"""Measure a trained block policy's capability envelope.

The goalie's behaviour system chooses what to do from what the policy *can* do, so the
envelope is the interface between them: for a shot crossing the goalie's line at ``dy``
with this much warning, how often does it get a touch on the ball, and how often does
it fall over trying?

Shots are rolled at a goalie driven by the checkpoint, and every resolved shot is
recorded against the crossing point and time-to-arrival measured when it was kicked.
The result is written as CSV next to the checkpoint so ``planning::PlanSave`` can load
the same numbers it was measured with.

    python -m mjlab.tasks.goalkeeper.scripts.measure_envelope \
        --checkpoint logs/rsl_rl/k1_block/<run>/model_1500.pt
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401  Populates the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.goalkeeper.mdp import ShotCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = "Mjlab-Block-Booster-K1"

DY_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2)
"""Bins of |dy|: how far sideways the ball crosses, in the goalie's frame."""
TIME_EDGES = (0.0, 0.5, 0.8, 1.2, 2.0, 99.0)
"""Bins of time-to-arrival at the moment of the kick: how much warning there was."""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _bin(value: float, edges: tuple[float, ...]) -> int:
  for i in range(len(edges) - 1):
    if edges[i] <= value < edges[i + 1]:
      return i
  return len(edges) - 2


def main(
  checkpoint: Path,
  num_envs: int = 256,
  steps: int = 6000,
  seed: int = 7,
  output: Path | None = None,
) -> None:
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  # Episodes have to end so a fallen goalie is reset and keeps taking shots. Shots cut
  # short by that reset are counted as failures below, not dropped.
  env_cfg.episode_length_s = 30.0
  agent_cfg = load_rl_cfg(TASK_ID)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  shot = env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)

  # Per shot, captured when it is kicked and resolved when it finishes.
  pending_dy = torch.zeros(num_envs, device=device)
  pending_time = torch.zeros(num_envs, device=device)
  pending_speed = torch.zeros(num_envs, device=device)
  live = torch.zeros(num_envs, dtype=torch.bool, device=device)
  was_moving = torch.zeros(num_envs, dtype=torch.bool, device=device)
  was_finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
  # Row of each env's last save, so its clearance can be filled in once the
  # follow-through has been judged.
  save_row = torch.full((num_envs,), -1, dtype=torch.long)

  rows: list[dict[str, float]] = []
  # Every fall, not only those that cut a shot short: a keeper that clears the ball
  # and lands on the floor after it has resolved its shot before it falls.
  total_falls = 0
  obs = wrapped.get_observations()
  for _ in range(steps):
    with torch.inference_mode():
      actions = policy(obs)
    obs, _, dones, _ = wrapped.step(actions)
    total_falls += int(env.termination_manager.terminated.sum())

    # A shot's axes are read once, just after the kick, before any contact bends it.
    just_kicked = shot.was_moving & ~was_moving
    if bool(just_kicked.any()):
      pending_dy[just_kicked] = shot.true_crossing[just_kicked]
      pending_time[just_kicked] = shot.true_time_to_cross[just_kicked]
      pending_speed[just_kicked] = shot.ball.data.root_link_lin_vel_w[
        just_kicked, :2
      ].norm(dim=-1)
      live[just_kicked] = True
    was_moving = shot.was_moving.clone()

    # A shot cut short because the goalie fell is a shot it did not block. Dropping
    # those would quietly score the envelope only over the shots it survived.
    interrupted = (dones > 0) & live & ~shot.finished
    if bool(interrupted.any()):
      for i in interrupted.nonzero(as_tuple=False).flatten().tolist():
        rows.append(
          {
            "dy": float(pending_dy[i]),
            "time_to_arrival": float(pending_time[i]),
            "speed": float(pending_speed[i]),
            "on_target": float(bool(shot.on_target_shot[i])),
            "saved": 0.0,
            "touched": float(bool(shot.touched[i])),
            "conceded": 0.0,
            "fell": 1.0,
            "rest_x": 0.0,
          }
        )
      live[interrupted] = False

    resolved = shot.finished & ~was_finished & live
    if bool(resolved.any()):
      fell = env.termination_manager.terminated
      for i in resolved.nonzero(as_tuple=False).flatten().tolist():
        rows.append(
          {
            "dy": float(pending_dy[i]),
            "time_to_arrival": float(pending_time[i]),
            "speed": float(pending_speed[i]),
            "on_target": float(bool(shot.on_target_shot[i])),
            "saved": float(bool(shot.saved_now[i])),
            "touched": float(bool(shot.touched[i])),
            "conceded": float(bool(shot.scored_now[i])),
            "fell": float(bool(fell[i])),
            # Where the ball will stop, in metres up the field from the goalie's
            # start, updated below with the best over the save's follow-through.
            "rest_x": float(shot.rest_x[i]),
          }
        )
        if shot.saved_now[i]:
          save_row[i] = len(rows) - 1
      live[resolved] = False
    was_finished = shot.finished.clone()

    for i in shot.clear_judged_now.nonzero(as_tuple=False).flatten().tolist():
      if save_row[i] >= 0:
        rows[int(save_row[i])]["rest_x"] = float(shot.best_rest[i])
        save_row[i] = -1

  play_minutes = num_envs * steps * env.step_dt / 60.0
  env.close()

  if not rows:
    raise SystemExit("no shots resolved; run for more steps")

  destination = output or checkpoint.parent / "envelope.csv"
  with destination.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  # An envelope describes one policy. Recording what it was measured from lets
  # planning::PlanSave refuse a mismatched pair rather than plan against numbers
  # belonging to a policy the robot is not running.
  sidecar = {
    "task": TASK_ID,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _sha256(checkpoint),
    "shots": len(rows),
    "shots_on_target": len([r for r in rows if r["on_target"]]),
    "save_rate": (
      sum(r["saved"] for r in rows) / max(1, len([r for r in rows if r["on_target"]]))
    ),
    "touch_rate": (
      sum(r["touched"] for r in rows) / max(1, len([r for r in rows if r["on_target"]]))
    ),
    "fell": sum(r["fell"] for r in rows),
    "falls_per_minute": total_falls / play_minutes,
    "shot_level": shot.levels[shot.level],
    "dy_edges": list(DY_EDGES),
    "time_edges": list(TIME_EDGES),
  }
  onnx_path = next(checkpoint.parent.glob("*.onnx"), None)
  if onnx_path is not None:
    sidecar["onnx"] = str(onnx_path)
    sidecar["onnx_sha256"] = _sha256(onnx_path)
  (destination.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2) + "\n")

  on_target = [r for r in rows if r["on_target"]]
  print(f"{len(rows)} shots ({len(on_target)} of them on target) -> {destination}\n")
  if on_target:
    saved = sum(r["saved"] for r in on_target) / len(on_target)
    touched = sum(r["touched"] for r in on_target) / len(on_target)
    print(f"Saved: {saved:.0%} of the shots that were going in")
    print(f"Touched: {touched:.0%} (a touch that goes in is still a goal)")
    saves = [r for r in on_target if r["saved"]]
    cleared = sum(r["rest_x"] > 1.0 for r in saves) / len(on_target)
    print(f"Cleared: {cleared:.0%} saved and sent more than 1 m up the field")
    if saves:
      rests = sorted(r["rest_x"] for r in saves)
      mean_rest = sum(rests) / len(rests)
      median_rest = rests[len(rests) // 2]
      print(
        f"Saved balls come to rest {mean_rest:.2f} m up the field on average "
        f"(median {median_rest:.2f} m)\n"
      )

  print("Save rate on shots that were going in, by |dy| (m) and warning (s):")
  header = "  |dy|      " + "".join(
    f"{TIME_EDGES[j]:>5.1f}-{TIME_EDGES[j + 1]:<5.1f}"
    if TIME_EDGES[j + 1] < 90
    else f"{TIME_EDGES[j]:>5.1f}+     "
    for j in range(len(TIME_EDGES) - 1)
  )
  print(header)
  for i in range(len(DY_EDGES) - 1):
    cells = []
    for j in range(len(TIME_EDGES) - 1):
      subset = [
        r
        for r in on_target
        if _bin(abs(r["dy"]), DY_EDGES) == i
        and _bin(r["time_to_arrival"], TIME_EDGES) == j
      ]
      if len(subset) < 5:
        cells.append(f"{'-':>10}")
      else:
        rate = sum(r["saved"] for r in subset) / len(subset)
        cells.append(f"{rate:>6.0%}({len(subset):3d})")
    print(f"  {DY_EDGES[i]:.1f}-{DY_EDGES[i + 1]:.1f}   " + "".join(cells))

  falls = sum(r["fell"] for r in rows)
  print(f"\nShots during which the goalie fell: {falls:.0f} of {len(rows)}")
  print(
    f"Falls per minute of play, after the shot too: {sidecar['falls_per_minute']:.2f}"
  )


if __name__ == "__main__":
  tyro.cli(main)
