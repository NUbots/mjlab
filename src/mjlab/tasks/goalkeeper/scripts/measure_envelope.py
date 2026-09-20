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
  # Episodes have to end, or a goalie that falls never gets reset and stops taking
  # shots, which would quietly bias the envelope towards the shots it survived.
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

  rows: list[dict[str, float]] = []
  obs = wrapped.get_observations()
  for _ in range(steps):
    with torch.inference_mode():
      actions = policy(obs)
    obs, _, _, _ = wrapped.step(actions)

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

    resolved = shot.finished & ~was_finished & live
    if bool(resolved.any()):
      fell = env.termination_manager.terminated
      for i in resolved.nonzero(as_tuple=False).flatten().tolist():
        rows.append(
          {
            "dy": float(pending_dy[i]),
            "time_to_arrival": float(pending_time[i]),
            "speed": float(pending_speed[i]),
            "blocked": float(bool(shot.touched[i])),
            "conceded": float(bool(shot.shots_conceded[i] > 0)),
            "fell": float(bool(fell[i])),
          }
        )
      live[resolved] = False
    was_finished = shot.finished.clone()

  env.close()

  if not rows:
    raise SystemExit("no shots resolved; run for more steps")

  destination = output or checkpoint.parent / "envelope.csv"
  with destination.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  print(f"{len(rows)} shots -> {destination}\n")
  overall = sum(r["blocked"] for r in rows) / len(rows)
  print(f"Blocked overall: {overall:.0%}\n")

  print("Block rate by |dy| (m) and time to arrival at the kick (s):")
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
        for r in rows
        if _bin(abs(r["dy"]), DY_EDGES) == i
        and _bin(r["time_to_arrival"], TIME_EDGES) == j
      ]
      if len(subset) < 5:
        cells.append(f"{'-':>10}")
      else:
        rate = sum(r["blocked"] for r in subset) / len(subset)
        cells.append(f"{rate:>6.0%}({len(subset):3d})")
    print(f"  {DY_EDGES[i]:.1f}-{DY_EDGES[i + 1]:.1f}   " + "".join(cells))

  falls = sum(r["fell"] for r in rows)
  print(f"\nShots during which the goalie fell: {falls:.0f} of {len(rows)}")


if __name__ == "__main__":
  tyro.cli(main)
