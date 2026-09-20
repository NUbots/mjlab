"""Check the task's simulated ball estimate against the real chain.

The block policy is trained on a command built from a simulated vision + UKF estimate
(``mdp/shot_command.py``). If that model drifts away from what the robot's own chain
does, the policy is trained for a robot that does not exist. This rolls shots at a
standing goalie and prints how fast the simulated estimate catches up with a kick,
next to what NUbots measured in NUSim.

Run it after changing the perception model, or after re-measuring the real one:

    python -m mjlab.tasks.goalkeeper.scripts.check_estimate_model
"""

from __future__ import annotations

import collections

import torch
import tyro

import mjlab.tasks  # noqa: F401  Populates the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Block-Booster-K1"

NUSIM_SPEED_RATIO = {0.0: 0.12, 0.1: 0.48, 0.2: 0.85, 0.3: 1.03, 0.4: 1.03, 0.5: 1.05}
"""Median estimated speed over true speed, by time since the kick, measured through
YOLO and the ball UKF in NUSim (September 2026). See ukf-validation.md in the goalie
plan for how it was measured."""


def main(num_envs: int = 64, steps: int = 1200, seed: int = 11) -> None:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset(seed=seed)

  shot = env.command_manager.get_term("shot")
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)

  speed_ratio: dict[float, list[float]] = collections.defaultdict(list)
  crossing_error: dict[float, list[float]] = collections.defaultdict(list)
  kick_time = torch.zeros(env.num_envs, device=device)
  was_kicked = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

  for _ in range(steps):
    env.step(action)
    just_kicked = shot.was_moving & ~was_kicked
    kick_time[just_kicked] = shot.time_since_resample[just_kicked]
    was_kicked = shot.was_moving.clone()

    live = shot.was_moving & ~shot.touched & ~shot.finished
    if not bool(live.any()):
      continue
    since_kick = (shot.time_since_resample - kick_time)[live]
    true_speed = shot.ball.data.root_link_lin_vel_w[live, :2].norm(dim=-1)
    estimated_speed = shot.est_vel[live].norm(dim=-1)
    ratio = (estimated_speed / true_speed.clamp(min=0.1)).cpu()
    error = (shot.command[live, 1] - shot.true_crossing[live]).abs().cpu()
    active = (shot.command[live, 0] > 0.5).cpu()

    for t, r, e, a in zip(
      since_kick.cpu().tolist(),
      ratio.tolist(),
      error.tolist(),
      active.tolist(),
      strict=True,
    ):
      bucket = min(round(int(t / 0.1) * 0.1, 1), 1.0)
      speed_ratio[bucket].append(r)
      if a:
        crossing_error[bucket].append(e)

  def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]

  print(
    f"{'since kick':>12} | {'est/true speed':>14} | {'NUSim':>6} | {'n':>5} | dy error"
  )
  for bucket in sorted(speed_ratio):
    reference = NUSIM_SPEED_RATIO.get(bucket)
    errors = crossing_error[bucket]
    print(
      f"{bucket:5.1f}-{bucket + 0.1:4.1f} s | {median(speed_ratio[bucket]):14.2f} | "
      f"{(f'{reference:.2f}' if reference else '-'):>6} | "
      f"{len(speed_ratio[bucket]):5d} | "
      f"{(f'{median(errors):.3f} m' if errors else '-')}"
    )

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
