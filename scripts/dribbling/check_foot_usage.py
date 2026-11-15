"""Count which foot a dribbling checkpoint touches the ball with.

A touch is a control step in which the ball's planar speed jumps while the
ball is within reach; it is credited to the nearer foot.

Run with: uv run python scripts/dribbling/check_foot_usage.py [checkpoint.pt]
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Dribble-Flat-Booster-K1"
SPEED_JUMP = 0.1  # m/s gained in one control step counts as a touch.
REACH = 0.35  # m, foot to ball centre.


def main() -> None:
  if len(sys.argv) > 1:
    ckpt = Path(sys.argv[1])
  else:
    ckpt = max(
      Path("logs/rsl_rl/k1_dribble").glob("*/model_*.pt"),
      key=lambda p: p.stat().st_mtime,
    )
  print(f"Checkpoint: {ckpt}")
  device = "cuda:0"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 256
  agent_cfg = load_rl_cfg(TASK)
  base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot, ball = base.scene["robot"], base.scene["ball"]
  site_ids, site_names = robot.find_sites(("left_foot", "right_foot"))
  assert list(site_names) == ["left_foot", "right_foot"], site_names

  touches = torch.zeros(base.num_envs, 2, device=device)
  obs, _ = env.reset()
  prev_speed = ball.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  with torch.no_grad():
    for _ in range(1000):
      obs, _, dones, _ = env.step(policy(obs))
      speed = ball.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
      feet = robot.data.site_pos_w[:, site_ids, :2]
      dist = (feet - ball.data.root_link_pos_w[:, None, :2]).norm(dim=-1)
      nearest = dist.argmin(dim=-1)
      hit = (speed - prev_speed > SPEED_JUMP) & (dist.min(dim=-1).values < REACH)
      hit &= ~dones.bool()
      touches[hit, nearest[hit]] += 1
      prev_speed = torch.where(dones.bool(), torch.zeros_like(speed), speed)

  left, right = touches.sum(dim=0).tolist()
  total = max(left + right, 1.0)
  print(
    f"Touches: left {left:.0f} ({left / total:.0%}), right {right:.0f} "
    f"({right / total:.0%})"
  )
  # Per-env share, to tell "each env uses both" from "half the envs use each".
  n = touches.sum(dim=-1)
  share = touches[n >= 5, 0] / n[n >= 5]
  print(
    f"Per-env left share over {share.numel()} envs: "
    f"median {share.median():.2f}, "
    f"mostly-left {(share > 0.8).float().mean():.0%}, "
    f"mostly-right {(share < 0.2).float().mean():.0%}, "
    f"mixed {((share >= 0.2) & (share <= 0.8)).float().mean():.0%}"
  )
  env.close()


if __name__ == "__main__":
  main()
