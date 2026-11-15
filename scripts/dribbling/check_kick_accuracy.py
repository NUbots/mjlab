"""Break a kick checkpoint's direction error down by commanded angle and foot.

Run with: uv run python scripts/dribbling/check_kick_accuracy.py [checkpoint.pt]
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.dribbling.mdp import KickCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Kick-Flat-Booster-K1"


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> None:
  if len(sys.argv) > 1:
    ckpt = Path(sys.argv[1])
  else:
    ckpt = max(
      Path("logs/rsl_rl/k1_kick").glob("*/model_*.pt"),
      key=lambda p: p.stat().st_mtime,
    )
  speed_range = (2.0, 3.0)
  if len(sys.argv) > 3:
    speed_range = (float(sys.argv[2]), float(sys.argv[3]))
  print(f"Checkpoint: {ckpt}, speeds {speed_range}")
  device = "cuda:0"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1024
  env_cfg.commands["ball_velocity"].speed_range = speed_range
  agent_cfg = load_rl_cfg(TASK)
  base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot, ball = base.scene["robot"], base.scene["ball"]
  kick = base.command_manager.get_term("ball_velocity")
  assert isinstance(kick, KickCommand)
  site_ids, _ = robot.find_sites(("left_foot", "right_foot"))

  obs, _ = env.reset()
  cmd = kick.command[:, :2].clone()
  cmd_angle = torch.atan2(cmd[:, 1], cmd[:, 0])
  yaw0 = robot.data.heading_w.clone()
  rel0 = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  bearing0 = torch.atan2(rel0[:, 1], rel0[:, 0])

  n = base.num_envs
  peak = torch.zeros(n, device=device)
  peak_vel = torch.zeros(n, 2, device=device)
  foot = torch.full((n,), -1, device=device)
  yaw_at_kick = torch.zeros(n, device=device)
  time_at_kick = torch.zeros(n, device=device)
  alive = torch.ones(n, dtype=torch.bool, device=device)
  with torch.no_grad():
    for step in range(290):
      obs, _, dones, _ = env.step(policy(obs))
      alive &= ~dones.bool()
      vel = ball.data.root_link_lin_vel_w[:, :2]
      speed = vel.norm(dim=-1)
      first = alive & kick.struck & (foot < 0)
      feet = robot.data.site_pos_w[:, site_ids, :2]
      dist = (feet - ball.data.root_link_pos_w[:, None, :2]).norm(dim=-1)
      foot = torch.where(first, dist.argmin(dim=-1), foot)
      yaw_at_kick = torch.where(first, robot.data.heading_w, yaw_at_kick)
      time_at_kick = torch.where(
        first, torch.full_like(peak, step * 0.02), time_at_kick
      )
      better = alive & kick.struck & (speed > peak)
      peak = torch.where(better, speed, peak)
      peak_vel = torch.where(better[:, None], vel, peak_vel)

  ok = foot >= 0
  err = torch.rad2deg(_wrap(torch.atan2(peak_vel[:, 1], peak_vel[:, 0]) - cmd_angle))
  a = err[ok].abs()
  print(
    f"Kicked: {ok.float().mean():.0%} of {n}; fell/reset early: "
    f"{(~alive).float().mean():.0%}"
  )
  print(
    f"|error| mean {a.mean():.1f}  median {a.median():.1f}  "
    f"p90 {a.quantile(0.9):.1f}  within 5deg {(a < 5).float().mean():.0%}  "
    f"within 10deg {(a < 10).float().mean():.0%}"
  )
  print(f"signed error mean {err[ok].mean():+.1f}")
  print(f"speed ratio mean {(peak[ok] / cmd[ok].norm(dim=-1)).mean():.2f}")
  print(f"time of strike mean {time_at_kick[ok].mean():.2f} s")
  for i, name in enumerate(("left", "right")):
    m = ok & (foot == i)
    if m.any():
      print(
        f"  {name} foot: {m.float().sum() / ok.float().sum():.0%} of kicks, "
        f"|err| {err[m].abs().mean():.1f}, signed {err[m].mean():+.1f}"
      )
  # Commanded direction relative to the robot's starting heading.
  rel_cmd = torch.rad2deg(_wrap(cmd_angle - yaw0))
  print(f"command vs start heading: range {rel_cmd.min():.0f}..{rel_cmd.max():.0f}")
  edges = [-180, -45, -30, -15, 0, 15, 30, 45, 180]
  for lo, hi in zip(edges[:-1], edges[1:], strict=True):
    m = ok & (rel_cmd >= lo) & (rel_cmd < hi)
    if m.sum() > 5:
      print(
        f"  cmd {lo:+4d}..{hi:+4d}: n={int(m.sum()):4d} |err| "
        f"{err[m].abs().mean():5.1f} signed {err[m].mean():+6.1f} "
        f"left-foot {(foot[m] == 0).float().mean():.0%}"
      )
  # Is the tail of bad kicks tied to when the strike happens or how far the
  # ball started?
  dist0 = rel0.norm(dim=-1)
  for label, x, edges_x in (
    ("strike time (s)", time_at_kick, [0.0, 0.3, 0.5, 0.7, 1.0, 6.0]),
    ("start distance (m)", dist0, [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]),
    ("speed ratio", peak / cmd.norm(dim=-1), [0.0, 0.7, 0.9, 1.1, 1.3, 9.0]),
  ):
    print(label)
    for lo, hi in zip(edges_x[:-1], edges_x[1:], strict=True):
      m = ok & (x >= lo) & (x < hi)
      if m.sum() > 5:
        print(
          f"  {lo:.1f}..{hi:.1f}: n={int(m.sum()):4d} |err| mean "
          f"{err[m].abs().mean():5.1f} median {err[m].abs().median():5.1f}"
        )
  # Close balls: does the error grow with how far the commanded direction is
  # from the robot->ball line (how far round the ball the robot must step)?
  off_line = torch.rad2deg(_wrap(cmd_angle - bearing0)).abs()
  for label, sel in (("ball < 0.6 m", dist0 < 0.6), ("ball >= 0.6 m", dist0 >= 0.6)):
    print(f"command off the robot->ball line, {label}")
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 90)):
      m = ok & sel & (off_line >= lo) & (off_line < hi)
      if m.sum() > 5:
        print(
          f"  {lo:2d}..{hi:2d} deg: n={int(m.sum()):4d} |err| mean "
          f"{err[m].abs().mean():5.1f} median {err[m].abs().median():5.1f} "
          f"strike at {time_at_kick[m].mean():.2f} s"
        )
  # Did the robot turn to face the commanded direction before kicking?
  face = torch.rad2deg(_wrap(cmd_angle - yaw_at_kick))
  print(f"heading vs command at strike: mean |.| {face[ok].abs().mean():.1f} deg")
  rel_bear = torch.rad2deg(_wrap(cmd_angle - bearing0))
  print(
    f"command vs robot->ball line at start: mean |.| "
    f"{rel_bear.abs().mean():.1f}, max {rel_bear.abs().max():.0f}"
  )
  env.close()


if __name__ == "__main__":
  main()
