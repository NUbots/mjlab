"""Check the kick task's strike detection and phased rewards.

Run with: uv run python scripts/dribbling/check_kick_logic.py
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.dribbling.config.booster_k1.kick_env_cfgs import (
  booster_k1_kick_env_cfg,
)
from mjlab.tasks.dribbling.mdp import KickCommand


def main() -> None:
  cfg = booster_k1_kick_env_cfg(play=False)
  cfg.scene.num_envs = 4
  cfg.events.pop("push_robot", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  ball = env.scene["ball"]
  kick = env.command_manager.get_term("ball_velocity")
  assert isinstance(kick, KickCommand)
  rm = env.reward_manager
  names = rm.active_terms
  zero = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
  ids = torch.arange(4, device=env.device)

  def terms() -> dict[str, float]:
    # Per-step weighted reward of env 0 for the phased terms.
    vals = rm._step_reward[0]
    keep = (
      "ball_exit_velocity",
      "ball_exit_direction",
      "ball_in_strike_zone",
      "face_ball",
      "stay_still_after_kick",
      "time_to_kick",
    )
    return {n: round(vals[names.index(n)].item(), 3) for n in keep}

  for _ in range(3):
    env.step(zero)
  print(f"[pre]    struck={kick.struck[0].item()} cmd={kick.command[0, :2].tolist()}")
  print(f"[pre]    {terms()}")

  # Put the ball at the feet (the reset can place it beyond the strike
  # radius) and give it the commanded velocity, as a perfect kick would.
  robot = env.scene["robot"]
  pose = ball.data.root_link_pose_w.clone()
  pose[:, :2] = robot.data.root_link_pos_w[:, :2]
  pose[:, 0] += 0.3
  ball.write_root_link_pose_to_sim(pose, env_ids=ids)
  vel = torch.zeros(4, 6, device=env.device)
  vel[:, :2] = kick.command[:, :2]
  ball.write_root_link_velocity_to_sim(vel, env_ids=ids)
  for i in range(3):
    env.step(zero)
    print(
      f"[kick+{i}] struck={kick.struck[0].item()} window={kick.in_window[0].item()}"
      f" {terms()}"
    )
  for _ in range(30):
    env.step(zero)
  print(
    f"[later]  struck={kick.struck[0].item()} window={kick.in_window[0].item()}"
    f" peak_ratio={kick.metrics['peak_speed_ratio'][0].item():.2f} {terms()}"
  )
  env.close()


if __name__ == "__main__":
  main()
