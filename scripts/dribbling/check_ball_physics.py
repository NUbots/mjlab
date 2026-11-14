"""Sanity-check the dribbling scene's ball physics.

Run with: uv run python scripts/dribbling/check_ball_physics.py
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.dribbling.config.booster_k1.env_cfgs import (
  BALL_RADIUS,
  booster_k1_dribble_env_cfg,
)


def main() -> None:
  cfg = booster_k1_dribble_env_cfg(play=True)
  cfg.scene.num_envs = 4
  cfg.events.pop("ball_friction", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  robot, ball = env.scene["robot"], env.scene["ball"]
  n = env.num_envs
  zero = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
  ids = torch.arange(n, device=env.device)

  def place_ball(offset, vel):
    """Place the ball at a robot-yaw-frame offset with a yaw-frame velocity."""
    env.reset()
    data = robot.data
    q = data.data.qpos[:, data.indexing.free_joint_q_adr]
    yaw = torch.atan2(
      2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
      1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2),
    )
    c, s = torch.cos(yaw), torch.sin(yaw)
    pos = torch.stack(
      [
        q[:, 0] + c * offset[0] - s * offset[1],
        q[:, 1] + s * offset[0] + c * offset[1],
        torch.full_like(yaw, offset[2]),
      ],
      dim=-1,
    )
    quat = torch.tensor([1.0, 0, 0, 0], device=env.device).repeat(n, 1)
    v = torch.zeros(n, 6, device=env.device)
    v[:, 0] = c * vel[0] - s * vel[1]
    v[:, 1] = s * vel[0] + c * vel[1]
    ball.write_root_link_pose_to_sim(torch.cat([pos, quat], -1), env_ids=ids)
    ball.write_root_link_velocity_to_sim(v, env_ids=ids)

  def rel_ball():
    return (ball.data.root_link_pos_w - robot.data.root_link_pos_w)[0]

  # 1. Rest height and roll-out deceleration, ball sent away from the robot.
  place_ball((1.5, 0.0, BALL_RADIUS), (2.0, 0.0))
  speeds = []
  for _ in range(100):
    env.step(zero)
    speeds.append(ball.data.root_link_lin_vel_w[0, :2].norm().item())
  z = ball.data.root_link_pos_w[0, 2].item()
  decel = (speeds[10] - speeds[60]) / (50 * env.step_dt)
  print(
    f"[roll]  ball z={z:.4f} (radius {BALL_RADIUS}), speed 0.2s={speeds[10]:.2f}"
    f" 1.2s={speeds[60]:.2f} m/s, decel={decel:.2f} m/s^2"
  )

  # 2. Ball rolled into the standing robot's feet: it must not pass through.
  place_ball((1.0, 0.09, BALL_RADIUS), (-2.0, 0.0))
  min_x = 9.0
  for _ in range(60):
    env.step(zero)
    min_x = min(min_x, rel_ball()[0].item())
  print(
    f"[foot]  closest ball x in robot frame={min_x:.3f} m, final rel="
    f"{[round(v, 2) for v in rel_ball().tolist()]}"
  )

  # 3. Ball fired at shin height: it must bounce back, not pass through.
  place_ball((0.45, 0.096, 0.22), (-6.0, 0.0))
  q = robot.data.data.qpos[0, robot.data.indexing.free_joint_q_adr]
  yaw = torch.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
  fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)])
  trace = []
  for _ in range(8):
    env.step(zero)
    v = ball.data.root_link_lin_vel_w[0, :2]
    trace.append((round(rel_ball()[0].item(), 2), round((v @ fwd).item(), 2)))
  print(f"[shin]  (ball x in robot frame, ball forward speed) per step: {trace}")

  # 4. Observation layout.
  print("[obs]   groups:", dict(env.observation_manager.group_obs_term_dim))
  env.close()


if __name__ == "__main__":
  main()
