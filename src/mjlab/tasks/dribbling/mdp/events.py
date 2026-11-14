"""Dribbling task events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _uniform(rng: tuple[float, float], n: int, device: str) -> torch.Tensor:
  return torch.empty(n, device=device).uniform_(*rng)


def reset_robot_and_ball(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  robot_xy_range: tuple[float, float] = (-0.5, 0.5),
  robot_z_range: tuple[float, float] = (0.01, 0.05),
  robot_yaw_range: tuple[float, float] = (-3.14159, 3.14159),
  ball_distance_range: tuple[float, float] = (0.4, 1.0),
  ball_bearing_range: tuple[float, float] = (-0.4, 0.4),
  ball_radius: float = 0.11,
) -> None:
  """Reset the robot's root pose and put a resting ball ahead of it.

  One event rather than two because the ball is placed relative to the pose
  the robot was just given, and kinematics are not forwarded between reset
  events, so the robot's world pose cannot be read back from the sim here.

  JOGA Table VI: robot xy in +/-0.5 m, yaw uniform, ball 0.4-1.0 m ahead.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  origins = env.scene.env_origins[env_ids]

  root = robot.data.default_root_state[env_ids].clone()
  x = _uniform(robot_xy_range, n, env.device)
  y = _uniform(robot_xy_range, n, env.device)
  z = _uniform(robot_z_range, n, env.device)
  yaw = _uniform(robot_yaw_range, n, env.device)
  pos = root[:, 0:3] + torch.stack([x, y, z], dim=-1) + origins
  zeros = torch.zeros_like(yaw)
  quat = quat_mul(root[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
  robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)

  dist = _uniform(ball_distance_range, n, env.device)
  bearing = yaw + _uniform(ball_bearing_range, n, env.device)
  ball_pos = torch.stack(
    [
      pos[:, 0] + dist * torch.cos(bearing),
      pos[:, 1] + dist * torch.sin(bearing),
      origins[:, 2] + ball_radius,
    ],
    dim=-1,
  )
  ball_quat = torch.zeros(n, 4, device=env.device)
  ball_quat[:, 0] = 1.0
  ball.write_root_link_pose_to_sim(
    torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
  )
  ball.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids
  )
