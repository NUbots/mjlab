"""Dribbling task rewards, after JOGA Table V (dribbling column)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")


def _ball_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  rel_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), rel_w)


def ball_in_zone(
  env: ManagerBasedRlEnv,
  zone: tuple[float, float] = (0.2, 0.0),
  std: float = 1.0,
) -> torch.Tensor:
  """exp(-||p_ball,xy^b - zone||^2 / std^2): keep the ball just ahead of the feet."""
  pos_b = _ball_pos_b(env)[:, :2]
  target = torch.tensor(zone, device=env.device)
  dist_sq = torch.sum(torch.square(pos_b - target), dim=-1)
  env.extras["log"]["Metrics/ball_distance_mean"] = torch.mean(
    torch.norm(pos_b, dim=-1)
  )
  return torch.exp(-dist_sq / std**2)


def ball_velocity_match(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
) -> torch.Tensor:
  """exp(-||v_ball,xy - v_cmd||^2 / std^2)."""
  ball: Entity = env.scene["ball"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  err = ball.data.root_link_lin_vel_w[:, :2] - command[:, :2]
  return torch.exp(-torch.sum(torch.square(err), dim=-1) / std**2)


def ball_velocity_along_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_speed: float = 2.5,
  near_base: float = 0.4,
  near_per_speed: float = 0.25,
) -> torch.Tensor:
  """I_near * clip(v_ball . v_cmd_hat, 0, max) / max.

  Pays for ball speed along the commanded direction, but only while the ball
  is close to the robot, so booting it away and watching it roll pays nothing.
  """
  ball: Entity = env.scene["ball"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, :2]
  cmd_speed = torch.norm(cmd, dim=-1)
  cmd_dir = cmd / cmd_speed.clamp(min=1e-6).unsqueeze(-1)
  along = torch.sum(ball.data.root_link_lin_vel_w[:, :2] * cmd_dir, dim=-1)
  dist = torch.norm(_ball_pos_b(env)[:, :2], dim=-1)
  near = (dist < near_base + near_per_speed * cmd_speed) & (cmd_speed > 0.1)
  env.extras["log"]["Metrics/ball_speed_along_cmd_mean"] = torch.mean(along)
  env.extras["log"]["Metrics/ball_near_fraction"] = torch.mean(near.float())
  return near.float() * along.clamp(0.0, max_speed) / max_speed


def face_ball(env: ManagerBasedRlEnv) -> torch.Tensor:
  """exp(-2 (1 - p_hat_x)), p_hat the unit base-to-ball vector in the yaw frame."""
  pos_b = _ball_pos_b(env)[:, :2]
  unit_x = pos_b[:, 0] / torch.norm(pos_b, dim=-1).clamp(min=1e-6)
  return torch.exp(-2.0 * (1.0 - unit_x))


def feet_too_close(
  env: ManagerBasedRlEnv,
  min_distance: float = 0.1,
  sharpness: float = 200.0,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """1 - exp(-k [d_min - ||p_L,xy - p_R,xy||]_+^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
  dist = torch.norm(feet[:, 0] - feet[:, 1], dim=-1)
  shortfall = (min_distance - dist).clamp(min=0.0)
  return 1.0 - torch.exp(-sharpness * torch.square(shortfall))


def drive_ball_along_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  near_base: float = 0.4,
  near_per_speed: float = 0.25,
) -> torch.Tensor:
  """Base velocity along the command, while the ball is near and lined up.

  Not in JOGA. A bridge out of the "stand beside the ball" optimum: the ball
  terms pay only once the ball is already moving, and a first tentative touch
  costs more (ball leaves the zone) than it earns. This pays the robot for
  walking in the commanded direction, scaled by how well the ball sits ahead
  of it in that direction, which is the state in which walking pushes the
  ball. Capped at the commanded speed, so it cannot pay for outrunning it.
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, :2]
  cmd_speed = torch.norm(cmd, dim=-1)
  cmd_dir = cmd / cmd_speed.clamp(min=1e-6).unsqueeze(-1)
  ball: Entity = env.scene["ball"]
  rel = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dist = torch.norm(rel, dim=-1)
  alignment = torch.sum(rel / dist.clamp(min=1e-6).unsqueeze(-1) * cmd_dir, dim=-1)
  base_along = torch.sum(robot.data.root_link_lin_vel_w[:, :2] * cmd_dir, dim=-1)
  near = (dist < near_base + near_per_speed * cmd_speed) & (cmd_speed > 0.1)
  progress = torch.minimum(base_along.clamp(min=0.0), cmd_speed) / cmd_speed.clamp(
    min=1e-6
  )
  env.extras["log"]["Metrics/base_speed_along_cmd_mean"] = torch.mean(base_along)
  env.extras["log"]["Metrics/ball_alignment_mean"] = torch.mean(alignment)
  return near.float() * alignment.clamp(min=0.0) * progress
