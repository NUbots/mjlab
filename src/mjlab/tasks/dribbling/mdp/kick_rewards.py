"""Kicking task rewards, after JOGA Table V (passing / shooting column)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.tasks.dribbling.mdp.kick_command import KickCommand
from mjlab.tasks.dribbling.mdp.rewards import _ball_pos_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _kick(env: ManagerBasedRlEnv, command_name: str) -> KickCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, KickCommand)
  return term


def ball_exit_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
  loft_penalty: float = 0.2,
) -> torch.Tensor:
  """I_win exp(-||v_ball,xy - v_cmd||^2 / std^2 - d_z v_ball,z^2)."""
  kick = _kick(env, command_name)
  ball: Entity = env.scene["ball"]
  vel = ball.data.root_link_lin_vel_w
  err = torch.sum(torch.square(vel[:, :2] - kick.command[:, :2]), dim=-1)
  loft = loft_penalty * torch.square(vel[:, 2])
  return kick.in_window.float() * torch.exp(-err / std**2 - loft)


def ball_in_strike_zone(
  env: ManagerBasedRlEnv,
  command_name: str,
  zone: tuple[float, float] = (0.1, 0.0),
  sharpness: float = 6.0,
) -> torch.Tensor:
  """I_pre exp(-k ||p_ball,xy^b - zone||^2): get to the ball before the kick."""
  kick = _kick(env, command_name)
  pos_b = _ball_pos_b(env)[:, :2]
  target = torch.tensor(zone, device=env.device)
  dist_sq = torch.sum(torch.square(pos_b - target), dim=-1)
  return kick.pre_strike.float() * torch.exp(-sharpness * dist_sq)


def face_ball_before_kick(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """I_pre exp(-2 (1 - p_hat_x))."""
  kick = _kick(env, command_name)
  pos_b = _ball_pos_b(env)[:, :2]
  unit_x = pos_b[:, 0] / torch.norm(pos_b, dim=-1).clamp(min=1e-6)
  return kick.pre_strike.float() * torch.exp(-2.0 * (1.0 - unit_x))


def stay_still_after_kick(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(1 - I_pre) exp(-2 ||v_base,xy||): do not chase the ball after kicking it."""
  kick = _kick(env, command_name)
  robot: Entity = env.scene["robot"]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
  return kick.struck.float() * torch.exp(-2.0 * speed)


def time_to_kick(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """I_pre: a constant cost for every step the ball has not been struck."""
  return _kick(env, command_name).pre_strike.float()


def leg_ball_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Count leg geoms (shin, knee) pressing on the ball: kick with the foot."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force = torch.norm(sensor.data.force, dim=-1)  # [B, N]
  return (force > force_threshold).sum(dim=-1).float()
