"""Goalkeeper rewards.

Every term here is computed from the *true* ball, not from the estimate the policy is
commanded with: the policy is paid for blocking the ball that is actually there. The
gap between the two is the perception model in ``shot_command.py``, and learning to
act well despite it is the point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _shot(env: ManagerBasedRlEnv, command_name: str) -> ShotCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, ShotCommand)
  return term


def line_up(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward standing where the ball will cross, while a shot is on its way.

  This is the dense signal that teaches side-stepping: the crossing point is expressed
  in the goalie's own frame, so driving it to zero means putting the body in the way.
  """
  shot = _shot(env, command_name)
  aligned = torch.exp(-shot.true_crossing.square() / std**2)
  return aligned * shot.on_target.float()


def urgency_weighted_line_up(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  horizon: float = 1.0,
) -> torch.Tensor:
  """``line_up``, worth more the closer the ball is to arriving.

  Being in place early is worth little if the goalie drifts off the line again, so the
  reward is concentrated in the last ``horizon`` seconds before the ball arrives.
  """
  shot = _shot(env, command_name)
  aligned = torch.exp(-shot.true_crossing.square() / std**2)
  urgency = (1.0 - shot.true_time_to_cross / horizon).clamp(0.0, 1.0)
  return aligned * urgency * shot.on_target.float()


def blocked(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off reward the step the ball first touches the robot."""
  return _shot(env, command_name).blocked_now.float()


def conceded(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off penalty the step a shot passes the goalie's line untouched."""
  return _shot(env, command_name).conceded_now.float()


def hold_line(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward staying on the goal line, rather than drifting up the pitch or into goal.

  The goalie's depth is the positioning layer's decision, not the policy's; the policy
  holds the line it was placed on and moves sideways along it.

  Bounded on purpose. As an unbounded ``(offset / std)^2`` penalty this grew without
  limit while the robot toppled, so ending the episode paid better than standing up:
  episode length collapsed to about 7 steps and stayed there.
  """
  asset: Entity = env.scene[asset_cfg.name]
  offset = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.exp(-torch.square(offset / std))


def face_shooter(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward facing the field.

  The command is expressed in the goalie's frame, so a policy that turns can shrink
  ``dy`` without moving. Holding the heading keeps the command honest and keeps the
  camera pointed at the ball. Bounded, for the reason given in ``hold_line``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.square(asset.data.heading_w / std))


def posture(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward holding the ready stance, per joint."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids else slice(None)
  error = (
    asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
  )
  return torch.exp(-torch.mean(torch.square(error), dim=-1) / std**2)


def ready_when_idle(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward standing still in the ready stance when there is no shot to block."""
  asset: Entity = env.scene[asset_cfg.name]
  shot = _shot(env, command_name)
  speed = torch.linalg.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
  still = torch.exp(-speed.square() / std**2)
  return still * (~shot.on_target).float()


def foot_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize a foot sliding while it carries load.

  The velocity task's version is gated on a twist command, which the goalie does not
  have; here it applies whenever the foot is down, shuffling or not.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  return torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)


def close_on_crossing(
  env: ManagerBasedRlEnv,
  command_name: str,
  reference_speed: float = 1.0,
  deadband: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward moving sideways towards where the ball will cross.

  ``line_up`` pays for *being* on the ball's line, and as a Gaussian it is already
  flat by 0.6 m: a goalie a metre off gets the same (zero) reward whether it steps
  towards the ball or stands still, so nothing tells it to move. This pays for closing
  the gap at any distance, which is the gradient that was missing. It is bounded, and
  it stops at the deadband so a goalie already on the line is not paid to jitter.
  """
  shot = _shot(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]

  heading = asset.data.heading_w
  velocity_w = asset.data.root_link_lin_vel_w[:, :2]
  lateral = (
    -torch.sin(heading) * velocity_w[:, 0] + torch.cos(heading) * velocity_w[:, 1]
  )

  closing = lateral * torch.sign(shot.true_crossing)
  worth_moving = (shot.true_crossing.abs() > deadband).float()
  return (
    (closing / reference_speed).clamp(-1.0, 1.0) * worth_moving * shot.on_target.float()
  )
