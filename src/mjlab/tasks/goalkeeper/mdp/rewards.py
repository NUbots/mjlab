"""Goalkeeper rewards.

Every term here is computed from the *true* ball, not from the estimate the policy is
commanded with: the policy is paid for blocking the ball that is actually there. The
gap between the two is the perception model in ``shot_command.py``, and learning to
act well despite it is the point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommand
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _shot(env: ManagerBasedRlEnv, command_name: str) -> ShotCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, ShotCommand)
  return term


def approach_crossing(
  env: ManagerBasedRlEnv,
  command_name: str,
  reach: float = 0.25,
  sharpness: float = 3.0,
  horizon: float = 1.0,
  switch_time: float = 0.35,
) -> torch.Tensor:
  """Reward being where the ball will cross, worth more the closer it is to arriving.

  Shaped as a soft step rather than a Gaussian: full value inside ``reach``, falling
  off over a width of about ``1 / sharpness``, and still meaningfully sloped a metre
  out. The Gaussian this replaces was flat past 0.6 m, so two thirds of the misses
  (the ones that never got a foot within 20 cm of the ball) sat in a part of the
  reward that could not tell moving towards the ball from standing still.

  The target switches late: until ``switch_time`` before arrival it is the predicted
  crossing point, and after that the ball's own position, because by then a prediction
  is worth less than what is actually in front of the goalie.
  """
  shot = _shot(env, command_name)
  late = shot.true_time_to_cross < switch_time
  target = torch.where(late, shot.ball_offset, shot.true_crossing)
  aligned = 1.0 - torch.sigmoid((target.abs() - reach) * sharpness)
  urgency = (1.0 - shot.true_time_to_cross / horizon).clamp(0.0, 1.0)
  return aligned * urgency * shot.on_target.float()


def touched(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off reward the step the ball first touches the robot.

  Getting in the way is progress, but it is not the job: a touch that deflects the
  ball into the goal is still a goal, so this is worth much less than a save.
  """
  return _shot(env, command_name).touched_now.float()


def saved(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off reward for keeping out a shot that was going in.

  Paid when a shot that would have crossed the goal line inside the posts ends up not
  doing so: stopped, deflected wide, or sent back out.
  """
  return _shot(env, command_name).saved_now.float()


def cleared(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off reward on a save for how far up the field the ball will come to rest.

  A ball stopped at the goalie's feet or knocked just wide is still there for the
  shooter to have another go at. This pays, on top of the save, for sending it away
  from the goal and towards the other end: nothing for a ball left at the goalie's
  feet, full value for one that will roll ``clear_distance`` or more up the field.
  """
  return _shot(env, command_name).cleared_now


def conceded(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One-off penalty the step a shot crosses the goal line inside the posts."""
  return _shot(env, command_name).scored_now.float()


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


def defused(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reward, after a touch, for where the ball is now headed.

  The goalie touches far more shots than it saves: it gets in the way of a fast ball
  and the ball comes off it into the goal. Waiting for the shot to resolve says so
  only once, and late. This pays every step after the first touch for the danger
  actually taken out of the ball, which is 1 once it can no longer reach the goal
  (stopped dead or sent wide) and 0 while it is still bound for the posts.
  """
  shot = _shot(env, command_name)
  live = shot.was_moving & shot.touched & ~shot.finished & shot.on_target_shot
  return shot.defused * live.float()


def upright_with_dead_zone(
  env: ManagerBasedRlEnv,
  std: float,
  dead_zone_deg: float = 25.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward staying off the floor, without insisting on standing straight.

  The velocity task's upright term charges for every degree of tilt, which is right
  for walking and wrong here: a keeper reaching a wide ball has to lean out over a
  foot, and that lean was being paid for out of the same term that stops it toppling.
  Leaning up to ``dead_zone_deg`` is free, and past that the cost rises as before, so
  the term still does the job it was added for while leaving the motion alone.

  The dead zone has to stay well inside the angle that ends the episode, or the keeper
  would be paid full value for a tilt it is about to be terminated for.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  else:
    quat = asset.data.root_link_quat_w
  projected_gravity = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  # Magnitude of the horizontal part of gravity in the body frame: sin of the tilt.
  tilt = torch.linalg.norm(projected_gravity[:, :2], dim=-1)
  excess = (tilt - math.sin(math.radians(dead_zone_deg))).clamp(min=0.0)
  return torch.exp(-excess.square() / std**2)
