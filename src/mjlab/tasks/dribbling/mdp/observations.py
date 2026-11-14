"""Dribbling observations: ball command, and the ball as a head camera sees it."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")


def ball_command_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded planar ball velocity, expressed in the robot's yaw frame.

  JOGA feeds the world-frame command plus the world-frame heading; rotating
  the command by the heading carries the same information in two numbers and
  needs no absolute heading on the robot beyond what makes the command.
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_b = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), command)
  return cmd_b[:, :2]


def ball_pos_yaw_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
  """Privileged: true ball position relative to the base, in its yaw frame."""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  rel_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), rel_w)


def ball_vel_yaw_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
  """Privileged: true ball linear velocity, in the base's yaw frame."""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  return quat_apply_inverse(
    yaw_quat(robot.data.root_link_quat_w), ball.data.root_link_lin_vel_w
  )


def ball_in_camera_frame(
  env: ManagerBasedRlEnv,
  camera_cfg: SceneEntityCfg,
  camera_offset: tuple[float, float, float],
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Ball position in the head camera frame (x forward, y left, z up)."""
  robot: Entity = env.scene[camera_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  head_pos = robot.data.body_link_pos_w[:, camera_cfg.body_ids, :].squeeze(1)
  head_quat = robot.data.body_link_quat_w[:, camera_cfg.body_ids, :].squeeze(1)
  offset = torch.tensor(camera_offset, device=env.device).expand_as(head_pos)
  cam_pos = head_pos + quat_apply(head_quat, offset)
  pos_c = quat_apply_inverse(head_quat, ball.data.root_link_pos_w - cam_pos)
  return pos_c, head_quat


class ball_camera_observation:
  """Ball as the onboard head camera reports it: ``(x, y, z, visible)``.

  Mirrors JOGA's ball observation (Appendix B.1): position in the camera
  frame with a binary visibility flag; when the ball is outside the field of
  view, beyond detection range, or dropped by the detector, the last
  observation is held. Because the camera rides on the actuated head, the
  policy has to learn where to look.

  JOGA samples detection error from a model fitted to hardware data (PAM).
  Without that data this uses the range-proportional Gaussian baseline: the
  error grows with distance, as monocular range from box size does.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._last = torch.zeros(env.num_envs, 4, device=env.device)
    # A plausible "ball just ahead on the ground" prior until first seen.
    self._prior = torch.tensor([0.5, 0.0, -0.8, 0.0], device=env.device)
    self._last[:] = self._prior

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last[env_ids] = self._prior

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    camera_cfg: SceneEntityCfg,
    camera_offset: tuple[float, float, float] = (0.06, 0.0, 0.07),
    hfov_deg: float = 90.0,
    vfov_deg: float = 60.0,
    max_range: float = 5.0,
    corrupt: bool = False,
    noise_std_base: float = 0.01,
    noise_std_per_m: float = 0.03,
    dropout_prob: float = 0.03,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
  ) -> torch.Tensor:
    pos_c, _ = ball_in_camera_frame(env, camera_cfg, camera_offset, ball_cfg)
    dist = torch.norm(pos_c, dim=-1)
    azimuth = torch.atan2(pos_c[:, 1], pos_c[:, 0])
    elevation = torch.atan2(pos_c[:, 2], pos_c[:, 0].clamp(min=1e-6))
    visible = (
      (pos_c[:, 0] > 0.0)
      & (azimuth.abs() < math.radians(hfov_deg) / 2.0)
      & (elevation.abs() < math.radians(vfov_deg) / 2.0)
      & (dist < max_range)
    )
    if corrupt:
      std = (noise_std_base + noise_std_per_m * dist).unsqueeze(-1)
      pos_c = pos_c + torch.randn_like(pos_c) * std
      visible &= torch.rand_like(dist) > dropout_prob
    fresh = torch.cat([pos_c, torch.ones_like(dist).unsqueeze(-1)], dim=-1)
    held = self._last.clone()
    held[:, 3] = 0.0
    self._last = torch.where(visible.unsqueeze(-1), fresh, held)
    return self._last.clone()


def ball_visible_fraction(
  env: ManagerBasedRlEnv,
  camera_cfg: SceneEntityCfg,
  camera_offset: tuple[float, float, float] = (0.06, 0.0, 0.07),
  hfov_deg: float = 90.0,
  vfov_deg: float = 60.0,
  max_range: float = 5.0,
) -> torch.Tensor:
  """Metric: 1 where the (noise-free) ball is inside the camera's view."""
  pos_c, _ = ball_in_camera_frame(env, camera_cfg, camera_offset)
  dist = torch.norm(pos_c, dim=-1)
  azimuth = torch.atan2(pos_c[:, 1], pos_c[:, 0])
  elevation = torch.atan2(pos_c[:, 2], pos_c[:, 0].clamp(min=1e-6))
  visible = (
    (pos_c[:, 0] > 0.0)
    & (azimuth.abs() < math.radians(hfov_deg) / 2.0)
    & (elevation.abs() < math.radians(vfov_deg) / 2.0)
    & (dist < max_range)
  )
  return visible.float()
