"""Observability metrics for gait geometry and limb motion.

Pure logging terms (no weight, no gradient) for watching behaviour evolve
over training: foot heel/toe rocking (fore-aft sole pitch), lateral
edge-rocking (medial-lateral sole roll), the duck-walk (foot yaw relative
to the torso), and per-limb-group joint speed (arm/head flail). None of
these shape the policy; they surface in ``Episode_Metrics/*`` so the
timeline of each behaviour is visible in W&B.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "foot_heel_toe_pitch_deg",
  "foot_lateral_roll_deg",
  "foot_toeout_deg",
  "joint_speed_abs",
]

_ROBOT = SceneEntityCfg("robot")
_RAD2DEG = 180.0 / math.pi


def _sole_tilt_components(
  asset: Entity, asset_cfg: SceneEntityCfg, sole_normal_axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (roll, pitch) sole-tilt angles (radians) for the config's feet.

  Gravity projected into each foot frame; the two components tangent to
  the sole normal measure tilt. For the NUgus foot the sole normal is the
  local X axis (``sole_normal_axis=0``); tangent axis 1 is medial-lateral
  roll, tangent axis 2 is fore-aft pitch (see ``feet_flat_orientation``).
  Angle = asin(|component|), i.e. tilt magnitude of the sole plane.
  """
  foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, F, 4]
  num_feet = foot_quat_w.shape[1]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
  gravity_b = quat_apply_inverse(foot_quat_w, gravity_w)  # [B, F, 3]
  tangent_axes = [a for a in range(3) if a != sole_normal_axis]
  roll = torch.asin(gravity_b[..., tangent_axes[0]].clamp(-1.0, 1.0))
  pitch = torch.asin(gravity_b[..., tangent_axes[1]].clamp(-1.0, 1.0))
  return roll, pitch


def _contact_mean(per_foot: torch.Tensor, in_contact: torch.Tensor) -> torch.Tensor:
  """Mean of a per-foot quantity over feet currently in contact.

  Falls back to 0 for envs with no contact (flight/all-swing) so the
  metric never divides by zero.
  """
  weighted = (per_foot * in_contact).sum(dim=1)
  denom = in_contact.sum(dim=1).clamp(min=1.0)
  return weighted / denom


def foot_heel_toe_pitch_deg(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  sole_normal_axis: int = 0,
) -> torch.Tensor:
  """Mean absolute fore-aft sole pitch (deg) over stance feet.

  Heel/toe rocking: a flat-slapping foot reads ~0, a heel-strike/toe-off
  roll reads higher. Stance-gated so swing-phase foot lift is excluded.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  _, pitch = _sole_tilt_components(asset, asset_cfg, sole_normal_axis)
  in_contact = (contact.data.found > 0).float()
  return _contact_mean(pitch.abs() * _RAD2DEG, in_contact)


def foot_lateral_roll_deg(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  sole_normal_axis: int = 0,
) -> torch.Tensor:
  """Mean absolute medial-lateral sole roll (deg) over stance feet.

  Lateral heel/toe: the outside-edge to inside-edge rocking. Flat foot
  reads ~0; edge-walking reads higher. Stance-gated.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  roll, _ = _sole_tilt_components(asset, asset_cfg, sole_normal_axis)
  in_contact = (contact.data.found > 0).float()
  return _contact_mean(roll.abs() * _RAD2DEG, in_contact)


def foot_toeout_deg(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
  torso_cfg: SceneEntityCfg = _ROBOT,
  foot_signs: tuple[float, ...] = (1.0, -1.0),
  command_name: str | None = "twist",
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Signed toe-out (duck) angle (deg): foot yaw relative to torso heading.

  Each foot's yaw is measured in the torso frame; ``foot_signs`` flips the
  right foot so a symmetric toe-out reads positive on both (matching the
  eval-probe ``duck°``). Toes-in (e.g. walking backward) reads negative.
  Averaged over feet; walking-gated so the standing pose (feet forward)
  does not dominate. Near-zero neutral offset (<0.5 deg) is ignored.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, F, 4]
  num_feet = foot_quat.shape[1]
  torso_quat = asset.data.body_link_quat_w[:, torso_cfg.body_ids, :][:, 0, :]  # [B, 4]
  # quat_mul requires matching shapes (no broadcast): expand torso to feet.
  torso_inv = quat_conjugate(torso_quat).unsqueeze(1).expand(-1, num_feet, -1)
  rel = quat_mul(torso_inv.reshape(-1, 4), foot_quat.reshape(-1, 4)).reshape(
    foot_quat.shape
  )  # [B, F, 4]
  w, x, y, z = rel[..., 0], rel[..., 1], rel[..., 2], rel[..., 3]
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # [B, F]
  signs = torch.tensor(foot_signs, device=yaw.device, dtype=yaw.dtype)
  toeout = (yaw * signs).mean(dim=1) * _RAD2DEG  # [B]

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = ((linear_norm + angular_norm) > command_threshold).float()
      toeout = toeout * active
  return toeout


def joint_speed_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Mean absolute joint velocity (rad/s) over the config's joints.

  Point it at the arm or head joint group to watch flail magnitude over
  training (the arms/head carry little balance load, so this is a fairly
  direct read on wasted, non-functional motion).
  """
  asset: Entity = env.scene[asset_cfg.name]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, J]
  return qd.abs().mean(dim=1)
