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
  "flight_fraction",
  "foot_heel_toe_pitch_deg",
  "foot_lateral_roll_deg",
  "foot_toein_deg",
  "foot_toeout_deg",
  "foot_torso_yaw_signed",
  "gait_stance_asymmetry",
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


def foot_torso_yaw_signed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  foot_signs: tuple[float, ...] = (1.0, -1.0),
) -> torch.Tensor:
  """Per-foot signed yaw relative to the torso heading, radians ``[B, F]``.

  ``foot_signs`` flips the right foot so toe-OUT reads positive on both
  feet and toe-IN negative. Shared by the ``foot_toeout_deg`` metric and
  the ``foot_toein_cost`` reward so the two cannot disagree on geometry.
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
  return yaw * signs


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
  signed_yaw = foot_torso_yaw_signed(env, asset_cfg, torso_cfg, foot_signs)
  toeout = signed_yaw.mean(dim=1) * _RAD2DEG  # [B]

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = ((linear_norm + angular_norm) > command_threshold).float()
      toeout = toeout * active
  return toeout


def foot_toein_deg(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
  torso_cfg: SceneEntityCfg = _ROBOT,
  foot_signs: tuple[float, ...] = (1.0, -1.0),
  command_name: str | None = "twist",
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Inward-only (pigeon-toe) foot yaw magnitude (deg), max over feet.

  ``foot_toeout_deg`` is a signed MEAN over feet and envs, so a policy
  that toes-in for one command regime while toeing-out hard elsewhere
  still reads positive — the v55 inward-duck sighting was invisible in
  the mean. This logs only the inward component (zero when toed out), so
  any pigeon-toe shows up regardless of what the average is doing.
  Walking-gated like the signed metric.
  """
  signed_yaw = foot_torso_yaw_signed(env, asset_cfg, torso_cfg, foot_signs)
  toein = torch.clamp(-signed_yaw, min=0.0).max(dim=1).values * _RAD2DEG  # [B]

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = ((linear_norm + angular_norm) > command_threshold).float()
      toein = toein * active
  return toein


def flight_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  command_name: str | None = "twist",
  command_threshold: float = 0.05,
  min_command_speed: float = 0.0,
  max_tilt_cos: float = 0.8,
  min_episode_steps: int = 25,
) -> torch.Tensor:
  """Fraction of steps in TRUE flight: all feet airborne while upright.

  Running is defined by a flight phase; this tracks the walk->run
  boundary crossing without letting falls masquerade as airtime: frames
  only count when every foot is off the ground AND the torso is upright
  (``-projected_gravity_b.z >= max_tilt_cos``, i.e. tilt under ~37 deg
  at the 0.8 default — a falling robot is tilted, a fallen one
  terminates). Walking-gated; ``min_command_speed`` optionally restricts
  to fast commands (e.g. 1.5 m/s, just under the NUgus walk->run Froude
  boundary v* = 1.56 m/s) so the boundary is visible undiluted by the
  slow half of the command envelope. Episode mean = flight fraction.
  """
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  airborne = (contact.data.found == 0).all(dim=1)  # [B]

  asset: Entity = env.scene[asset_cfg.name]
  upright = -asset.data.projected_gravity_b[:, 2] >= max_tilt_cos  # [B]

  # Spawn exclusion (Trent 2026-07-14): envs are dropped in at reset, so
  # every episode's first frames are airborne "flight" - a constant
  # artifact floor (~drop_frames/episode_len). Skip the settle window.
  settled = env.episode_length_buf > min_episode_steps
  flight = (airborne & upright & settled).float()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = (linear_norm + angular_norm) > command_threshold
      if min_command_speed > 0.0:
        active = active & (linear_norm >= min_command_speed)
      flight = flight * active.float()
  return flight


def gait_stance_asymmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  signed: bool = True,
  command_name: str | None = "twist",
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Left-minus-right stance-time asymmetry (chirality vs adaptation).

  Per step: ``contact(left) - contact(right)`` (sensor body order is
  (left, right)), walking-gated; the episode mean is the signed
  stance-time split. Interpretation under per-side DR draws (v60+,
  where per-foot friction and per-servo gains legitimately create
  asymmetric robots): the SIGNED metric near zero with per-env spread =
  the policy adapts its limp to each episode's draw (healthy); a biased
  signed mean across the population = a fixed chirality leaking through
  the mirror constraint (trigger for backlog 15e twin RNNs). Pair with
  ``signed=False`` for the magnitude of per-episode limping.
  """
  contact: ContactSensor = env.scene[sensor_name]
  assert contact.data.found is not None
  in_contact = (contact.data.found > 0).float()  # [B, 2] = (left, right)
  diff = in_contact[:, 0] - in_contact[:, 1]
  asym = diff if signed else diff.abs()
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = (linear_norm + angular_norm) > command_threshold
      asym = asym * active.float()
  return asym


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
