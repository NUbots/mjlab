from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  power: int = 1,
  only_below: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot speed.

  Horizontal foot speed acts as a swing/stance gate: at lift-off and touchdown
  the foot is necessarily low *and* slow, so the penalty fades there and the
  term only shapes clearance while the foot is travelling mid-swing.

  Args:
    power: Exponent on the height error. ``1`` is the original absolute
      deviation; ``2`` gives a squared error whose gradient grows the further
      below target the foot sits, so it is harder to "buy out" with a small
      constant offset.
    only_below: If ``True``, only penalize the foot for being *below* the
      target. Clearing higher than ``target_height`` is then never penalized,
      removing the symmetric pull that drags a high apex back down to target.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  if only_below:
    delta = (foot_height - target_height).neg().clamp(min=0.0)  # [B, F]
  else:
    delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta.pow(power) * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def _swing_height_profile(
  phase: torch.Tensor,
  amplitude: float,
  profile: Literal["sin", "bump"],
) -> torch.Tensor:
  """Desired foot height as a function of normalized swing phase ``psi``.

  ``phase`` (``psi``) runs from 0 at takeoff to 1 at the nominal landing. Both
  profiles vanish at the endpoints and peak at ``amplitude`` mid-swing, so the
  target traces a single lift-and-lower arc:

  - ``"sin"``: ``A * sin(pi * psi)``. Smooth, but has nonzero slope at the
    endpoints (the foot is still rising at takeoff / falling at landing).
  - ``"bump"``: ``A * 16 * psi^2 * (1 - psi)^2``. A quartic bump with zero
    slope at both endpoints, so the target eases away from and back to the
    ground -- gentler near takeoff and touchdown.
  """
  if profile == "sin":
    return amplitude * torch.sin(math.pi * phase)
  if profile == "bump":
    return amplitude * 16.0 * torch.square(phase) * torch.square(1.0 - phase)
  raise ValueError(f"Unknown swing-height profile '{profile}'.")


def feet_swing_height_clock(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  target_height: float,
  period: float,
  swing_ratio: float,
  std: float,
  foot_offsets: tuple[float, ...] = (0.0, 0.5),
  profile: Literal["sin", "bump"] = "sin",
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Dense swing-height tracking against an *independent* gait clock.

  Unlike :class:`feet_swing_height` (sparse, scored once at landing) and unlike
  an air-time-driven phase (self-referential: the target adapts to whatever the
  foot does, so a quick low step always matches a low target), this term drives
  the desired height from a fixed-frequency clock that the policy does not
  control. The clock advances with episode time and resets with the episode::

      base_phase = (episode_time / period) mod 1
      foot_phase = (base_phase + foot_offset) mod 1   # offsets put feet out of
                                                      # phase, e.g. (0, 0.5)

  Each foot's cycle is split into a swing window ``foot_phase < swing_ratio``
  (desired height follows the lift-and-lower arc, see
  :func:`_swing_height_profile`) and a stance window (desired height 0, i.e.
  foot on the ground). The foot is scored every step by
  ``exp(-(h - h_des)^2 / std^2)`` against its measured terrain clearance, summed
  over feet. Because the target reaches ``target_height`` mid-swing regardless
  of what the foot is doing, a foot that fails to lift on schedule is genuinely
  penalized -- which is what forces a higher, slower step.

  ``period`` is the full gait-cycle duration (both feet complete one
  swing+stance); a larger ``period`` commands a slower cadence. Feed the same
  clock to the policy as an observation (``mdp.gait_clock`` with a matching
  ``period``) so it can act periodically.
  """
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    "feet_swing_height_clock requires a TerrainHeightSensor, got "
    f"{type(height_sensor).__name__}"
  )

  foot_heights = height_sensor.data.heights  # [B, F]
  num_feet = foot_heights.shape[1]
  assert len(foot_offsets) == num_feet, (
    f"foot_offsets has {len(foot_offsets)} entries but sensor reports {num_feet} feet."
  )

  t = env.episode_length_buf.float() * env.step_dt  # [B]
  base_phase = torch.remainder(t / period, 1.0)  # [B]
  offsets = torch.tensor(foot_offsets, device=env.device, dtype=foot_heights.dtype)
  foot_phase = torch.remainder(base_phase[:, None] + offsets[None, :], 1.0)  # [B, F]

  in_swing = foot_phase < swing_ratio  # [B, F]
  psi = torch.clamp(foot_phase / swing_ratio, 0.0, 1.0)  # [B, F]
  arc = _swing_height_profile(psi, target_height, profile)  # [B, F]
  desired = torch.where(in_swing, arc, torch.zeros_like(arc))  # [B, F]

  error = foot_heights - desired  # [B, F]
  tracking = torch.exp(-torch.square(error) / std**2)  # [B, F]
  reward = torch.sum(tracking, dim=1)  # [B]

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      reward = reward * active

  swing_mask = in_swing.float()
  num_swing = torch.clamp(torch.sum(swing_mask), min=1.0)
  mean_swing_error = torch.sum(torch.abs(error) * swing_mask) / num_swing
  env.extras["log"]["Metrics/swing_clock_error_mean"] = mean_swing_error
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_flat_orientation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  sole_normal_axis: int = 2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot-sole tilt during swing to encourage flat-footed stepping.

  The foot sole lies in the plane perpendicular to one of the foot body's local
  axes (``sole_normal_axis``). Projecting world gravity into the foot frame gives
  a unit vector that points purely along that axis when the sole is level; its
  two in-plane (tangent) components measure tilt. Penalizing them keeps the sole
  parallel to the ground so the toe/front edge does not pitch down and dig in on
  touchdown.

  For the Nugus foot the four corner sites share the same local-X coordinate, so
  the sole normal is the local X axis (``sole_normal_axis=0``); the tangent
  components then correspond to fore-aft pitch and medial-lateral roll.

  Only swing feet (no ground contact) are penalized so the term does not fight
  terrain conformance during stance.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, F, 4]
  num_feet = foot_quat_w.shape[1]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
  gravity_b = quat_apply_inverse(foot_quat_w, gravity_w)  # [B, F, 3]

  tangent_axes = [a for a in range(3) if a != sole_normal_axis]
  tilt = torch.sum(torch.square(gravity_b[..., tangent_axes]), dim=-1)  # [B, F]

  in_air = (contact_sensor.data.found == 0).float()  # [B, F]
  cost = torch.sum(tilt * in_air, dim=1)  # [B]

  command = env.command_manager.get_command(command_name)
  if command is not None:
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    cost = cost * active

  num_in_air = torch.clamp(torch.sum(in_air), min=1)
  env.extras["log"]["Metrics/foot_tilt_mean"] = torch.sum(tilt * in_air) / num_in_air
  return cost


class cost_of_transport_proxy:
  """Penalize energy per traveled distance to improve transport efficiency.

  This computes an online CoT proxy:

    sum(max(0, tau * qd)) / max(horizontal_speed, speed_floor)

  For a fixed robot, true CoT differs by a constant factor (mass * gravity),
  so this proxy is sufficient for reward shaping.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    actuator_ids, _ = asset.find_actuators(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    speed_floor: float = 0.1,
    command_name: str | None = None,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech_pos = torch.clamp(tau * qd, min=0.0)
    mechanical_power = torch.sum(mech_pos, dim=1)

    horizontal_speed = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
    cot_proxy = mechanical_power / torch.clamp(horizontal_speed, min=speed_floor)

    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        cot_proxy = cot_proxy * active

    env.extras["log"]["Metrics/cot_proxy_mean"] = torch.mean(cot_proxy)
    env.extras["log"]["Metrics/locomotion_speed_mean"] = torch.mean(horizontal_speed)
    return cot_proxy


def gait_phase_regularity_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  eps: float = 1e-6,
) -> torch.Tensor:
  """Penalize irregular left-right gait timing using contact phase durations.

  Uses coefficient of variation (CV) across feet for completed swing and stance
  durations from the contact sensor's airtime tracker.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.last_air_time is not None
  assert sensor_data.last_contact_time is not None

  last_air_time = sensor_data.last_air_time
  last_contact_time = sensor_data.last_contact_time

  num_feet = last_air_time.shape[1]
  if num_feet < 2:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  air_var = torch.var(last_air_time, dim=1, unbiased=False)
  contact_var = torch.var(last_contact_time, dim=1, unbiased=False)
  air_mean = torch.mean(last_air_time, dim=1)
  contact_mean = torch.mean(last_contact_time, dim=1)

  air_cv = torch.sqrt(air_var) / torch.clamp(air_mean, min=eps)
  contact_cv = torch.sqrt(contact_var) / torch.clamp(contact_mean, min=eps)
  cost = air_cv + contact_cv

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active

  env.extras["log"]["Metrics/gait_air_cv_mean"] = torch.mean(air_cv)
  env.extras["log"]["Metrics/gait_contact_cv_mean"] = torch.mean(contact_cv)
  return cost


def feet_lateral_distance_cost(
  env: ManagerBasedRlEnv,
  nominal_distance: float,
  sharpness: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet whose lateral body-frame separation falls below nominal.

  The foot-to-foot vector is expressed in the robot's body frame so that yaw
  rotation does not affect the measurement. Only the body-Y component is used,
  isolating lateral spread from fore-aft offset during a stride.


  The penalty shape is ``exp(sharpness * shortfall) - 1`` where
  ``shortfall = max(0, abs(nominal_distance - lateral_distance))``.
  When nominal_distance == lateral_distance the cost is zero. For
  anything else, the cost grows exponentially with the difference.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, N, 3]

  num_feet = foot_pos_w.shape[1]
  if num_feet < 2:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  root_quat_w = asset.data.root_link_quat_w  # [B, 4]

  pair_i, pair_j = torch.triu_indices(num_feet, num_feet, offset=1)
  foot_a = foot_pos_w[:, pair_i, :]  # [B, P, 3]
  foot_b = foot_pos_w[:, pair_j, :]  # [B, P, 3]

  num_pairs = pair_i.shape[0]
  # Rotate foot-to-foot vectors into body frame.
  quat_exp = root_quat_w.unsqueeze(1).expand(-1, num_pairs, -1)  # [B, P, 4]
  delta_b = quat_apply_inverse(quat_exp, foot_a - foot_b)  # [B, P, 3]

  pair_distance = torch.abs(delta_b[..., 1])  # body-Y component [B, P]

  shortfall = torch.clamp(abs(nominal_distance - pair_distance), min=0.0)
  cost = torch.sum(torch.exp(sharpness * shortfall) - 1.0, dim=1)

  min_pair_distance = torch.amin(pair_distance, dim=1)
  max_pair_distance = torch.amax(pair_distance, dim=1)
  env.extras["log"]["Metrics/min_foot_lateral_distance_mean"] = torch.mean(
    min_pair_distance
  )
  env.extras["log"]["Metrics/max_foot_lateral_distance_mean"] = torch.mean(
    max_pair_distance
  )
  return cost


class actuator_torque_rate_l2:
  """Penalize rapid actuator torque changes (shuffle reversals)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    actuator_ids, _ = asset.find_actuators(
      cfg.params["asset_cfg"].joint_names,
    )
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)
    self._prev_tau = torch.zeros(
      (env.num_envs, len(actuator_ids)), device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    cost = torch.sum(torch.square(tau - self._prev_tau), dim=1)
    self._prev_tau = tau.clone()
    return cost

  def reset(self, env_ids: torch.Tensor) -> None:
    self._prev_tau[env_ids] = 0.0


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class left_right_joint_symmetry_cost:
  """Penalize left-right asymmetry for matched joint pairs.

  The cost compares left/right joint motion magnitudes around the default pose,
  which encourages balanced limb usage without forcing identical joint signs.
  """

  # Common naming conventions for left-right counterparts.
  _DEFAULT_NAME_SUBSTITUTIONS = (
    ("left_", "right_"),
    ("right_", "left_"),
    ("_left", "_right"),
    ("_right", "_left"),
    ("left", "right"),
    ("right", "left"),
    ("_l_", "_r_"),
    ("_r_", "_l_"),
    ("_l", "_r"),
    ("_r", "_l"),
    ("l_", "r_"),
    ("r_", "l_"),
    ("FL_", "FR_"),
    ("FR_", "FL_"),
    ("RL_", "RR_"),
    ("RR_", "RL_"),
    ("_FL", "_FR"),
    ("_FR", "_FL"),
    ("_RL", "_RR"),
    ("_RR", "_RL"),
  )

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    name_to_joint_id = dict(zip(joint_names, joint_ids, strict=False))

    substitutions = cfg.params.get(
      "name_substitutions", self._DEFAULT_NAME_SUBSTITUTIONS
    )
    substitution_pairs = [
      (str(left), str(right)) for left, right in substitutions if left and right
    ]

    pair_set: set[tuple[int, int]] = set()
    for name, joint_id in zip(joint_names, joint_ids, strict=False):
      for src, dst in substitution_pairs:
        if src in name:
          counterpart_name = re.sub(re.escape(src), dst, name, count=1)
          counterpart_id = name_to_joint_id.get(counterpart_name)
          if counterpart_id is not None and counterpart_id != joint_id:
            pair_set.add((min(joint_id, counterpart_id), max(joint_id, counterpart_id)))

    sorted_pairs = sorted(pair_set)
    if sorted_pairs:
      self._joint_a_ids = torch.tensor(
        [pair[0] for pair in sorted_pairs], device=env.device, dtype=torch.long
      )
      self._joint_b_ids = torch.tensor(
        [pair[1] for pair in sorted_pairs], device=env.device, dtype=torch.long
      )
    else:
      self._joint_a_ids = torch.empty(0, device=env.device, dtype=torch.long)
      self._joint_b_ids = torch.empty(0, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    position_weight: float = 1.0,
    velocity_weight: float = 0.1,
  ) -> torch.Tensor:
    if self._joint_a_ids.numel() == 0:
      return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None

    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel

    dev_a = joint_pos[:, self._joint_a_ids] - default_joint_pos[:, self._joint_a_ids]
    dev_b = joint_pos[:, self._joint_b_ids] - default_joint_pos[:, self._joint_b_ids]
    pos_cost = torch.mean(torch.square(torch.abs(dev_a) - torch.abs(dev_b)), dim=1)

    vel_a = joint_vel[:, self._joint_a_ids]
    vel_b = joint_vel[:, self._joint_b_ids]
    vel_cost = torch.mean(torch.square(torch.abs(vel_a) - torch.abs(vel_b)), dim=1)

    env.extras["log"]["Metrics/symmetry_pos_cost_mean"] = torch.mean(pos_cost)
    env.extras["log"]["Metrics/symmetry_vel_cost_mean"] = torch.mean(vel_cost)

    return position_weight * pos_cost + velocity_weight * vel_cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
