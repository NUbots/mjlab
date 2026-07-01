from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary

import torch

from mjlab.envs.mdp.dr.actuator import get_current_sensor_buffers
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Per-(env, asset) cache of the per-actuator torque-constant (Kt) tensor used by
# ``actuator_current``. Built once from the resolved actuator names and reused.
_CURRENT_KT_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _gait_base_phase(
  env: ManagerBasedRlEnv,
  period: float,
  phase_source: Literal["time", "policy"],
  action_term_name: str,
) -> torch.Tensor:
  """Normalized gait phase in ``[0, 1)`` from episode time or policy state."""
  if phase_source == "time":
    t = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(t / period, 1.0)
  if phase_source == "policy":
    from mjlab.envs.mdp.actions.phase_delta import get_phase_delta_action

    return get_phase_delta_action(env, action_term_name).policy_phase
  raise ValueError(f"Unknown phase_source {phase_source!r}; use 'time' or 'policy'.")


def gait_clock(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  silence_stages: list[dict[str, float]] | None = None,
  phase_source: Literal["time", "policy"] = "time",
  action_term_name: str = "phase_delta",
) -> torch.Tensor:
  """Independent gait-clock phase as ``[sin(2*pi*phase), cos(2*pi*phase)]``.

  The phase advances with episode time and resets with the episode,
  ``phase = (episode_time / period) mod 1``, giving the policy a periodic signal
  it can condition its swing on. ``period`` is the full gait-cycle duration and
  must match the ``period`` of the :func:`mjlab.tasks.velocity.mdp.rewards.
  feet_swing_height_clock` reward so the observed clock and the rewarded clock
  agree.

  When ``command_name`` is given, the clock is zeroed for environments whose
  command magnitude is below ``command_threshold``. This collapses the signal to
  the origin (off the unit circle) at standstill, presenting the policy with a
  distinct "standing" input rather than a walking phase. Without this gate the
  clock keeps sweeping at zero command and drives the policy to march on the
  spot even though the gait reward is gated off. Use the same ``command_name``
  and ``command_threshold`` as the matching
  :func:`mjlab.tasks.velocity.mdp.rewards.feet_swing_height_clock` reward so the
  observed clock vanishes exactly when the reward turns off.

  When ``silence_stages`` is given, the clock output is multiplied by a global
  stepped scale read from ``env.common_step_counter``. Each stage is a dict with
  an integer ``step`` threshold and a float ``scale``; the scale of the last
  stage whose ``step`` has been reached is applied (a step function, no
  interpolation, matching :class:`mjlab.envs.mdp.curriculums.reward_curriculum`).
  Supplying stages that ramp the scale ``1.0 -> 0.0`` over the same boundaries as
  the clock-reward anneal fades the observed clock out as the policy is weaned
  off the clock, so the deployed policy does not depend on a phase signal it
  would not have on hardware. Stages must be in nondecreasing ``step`` order.

  Returns:
    Tensor of shape [B, 2].
  """
  phase = _gait_base_phase(env, period, phase_source, action_term_name)  # [B]
  angle = 2.0 * math.pi * phase  # [B]
  clock = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, 2]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float().unsqueeze(-1)  # [B, 1]
      clock = clock * active
  if silence_stages:
    step = env.common_step_counter
    scale = 1.0
    for stage in silence_stages:
      if step >= stage["step"]:
        scale = stage["scale"]
    clock = clock * scale
  return clock  # [B, 2]


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def _build_kt_tensor(
  env: ManagerBasedRlEnv,
  asset: Entity,
  cache_key: str,
  actuator_ids: torch.Tensor,
  kt: float | dict[str, float],
) -> torch.Tensor:
  """Build (and cache) the [1, n] per-actuator torque-constant tensor.

  ``kt`` is either a scalar applied to every actuator, or a mapping of regex
  pattern -> Kt (Nm/A). For the dict form, each actuator's name is matched
  against the patterns in order and the first match wins; a ``"default"`` key
  (or 2.0) is used for unmatched actuators.
  """
  store = _CURRENT_KT_CACHE.setdefault(env, {})
  cached = store.get(cache_key)
  if cached is not None and cached.shape == (1, int(actuator_ids.numel())):
    return cached

  if isinstance(kt, (int, float)):
    values = [float(kt)] * int(actuator_ids.numel())
  else:
    default = float(kt.get("default", 2.0))
    patterns = [(p, v) for p, v in kt.items() if p != "default"]
    names = list(asset.actuator_names)
    values = []
    for idx in actuator_ids.tolist():
      name = names[idx]
      matched = default
      for pattern, value in patterns:
        if re.search(pattern, name):
          matched = float(value)
          break
      values.append(matched)
  kt_tensor = torch.tensor(values, device=env.device).unsqueeze(0)  # [1, n]
  store[cache_key] = kt_tensor
  return kt_tensor


def actuator_current(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  kt: float | dict[str, float] = 2.0,
  quantize: float | None = None,
) -> torch.Tensor:
  """Per-actuator estimated electrical current ``i = tau / Kt`` (Amps).

  Mirrors a Dynamixel "present current" sensor: the actuator output torque is
  divided by the per-servo torque constant ``Kt`` (Nm/A) to recover current.
  A per-servo gain/offset calibration buffer (randomized by the
  :func:`mjlab.envs.mdp.dr.current_sensor` reset event) is applied as
  ``sensed = i * gain + offset`` to model servo-to-servo variation and bias;
  the buffer defaults to identity (gain=1, offset=0) until that event runs.
  Optional ``quantize`` rounds to a current resolution in Amps (e.g. the
  XH540-W270 present-current unit of 2.69 mA = ``0.00269``).

  Returns:
    Tensor of shape [B, n_actuators].
  """
  asset: Entity = env.scene[asset_cfg.name]
  gain, offset, actuator_ids = get_current_sensor_buffers(env, asset_cfg)
  kt_tensor = _build_kt_tensor(env, asset, asset_cfg.name, actuator_ids, kt)
  tau = asset.data.actuator_force[:, actuator_ids]  # [B, n]
  current = tau / kt_tensor
  current = current * gain + offset
  if quantize is not None and quantize > 0:
    current = torch.round(current / quantize) * quantize
  return current
