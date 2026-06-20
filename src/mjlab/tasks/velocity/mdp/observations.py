from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_clock(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
  """Independent gait-clock phase as ``[sin(2*pi*phase), cos(2*pi*phase)]``.

  The phase advances with episode time and resets with the episode,
  ``phase = (episode_time / period) mod 1``, giving the policy a periodic signal
  it can condition its swing on. ``period`` is the full gait-cycle duration and
  must match the ``period`` of the :func:`mjlab.tasks.velocity.mdp.rewards.
  feet_swing_height_clock` reward so the observed clock and the rewarded clock
  agree.

  Returns:
    Tensor of shape [B, 2].
  """
  t = env.episode_length_buf.float() * env.step_dt  # [B]
  angle = 2.0 * math.pi * torch.remainder(t / period, 1.0)  # [B]
  return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, 2]


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
