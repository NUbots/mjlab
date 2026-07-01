"""Tests for the actuator_current observation and its current-sensor DR."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mjlab.envs.mdp.dr.actuator import current_sensor, get_current_sensor_buffers
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.observations import actuator_current


def _make_env(actuator_force: torch.Tensor, names: list[str]) -> MagicMock:
  num_envs, num_act = actuator_force.shape
  asset = MagicMock()
  asset.num_actuators = num_act
  asset.actuator_names = names
  asset.data.actuator_force = actuator_force
  env = MagicMock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.scene = {"robot": asset}
  return env


def test_actuator_current_scalar_kt():
  force = torch.tensor([[2.0, 4.0, 6.0]])
  env = _make_env(force, ["a", "b", "c"])
  cfg = SceneEntityCfg("robot")
  current = actuator_current(env, asset_cfg=cfg, kt=2.0)
  assert current.shape == (1, 3)
  torch.testing.assert_close(current, torch.tensor([[1.0, 2.0, 3.0]]))


def test_actuator_current_per_group_kt():
  # tau / Kt with different constants per matched group.
  force = torch.tensor([[3.0, 3.0]])
  env = _make_env(force, ["left_knee_pitch", "left_shoulder_pitch"])
  cfg = SceneEntityCfg("robot")
  kt = {r"(shoulder|elbow|neck|head)": 1.5, "default": 2.0}
  current = actuator_current(env, asset_cfg=cfg, kt=kt)
  # knee -> default 2.0 (3/2=1.5); shoulder -> 1.5 (3/1.5=2.0).
  torch.testing.assert_close(current, torch.tensor([[1.5, 2.0]]))


def test_actuator_current_quantization():
  force = torch.tensor([[1.0]])
  env = _make_env(force, ["a"])
  cfg = SceneEntityCfg("robot")
  # current = 1/2 = 0.5 A; quantize to 2.69 mA resolution.
  q = 0.00269
  current = actuator_current(env, asset_cfg=cfg, kt=2.0, quantize=q)
  expected = round(0.5 / q) * q
  torch.testing.assert_close(current, torch.tensor([[expected]]))


def test_current_sensor_dr_applies_gain_and_offset():
  force = torch.tensor([[2.0, 2.0]])
  env = _make_env(force, ["a", "b"])
  cfg = SceneEntityCfg("robot")
  # Deterministic gain=1.0, offset=0.5 across the (identical) ranges.
  current_sensor(
    env,
    env_ids=None,
    gain_range=(1.0, 1.0),
    offset_range=(0.5, 0.5),
    asset_cfg=cfg,
  )
  # Observation reads the same buffers: sensed = tau/Kt * gain + offset.
  current = actuator_current(env, asset_cfg=cfg, kt=2.0)
  torch.testing.assert_close(current, torch.tensor([[1.5, 1.5]]))


def test_current_sensor_buffers_default_identity():
  force = torch.tensor([[2.0, 2.0]])
  env = _make_env(force, ["a", "b"])
  cfg = SceneEntityCfg("robot")
  gain, offset, actuator_ids = get_current_sensor_buffers(env, cfg)
  assert torch.equal(gain, torch.ones(1, 2))
  assert torch.equal(offset, torch.zeros(1, 2))
  assert torch.equal(actuator_ids, torch.tensor([0, 1]))
