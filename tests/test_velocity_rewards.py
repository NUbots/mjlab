"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastData, RayCastSensor
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.rewards import (
  _swing_height_profile,
  feet_swing_height_tracking,
  upright,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def _identity_quat(B: int) -> torch.Tensor:
  """(w, x, y, z) = (1, 0, 0, 0)."""
  q = torch.zeros(B, 4)
  q[:, 0] = 1.0
  return q


def _quat_from_roll(roll_rad: float, B: int = 1) -> torch.Tensor:
  roll = torch.full((B,), roll_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(roll, zero, zero)


def _quat_from_pitch(pitch_rad: float, B: int = 1) -> torch.Tensor:
  pitch = torch.full((B,), pitch_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(zero, pitch, zero)


def _make_env_and_reward(
  terrain_sensor_names: tuple[str, ...] | None = None,
  body_quat_w: torch.Tensor | None = None,
  terrain_hit_z: float = 0.0,
  terrain_slope_x: float = 0.0,
):
  """Build mocked env + upright reward instance.

  Args:
    terrain_sensor_names: If set, enables terrain-aware mode.
    body_quat_w: [B, 4] root orientation. Defaults to identity.
    terrain_hit_z: Z value for flat terrain hits.
    terrain_slope_x: Slope in X (z = terrain_slope_x * x).
  """
  B = 1 if body_quat_w is None else body_quat_w.shape[0]
  if body_quat_w is None:
    body_quat_w = _identity_quat(B)

  # Mock asset data. Use explicit asset_cfg with no body_names so
  # body_ids stays None and the reward uses root_link_quat_w.
  asset = MagicMock()
  asset.data.root_link_quat_w = body_quat_w
  asset.data.root_link_pos_w = torch.zeros(B, 3)
  asset.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(B, 3)
  asset_cfg = SceneEntityCfg("robot", body_names=None, body_ids=[])

  # Mock terrain sensor if needed.
  sensors: dict = {"robot": asset}
  if terrain_sensor_names is not None:
    N = 100
    torch.manual_seed(0)
    hit_pos = torch.zeros(B, N, 3)
    hit_pos[:, :, 0] = torch.randn(B, N)
    hit_pos[:, :, 1] = torch.randn(B, N)
    hit_pos[:, :, 2] = terrain_hit_z + terrain_slope_x * hit_pos[:, :, 0]

    raycast_sensor = MagicMock(spec=RayCastSensor)
    raycast_data = RayCastData(
      distances=torch.ones(B, N),
      normals_w=torch.zeros(B, N, 3),
      hit_pos_w=hit_pos,
      pos_w=torch.zeros(B, 3),
      quat_w=torch.zeros(B, 4),
      frame_pos_w=torch.zeros(B, 1, 3),
      frame_quat_w=torch.zeros(B, 1, 4),
    )
    type(raycast_sensor).data = PropertyMock(return_value=raycast_data)
    for name in terrain_sensor_names:
      sensors[name] = raycast_sensor

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])

  params: dict = {"std": 1.0, "asset_cfg": asset_cfg}
  if terrain_sensor_names is not None:
    params["terrain_sensor_names"] = terrain_sensor_names
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = params

  reward_fn = upright(cfg, env)
  return env, reward_fn, params


def _make_swing_env(
  air_time: torch.Tensor,
  foot_heights: torch.Tensor,
  command: torch.Tensor | None = None,
):
  """Build a mocked env with a contact sensor + terrain height sensor."""
  contact = MagicMock(spec=ContactSensor)
  contact.data.current_air_time = air_time

  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.data.heights = foot_heights

  sensors = {"feet_ground_contact": contact, "foot_height_scan": height_sensor}
  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])
  env.extras = {"log": {}}
  if command is None:
    env.command_manager.get_command = MagicMock(return_value=None)
  else:
    env.command_manager.get_command = MagicMock(return_value=command)
  return env


def _swing_reward(env, **overrides):
  params = dict(
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target_height=0.1,
    swing_duration=0.3,
    std=0.05,
    profile="sin",
  )
  params.update(overrides)
  return feet_swing_height_tracking(env, **params)


def test_swing_profiles_peak_at_midswing():
  """Both profiles vanish at the endpoints and peak at the amplitude mid-swing."""
  psi = torch.tensor([0.0, 0.5, 1.0])
  for profile in ("sin", "bump"):
    h = _swing_height_profile(psi, amplitude=0.1, profile=profile)
    assert h[0].item() == 0.0
    assert h[2].item() < 1e-6
    torch.testing.assert_close(h[1], torch.tensor(0.1), atol=1e-6, rtol=0)


def test_swing_tracking_rewards_matching_height():
  """A swing foot sitting exactly on the desired profile scores ~1 per foot."""
  # Two feet, both mid-swing (psi = 0.5) → desired height = amplitude = 0.1.
  air_time = torch.tensor([[0.15, 0.15]])  # swing_duration/2
  foot_heights = torch.tensor([[0.1, 0.1]])
  env = _make_swing_env(air_time, foot_heights)
  r = _swing_reward(env)
  assert r.shape == (1,)
  torch.testing.assert_close(r, torch.tensor([2.0]), atol=1e-4, rtol=0)


def test_swing_tracking_penalizes_deviation():
  """A foot far from the target scores well below a foot that matches it."""
  air_time = torch.tensor([[0.15, 0.15]])
  foot_heights = torch.tensor([[0.1, 0.3]])  # second foot way too high
  env = _make_swing_env(air_time, foot_heights)
  r = _swing_reward(env)
  # First foot ~1, second foot strongly attenuated.
  assert 1.0 < r.item() < 1.05


def test_swing_tracking_ignores_stance_feet():
  """A foot in contact (air_time == 0) contributes nothing."""
  air_time = torch.tensor([[0.0, 0.15]])  # first foot in stance
  foot_heights = torch.tensor([[0.1, 0.1]])
  env = _make_swing_env(air_time, foot_heights)
  r = _swing_reward(env)
  torch.testing.assert_close(r, torch.tensor([1.0]), atol=1e-4, rtol=0)


def test_swing_tracking_command_gate():
  """Reward is zeroed when the command is below threshold."""
  air_time = torch.tensor([[0.15, 0.15]])
  foot_heights = torch.tensor([[0.1, 0.1]])
  command = torch.tensor([[0.0, 0.0, 0.0]])
  env = _make_swing_env(air_time, foot_heights, command=command)
  r = _swing_reward(env, command_name="twist", command_threshold=0.05)
  torch.testing.assert_close(r, torch.tensor([0.0]), atol=1e-6, rtol=0)


def test_swing_tracking_phase_saturates():
  """A foot lingering past the swing duration is pulled back toward the ground."""
  # Overrun: air_time > swing_duration → psi clamps to 1 → desired height ~0.
  air_time = torch.tensor([[0.6]])
  on_ground = _make_swing_env(air_time, torch.tensor([[0.0]]))
  still_high = _make_swing_env(air_time, torch.tensor([[0.1]]))
  r_ground = _swing_reward(on_ground)
  r_high = _swing_reward(still_high)
  # Being on the ground when phase has saturated is rewarded; staying high isn't.
  assert r_ground.item() > 0.99
  assert r_high.item() < r_ground.item()


def test_world_up_identity_gives_max_reward():
  """Perfectly upright robot on flat ground → reward ≈ 1."""
  env, reward, params = _make_env_and_reward()
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.shape == (1,)
  assert r.item() > 0.99


def test_world_up_tilted_gives_lower_reward():
  """30° roll → reward significantly below 1."""
  quat = _quat_from_roll(math.radians(30))
  env, reward, params = _make_env_and_reward(body_quat_w=quat)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.item() < 0.8


def test_terrain_aware_aligned_with_slope():
  """Robot pitched to match a slope → terrain-aware reward ≈ 1."""
  slope = 0.5  # z = 0.5 * x
  tilt = math.atan(slope)  # Pitch to match slope in XZ plane.
  quat = _quat_from_pitch(-tilt)
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be close to 1 since robot matches terrain.
  assert r.item() > 0.9


def test_terrain_aware_upright_on_slope_penalized():
  """Robot staying vertical on a slope → terrain-aware reward < 1."""
  slope = 0.5
  quat = _identity_quat(1)  # Robot is world-vertical, not matching slope.
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be penalized since robot doesn't match terrain.
  assert r.item() < 0.95


def test_terrain_aware_flat_ground_matches_world_up():
  """On flat terrain, terrain-aware and world-up should give same reward."""
  quat = _quat_from_roll(math.radians(15))
  env_t, reward_t, params_t = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
  )
  env_w, reward_w, params_w = _make_env_and_reward(body_quat_w=quat)

  r_terrain = reward_t(
    env_t,
    std=params_t["std"],
    asset_cfg=params_t["asset_cfg"],
    terrain_sensor_names=params_t["terrain_sensor_names"],
  )
  r_world = reward_w(env_w, std=params_w["std"], asset_cfg=params_w["asset_cfg"])

  torch.testing.assert_close(r_terrain, r_world, atol=0.02, rtol=0.02)


def test_batch_consistency():
  """Multiple envs with different orientations get independent rewards."""
  B = 4
  quats = torch.zeros(B, 4)
  quats[:, 0] = 1.0  # All identity.
  # Tilt env 2 by 45°.
  quats[2] = _quat_from_roll(math.radians(45))[0]

  env, reward, params = _make_env_and_reward(body_quat_w=quats)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])

  assert r.shape == (B,)
  # Env 0, 1, 3 should be ~1, env 2 should be lower.
  assert r[0].item() > 0.99
  assert r[1].item() > 0.99
  assert r[2].item() < 0.7
  assert r[3].item() > 0.99
