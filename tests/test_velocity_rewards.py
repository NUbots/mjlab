"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastData, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.observations import gait_clock
from mjlab.tasks.velocity.mdp.rewards import (
  _swing_height_profile,
  feet_swing_height_clock,
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


def test_swing_height_profile_endpoints_and_peak():
  psi = torch.tensor([0.0, 0.5, 1.0])
  for profile in ("sin", "bump"):
    h = _swing_height_profile(psi, amplitude=0.1, profile=profile)
    # Vanishes at takeoff/landing, peaks at the target amplitude mid-swing.
    assert math.isclose(h[0].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(h[2].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(h[1].item(), 0.1, rel_tol=1e-6)


def _make_clock_env(heights: torch.Tensor, episode_step: int, command: torch.Tensor):
  """Mock env exposing just what feet_swing_height_clock reads."""
  height_sensor = MagicMock(spec=TerrainHeightSensor)
  type(height_sensor).data = PropertyMock(return_value=MagicMock(heights=heights))
  env = MagicMock()
  env.device = torch.device("cpu")
  env.step_dt = 0.1
  env.episode_length_buf = torch.tensor([episode_step], dtype=torch.long)
  env.scene.__getitem__ = MagicMock(return_value=height_sensor)
  env.command_manager.get_command = MagicMock(return_value=command)
  env.extras = {"log": {}}
  return env


def _clock_reward(env):
  # period=0.8, step_dt=0.1, swing_ratio=0.5: at episode_step=2 the base phase is
  # 0.25, so foot 0 (offset 0) is mid-swing (psi=0.5, desired=target) and foot 1
  # (offset 0.5) is in stance (desired=0).
  return feet_swing_height_clock(
    env,
    height_sensor_name="foot_height_scan",
    target_height=0.1,
    period=0.8,
    swing_ratio=0.5,
    std=0.05,
    foot_offsets=(0.0, 0.5),
    profile="sin",
    command_name="twist",
    command_threshold=0.05,
  )


def test_clock_reward_rewards_on_schedule_lift():
  moving = torch.tensor([[0.5, 0.0, 0.0]])
  # Foot on the clock-prescribed arc (swing foot at target, stance foot down).
  matched = _clock_reward(_make_clock_env(torch.tensor([[0.1, 0.0]]), 2, moving))
  # Same clock, but the swing foot fails to lift (stays low) -> penalized.
  low = _clock_reward(_make_clock_env(torch.tensor([[0.03, 0.0]]), 2, moving))
  assert matched.item() > 1.99  # both feet match their targets
  assert low.item() < matched.item() - 0.5  # low swing foot is genuinely worse


def test_clock_reward_gated_off_when_not_commanded():
  still = torch.tensor([[0.0, 0.0, 0.0]])
  r = _clock_reward(_make_clock_env(torch.tensor([[0.1, 0.0]]), 2, still))
  assert r.item() == 0.0


def _make_clock_obs_env(episode_step: int, command: torch.Tensor):
  """Mock env exposing just what gait_clock reads."""
  env = MagicMock()
  env.step_dt = 0.1
  env.episode_length_buf = torch.tensor([episode_step], dtype=torch.long)
  env.command_manager.get_command = MagicMock(return_value=command)
  return env


def test_gait_clock_advances_with_episode_time():
  # period=0.8, step_dt=0.1: at episode_step=2 the phase is 0.25, so the clock
  # is at angle pi/2 -> [sin, cos] = [1, 0].
  moving = torch.tensor([[0.5, 0.0, 0.0]])
  clock = gait_clock(
    _make_clock_obs_env(2, moving),
    period=0.8,
    command_name="twist",
    command_threshold=0.05,
  )
  assert clock.shape == (1, 2)
  assert math.isclose(clock[0, 0].item(), 1.0, abs_tol=1e-6)
  assert math.isclose(clock[0, 1].item(), 0.0, abs_tol=1e-6)


def test_gait_clock_zeroed_when_not_commanded():
  # At zero command the clock must collapse to the origin so the policy gets a
  # distinct standing signal instead of a sweeping walking phase (which would
  # otherwise drive marching on the spot).
  still = torch.tensor([[0.0, 0.0, 0.0]])
  clock = gait_clock(
    _make_clock_obs_env(2, still),
    period=0.8,
    command_name="twist",
    command_threshold=0.05,
  )
  assert torch.equal(clock, torch.zeros(1, 2))


def test_gait_clock_ungated_without_command_name():
  # Without command_name the clock always ticks regardless of command.
  still = torch.tensor([[0.0, 0.0, 0.0]])
  clock = gait_clock(_make_clock_obs_env(2, still), period=0.8)
  assert math.isclose(clock[0, 0].item(), 1.0, abs_tol=1e-6)
  assert math.isclose(clock[0, 1].item(), 0.0, abs_tol=1e-6)


def _make_clock_silence_env(episode_step: int, step_counter: int) -> MagicMock:
  """Mock env for gait_clock silence tests (always commanded, moving)."""
  env = MagicMock()
  env.step_dt = 0.1
  env.episode_length_buf = torch.tensor([episode_step], dtype=torch.long)
  env.common_step_counter = step_counter
  env.command_manager.get_command = MagicMock(
    return_value=torch.tensor([[0.5, 0.0, 0.0]])
  )
  return env


def test_gait_clock_silence_scales_at_stage_boundaries():
  # A stepped silence schedule fades the clock 1.0 -> 0.0. The scale of the last
  # stage whose step has been reached is applied (step function, no interp). At
  # episode_step=2 (phase 0.25) the unsilenced clock is [1, 0], so the observed
  # magnitude equals the active scale.
  stages = [
    {"step": 0, "scale": 1.0},
    {"step": 100, "scale": 0.5},
    {"step": 200, "scale": 0.1},
    {"step": 300, "scale": 0.0},
  ]
  # Just before the first boundary: full-strength clock.
  clock = gait_clock(
    _make_clock_silence_env(2, 50),
    period=0.8,
    command_name="twist",
    silence_stages=stages,
  )
  assert math.isclose(clock[0, 0].item(), 1.0, abs_tol=1e-6)
  # Between stage 1 and 2: half scale.
  clock = gait_clock(
    _make_clock_silence_env(2, 150),
    period=0.8,
    command_name="twist",
    silence_stages=stages,
  )
  assert math.isclose(clock[0, 0].item(), 0.5, abs_tol=1e-6)
  # Past the final boundary: fully silenced.
  clock = gait_clock(
    _make_clock_silence_env(2, 500),
    period=0.8,
    command_name="twist",
    silence_stages=stages,
  )
  assert torch.equal(clock, torch.zeros(1, 2))


def test_gait_clock_no_silence_without_stages():
  # Without silence_stages the clock is never attenuated by the step counter.
  clock = gait_clock(
    _make_clock_silence_env(2, 10_000),
    period=0.8,
    command_name="twist",
  )
  assert math.isclose(clock[0, 0].item(), 1.0, abs_tol=1e-6)
