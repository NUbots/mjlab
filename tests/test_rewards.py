"""Tests for reward manager functionality."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.rewards import electrical_power_cost, joint_torques_l2
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.mdp.rewards import (
  cost_of_transport_proxy,
  feet_flat_orientation,
  feet_lateral_distance_cost,
  gait_phase_regularity_cost,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz

PARTIALLY_ACTUATED_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="actuated_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="0.5"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="passive_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.3"/>
          <body name="link3" pos="0 0 0.3">
            <joint name="actuated_joint2" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
            <geom name="link3_geom" type="box" size="0.05 0.05 0.1" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


class SimpleTestReward:
  """A simple class-based reward for testing that tracks state."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.num_envs = env.num_envs
    self.device = env.device
    self.current_air_time = torch.zeros((self.num_envs, 1), device=self.device)

  def __call__(self, env, **kwargs):
    self.current_air_time += 0.01
    return torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.current_air_time[env_ids] = 0


class StatelessReward:
  """A stateless class-based reward without reset method."""

  def __init__(self, cfg: RewardTermCfg, env):
    pass

  def __call__(self, env, **kwargs):
    return torch.ones(env.num_envs)


@pytest.fixture
def mock_env():
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  env.max_episode_length_s = 10.0
  robot = Mock()
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env


@pytest.fixture
def class_reward_config():
  """Config with a class-based reward."""
  return {
    "term": RewardTermCfg(
      func=SimpleTestReward,
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def function_reward_config():
  """Config with a function-based reward."""
  return {
    "term": RewardTermCfg(
      func=lambda env: torch.ones(env.num_envs),
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def stateless_reward_config():
  """Config with a stateless class-based reward."""
  return {
    "term": RewardTermCfg(
      func=StatelessReward,
      weight=1.0,
      params={},
    )
  }


def test_class_based_reward_reset(mock_env, class_reward_config):
  """Test that class-based reward terms are tracked and have reset called."""
  manager = RewardManager(class_reward_config, mock_env)
  term = manager._class_term_cfgs[0].func

  for _ in range(10):
    manager.compute(dt=0.01)
  assert (term.current_air_time > 0).all()

  manager.reset(env_ids=torch.tensor([0, 2]))

  # Check that only specified envs were reset.
  assert term.current_air_time[0, 0] == 0
  assert term.current_air_time[1, 0] > 0
  assert term.current_air_time[2, 0] == 0
  assert term.current_air_time[3, 0] > 0


def test_function_based_reward_not_tracked(mock_env, function_reward_config):
  """Test that function-based reward terms are not tracked as class terms."""
  manager = RewardManager(function_reward_config, mock_env)
  assert len(manager._class_term_cfgs) == 0


def test_stateless_class_reward_no_reset(mock_env, stateless_reward_config):
  """Test that stateless class-based rewards without reset don't break reset."""
  manager = RewardManager(stateless_reward_config, mock_env)

  # Stateless rewards without reset method should not be tracked.
  assert len(manager._class_term_cfgs) == 0

  # Reset should work without errors.
  manager.reset(env_ids=torch.tensor([0, 2]))


def test_electrical_power_cost_partially_actuated(device):
  """Test electrical_power_cost on robots with passive joints.

  This test verifies that:
  1. The reward correctly identifies and uses only actuated joints (ignoring passive joints)
  2. Power cost is computed as sum of max(0, force*velocity) for each actuator,
     meaning only positive work is penalized (not regenerative braking)
  """
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(PARTIALLY_ACTUATED_ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("actuated_joint1", "actuated_joint2"),
          effort_limit=10.0,
          stiffness=100.0,
          damping=10.0,
        ),
      )
    ),
  )
  entity = Entity(entity_cfg)
  model = entity.compile()

  num_envs = 2
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg(name="robot", joint_names=("actuated.*",))

  reward_cfg = RewardTermCfg(
    func=electrical_power_cost, weight=1.0, params={"asset_cfg": asset_cfg}
  )

  reward = electrical_power_cost(reward_cfg, env)

  assert len(reward._joint_ids) == 2

  # Test case 1: All forces and velocities aligned (all positive work).
  # actuated_joint1 (dof 0), actuated_joint2 (dof 2).
  # Note: dof 1 is the passive joint, not used in power calculation.
  sim.data.qfrc_actuator[:] = 0.0
  sim.data.qfrc_actuator[:, 0] = torch.tensor([2.0, 1.0], device=device)
  sim.data.qfrc_actuator[:, 2] = torch.tensor([3.0, 4.0], device=device)
  sim.data.qvel[:] = 0.0
  sim.data.qvel[:, 0] = torch.tensor([1.0, 2.0], device=device)
  sim.data.qvel[:, 2] = torch.tensor([2.0, 1.0], device=device)

  power_cost = reward(env, asset_cfg)

  # Expected: env0 = 2.0*1.0 + 3.0*2.0 = 8.0, env1 = 1.0*2.0 + 4.0*1.0 = 6.0.
  expected = torch.tensor([8.0, 6.0], device=device)
  assert torch.allclose(power_cost, expected)

  # Test case 2: Some negative forces (regenerative braking, not penalized).
  sim.data.qfrc_actuator[:] = 0.0
  sim.data.qfrc_actuator[:, 0] = torch.tensor([-2.0, 1.0], device=device)
  sim.data.qfrc_actuator[:, 2] = torch.tensor([3.0, -4.0], device=device)

  power_cost = reward(env, asset_cfg)

  # Expected: env0 = max(0,-2.0*1.0) + max(0,3.0*2.0) = 0 + 6.0 = 6.0.
  #           env1 = max(0,1.0*2.0) + max(0,-4.0*1.0) = 2.0 + 0 = 2.0.
  expected = torch.tensor([6.0, 2.0], device=device)
  assert torch.allclose(power_cost, expected)


def test_cost_of_transport_proxy_matches_expected_formula():
  """CoT proxy should be positive power divided by horizontal speed."""
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.extras = {"log": {}}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  )

  asset = Mock()
  asset.find_joints = Mock(return_value=([0, 2], ["joint0", "joint2"]))
  asset.find_actuators = Mock(return_value=([0, 1], ["act0", "act1"]))
  asset.data = SimpleNamespace(
    actuator_force=torch.tensor([[2.0, 3.0], [1.0, 4.0]], dtype=torch.float32),
    joint_vel=torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.float32),
    root_link_lin_vel_w=torch.tensor(
      [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    ),
  )
  env.scene = {"robot": asset}

  asset_cfg = SceneEntityCfg(name="robot", joint_names=(".*",))
  reward_cfg = RewardTermCfg(
    func=cost_of_transport_proxy,
    weight=1.0,
    params={"asset_cfg": asset_cfg},
  )
  reward = cost_of_transport_proxy(reward_cfg, env)

  cot_cost = reward(
    env,
    asset_cfg=asset_cfg,
    speed_floor=0.1,
    command_name="twist",
    command_threshold=0.05,
  )

  # Positive mechanical powers: [8.0, 6.0], speeds: [0.5, 1.0].
  expected = torch.tensor([16.0, 6.0], dtype=torch.float32)
  assert torch.allclose(cot_cost, expected)
  assert "Metrics/cot_proxy_mean" in env.extras["log"]
  assert "Metrics/locomotion_speed_mean" in env.extras["log"]


def test_gait_phase_regularity_cost_penalizes_asymmetry():
  """Gait phase regularity should be zero for symmetric timing and positive for asymmetric timing."""
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.extras = {"log": {}}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  )

  sensor = Mock()
  sensor.data = SimpleNamespace(
    last_air_time=torch.tensor([[0.2, 0.2], [0.1, 0.3]], dtype=torch.float32),
    last_contact_time=torch.tensor([[0.4, 0.4], [0.2, 0.5]], dtype=torch.float32),
  )
  env.scene = {"feet_ground_contact": sensor}

  cost = gait_phase_regularity_cost(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.05,
  )

  assert cost[0] == pytest.approx(0.0)
  assert cost[1] > 0.0
  assert "Metrics/gait_air_cv_mean" in env.extras["log"]
  assert "Metrics/gait_contact_cv_mean" in env.extras["log"]


def test_gait_phase_regularity_cost_respects_command_threshold():
  """Gait phase regularity should be disabled for environments with no motion command."""
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.extras = {"log": {}}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  )

  sensor = Mock()
  sensor.data = SimpleNamespace(
    last_air_time=torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32),
    last_contact_time=torch.tensor([[0.2, 0.5], [0.2, 0.5]], dtype=torch.float32),
  )
  env.scene = {"feet_ground_contact": sensor}

  cost = gait_phase_regularity_cost(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.05,
  )

  assert cost[0] > 0.0
  assert cost[1] == pytest.approx(0.0)


def _make_lateral_foot_env(lateral_distances: torch.Tensor) -> Mock:
  """Build a mock env with two feet at the given body-Y separations."""
  num_envs = lateral_distances.shape[0]
  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.extras = {"log": {}}

  half_sep = lateral_distances / 2.0
  foot_pos_w = torch.zeros(num_envs, 2, 3)
  foot_pos_w[:, 0, 1] = half_sep
  foot_pos_w[:, 1, 1] = -half_sep

  asset = Mock()
  asset.data = SimpleNamespace(
    site_pos_w=foot_pos_w,
    root_link_quat_w=quat_from_euler_xyz(
      torch.zeros(num_envs),
      torch.zeros(num_envs),
      torch.zeros(num_envs),
    ),
  )
  env.scene = {"robot": asset}
  return env


def test_feet_lateral_distance_cost_at_nominal_is_zero():
  nominal = 0.25
  env = _make_lateral_foot_env(torch.tensor([nominal]))
  asset_cfg = SceneEntityCfg(name="robot", site_ids=[0, 1])

  cost = feet_lateral_distance_cost(
    env,
    nominal_distance=nominal,
    sharpness=8.0,
    asset_cfg=asset_cfg,
  )

  assert cost[0] == pytest.approx(0.0, abs=1e-6)


def test_feet_lateral_distance_cost_penalizes_too_close_and_too_wide():
  nominal = 0.25
  sharpness = 8.0
  delta = 0.05
  env = _make_lateral_foot_env(torch.tensor([nominal - delta, nominal + delta]))
  asset_cfg = SceneEntityCfg(name="robot", site_ids=[0, 1])

  cost = feet_lateral_distance_cost(
    env,
    nominal_distance=nominal,
    sharpness=sharpness,
    asset_cfg=asset_cfg,
  )

  expected = math.exp(sharpness * delta) - 1.0
  assert cost[0] == pytest.approx(expected, rel=1e-5)
  assert cost[1] == pytest.approx(expected, rel=1e-5)


def _make_flat_foot_env(quats, found, command):
  env = Mock()
  num_envs = quats.shape[0]
  env.num_envs = num_envs
  env.device = "cpu"
  env.extras = {"log": {}}
  env.command_manager.get_command = Mock(return_value=command)

  asset = Mock()
  asset.data = SimpleNamespace(
    body_link_quat_w=quats,
    gravity_vec_w=torch.tensor([0.0, 0.0, -1.0]).expand(num_envs, 3),
  )
  sensor = Mock()
  sensor.data = SimpleNamespace(found=found)
  env.scene = {"robot": asset, "feet_ground_contact": sensor}
  return env


def test_feet_flat_orientation_penalizes_tilt_in_swing():
  """A level swing foot costs nothing; a pitched swing foot is penalized."""
  zero = torch.zeros(1)
  level = quat_from_euler_xyz(zero, zero, zero)[0]  # [4]
  pitched = quat_from_euler_xyz(zero, torch.tensor([0.5]), zero)[0]  # 0.5 rad pitch

  # Env 0: both feet level. Env 1: left foot pitched.
  quats = torch.stack(
    [torch.stack([level, level]), torch.stack([pitched, level])]
  )  # [2, 2, 4]
  found = torch.zeros(2, 2)  # all feet in the air (swing)
  command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

  env = _make_flat_foot_env(quats, found, command)
  asset_cfg = SceneEntityCfg(name="robot", body_ids=[0, 1])

  cost = feet_flat_orientation(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.05,
    sole_normal_axis=2,
    asset_cfg=asset_cfg,
  )

  assert cost[0] == pytest.approx(0.0, abs=1e-6)
  assert cost[1] == pytest.approx(math.sin(0.5) ** 2, abs=1e-5)
  assert "Metrics/foot_tilt_mean" in env.extras["log"]


def test_feet_flat_orientation_ignores_stance_and_zero_command():
  """Stance feet and standing-still environments are not penalized."""
  zero = torch.zeros(1)
  pitched = quat_from_euler_xyz(zero, torch.tensor([0.5]), zero)[0]

  quats = torch.stack(
    [torch.stack([pitched, pitched]), torch.stack([pitched, pitched])]
  )
  # Env 0: pitched feet but in contact (stance). Env 1: in air but no command.
  found = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
  command = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  env = _make_flat_foot_env(quats, found, command)
  asset_cfg = SceneEntityCfg(name="robot", body_ids=[0, 1])

  cost = feet_flat_orientation(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    command_threshold=0.05,
    sole_normal_axis=2,
    asset_cfg=asset_cfg,
  )

  assert cost[0] == pytest.approx(0.0, abs=1e-6)
  assert cost[1] == pytest.approx(0.0, abs=1e-6)


def test_reward_manager_handles_nan_values(mock_env):
  """Test that RewardManager converts NaN/Inf reward values to zero."""

  def nan_reward(env):
    """Reward function that returns NaN for some environments."""
    r = torch.ones(env.num_envs, device=env.device)
    r[1] = float("nan")
    r[3] = float("inf")
    return r

  cfg = {"nan_term": RewardTermCfg(func=nan_reward, weight=1.0, params={})}
  manager = RewardManager(cfg, mock_env)

  rewards = manager.compute(dt=0.01)

  # NaN and Inf should be converted to 0.
  assert not torch.isnan(rewards).any(), "Reward buffer contains NaN"
  assert not torch.isinf(rewards).any(), "Reward buffer contains Inf"
  assert rewards[0] == pytest.approx(0.01)
  assert rewards[1] == 0.0
  assert rewards[2] == pytest.approx(0.01)
  assert rewards[3] == 0.0


def test_reward_manager_handles_neginf_values(mock_env):
  """Test that RewardManager converts negative infinity to zero."""

  def neginf_reward(env):
    r = torch.ones(env.num_envs, device=env.device)
    r[2] = float("-inf")
    return r

  cfg = {"neginf_term": RewardTermCfg(func=neginf_reward, weight=1.0, params={})}
  manager = RewardManager(cfg, mock_env)

  rewards = manager.compute(dt=0.01)

  assert not torch.isinf(rewards).any()
  assert rewards[2] == 0.0


def test_reward_scaling_enabled(mock_env):
  """Test that rewards are scaled by dt when scale_by_dt=True (default)."""

  def constant_reward(env):
    return torch.ones(env.num_envs, device=env.device)

  cfg = {"term": RewardTermCfg(func=constant_reward, weight=2.0, params={})}
  manager = RewardManager(cfg, mock_env, scale_by_dt=True)

  dt = 0.02
  rewards = manager.compute(dt=dt)

  # With scaling: reward = raw_value (1.0) * weight (2.0) * dt (0.02) = 0.04
  expected = 1.0 * 2.0 * dt
  assert torch.allclose(rewards, torch.full((4,), expected))

  # _step_reward should be unscaled (raw_value * weight)
  step_reward = manager._step_reward[:, 0]
  expected_step = 1.0 * 2.0
  assert torch.allclose(step_reward, torch.full((4,), expected_step))


def test_reward_scaling_disabled(mock_env):
  """Test that rewards are not scaled by dt when scale_by_dt=False."""

  def constant_reward(env):
    return torch.ones(env.num_envs, device=env.device)

  cfg = {"term": RewardTermCfg(func=constant_reward, weight=2.0, params={})}
  manager = RewardManager(cfg, mock_env, scale_by_dt=False)

  dt = 0.02
  rewards = manager.compute(dt=dt)

  # Without scaling: reward = raw_value (1.0) * weight (2.0) = 2.0
  expected = 1.0 * 2.0
  assert torch.allclose(rewards, torch.full((4,), expected))

  # _step_reward should still be unscaled (same as reward when not scaling)
  step_reward = manager._step_reward[:, 0]
  assert torch.allclose(step_reward, torch.full((4,), expected))


def test_reward_scaling_default_is_enabled(mock_env):
  """Test that scale_by_dt defaults to True for backward compatibility."""

  def constant_reward(env):
    return torch.ones(env.num_envs, device=env.device)

  cfg = {"term": RewardTermCfg(func=constant_reward, weight=1.0, params={})}
  # Don't pass scale_by_dt - should default to True
  manager = RewardManager(cfg, mock_env)

  dt = 0.01
  rewards = manager.compute(dt=dt)

  # Default (scaling enabled): reward = 1.0 * 1.0 * 0.01 = 0.01
  assert torch.allclose(rewards, torch.full((4,), 0.01))


def test_joint_torques_l2_with_actuator_ids(mock_env):
  """Test that joint_torques_l2 only penalizes specified actuators."""
  mock_env.scene["robot"].data.actuator_force = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4)

  asset_cfg = SceneEntityCfg(name="robot", actuator_ids=[0, 2])
  result = joint_torques_l2(mock_env, asset_cfg)

  # Only actuators 0 and 2: 1^2 + 3^2 = 10.0
  expected = torch.full((4,), 10.0)
  assert torch.allclose(result, expected)


def test_joint_torques_l2_all_actuators(mock_env):
  """Test that joint_torques_l2 uses all actuators by default."""
  mock_env.scene["robot"].data.actuator_force = torch.tensor([[1.0, 2.0, 3.0]] * 4)

  asset_cfg = SceneEntityCfg(name="robot")
  result = joint_torques_l2(mock_env, asset_cfg)

  # All actuators: 1^2 + 2^2 + 3^2 = 14.0
  expected = torch.full((4,), 14.0)
  assert torch.allclose(result, expected)
