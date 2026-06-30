"""Tests for stand-still rewards and command gating."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import (
  _is_commanded_to_stand,
  stand_still_pose_deviation,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


def test_is_commanded_to_stand_uses_per_axis_not_sum():
  # Pure strafing: lateral command with zero yaw must not count as standing.
  strafe = torch.tensor([[0.0, 0.12, 0.0], [0.0, 0.0, 0.0]])
  standing = _is_commanded_to_stand(strafe, threshold=0.05)
  assert standing[0].item() is False
  assert standing[1].item() is True

  # Sum-of-norms would also miss this, but mixed small axes should stay walking.
  mixed = torch.tensor([[0.04, 0.04, 0.0]])
  assert _is_commanded_to_stand(mixed, threshold=0.05)[0].item() is False


def _make_stand_reward_env(
  *,
  num_envs: int,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  default_joint_pos: torch.Tensor,
  command: torch.Tensor,
  is_stop_ramping: torch.Tensor | None = None,
):
  asset = MagicMock()
  type(asset).data = PropertyMock(
    return_value=SimpleNamespace(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      default_joint_pos=default_joint_pos,
    )
  )
  asset.find_joints.return_value = (list(range(joint_pos.shape[1])), [])

  twist = MagicMock(spec=UniformVelocityCommand)
  twist.is_stop_ramping = (
    is_stop_ramping
    if is_stop_ramping is not None
    else torch.zeros(num_envs, dtype=torch.bool)
  )

  env = MagicMock()
  env.device = torch.device("cpu")
  env.num_envs = num_envs
  env.scene.__getitem__ = MagicMock(return_value=asset)
  env.command_manager.get_command = MagicMock(return_value=command)
  env.command_manager.get_term = MagicMock(return_value=twist)
  return env


def test_stand_still_pose_inactive_during_strafe():
  default = torch.zeros(1, 2)
  deviated = default.clone()
  deviated[0, 0] = 0.5
  env = _make_stand_reward_env(
    num_envs=1,
    joint_pos=deviated,
    joint_vel=torch.zeros(1, 2),
    default_joint_pos=default,
    command=torch.tensor([[0.0, 0.2, 0.0]]),
  )
  cfg = RewardTermCfg(
    func=stand_still_pose_deviation,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  term = stand_still_pose_deviation(cfg, env)
  cost = term(
    env,
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    command_name="twist",
    command_threshold=0.05,
    grace_steps=0,
  )
  assert cost[0].item() == pytest.approx(0.0)


def test_stand_still_pose_respects_grace_and_ramp():
  default = torch.zeros(1, 1)
  deviated = torch.tensor([[0.4]])
  env = _make_stand_reward_env(
    num_envs=1,
    joint_pos=deviated,
    joint_vel=torch.zeros(1, 1),
    default_joint_pos=default,
    command=torch.tensor([[0.0, 0.0, 0.0]]),
    is_stop_ramping=torch.tensor([True]),
  )
  cfg = RewardTermCfg(
    func=stand_still_pose_deviation,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  term = stand_still_pose_deviation(cfg, env)
  cost_ramp = term(
    env,
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    command_name="twist",
    grace_steps=0,
  )
  assert cost_ramp[0].item() == pytest.approx(0.0)

  env.command_manager.get_term.return_value.is_stop_ramping = torch.tensor([False])
  for _ in range(2):
    cost_before_grace = term(
      env,
      asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
      command_name="twist",
      grace_steps=3,
    )
    assert cost_before_grace[0].item() == pytest.approx(0.0)
  cost_after_grace = term(
    env,
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    command_name="twist",
    grace_steps=3,
  )
  assert cost_after_grace[0].item() == pytest.approx(0.4, abs=1e-5)


def test_velocity_command_stop_tail_ramps_then_zeros():
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 4.0),
    rel_stop_envs=1.0,
    stop_ramp_time=1.0,
    stop_settle_time=0.5,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
  )
  env = MagicMock()
  env.device = torch.device("cpu")
  env.num_envs = 1
  env.step_dt = 0.1
  robot = MagicMock()
  robot.data = SimpleNamespace(
    root_link_pos_w=torch.zeros(1, 3),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    root_link_lin_vel_b=torch.zeros(1, 3),
    root_link_ang_vel_b=torch.zeros(1, 3),
    heading_w=torch.zeros(1),
  )
  env.scene.__getitem__ = MagicMock(return_value=robot)

  term = UniformVelocityCommand(cfg, env)
  term.vel_command_b[0] = torch.tensor([1.0, 0.0, 0.0])
  term.vel_command_w[0] = term.vel_command_b[0].clone()
  term.time_left[0] = 4.0
  term.has_stop_tail[0] = True
  term.is_standing_env[0] = False

  term._update_command()
  assert term.vel_command_b[0, 0].item() == pytest.approx(1.0, abs=1e-5)
  assert term.is_stop_ramping[0].item() is False

  term.time_left[0] = 1.2
  term._update_command()
  assert term.is_stop_ramping[0].item() is True
  assert 0.0 < term.vel_command_b[0, 0].item() < 1.0

  term.time_left[0] = 0.4
  term._update_command()
  assert term.is_stop_ramping[0].item() is False
  assert term.vel_command_b[0, 0].item() == pytest.approx(0.0, abs=1e-5)
