"""Tests for NUgus velocity-saturation telemetry."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ACTUATOR_LEGS,
  NUGUS_MOTOR_JOINT_REGEX,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.nugus.vel_sat_telemetry import log_vel_sat_frac


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _mock_robot_with_joint_vel(
  device: str, *, knee_qd: float, num_envs: int = 2
) -> Mock:
  entity = Entity(get_nugus_robot_cfg())
  joint_ids, joint_names = entity.find_joints(NUGUS_MOTOR_JOINT_REGEX)
  num_motor_joints = len(joint_names)
  qvel = torch.zeros(num_envs, num_motor_joints, device=device)
  knee_col = joint_names.index("right_knee_pitch")
  qvel[:, knee_col] = knee_qd

  data = Mock()
  data.joint_vel = qvel

  robot = Mock()
  robot.find_joints = Mock(
    return_value=(torch.arange(num_motor_joints, device=device), joint_names)
  )
  robot.data = data
  return robot


def test_vel_sat_frac_legs_saturated(device: str) -> None:
  vel_limit = NUGUS_ACTUATOR_LEGS.velocity_limit
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.extras = {"log": {}}
  env.scene = {
    "robot": _mock_robot_with_joint_vel(device, knee_qd=0.95 * vel_limit),
  }

  log_vel_sat_frac(env)

  assert env.extras["log"]["Metrics/vel_sat_frac_legs"] == pytest.approx(0.1, abs=0.01)
  assert env.extras["log"]["Metrics/vel_sat_frac_hips"] == pytest.approx(0.0, abs=1e-6)


def test_vel_sat_frac_below_threshold_not_counted(device: str) -> None:
  vel_limit = NUGUS_ACTUATOR_LEGS.velocity_limit
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.extras = {"log": {}}
  env.scene = {
    "robot": _mock_robot_with_joint_vel(device, knee_qd=0.5 * vel_limit),
  }

  log_vel_sat_frac(env)

  assert env.extras["log"]["Metrics/vel_sat_frac_legs"] == pytest.approx(0.0, abs=1e-6)
