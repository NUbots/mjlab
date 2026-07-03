"""Tests for NUgus actuator velocity limits and stall-torque modeling."""

from __future__ import annotations

import re

import pytest

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ACTUATOR_ARMS,
  NUGUS_ACTUATOR_HEAD,
  NUGUS_ACTUATOR_HIPS,
  NUGUS_ACTUATOR_LEGS,
  VELOCITY_LIMIT_XH540,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def nugus_entity() -> Entity:
  return Entity(get_nugus_robot_cfg())


def test_xh540_velocity_limit_in_expected_range() -> None:
  # 46 rpm @ 14.8 V (e-Manual) → 4.817 rad/s; doc 06 allows up to ~4.8.
  assert 4.1 <= VELOCITY_LIMIT_XH540 <= 4.82


@pytest.mark.parametrize(
  "actuator_cfg",
  [
    NUGUS_ACTUATOR_ARMS,
    NUGUS_ACTUATOR_HIPS,
    NUGUS_ACTUATOR_LEGS,
    NUGUS_ACTUATOR_HEAD,
  ],
)
def test_saturation_effort_equals_effort_limit(
  actuator_cfg: DcMotorActuatorCfg,
) -> None:
  assert actuator_cfg.saturation_effort == actuator_cfg.effort_limit


def test_xh540_knee_actuators_use_correct_velocity_limit(
  nugus_entity: Entity,
) -> None:
  model = nugus_entity.spec.compile()
  knee_pattern = re.compile(r".*knee_pitch$")
  for actuator_idx in range(model.nu):
    actuator = model.actuator(actuator_idx)
    if knee_pattern.match(actuator.name):
      cfg = NUGUS_ACTUATOR_LEGS
      assert cfg.velocity_limit == pytest.approx(VELOCITY_LIMIT_XH540, rel=1e-3)
