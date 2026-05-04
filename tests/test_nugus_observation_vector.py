"""Tests for the NUgus actor observation vector structure.

Verifies the order of observation terms and their index slices within the
concatenated actor observation vector produced by the NUgus velocity task.
"""

import pytest

from mjlab.asset_zoo.robots.nugus.nugus_constants import get_nugus_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg


@pytest.fixture(scope="module")
def nugus_entity() -> Entity:
  return Entity(get_nugus_robot_cfg())


@pytest.fixture(scope="module")
def nugus_actor_terms() -> dict:
  cfg = nubots_nugus_flat_env_cfg()
  return {
    name: term
    for name, term in cfg.observations["actor"].terms.items()
    if term is not None
  }


def _compute_slices(
  term_names: list[str], term_dims: dict[str, int]
) -> dict[str, slice]:
  """Compute the start/end index slice for each term in the concatenated vector."""
  slices: dict[str, slice] = {}
  offset = 0
  for name in term_names:
    dim = term_dims[name]
    slices[name] = slice(offset, offset + dim)
    offset += dim
  return slices


def test_nugus_has_20_joints(nugus_entity: Entity) -> None:
  assert nugus_entity.num_joints == 20
  assert nugus_entity.num_actuators == 20


def test_actor_terms_present(nugus_actor_terms: dict) -> None:
  expected = {
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  }
  assert set(nugus_actor_terms.keys()) == expected


def test_base_lin_vel_absent(nugus_actor_terms: dict) -> None:
  """base_lin_vel is not observable by the NUgus policy (no direct velocity sensor)."""
  assert "base_lin_vel" not in nugus_actor_terms


def test_height_scan_absent(nugus_actor_terms: dict) -> None:
  """height_scan is removed for the flat terrain NUgus config."""
  assert "height_scan" not in nugus_actor_terms


def test_actor_term_order(nugus_actor_terms: dict) -> None:
  expected_order = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  ]
  assert list(nugus_actor_terms.keys()) == expected_order


def test_observation_vector_slices(
  nugus_entity: Entity, nugus_actor_terms: dict
) -> None:
  """Verify each term occupies the expected index slice in the actor observation vector.

  With 20 joints the layout is:
    base_ang_vel       [0:3]
    projected_gravity  [3:6]
    joint_pos          [6:26]
    joint_vel          [26:46]
    actions            [46:66]
    command            [66:69]
  """
  n = nugus_entity.num_joints  # 20

  # Known dimensions: sensor/gravity outputs are 3D, joints give n dims, command is 3D.
  term_dims = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": n,
    "joint_vel": n,
    "actions": n,
    "command": 3,
  }

  slices = _compute_slices(list(nugus_actor_terms.keys()), term_dims)

  assert slices["base_ang_vel"] == slice(0, 3)
  assert slices["projected_gravity"] == slice(3, 6)
  assert slices["joint_pos"] == slice(6, 6 + n)
  assert slices["joint_vel"] == slice(6 + n, 6 + 2 * n)
  assert slices["actions"] == slice(6 + 2 * n, 6 + 3 * n)
  assert slices["command"] == slice(6 + 3 * n, 6 + 3 * n + 3)


def test_total_observation_dim(nugus_entity: Entity, nugus_actor_terms: dict) -> None:
  n = nugus_entity.num_joints  # 20
  expected_dim = 3 + 3 + n + n + n + 3  # 69
  assert expected_dim == 69
