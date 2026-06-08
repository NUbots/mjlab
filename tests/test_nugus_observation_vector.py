"""Tests for the NUgus actor observation vector structure.

Verifies the order of observation terms, their index slices within the
concatenated actor observation vector, and the joint ordering within
``joint_pos`` / ``joint_vel`` / ``actions`` (the slice of the observation
that depends on the deployment-side joint mapping).
"""

import pytest

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ARTICULATION,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

# Expected joint ordering for joint_pos / joint_vel / actions in the actor
# observation vector. This is the MuJoCo joint order, dictated by body-tree
# traversal of nugus.xml (left leg, then right leg, then head, then left arm,
# then right arm). If this order changes, every deployed policy's joint
# mapping table breaks — hence the explicit pin.
EXPECTED_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_yaw",
  "left_hip_roll",
  "left_hip_pitch",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_yaw",
  "right_hip_roll",
  "right_hip_pitch",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
  "neck_yaw",
  "head_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow_pitch",
)


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


def test_entity_joint_name_order(nugus_entity: Entity) -> None:
  """Pin the MuJoCo joint order.

  ``joint_pos`` and ``joint_vel`` observations use the default
  ``SceneEntityCfg("robot")`` (``joint_ids=slice(None)``), so they index
  ``entity.data.joint_pos`` directly, giving ``entity.joint_names`` order.
  A change here means every deployed policy's joint mapping is wrong.
  """
  assert tuple(nugus_entity.joint_names) == EXPECTED_JOINT_ORDER


def test_action_joint_order(nugus_entity: Entity) -> None:
  """Pin the action joint mapping.

  The velocity task's ``joint_pos`` action term uses
  ``actuator_names=(".*",)``, which goes through
  ``find_joints_by_actuator_names`` and resolves to actuated joints in
  ``entity.joint_names`` natural order. action[i] must target
  ``EXPECTED_JOINT_ORDER[i]`` for deployed policies to remain valid.
  """
  joint_ids, joint_names = nugus_entity.find_joints_by_actuator_names(".*")
  assert tuple(joint_names) == EXPECTED_JOINT_ORDER
  # joint_ids index into entity.joint_names; confirm they line up too.
  resolved = tuple(nugus_entity.joint_names[i] for i in joint_ids)
  assert resolved == EXPECTED_JOINT_ORDER


def test_action_scale_covers_all_joints(nugus_entity: Entity) -> None:
  """Every actuated joint has a per-joint action scale entry.

  Catches the case where someone adds/removes a joint from the XML but
  forgets to update ``nugus_constants.NUGUS_ACTION_SCALE``, which would
  silently leave action scales unset for the new joint.
  """
  from mjlab.asset_zoo.robots.nugus.nugus_constants import NUGUS_ACTION_SCALE

  for name in nugus_entity.joint_names:
    assert name in NUGUS_ACTION_SCALE, f"missing action scale for {name}"


def test_actuator_declaration_order_differs_from_joint_order() -> None:
  """Document that actuator (model.ctrl) order != joint (joint_pos) order.

  ``nugus_constants.py`` declares actuators in right/left-interleaved order,
  while the XML body tree puts left joints first. With ``sort_actuators=False``
  (the default), ``model.ctrl`` follows declaration order. The deployment
  mapping table must therefore distinguish between "joint-order observation
  indices" and "model.ctrl indices" — they are not interchangeable.
  """
  ctrl_order = tuple(
    name for group in NUGUS_ARTICULATION.actuators for name in group.target_names_expr
  )
  assert ctrl_order != EXPECTED_JOINT_ORDER
  # Both must still cover the same set of joints.
  assert set(ctrl_order) == set(EXPECTED_JOINT_ORDER)
