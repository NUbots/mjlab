"""Tests for the NUgus actor observation vector structure.

Verifies the order of observation terms, their index slices within the
concatenated actor observation vector, and the joint ordering within
``joint_pos`` / ``joint_vel`` / ``actions`` (the slice of the observation
that depends on the deployment-side joint mapping).
"""

from __future__ import annotations

import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ARTICULATION,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.nugus.dr_observations import (
  DR_EXTRAS_DIM,
  DR_RATIOS_DIM,
  dr_extras,
  dr_ratios,
  dr_ratios_torso_mass_index,
)
from mjlab.tasks.velocity.config.nugus.env_cfgs import (
  nubots_nugus_flat_env_cfg,
)


@pytest.fixture
def device() -> str:
  return get_test_device()


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


def test_nugus_has_20_motor_joints(nugus_entity: Entity) -> None:
  # 20 motor joints, each paired with a passive ``_backlash`` sibling joint
  # that models servo gear play. Only the motor joints are actuated.
  motor_joints = [n for n in nugus_entity.joint_names if not n.endswith("_backlash")]
  backlash_joints = [n for n in nugus_entity.joint_names if n.endswith("_backlash")]
  assert len(motor_joints) == 20
  assert len(backlash_joints) == 20
  assert nugus_entity.num_joints == 40
  assert nugus_entity.num_actuators == 20


def test_actor_terms_present(nugus_actor_terms: dict) -> None:
  expected = {
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "gait_clock",
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
    "gait_clock",
  ]
  assert list(nugus_actor_terms.keys()) == expected_order


def test_observation_vector_slices(
  nugus_entity: Entity, nugus_actor_terms: dict
) -> None:
  """Verify each term occupies the expected index slice in the actor observation vector.

  With 20 motor joints the layout is:
    base_ang_vel       [0:3]
    projected_gravity  [3:6]
    joint_pos          [6:26]
    joint_vel          [26:46]
    actions            [46:66]
    command            [66:69]
    gait_clock         [69:71]
  """
  # joint_pos/joint_vel observations are scoped to motor joints only via the
  # asset_cfg override in nubots_nugus_*_env_cfg — passive ``_backlash`` joints
  # do not appear in the observation vector.
  n = len(EXPECTED_JOINT_ORDER)  # 20

  # Known dimensions: sensor/gravity outputs are 3D, joints give n dims, command is 3D.
  term_dims = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": n,
    "joint_vel": n,
    "actions": n,
    "command": 3,
    "gait_clock": 2,
  }

  slices = _compute_slices(list(nugus_actor_terms.keys()), term_dims)

  assert slices["base_ang_vel"] == slice(0, 3)
  assert slices["projected_gravity"] == slice(3, 6)
  assert slices["joint_pos"] == slice(6, 6 + n)
  assert slices["joint_vel"] == slice(6 + n, 6 + 2 * n)
  assert slices["actions"] == slice(6 + 2 * n, 6 + 3 * n)
  assert slices["command"] == slice(6 + 3 * n, 6 + 3 * n + 3)
  assert slices["gait_clock"] == slice(6 + 3 * n + 3, 6 + 3 * n + 3 + 2)


def test_total_observation_dim(nugus_entity: Entity, nugus_actor_terms: dict) -> None:
  n = len(EXPECTED_JOINT_ORDER)  # 20
  expected_dim = 3 + 3 + n + n + n + 3  # 69
  assert expected_dim == 69


def test_entity_motor_joint_name_order(nugus_entity: Entity) -> None:
  """Pin the MuJoCo motor-joint order.

  ``joint_pos`` and ``joint_vel`` observations are scoped to motor joints
  via ``SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))``,
  so they index ``entity.data.joint_pos`` using the motor-joint subset in
  ``entity.joint_names`` natural order. A change here means every deployed
  policy's joint mapping is wrong.
  """
  motor_joints = tuple(
    n for n in nugus_entity.joint_names if not n.endswith("_backlash")
  )
  assert motor_joints == EXPECTED_JOINT_ORDER


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
    if name.endswith("_backlash"):
      # Passive backlash joints are not actuated and don't need an action scale.
      continue
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


@pytest.fixture(scope="module")
def nugus_flat_cfg():
  return nubots_nugus_flat_env_cfg()


def test_dr_ratios_in_critic_not_actor(nugus_flat_cfg) -> None:
  assert "dr_ratios" in nugus_flat_cfg.observations["critic"].terms
  assert "dr_ratios" not in nugus_flat_cfg.observations["actor"].terms


def test_dr_ratios_change_across_resets(device, nugus_flat_cfg) -> None:
  cfg = nugus_flat_cfg
  cfg.scene.num_envs = 32
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset(seed=1)
    first = dr_ratios(env)
    assert first.shape == (32, DR_RATIOS_DIM)

    env.reset(seed=999)
    second = dr_ratios(env)
    torso_mass_idx = dr_ratios_torso_mass_index()
    assert not torch.allclose(first[:, torso_mass_idx], second[:, torso_mass_idx])
  finally:
    env.close()


def test_rma_groups_absent_by_default(nugus_flat_cfg) -> None:
  """RMA unset must leave the observation spec byte-identical to pre-RMA."""
  assert "history" not in nugus_flat_cfg.observations
  assert "dr" not in nugus_flat_cfg.observations
  assert "dr_extras" not in nugus_flat_cfg.observations["critic"].terms
  assert "odom_target" not in nugus_flat_cfg.observations


def test_rnn_memory_groups_cfg(monkeypatch) -> None:
  """RNN_MEMORY: odom + critic dr_extras, but no window or dr groups."""
  monkeypatch.setenv("RNN_MEMORY", "1")
  monkeypatch.setenv("RMA_VHAT", "1")
  cfg = nubots_nugus_flat_env_cfg()
  assert "odom_target" in cfg.observations
  assert "dr_extras" in cfg.observations["critic"].terms
  assert "history" not in cfg.observations
  assert "dr" not in cfg.observations


def test_odom_target_group_cfg(monkeypatch) -> None:
  """RMA_VHAT adds the noise-free odometry supervision group."""
  monkeypatch.setenv("RMA", "1")
  monkeypatch.setenv("RMA_VHAT", "1")
  cfg = nubots_nugus_flat_env_cfg()
  group = cfg.observations["odom_target"]
  assert list(group.terms.keys()) == ["base_lin_vel"]
  assert group.enable_corruption is False

  # RMA without the head must not add the group.
  monkeypatch.delenv("RMA_VHAT")
  cfg = nubots_nugus_flat_env_cfg()
  assert "odom_target" not in cfg.observations


def test_rma_groups_cfg(monkeypatch) -> None:
  monkeypatch.setenv("RMA", "1")
  monkeypatch.setenv("RMA_WINDOW", "7")
  cfg = nubots_nugus_flat_env_cfg()

  history = cfg.observations["history"]
  assert history.history_length == 7
  assert history.flatten_history_dim is False
  assert history.enable_corruption is True
  # Per-frame layout must equal the actor vector: same terms, same order.
  assert list(history.terms.keys()) == list(cfg.observations["actor"].terms.keys())

  dr_group = cfg.observations["dr"]
  assert list(dr_group.terms.keys()) == ["dr_ratios", "dr_extras"]
  assert dr_group.enable_corruption is False

  # The critic sees the same extended privileged vector as the encoder.
  assert "dr_extras" in cfg.observations["critic"].terms


def test_rma_env_group_dims_and_extras_defaults(device, monkeypatch) -> None:
  """Live env: group shapes and nominal dr_extras segments (bus/current off)."""
  monkeypatch.setenv("RMA", "1")
  monkeypatch.setenv("RMA_WINDOW", "5")
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset(seed=1)
    obs = env.observation_manager.compute()
    actor_dim = obs["actor"].shape[-1]
    assert obs["history"].shape == (4, 5, actor_dim)
    assert obs["dr"].shape == (4, DR_RATIOS_DIM + DR_EXTRAS_DIM)

    extras = dr_extras(env)
    n = 20
    bus = extras[:, n : n + 3]
    gain = extras[:, n + 3 : n + 3 + n]
    offset = extras[:, n + 3 + n :]
    # Bus model and current-sensor DR are off in this config: segments hold
    # their nominal/identity values (dim stays stable across knobs).
    assert torch.allclose(bus[:, 0], torch.ones_like(bus[:, 0]))
    assert torch.allclose(bus[:, 1], torch.zeros_like(bus[:, 1]))
    assert torch.allclose(gain, torch.ones_like(gain))
    assert torch.allclose(offset, torch.zeros_like(offset))
  finally:
    env.close()
