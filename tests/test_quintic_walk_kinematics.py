"""Tests for the ported NUbots leg IK and its Euler helpers.

The correctness test here is :func:`test_ik_matches_nubots_cpp`, which diffs the
port against golden data dumped from the real C++ ``calculate_leg_joints``. See
``tests/fixtures/quintic_walk_cpp/README.md`` for how that data is regenerated.

The MuJoCo round-trip tests serve a different purpose: they check that the frame
conventions wiring the engine to the mjlab MJCF are right, and they pin down how
much foot-placement error the engine's idealised leg model carries against the
real robot geometry.
"""

import csv
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_XML,
  STAND_BENT_KNEES_KEYFRAME,
)
from mjlab.controllers.quintic_walk.kinematics import (
  JOINT_NAMES,
  NUGUS_LEG,
  NUGUS_SOLE_OFFSET,
  NUGUS_SOLE_ROTATION,
  NUGUS_TORSO_FRAME_OFFSET,
  calculate_leg_joints,
  invert_transform,
  make_transform,
  mat_to_rpy_intrinsic,
  rpy_intrinsic_to_mat,
)

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "quintic_walk_ik_golden.csv"

# The engine models the leg as two straight 0.2 m links with coincident hip
# axes. The MJCF's thigh is 0.2138 m and tilts 15.4 deg backwards (its knee sits
# 57 mm behind the hip), so a commanded foot pose lands up to ~10 cm away. This
# bound exists to catch frame and sign errors, which are an order of magnitude
# larger; the actual residual is reported by
# ``test_report_idealised_model_placement_error``.
MAX_MODEL_PLACEMENT_ERROR = 0.12


@pytest.fixture
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def golden():
  """Golden IK solutions produced by the real NUbots C++."""
  with GOLDEN_CSV.open() as handle:
    rows = list(csv.DictReader(handle))
  assert rows, "golden fixture is empty"
  return rows


def _golden_tensors(rows, limb: str, device: str):
  subset = [row for row in rows if row["limb"] == limb]
  assert subset, f"no {limb} rows in golden fixture"
  as_tensor = lambda keys: torch.tensor(  # noqa: E731
    [[float(row[key]) for key in keys] for row in subset],
    dtype=torch.float64,
    device=device,
  )
  translation = as_tensor(("x", "y", "z"))
  rpy = as_tensor(("roll", "pitch", "yaw"))
  joints = as_tensor(JOINT_NAMES)
  return make_transform(translation, rpy_intrinsic_to_mat(rpy)), joints


@pytest.mark.parametrize("limb", ["left", "right"])
def test_ik_matches_nubots_cpp(golden, device, limb):
  """The port reproduces the C++ solver to float64 precision.

  Both sides use the same double-precision leg model, so any disagreement above
  numerical noise is a transcription error, not a modelling choice.
  """
  htf, expected = _golden_tensors(golden, limb, device)

  joints = calculate_leg_joints(htf, left=limb == "left")

  assert torch.allclose(joints, expected, atol=1e-12, rtol=0.0)


def test_golden_fixture_exercises_the_overextension_clamp(golden, device):
  """The sweep reaches past maximum leg length, not just the walk envelope.

  The clamp is the one place the solver saturates rather than solving, so a
  fixture that never triggers it would leave that path untested.
  """
  count = 0
  for limb in ("left", "right"):
    htf, _ = _golden_tensors(golden, limb, device)
    rotation, translation = htf[:, :3, :3], htf[:, :3, 3]
    translation = translation + rotation[:, :, 2] * NUGUS_LEG.foot_height
    frame = torch.tensor(
      [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
      dtype=torch.float64,
      device=device,
    )
    translation = torch.einsum("ij,nj->ni", frame, translation)
    if limb == "right":
      translation = translation * torch.tensor(
        [-1.0, 1.0, 1.0], dtype=torch.float64, device=device
      )
    hip = torch.tensor(
      [
        0.5 * NUGUS_LEG.length_between_legs,
        NUGUS_LEG.hip_offset_x,
        NUGUS_LEG.hip_offset_z,
      ],
      dtype=torch.float64,
      device=device,
    )
    count += int(
      ((translation - hip).norm(dim=-1) > NUGUS_LEG.max_leg_length).sum().item()
    )
  assert count > 0, "golden sweep never over-extends the leg"


def test_ik_is_batched_over_envs(device):
  """Independent targets in one batch solve independently."""
  translation = torch.tensor(
    [[0.0, 0.135, -0.44], [0.05, 0.135, -0.40]], device=device, dtype=torch.float64
  )
  rpy = torch.zeros(2, 3, device=device, dtype=torch.float64)
  htf = make_transform(translation, rpy_intrinsic_to_mat(rpy))

  batched = calculate_leg_joints(htf, left=True)
  singles = torch.cat(
    [calculate_leg_joints(htf[i : i + 1], left=True) for i in range(2)], dim=0
  )

  assert torch.allclose(batched, singles, atol=1e-12)
  assert not torch.allclose(batched[0], batched[1])


def test_euler_helpers_round_trip(device):
  """rpy -> matrix -> rpy is the identity away from gimbal lock."""
  rpy = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.1, -0.2, 0.3],
      [-0.4, 0.5, -0.6],
      [0.0, math.pi / 12, 0.0],
    ],
    device=device,
    dtype=torch.float64,
  )

  recovered = mat_to_rpy_intrinsic(rpy_intrinsic_to_mat(rpy))

  assert torch.allclose(recovered, rpy, atol=1e-12)


def test_rpy_to_matrix_composes_as_z_y_x(device):
  """The convention is intrinsic ZYX, i.e. Rz(yaw) Ry(pitch) Rx(roll)."""
  rpy = torch.tensor([[0.3, -0.2, 0.5]], device=device, dtype=torch.float64)
  roll, pitch, yaw = rpy[0].tolist()

  def axis(angle, which):
    c, s = math.cos(angle), math.sin(angle)
    if which == "x":
      return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if which == "y":
      return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

  expected = axis(yaw, "z") @ axis(pitch, "y") @ axis(roll, "x")

  assert np.allclose(rpy_intrinsic_to_mat(rpy)[0].cpu().numpy(), expected, atol=1e-12)


def test_invert_transform_is_an_inverse(device):
  """Inverting a transform and composing gives the identity."""
  translation = torch.tensor([[0.1, -0.2, 0.3]], device=device, dtype=torch.float64)
  rpy = torch.tensor([[0.2, 0.3, -0.4]], device=device, dtype=torch.float64)
  transform = make_transform(translation, rpy_intrinsic_to_mat(rpy))

  product = transform @ invert_transform(transform)

  identity = torch.eye(4, device=device, dtype=torch.float64).unsqueeze(0)
  assert torch.allclose(product, identity, atol=1e-12)


##
# MuJoCo frame-convention checks.
##


@pytest.fixture(scope="module")
def nugus_model():
  return mujoco.MjModel.from_xml_path(str(NUGUS_XML))


def _sole_pose_in_torso_frame(model, data, side: str):
  """Foot sole pose in the walk engine's torso frame, via MuJoCo kinematics.

  Assumes the torso free joint is at the world origin with identity rotation,
  so world coordinates are torso-body coordinates.
  """
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
  rotation = data.xmat[body].reshape(3, 3)
  translation = (
    data.xpos[body]
    + rotation @ np.array(NUGUS_SOLE_OFFSET)
    - np.array(NUGUS_TORSO_FRAME_OFFSET)
  )
  return translation, rotation @ np.array(NUGUS_SOLE_ROTATION)


def _set_leg_joints(model, data, side: str, joints):
  data.qpos[:] = 0.0
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in zip(JOINT_NAMES, joints, strict=True):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value
  mujoco.mj_forward(model, data)


def test_sole_frame_convention_holds_at_stance(nugus_model):
  """The sole frame is right-handed and oriented as the engine expects.

  At the standing keyframe the sole normal must point up and the heel-to-toe
  axis forward. A wrong sign in :data:`NUGUS_SOLE_ROTATION` shows up here rather
  than as a mysterious IK residual later.
  """
  data = mujoco.MjData(nugus_model)
  data.qpos[:3] = STAND_BENT_KNEES_KEYFRAME.pos
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  assert STAND_BENT_KNEES_KEYFRAME.joint_pos is not None
  for name, value in STAND_BENT_KNEES_KEYFRAME.joint_pos.items():
    joint_id = mujoco.mj_name2id(nugus_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[nugus_model.jnt_qposadr[joint_id]] = value
  mujoco.mj_forward(nugus_model, data)

  for side in ("left", "right"):
    _, rotation = _sole_pose_in_torso_frame(nugus_model, data, side)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6), "not right-handed"
    assert rotation[:, 0] @ np.array([1.0, 0.0, 0.0]) > 0.9, "forward axis is wrong"
    assert rotation[:, 2] @ np.array([0.0, 0.0, 1.0]) > 0.9, "sole normal is wrong"


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_reproduces_commanded_foot_orientation(nugus_model, side):
  """Driving MuJoCo with the IK solution recovers the commanded orientation.

  Orientation is unaffected by the link-length idealisation, so this pins down
  every rotational convention at once: the IK's internal frame swap, the
  left/right mirror, the joint sign table, and the sole frame.

  The residual floor is the MJCF's own precision, not the solver's. Body quats
  are written to six decimals -- ``quat="0.707105 0.707108 0 0"`` is 4.2e-6 rad
  away from a true quarter turn -- so the tolerance sits just above that.
  """
  data = mujoco.MjData(nugus_model)
  rng = np.random.default_rng(7)
  y_sign = 1.0 if side == "left" else -1.0

  worst = 0.0
  for _ in range(64):
    translation = np.array(
      [
        rng.uniform(-0.08, 0.08),
        y_sign * rng.uniform(0.10, 0.17),
        rng.uniform(-0.46, -0.38),
      ]
    )
    rpy = np.array(
      [rng.uniform(-0.10, 0.10), rng.uniform(-0.30, 0.05), rng.uniform(-0.20, 0.20)]
    )
    htf = make_transform(
      torch.tensor(translation).unsqueeze(0),
      rpy_intrinsic_to_mat(torch.tensor(rpy).unsqueeze(0)),
    )
    joints = calculate_leg_joints(htf, left=side == "left")[0].numpy()

    _set_leg_joints(nugus_model, data, side, joints)
    _, rotation = _sole_pose_in_torso_frame(nugus_model, data, side)

    actual = mat_to_rpy_intrinsic(torch.tensor(rotation).unsqueeze(0))[0].numpy()
    error = np.abs((actual - rpy + np.pi) % (2 * np.pi) - np.pi).max()
    worst = max(worst, float(error))

  assert worst < 1e-5, f"orientation error {worst:.2e} rad"


@pytest.mark.parametrize("side", ["left", "right"])
def test_report_idealised_model_placement_error(nugus_model, side, capsys):
  """Quantify the foot-placement error the idealised leg model introduces.

  This is characterisation, not a correctness check: the engine plans against
  two straight 0.2 m links, the MJCF thigh is 0.2138 m and tilted, and the gap
  between them is a real property of the deployed controller that the
  comparison inherits. The assertion only guards against frame or sign errors,
  which are far larger.
  """
  data = mujoco.MjData(nugus_model)
  rng = np.random.default_rng(11)
  y_sign = 1.0 if side == "left" else -1.0

  errors = []
  for _ in range(64):
    translation = np.array(
      [
        rng.uniform(-0.08, 0.08),
        y_sign * rng.uniform(0.10, 0.17),
        rng.uniform(-0.46, -0.38),
      ]
    )
    rpy = np.array(
      [rng.uniform(-0.10, 0.10), rng.uniform(-0.30, 0.05), rng.uniform(-0.20, 0.20)]
    )
    htf = make_transform(
      torch.tensor(translation).unsqueeze(0),
      rpy_intrinsic_to_mat(torch.tensor(rpy).unsqueeze(0)),
    )
    joints = calculate_leg_joints(htf, left=side == "left")[0].numpy()

    _set_leg_joints(nugus_model, data, side, joints)
    actual, _ = _sole_pose_in_torso_frame(nugus_model, data, side)
    errors.append(actual - translation)

  errors = np.array(errors)
  magnitude = np.linalg.norm(errors, axis=1)
  with capsys.disabled():
    print(
      f"\n  {side:>5} foot placement error vs idealised model: "
      f"mean {magnitude.mean() * 1000:.1f} mm, max {magnitude.max() * 1000:.1f} mm, "
      f"bias (x, y, z) = {np.round(errors.mean(0) * 1000, 1)} mm"
    )

  assert magnitude.max() < MAX_MODEL_PLACEMENT_ERROR
