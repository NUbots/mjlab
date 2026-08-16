"""Tests for the exact-geometry leg kinematics.

Unlike the idealised solver, which is validated against the NUbots C++, this one
has no upstream to diff against -- it is new code solving against the MJCF. So
it is pinned from two directions instead: the forward kinematics must agree with
MuJoCo's own, and the analytic Jacobian must agree with finite differences of
that forward kinematics. Given both, the solver is checked by round-trip.
"""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.nugus.nugus_constants import get_nugus_robot_cfg
from mjlab.controllers.quintic_walk.exact_kinematics import (
  DEFAULT_SEED,
  LEG_JOINT_SUFFIXES,
  LegChain,
  forward_kinematics,
  rotation_log,
  solve_leg_ik,
  target_from_engine_frame,
)
from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_SOLE_OFFSET,
  NUGUS_SOLE_ROTATION,
  NUGUS_TORSO_FRAME_OFFSET,
)
from mjlab.entity import Entity

DTYPE = torch.float64


@pytest.fixture(scope="module")
def model():
  """The NUgus with mjlab's actuators, matching what the controller drives."""
  return Entity(get_nugus_robot_cfg()).spec.compile()


@pytest.fixture
def device():
  return get_test_device()


def _joint_ids(model, side: str):
  return [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{suffix}")
    for suffix in LEG_JOINT_SUFFIXES
  ]


def _mujoco_sole_pose(model, data, side: str):
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
  rotation = data.xmat[body].reshape(3, 3)
  translation = data.xpos[body] + rotation @ np.array(NUGUS_SOLE_OFFSET)
  return translation, rotation @ np.array(NUGUS_SOLE_ROTATION)


@pytest.mark.parametrize("left", [True, False])
def test_forward_kinematics_matches_mujoco(model, left):
  """The chain reproduces MuJoCo's kinematics exactly.

  This is what makes 'exact' meaningful: the solver targets the same geometry
  the physics integrates, so a solved pose is genuinely realised.
  """
  side = "left" if left else "right"
  chain = LegChain.from_model(model, left=left, dtype=DTYPE)
  data = mujoco.MjData(model)
  joint_ids = _joint_ids(model, side)
  ranges = np.array([model.jnt_range[j] for j in joint_ids])
  qpos_adr = [model.jnt_qposadr[j] for j in joint_ids]

  rng = np.random.default_rng(3)
  worst_pos = worst_rot = 0.0
  for _ in range(64):
    q = rng.uniform(ranges[:, 0], ranges[:, 1])
    data.qpos[:] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = q
    mujoco.mj_forward(model, data)

    expected_pos, expected_rot = _mujoco_sole_pose(model, data, side)
    pose, _, _ = forward_kinematics(chain, torch.tensor(q, dtype=DTYPE).unsqueeze(0))
    worst_pos = max(
      worst_pos, float(np.abs(pose[0, :3, 3].numpy() - expected_pos).max())
    )
    worst_rot = max(
      worst_rot, float(np.abs(pose[0, :3, :3].numpy() - expected_rot).max())
    )

  assert worst_pos < 1e-12, f"position mismatch {worst_pos:.2e}"
  assert worst_rot < 1e-12, f"rotation mismatch {worst_rot:.2e}"


def test_jacobian_matches_finite_differences(model):
  """The analytic Jacobian is the derivative of the forward kinematics.

  A wrong Jacobian would still converge sometimes, just slowly and from fewer
  seeds, so this is checked directly rather than inferred from solver success.
  """
  chain = LegChain.from_model(model, left=True, dtype=DTYPE)
  rng = np.random.default_rng(1)
  q = torch.tensor(rng.uniform(-0.4, 0.6, size=(1, 6)), dtype=DTYPE)

  pose, axes, anchors = forward_kinematics(chain, q)
  lever = pose[:, None, :3, 3] - anchors
  analytic = torch.cat((torch.cross(axes, lever, dim=-1), axes), dim=-1).transpose(1, 2)

  step = 1e-7
  numeric = torch.zeros(1, 6, 6, dtype=DTYPE)
  for i in range(6):
    plus, minus = q.clone(), q.clone()
    plus[0, i] += step
    minus[0, i] -= step
    pose_plus, _, _ = forward_kinematics(chain, plus)
    pose_minus, _, _ = forward_kinematics(chain, minus)
    numeric[0, :3, i] = (pose_plus[0, :3, 3] - pose_minus[0, :3, 3]) / (2 * step)
    numeric[0, 3:, i] = rotation_log(
      pose_plus[:, :3, :3] @ pose_minus[:, :3, :3].transpose(-1, -2)
    )[0] / (2 * step)

  assert float((analytic - numeric).abs().max()) < 1e-7


@pytest.mark.parametrize("left", [True, False])
def test_solver_round_trips_from_the_default_seed(model, left):
  """Poses reachable in the walk workspace are recovered from a cold start."""
  chain = LegChain.from_model(model, left=left, dtype=DTYPE)
  rng = np.random.default_rng(11)
  y_sign = 1.0 if left else -1.0

  # Joint configurations spanning a plausible walk posture, converted to poses.
  q_true = torch.tensor(
    np.stack(
      [
        rng.uniform(-0.1, 0.1, 32),
        y_sign * rng.uniform(0.05, 0.3, 32),
        rng.uniform(-1.0, -0.5, 32),
        rng.uniform(0.9, 1.8, 32),
        rng.uniform(-0.8, -0.3, 32),
        y_sign * rng.uniform(-0.3, 0.0, 32),
      ],
      axis=-1,
    ),
    dtype=DTYPE,
  )
  target, _, _ = forward_kinematics(chain, q_true)
  seed = torch.tensor(DEFAULT_SEED, dtype=DTYPE).expand(32, 6).clone()

  solution = solve_leg_ik(chain, target, seed, iterations=40)

  pose, _, _ = forward_kinematics(chain, solution)
  position_error = (pose[:, :3, 3] - target[:, :3, 3]).norm(dim=-1).max()
  rotation_error = (
    rotation_log(target[:, :3, :3] @ pose[:, :3, :3].transpose(-1, -2))
    .norm(dim=-1)
    .max()
  )

  # Thresholds sit just above the solver's own early-stop tolerance (1e-8) and
  # are still a micron -- orders of magnitude below anything physically
  # meaningful for foot placement.
  assert float(position_error) < 1e-6
  assert float(rotation_error) < 1e-6


def test_solver_converges_in_a_few_warm_started_iterations(model):
  """At control rates the previous solution is close, so 8 iterations suffice.

  The controller runs a fixed iteration budget rather than looping to a
  tolerance, so that budget has to be enough from a realistic warm start.

  Sampled over walk postures rather than the full joint range: convergence
  slows near the straight-leg singularity, which the walk never visits (its
  knee stays around 0.8-1.8 rad).
  """
  chain = LegChain.from_model(model, left=True, dtype=DTYPE)
  rng = np.random.default_rng(2)
  q_true = torch.tensor(
    np.stack(
      [
        rng.uniform(-0.1, 0.1, 16),
        rng.uniform(0.05, 0.3, 16),
        rng.uniform(-1.0, -0.5, 16),
        rng.uniform(0.9, 1.8, 16),
        rng.uniform(-0.8, -0.3, 16),
        rng.uniform(-0.3, 0.0, 16),
      ],
      axis=-1,
    ),
    dtype=DTYPE,
  )
  target, _, _ = forward_kinematics(chain, q_true)

  # A control step moves the joints by well under 0.05 rad at 100 Hz.
  warm = q_true + 0.05

  solution = solve_leg_ik(chain, target, warm, iterations=8)

  pose, _, _ = forward_kinematics(chain, solution)
  assert float((pose[:, :3, 3] - target[:, :3, 3]).norm(dim=-1).max()) < 1e-6


def test_zero_seed_is_documented_as_singular(model):
  """The straight-leg configuration really is a bad seed.

  Guards the DEFAULT_SEED rationale: if this ever starts converging, the
  comment explaining why zero is avoided has gone stale.
  """
  chain = LegChain.from_model(model, left=True, dtype=DTYPE)
  rng = np.random.default_rng(7)
  q_true = torch.tensor(rng.uniform(-0.4, 0.9, size=(16, 6)), dtype=DTYPE)
  target, _, _ = forward_kinematics(chain, q_true)

  from_zero = solve_leg_ik(chain, target, torch.zeros(16, 6, dtype=DTYPE), 40)
  from_default = solve_leg_ik(
    chain, target, torch.tensor(DEFAULT_SEED, dtype=DTYPE).expand(16, 6).clone(), 40
  )

  def worst(solution):
    pose, _, _ = forward_kinematics(chain, solution)
    return float((pose[:, :3, 3] - target[:, :3, 3]).norm(dim=-1).max())

  assert worst(from_default) < 1e-6
  assert worst(from_default) < worst(from_zero)


def test_engine_frame_conversion_shifts_by_the_hip_offset(device):
  """Engine-frame poses drop by the torso frame offset, orientation untouched."""
  htf = torch.eye(4, device=device).unsqueeze(0).clone()
  htf[0, :3, 3] = torch.tensor([0.08, 0.135, -0.432], device=device)

  converted = target_from_engine_frame(htf)

  assert float(converted[0, 2, 3]) == pytest.approx(
    -0.432 + NUGUS_TORSO_FRAME_OFFSET[2]
  )
  assert torch.allclose(converted[0, :3, :3], htf[0, :3, :3])
