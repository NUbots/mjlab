"""Tests for the batched quintic splines of the ported NUbots walk engine."""

import math

import pytest
import torch
from conftest import get_test_device

from mjlab.controllers.quintic_walk.spline import (
  Trajectory,
  acceleration,
  build_trajectory,
  make_waypoint,
  position,
  quintic_coefficients,
  velocity,
)


@pytest.fixture
def device():
  return get_test_device()


def _waypoint(p, v=0.0, a=0.0, device="cpu"):
  return torch.tensor([[p, v, a]], device=device)


def test_coefficients_match_hand_computed_smoothstep(device):
  """A 0->1 rest-to-rest quintic is the classic 10t^3 - 15t^4 + 6t^5."""
  start = _waypoint(0.0, device=device)
  end = _waypoint(1.0, device=device)
  duration = torch.ones(1, device=device)

  coeffs = quintic_coefficients(start, end, duration)

  expected = torch.tensor([[0.0, 0.0, 0.0, 10.0, -15.0, 6.0]], device=device)
  assert torch.allclose(coeffs, expected, atol=1e-6)


def test_endpoints_reproduce_boundary_conditions(device):
  """Position, velocity and acceleration are honoured at both ends."""
  start = _waypoint(0.3, v=-0.2, a=1.5, device=device)
  end = _waypoint(-0.7, v=0.9, a=-2.0, device=device)
  duration = torch.full((1,), 0.32, device=device)

  coeffs = quintic_coefficients(start, end, duration)

  zero = torch.zeros(1, device=device)
  assert torch.allclose(position(coeffs, zero), start[:, 0], atol=1e-6)
  assert torch.allclose(velocity(coeffs, zero), start[:, 1], atol=1e-6)
  assert torch.allclose(acceleration(coeffs, zero), start[:, 2], atol=1e-6)
  assert torch.allclose(position(coeffs, duration), end[:, 0], atol=1e-5)
  assert torch.allclose(velocity(coeffs, duration), end[:, 1], atol=1e-4)
  assert torch.allclose(acceleration(coeffs, duration), end[:, 2], atol=1e-3)


def test_derivatives_agree_with_finite_differences(device):
  """The analytic derivatives match numerical ones away from the endpoints."""
  start = _waypoint(0.1, v=0.4, a=-0.3, device=device)
  end = _waypoint(0.6, v=-0.1, a=0.2, device=device)
  duration = torch.full((1,), 0.5, device=device)
  coeffs = quintic_coefficients(start, end, duration).double()

  t = torch.full((1,), 0.2, device=device, dtype=torch.float64)
  h = 1e-6
  numeric_velocity = (position(coeffs, t + h) - position(coeffs, t - h)) / (2 * h)
  numeric_acceleration = (velocity(coeffs, t + h) - velocity(coeffs, t - h)) / (2 * h)

  assert torch.allclose(velocity(coeffs, t), numeric_velocity, atol=1e-6)
  assert torch.allclose(acceleration(coeffs, t), numeric_acceleration, atol=1e-4)


def test_batched_envs_are_independent(device):
  """Each batch element gets its own coefficients."""
  start = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
  end = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
  duration = torch.tensor([1.0, 2.0], device=device)

  coeffs = quintic_coefficients(start, end, duration)
  half = position(coeffs, duration * 0.5)

  assert torch.allclose(half, torch.tensor([0.5, 0.5], device=device), atol=1e-6)


def _three_waypoint_trajectory(device, t_mid=0.16, t_end=0.32) -> Trajectory:
  """Trajectory shaped like the walk engine's swing foot: rise then plant."""
  first = make_waypoint(
    torch.tensor([[0.0, -0.27, 0.0]], device=device),
    torch.zeros(1, 3, device=device),
  )
  middle = make_waypoint(
    torch.tensor([[0.0, -0.27, 0.085]], device=device),
    torch.zeros(1, 3, device=device),
    linear_velocity=torch.tensor([[0.2, 0.0, 0.0]], device=device),
  )
  last = make_waypoint(
    torch.tensor([[0.064, -0.27, 0.0]], device=device),
    torch.zeros(1, 3, device=device),
  )
  return build_trajectory(
    first,
    middle,
    last,
    torch.full((1,), t_mid, device=device),
    torch.full((1,), t_end, device=device),
  )


def test_trajectory_interpolates_all_three_waypoints(device):
  """The trajectory passes through each waypoint at its own timepoint."""
  traj = _three_waypoint_trajectory(device)

  for t, expected in (
    (0.0, [0.0, -0.27, 0.0]),
    (0.16, [0.0, -0.27, 0.085]),
    (0.32, [0.064, -0.27, 0.0]),
  ):
    pos, _ = traj.evaluate(torch.full((1,), t, device=device))
    assert torch.allclose(pos, torch.tensor([expected], device=device), atol=1e-5), (
      f"mismatch at t={t}"
    )


def test_trajectory_selects_segment_on_strict_greater_than(device):
  """Segment selection matches the C++: segment 1 only for t > t_mid.

  At exactly ``t == t_mid`` the C++ loop condition ``time > timepoints[i + 1]``
  is false, so the first segment is used. Both segments agree at the shared
  waypoint, so this is checked via the segment index itself.
  """
  traj = _three_waypoint_trajectory(device)
  t_mid = traj.t_mid

  _, at_mid = traj._segment_coeffs(t_mid)
  _, just_after = traj._segment_coeffs(t_mid + 1e-4)

  # First segment => local time is measured from zero, so equals absolute time.
  assert torch.allclose(at_mid, t_mid, atol=1e-9)
  # Second segment => local time restarts from t_mid, so collapses to ~zero.
  assert (just_after < 1e-3).all()


def test_two_waypoint_trajectory_ignores_degenerate_segment(device):
  """The starting-step trajectory stays finite and hits both waypoints.

  The walk engine builds two-waypoint trajectories by collapsing the middle and
  final waypoints. The resulting zero-duration second segment must never be
  selected, and must not poison the result with inf/NaN.
  """
  start = make_waypoint(
    torch.tensor([[0.0, -0.27, 0.0]], device=device),
    torch.zeros(1, 3, device=device),
  )
  end = make_waypoint(
    torch.tensor([[0.02, -0.27, 0.0]], device=device),
    torch.tensor([[0.0, 0.0, 0.4]], device=device),
  )
  period = torch.full((1,), 0.32, device=device)

  traj = build_trajectory(start, end, end, period, period)

  for t in (0.0, 0.1, 0.2, 0.32):
    pos, rpy = traj.evaluate(torch.full((1,), t, device=device))
    assert torch.isfinite(pos).all(), f"non-finite position at t={t}"
    assert torch.isfinite(rpy).all(), f"non-finite orientation at t={t}"

  pos, rpy = traj.evaluate(period)
  assert torch.allclose(
    pos, torch.tensor([[0.02, -0.27, 0.0]], device=device), atol=1e-5
  )
  assert torch.allclose(rpy, torch.tensor([[0.0, 0.0, 0.4]], device=device), atol=1e-5)


def test_trajectory_batches_distinct_step_periods(device):
  """Envs with different periods and mid-times evaluate independently."""
  first = make_waypoint(
    torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)
  )
  middle = make_waypoint(
    torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], device=device),
    torch.zeros(2, 3, device=device),
  )
  last = make_waypoint(
    torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], device=device),
    torch.zeros(2, 3, device=device),
  )
  t_mid = torch.tensor([0.16, 0.10], device=device)
  t_end = torch.tensor([0.32, 0.25], device=device)

  traj = build_trajectory(first, middle, last, t_mid, t_end)
  pos, _ = traj.evaluate(t_mid)

  assert torch.allclose(pos[:, 2], torch.tensor([0.1, 0.2], device=device), atol=1e-5)


def test_orientation_channels_are_splined_independently(device):
  """Roll, pitch and yaw follow their own quintics."""
  first = make_waypoint(
    torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device)
  )
  middle = make_waypoint(
    torch.zeros(1, 3, device=device),
    torch.tensor([[0.0, math.pi / 12, 0.1]], device=device),
  )
  last = make_waypoint(
    torch.zeros(1, 3, device=device),
    torch.tensor([[0.0, math.pi / 12, 0.4]], device=device),
  )
  traj = build_trajectory(
    first,
    middle,
    last,
    torch.full((1,), 0.16, device=device),
    torch.full((1,), 0.32, device=device),
  )

  _, rpy = traj.evaluate(torch.full((1,), 0.32, device=device))

  assert torch.allclose(
    rpy, torch.tensor([[0.0, math.pi / 12, 0.4]], device=device), atol=1e-5
  )
