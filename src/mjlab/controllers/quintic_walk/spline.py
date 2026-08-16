"""Batched quintic splines and 6-D pose trajectories.

Ported from the NUbots quintic walk engine, which in turn derives from the
Hamburg Bit-Bots / Rhoban ``model`` splines:

- ``shared/utility/skill/splines/QuinticSpline.hpp``
- ``shared/utility/skill/splines/Trajectory.hpp``

The C++ evaluates one robot at a time using ``std::vector`` of spline segments.
Here every quantity carries a leading batch dimension so that ``num_envs``
robots are evaluated in lockstep on the GPU.

The walk engine only ever builds trajectories with two or three waypoints (one
or two segments), so :class:`Trajectory` fixes the segment count at two and
degenerates the second segment for the two-waypoint case. See
:func:`build_trajectory` for how that degeneracy is made harmless.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Trajectory dimension ordering, matching ``TrajectoryDimension`` in the C++.
X, Y, Z, ROLL, PITCH, YAW = range(6)
NUM_DIMS = 6

# Smallest segment duration used when building spline coefficients. The C++
# throws for durations <= 1e-5; batched evaluation cannot throw per-element, so
# durations are clamped instead and the affected segment is never selected.
MIN_DURATION = 1e-5


def quintic_coefficients(
  start: torch.Tensor, end: torch.Tensor, duration: torch.Tensor
) -> torch.Tensor:
  """Solve for the coefficients of a quintic connecting two waypoints.

  Args:
    start: Shape ``(..., 3)`` holding (position, velocity, acceleration).
    end: Shape ``(..., 3)`` holding (position, velocity, acceleration).
    duration: Shape ``(...)`` segment duration in seconds.

  Returns:
    Shape ``(..., 6)`` polynomial coefficients, lowest order first.
  """
  p0, v0, a0 = start.unbind(-1)
  p1, v1, a1 = end.unbind(-1)

  t = duration.clamp_min(MIN_DURATION)
  t2 = t * t
  t3 = t2 * t
  t4 = t3 * t
  t5 = t4 * t

  c0 = p0
  c1 = v0
  c2 = a0 / 2
  c3 = -(-a1 * t2 + 3 * a0 * t2 + 8 * v1 * t + 12 * v0 * t - 20 * p1 + 20 * p0) / (
    2 * t3
  )
  c4 = (-2 * a1 * t2 + 3 * a0 * t2 + 14 * v1 * t + 16 * v0 * t - 30 * p1 + 30 * p0) / (
    2 * t4
  )
  c5 = -(-a1 * t2 + a0 * t2 + 6 * v1 * t + 6 * v0 * t - 12 * p1 + 12 * p0) / (2 * t5)

  return torch.stack((c0, c1, c2, c3, c4, c5), dim=-1)


def _polyval(
  coeffs: torch.Tensor, t: torch.Tensor, derivative: int = 0
) -> torch.Tensor:
  """Evaluate a polynomial (or one of its derivatives) via Horner-free powers.

  Args:
    coeffs: Shape ``(..., 6)`` coefficients, lowest order first.
    t: Shape ``(...)`` evaluation time, broadcast against ``coeffs[..., 0]``.
    derivative: 0 for position, 1 for velocity, 2 for acceleration.

  Returns:
    Shape ``(...)`` polynomial value.
  """
  order = coeffs.shape[-1]
  powers = torch.arange(order, device=coeffs.device, dtype=coeffs.dtype)
  # d-th derivative of t^i is (i!/(i-d)!) * t^(i-d), and vanishes for i < d.
  scale = torch.ones_like(powers)
  for k in range(derivative):
    scale = scale * (powers - k).clamp_min(0.0)
  exponent = (powers - derivative).clamp_min(0.0)
  basis = t.unsqueeze(-1) ** exponent
  return (coeffs * scale * basis).sum(-1)


def position(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Position of a quintic at time ``t``."""
  return _polyval(coeffs, t, derivative=0)


def velocity(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Velocity of a quintic at time ``t``."""
  return _polyval(coeffs, t, derivative=1)


def acceleration(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Acceleration of a quintic at time ``t``."""
  return _polyval(coeffs, t, derivative=2)


@dataclass
class Trajectory:
  """A batch of two-segment, 6-D piecewise-quintic trajectories.

  The six dimensions are (x, y, z, roll, pitch, yaw), matching the C++
  ``TrajectoryDimension`` ordering, and each is splined independently.

  Attributes:
    coeffs: Shape ``(N, 2, 6, 6)`` indexed ``[env, segment, dim, coefficient]``.
    t_mid: Shape ``(N,)`` time of the middle waypoint; the second segment is
      selected for ``t > t_mid``.
  """

  coeffs: torch.Tensor
  t_mid: torch.Tensor

  @property
  def num_envs(self) -> int:
    return self.coeffs.shape[0]

  def _segment_coeffs(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the active segment's coefficients and segment-local time.

    Mirrors ``Trajectory::eval``: the C++ walks the timepoint list and stops at
    the first segment whose end time is not exceeded, so with three waypoints
    the second segment is chosen exactly when ``t > t_mid``. Times are clamped
    to the step period upstream, so the search never runs past the last
    segment.
    """
    segment = (t > self.t_mid).long()
    t_local = t - torch.where(segment.bool(), self.t_mid, torch.zeros_like(self.t_mid))
    index = segment.view(-1, 1, 1, 1).expand(-1, 1, NUM_DIMS, self.coeffs.shape[-1])
    coeffs = torch.gather(self.coeffs, 1, index).squeeze(1)
    return coeffs, t_local

  def evaluate(
    self, t: torch.Tensor, derivative: int = 0
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate all six dimensions at ``t``.

    Args:
      t: Shape ``(N,)`` time within the step cycle.
      derivative: 0 for pose, 1 for twist, 2 for acceleration.

    Returns:
      Tuple of ``(N, 3)`` translation and ``(N, 3)`` intrinsic roll-pitch-yaw
      (or their derivatives).
    """
    coeffs, t_local = self._segment_coeffs(t)
    values = _polyval(coeffs, t_local.unsqueeze(-1), derivative=derivative)
    return values[:, X : Z + 1], values[:, ROLL : YAW + 1]


def build_trajectory(
  first: torch.Tensor,
  middle: torch.Tensor,
  last: torch.Tensor,
  t_mid: torch.Tensor,
  t_end: torch.Tensor,
) -> Trajectory:
  """Build a batch of two-segment trajectories from three waypoints.

  Args:
    first: Shape ``(N, 6, 3)`` waypoint at ``t = 0``, indexed
      ``[env, dim, (position, velocity, acceleration)]``.
    middle: Shape ``(N, 6, 3)`` waypoint at ``t = t_mid``.
    last: Shape ``(N, 6, 3)`` waypoint at ``t = t_end``.
    t_mid: Shape ``(N,)`` time of the middle waypoint.
    t_end: Shape ``(N,)`` time of the final waypoint.

  Returns:
    The assembled :class:`Trajectory`.

  For the two-waypoint trajectories the walk engine builds while starting a
  step, pass ``middle == last`` and ``t_mid == t_end``. The second segment then
  has zero duration and its coefficients are meaningless, but ``t > t_mid`` is
  false for every ``t <= t_end`` so it is never selected. The duration clamp in
  :func:`quintic_coefficients` keeps those unused coefficients finite.
  """
  first_seg = quintic_coefficients(first, middle, t_mid.unsqueeze(-1))
  second_seg = quintic_coefficients(middle, last, (t_end - t_mid).unsqueeze(-1))
  return Trajectory(coeffs=torch.stack((first_seg, second_seg), dim=1), t_mid=t_mid)


def make_waypoint(
  translation: torch.Tensor,
  orientation: torch.Tensor,
  linear_velocity: torch.Tensor | None = None,
  angular_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
  """Pack a 6-D waypoint into the ``(N, 6, 3)`` layout used by trajectories.

  Accelerations are always zero, matching every waypoint the walk engine
  builds.

  Args:
    translation: Shape ``(N, 3)`` position.
    orientation: Shape ``(N, 3)`` intrinsic roll-pitch-yaw.
    linear_velocity: Shape ``(N, 3)``, defaults to zero.
    angular_velocity: Shape ``(N, 3)``, defaults to zero.

  Returns:
    Shape ``(N, 6, 3)`` waypoint.
  """
  linear = torch.zeros_like(translation) if linear_velocity is None else linear_velocity
  angular = (
    torch.zeros_like(orientation) if angular_velocity is None else angular_velocity
  )
  values = torch.cat((translation, orientation), dim=-1)
  velocities = torch.cat((linear, angular), dim=-1)
  return torch.stack((values, velocities, torch.zeros_like(values)), dim=-1)
