"""GPU-parallel port of the NUbots quintic spline walk engine.

The engine is the pre-RL walk controller from https://github.com/NUbots/NUbots.
It is reproduced here so that it can be run inside mjlab, against the same
robot model, actuator model and domain randomisation as a learned policy, to
give a controlled comparison between the two.
"""

from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_LEG,
  LegModel,
  calculate_leg_joints,
  invert_transform,
  make_transform,
  mat_to_rpy_intrinsic,
  rpy_intrinsic_to_mat,
)
from mjlab.controllers.quintic_walk.spline import (
  Trajectory,
  build_trajectory,
  make_waypoint,
  quintic_coefficients,
)

__all__ = (
  "NUGUS_LEG",
  "LegModel",
  "Trajectory",
  "build_trajectory",
  "calculate_leg_joints",
  "invert_transform",
  "make_transform",
  "make_waypoint",
  "mat_to_rpy_intrinsic",
  "quintic_coefficients",
  "rpy_intrinsic_to_mat",
)
