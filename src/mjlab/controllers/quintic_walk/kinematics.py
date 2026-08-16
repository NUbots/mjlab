"""Batched leg inverse kinematics and Euler helpers for the quintic walk.

Ported line-by-line from the NUbots C++:

- ``shared/utility/actuation/InverseKinematics.hpp`` (``calculate_leg_joints``)
- ``shared/utility/math/euler.hpp``
- ``shared/utility/math/angle.hpp``

The solver is closed form -- no iteration, no solver state -- so the only work
needed to run thousands of robots at once is to give every quantity a leading
batch dimension and turn the handful of scalar branches into ``torch.where``
selects.

Fidelity note: this reproduces the deployed controller's *idealised* leg model
(equal 0.2 m links, coincident hip axes) rather than the URDF's true link
lengths. The mismatch is deliberate. The on-robot walk engine plans against
this model, so a faithful comparison has to inherit its model error too. See
``LegModel`` for the numbers and ``tests/test_quintic_walk_kinematics.py`` for
the measured residual against the MuJoCo model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Guard for divisions by a vector norm. Eigen's ``normalized()`` has no such
# guard and produces NaN at the origin; batched evaluation cannot afford a
# per-element NaN, so degenerate directions collapse to zero instead.
EPS = 1e-9


@dataclass(frozen=True)
class LegModel:
  """Idealised leg geometry used by the analytic IK.

  Mirrors the subset of the NUbots ``KinematicsModel`` that
  ``calculate_leg_joints`` reads, populated from
  ``module/actuation/KinematicsConfiguration/data/config/KinematicsConfiguration.yaml``.
  """

  upper_leg_length: float
  lower_leg_length: float
  hip_offset_x: float
  hip_offset_y: float
  hip_offset_z: float
  foot_height: float
  left_to_right_hip_yaw: float
  left_to_right_hip_roll: float
  left_to_right_hip_pitch: float
  left_to_right_knee: float
  left_to_right_ankle_pitch: float
  left_to_right_ankle_roll: float

  @property
  def length_between_legs(self) -> float:
    """Lateral hip separation; ``2 * HIP_OFFSET_Y`` in the C++ config loader."""
    return 2.0 * self.hip_offset_y

  @property
  def max_leg_length(self) -> float:
    return self.upper_leg_length + self.lower_leg_length

  @property
  def left_to_right(self) -> tuple[float, ...]:
    """Per-joint sign flips taking a left-leg solution to the right leg."""
    return (
      self.left_to_right_hip_yaw,
      self.left_to_right_hip_roll,
      self.left_to_right_hip_pitch,
      self.left_to_right_knee,
      self.left_to_right_ankle_pitch,
      self.left_to_right_ankle_roll,
    )


NUGUS_LEG = LegModel(
  upper_leg_length=0.2,
  lower_leg_length=0.2,
  hip_offset_x=0.0,
  hip_offset_y=0.055,
  hip_offset_z=0.045,
  foot_height=0.04,
  left_to_right_hip_yaw=-1.0,
  left_to_right_hip_roll=-1.0,
  left_to_right_hip_pitch=1.0,
  left_to_right_knee=1.0,
  left_to_right_ankle_pitch=1.0,
  left_to_right_ankle_roll=-1.0,
)
"""Leg model for the NUgus, matching the robot the walk engine is deployed on."""

##
# Frame adapters between the walk engine's conventions and the mjlab MJCF.
##

NUGUS_TORSO_FRAME_OFFSET = (0.0, 0.0, 0.045)
"""Walk-engine torso frame origin, in the mjlab ``torso`` body frame.

The engine's torso frame sits ``hip_offset_z`` above the hip axes, whereas the
MJCF puts the ``torso`` body origin *on* them (``left_hip_yaw`` is at
``pos="0 0.055 0"``). Subtract this from a point in the ``torso`` body frame to
get its coordinates in the engine's frame.
"""

NUGUS_SOLE_OFFSET = (0.039, 0.0, 0.0)
"""Foot sole origin, in the ``{left,right}_foot`` body frame.

Read off the four ``*_foot_c[0-3]`` corner sites, which all share a local x of
0.039 and so span the sole plane. The engine's own ``foot_height`` is 0.04, a
1 mm idealisation of this.
"""

NUGUS_SOLE_ROTATION = (
  (0.0, 0.0, -1.0),
  (0.0, 1.0, 0.0),
  (1.0, 0.0, 0.0),
)
"""Sole frame orientation, in the ``{left,right}_foot`` body frame.

Columns are the sole axes: x from heel to toe (body +z, the long axis of the
corner sites), z the upward sole normal (body -x), y completing a right-handed
frame. This matches the engine's foot convention: x forward, y left, z normal
to the sole.
"""

# Joint ordering of the IK solution, matching the C++ ``emplace_back`` order.
JOINT_NAMES = (
  "hip_yaw",
  "hip_roll",
  "hip_pitch",
  "knee_pitch",
  "ankle_pitch",
  "ankle_roll",
)


def acos_clamped(x: torch.Tensor) -> torch.Tensor:
  """``acos`` with its argument clamped into the valid domain."""
  return torch.acos(x.clamp(-1.0, 1.0))


def asin_clamped(x: torch.Tensor) -> torch.Tensor:
  """``asin`` with its argument clamped into the valid domain."""
  return torch.asin(x.clamp(-1.0, 1.0))


def normalize(v: torch.Tensor) -> torch.Tensor:
  """Normalise along the last dimension, mapping the zero vector to itself."""
  return v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)


def angle_between(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Angle between two unit vectors, stable when they are near parallel.

  Uses the half-chord formulation from https://www.plunk.org/~hatch/rightway.html,
  as the C++ does.
  """
  dot = (u * v).sum(-1)
  near = 2.0 * asin_clamped((v - u).norm(dim=-1) * 0.5)
  far = math.pi - 2.0 * asin_clamped((-v - u).norm(dim=-1) * 0.5)
  return torch.where(dot < 0, far, near)


def mat_to_rpy_intrinsic(mat: torch.Tensor) -> torch.Tensor:
  """Rotation matrices to intrinsic ZYX (roll, pitch, yaw) Euler angles.

  Args:
    mat: Shape ``(..., 3, 3)`` rotation matrices.

  Returns:
    Shape ``(..., 3)`` angles ordered (roll, pitch, yaw).
  """
  m21 = mat[..., 2, 1]
  m22 = mat[..., 2, 2]
  roll = torch.atan2(m21, m22)
  pitch = torch.atan2(-mat[..., 2, 0], torch.sqrt(m21 * m21 + m22 * m22))
  yaw = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
  return torch.stack((roll, pitch, yaw), dim=-1)


def rpy_intrinsic_to_mat(rpy: torch.Tensor) -> torch.Tensor:
  """Intrinsic ZYX (roll, pitch, yaw) Euler angles to rotation matrices.

  Args:
    rpy: Shape ``(..., 3)`` angles ordered (roll, pitch, yaw).

  Returns:
    Shape ``(..., 3, 3)`` rotation matrices equal to ``Rz(yaw) Ry(pitch) Rx(roll)``.
  """
  roll, pitch, yaw = rpy.unbind(-1)
  cr, sr = torch.cos(roll), torch.sin(roll)
  cp, sp = torch.cos(pitch), torch.sin(pitch)
  cy, sy = torch.cos(yaw), torch.sin(yaw)
  row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
  row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
  row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
  return torch.stack((row0, row1, row2), dim=-2)


def make_transform(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
  """Assemble homogeneous transforms from a translation and a rotation.

  Args:
    translation: Shape ``(..., 3)``.
    rotation: Shape ``(..., 3, 3)``.

  Returns:
    Shape ``(..., 4, 4)`` homogeneous transforms.
  """
  batch = translation.shape[:-1]
  transform = torch.zeros(
    (*batch, 4, 4), device=translation.device, dtype=translation.dtype
  )
  transform[..., :3, :3] = rotation
  transform[..., :3, 3] = translation
  transform[..., 3, 3] = 1.0
  return transform


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
  """Invert homogeneous transforms, exploiting orthonormality of the rotation."""
  rotation = transform[..., :3, :3]
  translation = transform[..., :3, 3]
  inv_rotation = rotation.transpose(-1, -2)
  inv_translation = -torch.einsum("...ij,...j->...i", inv_rotation, translation)
  return make_transform(inv_translation, inv_rotation)


# Maps the walk engine's torso convention (x forward, y left, z up) into the
# convention the IK derivation works in. Applied as ``R p`` and ``R M R^T``.
_IK_FRAME = (
  (0.0, 1.0, 0.0),
  (1.0, 0.0, 0.0),
  (0.0, 0.0, -1.0),
)


def calculate_leg_joints(
  htf: torch.Tensor, left: bool, model: LegModel = NUGUS_LEG
) -> torch.Tensor:
  """Solve the six leg joint angles placing a foot at a desired pose.

  Args:
    htf: Shape ``(N, 4, 4)`` desired pose of the foot ``{f}`` in the torso
      ``{t}`` frame. The torso frame has x forward, y left, z up, with its
      origin ``hip_offset_z`` above the midpoint of the hips. The foot frame is
      referenced to the *sole*: x from heel to toe, y left, z normal to the
      sole plane.
    left: Whether to solve for the left leg. Right-leg solutions apply the
      model's ``left_to_right`` sign flips.
    model: Leg geometry.

  Returns:
    Shape ``(N, 6)`` joint angles ordered as :data:`JOINT_NAMES`.
  """
  frame = torch.tensor(_IK_FRAME, device=htf.device, dtype=htf.dtype)

  rotation = htf[..., :3, :3]
  translation = htf[..., :3, 3]

  # Shift from the sole to the ankle. Eigen's ``translate`` is a local-frame
  # (right) multiply, so the offset is rotated by the foot's own orientation.
  translation = translation + rotation[..., :, 2] * model.foot_height

  translation = torch.einsum("ij,...j->...i", frame, translation)
  rotation = frame @ rotation @ frame.transpose(-1, -2)

  if not left:
    # Mirror the target about the sagittal plane so the left-leg derivation
    # applies; the joint signs are flipped back at the end.
    mirror = torch.tensor(
      [[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
      device=htf.device,
      dtype=htf.dtype,
    )
    rotation = rotation * mirror
    translation = translation * torch.tensor(
      [-1.0, 1.0, 1.0], device=htf.device, dtype=htf.dtype
    )

  ankle_x = rotation[..., :, 0]
  ankle_y = rotation[..., :, 1]

  hip_offset = torch.tensor(
    [0.5 * model.length_between_legs, model.hip_offset_x, model.hip_offset_z],
    device=htf.device,
    dtype=htf.dtype,
  )
  r_ftt = translation - hip_offset

  # NOTE: ``length`` is deliberately *not* recomputed after the clamp below.
  # The C++ leaves the pre-clamp length in place, so an over-extended target
  # drives ``cos_knee`` past -1 and ``acos_clamped`` straightens the leg. That
  # saturation behaviour is part of what the deployed controller does.
  length = r_ftt.norm(dim=-1)
  overextended = length > model.max_leg_length
  scale = torch.where(
    overextended,
    model.max_leg_length / length.clamp_min(EPS),
    torch.ones_like(length),
  )
  r_ftt = r_ftt * scale.unsqueeze(-1)
  safe_length = length.clamp_min(EPS)

  # Knee pitch.
  sqr_length = length * length
  sqr_upper = model.upper_leg_length**2
  sqr_lower = model.lower_leg_length**2
  cos_knee = (sqr_upper + sqr_lower - sqr_length) / (
    2.0 * model.upper_leg_length * model.lower_leg_length
  )
  knee_pitch = math.pi - acos_clamped(cos_knee)

  # Ankle pitch.
  cos_lower_leg = (sqr_lower + sqr_length - sqr_upper) / (
    2.0 * model.lower_leg_length * safe_length
  )
  lower_leg = acos_clamped(cos_lower_leg)
  phi2 = acos_clamped((r_ftt * ankle_y).sum(-1) / safe_length)
  ankle_pitch = -(lower_leg + phi2 - 0.5 * math.pi)

  # Ankle roll.
  u_ftt = normalize(r_ftt)
  hip_x = normalize(torch.cross(ankle_y, u_ftt, dim=-1))
  leg_plane_tangent = torch.cross(ankle_y, hip_x, dim=-1)
  ankle_roll = torch.atan2(
    (ankle_x * leg_plane_tangent).sum(-1), (ankle_x * hip_x).sum(-1)
  )

  # Hip roll.
  unit_z = torch.zeros_like(hip_x)
  unit_z[..., 2] = 1.0
  is_ankle_above_waist = u_ftt[..., 2] < 0
  above_sign = torch.where(
    is_ankle_above_waist, -torch.ones_like(length), torch.ones_like(length)
  )
  cos_z_and_hip_x = hip_x[..., 2]
  hip_roll_positive = cos_z_and_hip_x <= 0
  leg_plane_global_z = above_sign.unsqueeze(-1) * (
    unit_z - cos_z_and_hip_x.unsqueeze(-1) * hip_x
  )
  leg_plane_global_z = normalize(leg_plane_global_z)
  hip_roll = torch.where(
    hip_roll_positive, torch.ones_like(length), -torch.ones_like(length)
  ) * angle_between(leg_plane_global_z, unit_z)

  # Hip pitch.
  phi4 = knee_pitch - lower_leg
  sin_pi_minus_phi2 = torch.sin(math.pi - phi2)
  sin_pi_minus_phi2 = torch.where(
    sin_pi_minus_phi2.abs() < EPS,
    torch.full_like(sin_pi_minus_phi2, EPS),
    sin_pi_minus_phi2,
  )
  unit_upper_leg = u_ftt * (torch.sin(phi2 - phi4) / sin_pi_minus_phi2).unsqueeze(
    -1
  ) + ankle_y * (torch.sin(phi4) / sin_pi_minus_phi2).unsqueeze(-1)
  is_hip_pitch_positive = (
    hip_x * torch.cross(unit_upper_leg, leg_plane_global_z, dim=-1)
  ).sum(-1) >= 0
  hip_pitch = torch.where(
    is_hip_pitch_positive, -torch.ones_like(length), torch.ones_like(length)
  ) * angle_between(unit_upper_leg, leg_plane_global_z)

  # Hip yaw.
  unit_x = torch.zeros_like(hip_x)
  unit_x[..., 0] = 1.0
  hip_x_projected = above_sign.unsqueeze(-1) * hip_x
  hip_x_projected = normalize(
    torch.stack(
      (
        hip_x_projected[..., 0],
        hip_x_projected[..., 1],
        torch.zeros_like(hip_x_projected[..., 2]),
      ),
      dim=-1,
    )
  )
  is_hip_yaw_positive = hip_x_projected[..., 1] >= 0
  hip_yaw = torch.where(
    is_hip_yaw_positive, -torch.ones_like(length), torch.ones_like(length)
  ) * angle_between(hip_x_projected, unit_x)

  joints = torch.stack(
    (hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll), dim=-1
  )
  if not left:
    signs = torch.tensor(model.left_to_right, device=htf.device, dtype=htf.dtype)
    joints = joints * signs
  return joints
