"""Exact leg kinematics for the real MJCF geometry.

The NUbots IK plans against an idealised leg -- two straight 0.2 m links with
coincident hip axes -- which is what the robot deploys. The NUgus MJCF is not
that leg: its thigh is 0.2138 m and tilts 15.4 degrees backwards, and the hip
yaw and roll axes are 22 mm apart. Solving against the idealised model therefore
places the foot up to ~9 cm from where the engine asked, and leaves the robot
standing 86 mm taller with 0.36 rad less knee flexion than it does on hardware.

This module solves the same problem against the actual model, so the engine's
commanded foot poses are realised as intended. It is the *second condition* for
the walk-engine comparison, not a replacement: see
:class:`~mjlab.controllers.quintic_walk.controller.QuinticWalkController` for how
to select between them.

The chain is read out of the compiled ``MjModel`` rather than hard-coded, so it
tracks whatever geometry the asset actually has. The solver is damped
least-squares Gauss-Newton over the exact forward kinematics, warm-started from
the previous solution, which converges in a handful of iterations at control
rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_SOLE_OFFSET,
  NUGUS_SOLE_ROTATION,
  NUGUS_TORSO_FRAME_OFFSET,
  make_transform,
)
from mjlab.controllers.quintic_walk.walk_generator import ENGINE_DTYPE
from mjlab.utils.lab_api.math import matrix_from_quat

# Motor joints of each leg, in chain order from the torso.
LEG_JOINT_SUFFIXES = (
  "hip_yaw",
  "hip_roll",
  "hip_pitch",
  "knee_pitch",
  "ankle_pitch",
  "ankle_roll",
)

DEFAULT_SEED: tuple[float, ...] = (0.0, 0.0, -0.6, 1.2, -0.6, 0.0)
"""Cold-start seed for the solver: a mildly crouched leg.

Never seed from zero. A straight leg is a kinematic singularity -- the knee
axis contributes nothing to vertical motion there -- so Gauss-Newton takes a
huge first step and can land in a different branch. Any bent-knee posture
escapes it; convergence from this seed is well inside the walk workspace.
"""


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Rodrigues' rotation formula.

  Args:
    axis: Shape ``(..., 3)`` unit rotation axis.
    angle: Shape ``(...)`` rotation angle in radians.

  Returns:
    Shape ``(..., 3, 3)`` rotation matrices.
  """
  sin = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
  cos = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
  x, y, z = axis.unbind(-1)
  zero = torch.zeros_like(x)
  skew = torch.stack(
    (
      torch.stack((zero, -z, y), dim=-1),
      torch.stack((z, zero, -x), dim=-1),
      torch.stack((-y, x, zero), dim=-1),
    ),
    dim=-2,
  )
  eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(skew)
  return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def rotation_log(rotation: torch.Tensor) -> torch.Tensor:
  """Axis-angle (log map) of a rotation matrix.

  Args:
    rotation: Shape ``(..., 3, 3)``.

  Returns:
    Shape ``(..., 3)`` rotation vector, whose norm is the rotation angle.
  """
  trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
  angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
  vee = torch.stack(
    (
      rotation[..., 2, 1] - rotation[..., 1, 2],
      rotation[..., 0, 2] - rotation[..., 2, 0],
      rotation[..., 1, 0] - rotation[..., 0, 1],
    ),
    dim=-1,
  )
  sin = torch.sin(angle)
  # Near zero the 1/(2 sin) factor degenerates; the first-order term 0.5 * vee
  # is accurate there and is what the small-angle limit converges to anyway.
  small = sin.abs() < 1e-6
  scale = torch.where(small, torch.full_like(sin, 0.5), angle / (2.0 * sin))
  return scale.unsqueeze(-1) * vee


@dataclass
class LegChain:
  """Fixed geometry of one leg, read out of a compiled ``MjModel``.

  Attributes:
    body_pos: Shape ``(6, 3)`` body origin in its parent's frame.
    body_rot: Shape ``(6, 3, 3)`` body orientation in its parent's frame.
    joint_pos: Shape ``(6, 3)`` joint anchor in its own body's frame.
    joint_axis: Shape ``(6, 3)`` joint axis in its own body's frame.
    sole_pos: Shape ``(3,)`` sole origin in the foot body frame.
    sole_rot: Shape ``(3, 3)`` sole orientation in the foot body frame.
  """

  body_pos: torch.Tensor
  body_rot: torch.Tensor
  joint_pos: torch.Tensor
  joint_axis: torch.Tensor
  sole_pos: torch.Tensor
  sole_rot: torch.Tensor

  @classmethod
  def from_model(
    cls,
    model: mujoco.MjModel,
    left: bool,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = ENGINE_DTYPE,
  ) -> LegChain:
    """Extract a leg chain from a compiled model.

    Args:
      model: Compiled NUgus model.
      left: Whether to read the left leg.
      device: Device for the extracted tensors.
      dtype: Floating point precision.

    Returns:
      The leg's fixed geometry.
    """
    side = "left" if left else "right"
    body_pos, body_quat, joint_pos, joint_axis = [], [], [], []
    for suffix in LEG_JOINT_SUFFIXES:
      joint_name = f"{side}_{suffix}"
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      if joint_id < 0:
        raise ValueError(f"joint {joint_name!r} not found in model")
      body_id = model.jnt_bodyid[joint_id]
      body_pos.append(model.body_pos[body_id].copy())
      body_quat.append(model.body_quat[body_id].copy())
      joint_pos.append(model.jnt_pos[joint_id].copy())
      joint_axis.append(model.jnt_axis[joint_id].copy())

    def tensor(values) -> torch.Tensor:
      return torch.tensor(np.asarray(values), device=device, dtype=dtype)

    quats = tensor(body_quat)
    return cls(
      body_pos=tensor(body_pos),
      body_rot=matrix_from_quat(quats),
      joint_pos=tensor(joint_pos),
      joint_axis=tensor(joint_axis),
      sole_pos=tensor(NUGUS_SOLE_OFFSET),
      sole_rot=tensor(NUGUS_SOLE_ROTATION),
    )


def forward_kinematics(
  chain: LegChain, joint_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Sole pose and geometric Jacobian for a batch of joint configurations.

  Args:
    chain: Leg geometry.
    joint_pos: Shape ``(N, 6)`` joint angles, ordered as
      :data:`LEG_JOINT_SUFFIXES`.

  Returns:
    Tuple of the ``(N, 4, 4)`` sole pose in the torso body frame, the
    ``(N, 6, 3)`` joint axes in that frame, and the ``(N, 6, 3)`` joint anchor
    positions in that frame. The latter two build the Jacobian.
  """
  num_envs = joint_pos.shape[0]
  rotation = torch.eye(3, device=joint_pos.device, dtype=joint_pos.dtype).expand(
    num_envs, 3, 3
  )
  translation = torch.zeros(num_envs, 3, device=joint_pos.device, dtype=joint_pos.dtype)

  axes, anchors = [], []
  for index in range(len(LEG_JOINT_SUFFIXES)):
    # Into the child body's frame.
    translation = translation + torch.einsum(
      "nij,j->ni", rotation, chain.body_pos[index]
    )
    rotation = rotation @ chain.body_rot[index]

    # The joint anchor and axis, now expressed in the torso frame.
    anchor = translation + torch.einsum("nij,j->ni", rotation, chain.joint_pos[index])
    axis = torch.einsum("nij,j->ni", rotation, chain.joint_axis[index])
    anchors.append(anchor)
    axes.append(axis)

    # Rotate about the joint, which pivots the frame around its anchor.
    joint_rotation = _axis_angle_to_matrix(
      chain.joint_axis[index].expand(num_envs, 3), joint_pos[:, index]
    )
    translation = anchor + torch.einsum(
      "nij,nj->ni",
      rotation @ joint_rotation,
      -chain.joint_pos[index].expand(num_envs, 3),
    )
    rotation = rotation @ joint_rotation

  sole_translation = translation + torch.einsum("nij,j->ni", rotation, chain.sole_pos)
  sole_rotation = rotation @ chain.sole_rot
  return (
    make_transform(sole_translation, sole_rotation),
    torch.stack(axes, dim=1),
    torch.stack(anchors, dim=1),
  )


def solve_leg_ik(
  chain: LegChain,
  target: torch.Tensor,
  seed: torch.Tensor,
  iterations: int = 12,
  damping: float = 1e-3,
  tolerance: float = 1e-8,
) -> torch.Tensor:
  """Solve joint angles placing the sole at a target pose.

  Damped least-squares Gauss-Newton over :func:`forward_kinematics`. The damping
  keeps the step bounded through the knee-extension singularity, where the
  idealised solver instead saturates.

  Args:
    chain: Leg geometry.
    target: Shape ``(N, 4, 4)`` desired sole pose in the torso body frame.
    seed: Shape ``(N, 6)`` initial guess; the previous solution works well at
      control rates.
    iterations: Maximum Gauss-Newton iterations.
    damping: Least-squares damping factor.
    tolerance: Stop once the largest pose error falls below this.

  Returns:
    Shape ``(N, 6)`` joint angles.
  """
  joint_pos = seed.clone()
  eye = torch.eye(6, device=seed.device, dtype=seed.dtype)

  for _ in range(iterations):
    pose, axes, anchors = forward_kinematics(chain, joint_pos)

    position_error = target[:, :3, 3] - pose[:, :3, 3]
    rotation_error = rotation_log(target[:, :3, :3] @ pose[:, :3, :3].transpose(-1, -2))
    error = torch.cat((position_error, rotation_error), dim=-1)
    if float(error.abs().max()) < tolerance:
      break

    # Geometric Jacobian: a revolute joint contributes w x (p_ee - p_joint) to
    # the sole's linear velocity and w to its angular velocity.
    lever = pose[:, None, :3, 3] - anchors
    jacobian = torch.cat((torch.cross(axes, lever, dim=-1), axes), dim=-1).transpose(
      1, 2
    )

    jjt = jacobian @ jacobian.transpose(-1, -2) + (damping**2) * eye
    step = jacobian.transpose(-1, -2) @ torch.linalg.solve(jjt, error.unsqueeze(-1))
    joint_pos = joint_pos + step.squeeze(-1)

  return joint_pos


def target_from_engine_frame(htf: torch.Tensor) -> torch.Tensor:
  """Convert a foot pose from the walk engine's torso frame to the body frame.

  The engine's torso frame sits :data:`NUGUS_TORSO_FRAME_OFFSET` above the MJCF
  ``torso`` body origin, so the commanded pose shifts down by that offset.

  Args:
    htf: Shape ``(N, 4, 4)`` foot pose in the engine's torso frame.

  Returns:
    Shape ``(N, 4, 4)`` foot pose in the ``torso`` body frame.
  """
  offset = torch.tensor(NUGUS_TORSO_FRAME_OFFSET, device=htf.device, dtype=htf.dtype)
  shifted = htf.clone()
  shifted[:, :3, 3] = shifted[:, :3, 3] + offset
  return shifted
