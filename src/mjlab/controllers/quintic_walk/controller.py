"""The quintic walk engine wired up as a joint-position controller.

Combines the pieces the NUbots stack spreads across three modules:

- ``module/skill/Walk`` -- command slew limiting and the engine update.
- ``utility/skill/WalkGenerator`` -- the trajectories themselves.
- ``module/actuation/FootController`` -- the balance correction applied to the
  desired foot orientation, and the leg IK.

The result maps (velocity command, torso orientation, gyro) to twelve leg joint
position targets, which is exactly the interface a learned policy exposes, so
the two can be swapped in the same environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.controllers.quintic_walk.exact_kinematics import (
  DEFAULT_SEED,
  LegChain,
  solve_leg_ik,
  target_from_engine_frame,
)
from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_LEG,
  NUGUS_SOLE_OFFSET,
  NUGUS_SOLE_ROTATION,
  LegModel,
  calculate_leg_joints,
  invert_transform,
  mat_to_rpy_intrinsic,
  rpy_intrinsic_to_mat,
)
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_MAX_ACCELERATION,
  NUGUS_WALK_PARAMETERS,
  Phase,
  WalkGenerator,
  WalkParameters,
)

JOINT_NAMES: tuple[str, ...] = (
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
)
"""Column order of the joint targets returned by :meth:`QuinticWalkController.compute`."""

FOOT_DOWN_THRESHOLD = 0.01
"""Z-height threshold for foot contact, from ``SensorFilter.yaml``."""


@dataclass(frozen=True)
class BalanceGains:
  """Torso-orientation PID gains, from ``FootController.yaml``."""

  roll_p: float = 0.2
  pitch_p: float = 0.2
  roll_i: float = 0.0
  pitch_i: float = 0.0
  roll_d: float = 0.01
  pitch_d: float = 0.01
  max_i_error: float = 0.0
  max_roll_error: float = 0.6
  max_pitch_error: float = 0.6


NUGUS_BALANCE_GAINS = BalanceGains()
"""NUgus balance tuning. The integral term is disabled on the robot."""


def detect_planted_phase(
  left_foot_pose_t: torch.Tensor,
  right_foot_pose_t: torch.Tensor,
  threshold: float = FOOT_DOWN_THRESHOLD,
) -> torch.Tensor:
  """Classify the planted foot from relative foot height.

  Reproduces the ``Z_HEIGHT`` branch of ``SensorFilter::update_kinematics``:
  the right foot's height in the left foot's frame decides which feet are
  down, with a dead band around zero giving double support.

  Args:
    left_foot_pose_t: Shape ``(N, 4, 4)`` left foot pose in the torso frame.
    right_foot_pose_t: Shape ``(N, 4, 4)`` right foot pose in the torso frame.
    threshold: Height dead band, in metres.

  Returns:
    Shape ``(N,)`` :class:`Phase` per environment.
  """
  hlr = invert_transform(left_foot_pose_t) @ right_foot_pose_t
  height = hlr[:, 2, 3]
  right_down = height <= threshold
  left_down = height >= -threshold
  phase = torch.where(
    left_down & ~right_down,
    torch.full_like(height, int(Phase.LEFT)),
    torch.full_like(height, int(Phase.RIGHT)),
  )
  return torch.where(
    left_down & right_down, torch.full_like(height, int(Phase.DOUBLE)), phase
  ).long()


def sole_poses_in_torso(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  device: torch.device | str = "cpu",
  dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Measure both sole poses in the torso frame from simulator state.

  This is the input :func:`detect_planted_phase` wants. ``SensorFilter`` gets
  the same quantity by running forward kinematics on the measured servo
  positions and reading ``Htx[L_FOOT_BASE]``; reading the simulator's body poses
  is the same computation with the same inputs.

  The mjlab NUgus has no ``*_foot_base`` body, so the sole frame is
  reconstructed from the foot body with :data:`NUGUS_SOLE_OFFSET` and
  :data:`NUGUS_SOLE_ROTATION`. Those reproduce the ``left_foot_base`` fixed
  joint of NUbots' URDF -- ``xyz="0.038 0 0"``, ``rpy="0 -pi/2 0"`` -- exactly
  in rotation and to within a millimetre in translation.

  Args:
    model: Compiled model containing ``torso``, ``left_foot`` and
      ``right_foot`` bodies.
    data: Simulator state, already forwarded to the current ``qpos``.
    device: Device of the returned tensors.
    dtype: Dtype of the returned tensors.

  Returns:
    Left and right sole poses in the torso frame, each shape ``(1, 4, 4)``.
  """
  sole_rotation = np.array(NUGUS_SOLE_ROTATION)
  sole_offset = np.array(NUGUS_SOLE_OFFSET)
  torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
  torso_rotation = data.xmat[torso].reshape(3, 3)
  torso_position = data.xpos[torso]

  poses = []
  for side in ("left", "right"):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    body_rotation = data.xmat[body].reshape(3, 3)
    pose = np.eye(4)
    pose[:3, :3] = torso_rotation.T @ body_rotation @ sole_rotation
    pose[:3, 3] = torso_rotation.T @ (
      data.xpos[body] + body_rotation @ sole_offset - torso_position
    )
    poses.append(torch.tensor(pose, device=device, dtype=dtype).unsqueeze(0))
  return poses[0], poses[1]


class QuinticWalkController:
  """Batched quintic walk controller producing leg joint position targets."""

  joint_names = JOINT_NAMES

  def __init__(
    self,
    num_envs: int,
    device: torch.device | str = "cpu",
    walk_params: WalkParameters = NUGUS_WALK_PARAMETERS,
    balance_gains: BalanceGains = NUGUS_BALANCE_GAINS,
    leg_model: LegModel = NUGUS_LEG,
    use_balance_control: bool = True,
    max_acceleration: tuple[float, float, float] = NUGUS_MAX_ACCELERATION,
    dtype: torch.dtype = torch.float32,
    exact_ik_model: mujoco.MjModel | None = None,
    exact_ik_iterations: int = 8,
  ) -> None:
    """Build a controller.

    Args:
      exact_ik_model: When given, solve the legs against this model's real
        geometry instead of the engine's idealised leg. This is the second
        experimental condition, not the faithful one: the deployed robot runs
        the idealised solver, so leaving this ``None`` is what reproduces
        hardware. Supplying it answers "how would the engine do if its
        kinematic model were right?".
      exact_ik_iterations: Gauss-Newton iterations per control step. Warm
        started from the previous solution, so a handful suffices.
    """
    self.device = torch.device(device)
    self.dtype = dtype
    self.num_envs = num_envs
    self.leg_model = leg_model
    self.gains = balance_gains
    self.use_balance_control = use_balance_control
    self.exact_ik_iterations = exact_ik_iterations

    self._chains: dict[bool, LegChain] | None = None
    self._previous: dict[bool, torch.Tensor] | None = None
    if exact_ik_model is not None:
      self._chains = {
        left: LegChain.from_model(
          exact_ik_model, left=left, device=self.device, dtype=dtype
        )
        for left in (True, False)
      }
      seed = torch.tensor(DEFAULT_SEED, device=self.device, dtype=dtype)
      self._previous = {
        left: seed.expand(num_envs, 6).clone() for left in (True, False)
      }

    self.generator = WalkGenerator(
      num_envs, device=self.device, params=walk_params, dtype=dtype
    )
    self._max_acceleration = torch.tensor(
      max_acceleration, device=self.device, dtype=dtype
    )
    self._command = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
    self._integral_roll = torch.zeros(num_envs, device=self.device, dtype=dtype)
    self._integral_pitch = torch.zeros(num_envs, device=self.device, dtype=dtype)

  @property
  def velocity_command(self) -> torch.Tensor:
    """Shape ``(N, 3)`` slew-limited command actually given to the engine."""
    return self._command

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset the engine, the command filter and the balance integrators."""
    self.generator.reset(env_ids)
    index = slice(None) if env_ids is None else env_ids
    self._command[index] = 0.0
    self._integral_roll[index] = 0.0
    self._integral_pitch[index] = 0.0
    if self._previous is not None:
      seed = torch.tensor(DEFAULT_SEED, device=self.device, dtype=self.dtype)
      for left in (True, False):
        self._previous[left][index] = seed

  def compute(
    self,
    dt: float,
    velocity_command: torch.Tensor,
    torso_rotation_w: torch.Tensor | None = None,
    gyro_b: torch.Tensor | None = None,
    sensed_phase: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Step the engine and solve for leg joint targets.

    Args:
      dt: Control period in seconds.
      velocity_command: Shape ``(N, 3)`` requested (dx, dy, dtheta). Slew
        limited internally, as ``Walk.cpp`` does before calling the engine.
      torso_rotation_w: Shape ``(N, 3, 3)`` torso orientation in the world
        frame. Required when balance control is enabled.
      gyro_b: Shape ``(N, 3)`` body-frame angular velocity. Required when
        balance control is enabled.
      sensed_phase: Shape ``(N,)`` planted foot phase; see
        :func:`detect_planted_phase`. Only read when the engine is configured
        with ``only_switch_when_planted``.

    Returns:
      Shape ``(N, 12)`` joint position targets ordered as :data:`JOINT_NAMES`.
    """
    delta = self._max_acceleration * min(dt, 1.0)
    self._command = self._command + (velocity_command - self._command).clamp(
      -delta, delta
    )

    self.generator.update(dt, self._command, sensed_phase)

    htl = self.generator.foot_pose(left=True)
    htr = self.generator.foot_pose(left=False)

    if self.use_balance_control:
      if torso_rotation_w is None or gyro_b is None:
        raise ValueError(
          "balance control needs torso_rotation_w and gyro_b; pass "
          "use_balance_control=False to run the engine open loop"
        )
      htl = self._apply_balance(htl, torso_rotation_w, gyro_b, dt, accumulate=True)
      htr = self._apply_balance(htr, torso_rotation_w, gyro_b, dt, accumulate=False)

    return torch.cat(
      (self._solve_leg(htl, left=True), self._solve_leg(htr, left=False)), dim=-1
    )

  @property
  def uses_exact_ik(self) -> bool:
    """Whether the legs are solved against the real geometry."""
    return self._chains is not None

  def _solve_leg(self, htf: torch.Tensor, left: bool) -> torch.Tensor:
    """Solve one leg with whichever IK this controller was built for."""
    if self._chains is None or self._previous is None:
      return calculate_leg_joints(htf, left=left, model=self.leg_model)
    solution = solve_leg_ik(
      self._chains[left],
      target_from_engine_frame(htf),
      self._previous[left],
      iterations=self.exact_ik_iterations,
    )
    self._previous[left] = solution
    return solution

  def _apply_balance(
    self,
    htf: torch.Tensor,
    torso_rotation_w: torch.Tensor,
    gyro_b: torch.Tensor,
    dt: float,
    accumulate: bool,
  ) -> torch.Tensor:
    """Rotate the desired foot pose to hold the torso level.

    Mirrors ``FootController::control_foot``: proportional on the error between
    the desired and measured torso roll/pitch, derivative on the gyro, and the
    yaw left untouched.

    The integrators are shared between the two feet, so only the first call per
    control step accumulates -- matching the single ``last_update_time`` the C++
    module keeps for both.
    """
    g = self.gains
    measured = mat_to_rpy_intrinsic(torso_rotation_w)
    desired = mat_to_rpy_intrinsic(invert_transform(htf)[:, :3, :3])

    roll_error = desired[:, 0] - measured[:, 0]
    pitch_error = desired[:, 1] - measured[:, 1]

    desired_roll = desired[:, 0] + g.roll_p * roll_error.clamp(
      -g.max_roll_error, g.max_roll_error
    )
    desired_pitch = desired[:, 1] + g.pitch_p * pitch_error.clamp(
      -g.max_pitch_error, g.max_pitch_error
    )

    if accumulate:
      self._integral_roll = (self._integral_roll + roll_error * dt).clamp(
        -g.max_i_error, g.max_i_error
      )
      self._integral_pitch = (self._integral_pitch + pitch_error * dt).clamp(
        -g.max_i_error, g.max_i_error
      )
    desired_roll = desired_roll + g.roll_i * self._integral_roll
    desired_pitch = desired_pitch + g.pitch_i * self._integral_pitch

    desired_roll = desired_roll - g.roll_d * gyro_b[:, 0]
    desired_pitch = desired_pitch - g.pitch_d * gyro_b[:, 1]

    zeros = torch.zeros_like(desired_roll)
    yaw_only = rpy_intrinsic_to_mat(torch.stack((zeros, zeros, desired[:, 2]), dim=-1))
    roll_pitch = rpy_intrinsic_to_mat(
      torch.stack((desired_roll, desired_pitch, zeros), dim=-1)
    )
    rotation_ft = yaw_only @ roll_pitch

    corrected = htf.clone()
    corrected[:, :3, :3] = rotation_ft.transpose(-1, -2)
    return corrected
