"""Batched port of the NUbots quintic walk trajectory generator.

Ported from ``shared/utility/skill/WalkGenerator.hpp``. The engine is a phase
clock driving two piecewise-quintic trajectories -- one for the swing foot, one
for the torso -- both expressed in the planted foot's frame, plus a four-state
machine controlling when steps start, switch and stop.

Everything here carries a leading ``num_envs`` dimension. The C++ ``switch`` on
engine state becomes a set of boolean masks: each branch is evaluated for all
environments and selected per environment, which costs a few extra elementwise
ops and keeps the whole update branch-free on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import torch

from mjlab.controllers.quintic_walk.kinematics import (
  invert_transform,
  make_transform,
  mat_to_rpy_intrinsic,
  rpy_intrinsic_to_mat,
)
from mjlab.controllers.quintic_walk.spline import (
  Trajectory,
  build_trajectory,
  make_waypoint,
)


class EngineState(IntEnum):
  """Walk engine states, matching ``WalkState::State``."""

  UNKNOWN = 0
  STARTING = 1
  WALKING = 2
  STOPPING = 3
  STOPPED = 4


class Phase(IntEnum):
  """Planted foot phase, matching ``WalkState::Phase`` in the proto."""

  DOUBLE = 0
  LEFT = 1
  RIGHT = 2


@dataclass(frozen=True)
class WalkParameters:
  """Walk engine tuning, mirroring the C++ ``WalkParameters`` struct."""

  step_limits: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Maximum step displacement in x, y (metres) and theta (radians)."""
  step_period: float = 0.0
  """Duration of one complete step, in seconds."""
  step_height: float = 0.0
  """Peak swing foot height, in metres."""
  step_width: float = 0.0
  """Lateral distance between the feet, in metres."""
  step_apex_ratio: float = 0.0
  """Fraction of the step period at which the swing foot peaks."""
  torso_height: float = 0.0
  """Torso height above the planted foot, in metres."""
  torso_pitch: float = 0.0
  """Constant torso pitch, in radians."""
  torso_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Constant torso position offset, in metres."""
  torso_sway_ratio: float = 0.0
  """Fraction of the step period at which the torso sway peaks."""
  torso_sway_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Torso offset from the planted foot at peak sway, in metres."""
  torso_start_sway_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Torso offset from the planted foot at the end of the starting step."""
  torso_final_position_ratio: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Where to place the torso at step end, as a ratio of the next step."""
  only_switch_when_planted: bool = False
  """Whether to defer the foot switch until the sensed phase agrees.

  ``False`` reproduces the C++ struct's own default. See
  :data:`NUGUS_WALK_PARAMETERS` for why the NUgus tuning leaves it there even
  though ``Walk.yaml`` asks for ``true``.
  """


NUGUS_WALK_PARAMETERS = WalkParameters(
  step_limits=(0.5, 0.2, 0.4),
  step_period=0.32,
  step_height=0.085,
  step_width=0.27,
  step_apex_ratio=0.4,
  torso_height=0.44,
  torso_pitch=12.0 * math.pi / 180.0,
  torso_position_offset=(0.01, 0.0, 0.0),
  torso_sway_ratio=0.5,
  torso_sway_offset=(0.0, 0.1, 0.0),
  torso_start_sway_offset=(0.0, 0.1, 0.0),
  torso_final_position_ratio=(0.5, 0.5, 1.0),
)
"""NUgus walk tuning, from ``module/skill/Walk/data/config/Walk.yaml``.

Every value here is the config file's, with one deliberate exception:
``only_switch_when_planted``, which ``Walk.yaml`` sets to ``true`` and which is
left ``False`` here because the robot does not run it. ``Walk.cpp`` reads the
config into ``cfg.walk_generator_parameters``, calls
``walk_generator.set_parameters(...)`` -- which copies by value -- and only
*then* assigns this one field, so the generator keeps the struct default of
``false``. The configuration reaction runs once per process (the
``Configuration`` DSL suppresses the extra per-hierarchy triggers at install
time), so nothing later repairs it, and the deployed engine switches feet on the
clock alone.

Enabling it is a supported experiment rather than a fidelity fix, and playback
always supplies the sensed phase so that it works. Measured on the eval plant,
20 s per command: walking forward it defers each switch by a single 10 ms
control tick and changes nothing. Turning in place at 0.5 rad/s it stalls the
gait for up to 160 ms at a time and topples the robot after four seconds, where
the clock-driven engine stays up. The reason is the ``Z_HEIGHT`` foot-down
detector it consumes: it reads the swing sole's height in the *stance* sole's
frame, so a stance foot rolled onto its edge tilts the reference plane and the
swing foot never registers as landed. NUbots' own ``FSR`` method does not have
that failure mode.
"""

NUGUS_MAX_ACCELERATION: tuple[float, float, float] = (0.2, 0.2, 0.5)
"""Per-second command slew limit applied by ``Walk.cpp`` before the engine."""


@dataclass
class _Params:
  """Walk parameters broadcast to per-environment tensors."""

  step_limits: torch.Tensor = field(init=False)
  step_period: torch.Tensor = field(init=False)

  def __init__(self, params: WalkParameters, num_envs: int, device, dtype):
    def scalar(value: float) -> torch.Tensor:
      return torch.full((num_envs,), value, device=device, dtype=dtype)

    def vector(value: tuple[float, float, float]) -> torch.Tensor:
      return torch.tensor(value, device=device, dtype=dtype).expand(num_envs, 3)

    self.step_limits = vector(params.step_limits)
    self.step_period = scalar(params.step_period)
    self.step_height = scalar(params.step_height)
    self.step_width = scalar(params.step_width)
    self.step_apex_ratio = scalar(params.step_apex_ratio)
    self.torso_height = scalar(params.torso_height)
    self.torso_pitch = scalar(params.torso_pitch)
    self.torso_position_offset = vector(params.torso_position_offset)
    self.torso_sway_ratio = scalar(params.torso_sway_ratio)
    self.torso_sway_offset = vector(params.torso_sway_offset)
    self.torso_start_sway_offset = vector(params.torso_start_sway_offset)
    self.torso_final_position_ratio = vector(params.torso_final_position_ratio)


class WalkGenerator:
  """A batch of quintic walk engines, one per environment."""

  def __init__(
    self,
    num_envs: int,
    device: torch.device | str = "cpu",
    params: WalkParameters = NUGUS_WALK_PARAMETERS,
    dtype: torch.dtype = torch.float32,
  ) -> None:
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.dtype = dtype
    self.cfg = params
    self._p = _Params(params, num_envs, self.device, dtype)

    self._state = torch.full(
      (num_envs,), int(EngineState.STOPPED), device=self.device, dtype=torch.long
    )
    self._phase = torch.full(
      (num_envs,), int(Phase.LEFT), device=self.device, dtype=torch.long
    )
    self._t = torch.zeros(num_envs, device=self.device, dtype=dtype)
    eye = torch.eye(4, device=self.device, dtype=dtype)
    self._hps_start = eye.expand(num_envs, 4, 4).clone()
    self._hpt_start = eye.expand(num_envs, 4, 4).clone()

    zero = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
    waypoint = make_waypoint(zero, zero)
    period = self._p.step_period
    self._swing = build_trajectory(waypoint, waypoint, waypoint, period * 0.5, period)
    self._torso = build_trajectory(waypoint, waypoint, waypoint, period * 0.5, period)

    self.reset()

  # Properties.

  @property
  def state(self) -> torch.Tensor:
    """Shape ``(N,)`` current :class:`EngineState` per environment."""
    return self._state

  @property
  def phase(self) -> torch.Tensor:
    """Shape ``(N,)`` planted foot :class:`Phase` per environment."""
    return self._phase

  @property
  def time(self) -> torch.Tensor:
    """Shape ``(N,)`` time within the step cycle."""
    return self._t

  @property
  def step_period(self) -> float:
    return self.cfg.step_period

  # Trajectory queries.

  def swing_foot_pose(self, t: torch.Tensor | None = None) -> torch.Tensor:
    """Swing foot pose in the planted foot frame. Shape ``(N, 4, 4)``."""
    translation, rpy = self._swing.evaluate(self._t if t is None else t)
    return make_transform(translation, rpy_intrinsic_to_mat(rpy))

  def torso_pose(self, t: torch.Tensor | None = None) -> torch.Tensor:
    """Torso pose in the planted foot frame. Shape ``(N, 4, 4)``."""
    translation, rpy = self._torso.evaluate(self._t if t is None else t)
    return make_transform(translation, rpy_intrinsic_to_mat(rpy))

  def foot_pose(self, left: bool, t: torch.Tensor | None = None) -> torch.Tensor:
    """Desired pose of one foot in the torso frame. Shape ``(N, 4, 4)``.

    The planted foot sits at the inverse of the torso pose; the swing foot is
    that composed with the swing trajectory.
    """
    torso_inv = invert_transform(self.torso_pose(t))
    swinging = torso_inv @ self.swing_foot_pose(t)
    is_planted = (self._phase == int(Phase.LEFT)) == left
    return torch.where(is_planted.view(-1, 1, 1), torso_inv, swinging)

  # State updates.

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset to a standing stance with the step clock parked at its end."""
    width = self._foot_width_offset()
    zeros = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)

    hps = make_transform(
      torch.stack((zeros, width, zeros), dim=-1),
      torch.eye(4, device=self.device, dtype=self.dtype)[:3, :3].expand(
        self.num_envs, 3, 3
      ),
    )
    torso_rpy = torch.stack((zeros, self._p.torso_pitch, zeros), dim=-1)
    hpt = make_transform(
      self._p.torso_position_offset
      + torch.stack((zeros, width * 0.5, self._p.torso_height), dim=-1),
      rpy_intrinsic_to_mat(torso_rpy),
    )

    if env_ids is None:
      self._hps_start = hps
      self._hpt_start = hpt
      self._t = self._p.step_period.clone()
      self._state = torch.full_like(self._state, int(EngineState.STOPPED))
    else:
      self._hps_start[env_ids] = hps[env_ids]
      self._hpt_start[env_ids] = hpt[env_ids]
      self._t[env_ids] = self._p.step_period[env_ids]
      self._state[env_ids] = int(EngineState.STOPPED)

    # Standing still still needs trajectories, generated at zero velocity.
    self._generate_walking_trajectories(
      torch.zeros(self.num_envs, 3, device=self.device, dtype=self.dtype)
    )

  def update(
    self,
    dt: float,
    velocity_target: torch.Tensor,
    sensed_phase: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Advance the engine by ``dt``.

    Args:
      dt: Control period in seconds. Must be positive and no larger than the
        step period; the C++ warns and skips the update otherwise.
      velocity_target: Shape ``(N, 3)`` requested (dx, dy, dtheta).
      sensed_phase: Shape ``(N,)`` planted foot phase from foot contact. Only
        read when :attr:`WalkParameters.only_switch_when_planted` is set.

    Returns:
      Shape ``(N,)`` engine state after the update.
    """
    if dt <= 0.0 or dt > self.cfg.step_period:
      return self._state

    period = self._p.step_period
    is_zero_command = (velocity_target == 0).all(dim=-1)

    # State transitions, in the same order as the C++ if-chain, evaluated
    # against the pre-update clock.
    stopping = is_zero_command & (self._t < period)
    stopped = (
      (self._t >= period)
      & (is_zero_command | (self._state == int(EngineState.STOPPING)))
      & ~stopping
    )
    starting = (
      ~is_zero_command
      & (self._state == int(EngineState.STOPPED))
      & ~stopping
      & ~stopped
    )
    self._state = torch.where(
      stopping, torch.full_like(self._state, int(EngineState.STOPPING)), self._state
    )
    self._state = torch.where(
      stopped, torch.full_like(self._state, int(EngineState.STOPPED)), self._state
    )
    self._state = torch.where(
      starting, torch.full_like(self._state, int(EngineState.STARTING)), self._state
    )
    self._t = torch.where(starting, torch.zeros_like(self._t), self._t)

    is_starting = self._state == int(EngineState.STARTING)
    is_walking = self._state == int(EngineState.WALKING)
    is_stopping = self._state == int(EngineState.STOPPING)
    is_stopped = self._state == int(EngineState.STOPPED)

    # Advance the clock everywhere except the stopped state.
    advanced = (self._t + dt).clamp_max(period)
    self._t = torch.where(is_stopped, self._t, advanced)

    # Walking: switch the planted foot at the end of the step.
    if self.cfg.only_switch_when_planted:
      if sensed_phase is None:
        raise ValueError(
          "only_switch_when_planted is set but no sensed_phase was provided"
        )
      can_switch = sensed_phase != self._phase
    else:
      can_switch = torch.ones_like(is_walking)
    self._switch_planted_foot(is_walking & (self._t >= period) & can_switch)

    # Starting: hold the swing foot still and sway onto the planted foot.
    self._generate_starting_trajectories(is_starting)

    # Walking and stopping share the walking trajectories.
    self._generate_walking_trajectories(
      velocity_target, select=is_walking | is_stopping
    )

    # Stopped environments re-park at the standing stance.
    if bool(is_stopped.any()):
      self.reset(is_stopped.nonzero(as_tuple=False).squeeze(-1))

    # Starting completes once the clock runs out.
    finished_starting = is_starting & (self._t >= period)
    self._state = torch.where(
      finished_starting,
      torch.full_like(self._state, int(EngineState.WALKING)),
      self._state,
    )
    self._t = torch.where(finished_starting, torch.zeros_like(self._t), self._t)

    return self._state

  # Internals.

  def _foot_width_offset(self) -> torch.Tensor:
    """Lateral offset of the swing foot in the planted foot frame."""
    return torch.where(
      self._phase == int(Phase.LEFT), -self._p.step_width, self._p.step_width
    )

  def _switch_planted_foot(self, select: torch.Tensor) -> None:
    """Re-anchor the trajectories onto the other foot."""
    if not bool(select.any()):
      return
    period = self._p.step_period
    hps = invert_transform(self.swing_foot_pose(period))
    hpt = hps @ self.torso_pose(period)

    mask = select.view(-1, 1, 1)
    self._hps_start = torch.where(mask, hps, self._hps_start)
    self._hpt_start = torch.where(mask, hpt, self._hpt_start)
    flipped = torch.where(
      self._phase == int(Phase.LEFT),
      torch.full_like(self._phase, int(Phase.RIGHT)),
      torch.full_like(self._phase, int(Phase.LEFT)),
    )
    self._phase = torch.where(select, flipped, self._phase)
    self._t = torch.where(select, torch.zeros_like(self._t), self._t)

  def _step_from_velocity(self, velocity_target: torch.Tensor) -> torch.Tensor:
    """Next step placement in the planted foot frame, clamped to the limits."""
    period = self._p.step_period.unsqueeze(-1)
    return (velocity_target * period).clamp(-self._p.step_limits, self._p.step_limits)

  def _generate_walking_trajectories(
    self, velocity_target: torch.Tensor, select: torch.Tensor | None = None
  ) -> None:
    step = self._step_from_velocity(velocity_target)
    p = self._p
    zeros = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
    width = self._foot_width_offset()
    period = p.step_period

    # Torso: start where it is, sway over the planted foot, finish part-way
    # towards the next step.
    sway_y = torch.where(
      self._phase == int(Phase.LEFT),
      -p.torso_sway_offset[:, 1],
      p.torso_sway_offset[:, 1],
    )
    torso_first = make_waypoint(
      self._hpt_start[:, :3, 3], mat_to_rpy_intrinsic(self._hpt_start[:, :3, :3])
    )
    torso_middle = make_waypoint(
      torch.stack(
        (
          p.torso_sway_offset[:, 0],
          sway_y,
          p.torso_height + p.torso_sway_offset[:, 2],
        ),
        dim=-1,
      )
      + p.torso_position_offset,
      torch.stack(
        (zeros, p.torso_pitch, velocity_target[:, 2] * p.torso_sway_ratio * period),
        dim=-1,
      ),
      linear_velocity=torch.stack(
        (velocity_target[:, 0], velocity_target[:, 1], zeros), dim=-1
      ),
      angular_velocity=torch.stack((zeros, zeros, velocity_target[:, 2]), dim=-1),
    )
    torso_last = make_waypoint(
      torch.stack(
        (
          p.torso_final_position_ratio[:, 0] * step[:, 0],
          width * 0.5 + p.torso_final_position_ratio[:, 1] * step[:, 1],
          p.torso_height,
        ),
        dim=-1,
      )
      + p.torso_position_offset,
      torch.stack(
        (zeros, p.torso_pitch, p.torso_final_position_ratio[:, 2] * step[:, 2]), dim=-1
      ),
    )
    torso = build_trajectory(
      torso_first, torso_middle, torso_last, p.torso_sway_ratio * period, period
    )

    # Swing foot: lift to the apex, then plant at the next step placement.
    swing_first = make_waypoint(
      self._hps_start[:, :3, 3], mat_to_rpy_intrinsic(self._hps_start[:, :3, :3])
    )
    swing_middle = make_waypoint(
      torch.stack((zeros, width, p.step_height), dim=-1),
      torch.stack(
        (zeros, zeros, velocity_target[:, 2] * p.step_apex_ratio * period), dim=-1
      ),
      linear_velocity=torch.stack(
        (velocity_target[:, 0], velocity_target[:, 1], zeros), dim=-1
      ),
      angular_velocity=torch.stack((zeros, zeros, velocity_target[:, 2]), dim=-1),
    )
    swing_last = make_waypoint(
      torch.stack((step[:, 0], width + step[:, 1], zeros), dim=-1),
      torch.stack((zeros, zeros, step[:, 2]), dim=-1),
    )
    swing = build_trajectory(
      swing_first, swing_middle, swing_last, p.step_apex_ratio * period, period
    )

    self._select_trajectories(torso, swing, select)

  def _generate_starting_trajectories(self, select: torch.Tensor) -> None:
    if not bool(select.any()):
      return
    p = self._p
    zeros = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
    period = p.step_period

    # Swing foot holds still for the whole starting step.
    swing_at_start = make_waypoint(
      self._hps_start[:, :3, 3], mat_to_rpy_intrinsic(self._hps_start[:, :3, :3])
    )
    swing = build_trajectory(
      swing_at_start, swing_at_start, swing_at_start, period, period
    )

    sway_y = torch.where(
      self._phase == int(Phase.LEFT),
      -p.torso_start_sway_offset[:, 1],
      p.torso_start_sway_offset[:, 1],
    )
    torso_first = make_waypoint(
      self._hpt_start[:, :3, 3], mat_to_rpy_intrinsic(self._hpt_start[:, :3, :3])
    )
    torso_last = make_waypoint(
      torch.stack(
        (
          p.torso_start_sway_offset[:, 0],
          sway_y,
          p.torso_height + p.torso_start_sway_offset[:, 2],
        ),
        dim=-1,
      )
      + p.torso_position_offset,
      torch.stack((zeros, p.torso_pitch, zeros), dim=-1),
    )
    torso = build_trajectory(torso_first, torso_last, torso_last, period, period)

    self._select_trajectories(torso, swing, select)

  def _select_trajectories(
    self, torso: Trajectory, swing: Trajectory, select: torch.Tensor | None
  ) -> None:
    """Adopt new trajectories for the selected environments."""
    if select is None:
      self._torso = torso
      self._swing = swing
      return
    if not bool(select.any()):
      return
    coeff_mask = select.view(-1, 1, 1, 1)
    self._torso = Trajectory(
      coeffs=torch.where(coeff_mask, torso.coeffs, self._torso.coeffs),
      t_mid=torch.where(select, torso.t_mid, self._torso.t_mid),
    )
    self._swing = Trajectory(
      coeffs=torch.where(coeff_mask, swing.coeffs, self._swing.coeffs),
      t_mid=torch.where(select, swing.t_mid, self._swing.t_mid),
    )
