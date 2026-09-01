"""Push recovery: a measured shove, and what the controller does about it.

The command sweeps in :mod:`mjlab.evaluation.harness` ask how fast a controller
can walk and a profile run asks how well it follows a command that moves.
Neither disturbs the robot. This module asks the third question -- what happens
when something shoves it -- and asks it the same way of every controller, so
the answer is a property of the controller rather than of the push.

A *trial* is one shove. The robot walks under a fixed command until the gait is
established, a constant force is applied to the torso for
:attr:`PushCfg.duration` seconds, and the run continues for
:attr:`PushCfg.recovery` seconds while the outcome is measured. A *battery* is
a grid of trials over three variables:

``magnitude``
  How hard. Parameterised as the velocity change the impulse would produce on a
  free body of the robot's mass -- :attr:`PushCfg.delta_v`, in m/s -- because
  that is comparable across plants that do not weigh the same. The force
  actually applied is ``mass * delta_v / duration`` and is reported alongside.
``direction``
  Which way, as a heading in the robot's own yaw frame at the instant the push
  lands: 0 shoves it forwards, 90 degrees shoves it to its left. Latched at
  onset and held in the world frame for the duration, which is what a shove is.
``phase``
  When, within a gait cycle. A push during single support is a different event
  from the same push during double support, and a controller has no say in
  which one it gets. The onsets are spread evenly across
  :attr:`PushCfg.phase_window`, so every reported number is an average over
  gait phase rather than a measurement at one arbitrary point in the stride.

The battery is run one magnitude at a time -- :func:`push_battery` returns one
:class:`PushPlan` per magnitude -- so the batch size is set by the direction,
phase and replica counts alone and does not grow when the magnitude axis is
refined. See :func:`run_push_battery`.

Both the actuation and the measurement read the same :class:`PushPlan`, so the
push that :class:`PushDriver` applies and the push that :class:`PushMetrics`
dates its window from cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Protocol, Sequence

import torch

from mjlab.entity import Entity
from mjlab.evaluation.metrics import (
  FALL_UPRIGHT_THRESHOLD,
  WALK_QUALITY_METRICS,
  EvalState,
  PerEnvMetrics,
  WalkMetrics,
  summarise,
  upright_from_quat,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

PUSH_BODY = "torso"
"""Body the force is applied to.

MuJoCo applies ``xfrc_applied`` at the body's centre of mass, so this is a
shove through the torso's CoM: no contrived moment arm, and the same geometry
on every plant. Where on the torso a real shove lands would change the tipping
moment, which is a study of its own and not what this measures.
"""

PUSH_DURATION_S = 0.2
"""Seconds the force is held.

Short enough to be an impulse against a 0.64 s gait cycle and long enough to
span several control steps of either controller, so the outcome is not decided
by which single step the force happened to land on.
"""

SETTLE_S = 8.0
"""Seconds of undisturbed walking before the earliest push.

The same figure the sweeps discard as warm-up: long enough that the run-up is
over and the controller is at the speed it was asked for.
"""

PHASE_WINDOW_S = 0.64
"""Seconds the push onsets are spread over, one full gait cycle of the engine.

Two of the 0.32 s step periods the NUgus walk tuning runs -- see
``NUGUS_WALK_PARAMETERS``. A learned policy sets its own cadence, so this is not
its cycle exactly; with a dozen onsets spread across it the sampled gait phases
still cover a cycle of any period near this one, which is all the average needs.
"""

RECOVERY_S = 4.0
"""Seconds after the push that the outcome is measured over.

A robot still upright at the end of it withstood the push. Long enough to
contain the stumble and several strides of walking after it; not so long that
an unrelated fall lands inside the window.
"""

RECOVERY_TOLERANCE = 0.1
"""Planar velocity error, in m/s, that counts as back on command."""

RECOVERY_HOLD_S = 0.5
"""Seconds the error has to stay inside the tolerance to count as recovered.

Without it a trajectory that sweeps through the band on its way past would be
recorded as a recovery.
"""

SMOOTH_S = 0.64
"""Window of the moving average the recovery test is applied to, in seconds.

One full gait cycle, and the length matters. A raw base velocity swings by more
than the command does within a stride -- the torso sways sideways and
counter-rotates every step -- so an untouched robot never sits inside a 0.1 m/s
band and no run would ever be recorded as recovered. Cancelling the sway needs
a whole cycle rather than a step: walking forward at 0.2 m/s on the evaluation
plant, the engine's planar velocity error averages 0.23 m/s raw, 0.14 m/s over
a 0.30 s window and 0.03 m/s over this one, peaking at 0.05.

It also delays the measurement by up to a window, which is a constant offset on
every controller's recovery time rather than a difference between them.
"""


@dataclass(frozen=True)
class PushCfg:
  """The battery: what to push with, from where, and when."""

  vx: float = 0.2
  """Forward velocity command held for the whole run, in m/s."""
  vy: float = 0.0
  """Lateral velocity command, in m/s."""
  wz: float = 0.0
  """Yaw rate command, in rad/s. The recovery test reads the planar velocity
  error only, so a turning battery measures recovery of the linear axes."""

  delta_v: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
  )
  """Push magnitudes, as the velocity change each impulse would produce on a
  free body of the robot's mass, in m/s. One run per value.

  Stepped by 0.1 up to 1.0 and coarser above it. On the evaluation plant the
  walk engine's envelope falls around 0.3 to 0.6 and a trained policy's around
  0.5 to 1.0, so the fine half is where the crossing is; the tail is there so
  the better controller's envelope has an outside rather than running off the
  end of the battery."""
  directions: int = 12
  """Push headings, evenly spaced over the full circle. 0 shoves the robot
  forwards, 90 degrees shoves it to its left."""
  phases: int = 12
  """Push onsets, evenly spaced across :attr:`phase_window`."""
  replicas: int = 4
  """Environments per (direction, phase) pair.

  Not redundant even though the plant is deterministic and the randomisation is
  off: two identical robots in one batch are separated by reduction ordering
  within a few steps, and a gait amplifies that, so a trial near the edge of
  the envelope is effectively a coin flip. Replicas sample it."""

  duration: float = PUSH_DURATION_S
  """Seconds the force is held; see :data:`PUSH_DURATION_S`."""
  settle: float = SETTLE_S
  """Seconds of undisturbed walking before the earliest push."""
  phase_window: float = PHASE_WINDOW_S
  """Seconds the onsets are spread over; see :data:`PHASE_WINDOW_S`."""
  recovery: float = RECOVERY_S
  """Seconds after the push the outcome is measured over."""

  tolerance: float = RECOVERY_TOLERANCE
  """Planar velocity error that counts as back on command, in m/s."""
  hold: float = RECOVERY_HOLD_S
  """Seconds inside the tolerance that count as a recovery."""
  smooth: float = SMOOTH_S
  """Moving average the recovery test reads, in seconds."""

  def __post_init__(self) -> None:
    if not self.delta_v:
      raise ValueError("push battery has no magnitudes")
    if min(self.delta_v) <= 0.0:
      raise ValueError("push magnitudes must be positive")
    for name in ("directions", "phases", "replicas"):
      if getattr(self, name) < 1:
        raise ValueError(f"push {name} must be at least 1")
    for name in ("duration", "recovery", "hold", "smooth", "tolerance"):
      if getattr(self, name) <= 0.0:
        raise ValueError(f"push {name} must be positive")
    if self.settle < 0.0 or self.phase_window < 0.0:
      raise ValueError("push settle and phase_window must not be negative")

  @property
  def trials_per_pass(self) -> int:
    """Environments one magnitude needs, i.e. the batch size."""
    return self.directions * self.phases * self.replicas

  @property
  def num_trials(self) -> int:
    return self.trials_per_pass * len(self.delta_v)

  @property
  def trials_per_cell(self) -> int:
    """Trials behind one (direction, magnitude) survival fraction."""
    return self.phases * self.replicas

  @property
  def headings(self) -> tuple[float, ...]:
    """Push headings in radians, in batch order."""
    step = 2.0 * math.pi / self.directions
    return tuple(index * step for index in range(self.directions))

  @property
  def command(self) -> tuple[float, float, float]:
    return (self.vx, self.vy, self.wz)


@dataclass(frozen=True)
class PushPlan:
  """One pass of a battery: one trial per environment, all at one magnitude.

  Every field but :attr:`dt` and the step counts is shape ``(N,)`` or
  ``(N, 3)``, indexed by environment.

  Attributes:
    command: Shape ``(N, 3)`` velocity command held for the whole run.
    delta_v: Push magnitude, as the free-body velocity change in m/s.
    impulse: The same magnitude as an impulse, in N s.
    force: Force applied over :attr:`hold_steps`, in N.
    heading: Push direction in the robot's yaw frame at onset, in radians.
    push_step: Control step the force switches on.
    hold_steps: Steps the force is held.
    recovery_steps: Steps after the onset the outcome is measured over.
    settle_steps: Steps before the earliest onset.
    num_steps: Steps in the whole pass.
    dt: Control period, in seconds.
    tolerance: Planar velocity error that counts as back on command, in m/s.
    band_steps: Steps inside the tolerance that count as a recovery.
    smooth_steps: Steps the recovery test's moving average covers.
  """

  command: torch.Tensor
  delta_v: torch.Tensor
  impulse: torch.Tensor
  force: torch.Tensor
  heading: torch.Tensor
  push_step: torch.Tensor
  hold_steps: int
  recovery_steps: int
  settle_steps: int
  num_steps: int
  dt: float
  tolerance: float
  band_steps: int
  smooth_steps: int

  @property
  def num_envs(self) -> int:
    return int(self.command.shape[0])

  @property
  def push_time(self) -> torch.Tensor:
    """Seconds from the start of the run to each environment's onset."""
    return self.push_step.float() * self.dt


def push_plan(
  cfg: PushCfg,
  delta_v: float,
  mass: float,
  dt: float,
  device: torch.device | str = "cpu",
) -> PushPlan:
  """Lay one magnitude's trials out over a batch.

  The batch is ordered direction-major, then phase, then replica, so a reader
  of ``per_env.csv`` can find a cell without consulting this function -- though
  every trial also carries its own magnitude, heading and onset as columns, so
  nothing downstream has to reconstruct the layout.

  Args:
    cfg: The battery.
    delta_v: This pass's magnitude, in m/s of free-body velocity change.
    mass: Total mass of the robot, in kg.
    dt: Control period, in seconds.
    device: Where the plan's tensors live.
  """
  if mass <= 0.0:
    raise ValueError(f"robot mass must be positive, got {mass}")

  hold_steps = max(1, round(cfg.duration / dt))
  settle_steps = max(1, round(cfg.settle / dt))
  recovery_steps = max(1, round(cfg.recovery / dt))
  window_steps = max(1, round(cfg.phase_window / dt))

  # Onsets evenly spaced across the phase window. Distinct steps where the
  # window allows it; where it does not -- a coarse control rate against a
  # short window -- several phases land on one step and the duplicates become
  # replicas, which is the honest outcome rather than an error.
  offsets = [round(index * window_steps / cfg.phases) for index in range(cfg.phases)]

  heading = torch.tensor(cfg.headings, device=device, dtype=torch.float32)
  heading = heading.repeat_interleave(cfg.phases * cfg.replicas)
  offset = torch.tensor(offsets, device=device, dtype=torch.long)
  offset = offset.repeat_interleave(cfg.replicas).repeat(cfg.directions)

  num_envs = cfg.trials_per_pass
  impulse = mass * delta_v
  duration = hold_steps * dt
  return PushPlan(
    command=torch.tensor([cfg.command], device=device, dtype=torch.float32).repeat(
      num_envs, 1
    ),
    delta_v=torch.full((num_envs,), delta_v, device=device),
    impulse=torch.full((num_envs,), impulse, device=device),
    # The impulse is exact by construction: the force is divided by the
    # duration actually held, which is a whole number of control steps and so
    # is not quite the duration that was asked for.
    force=torch.full((num_envs,), impulse / duration, device=device),
    heading=heading,
    push_step=settle_steps + offset,
    hold_steps=hold_steps,
    recovery_steps=recovery_steps,
    settle_steps=settle_steps,
    num_steps=settle_steps + window_steps + recovery_steps,
    dt=dt,
    tolerance=cfg.tolerance,
    band_steps=max(1, round(cfg.hold / dt)),
    smooth_steps=max(1, round(cfg.smooth / dt)),
  )


def push_battery(
  cfg: PushCfg, mass: float, dt: float, device: torch.device | str = "cpu"
) -> tuple[PushPlan, ...]:
  """The whole battery, one plan per magnitude.

  Every plan has the same number of environments, so one harness runs all of
  them; see :func:`run_push_battery`.
  """
  return tuple(push_plan(cfg, delta_v, mass, dt, device) for delta_v in cfg.delta_v)


class PushDriver:
  """Applies a :class:`PushPlan`'s forces, one control step at a time.

  Call :meth:`apply` with the loop index *before* stepping the simulation, and
  :meth:`clear` when the run is over -- ``xfrc_applied`` persists until it is
  overwritten, so a plan that ended mid-push would otherwise keep shoving the
  next run.
  """

  def __init__(self, plan: PushPlan, robot: Entity, body_id: int) -> None:
    """
    Args:
      plan: The trials to apply.
      robot: The entity to push.
      body_id: Index of :data:`PUSH_BODY` in ``robot.body_names``.
    """
    self._plan = plan
    self._robot = robot
    self._body_ids = [body_id]
    device = plan.push_step.device
    self._force_w = torch.zeros(plan.num_envs, 1, 3, device=device)
    self._zeros = torch.zeros(plan.num_envs, 1, 3, device=device)

  def apply(self, step: int) -> None:
    """Write the wrench for control step ``step``.

    The direction is latched on the step the push starts, out of the yaw the
    robot holds at that moment, and the resulting world-frame force is held
    constant until the push expires. A push does not steer itself: it comes
    from somewhere, and where that is stops depending on the robot as soon as
    it starts to fall.
    """
    plan = self._plan
    starting = plan.push_step == step
    if bool(starting.any()):
      yaw = _yaw(self._robot.data.root_link_quat_w)
      angle = yaw + plan.heading
      direction = torch.stack(
        (angle.cos(), angle.sin(), torch.zeros_like(angle)), dim=-1
      )
      latched = (direction * plan.force.unsqueeze(-1)).unsqueeze(1)
      self._force_w = torch.where(starting[:, None, None], latched, self._force_w)

    active = (plan.push_step <= step) & (plan.push_step + plan.hold_steps > step)
    force = torch.where(active[:, None, None], self._force_w, self._zeros)
    self._robot.write_external_wrench_to_sim(
      force, self._zeros, body_ids=self._body_ids
    )

  def clear(self) -> None:
    """Zero the wrench on every environment."""
    self._robot.write_external_wrench_to_sim(
      self._zeros, self._zeros, body_ids=self._body_ids
    )


def _yaw(quaternion_w: torch.Tensor) -> torch.Tensor:
  return euler_xyz_from_quat(quaternion_w)[2]


@dataclass(frozen=True)
class PerEnvPushMetrics(PerEnvMetrics):
  """One row per trial: every walking metric, and what the push did.

  The inherited fields are :class:`~mjlab.evaluation.metrics.PerEnvMetrics`
  measured over the window that opens when the settle time ends, so they
  describe the robot around and after its push rather than the run-up to it.
  """

  push_delta_v: torch.Tensor
  """Push magnitude as a free-body velocity change, in m/s."""
  push_impulse: torch.Tensor
  """The same magnitude in N s."""
  push_force: torch.Tensor
  """Force applied over the push duration, in N."""
  push_heading_deg: torch.Tensor
  """Push direction in the robot's yaw frame at onset. 0 shoves it forwards."""
  push_time: torch.Tensor
  """Seconds from the start of the run to the onset."""
  push_offset: torch.Tensor
  """Seconds into the phase window the onset fell, i.e. which gait phase."""
  fell_before_push: torch.Tensor
  """1.0 if the robot was already down when the push landed.

  Nothing to do with the push: the controller could not hold the command for
  the settle time. Such a trial says nothing about push recovery and is left
  out of the survival fractions."""
  withstood: torch.Tensor
  """1.0 if the robot was still upright a recovery window after the push.

  The headline outcome. NaN where the robot fell before it was pushed."""
  time_to_fall: torch.Tensor
  """Seconds from the onset to the fall; NaN if it did not fall."""
  recovered: torch.Tensor
  """1.0 if the robot came back onto its command and was still up at the end of
  the window.

  Recovering is strictly more than withstanding: a robot can stay upright and
  never get back to the speed it was asked for. It is never less -- a robot
  that went down inside the window did not recover, whatever its velocity was
  doing on the way. NaN where the robot fell before it was pushed."""
  recovery_time: torch.Tensor
  """Seconds from the onset until the error was back inside the tolerance for
  the hold time; NaN if it never was, or if the robot went down anyway.

  Zero where the push never took the robot out of the band at all."""
  peak_speed_error: torch.Tensor
  """Largest planar velocity error reached after the push, in m/s.

  Off the smoothed velocity, so it is the excursion the push caused and not the
  sway every stride carries."""
  min_upright_after: torch.Tensor
  """Smallest up-axis component reached in the recovery window."""
  heading_error: torch.Tensor
  """Yaw at the end of the recovery window less the yaw at onset, less what the
  command asked for over that time, in radians."""


PUSH_QUALITY_METRICS: tuple[str, ...] = (
  "recovery_time",
  "peak_speed_error",
  "min_upright_after",
  "heading_error",
  "time_to_fall",
)
"""Push metrics that describe *how* the robot took it, as opposed to whether it
stayed up."""


class PushMetrics:
  """Accumulates :class:`EvalState` samples into :class:`PerEnvPushMetrics`.

  Wraps a :class:`~mjlab.evaluation.metrics.WalkMetrics` rather than
  reimplementing it: uprightness, the fall and the walking quality are measured
  by exactly the code every other run in this pipeline uses, and this class
  adds only what a push needs on top. Survival is one of the things it does
  *not* add -- ``withstood`` is read off the shared ``fall_time`` against each
  trial's own onset, so a push outcome and a sweep outcome cannot disagree
  about what a fall is.
  """

  def __init__(
    self,
    plan: PushPlan,
    fall_threshold: float = FALL_UPRIGHT_THRESHOLD,
  ) -> None:
    """
    Args:
      plan: The trials being run. Supplies the command, the onsets, the window
        the outcome is measured over and what counts as a recovery -- the same
        object :class:`PushDriver` takes its forces from, so a run cannot
        measure a different push from the one it applied.
      fall_threshold: See
        :data:`~mjlab.evaluation.metrics.FALL_UPRIGHT_THRESHOLD`.
    """
    self.plan = plan
    self.dt = plan.dt
    self.tolerance = plan.tolerance
    self.fall_threshold = fall_threshold
    self.band_steps = plan.band_steps
    self.smooth_steps = plan.smooth_steps

    # The walking metrics describe the window the push happens in, so the
    # settle time is discarded exactly as a sweep discards its warm-up.
    self.walk = WalkMetrics(
      command_b=plan.command,
      dt=plan.dt,
      fall_threshold=fall_threshold,
      warmup_s=plan.settle_steps * plan.dt,
    )

    num_envs = plan.num_envs
    device = plan.command.device
    self._step = 0
    self._alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    self._window = torch.zeros(self.smooth_steps, num_envs, 2, device=device)
    self._window_sum = torch.zeros(num_envs, 2, device=device)
    self._band_start = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self._recovery_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self._peak_error = torch.zeros(num_envs, device=device)
    self._min_upright = torch.full((num_envs,), math.inf, device=device)
    self._yaw_at_push = torch.zeros(num_envs, device=device)
    self._yaw_at_end = torch.zeros(num_envs, device=device)

  def start(self, state: EvalState) -> None:
    """Record the starting pose. Call once, before stepping."""
    self.walk.start(state)

  def record(self, state: EvalState) -> None:
    """Accumulate one control step.

    Call once per control step, after the step, with the same state the
    :class:`~mjlab.evaluation.metrics.WalkMetrics` would be given.
    """
    self.walk.record(state)
    plan = self.plan
    step = self._step
    self._step += 1

    upright = upright_from_quat(state.quaternion_w)
    self._alive = self._alive & (upright >= self.fall_threshold)

    # The yaw the driver read when it latched this trial's direction: the state
    # after the step before the onset is the state the driver saw.
    yaw = _yaw(state.quaternion_w)
    self._yaw_at_push = torch.where(plan.push_step == step + 1, yaw, self._yaw_at_push)
    ends = plan.push_step + plan.recovery_steps
    self._yaw_at_end = torch.where(ends == step + 1, yaw, self._yaw_at_end)

    # Moving average of the planar velocity, as a ring buffer with a running
    # sum: the recovery test needs the excursion, and the raw signal is
    # dominated by the stride.
    slot = step % self.smooth_steps
    planar = state.lin_vel_b[:, :2]
    self._window_sum = self._window_sum - self._window[slot] + planar
    self._window[slot] = planar
    filled = min(step + 1, self.smooth_steps)
    error = torch.linalg.vector_norm(
      self._window_sum / filled - plan.command[:, :2], dim=-1
    )

    pushed = plan.push_step <= step
    measuring = pushed & (ends > step) & self._alive
    self._peak_error = torch.where(
      measuring, torch.maximum(self._peak_error, error), self._peak_error
    )
    self._min_upright = torch.where(
      pushed & (ends > step),
      torch.minimum(self._min_upright, upright),
      self._min_upright,
    )

    # The first run of in-band samples that reaches the onset and lasts the
    # hold time from there. Clamping the run's start to the onset is what makes
    # a push too small to move the robot out of the band recover at zero rather
    # than not at all: the run it is inside began before the push landed.
    in_band = (error <= self.tolerance) & self._alive
    opening = in_band & (self._band_start < 0)
    self._band_start = torch.where(
      opening, torch.full_like(self._band_start, step), self._band_start
    )
    self._band_start = torch.where(
      in_band, self._band_start, torch.full_like(self._band_start, -1)
    )
    start = torch.maximum(self._band_start, plan.push_step)
    sustained = (
      in_band
      & (plan.push_step <= step)
      & (start + self.band_steps <= step + 1)
      & (self._recovery_step < 0)
    )
    self._recovery_step = torch.where(sustained, start, self._recovery_step)

  def result(self) -> PerEnvPushMetrics:
    """Reduce the accumulated samples. Safe to call more than once."""
    plan = self.plan
    walk = self.walk.result()
    nan = torch.full_like(walk.fall_time, float("nan"))

    push_time = plan.push_time
    fell = ~walk.fall_time.isnan()
    fell_before = fell & (walk.fall_time <= push_time)
    # A trial the controller had already lost before the push landed measures
    # the controller's ability to hold the command, which the sweeps measure;
    # it says nothing about the push, so it is reported as missing rather than
    # as a failure to recover.
    valid = ~fell_before

    def only_valid(values: torch.Tensor) -> torch.Tensor:
      return torch.where(valid, values, nan)

    deadline = push_time + plan.recovery_steps * self.dt
    withstood = ~(fell & (walk.fall_time <= deadline))
    recovered = self._recovery_step >= 0
    recovery_time = (self._recovery_step - plan.push_step).float() * self.dt
    # A recovery has to land inside the window the trial is judged over. An
    # environment pushed early keeps walking after its window closes, and a
    # robot that came back on command out there did not recover from the push.
    recovery_time = torch.where(
      recovered & withstood & (recovery_time <= plan.recovery_steps * self.dt),
      recovery_time,
      nan,
    )
    commanded_yaw = plan.command[:, 2] * plan.recovery_steps * self.dt

    return PerEnvPushMetrics(
      **{field.name: getattr(walk, field.name) for field in fields(PerEnvMetrics)},
      push_delta_v=plan.delta_v,
      push_impulse=plan.impulse,
      push_force=plan.force,
      push_heading_deg=torch.rad2deg(plan.heading),
      push_time=push_time,
      push_offset=(plan.push_step - plan.settle_steps).float() * self.dt,
      fell_before_push=fell_before.float(),
      withstood=only_valid(withstood.float()),
      time_to_fall=only_valid(walk.fall_time - push_time),
      recovered=only_valid(recovery_time.isnan().logical_not().float()),
      recovery_time=only_valid(recovery_time),
      peak_speed_error=only_valid(self._peak_error),
      min_upright_after=only_valid(
        torch.where(self._min_upright.isinf(), nan, self._min_upright)
      ),
      heading_error=only_valid(
        wrap_to_pi(self._yaw_at_end - self._yaw_at_push - commanded_yaw)
      ),
    )


def concat_push_metrics(
  parts: Sequence[PerEnvPushMetrics],
) -> PerEnvPushMetrics:
  """Join the passes of a battery into one table of trials."""
  if not parts:
    raise ValueError("nothing to concatenate")
  return PerEnvPushMetrics(
    **{
      field.name: torch.cat([getattr(part, field.name) for part in parts])
      for field in fields(PerEnvPushMetrics)
    }
  )


class PushHarness(Protocol):
  """What :func:`run_push_battery` needs from a harness."""

  num_envs: int
  control_dt: float
  device: str

  def robot_mass(self) -> float: ...

  def run_push(
    self, plan: PushPlan, on_step: Callable[[int], None] | None = None
  ) -> PushMetrics: ...


def run_push_battery(
  harness: PushHarness,
  cfg: PushCfg,
  on_pass: Callable[[int, PushPlan, PerEnvPushMetrics], None] | None = None,
  on_step: Callable[[int], None] | None = None,
) -> PerEnvPushMetrics:
  """Run every magnitude of a battery through one harness.

  The passes are separate runs of the same batch: the harness is reset at the
  top of each, so nothing a magnitude does carries into the next one.

  Args:
    harness: Built with ``cfg.trials_per_pass`` environments.
    cfg: The battery.
    on_pass: Called after each pass with its index, plan and results.
    on_step: Called with the step index after each control step of every pass.

  Returns:
    Every trial in the battery, in magnitude order.
  """
  if harness.num_envs != cfg.trials_per_pass:
    raise ValueError(
      f"harness has {harness.num_envs} environments; the battery needs "
      f"{cfg.trials_per_pass} "
      f"({cfg.directions} directions x {cfg.phases} phases x {cfg.replicas} "
      f"replicas)"
    )
  plans = push_battery(cfg, harness.robot_mass(), harness.control_dt, harness.device)
  results = []
  for index, plan in enumerate(plans):
    result = harness.run_push(plan, on_step=on_step).result()
    results.append(result)
    if on_pass is not None:
      on_pass(index, plan, result)
  return concat_push_metrics(results)


def push_envelope(metrics: PerEnvPushMetrics, threshold: float = 0.5) -> list[dict]:
  """The largest push each direction withstands, direction by direction.

  For one heading the survival fraction falls with magnitude; the critical
  magnitude is where it crosses ``threshold``, linearly interpolated between
  the two tested magnitudes that straddle the crossing. Reading the crossing
  rather than the last magnitude that survived is what makes the number stable:
  it is set by the whole curve, so a cell that lands a few trials either side
  of the threshold moves it by a fraction of a step rather than by a whole one.

  Returns:
    One entry per heading, in increasing heading order, each carrying the
    heading in degrees, the critical magnitude and impulse, the survival
    fraction at every magnitude, and whether the curve ever crossed.
  """
  heading = metrics.push_heading_deg
  magnitude = metrics.push_delta_v
  withstood = metrics.withstood
  # Mass is a property of the plant, so it is the same for every trial; taking
  # it off the ratio keeps the envelope reportable in newton-seconds without
  # the caller having to supply it again.
  per_delta_v = float(metrics.push_impulse[0] / metrics.push_delta_v[0])

  headings = sorted({float(value) for value in heading.tolist()})
  levels = sorted({float(value) for value in magnitude.tolist()})

  envelope = []
  for angle in headings:
    at_angle = heading == angle
    magnitudes, fractions, counts = [], [], []
    for level in levels:
      cell = at_angle & (magnitude == level)
      outcomes = withstood[cell]
      outcomes = outcomes[outcomes.isfinite()]
      if outcomes.numel() == 0:
        continue
      magnitudes.append(level)
      fractions.append(float(outcomes.mean()))
      counts.append(int(outcomes.numel()))
    critical = _crossing(magnitudes, fractions, threshold)
    envelope.append(
      {
        "heading_deg": angle,
        "critical_delta_v": critical,
        "critical_impulse": (
          float("nan") if math.isnan(critical) else critical * per_delta_v
        ),
        "crossed": not math.isnan(critical),
        "delta_v": magnitudes,
        "survival": fractions,
        "trials": counts,
      }
    )
  return envelope


def _crossing(
  magnitudes: list[float], fractions: list[float], threshold: float
) -> float:
  """Where a falling survival curve first drops below ``threshold``.

  A curve that never drops returns NaN: the envelope lies outside the
  magnitudes that were tested, and inventing a number for it would be reporting
  the battery's range rather than the controller's. A curve already below the
  threshold at its smallest magnitude is interpolated back towards the origin,
  where survival is 1.0 by construction -- no push, no fall.
  """
  for index, fraction in enumerate(fractions):
    if fraction >= threshold:
      continue
    low, low_fraction = (
      (0.0, 1.0) if index == 0 else (magnitudes[index - 1], fractions[index - 1])
    )
    high, high_fraction = magnitudes[index], fraction
    span = low_fraction - high_fraction
    if span <= 0.0:
      return low
    return low + (high - low) * (low_fraction - threshold) / span
  return float("nan")


def summarise_push(metrics: PerEnvPushMetrics, cfg: PushCfg | None = None) -> dict:
  """Aggregate a battery, on top of the usual walking summary.

  The walking blocks are the ones every other run writes, so a push run's
  ``summary.json`` can be read by anything that reads a sweep's. The ``push``
  block on top carries what only a battery has: how many trials were spoiled
  before their push landed, the survival fraction over the whole battery, and
  the envelope.
  """
  summary = summarise(metrics, WALK_QUALITY_METRICS + PUSH_QUALITY_METRICS)
  valid = metrics.withstood.isfinite()
  withstood = metrics.withstood[valid]
  recovered = metrics.recovered[valid]
  push: dict = {
    "num_trials": int(metrics.withstood.numel()),
    "num_spoiled": int((metrics.fell_before_push > 0.5).sum()),
    "withstood_rate": float(withstood.mean()) if withstood.numel() else float("nan"),
    "recovered_rate": (float(recovered.mean()) if recovered.numel() else float("nan")),
    "envelope": push_envelope(metrics),
  }
  if cfg is not None:
    push["trials_per_cell"] = cfg.trials_per_cell
    push["command"] = {"vx": cfg.vx, "vy": cfg.vy, "wz": cfg.wz}
  summary["push"] = push
  return summary


def format_push_summary(summary: dict) -> str:
  """One-screen rendering of :func:`summarise_push`."""
  push = summary["push"]
  lines = [
    f"trials            : {push['num_trials']}",
    f"withstood         : {100.0 * push['withstood_rate']:.1f}%",
    f"recovered         : {100.0 * push['recovered_rate']:.1f}%",
  ]
  if push["num_spoiled"]:
    lines.append(f"spoiled           : {push['num_spoiled']} fell before being pushed")
  crossed = [entry for entry in push["envelope"] if entry["crossed"]]
  if crossed:
    weakest = min(crossed, key=lambda entry: entry["critical_delta_v"])
    strongest = max(crossed, key=lambda entry: entry["critical_delta_v"])
    lines.append(
      f"weakest direction : {weakest['heading_deg']:.0f} deg at "
      f"{weakest['critical_delta_v']:.2f} m/s "
      f"({weakest['critical_impulse']:.2f} N s)"
    )
    lines.append(
      f"strongest         : {strongest['heading_deg']:.0f} deg at "
      f"{strongest['critical_delta_v']:.2f} m/s "
      f"({strongest['critical_impulse']:.2f} N s)"
    )
  if len(crossed) < len(push["envelope"]):
    lines.append(
      f"open directions   : {len(push['envelope']) - len(crossed)} withstood "
      f"every magnitude tested"
    )
  return "\n".join(lines)
