"""Push survival: a measured shove, and whether the robot stays up.

A *trial* is one shove. The robot walks under a fixed command until the gait is
established, a constant force is applied to its root body for
:attr:`PushCfg.duration` seconds, and the run continues for
:attr:`PushCfg.recovery` seconds. A robot still upright at the end of that
window *withstood* the push. A *battery* is a grid of trials over three
variables:

``magnitude``
  How hard. Parameterised as the velocity change the impulse would produce on a
  free body of the robot's mass -- :attr:`PushCfg.delta_v`, in m/s -- because
  that is comparable across robots that do not weigh the same. The force
  actually applied is ``mass * delta_v / duration``.
``direction``
  Which way, as a heading in the robot's own yaw frame at the instant the push
  lands: 0 shoves it forwards, 90 degrees shoves it to its left. Latched at
  onset and held in the world frame for the duration, which is what a shove is.
``phase``
  When, within a gait cycle. The onsets are spread evenly across
  :attr:`PushCfg.phase_window`, so every reported number is an average over
  gait phase rather than a measurement at one arbitrary point in the stride.

The battery is run one magnitude at a time -- :func:`push_battery` returns one
:class:`PushPlan` per magnitude -- so the batch size is set by the direction,
phase and replica counts alone. The per-direction survival curves reduce to the
*envelope* (:func:`push_envelope`): the magnitude at which half the trials in
that direction end on the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Protocol, Sequence

import torch

from mjlab.entity import Entity
from mjlab.evaluation.metrics import (
  FALL_UPRIGHT_THRESHOLD,
  EvalState,
  PerEnvMetrics,
  WalkMetrics,
  summarise,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat


@dataclass(frozen=True)
class PushCfg:
  """The battery: what to push with, from where, and when."""

  vx: float = 0.5
  """Forward velocity command held for the whole run, in m/s."""
  vy: float = 0.0
  """Lateral velocity command, in m/s."""
  wz: float = 0.0
  """Yaw rate command, in rad/s."""

  delta_v: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.5,
    4.0,
  )
  """Push magnitudes, as the velocity change each impulse would produce on a
  free body of the robot's mass, in m/s. One pass per value.

  Fine where a walking Booster K1 policy's envelope lies (roughly 1.5 to 2.5
  when walking at 0.5 m/s) and coarse above it, so a stronger controller's
  envelope still has an outside rather than running off the end."""
  directions: int = 12
  """Push headings, evenly spaced over the full circle. 0 shoves the robot
  forwards, 90 degrees shoves it to its left."""
  phases: int = 12
  """Push onsets, evenly spaced across :attr:`phase_window`."""
  replicas: int = 4
  """Environments per (direction, phase) pair.

  Not redundant: two identical robots in one batch drift apart within a few
  steps and a gait amplifies that, so a trial near the edge of the envelope is
  effectively a coin flip. Replicas sample it."""

  duration: float = 0.2
  """Seconds the force is held. Short enough to be an impulse against a gait
  cycle, long enough to span several control steps."""
  settle: float = 8.0
  """Seconds of undisturbed walking before the earliest push, so the run-up is
  over and the controller is at the speed it was asked for."""
  phase_window: float = 0.64
  """Seconds the onsets are spread over; about one gait cycle. A learned policy
  sets its own cadence, but with a dozen onsets across this window the sampled
  phases cover any cycle near this length, which is all the average needs."""
  recovery: float = 4.0
  """Seconds after the onset the outcome is judged over. Long enough to contain
  the stumble, not so long that an unrelated fall lands inside it."""

  def __post_init__(self) -> None:
    if not self.delta_v:
      raise ValueError("push battery has no magnitudes")
    if min(self.delta_v) <= 0.0:
      raise ValueError("push magnitudes must be positive")
    for name in ("directions", "phases", "replicas"):
      if getattr(self, name) < 1:
        raise ValueError(f"push {name} must be at least 1")
    for name in ("duration", "recovery"):
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

  Attributes:
    command: Shape ``(N, 3)`` velocity command held for the whole run.
    delta_v: Shape ``(N,)`` push magnitude, as free-body velocity change in m/s.
    impulse: Shape ``(N,)`` the same magnitude as an impulse, in N s.
    force: Shape ``(N,)`` force applied over :attr:`hold_steps`, in N.
    heading: Shape ``(N,)`` push direction in the robot's yaw frame, radians.
    push_step: Shape ``(N,)`` control step the force switches on.
    hold_steps: Steps the force is held.
    recovery_steps: Steps after the onset the outcome is judged over.
    settle_steps: Steps before the earliest onset.
    num_steps: Steps in the whole pass.
    dt: Control period, in seconds.
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

  The batch is ordered direction-major, then phase, then replica. Every trial
  also carries its own magnitude and heading as columns, so nothing downstream
  has to reconstruct the layout.
  """
  if mass <= 0.0:
    raise ValueError(f"robot mass must be positive, got {mass}")

  hold_steps = max(1, round(cfg.duration / dt))
  settle_steps = max(1, round(cfg.settle / dt))
  recovery_steps = max(1, round(cfg.recovery / dt))
  window_steps = max(1, round(cfg.phase_window / dt))

  # Onsets evenly spaced across the phase window. Where the control rate is too
  # coarse for the window, several phases land on one step and the duplicates
  # become replicas.
  offsets = [round(index * window_steps / cfg.phases) for index in range(cfg.phases)]

  heading = torch.tensor(cfg.headings, device=device, dtype=torch.float32)
  heading = heading.repeat_interleave(cfg.phases * cfg.replicas)
  offset = torch.tensor(offsets, device=device, dtype=torch.long)
  offset = offset.repeat_interleave(cfg.replicas).repeat(cfg.directions)

  num_envs = cfg.trials_per_pass
  impulse = mass * delta_v
  return PushPlan(
    command=torch.tensor([cfg.command], device=device, dtype=torch.float32).repeat(
      num_envs, 1
    ),
    delta_v=torch.full((num_envs,), delta_v, device=device),
    impulse=torch.full((num_envs,), impulse, device=device),
    # Divided by the duration actually held -- a whole number of control steps
    # -- so the impulse is exact.
    force=torch.full((num_envs,), impulse / (hold_steps * dt), device=device),
    heading=heading,
    push_step=settle_steps + offset,
    hold_steps=hold_steps,
    recovery_steps=recovery_steps,
    settle_steps=settle_steps,
    num_steps=settle_steps + window_steps + recovery_steps,
    dt=dt,
  )


def push_battery(
  cfg: PushCfg, mass: float, dt: float, device: torch.device | str = "cpu"
) -> tuple[PushPlan, ...]:
  """The whole battery, one plan per magnitude, all the same batch size."""
  return tuple(push_plan(cfg, delta_v, mass, dt, device) for delta_v in cfg.delta_v)


def _yaw(quaternion_w: torch.Tensor) -> torch.Tensor:
  return euler_xyz_from_quat(quaternion_w)[2]


class PushDriver:
  """Applies a :class:`PushPlan`'s forces, one control step at a time.

  Call :meth:`apply` with the loop index *before* stepping the simulation, and
  :meth:`clear` when the run is over -- ``xfrc_applied`` persists until it is
  overwritten, so a plan that ended mid-push would keep shoving the next run.
  """

  def __init__(self, plan: PushPlan, robot: Entity, body_id: int) -> None:
    """
    Args:
      plan: The trials to apply.
      robot: The entity to push.
      body_id: Index into ``robot.body_names`` of the body to push. MuJoCo
        applies ``xfrc_applied`` at that body's centre of mass.
    """
    self._plan = plan
    self._robot = robot
    self._body_ids = [body_id]
    device = plan.push_step.device
    self._force_w = torch.zeros(plan.num_envs, 1, 3, device=device)
    self._zeros = torch.zeros(plan.num_envs, 1, 3, device=device)

  def apply(self, step: int) -> None:
    """Write the wrench for control step ``step``.

    The direction is latched on the step the push starts, from the yaw the
    robot holds at that moment, and held constant in the world frame until the
    push expires.
    """
    plan = self._plan
    starting = plan.push_step == step
    if bool(starting.any()):
      angle = _yaw(self._robot.data.root_link_quat_w) + plan.heading
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


@dataclass(frozen=True)
class PerEnvPushMetrics(PerEnvMetrics):
  """One row per trial: the walking metrics, and what the push did.

  The inherited fields are measured over the window that opens when the settle
  time ends, so they describe the robot around and after its push.
  """

  push_delta_v: torch.Tensor
  """Push magnitude as a free-body velocity change, in m/s."""
  push_impulse: torch.Tensor
  """The same magnitude in N s."""
  push_heading_deg: torch.Tensor
  """Push direction in the robot's yaw frame at onset. 0 shoves it forwards."""
  push_time: torch.Tensor
  """Seconds from the start of the run to the onset."""
  fell_before_push: torch.Tensor
  """1.0 if the robot was already down when the push landed. Such a trial says
  nothing about the push and is left out of the survival fractions."""
  withstood: torch.Tensor
  """1.0 if the robot was still upright a recovery window after the push. NaN
  where it fell before it was pushed."""


class PushMetrics:
  """Accumulates :class:`EvalState` samples into :class:`PerEnvPushMetrics`.

  Wraps a :class:`~mjlab.evaluation.metrics.WalkMetrics`: ``withstood`` is read
  off its ``fall_time`` against each trial's own onset, so a push outcome and a
  grid outcome cannot disagree about what a fall is.
  """

  def __init__(
    self, plan: PushPlan, fall_threshold: float = FALL_UPRIGHT_THRESHOLD
  ) -> None:
    self.plan = plan
    self.walk = WalkMetrics(
      command_b=plan.command,
      dt=plan.dt,
      fall_threshold=fall_threshold,
      warmup_s=plan.settle_steps * plan.dt,
    )

  def record(self, state: EvalState) -> None:
    """Accumulate one control step."""
    self.walk.record(state)

  def result(self) -> PerEnvPushMetrics:
    """Reduce the accumulated samples. Safe to call more than once."""
    plan = self.plan
    walk = self.walk.result()
    push_time = plan.push_time
    fell = ~walk.fall_time.isnan()
    fell_before = fell & (walk.fall_time <= push_time)
    deadline = push_time + plan.recovery_steps * plan.dt
    withstood = (~(fell & (walk.fall_time <= deadline))).float()

    return PerEnvPushMetrics(
      **{field.name: getattr(walk, field.name) for field in fields(PerEnvMetrics)},
      push_delta_v=plan.delta_v,
      push_impulse=plan.impulse,
      push_heading_deg=torch.rad2deg(plan.heading),
      push_time=push_time,
      fell_before_push=fell_before.float(),
      withstood=torch.where(fell_before, float("nan"), withstood),
    )


def concat_push_metrics(parts: Sequence[PerEnvPushMetrics]) -> PerEnvPushMetrics:
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

  def run_push(self, plan: PushPlan) -> PushMetrics: ...


def run_push_battery(
  harness: PushHarness,
  cfg: PushCfg,
  on_pass: Callable[[int, PushPlan, PerEnvPushMetrics], None] | None = None,
) -> PerEnvPushMetrics:
  """Run every magnitude of a battery through one harness.

  The harness is reset at the top of each pass, so nothing a magnitude does
  carries into the next one.

  Args:
    harness: Built with ``cfg.trials_per_pass`` environments.
    cfg: The battery.
    on_pass: Called after each pass with its index, plan and results.

  Returns:
    Every trial in the battery, in magnitude order.
  """
  if harness.num_envs != cfg.trials_per_pass:
    raise ValueError(
      f"harness has {harness.num_envs} environments; the battery needs "
      f"{cfg.trials_per_pass} ({cfg.directions} directions x {cfg.phases} "
      f"phases x {cfg.replicas} replicas)"
    )
  plans = push_battery(cfg, harness.robot_mass(), harness.control_dt, harness.device)
  results = []
  for index, plan in enumerate(plans):
    result = harness.run_push(plan).result()
    results.append(result)
    if on_pass is not None:
      on_pass(index, plan, result)
  return concat_push_metrics(results)


def push_envelope(metrics: PerEnvPushMetrics, threshold: float = 0.5) -> list[dict]:
  """The largest push each direction withstands, direction by direction.

  For one heading the survival fraction falls with magnitude; the critical
  magnitude is where it crosses ``threshold``, linearly interpolated between
  the two tested magnitudes that straddle the crossing, so a cell landing a few
  trials either side of the threshold moves it by a fraction of a step rather
  than by a whole one.

  Returns:
    One entry per heading, in increasing heading order, each carrying the
    heading in degrees, the critical magnitude and impulse, the survival
    fraction at every magnitude, and whether the curve ever crossed.
  """
  heading = metrics.push_heading_deg
  magnitude = metrics.push_delta_v
  withstood = metrics.withstood
  per_delta_v = float(metrics.push_impulse[0] / metrics.push_delta_v[0])

  envelope = []
  for angle in sorted({float(value) for value in heading.tolist()}):
    at_angle = heading == angle
    magnitudes, fractions, counts = [], [], []
    for level in sorted({float(value) for value in magnitude.tolist()}):
      outcomes = withstood[at_angle & (magnitude == level)]
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

  A curve that never drops returns NaN: the envelope lies outside the tested
  magnitudes. A curve already below the threshold at its smallest magnitude is
  interpolated back towards the origin, where survival is 1.0 by construction.
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


def summarise_push(metrics: PerEnvPushMetrics, cfg: PushCfg) -> dict:
  """The usual walking summary, plus a ``push`` block with the envelope."""
  summary = summarise(metrics)
  withstood = metrics.withstood[metrics.withstood.isfinite()]
  summary["push"] = {
    "num_trials": int(metrics.withstood.numel()),
    "num_spoiled": int((metrics.fell_before_push > 0.5).sum()),
    "withstood_rate": float(withstood.mean()) if withstood.numel() else float("nan"),
    "trials_per_cell": cfg.trials_per_cell,
    "command": {"vx": cfg.vx, "vy": cfg.vy, "wz": cfg.wz},
    "envelope": push_envelope(metrics),
  }
  return summary


def format_push_summary(summary: dict) -> str:
  """One-screen rendering of :func:`summarise_push`."""
  push = summary["push"]
  lines = [
    f"trials            : {push['num_trials']}",
    f"withstood         : {100.0 * push['withstood_rate']:.1f}%",
  ]
  if push["num_spoiled"]:
    lines.append(f"spoiled           : {push['num_spoiled']} fell before being pushed")
  crossed = [entry for entry in push["envelope"] if entry["crossed"]]
  if crossed:
    weakest = min(crossed, key=lambda entry: entry["critical_delta_v"])
    strongest = max(crossed, key=lambda entry: entry["critical_delta_v"])
    for label, entry in (
      ("weakest direction", weakest),
      ("strongest        ", strongest),
    ):
      lines.append(
        f"{label} : {entry['heading_deg']:.0f} deg at "
        f"{entry['critical_delta_v']:.2f} m/s ({entry['critical_impulse']:.2f} N s)"
      )
  if len(crossed) < len(push["envelope"]):
    lines.append(
      f"open directions   : {len(push['envelope']) - len(crossed)} withstood "
      f"every magnitude tested"
    )
  return "\n".join(lines)
