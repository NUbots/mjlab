"""Time-varying velocity commands for tracking runs.

The command sweeps in :mod:`mjlab.evaluation.harness` hold one command for a
whole episode and report a mean, which measures steady-state tracking and says
nothing about how the robot gets there. A *profile* run instead moves the
command during the episode -- forward, then sideways, then turning, then the
combinations -- and records the response step by step, which is the figure
DeepWalk (Rodriguez and Behnke, ICRA 2021, Fig. 3) uses to show a gait is
omnidirectional.

The batch is used for *independent* profiles rather than for replicas of one
long sequence. A single sequence would be contaminated by its own history: the
quintic engine falls over under a backwards command, and everything after that
point in the sequence would then be a measurement of a robot on the floor. So
each :class:`Lane` is its own command schedule, every environment runs one
lane, and a lane that ends early is held at rest while the others finish.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

RAMP_S = 1.5
"""Seconds a command takes to slew between two plateaus.

The command is a trapezoid, not a step: on the robot the operator's stick moves
at a finite rate, and a step would measure the controller's response to a
discontinuity that never happens.
"""

HOLD_S = 4.0
"""Seconds a command is held at a plateau, after the ramp onto it."""


@dataclass(frozen=True)
class Segment:
  """One plateau of a command schedule.

  Attributes:
    vx: Forward velocity command, in m/s.
    vy: Lateral velocity command, in m/s.
    wz: Yaw rate command, in rad/s.
    hold: Seconds to hold this plateau, after ramping onto it.
    ramp: Seconds to slew from the previous plateau to this one.
  """

  vx: float = 0.0
  vy: float = 0.0
  wz: float = 0.0
  hold: float = HOLD_S
  ramp: float = RAMP_S

  @property
  def duration(self) -> float:
    return self.ramp + self.hold

  @property
  def value(self) -> tuple[float, float, float]:
    return (self.vx, self.vy, self.wz)


@dataclass(frozen=True)
class Lane:
  """A named command schedule, run by one slice of the batch.

  Attributes:
    name: Identifies the lane in the trace file and in a figure's panels.
    axes: Which of ``("vx", "vy", "wz")`` this lane exercises. Only a labelling
      convenience -- the schedule is what it is -- but it is what a plot needs
      to know which traces to draw.
    segments: The schedule, in order.
  """

  name: str
  axes: tuple[str, ...]
  segments: tuple[Segment, ...]

  @property
  def duration(self) -> float:
    return sum(segment.duration for segment in self.segments)

  def command_at(self, t: float) -> tuple[float, float, float]:
    """The command at time ``t`` seconds, ramps interpolated.

    Past the end of the schedule the lane stands still, so a short lane can
    share a batch with a long one.
    """
    previous = (0.0, 0.0, 0.0)
    elapsed = 0.0
    for segment in self.segments:
      if t < elapsed + segment.ramp:
        if segment.ramp <= 0.0:
          return segment.value
        alpha = (t - elapsed) / segment.ramp
        end = segment.value
        return (
          previous[0] + alpha * (end[0] - previous[0]),
          previous[1] + alpha * (end[1] - previous[1]),
          previous[2] + alpha * (end[2] - previous[2]),
        )
      if t < elapsed + segment.duration:
        return segment.value
      previous = segment.value
      elapsed += segment.duration
    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Profile:
  """A set of lanes, tiled over a batch of environments.

  Attributes:
    lanes: The schedules to run.
    replicas: Environments per lane. One is enough for a deterministic
      controller; a policy that sees noisy observations gives a slightly
      different trace each time, and replicas turn that into a band.
  """

  lanes: tuple[Lane, ...]
  replicas: int = 1

  @property
  def num_envs(self) -> int:
    return len(self.lanes) * self.replicas

  @property
  def duration(self) -> float:
    """Seconds needed for every lane to finish its schedule."""
    return max(lane.duration for lane in self.lanes)

  def lane_of_env(self) -> tuple[str, ...]:
    """Lane name per environment, in batch order."""
    return tuple(lane.name for lane in self.lanes for _ in range(self.replicas))

  def commands(self, dt: float, duration: float | None = None) -> torch.Tensor:
    """The whole schedule, precomputed.

    Args:
      dt: Control period, in seconds.
      duration: Seconds to generate. Defaults to :attr:`duration`.

    Returns:
      Shape ``(T, num_envs, 3)`` commands, ordered ``(vx, vy, wz)``. Row ``k``
      is the command in force for control step ``k``.
    """
    span = self.duration if duration is None else duration
    num_steps = int(round(span / dt))
    times = [step * dt for step in range(num_steps)]
    per_lane = torch.tensor(
      [[lane.command_at(t) for lane in self.lanes] for t in times],
      dtype=torch.float32,
    )
    return per_lane.repeat_interleave(self.replicas, dim=1).contiguous()


def _sweep(
  axis: str, amplitude: float, hold: float, ramp: float
) -> tuple[Segment, ...]:
  """Rest, positive, rest, negative, rest, on one axis."""
  rest = Segment(hold=hold, ramp=ramp)
  positive = Segment(**{axis: amplitude}, hold=hold, ramp=ramp)
  negative = Segment(**{axis: -amplitude}, hold=hold, ramp=ramp)
  return (Segment(hold=hold, ramp=0.0), positive, rest, negative, rest)


def _pair(
  first: tuple[str, float],
  second: tuple[str, float],
  hold: float,
  ramp: float,
) -> tuple[Segment, ...]:
  """Rest, both positive, rest, one sign flipped, rest."""
  (axis_a, value_a), (axis_b, value_b) = first, second
  rest = Segment(hold=hold, ramp=ramp)
  same = Segment(**{axis_a: value_a, axis_b: value_b}, hold=hold, ramp=ramp)
  opposed = Segment(**{axis_a: value_a, axis_b: -value_b}, hold=hold, ramp=ramp)
  return (Segment(hold=hold, ramp=0.0), same, rest, opposed, rest)


@dataclass(frozen=True)
class ProfileCfg:
  """Amplitudes and timing of :func:`omnidirectional_profile`."""

  vx: float = 0.35
  """Forward amplitude, in m/s. The negative half of the sweep uses ``-vx``."""
  vy: float = 0.20
  """Lateral amplitude, in m/s."""
  wz: float = 0.60
  """Yaw amplitude, in rad/s."""
  combined_vx: float = 0.25
  """Forward component of the combined lanes, held while another axis moves."""
  combined_vy: float = 0.15
  combined_wz: float = 0.40
  hold: float = HOLD_S
  ramp: float = RAMP_S
  replicas: int = 4


DEFAULT_PROFILE_CFG = ProfileCfg()


def omnidirectional_profile(cfg: ProfileCfg = DEFAULT_PROFILE_CFG) -> Profile:
  """Six lanes: the three axes alone, then the three pairs of them.

  The order is DeepWalk's -- sagittal, lateral, turning, then sagittal with
  lateral, sagittal with turning, lateral with turning -- and each lane visits
  both signs of the axis it is exercising, so a controller that walks forwards
  but not backwards shows the asymmetry rather than averaging it away.
  """
  hold, ramp = cfg.hold, cfg.ramp
  return Profile(
    lanes=(
      Lane("sagittal", ("vx",), _sweep("vx", cfg.vx, hold, ramp)),
      Lane("lateral", ("vy",), _sweep("vy", cfg.vy, hold, ramp)),
      Lane("turning", ("wz",), _sweep("wz", cfg.wz, hold, ramp)),
      Lane(
        "sagittal+lateral",
        ("vx", "vy"),
        _pair(("vx", cfg.combined_vx), ("vy", cfg.combined_vy), hold, ramp),
      ),
      Lane(
        "sagittal+turning",
        ("vx", "wz"),
        _pair(("vx", cfg.combined_vx), ("wz", cfg.combined_wz), hold, ramp),
      ),
      Lane(
        "lateral+turning",
        ("vy", "wz"),
        _pair(("vy", cfg.combined_vy), ("wz", cfg.combined_wz), hold, ramp),
      ),
    ),
    replicas=cfg.replicas,
  )
