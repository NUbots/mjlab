"""High-level training focus: what the velocity task should care about.

The competence machinery in :mod:`.competence` decides *when* pressure is
safe to apply. This module decides *what* that pressure is aimed at, in
terms a human tuning a run actually thinks in: "I want forward speed above
all else", "I need stability at walking pace but not at a sprint", "I do
not care about stability at all right now".

Everything is expressed over four **motion channels** -- ``forward``
(+x), ``backward`` (-x), ``strafe`` (|y|) and ``yaw`` (|wz|). Each channel
carries three numbers:

* ``speed`` -- how much delivered velocity in this channel pays. Relative
  only; the attainment reward is scale-invariant in these weights, so
  doubling every channel changes nothing and only ratios matter.
* ``target_speed`` -- how far the command curriculum ramps this channel.
  Deliberately separate from ``speed``: one is reward per unit of
  delivery, the other is how much you *ask* for, and they fail in
  different ways. Commanding well past the frontier teaches surrender no
  matter how richly delivery pays.
* ``stability`` -- a :class:`SpeedProfile` mapping speed to how much the
  stability reward group matters at that speed.

Two global scalars, ``stability_scale`` and ``speed_scale``, slide the
whole balance without disturbing its shape;
:meth:`TrainingFocusCfg.with_balance` sets both from one number so the
stability-versus-speed axis can be swept directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

CHANNELS: tuple[str, ...] = ("forward", "backward", "strafe", "yaw")
"""Motion channels, in the order the runtime packs them into tensors."""

DEFAULT_STABILITY_TERMS: tuple[str, ...] = (
  "upright",
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "feet_distance",
  "foot_slip",
  "foot_flat",
)
"""Reward terms the stability gate scales.

``termination_penalty`` is deliberately absent. It is the strongest
stability signal in the task, and gating it would both make falling at
speed cheap and decouple the reward from the fall statistics that drive
competence promotion -- the gate would be quietly editing its own
evidence. Add it explicitly if you want to experiment with that.
"""


@dataclass(frozen=True)
class SpeedProfile:
  """Piecewise-linear emphasis as a function of speed.

  ``knots`` is an ascending sequence of ``(speed, emphasis)`` pairs.
  Between knots the emphasis interpolates linearly; outside the first and
  last knot it is held flat, so a profile never extrapolates into
  emphasis values its author did not write down.
  """

  knots: tuple[tuple[float, float], ...]

  def __post_init__(self) -> None:
    if len(self.knots) == 0:
      raise ValueError("SpeedProfile needs at least one knot.")
    speeds = [s for s, _ in self.knots]
    if any(b < a for a, b in zip(speeds, speeds[1:], strict=False)):
      raise ValueError(f"SpeedProfile knots must ascend in speed, got {speeds}.")
    if any(s < 0.0 for s in speeds):
      raise ValueError(f"SpeedProfile speeds must be non-negative, got {speeds}.")
    if any(e < 0.0 for _, e in self.knots):
      raise ValueError("SpeedProfile emphasis values must be non-negative.")

  @classmethod
  def constant(cls, emphasis: float = 1.0) -> SpeedProfile:
    """Same emphasis at every speed: "I care about this everywhere"."""
    return cls(knots=((0.0, emphasis),))

  @classmethod
  def decaying(
    cls,
    slow: float = 1.0,
    fast: float = 0.2,
    from_speed: float = 0.5,
    to_speed: float = 1.0,
  ) -> SpeedProfile:
    """High below ``from_speed``, fading to ``fast`` above ``to_speed``.

    "I need this to be solid at walking pace, but I will trade it away
    once it is actually moving."
    """
    return cls(knots=((from_speed, slow), (to_speed, fast)))

  @classmethod
  def rising(
    cls,
    slow: float = 0.2,
    fast: float = 1.0,
    from_speed: float = 0.6,
    to_speed: float = 1.2,
  ) -> SpeedProfile:
    """Low at low speed, climbing to ``fast`` above ``to_speed``.

    "Shuffling around slowly is not where my failures are; hold it
    together when it is moving."
    """
    return cls(knots=((from_speed, slow), (to_speed, fast)))

  @classmethod
  def band(
    cls,
    peak: float = 1.0,
    outside: float = 0.2,
    low: float = 0.4,
    high: float = 0.9,
    edge: float = 0.2,
  ) -> SpeedProfile:
    """Emphasis peaking across ``[low, high]`` and falling off either side.

    ``edge`` is the width of each linear shoulder.
    """
    if high < low:
      raise ValueError(f"band() needs low <= high, got {low} > {high}.")
    if edge <= 0.0:
      raise ValueError(f"band() needs a positive edge width, got {edge}.")
    return cls(
      knots=(
        (max(low - edge, 0.0), outside),
        (low, peak),
        (high, peak),
        (high + edge, outside),
      )
    )

  def evaluate(self, speed: float) -> float:
    """Scalar evaluation, for config-time reasoning and tests.

    The runtime path evaluates all channels at once on device; see
    :class:`~mjlab.tasks.velocity.mdp.stability_gate.StabilityGate`.
    """
    knots = self.knots
    if speed <= knots[0][0]:
      return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False):
      if speed <= x1:
        if x1 == x0:
          return y1
        return y0 + (y1 - y0) * (speed - x0) / (x1 - x0)
    return knots[-1][1]

  def scaled(self, factor: float) -> SpeedProfile:
    """This profile with every emphasis multiplied by ``factor``."""
    return SpeedProfile(knots=tuple((s, e * factor) for s, e in self.knots))

  @property
  def max_emphasis(self) -> float:
    return max(e for _, e in self.knots)


@dataclass(frozen=True)
class ChannelFocus:
  """What one motion channel is worth and how stable it must be."""

  speed: float = 1.0
  """Relative weight on delivered velocity in this channel."""

  target_speed: float = 1.0
  """Top of this channel's command range once the curriculum tops out."""

  stability: SpeedProfile = field(default_factory=SpeedProfile.constant)
  """Stability emphasis as a function of the speed achieved here."""

  def __post_init__(self) -> None:
    if self.speed < 0.0:
      raise ValueError(f"ChannelFocus.speed must be non-negative, got {self.speed}.")
    if self.target_speed < 0.0:
      raise ValueError(
        f"ChannelFocus.target_speed must be non-negative, got {self.target_speed}."
      )


@dataclass(frozen=True)
class TrainingFocusCfg:
  """The complete "what should this run care about" configuration."""

  forward: ChannelFocus = field(default_factory=ChannelFocus)
  backward: ChannelFocus = field(default_factory=ChannelFocus)
  strafe: ChannelFocus = field(default_factory=ChannelFocus)
  yaw: ChannelFocus = field(default_factory=ChannelFocus)

  stability_scale: float = 1.0
  """Global multiplier on every channel's stability profile."""

  speed_scale: float = 1.0
  """Global multiplier on the attainment reward weights."""

  linear_attainment_weight: float = 2.0
  """Base weight of the linear attainment term, before ``speed_scale``."""

  angular_attainment_weight: float = 1.0
  """Base weight of the angular attainment term, before ``speed_scale``.

  Lower than linear by default: the yaw tracking kernel is an order of
  magnitude wider (std^2 = 0.5 against 0.05), so yaw already has usable
  gradient where the linear kernel has none.
  """

  stability_terms: tuple[str, ...] = DEFAULT_STABILITY_TERMS
  """Reward terms the stability gate scales."""

  standing_stability: float | None = None
  """Emphasis used when the command is ~zero. ``None`` reads each profile
  at speed 0, which is what you want when low speed is where stability
  matters. Set it explicitly to decouple standing from the low-speed end
  of the curves."""

  command_stages: int = 3
  """Number of stages in the derived command curriculum ramp."""

  command_ramp_start: float = 0.6
  """First stage commands this fraction of each channel's target."""

  command_ramp_iters: int = 11_000
  """Training iteration by which the command ramp reaches its target."""

  def __post_init__(self) -> None:
    if self.stability_scale < 0.0:
      raise ValueError("stability_scale must be non-negative.")
    if self.speed_scale < 0.0:
      raise ValueError("speed_scale must be non-negative.")
    if self.command_stages < 1:
      raise ValueError("command_stages must be at least 1.")
    if not 0.0 < self.command_ramp_start <= 1.0:
      raise ValueError(
        f"command_ramp_start must lie in (0, 1], got {self.command_ramp_start}."
      )
    if sum(self.channel(name).speed for name in CHANNELS) <= 0.0:
      raise ValueError(
        "At least one channel must have a non-zero speed weight, otherwise "
        "the attainment reward has no direction to pay for."
      )

  def channel(self, name: str) -> ChannelFocus:
    if name not in CHANNELS:
      raise KeyError(f"Unknown motion channel {name!r}; expected one of {CHANNELS}.")
    return getattr(self, name)

  def channels(self) -> dict[str, ChannelFocus]:
    return {name: self.channel(name) for name in CHANNELS}

  def speed_weights(self) -> tuple[float, ...]:
    """Per-channel speed weights in :data:`CHANNELS` order."""
    return tuple(self.channel(name).speed for name in CHANNELS)

  def stability_knots(self) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Per-channel stability knots, already scaled by ``stability_scale``."""
    return tuple(
      self.channel(name).stability.scaled(self.stability_scale).knots
      for name in CHANNELS
    )

  def with_balance(self, balance: float) -> TrainingFocusCfg:
    """Slide between stability (``0.0``) and speed (``1.0``).

    ``0.5`` leaves both scales at 1.0, so the shape of the focus is
    untouched and only the emphasis between the two families moves. This
    is the single knob to sweep when looking for the middle ground.
    """
    if not 0.0 <= balance <= 1.0:
      raise ValueError(f"balance must lie in [0, 1], got {balance}.")
    return replace(
      self,
      stability_scale=2.0 * (1.0 - balance),
      speed_scale=2.0 * balance,
    )

  def command_ranges(
    self,
  ) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
  ]:
    """Fully-ramped ``(lin_vel_x, lin_vel_y, ang_vel_z)`` command ranges."""
    return (
      (-self.backward.target_speed, self.forward.target_speed),
      (-self.strafe.target_speed, self.strafe.target_speed),
      (-self.yaw.target_speed, self.yaw.target_speed),
    )


##
# Presets.
##


def stability_first() -> TrainingFocusCfg:
  """Conservative: full stability pressure everywhere, modest reach.

  Closest to the task's behaviour before the focus layer existed.
  """
  return TrainingFocusCfg(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    backward=ChannelFocus(1.0, 0.6, SpeedProfile.constant(1.0)),
    strafe=ChannelFocus(1.0, 0.4, SpeedProfile.constant(1.0)),
    yaw=ChannelFocus(1.0, 2.0, SpeedProfile.constant(1.0)),
    speed_scale=0.5,
  )


def balanced() -> TrainingFocusCfg:
  """Stability held at walking pace, traded away approaching the frontier."""
  return TrainingFocusCfg(
    forward=ChannelFocus(1.0, 1.15, SpeedProfile.decaying(1.0, 0.5, 0.5, 1.1)),
    backward=ChannelFocus(1.0, 0.7, SpeedProfile.decaying(1.0, 0.5, 0.4, 0.7)),
    strafe=ChannelFocus(1.0, 0.5, SpeedProfile.decaying(1.0, 0.6, 0.3, 0.5)),
    yaw=ChannelFocus(1.0, 2.5, SpeedProfile.decaying(1.0, 0.6, 1.0, 2.0)),
  )


def speed_first() -> TrainingFocusCfg:
  """Reach prioritised: stability only where the robot is slow."""
  return TrainingFocusCfg(
    forward=ChannelFocus(1.0, 1.25, SpeedProfile.decaying(1.0, 0.2, 0.4, 1.0)),
    backward=ChannelFocus(1.0, 0.8, SpeedProfile.decaying(1.0, 0.2, 0.3, 0.7)),
    strafe=ChannelFocus(1.0, 0.55, SpeedProfile.decaying(1.0, 0.3, 0.25, 0.5)),
    yaw=ChannelFocus(1.0, 2.5, SpeedProfile.decaying(1.0, 0.3, 0.8, 2.0)),
    speed_scale=1.5,
  )


def forward_sprint() -> TrainingFocusCfg:
  """Forward speed above all else; lateral and yaw kept only as capability."""
  return TrainingFocusCfg(
    forward=ChannelFocus(3.0, 1.4, SpeedProfile.decaying(1.0, 0.25, 0.5, 1.2)),
    backward=ChannelFocus(0.5, 0.5, SpeedProfile.constant(1.0)),
    strafe=ChannelFocus(0.3, 0.3, SpeedProfile.constant(1.0)),
    yaw=ChannelFocus(0.5, 1.5, SpeedProfile.constant(1.0)),
    speed_scale=1.5,
  )


def strafe_first() -> TrainingFocusCfg:
  """Lateral speed above all else -- the mirror of :func:`forward_sprint`."""
  return TrainingFocusCfg(
    forward=ChannelFocus(0.5, 0.7, SpeedProfile.constant(1.0)),
    backward=ChannelFocus(0.5, 0.5, SpeedProfile.constant(1.0)),
    strafe=ChannelFocus(3.0, 0.9, SpeedProfile.decaying(1.0, 0.3, 0.3, 0.8)),
    yaw=ChannelFocus(0.5, 1.5, SpeedProfile.constant(1.0)),
    speed_scale=1.5,
  )


def low_speed_precision() -> TrainingFocusCfg:
  """Tight, dependable slow walking; no interest in the top of the range."""
  return TrainingFocusCfg(
    forward=ChannelFocus(1.0, 0.7, SpeedProfile.decaying(1.5, 0.5, 0.5, 0.8)),
    backward=ChannelFocus(1.0, 0.5, SpeedProfile.decaying(1.5, 0.5, 0.4, 0.6)),
    strafe=ChannelFocus(1.0, 0.35, SpeedProfile.decaying(1.5, 0.5, 0.25, 0.4)),
    yaw=ChannelFocus(1.0, 1.5, SpeedProfile.decaying(1.5, 0.5, 0.8, 1.5)),
  )


def high_speed_stability() -> TrainingFocusCfg:
  """Slow wobble is tolerated; hold it together once it is actually moving."""
  return TrainingFocusCfg(
    forward=ChannelFocus(1.0, 1.25, SpeedProfile.rising(0.3, 1.5, 0.5, 1.1)),
    backward=ChannelFocus(1.0, 0.7, SpeedProfile.rising(0.3, 1.5, 0.3, 0.7)),
    strafe=ChannelFocus(1.0, 0.5, SpeedProfile.rising(0.3, 1.5, 0.25, 0.5)),
    yaw=ChannelFocus(1.0, 2.5, SpeedProfile.rising(0.3, 1.5, 1.0, 2.0)),
  )


def stability_agnostic() -> TrainingFocusCfg:
  """Stability group switched off entirely; pure reach experiment."""
  return replace(speed_first(), stability_scale=0.0)


FOCUS_PRESETS: dict[str, Callable[[], TrainingFocusCfg]] = {
  "stability_first": stability_first,
  "balanced": balanced,
  "speed_first": speed_first,
  "forward_sprint": forward_sprint,
  "strafe_first": strafe_first,
  "low_speed_precision": low_speed_precision,
  "high_speed_stability": high_speed_stability,
  "stability_agnostic": stability_agnostic,
}
"""Named starting points, by intent. Resolve with :func:`get_focus_preset`."""


def get_focus_preset(name: str) -> TrainingFocusCfg:
  """Build the named preset, with a helpful error for typos."""
  if name not in FOCUS_PRESETS:
    raise KeyError(
      f"Unknown focus preset {name!r}. Available: {sorted(FOCUS_PRESETS)}."
    )
  return FOCUS_PRESETS[name]()
