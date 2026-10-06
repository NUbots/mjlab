"""Scripted velocity-command sequences for system identification of a walk.

The data these sequences produce is fitted, per axis, with a model from commanded
body velocity ``(vx, vy, wz)`` to achieved body velocity. To fit gain, lag, delay,
dead zone, saturation and cross-axis coupling, the sequences need to excite each
of those separately:

``step``
  zero to one level and back, on one axis: gain, lag and delay at each amplitude.
``level``
  between two non-zero levels, including through zero: a change made mid-walk,
  as distinct from the start-up from a stand that ``step`` measures.
``ramp``
  a slow sweep from zero to full range: where the dead zone ends, and where the
  response saturates.
``multilevel``
  a random staircase: broadband data for fitting a dynamic model.
``chirp``
  a swept sine: the frequency response directly.
``combined``
  a step on one axis while another is held: cross-axis coupling.

Amplitudes are fractions of the training command range, which the caller passes
in, so the same sequences scale to any policy. A positive fraction is scaled by
the top of an axis's range and a negative one by the magnitude of its bottom,
which is the same thing for the symmetric ranges a velocity task trains on.

This module is numpy-only on purpose: it is the one piece that has to run off
the simulator too, to replay the exact same commands on the robot. Every
sequence is built from integer step counts at the control period, so the
commands are bit-identical wherever they are generated.

Every sequence starts with :data:`LEAD_IN_S` of zero command (standing).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

AXES: tuple[str, ...] = ("vx", "vy", "wz")
"""Command axes, in the order a command vector stores them."""

AXIS_UNITS: dict[str, str] = {"vx": "m/s", "vy": "m/s", "wz": "rad/s"}

Family = Literal["step", "level", "ramp", "multilevel", "chirp", "combined"]
Split = Literal["identification", "validation"]

LEAD_IN_S = 2.0
"""Zero command at the start of every sequence, so each starts from a stand."""

STEP_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
STEP_HOLD_S = 5.0
RETURN_S = 3.0
"""Zero command after the excitation, to see the robot come back to a stand."""

LEVEL_PAIRS: tuple[tuple[float, float], ...] = (
  (0.25, 0.75),
  (0.75, 0.25),
  (-0.25, -0.75),
  (-0.75, -0.25),
  (0.50, -0.50),
  (-0.50, 0.50),
)
LEVEL_HOLD_S = 5.0

RAMP_S = 20.0

MULTILEVEL_S = 120.0
MULTILEVEL_MAX_FRACTION = 0.80
MULTILEVEL_HOLD_RANGE_S = (0.5, 3.0)
IDENTIFICATION_SEED = 1000
"""Base seed of the identification staircases; axis ``i`` uses ``seed + i``."""
VALIDATION_SEED = 2000
"""Base seed of the held-out staircases. Distinct from every identification seed."""

CHIRP_FRACTION = 0.50
CHIRP_F0_HZ = 0.05
CHIRP_F1_HZ = 2.0
CHIRP_S = 60.0

COMBINED_PAIRS: tuple[tuple[str, str], ...] = (
  ("vx", "wz"),
  ("wz", "vx"),
  ("vy", "vx"),
)
"""``(stepped axis, held axis)`` pairs of the combined-axis runs."""
COMBINED_HOLD_FRACTIONS: tuple[float, ...] = (0.50, -0.50)
COMBINED_STEP_FRACTIONS: tuple[float, ...] = (
  0.25,
  0.50,
  0.75,
  1.00,
  -0.25,
  -0.50,
  -0.75,
  -1.00,
)
COMBINED_SETTLE_S = 4.0
"""Held axis alone, before the step, so the step lands on a settled gait."""
COMBINED_STEP_S = 5.0
COMBINED_BACK_S = 3.0
"""Stepped axis back at zero with the held axis still held."""

Ranges = dict[str, tuple[float, float]]
"""Training command range per axis, ``{"vx": (low, high), ...}``."""


@dataclass(frozen=True)
class CommandSequence:
  """One scripted command run.

  Attributes:
    name: Unique, filesystem-safe identifier.
    family: Which kind of excitation this is; see the module docstring.
    split: ``"validation"`` for the held-out staircases, else
      ``"identification"``.
    dt: Control period, in seconds. Row ``k`` of :attr:`commands` is in force
      over ``[k * dt, (k + 1) * dt)``.
    commands: Shape ``(T, 3)`` absolute commands, ordered as :data:`AXES`.
    params: How the sequence was built, as fractions of range and as absolute
      values. JSON-serialisable.
  """

  name: str
  family: Family
  split: Split
  dt: float
  commands: np.ndarray
  params: dict[str, Any] = field(default_factory=dict)

  @property
  def num_steps(self) -> int:
    return int(self.commands.shape[0])

  @property
  def duration(self) -> float:
    return self.num_steps * self.dt

  def times(self) -> np.ndarray:
    return np.arange(self.num_steps) * self.dt

  def metadata(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "family": self.family,
      "split": self.split,
      "dt_s": self.dt,
      "num_steps": self.num_steps,
      "duration_s": self.duration,
      "lead_in_s": LEAD_IN_S,
      "params": self.params,
    }


def scale(fraction: float, axis: str, ranges: Ranges) -> float:
  """Absolute command for a fraction of an axis's training range."""
  low, high = ranges[axis]
  return fraction * high if fraction >= 0.0 else fraction * abs(low)


def _steps(seconds: float, dt: float) -> int:
  steps = round(seconds / dt)
  if abs(steps * dt - seconds) > 1e-9:
    raise ValueError(f"{seconds} s is not a whole number of {dt} s control steps")
  return steps


def _vector(**values: float) -> np.ndarray:
  return np.array([values.get(axis, 0.0) for axis in AXES], dtype=np.float64)


def _hold(seconds: float, dt: float, **values: float) -> np.ndarray:
  return np.tile(_vector(**values), (_steps(seconds, dt), 1))


def _lead_in(dt: float) -> np.ndarray:
  return _hold(LEAD_IN_S, dt)


def _tag(fraction: float) -> str:
  """``0.25 -> "+025"``: a signed percentage that sorts and globs cleanly."""
  return f"{round(fraction * 100):+04d}"


def _level(axis: str, fraction: float, ranges: Ranges) -> dict[str, Any]:
  return {
    "axis": axis,
    "fraction": fraction,
    "value": scale(fraction, axis, ranges),
    "units": AXIS_UNITS[axis],
  }


def step_sequences(ranges: Ranges, dt: float) -> list[CommandSequence]:
  out = []
  for axis in AXES:
    for magnitude in STEP_FRACTIONS:
      for fraction in (magnitude, -magnitude):
        value = scale(fraction, axis, ranges)
        commands = np.concatenate(
          [
            _lead_in(dt),
            _hold(STEP_HOLD_S, dt, **{axis: value}),
            _hold(RETURN_S, dt),
          ]
        )
        out.append(
          CommandSequence(
            name=f"step_{axis}_{_tag(fraction)}",
            family="step",
            split="identification",
            dt=dt,
            commands=commands,
            params={
              **_level(axis, fraction, ranges),
              "onset_s": LEAD_IN_S,
              "hold_s": STEP_HOLD_S,
              "return_s": RETURN_S,
            },
          )
        )
  return out


def level_sequences(ranges: Ranges, dt: float) -> list[CommandSequence]:
  out = []
  for axis in AXES:
    for first, second in LEVEL_PAIRS:
      a, b = scale(first, axis, ranges), scale(second, axis, ranges)
      commands = np.concatenate(
        [
          _lead_in(dt),
          _hold(LEVEL_HOLD_S, dt, **{axis: a}),
          _hold(LEVEL_HOLD_S, dt, **{axis: b}),
          _hold(RETURN_S, dt),
        ]
      )
      out.append(
        CommandSequence(
          name=f"level_{axis}_{_tag(first)}_to_{_tag(second)}",
          family="level",
          split="identification",
          dt=dt,
          commands=commands,
          params={
            "axis": axis,
            "units": AXIS_UNITS[axis],
            "from_fraction": first,
            "to_fraction": second,
            "from_value": a,
            "to_value": b,
            "first_onset_s": LEAD_IN_S,
            "level_change_s": LEAD_IN_S + LEVEL_HOLD_S,
            "hold_s": LEVEL_HOLD_S,
            "return_s": RETURN_S,
          },
        )
      )
  return out


def ramp_sequences(ranges: Ranges, dt: float) -> list[CommandSequence]:
  out = []
  n = _steps(RAMP_S, dt)
  for axis in AXES:
    for fraction in (1.0, -1.0):
      end = scale(fraction, axis, ranges)
      # Row k holds the value the ramp has reached at its start, so the last row
      # is one step short of ``end`` and the ramp's slope is exactly end / RAMP_S.
      ramp = np.zeros((n, 3))
      ramp[:, AXES.index(axis)] = end * np.arange(n) / n
      commands = np.concatenate([_lead_in(dt), ramp, _hold(RETURN_S, dt)])
      out.append(
        CommandSequence(
          name=f"ramp_{axis}_{_tag(fraction)}",
          family="ramp",
          split="identification",
          dt=dt,
          commands=commands,
          params={
            "axis": axis,
            "units": AXIS_UNITS[axis],
            "end_fraction": fraction,
            "end_value": end,
            "start_s": LEAD_IN_S,
            "ramp_s": RAMP_S,
            "slope_per_s": end / RAMP_S,
            "return_s": RETURN_S,
          },
        )
      )
  return out


def multilevel_sequence(
  axis: str, ranges: Ranges, dt: float, seed: int, split: Split
) -> CommandSequence:
  """A staircase of random levels held for random times.

  Levels are uniform in ``+-MULTILEVEL_MAX_FRACTION`` of range and hold times
  uniform in :data:`MULTILEVEL_HOLD_RANGE_S`, rounded to whole control steps.
  The last level is cut short so the staircase lasts exactly
  :data:`MULTILEVEL_S`.
  """
  rng = np.random.default_rng(seed)
  total = _steps(MULTILEVEL_S, dt)
  low, high = MULTILEVEL_HOLD_RANGE_S
  values = np.zeros(total)
  levels: list[dict[str, float]] = []
  k = 0
  while k < total:
    fraction = float(rng.uniform(-MULTILEVEL_MAX_FRACTION, MULTILEVEL_MAX_FRACTION))
    hold = max(1, round(float(rng.uniform(low, high)) / dt))
    hold = min(hold, total - k)
    value = scale(fraction, axis, ranges)
    values[k : k + hold] = value
    levels.append(
      {
        "start_s": LEAD_IN_S + k * dt,
        "hold_s": hold * dt,
        "fraction": fraction,
        "value": value,
      }
    )
    k += hold
  staircase = np.zeros((total, 3))
  staircase[:, AXES.index(axis)] = values
  commands = np.concatenate([_lead_in(dt), staircase, _hold(RETURN_S, dt)])
  return CommandSequence(
    name=f"multilevel_{axis}_seed{seed}",
    family="multilevel",
    split=split,
    dt=dt,
    commands=commands,
    params={
      "axis": axis,
      "units": AXIS_UNITS[axis],
      "seed": seed,
      "rng": "numpy.random.default_rng (PCG64)",
      "max_fraction": MULTILEVEL_MAX_FRACTION,
      "hold_range_s": list(MULTILEVEL_HOLD_RANGE_S),
      "start_s": LEAD_IN_S,
      "staircase_s": MULTILEVEL_S,
      "return_s": RETURN_S,
      "levels": levels,
    },
  )


def multilevel_sequences(
  ranges: Ranges, dt: float, seed: int, split: Split
) -> list[CommandSequence]:
  return [
    multilevel_sequence(axis, ranges, dt, seed + index, split)
    for index, axis in enumerate(AXES)
  ]


def chirp_sequences(ranges: Ranges, dt: float) -> list[CommandSequence]:
  """Linear chirps, ``A sin(2 pi (f0 t + (f1 - f0) t^2 / (2 T))))``.

  The sweep starts at zero phase, so it leaves the stand without a jump, and it
  ends on a whole number of half cycles, so it returns to zero without one
  either for the default parameters.
  """
  out = []
  n = _steps(CHIRP_S, dt)
  t = np.arange(n) * dt
  phase = (
    2.0
    * math.pi
    * (CHIRP_F0_HZ * t + (CHIRP_F1_HZ - CHIRP_F0_HZ) * t**2 / (2 * CHIRP_S))
  )
  wave = np.sin(phase)
  for axis in AXES:
    up = scale(CHIRP_FRACTION, axis, ranges)
    down = scale(-CHIRP_FRACTION, axis, ranges)
    sweep = np.zeros((n, 3))
    sweep[:, AXES.index(axis)] = np.where(wave >= 0.0, wave * up, -wave * down)
    commands = np.concatenate([_lead_in(dt), sweep, _hold(RETURN_S, dt)])
    out.append(
      CommandSequence(
        name=f"chirp_{axis}",
        family="chirp",
        split="identification",
        dt=dt,
        commands=commands,
        params={
          "axis": axis,
          "units": AXIS_UNITS[axis],
          "fraction": CHIRP_FRACTION,
          "amplitude_positive": up,
          "amplitude_negative": down,
          "f0_hz": CHIRP_F0_HZ,
          "f1_hz": CHIRP_F1_HZ,
          "sweep": "linear",
          "start_s": LEAD_IN_S,
          "sweep_s": CHIRP_S,
          "return_s": RETURN_S,
          "instantaneous_frequency_hz": "f0 + (f1 - f0) * (t - start_s) / sweep_s",
        },
      )
    )
  return out


def combined_sequences(ranges: Ranges, dt: float) -> list[CommandSequence]:
  out = []
  for stepped, held in COMBINED_PAIRS:
    for hold_fraction in COMBINED_HOLD_FRACTIONS:
      hold_value = scale(hold_fraction, held, ranges)
      for step_fraction in COMBINED_STEP_FRACTIONS:
        step_value = scale(step_fraction, stepped, ranges)
        commands = np.concatenate(
          [
            _lead_in(dt),
            _hold(COMBINED_SETTLE_S, dt, **{held: hold_value}),
            _hold(COMBINED_STEP_S, dt, **{held: hold_value, stepped: step_value}),
            _hold(COMBINED_BACK_S, dt, **{held: hold_value}),
            _hold(RETURN_S, dt),
          ]
        )
        step_onset = LEAD_IN_S + COMBINED_SETTLE_S
        out.append(
          CommandSequence(
            name=(
              f"combined_{stepped}_{_tag(step_fraction)}"
              f"_hold_{held}_{_tag(hold_fraction)}"
            ),
            family="combined",
            split="identification",
            dt=dt,
            commands=commands,
            params={
              "stepped": _level(stepped, step_fraction, ranges),
              "held": _level(held, hold_fraction, ranges),
              "hold_onset_s": LEAD_IN_S,
              "step_onset_s": step_onset,
              "step_off_s": step_onset + COMBINED_STEP_S,
              "hold_off_s": step_onset + COMBINED_STEP_S + COMBINED_BACK_S,
              "return_s": RETURN_S,
            },
          )
        )
  return out


FAMILIES: tuple[Family, ...] = (
  "step",
  "level",
  "ramp",
  "multilevel",
  "chirp",
  "combined",
)


def build_sequences(
  ranges: Ranges,
  dt: float,
  families: tuple[Family, ...] = FAMILIES,
  identification_seed: int = IDENTIFICATION_SEED,
  validation_seed: int = VALIDATION_SEED,
) -> list[CommandSequence]:
  """Every sequence of the requested families.

  ``multilevel`` yields both the identification staircases and the held-out
  validation ones.
  """
  for axis in AXES:
    if axis not in ranges:
      raise ValueError(f"no training range for axis {axis!r}")
  if {identification_seed + i for i in range(len(AXES))} & {
    validation_seed + i for i in range(len(AXES))
  }:
    raise ValueError("identification and validation seeds overlap")
  out: list[CommandSequence] = []
  for family in families:
    if family == "step":
      out += step_sequences(ranges, dt)
    elif family == "level":
      out += level_sequences(ranges, dt)
    elif family == "ramp":
      out += ramp_sequences(ranges, dt)
    elif family == "multilevel":
      out += multilevel_sequences(ranges, dt, identification_seed, "identification")
      out += multilevel_sequences(ranges, dt, validation_seed, "validation")
    elif family == "chirp":
      out += chirp_sequences(ranges, dt)
    elif family == "combined":
      out += combined_sequences(ranges, dt)
    else:
      raise ValueError(f"unknown family {family!r}")
  names = [sequence.name for sequence in out]
  if len(set(names)) != len(names):
    raise ValueError("sequence names are not unique")
  return out


def write_command_csv(sequence: CommandSequence, path: Path) -> None:
  """The commands alone, for replay on the robot: ``t, vx, vy, wz``."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz"])
    for t, row in zip(sequence.times(), sequence.commands, strict=True):
      writer.writerow([f"{t:.6f}", *(repr(float(value)) for value in row)])


def export_commands(
  sequences: list[CommandSequence], ranges: Ranges, out_dir: Path
) -> None:
  """Write every sequence as a command CSV plus one JSON index.

  Layout is ``<out_dir>/<split>/<family>/<name>.csv``, the same tree the
  simulator data uses.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  index = []
  for sequence in sequences:
    relative = Path(sequence.split) / sequence.family / f"{sequence.name}.csv"
    write_command_csv(sequence, out_dir / relative)
    index.append({**sequence.metadata(), "file": str(relative)})
  (out_dir / "sequences.json").write_text(
    json.dumps(
      {
        "command_ranges": {axis: list(ranges[axis]) for axis in AXES},
        "units": AXIS_UNITS,
        "row_semantics": "row k is the command in force over [t_k, t_k + dt)",
        "sequences": index,
      },
      indent=1,
    )
  )
