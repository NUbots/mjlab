"""One profile strip stitched from two motion-capture logs.

A one-off for a capture that had to be flown twice. The schedule is twelve
phases; the first attempt went down partway through the yaw stage, so the run
was repeated, and between them the two logs hold the whole thing exactly once:

``run2_trimmed.json``
  the head -- the sagittal and lateral stages, ``vx±0.35`` and ``vy±0.20``.
``run1_trimmed.json``
  the rest -- the yaw stage and the six combined phases.

  uv run python scripts/eval/plot_mocap_profile_pair.py \\
    --head logs/eval/run2_trimmed.json \\
    --tail logs/eval/run1_trimmed.json

Everything except the drawing is imported from ``plot_mocap_profile``, which
this is deliberately not a copy of: reading, the frame calibration, the stride
measurement and the off-its-feet test are the parts that must not drift between
the two scripts, and the only thing that genuinely differs is that the phases
come from two runs instead of one.

The two captures are calibrated separately, because they are separate captures
-- the volume was not recalibrated between them, but the tracked body's pose
when Motive built it is not something to assume. They are drawn with one fit
window, measured from the tail run, so the two halves of the strip are the same
measurement rather than two.

A stitch is not a change of command, it is a change of *run*, so the seam is
drawn heavier than a phase boundary and named in the key. Nothing carries
across it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
  AXIS_COLOUR,
  AXIS_LABEL,
  BASELINE,
  GRID,
  INK,
  INK_2,
  RED,
  despine,
  save,
  use_house_style,
)
from matplotlib.lines import Line2D
from plot_mocap_profile import (
  AXIS_INDEX,
  RAW_WINDOW_S,
  Run,
  _phases,
  build_run,
  calibrate,
  read_log,
)

import mjlab


@dataclass
class Segment:
  """One commanded phase, and which of the two captures it came from."""

  run: Run
  source: str
  start: float
  end: float
  driving: dict[str, float]


def stitch(head: Run, tail: Run, head_phases: int | None) -> list[Segment]:
  """The head run's phases, then the tail run's.

  ``head_phases`` truncates the head, for a capture whose tail overlaps the
  other log's head. Left alone, every phase of both runs is drawn.
  """
  segments = []
  for run, source, limit in ((head, "head", head_phases), (tail, "tail", None)):
    for start, end, driving in _phases(run)[:limit]:
      segments.append(Segment(run, source, start, end, driving))
  return segments


def figure_profile(segments: list[Segment], label: str, path: Path) -> None:
  """The profile strip, over phases drawn from two captures.

  The same figure ``plot_mocap_profile`` draws, with one addition: the boundary
  between the two runs is a heavier line than the boundaries between phases,
  because a reader has to know the strip is stitched.
  """
  if not segments:
    raise SystemExit("no commanded motion in either capture; nothing to draw")

  fig, ax = plt.subplots(figsize=(16, 4.6))
  despine(ax)

  span = max(
    0.5,
    1.7 * max(abs(v) for segment in segments for v in segment.driving.values()),
  )

  offset = 0.0
  boundaries: list[float] = []
  seams: list[float] = []
  for order, segment in enumerate(segments):
    run = segment.run
    window = (run.t >= segment.start) & (run.t <= segment.end)
    time = run.t[window] - segment.start + offset
    width = segment.end - segment.start
    off = run.off_feet[window]

    if order % 2:
      ax.axvspan(offset, offset + width, color=GRID, alpha=0.28, zorder=0)
    if order and segment.source != segments[order - 1].source:
      seams.append(offset)

    for name in segment.driving:
      column = AXIS_INDEX[name]
      colour = AXIS_COLOUR[name]
      commanded = run.command[window, column]
      smoothed = np.where(off, np.nan, run.smooth[window, column])
      raw = np.where(off, np.nan, run.raw[window, column])
      ax.plot(
        time,
        np.clip(raw, -span, span),
        color=colour,
        linewidth=0.4,
        alpha=0.13,
        zorder=2,
      )
      ax.plot(time, commanded, color=colour, linewidth=2.0, zorder=3)
      ax.plot(
        time,
        np.clip(smoothed, -span, span),
        color=colour,
        linewidth=0.8,
        linestyle=(0, (2, 1)),
        alpha=0.6,
        zorder=4,
        solid_capstyle="round",
      )

    if run.fall_t is not None and segment.start <= run.fall_t <= segment.end:
      at = run.fall_t - segment.start + offset
      ax.axvspan(at, offset + width, color=RED, alpha=0.09, zorder=1)
      ax.axvline(at, color=RED, linewidth=1.2, linestyle=":", zorder=5)
      ax.annotate(
        f"fell {run.fall_t - segment.start:.1f} s in",
        xy=(at, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color=RED,
        fontsize=7.5,
        fontweight="bold",
        zorder=6,
      )

    offset += width
    boundaries.append(offset)

  for edge in boundaries[:-1]:
    ax.axvline(edge, color=BASELINE, linewidth=1.0, zorder=5)
  for edge in seams:
    ax.axvline(edge, color=INK, linewidth=1.6, zorder=6)
  ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
  ax.set_ylim(-span, span)
  ax.set_xlim(0.0, offset)
  ax.margins(x=0)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("velocity (m/s) · yaw rate (rad/s)")

  smooth_s = segments[0].run.smooth_s
  handles = [
    Line2D(
      [],
      [],
      color=INK_2,
      linewidth=0.8,
      linestyle=(0, (2, 1)),
      alpha=0.6,
      label=f"measured ({smooth_s:.2f} s fit)",
    ),
    Line2D(
      [],
      [],
      color=INK_2,
      linewidth=0.7,
      alpha=0.35,
      label=f"measured ({RAW_WINDOW_S:.2f} s fit)",
    ),
    Line2D([], [], color=INK_2, linewidth=2.0, label="commanded"),
  ]
  handles += [
    Line2D(
      [],
      [],
      color=AXIS_COLOUR[a],
      marker="o",
      markersize=6.5,
      linestyle="none",
      label=f"{AXIS_LABEL[a]} axis",
    )
    for a in ("vx", "vy", "wz")
  ]
  if seams:
    handles.append(Line2D([], [], color=INK, linewidth=1.6, label="second capture"))
  fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    ncol=len(handles),
    columnspacing=1.6,
  )
  fig.suptitle(
    f"Velocity tracking under a moving command — {label}",
    x=0.006,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.tight_layout(rect=(0, 0.0, 1, 0.93))
  save(fig, path)


@dataclass
class Args:
  head: Path
  """Capture supplying the opening phases of the schedule."""
  tail: Path
  """Capture supplying the rest of it."""
  label: str = "RL Profilewalk Mocap Results"
  """What the figure calls the pair."""
  output_dir: Path | None = None
  """Where the figure goes. Defaults to ``<head-directory>``."""
  name: str = "combined"
  """Slug for the output filename."""
  head_phases: int | None = None
  """Draw only this many phases from the head capture. All of them by default."""
  smooth: float | None = None
  """Window both captures are fitted over, in seconds.

  Defaults to the stride measured from the tail capture, applied to both, so
  the two halves of the strip are the same measurement."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()

  for path in (args.head, args.tail):
    if not path.is_file():
      raise SystemExit(f"no such log: {path}")

  head_log = read_log(args.head)
  head_frame, _ = calibrate(head_log, None, None, None)
  tail_log = read_log(args.tail)
  tail_frame, _ = calibrate(tail_log, None, None, None)

  # One window for both halves, so the strip is a single measurement rather
  # than two that happen to sit next to each other.
  window = args.smooth
  if window is None:
    window = build_run(tail_log, tail_frame).smooth_s
  head = build_run(head_log, head_frame, window)
  tail = build_run(tail_log, tail_frame, window)

  segments = stitch(head, tail, args.head_phases)
  out = args.output_dir or args.head.parent

  print()
  for role, run, path in (("head", head, args.head), ("tail", tail, args.tail)):
    drawn = [s for s in segments if s.source == role]
    print(f"{role:<5} {path.name}")
    print(
      f"      {run.t[-1] - run.t[0]:.1f} s, {len(run.t)} frames, "
      f"{len(drawn)} of {len(_phases(run))} phases drawn"
    )
    for note in run.frame.notes:
      print(f"      {note}")
    if run.fall_t is not None:
      print(f"      fell at {run.fall_t:.1f} s")
  print(f"\nvelocity window   : {window:.2f} s, both captures")
  print(
    "phases            : "
    + "   ".join(
      " ".join(f"{n}{v:+.2f}" for n, v in s.driving.items()) for s in segments
    )
  )

  figure_profile(segments, args.label, out / f"fig1_mocap_profile_{args.name}")


if __name__ == "__main__":
  main()
