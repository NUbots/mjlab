"""Tracking error over the command plane, from ``eval_competence_grid.py``.

The competence grid samples episodes over a grid of commands and shove
magnitudes and reports, per cell, how much of the command was delivered. This
script turns that into the figure the sweep plotter draws for a velocity
sweep: the commanded velocity plane, coloured by how far the robot was from
the command it was given, one panel per shove magnitude and one row per run.

**What "tracking error" means here.** The grid measures *attainment* -- the
achieved velocity projected onto the commanded direction, as a fraction of
the commanded speed -- so the recoverable error is the shortfall along the
command:

    speed_error = |c| * (1 - attain)     [m/s]

That is a real velocity error and it is the one the command asked about. It
is deliberately not a two-dimensional error: the component orthogonal to the
command is not recoverable from a projection, and reconstructing it from the
per-axis attainments would invent a number wherever an axis is commanded
below the per-axis floor. Pass ``--field shortfall`` for the dimensionless
``1 - attain`` instead, which compares cells of different speeds on equal
terms; ``speed_error`` compares them in metres per second, where a 10%
shortfall at 1.2 m/s counts for four times one at 0.3 m/s.

``--source attain_post`` reads the post-shove window instead of the whole
episode, which is the honest column to read on a shoved row: it excludes the
settling time the shove itself caused.

Usage::

    # every run under logs/eval, tracking error over the plane
    uv run python scripts/eval/plot_command_plane.py

    # one run, dimensionless, post-shove window
    uv run python scripts/eval/plot_command_plane.py \\
        --only v57 --field shortfall --source attain_post
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from figure_style import (
  INK,
  INK_2,
  MUTED,
  SEQUENTIAL,
  despine,
  plt,
  save,
  use_house_style,
)
from plot_competence_grid import Run, axis_values, load_runs, plane_cells

import mjlab

Field = Literal["speed_error", "shortfall"]
Source = Literal["attain", "attain_post"]
Statistic = Literal["median", "q25", "q75"]

FIELD_LABEL = {
  "speed_error": "Tracking error (m/s)",
  "shortfall": "Tracking shortfall (fraction of command)",
}

SOURCE_NOTE = {
  "attain": "whole episode",
  "attain_post": "post-shove window",
}

PANEL_IN = 3.1
"""Side of one square command-plane panel, in inches."""

HEADER_IN = 1.05
FOOTER_IN = 0.62


def cell_error(
  run: Run, cell: dict, field: Field, source: Source, stat: Statistic
) -> float:
  """The tracking error of one cell, or NaN where it has no evidence.

  A cell whose command is too small for attainment to be defined reports
  nothing rather than zero: the grid did not measure a shortfall there, and
  colouring it as a perfect cell would be a claim the data does not make.
  """
  if not cell.get("attain_defined", True):
    return math.nan
  attain = run.value(cell, source, stat)
  if not math.isfinite(attain):
    return math.nan
  shortfall = 1.0 - attain
  if field == "shortfall":
    return shortfall
  speed = math.hypot(cell["vx"], cell["vy"])
  return speed * shortfall


def plane_grid(
  run: Run, shove: float, field: Field, source: Source, stat: Statistic
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The ``(vx, vy)`` plane at one shove magnitude, as ``xs, ys, values``."""
  cells = [c for c in plane_cells(run) if abs(c["shove"] - shove) < 1e-9]
  xs = np.array(axis_values(cells, "vx"))
  ys = np.array(axis_values(cells, "vy"))
  values = np.full((ys.size, xs.size), np.nan)
  x_index = {v: i for i, v in enumerate(xs)}
  y_index = {v: i for i, v in enumerate(ys)}
  for cell in cells:
    values[y_index[cell["vy"]], x_index[cell["vx"]]] = cell_error(
      run, cell, field, source, stat
    )
  return xs, ys, values


def _edges(values: np.ndarray) -> np.ndarray:
  """Cell-centre positions turned into the mesh edges around them."""
  if values.size == 1:
    return np.array([values[0] - 0.5, values[0] + 0.5])
  step = np.diff(values)
  return np.concatenate(
    [[values[0] - step[0] / 2], values[:-1] + step / 2, [values[-1] + step[-1] / 2]]
  )


def _pcolor(ax, xs, ys, values, **kwargs):
  """Cell-centred mesh that shows the missing cells as missing."""
  ax.set_facecolor("#eceae4")
  return ax.pcolormesh(
    _edges(xs), _edges(ys), np.ma.masked_invalid(values), shading="flat", **kwargs
  )


def figure_command_plane(
  runs: list[Run],
  shoves: list[float],
  path: Path,
  field: Field,
  source: Source,
  stat: Statistic,
) -> None:
  """One row per run, one column per shove magnitude, one shared scale."""
  rows, cols = len(runs), len(shoves)
  height = HEADER_IN + FOOTER_IN + PANEL_IN * rows
  width = 3.0 + PANEL_IN * cols
  fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
  fig.subplots_adjust(
    left=1.45 / width,
    right=1.0 - 1.55 / width,
    top=1.0 - HEADER_IN / height,
    bottom=FOOTER_IN / height,
    wspace=0.28,
    hspace=0.32,
  )

  # One colour scale across every panel: the figure exists to compare runs
  # and shove levels, and a per-panel scale would make the worst cell of the
  # best run look like the worst cell of the worst one.
  grids = {
    (run.name, shove): plane_grid(run, shove, field, source, stat)
    for run in runs
    for shove in shoves
  }
  finite = [
    v for _, _, values in grids.values() for v in values.ravel() if np.isfinite(v)
  ]
  vmax = max(finite) if finite else 1.0
  mesh = None

  for row, run in enumerate(runs):
    for col, shove in enumerate(shoves):
      ax = axes[row][col]
      xs, ys, values = grids[(run.name, shove)]
      mesh = _pcolor(ax, xs, ys, values, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax)
      ax.set_aspect("equal", adjustable="box")
      if row == 0:
        ax.set_title(
          "undisturbed" if shove == 0.0 else f"shove {shove:g} m/s",
          color=INK,
          fontsize=10,
          pad=8,
        )
      if row == rows - 1:
        ax.set_xlabel("$v_x$ command (m/s)", color=INK_2)
      if col == 0:
        ax.set_ylabel("$v_y$ command (m/s)", color=INK_2)
      despine(ax)

  assert mesh is not None

  # Row labels are placed from the drawn axes positions, not in axes
  # coordinates: ``set_aspect("equal")`` resizes each panel's box at draw
  # time, so a label offset in axes fractions lands somewhere different for
  # every grid shape -- and off the figure entirely for a wide one.
  fig.canvas.draw()
  for row, run in enumerate(runs):
    box = axes[row][0].get_position()
    fig.text(
      box.x0 - 0.082,
      box.y0 + box.height / 2,
      run.label,
      rotation=90,
      va="center",
      ha="center",
      color=INK,
      fontsize=11,
    )

  # Placed at figure coordinates, vertically centred on the panel block, with
  # the right margin above sized to leave room for its label.
  panels_bottom = FOOTER_IN / height
  panels_top = 1.0 - HEADER_IN / height
  bar_height = 0.62 * (panels_top - panels_bottom)
  bar = fig.add_axes(
    (
      1.0 - 1.20 / width,
      panels_bottom + (panels_top - panels_bottom - bar_height) / 2,
      0.20 / width,
      bar_height,
    )
  )
  fig.colorbar(mesh, cax=bar)
  bar.set_ylabel(FIELD_LABEL[field], color=INK_2, fontsize=9)

  fig.text(
    0.055 / width,
    1.0 - 0.42 / height,
    "Tracking error over the command plane",
    color=INK,
    fontsize=13,
    ha="left",
  )
  fig.text(
    0.055 / width,
    1.0 - 0.72 / height,
    f"{stat} of {source} ({SOURCE_NOTE[source]}); blank cells were not measured",
    color=MUTED,
    fontsize=9,
    ha="left",
  )
  save(fig, path, tight=False)


@dataclass
class Args:
  input_dir: Path = Path("logs/eval")
  """Directory holding one subdirectory per run, each with a ``cells.json``."""
  only: str | None = None
  """Comma-separated run names to include. Default: every run found."""
  field: Field = "speed_error"
  """``speed_error`` (m/s along the command) or ``shortfall`` (dimensionless)."""
  source: Source = "attain"
  """``attain`` over the whole episode, or ``attain_post`` after the shove."""
  stat: Statistic = "median"
  """Which per-cell statistic to colour by: ``median``, ``q25`` or ``q75``."""
  output: Path = Path("logs/eval/figures/command_plane.pdf")
  """Where to write the figure."""


def main(args: Args) -> None:
  use_house_style()
  runs = load_runs(args.input_dir, args.only)
  shoves = sorted({cell["shove"] for run in runs for cell in plane_cells(run)})
  if not shoves:
    raise ValueError(
      "no command-plane cells found: every cell has a non-zero yaw command, "
      "so there is no (vx, vy) plane to draw."
    )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure_command_plane(runs, shoves, args.output, args.field, args.source, args.stat)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main(tyro.cli(Args, config=mjlab.TYRO_FLAGS))
