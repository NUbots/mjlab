"""Figures for the command x disturbance competence grid.

Reads what ``eval_competence_grid.py`` wrote and draws it. Nothing is computed
here that the eval could have computed: ``cells.json`` already carries each
cell's quartiles, its fall rate and its binomial interval, and this file's job
is to choose what to put next to what.

  uv run python scripts/eval/plot_competence_grid.py --input-dir logs/eval

Every subdirectory holding a ``cells.json`` is a run, and any number of them is
drawn. ``--runs a,b`` narrows and reorders the set; a run keeps its colour from
the full comparison so a narrowed figure and the whole one can sit on a page
together.

Figures land in ``<input-dir>/figures`` as PNG (300 dpi) and PDF:

* ``<run>_envelope`` -- the headline. One row per quantity, one column per shove
  bin, each panel the commanded velocity plane. This is the envelope: read down
  a column for what one disturbance level costs, across a row for how the cost
  grows with the shove. Every row's ramp is oriented so that darker is better,
  falls included, so the trouble is wherever the grid goes pale. The wobble row
  reports the lead time a fall was given, so it is hatched wherever nothing
  fell.
* ``<run>_spread`` -- the same grid showing the interquartile range instead of
  the median, because the interesting cells are the high-variance ones and a
  median cannot show that.
* ``<run>_axes`` -- signed per-axis attainment, diverging about 1.0. Asked for
  lateral and delivered forward is a distinct failure the scalar hides.
* ``<run>_yaw`` -- the yaw slice, which is one dimensional in command and so
  does not fit the plane panels.
* ``curves_<quantity>`` -- the same numbers against shove magnitude, one panel
  per commanded velocity, one line per run with its interquartile band. The
  heatmaps find the interesting cell; these read what happens inside it.
* ``difference`` -- drawn only for a pair of runs: the second minus the first,
  diverging about zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
  DIVERGING,
  GRID,
  INK,
  INK_2,
  MUTED,
  PALETTE,
  SEQUENTIAL,
  SURFACE,
  despine,
  hide,
  save,
  use_house_style,
)
from matplotlib.patches import Rectangle

import mjlab

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
  """One evaluated policy: its cells, and what to call it."""

  name: str
  colour: str
  cells: list[dict]
  meta: dict

  @property
  def label(self) -> str:
    return f"{self.name}"

  def value(self, cell: dict | None, quantity: str, statistic: str) -> float:
    """One number out of one cell, or NaN where the cell has no evidence."""
    if cell is None:
      return float("nan")
    if quantity == "fell":
      if statistic == "median":
        return cell["fell_rate"]
      return cell["fell_ci_high"] - cell["fell_ci_low"]
    if quantity not in cell:
      raise KeyError(
        f"run {self.name!r} has no {quantity!r}: it was collected before that "
        "quantity existed. Re-run eval_competence_grid.py for it -- the "
        "episodes.csv beside it cannot be re-summarised, since cells.json is "
        "what this reads."
      )
    return cell[quantity][statistic]

  def find(self, vx: float, vy: float, wz: float, shove: float) -> dict | None:
    for cell in self.cells:
      if (
        _close(cell["vx"], vx)
        and _close(cell["vy"], vy)
        and _close(cell["wz"], wz)
        and _close(cell["shove"], shove)
      ):
        return cell
    return None


def _close(a: float, b: float) -> bool:
  return abs(a - b) < 1e-6


def load_runs(input_dir: Path, wanted: str | None) -> list[Run]:
  """Every subdirectory holding a ``cells.json``, in name order.

  Colours are handed out over the *full* set before any narrowing, so
  ``--runs`` changes which runs are drawn and never what colour they are.
  """
  found = sorted(path.parent for path in input_dir.glob("*/cells.json"))
  if not found:
    raise FileNotFoundError(
      f"no run in {input_dir}: expected <input-dir>/<tag>/cells.json, as "
      "eval_competence_grid.py writes"
    )
  runs = []
  for index, directory in enumerate(found):
    with (directory / "cells.json").open() as handle:
      summary = json.load(handle)
    runs.append(
      Run(
        name=directory.name,
        colour=PALETTE[index % len(PALETTE)],
        cells=summary["cells"],
        meta=summary.get("run", {}),
      )
    )
  if wanted is None:
    return runs
  by_name = {run.name: run for run in runs}
  missing = [name for name in wanted.split(",") if name not in by_name]
  if missing:
    raise KeyError(f"no such run: {', '.join(missing)}; have {list(by_name)}")
  return [by_name[name] for name in wanted.split(",")]


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------


def axis_values(cells: list[dict], key: str) -> list[float]:
  return sorted({cell[key] for cell in cells})


def plane_cells(run: Run) -> list[dict]:
  """The commanded velocity plane: everything at zero yaw rate."""
  return [cell for cell in run.cells if _close(cell["wz"], 0.0)]


def yaw_cells(run: Run) -> list[dict]:
  """The yaw slice: one forward speed, a range of yaw rates."""
  return [cell for cell in run.cells if not _close(cell["wz"], 0.0)]


QUANTITIES: tuple[tuple[str, str, str, bool], ...] = (
  ("attain", "Attainment", "delivered / commanded", True),
  ("wobble_lead", "Wobble lead", "seconds from 25 deg to the fall", True),
  ("fell", "fall rate", "episodes ending in a fall", False),
  ("ep_len_frac", "Survival", "ep. length / maximum", True),
)
"""What to draw, in the order the panels stack: the headline, the warning the
near-miss channel gave before the failure, the binary, and the survival that
disambiguates a low attainment from an early termination.

Wobble lead is measured per fall, not per episode: seconds from the first tilt
past 25 degrees to the termination. More of it is better -- a robot that fought
for a second before going over gave a behaviour tree a second to react, and one
that snapped over in two control steps gave it nothing. It is undefined
wherever nothing fell, so that row is hatched exactly where the fall-rate row
below it reads zero.

The fourth field says whether more of the quantity is better. It sets which way
up the ramp goes on the median heatmaps, so that dark reads as *good* in every
row rather than as *more*: a reader scanning four rows at once should be able to
find the trouble by looking for the pale corner, without stopping to remember
which two rows invert. The colourbars carry the reversal for anyone reading a
row on its own."""


def ramp(higher_is_better: bool):
  """The sequential ramp, oriented so its dark end is the good end."""
  return SEQUENTIAL if higher_is_better else SEQUENTIAL.reversed()


SPREAD_LABEL = {
  "attain": "IQR of attainment",
  "wobble_lead": "IQR of wobble lead",
  "fell": "width of the 95% interval",
  "ep_len_frac": "IQR of survival",
}


def plane_array(
  run: Run,
  quantity: str,
  statistic: str,
  shove: float,
  vxs: list[float],
  vys: list[float],
) -> np.ndarray:
  """Shape ``(len(vys), len(vxs))`` of one quantity over the plane."""
  grid = np.full((len(vys), len(vxs)), np.nan)
  for row, vy in enumerate(vys):
    for column, vx in enumerate(vxs):
      grid[row, column] = run.value(run.find(vx, vy, 0.0, shove), quantity, statistic)
  return grid


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def note(fig, text: str) -> None:
  """A caption under the panels.

  Attached as the figure's shared x label so the constrained layout reserves
  room for it, rather than as free text that lands on top of the tick labels
  whenever the panels are short.
  """
  fig.supxlabel(text, fontsize=7.5, color=MUTED, x=0.01, ha="left", wrap=True)


def min_episodes(cells: list[dict]) -> int:
  """Episodes behind the worst-covered cell drawn."""
  return min((cell["episodes"] for cell in cells), default=0)


def diverging_span(values: list[np.ndarray], centre: float) -> float:
  """Half-width of a diverging scale about ``centre``.

  The 98th percentile of the deviation rather than the largest one. A
  diverging map has to stay symmetric about its neutral to mean anything, so a
  single cell where the robot delivered a tenth of what was asked would
  otherwise set both limits and leave every other cell in the middle third of
  the ramp. The colourbar is drawn with both ends extended, which is what says
  the outliers were clipped rather than absent.
  """
  finite = np.concatenate(
    [grid[np.isfinite(grid)].ravel() for grid in values] + [np.zeros(0)]
  )
  if not finite.size:
    return 1.0
  return max(float(np.percentile(np.abs(finite - centre), 98.0)), 1e-6)


def draw_heatmap(
  ax,
  values: np.ndarray,
  x_ticks: list[float],
  y_ticks: list[float],
  cmap,
  vmin: float,
  vmax: float,
):
  """One panel of a heatmap grid, with the undefined cells struck out.

  A cell with no evidence is not a zero. Attainment is not sampled at all below
  a commanded speed of 0.15 m/s, so those cells are hatched rather than
  coloured: painting them at the bottom of the ramp would read as the worst
  sandbagging on the grid, which is the opposite of what happened.
  """
  masked = np.ma.masked_invalid(values)
  image = ax.imshow(
    masked,
    origin="lower",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
    interpolation="nearest",
    extent=(-0.5, len(x_ticks) - 0.5, -0.5, len(y_ticks) - 0.5),
  )
  for row in range(values.shape[0]):
    for column in range(values.shape[1]):
      if np.isnan(values[row, column]):
        ax.add_patch(
          Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor=SURFACE,
            # Matplotlib draws a hatch in the patch's edge colour, so the edge
            # cannot be the surface gap; the separators come from the minor
            # grid instead, which this has to sit under. The house style puts
            # the grid at 0.5 and patches default to 1.
            edgecolor=GRID,
            hatch="///",
            linewidth=0.0,
            zorder=0.25,
          )
        )
  # A surface-coloured gap between cells, so adjacent values read as separate
  # marks rather than as one continuous field.
  ax.set_xticks(np.arange(len(x_ticks)))
  ax.set_yticks(np.arange(len(y_ticks)))
  ax.set_xticks(np.arange(len(x_ticks) + 1) - 0.5, minor=True)
  ax.set_yticks(np.arange(len(y_ticks) + 1) - 0.5, minor=True)
  ax.grid(which="minor", color=SURFACE, linewidth=1.5)
  ax.grid(which="major", visible=False)
  ax.tick_params(which="minor", length=0)
  ax.tick_params(which="major", length=0)
  # Seven values across a narrow panel collide read horizontally.
  ax.set_xticklabels([f"{value:g}" for value in x_ticks], rotation=90, fontsize=7)
  ax.set_yticklabels([f"{value:g}" for value in y_ticks], fontsize=7)
  for spine in ax.spines.values():
    hide(spine)
  return image


def plane_figure(
  run: Run,
  statistic: str,
  path: Path,
) -> None:
  """One row per quantity, one column per shove bin, each panel the plane."""
  cells = plane_cells(run)
  vxs = axis_values(cells, "vx")
  vys = axis_values(cells, "vy")
  shoves = axis_values(cells, "shove")

  fig, axes = plt.subplots(
    len(QUANTITIES),
    len(shoves),
    figsize=(1.55 * len(shoves) + 1.9, 1.7 * len(QUANTITIES) + 0.9),
    squeeze=False,
    layout="constrained",
  )
  for row, (quantity, title, unit, higher_is_better) in enumerate(QUANTITIES):
    grids = [plane_array(run, quantity, statistic, shove, vxs, vys) for shove in shoves]
    finite = np.concatenate([grid[np.isfinite(grid)].ravel() for grid in grids])
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax - vmin < 1e-9:
      vmax = vmin + 1e-9
    # Spread is spread: a wide band is not "bad" the way a fall is, and all
    # four rows already read the same way, so only the medians are oriented.
    cmap = ramp(higher_is_better) if statistic == "median" else SEQUENTIAL
    image = None
    for column, (shove, grid) in enumerate(zip(shoves, grids, strict=True)):
      ax = axes[row][column]
      image = draw_heatmap(ax, grid, vxs, vys, cmap, vmin, vmax)
      if row == 0:
        ax.set_title(f"$|\\Delta v|$ = {shove:g} m/s", pad=6)
      if column == 0:
        ax.set_ylabel("$v_y$ (m/s)")
      else:
        ax.set_yticklabels([])
      if row == len(QUANTITIES) - 1:
        ax.set_xlabel("$v_x$ (m/s)")
      else:
        ax.set_xticklabels([])
    label = unit if statistic == "median" else SPREAD_LABEL[quantity]
    assert image is not None
    bar = fig.colorbar(image, ax=axes[row], fraction=0.02, pad=0.015)
    hide(bar.outline)
    bar.ax.tick_params(length=0, labelsize=7, colors=MUTED)
    bar.set_label(f"{title}\n{label}", fontsize=8, color=INK_2, labelpad=6)

  fig.suptitle(
    f"{run.label} — Competence Envelope",
    fontsize=11,
    color=INK,
    fontweight="semibold",
    x=0.5,
    ha="center",
  )
  scale = "Darker is better" if statistic == "median" else "Darker is a wider spread"
  note(
    fig,
    f"{scale}. Hatched: nothing to measure -- no attainment sample below a "
    "commanded 0.15 m/s, no wobble lead where nothing fell.",
  )
  save(fig, path)


def axes_figure(run: Run, path: Path) -> None:
  """Signed per-axis attainment, diverging about 1.0.

  One is delivering what was asked; below is undershoot, and negative is
  travelling the other way. A row is undefined wherever the command asks for
  under 0.10 m/s on that axis, which is by construction the whole of the
  orthogonal single-axis column.
  """
  cells = plane_cells(run)
  vxs = axis_values(cells, "vx")
  vys = axis_values(cells, "vy")
  shoves = axis_values(cells, "shove")

  fig, axes = plt.subplots(
    2,
    len(shoves),
    figsize=(1.55 * len(shoves) + 1.9, 4.3),
    squeeze=False,
    layout="constrained",
  )
  span = diverging_span(
    [
      plane_array(run, quantity, "median", shove, vxs, vys)
      for quantity in ("attain_x", "attain_y")
      for shove in shoves
    ],
    centre=1.0,
  )

  for row, quantity in enumerate(("attain_x", "attain_y")):
    image = None
    for column, shove in enumerate(shoves):
      ax = axes[row][column]
      grid = plane_array(run, quantity, "median", shove, vxs, vys)
      image = draw_heatmap(ax, grid, vxs, vys, DIVERGING, 1.0 - span, 1.0 + span)
      if row == 0:
        ax.set_title(f"$|\\Delta v|$ = {shove:g} m/s", pad=6)
        ax.set_xticklabels([])
      else:
        ax.set_xlabel("$v_x$ (m/s)")
      if column == 0:
        ax.set_ylabel("$v_y$ (m/s)")
      else:
        ax.set_yticklabels([])
    assert image is not None
    bar = fig.colorbar(image, ax=axes[row], fraction=0.02, pad=0.015, extend="both")
    hide(bar.outline)
    bar.ax.tick_params(length=0, labelsize=7, colors=MUTED)
    label = "forward" if quantity == "attain_x" else "lateral"
    bar.set_label(f"{label} axis\ndelivered / commanded", fontsize=8, color=INK_2)

  fig.suptitle(
    f"{run.label} — per-axis attainment, median over episodes",
    fontsize=11,
    color=INK,
    fontweight="semibold",
    x=0.02,
    ha="left",
  )
  note(
    fig,
    "gray is delivering the commanded speed on that axis, blue is undershoot "
    "and red overshoot; hatched cells ask for under 0.10 m/s on it, which is "
    "by construction the whole of the orthogonal single-axis line.",
  )
  save(fig, path)


def yaw_figure(run: Run, path: Path) -> None:
  """The yaw slice: shove magnitude against commanded yaw rate."""
  cells = yaw_cells(run)
  if not cells:
    return
  wzs = axis_values(cells, "wz")
  shoves = axis_values(cells, "shove")
  vx = cells[0]["vx"]

  fig, axes = plt.subplots(
    1,
    len(QUANTITIES),
    figsize=(2.3 * len(QUANTITIES) + 0.6, 3.2),
    squeeze=False,
    layout="constrained",
  )
  for column, (quantity, title, unit, higher_is_better) in enumerate(QUANTITIES):
    grid = np.full((len(shoves), len(wzs)), np.nan)
    for row, shove in enumerate(shoves):
      for index, wz in enumerate(wzs):
        grid[row, index] = run.value(run.find(vx, 0.0, wz, shove), quantity, "median")
    ax = axes[0][column]
    finite = grid[np.isfinite(grid)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    image = draw_heatmap(
      ax, grid, wzs, shoves, ramp(higher_is_better), vmin, max(vmax, vmin + 1e-9)
    )
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel(r"$\omega_z$ (rad/s)")
    if column == 0:
      ax.set_ylabel(r"$|\Delta v|$ (m/s)")
    else:
      ax.set_yticklabels([])
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    hide(bar.outline)
    bar.ax.tick_params(length=0, labelsize=7, colors=MUTED)
    bar.set_label(unit, fontsize=7.5, color=INK_2)

  fig.suptitle(
    f"{run.label} — yaw slice at $v_x$ = {vx:g} m/s, median over episodes",
    fontsize=11,
    color=INK,
    fontweight="semibold",
    x=0.02,
    ha="left",
  )
  save(fig, path)


def cells_at(run: Run, command: tuple[float, float, float]) -> list[dict]:
  """A run's cells at one command, ordered by shove magnitude."""
  vx, vy, wz = command
  cells = [
    cell
    for cell in run.cells
    if _close(cell["vx"], vx) and _close(cell["vy"], vy) and _close(cell["wz"], wz)
  ]
  return sorted(cells, key=lambda cell: cell["shove"])


def curve_figure(
  runs: list[Run],
  quantity: str,
  title: str,
  unit: str,
  commands: list[tuple[float, float, float]],
  path: Path,
) -> None:
  """One panel per commanded velocity, one line per run, against the shove.

  The band is the interquartile range, which is what makes this figure worth
  drawing next to the heatmaps: a median that holds up while its band opens is
  a controller that is still delivering on average and has started to fail some
  of the time, and only the band says so.
  """
  # Commands no run measured would draw as blank panels, which is not the same
  # statement as a command that was measured and produced nothing.
  commands = [
    command for command in commands if any(cells_at(run, command) for run in runs)
  ]
  if not commands:
    print(f"curves_{quantity} skipped: no run holds any of the chosen commands")
    return
  columns = min(len(commands), 3)
  rows = -(-len(commands) // columns)
  fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(2.7 * columns + 0.4, 2.3 * rows + 0.7),
    squeeze=False,
    sharey=True,
    sharex=True,
    layout="constrained",
  )
  for index, (vx, vy, wz) in enumerate(commands):
    ax = axes[index // columns][index % columns]
    for run in runs:
      cells = cells_at(run, (vx, vy, wz))
      if not cells:
        continue
      shoves = np.array([cell["shove"] for cell in cells])
      median = np.array([run.value(cell, quantity, "median") for cell in cells])
      if quantity == "fell":
        low = np.array([cell["fell_ci_low"] for cell in cells])
        high = np.array([cell["fell_ci_high"] for cell in cells])
      else:
        low = np.array([cell[quantity]["q25"] for cell in cells])
        high = np.array([cell[quantity]["q75"] for cell in cells])
      ax.fill_between(shoves, low, high, color=run.colour, alpha=0.16, linewidth=0)
      ax.plot(
        shoves,
        median,
        color=run.colour,
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label=run.label,
        zorder=3,
      )
    label = f"$v_x$={vx:g}, $v_y$={vy:g}"
    if not _close(wz, 0.0):
      label += rf", $\omega_z$={wz:g}"
    ax.set_title(label, fontsize=9)
    despine(ax)
    if index % columns == 0:
      ax.set_ylabel(unit)
    if index // columns == rows - 1:
      ax.set_xlabel(r"shove magnitude $|\Delta v|$ (m/s)")
  for index in range(len(commands), rows * columns):
    axes[index // columns][index % columns].set_visible(False)

  if len(runs) > 1:
    # From a panel that actually drew lines: the first one need not have, if a
    # run is missing that command.
    handles, labels = [], []
    for ax in axes.ravel():
      found, names = ax.get_legend_handles_labels()
      if len(names) > len(labels):
        handles, labels = found, names
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(runs), 4))
  fig.suptitle(
    f"{title} against shove magnitude — median and interquartile range",
    fontsize=11,
    color=INK,
    fontweight="semibold",
    x=0.02,
    ha="left",
  )
  save(fig, path)


def difference_figure(before: Run, after: Run, path: Path) -> None:
  """The second run minus the first, over the plane, diverging about zero."""
  cells = plane_cells(after)
  vxs = axis_values(cells, "vx")
  vys = axis_values(cells, "vy")
  shoves = axis_values(cells, "shove")

  fig, axes = plt.subplots(
    len(QUANTITIES),
    len(shoves),
    figsize=(1.55 * len(shoves) + 1.9, 1.7 * len(QUANTITIES) + 0.9),
    squeeze=False,
    layout="constrained",
  )
  for row, (quantity, title, _, _better) in enumerate(QUANTITIES):
    grids = [
      plane_array(after, quantity, "median", shove, vxs, vys)
      - plane_array(before, quantity, "median", shove, vxs, vys)
      for shove in shoves
    ]
    span = diverging_span(grids, centre=0.0)
    image = None
    for column, (shove, grid) in enumerate(zip(shoves, grids, strict=True)):
      ax = axes[row][column]
      image = draw_heatmap(ax, grid, vxs, vys, DIVERGING, -span, span)
      if row == 0:
        ax.set_title(f"$|\\Delta v|$ = {shove:g} m/s", pad=6)
      if column == 0:
        ax.set_ylabel("$v_y$ (m/s)")
      else:
        ax.set_yticklabels([])
      if row == len(QUANTITIES) - 1:
        ax.set_xlabel("$v_x$ (m/s)")
      else:
        ax.set_xticklabels([])
    assert image is not None
    bar = fig.colorbar(image, ax=axes[row], fraction=0.02, pad=0.015, extend="both")
    hide(bar.outline)
    bar.ax.tick_params(length=0, labelsize=7, colors=MUTED)
    bar.set_label(f"{title}\ndifference in median", fontsize=8, color=INK_2)

  fig.suptitle(
    f"{after.label} minus {before.label}",
    fontsize=11,
    color=INK,
    fontweight="semibold",
    x=0.02,
    ha="left",
  )
  note(
    fig,
    "gray is no change; which direction is an improvement differs by row -- "
    "more attainment, more warning before a fall and more survival is better; "
    "fewer falls is better.",
  )
  save(fig, path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


DEFAULT_CURVE_COMMANDS = (
  (0.25, 0.0, 0.0),
  (0.5, 0.0, 0.0),
  (0.75, 0.0, 0.0),
  (-0.5, 0.0, 0.0),
  (0.0, 0.5, 0.0),
  (0.3, 0.0, 0.5),
)
"""Commands the curve figures cut through: a forward-speed ladder, one
backwards ask, one purely lateral one, and one turn taken while walking. Six
because that is how many distinguishable colours the house palette has, and a
seventh line would have to reuse one.

``vx = 1.0`` is deliberately absent. Measured on the competence-trained
policies, both stop walking and march in place above roughly 0.81 m/s
commanded -- perfect tracking at 0.80, 0.04 m/s delivered at 0.82 -- so a panel
at 1.0 m/s puts a policy that refused the command beside one that attempted it
and reads as a tracking collapse. That comparison is worth making, but it is a
statement about where each controller's band ends, and the envelope heatmaps
are where it belongs. Put a command back here only if every controller drawn
actually attempts it."""


def parse_commands(text: str | None) -> list[tuple[float, float, float]]:
  if text is None:
    return list(DEFAULT_CURVE_COMMANDS)
  commands = []
  for chunk in text.split(";"):
    parts = [float(value) for value in chunk.split(",")]
    if len(parts) != 3:
      raise ValueError(f"a command is three numbers, got {chunk!r}")
    commands.append((parts[0], parts[1], parts[2]))
  return commands


@dataclass
class Args:
  input_dir: Path = Path("logs/eval")
  """Directory holding one subdirectory per run, each with a ``cells.json``."""
  output_dir: Path | None = None
  """Where the figures go. Defaults to ``<input-dir>/figures``."""
  runs: str | None = None
  """Draw only these runs, in this order: a comma-separated list of directory
  names. Defaults to every run in the directory, in name order."""
  curve_commands: str | None = None
  """Commands the curve figures cut through, as ``vx,vy,wz`` triples separated
  by semicolons, e.g. ``0.5,0,0;0,0.5,0``. Defaults to a forward-speed ladder
  plus one lateral and one turning command."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()

  runs = load_runs(args.input_dir, args.runs)
  output_dir = args.output_dir or args.input_dir / "figures"
  commands = parse_commands(args.curve_commands)
  print(f"drawing {len(runs)} run(s): {', '.join(run.name for run in runs)}")

  for run in runs:
    plane_figure(run, "median", output_dir / f"{run.name}_envelope")
    plane_figure(run, "iqr", output_dir / f"{run.name}_spread")
    axes_figure(run, output_dir / f"{run.name}_axes")
    yaw_figure(run, output_dir / f"{run.name}_yaw")

  for quantity, title, unit, _ in QUANTITIES:
    curve_figure(
      runs, quantity, title, unit, commands, output_dir / f"curves_{quantity}"
    )

  if len(runs) == 2:
    difference_figure(runs[0], runs[1], output_dir / "difference")
  elif len(runs) > 2:
    print("difference figure skipped: it is drawn for a pair of runs only")


if __name__ == "__main__":
  main()
