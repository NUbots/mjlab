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

import csv
import json
import math
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
    run = Run(
      name=directory.name,
      colour=PALETTE[index % len(PALETTE)],
      cells=summary["cells"],
      meta=summary.get("run", {}),
    )
    attach_displacement(run, directory)
    runs.append(run)
  if wanted is None:
    return runs
  by_name = {run.name: run for run in runs}
  missing = [name for name in wanted.split(",") if name not in by_name]
  if missing:
    raise KeyError(f"no such run: {', '.join(missing)}; have {list(by_name)}")
  return [by_name[name] for name in wanted.split(",")]


DISPLACEMENT = "attain_displacement"
"""Cell key: how far a shove moved delivered speed, either way."""

SURVIVED = "attain_survived"
"""Cell key: attainment over the episodes that ran their full length."""

EFFECTIVE = "attain_effective"
"""Cell key: delivery over the whole nominal episode, a fall counting as zero.

All three are derived here from the ``episodes.csv`` beside each
``cells.json``, so they need no new data collection.
"""


def _quantile_record(values: list[float]) -> dict:
  """The five numbers ``summarise_cells`` reports, over a list of episodes."""
  finite = [v for v in values if not math.isnan(v)]
  if not finite:
    nan = float("nan")
    return {"n": 0, "median": nan, "q25": nan, "q75": nan, "iqr": nan}
  q25, median, q75 = (float(v) for v in np.percentile(finite, [25, 50, 75]))
  return {
    "n": len(finite),
    "median": median,
    "q25": q25,
    "q75": q75,
    "iqr": q75 - q25,
  }


def attach_displacement(run: Run, directory: Path) -> bool:
  """Add :data:`DISPLACEMENT` to a run's cells, from its ``episodes.csv``.

  Per episode, ``|attain - undisturbed attain at the same command|``: how far
  the shove moved delivered speed away from where it sits with no shove, in
  either direction.

  This exists because the median of ``attain`` cannot answer "does a shove cost
  tracking". Shove headings are drawn uniformly, so a shove along the command
  and one against it are equally likely and their effects cancel in any
  centre-of-distribution statistic. Measured over the 38 commands of one run,
  the median displacement rises monotonically with shove magnitude in 28 of
  them; the shift in the median attainment does so in 7. The effect was always
  in the data -- it lives in the spread, and a median throws the spread away.

  Displacement rather than a one-sided shortfall because the median of a
  one-sided quantity floors at zero whenever most episodes are unharmed, which
  is most cells. Absolute deviation has an informative median everywhere and
  keeps the same quantile shape as every other quantity here.

  Both are taken over the episodes that ran their full length, and this is the
  important part. Attainment is a mean over the steps an episode actually had,
  so an episode that falls is averaged over the run-up that preceded the shove
  that killed it. The shoves land at fixed times -- 3, 7, 11 and 15 s of a 20 s
  episode -- so a controller that goes over on the first one keeps only about
  three seconds of undisturbed walking, and *that* is what its attainment
  reports. Worse, the effect grows with the shove: harder pushes end episodes
  sooner, so the fraction of each episode that predates any disturbance rises,
  and the number drifts toward the undisturbed value exactly where the
  controller is doing worst. Measured on the walk engine at ``vx=0.3, wz=0.5``,
  the median episode at a 1.2 m/s shove is 3.66 s long, takes one shove and is
  82% pre-shove samples -- so its "0.53 attainment" is mostly a measurement of
  the three seconds before it was touched.

  Restricting to full-length episodes removes that entirely: every one of them
  took all four shoves over the same number of steps. The price is survivor
  bias, and the honest reading is to keep the fall rate beside it -- a cell
  where nothing survived reports no attainment at all, which is the truth.

  Returns whether the file was there to read.
  """
  path = directory / "episodes.csv"
  if not path.exists():
    return False

  # Rounded so a cell written as 0.4 and an episode written as
  # 0.4000000059604645 -- the float32 the table was built from -- land on one
  # key. Six places is far finer than any command this grid sweeps.
  def key(vx: float, vy: float, wz: float, shove: float) -> tuple:
    return tuple(round(v, 6) for v in (vx, vy, wz, shove))

  by_cell: dict[tuple, list[float]] = {}
  with path.open() as handle:
    for row in csv.DictReader(handle):
      cell_key = key(
        float(row["command_vx"]),
        float(row["command_vy"]),
        float(row["command_wz"]),
        float(row["shove"]),
      )
      by_cell.setdefault(cell_key, []).append(
        (float(row["attain"]), float(row["ep_len_frac"]), float(row["fell"]) < 0.5)
      )

  # The undisturbed delivery for each command, which the shove is measured
  # against. It is a median over episodes that are near enough identical --
  # with no shove the protocol has no stochastic input at all.
  baseline: dict[tuple, float] = {}
  for cell_key, episodes in by_cell.items():
    if _close(cell_key[3], 0.0):
      survived = [a for a, _frac, ok in episodes if ok]
      baseline[cell_key[:3]] = _quantile_record(survived)["median"]

  for cell in run.cells:
    cell_key = key(cell["vx"], cell["vy"], cell["wz"], cell["shove"])
    episodes = by_cell.get(cell_key, [])
    clean = baseline.get(cell_key[:3], float("nan"))
    survived = [a for a, _frac, ok in episodes if ok]

    cell[SURVIVED] = _quantile_record(survived)
    cell[DISPLACEMENT] = _quantile_record(
      [abs(a - clean) for a in survived] if not math.isnan(clean) else []
    )
    # Zero-fill: an episode that fell delivered nothing for the rest of its
    # nominal length, so its mean over the sampled steps is scaled by the share
    # of the episode it lasted. Every episode then has the same denominator and
    # nothing is censored -- a fall costs delivery instead of hiding it.
    cell[EFFECTIVE] = _quantile_record([a * frac for a, frac, _ok in episodes])
  return True


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
    fontsize=15,
    color=INK,
    fontweight="semibold",
    x=0.5,
    ha="center",
  )
  scale = "Darker is better" if statistic == "median" else "Darker is a wider spread"
  note(
    fig,
    f"{scale}",
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
    # Gathered over every panel, not taken from the best single one: the runs
    # are not swept over one command grid, so a run can be absent from the
    # panel that drew the most lines and would go unnamed.
    handles, labels = [], []
    for ax in axes.ravel():
      for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
        if name not in labels:
          handles.append(handle)
          labels.append(name)
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


def default_curve_commands(
  runs: list[Run], forward: int = 4
) -> list[tuple[float, float, float]]:
  """Commands to cut the curve figures through, read off the data.

  A hardcoded ladder cannot work here, because the runs are not swept over one
  command grid: the walk engine is stepped in 0.1 m/s out to 0.5, a policy in
  0.33 out to 1.67, and the only forward speed the two share is zero. Fixed
  speeds then land between the values a dataset actually holds and their panels
  come out empty, while the ones that do land sit in the middle of the range
  where nothing interesting happens.

  So the ladder is taken from the widest forward sweep any run has, spread
  across it with both ends included, and the lateral and turning probes from
  the largest each dataset holds. A run that was not asked a command simply has
  no line in that panel; the legend is built from the runs that drew.
  """
  widest: list[float] = []
  for run in runs:
    vxs = sorted(
      {c["vx"] for c in run.cells if _close(c["vy"], 0.0) and _close(c["wz"], 0.0)}
    )
    if len(vxs) >= 2 and (not widest or vxs[-1] - vxs[0] > widest[-1] - widest[0]):
      widest = vxs

  commands: list[tuple[float, float, float]] = []
  if widest:
    picks = sorted(
      {
        int(round(i))
        for i in np.linspace(0, len(widest) - 1, min(forward, len(widest)))
      }
    )
    commands = [(widest[i], 0.0, 0.0) for i in picks]

  # One lateral and one turning probe, so the figure keeps asking the two
  # questions a forward ladder cannot.
  lateral = max(
    (
      c["vy"]
      for run in runs
      for c in run.cells
      if _close(c["vx"], 0.0) and _close(c["wz"], 0.0)
    ),
    default=0.0,
  )
  if lateral > 0.0:
    commands.append((0.0, lateral, 0.0))
  turning = max(
    ((c["wz"], c["vx"]) for run in runs for c in run.cells if not _close(c["wz"], 0.0)),
    default=None,
  )
  if turning is not None:
    commands.append((turning[1], 0.0, turning[0]))
  return commands


def parse_commands(
  text: str | None, runs: list[Run]
) -> list[tuple[float, float, float]]:
  if text is None:
    return default_curve_commands(runs)
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
  spanning the widest sweep in the data, plus the largest lateral and turning
  commands it holds -- see :func:`default_curve_commands`."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()

  runs = load_runs(args.input_dir, args.runs)
  output_dir = args.output_dir or args.input_dir / "figures"
  commands = parse_commands(args.curve_commands, runs)
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

  # Drawn only as a curve, and only when the episodes were on disk to derive it
  # from: it is the figure that answers whether a shove costs tracking at all,
  # which the attainment curve above cannot -- see attach_displacement.
  if any(DISPLACEMENT in cell for run in runs for cell in run.cells):
    for quantity, title, unit in (
      (
        SURVIVED,
        "Attainment, full-length episodes only",
        "delivered / commanded",
      ),
      (
        DISPLACEMENT,
        "Attainment displacement",
        "|attain - undisturbed attain|",
      ),
      (
        EFFECTIVE,
        "Effective delivery over the nominal episode",
        "attain x share of episode survived",
      ),
    ):
      curve_figure(
        runs, quantity, title, unit, commands, output_dir / f"curves_{quantity}"
      )

  if len(runs) == 2:
    difference_figure(runs[0], runs[1], output_dir / "difference")
  elif len(runs) > 2:
    print("difference figure skipped: it is drawn for a pair of runs only")


if __name__ == "__main__":
  main()
