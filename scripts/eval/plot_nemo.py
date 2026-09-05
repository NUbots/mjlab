"""Figures for the NEmo paper.

One categorical panel per run per shove level: the commanded velocity plane,
every cell painted by what the controller did there rather than by how well it
did it. Four outcomes, and the distinction that matters is between the two
middle ones -- a controller that declined the command and stood there is not
the same as one that tried and went over, and a heatmap of attainment alone
renders them identically. See ``collate_nemo_results`` for the thresholds.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
  AQUA,
  BLUE,
  DIVERGING,
  GOLD,
  ORANGE,
  DAQUA,
  GRID,
  INK,
  INK_2,
  MUTED,
  PALETTE,
  RED,
  SEQUENTIAL,
  SURFACE,
  hide,
  save,
  use_nemo_style,
)
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

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


def cells_at(run: Run, command: tuple[float, float, float]) -> list[dict]:
  """A run's cells at one command, ordered by shove magnitude."""
  vx, vy, wz = command
  cells = [
    cell
    for cell in run.cells
    if _close(cell["vx"], vx) and _close(cell["vy"], vy) and _close(cell["wz"], wz)
  ]
  return sorted(cells, key=lambda cell: cell["shove"])


NEMO_CATEGORIES: tuple[tuple[str, str], ...] = (
  ("Delivered", AQUA),
  ("Degraded", DAQUA),
  ("Reversed", ORANGE),
  ("Refused", GOLD),
  ("Failed", RED),
  ("Other", BLUE),
)
"""The 6 outcomes and their colours, in legend order.

Taken from the house palette so this figure sits with the rest of the paper's.
The pair that has to stay apart is Refused and Failed -- they are the two the
scalar metrics conflate -- and gold against red is the worst-separated adjacent
pair here at a colour-blind Delta E of 8.2, which clears the bar with the
legend carrying the names.
"""

CATEGORY_INDEX = {name: index for index, (name, _) in enumerate(NEMO_CATEGORIES)}
CATEGORY_CMAP = ListedColormap([colour for _, colour in NEMO_CATEGORIES])


def collate_nemo_results(run: Run):
  # For each run, collect the needed data and store for plotting onto a single plot
  print(f"Collating run information for run {run.name}")

  # Categories: 
  # Failed = fell > 0.3 & ep_len_frac < 0.7
  # Reversed = attain < 0 and not failed
  # Refused = attain < 0.3 & fell < 0.1 & ep_len_frac > 0.9
  # Degraded = 0.15 <= attain <= 0.7
  # Delivered = attain >= 0.7

  for cell in run.cells:
    attain_median = (cell.get("attain")).get("median")
    fell_rate = cell.get("fell_rate")
    ep_len_frac_first_quartile = (cell.get("ep_len_frac")).get("q25")
    vx = cell.get("vx")
    vy = cell.get("vy")
    wz = cell.get("wz")

    # Failed
    if (fell_rate > 0.4): # & (ep_len_frac_first_quartile < 0.7):
      cell["category"] = "Failed"
    # Reversed
    # elif (attain_median < 0.0):
    #   cell["category"] = "Reversed"
    #   print("reversed attain median: ", attain_median)
    # Refused
    elif ((np.abs(attain_median) < 0.3) and not (np.isnan(attain_median))) & (fell_rate <= 0.3) & (ep_len_frac_first_quartile > 0.8):
      cell["category"] = "Refused"
    # Degraded
    elif (0.15 <= attain_median < 0.8):
      cell["category"] = "Degraded"
    # Delivered
    elif ((attain_median >= 0.8) or (np.isnan(attain_median) and vx==0 and vy==0 and wz==0)):
      cell["category"] = "Delivered"
    # Other
    else:
      cell["category"] = "Other"
      if ((cell.get("shove") == 0) or (cell.get("shove") == 0.8)):
        print("Cell Category = other, vx = {:3.2f}, vy = {:3.2f}, wz = {:3.2f}, episodes = {:4.1f}, attain_median = {:5.4f}, fell_rate = {:6.5f}, ep_len_frac_first_quartile = {:4.3f}".format(cell.get("vx"), cell.get("vy"), cell.get("wz"), cell.get("episodes"), attain_median, fell_rate, ep_len_frac_first_quartile))


NEMO_SHOVES: tuple[float, ...] = (0.0, 0.8)
"""The two disturbance levels drawn: undisturbed, and the one that separates
the controllers. Everything else in the grid is collected and ignored here."""


def category_grid(
  run: Run, shove: float, vxs: list[float], vys: list[float]
) -> np.ndarray:
  """Shape ``(len(vys), len(vxs))`` of category indices, -1 where no cell.

  Restricted to the commanded velocity plane. The yaw slice sits at a
  ``(vx, vy)`` the plane also covers, so letting it in would put two different
  commands in one square and whichever came last would win.
  """
  grid = np.full((len(vys), len(vxs)), -1, dtype=int)
  for cell in plane_cells(run):
    if not _close(cell["shove"], shove):
      continue
    row = vys.index(cell["vy"])
    column = vxs.index(cell["vx"])
    grid[row, column] = CATEGORY_INDEX[cell["category"]]
  return grid


def draw_categories(
  ax, grid: np.ndarray, x_ticks: list[float], y_ticks: list[float]
) -> None:
  """One categorical panel: a colour per outcome, hatched where no cell ran."""
  ax.imshow(
    np.ma.masked_less(grid, 0),
    origin="lower",
    cmap=CATEGORY_CMAP,
    vmin=-0.5,
    vmax=len(NEMO_CATEGORIES) - 0.5,
    aspect="auto",
    interpolation="nearest",
    extent=(-0.5, len(x_ticks) - 0.5, -0.5, len(y_ticks) - 0.5),
  )
  for row in range(grid.shape[0]):
    for column in range(grid.shape[1]):
      if grid[row, column] < 0:
        ax.add_patch(
          Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor=SURFACE,
            edgecolor=GRID,
            hatch="///",
            linewidth=0.0,
            zorder=0.25,
          )
        )
  ax.set_xticks(np.arange(len(x_ticks)))
  ax.set_yticks(np.arange(len(y_ticks)))
  ax.set_xticks(np.arange(len(x_ticks) + 1) - 0.5, minor=True)
  ax.set_yticks(np.arange(len(y_ticks) + 1) - 0.5, minor=True)
  ax.grid(which="minor", color=SURFACE, linewidth=1.5)
  ax.grid(which="major", visible=False)
  ax.tick_params(which="minor", length=0)
  ax.tick_params(which="major", length=0)
  ax.set_xticklabels([f"{value:g}" for value in x_ticks], rotation=90, fontsize=7)
  ax.set_yticklabels([f"{value:g}" for value in y_ticks], fontsize=7)
  for spine in ax.spines.values():
    hide(spine)


def plot_nemo(runs: list[Run], output_dir: Path) -> None:
  """One row per run, one column per shove level in :data:`NEMO_SHOVES`."""
  fig, axes = plt.subplots(
    len(runs),
    len(NEMO_SHOVES),
    figsize=(1.55 * len(NEMO_SHOVES) + 1.9, 1.7 * len(runs) + 0.9),
    squeeze=False,
    layout="constrained",
  )

  for row, run in enumerate(runs):
    cells = plane_cells(run)
    vxs = axis_values(cells, "vx")
    vys = axis_values(cells, "vy")
    for column, shove in enumerate(NEMO_SHOVES):
      ax = axes[row][column]
      draw_categories(ax, category_grid(run, shove, vxs, vys), vxs, vys)
      if row == 0:
        ax.set_title(f"$|\\Delta v|$ = {shove:g} m/s", pad=6)
      if column == 0:
        # Wrapped, because a run name is a sentence and a panel is 1.7 inches
        # tall: unwrapped they run into each other between rows.
        ax.set_ylabel(textwrap.fill(run.label, 22), fontsize=8)
      else:
        ax.set_yticklabels([])
      if row == len(runs) - 1:
        ax.set_xlabel("$v_x$ (m/s)")
      else:
        ax.set_xticklabels([])

  # One shared axis label rather than four, so each row's ylabel is free to
  # carry the run's name.
  fig.supylabel("$v_y$ (m/s)", fontsize=9, color=INK_2)
  # Colour is the only encoding here, so the legend is not optional.
  fig.legend(
    handles=[
      Patch(facecolor=colour, edgecolor="none", label=name)
      for name, colour in NEMO_CATEGORIES
    ],
    loc="outside lower center",
    ncol=len(NEMO_CATEGORIES),
  )
  save(fig, output_dir / "nemo_envelope")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclass
class Args:
  input_dir: Path = Path("logs/eval")
  """Directory holding one subdirectory per run, each with a ``cells.json``."""
  output_dir: Path | None = None
  """Where the figures go. Defaults to ``<input-dir>/nemo-figures``."""
  runs: str | None = None
  """Draw only these runs, in this order: a comma-separated list of directory
  names. Defaults to every run in the directory, in name order."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_nemo_style()

  runs = load_runs(args.input_dir, args.runs)
  output_dir = args.output_dir or args.input_dir / "nemo-figures"
  print(f"drawing {len(runs)} run(s): {', '.join(run.name for run in runs)}")

  for run in runs:
    collate_nemo_results(run)

  plot_nemo(runs, output_dir)


if __name__ == "__main__":
  main()
