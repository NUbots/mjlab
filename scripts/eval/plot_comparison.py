"""Figures for the policy comparison.

Reads what ``collect_comparison.sh`` wrote and draws three kinds of figure:

- ``fig1_profile_<name>``: velocity tracking under a moving command, one per
  controller;
- ``fig3c_normalised_error_plane``: tracking error over three command planes,
  each axis' error divided by the largest command on that axis;
- ``fig7_push_envelope``: the push magnitude each direction withstands, walking
  and standing.

  uv run python scripts/eval/plot_comparison.py --input-dir logs/eval/comparison

Which controllers are in a directory, what to call them and what colour to give
them come from the ``controllers.json`` the collection wrote.
``--controllers a,b`` narrows and reorders the set, keeping each controller's
colour. Figures land in ``<input-dir>/figures`` as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
  AXIS_COLOUR,
  AXIS_LABEL,
  AXIS_UNIT,
  BASELINE,
  GRID,
  INK,
  INK_2,
  MUTED,
  PALETTE,
  RED,
  SEQUENTIAL,
  SMOOTH_S,
  SURFACE,
  despine,
  hide,
  moving_average,
  save,
  use_house_style,
)
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import mjlab

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_csv(path: Path) -> dict[str, np.ndarray]:
  """A CSV with a header row, as a dict of float columns."""
  with path.open() as handle:
    reader = csv.reader(handle)
    header = next(reader)
    rows = [[float(value) for value in row] for row in reader]
  columns = np.array(rows, dtype=float).T if rows else np.zeros((len(header), 0))
  return dict(zip(header, columns, strict=True))


def read_json(path: Path) -> dict:
  with path.open() as handle:
    return json.load(handle)


@dataclass(frozen=True)
class Controller:
  """One controller in a comparison: what to load, and what to call it."""

  name: str
  """Slug the collection tagged its runs with, e.g. ``grid_vx_vy_<name>``."""
  label: str
  """Name shown on the figures."""
  colour: str
  """Series colour."""


def load_controllers(input_dir: Path, only: str | None) -> list[Controller]:
  """The controllers in a collection, in the order they were given to it."""
  manifest = input_dir / "controllers.json"
  if not manifest.is_file():
    raise SystemExit(
      f"no controllers.json in {input_dir}. Point --input-dir at a directory "
      f"collect_comparison.sh wrote into."
    )
  entries = read_json(manifest)["controllers"]

  # Colours are handed out over the whole directory before any narrowing, so a
  # controller keeps its colour however the set is cut down.
  controllers = [
    Controller(
      name=entry["name"],
      label=entry.get("label") or entry["name"],
      colour=entry.get("colour") or PALETTE[index % len(PALETTE)],
    )
    for index, entry in enumerate(entries)
  ]
  if only is None:
    return controllers

  wanted = [name.strip() for name in only.split(",") if name.strip()]
  by_name = {controller.name: controller for controller in controllers}
  unknown = [name for name in wanted if name not in by_name]
  if unknown:
    raise SystemExit(
      f"no such controller(s) in {input_dir}: {', '.join(unknown)}. "
      f"It holds: {', '.join(sorted(by_name))}."
    )
  return [by_name[name] for name in wanted]


@dataclass
class Grid:
  """One two-axis command grid, per environment."""

  data: dict[str, np.ndarray]
  summary: dict


@dataclass
class Trace:
  """One profile run: the schedule that was issued and the response to it."""

  controller: Controller
  run: dict
  time: np.ndarray
  command: np.ndarray  # (T, N, 3)
  achieved: np.ndarray  # (T, N, 3)
  upright: np.ndarray  # (T, N)

  def lane_envs(self, name: str) -> np.ndarray:
    return np.array(
      [i for i, lane in enumerate(self.run["lane_of_env"]) if lane == name]
    )

  @property
  def dt(self) -> float:
    return 1.0 / float(self.run["control_hz"])


def load_trace(directory: Path, controller: Controller) -> Trace:
  run = read_json(directory / "run.json")
  flat = read_csv(directory / "trace.csv")
  num_envs = int(run["num_envs"])
  num_steps = int(flat["step"].max()) + 1

  def reshape(*names: str) -> np.ndarray:
    return np.stack(
      [flat[name].reshape(num_steps, num_envs) for name in names], axis=-1
    )

  return Trace(
    controller=controller,
    run=run,
    time=flat["time"].reshape(num_steps, num_envs)[:, 0],
    command=reshape("command_vx", "command_vy", "command_wz"),
    achieved=reshape("achieved_vx", "achieved_vy", "achieved_wz"),
    upright=flat["upright"].reshape(num_steps, num_envs),
  )


@dataclass
class Battery:
  """One push battery: every trial, and the aggregate the run wrote."""

  data: dict[str, np.ndarray]
  summary: dict

  @property
  def envelope(self) -> list[dict]:
    """Critical magnitude per direction, as the run computed it. Read rather
    than recomputed, so the figure cannot disagree with the summary."""
    return self.summary["push"]["envelope"]

  @property
  def magnitudes(self) -> np.ndarray:
    return np.unique(self.data["push_delta_v"])


def load_run(directory: Path) -> tuple[dict[str, np.ndarray], dict]:
  return read_csv(directory / "per_env.csv"), read_json(directory / "summary.json")


# --------------------------------------------------------------------------
# Figure 1: velocity tracking under a moving command
# --------------------------------------------------------------------------


def figure_profile(trace: Trace, path: Path) -> None:
  """DeepWalk Fig. 3, as one continuous trace.

  The six schedules ran in parallel slices of the batch, not end to end -- a
  fall under one command must not contaminate the next -- so this lays their
  recorded windows side by side on one time axis. A boundary between two
  schedules is a change of robot, not a change of command.

  Only each lane's own schedule is drawn. A lane that finishes before the
  longest one is held at rest for the remainder of the run, and that tail is
  padding rather than measurement.
  """
  fig, ax = plt.subplots(figsize=(16, 4.6))
  despine(ax)
  window = max(1, int(round(SMOOTH_S / trace.dt)))
  axis_index = {"vx": 0, "vy": 1, "wz": 2}

  # One y-scale for the whole strip, set by the command rather than by the
  # response: within a single step the torso sways by several times what the
  # command asks for, and scaling to that would flatten the tracking. The raw
  # trace is clipped to the frame instead.
  span = max(0.5, 1.7 * float(np.abs(trace.command).max()))

  offset = 0.0
  boundaries: list[float] = []
  for order, lane in enumerate(trace.run["lanes"]):
    envs = trace.lane_envs(lane["name"])
    # Half a step of slack: the schedule's length is a sum of floats and the
    # sample times are multiples of dt.
    keep = trace.time <= float(lane["duration_s"]) + trace.dt / 2
    time = trace.time[keep] + offset
    width = float(lane["duration_s"])

    if order % 2:
      ax.axvspan(offset, offset + width, color=GRID, alpha=0.28, zorder=0)

    fell_at = None
    for env in envs:
      below = np.flatnonzero(trace.upright[keep][:, env] < 0.5)
      if below.size:
        step = float(time[below[0]])
        fell_at = step if fell_at is None else min(fell_at, step)

    # Label placement is worked out first: in a combined schedule both axes
    # move together, so the higher trace's label goes above it and the lower
    # trace's below, wherever the two are close.
    series = []
    for name in lane["axes"]:
      column = axis_index[name]
      raw = trace.achieved[keep][:, envs, column]
      smoothed = moving_average(raw.mean(axis=1), window)
      command = trace.command[keep][:, envs[0], column]
      # The middle of the first commanded plateau: the measurement lags the
      # command, so the plateau's first sample would label a response that is
      # still climbing.
      magnitude = np.abs(command[: command.size // 2])
      at_peak = np.flatnonzero(magnitude >= magnitude.max() - 1e-6)
      plateau = int(at_peak[at_peak.size // 2])
      series.append((name, raw, smoothed, command, plateau))
    highest = max(range(len(series)), key=lambda i: series[i][2][series[i][4]])

    for depth, (name, raw, smoothed, command, plateau) in enumerate(series):
      colour = AXIS_COLOUR[name]
      ax.plot(
        time,
        np.clip(raw.mean(axis=1), -span, span),
        color=colour,
        linewidth=0.5,
        alpha=0.22,
        zorder=2,
      )
      if raw.shape[1] > 1:
        ax.fill_between(
          time,
          np.clip(raw.min(axis=1), -span, span),
          np.clip(raw.max(axis=1), -span, span),
          color=colour,
          alpha=0.10,
          linewidth=0,
          zorder=1,
        )
      ax.plot(
        time, smoothed, color=colour, linewidth=2.0, zorder=4, solid_capstyle="round"
      )
      ax.plot(
        time,
        command,
        color=colour,
        linewidth=1.3,
        linestyle=(0, (5, 3)),
        alpha=0.95,
        zorder=3,
      )
      above = depth == highest
      ax.annotate(
        AXIS_LABEL[name],
        xy=(float(time[plateau]), float(smoothed[plateau])),
        xytext=(0, 11 if above else -12),
        textcoords="offset points",
        color=colour,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom" if above else "top",
      )

    if fell_at is not None:
      ax.axvspan(fell_at, offset + width, color=RED, alpha=0.09, zorder=1)
      ax.axvline(fell_at, color=RED, linewidth=1.2, linestyle=":", zorder=5)
      ax.annotate(
        f"fell at {fell_at - offset:.1f} s",
        xy=(fell_at, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color=RED,
        fontsize=7.5,
        fontweight="bold",
      )

    ax.annotate(
      lane["name"].replace("+", " + "),
      xy=(offset + width / 2, 1.0),
      xycoords=("data", "axes fraction"),
      xytext=(0, 5),
      textcoords="offset points",
      ha="center",
      va="bottom",
      fontsize=9,
      fontweight="semibold",
      color=INK,
    )

    offset += width
    boundaries.append(offset)

  for edge in boundaries[:-1]:
    ax.axvline(edge, color=BASELINE, linewidth=1.0, zorder=5)
  ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
  ax.set_ylim(-span, span)
  ax.set_xlim(0.0, offset)
  ax.margins(x=0)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("velocity (m/s) · yaw rate (rad/s)")

  handles = [
    Line2D([], [], color=INK_2, linewidth=2.0, label=f"measured ({SMOOTH_S} s mean)"),
    Line2D([], [], color=INK_2, linewidth=0.7, alpha=0.4, label="measured (raw)"),
    Line2D(
      [], [], color=INK_2, linewidth=1.4, linestyle=(0, (5, 3)), label="commanded"
    ),
  ]
  handles += [
    Line2D([], [], color=AXIS_COLOUR[a], linewidth=2.4, label=f"{AXIS_LABEL[a]} axis")
    for a in ("vx", "vy", "wz")
  ]
  fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    ncol=6,
    columnspacing=1.6,
  )
  fig.suptitle(
    f"Velocity tracking under a moving command — {trace.controller.label}",
    x=0.006,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.text(
    0.006,
    0.012,
    "Each block is an independent robot, so nothing carries across a boundary. "
    "Single axes first, then pairs; every schedule visits both signs.",
    fontsize=7.5,
    color=MUTED,
  )
  fig.tight_layout(rect=(0, 0.05, 1, 0.90))
  save(fig, path)


# --------------------------------------------------------------------------
# Figure 3c: normalised error over the command planes
# --------------------------------------------------------------------------

# Fixed vertical margins of the command-plane figure, in inches, so adding a
# controller adds a row instead of shrinking every row.
HEADER_IN = 1.12
FOOTER_IN = 0.61
ROW_IN = 2.74

GRID_PAIRS = (
  ("vx", "vy", "grid_vx_vy"),
  ("vx", "wz", "grid_vx_wz"),
  ("vy", "wz", "grid_vy_wz"),
)


def grid_field(
  data: dict[str, np.ndarray], x: str, y: str, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Per-environment values reshaped onto the two command axes swept."""
  xs = np.unique(data[f"command_{x}"])
  ys = np.unique(data[f"command_{y}"])
  field = np.full((ys.size, xs.size), np.nan)
  rows = np.searchsorted(ys, data[f"command_{y}"])
  columns = np.searchsorted(xs, data[f"command_{x}"])
  field[rows, columns] = values
  return xs, ys, field


def _edges(values: np.ndarray) -> np.ndarray:
  step = np.diff(values)
  return np.concatenate(
    [[values[0] - step[0] / 2], values[:-1] + step / 2, [values[-1] + step[-1] / 2]]
  )


def command_scales(grids: dict[str, dict[str, Grid]]) -> dict[str, float]:
  """Per-axis reference spans for the normalised error.

  The largest magnitude commanded on each axis, over every grid and every
  controller: over the whole figure, so a cell means the same thing in every
  panel, and over the *commands*, so the scale is a property of the collection
  rather than of whichever controller tracked worst.
  """
  scales: dict[str, float] = {}
  for axis in ("vx", "vy", "wz"):
    largest = max(
      float(np.abs(grid.data[f"command_{axis}"]).max())
      for per_controller in grids.values()
      for grid in per_controller.values()
    )
    # An axis no grid moved keeps its error in its own units instead of
    # dividing by zero.
    scales[axis] = largest or 1.0
  return scales


def normalised_error(data: dict[str, np.ndarray], scales: dict[str, float]):
  """Root mean square over the three axes of error / commanded span.

  Dimensionless, so yaw sits beside the planar axes; an axis missed by the full
  commanded span contributes one, so a cell reads as a fraction. Comparable
  between the controllers of one collection, not across collections swept
  over different ranges.
  """
  return np.sqrt(
    np.mean(
      [(data[f"error_{axis}"] / scales[axis]) ** 2 for axis in ("vx", "vy", "wz")],
      axis=0,
    )
  )


def figure_normalised_error_plane(
  controllers: list[Controller], grids: dict[str, dict[str, Grid]], path: Path
) -> None:
  """Normalised error over three command planes, one row per controller.

  The outline on each panel encloses the commands the robot held for the
  whole run; outside it, the error is an average over the seconds before it
  fell.
  """
  rows = len(controllers)
  height = HEADER_IN + FOOTER_IN + ROW_IN * rows
  fig, axes = plt.subplots(rows, 3, figsize=(12.2, height), squeeze=False)
  top = 1.0 - HEADER_IN / height
  bottom = FOOTER_IN / height
  fig.subplots_adjust(
    left=0.072, right=0.895, top=top, bottom=bottom, wspace=0.40, hspace=0.50
  )
  scales = command_scales(grids)
  # Fixed rather than read off the data: one means the controller missed by
  # everything it was asked for. A cell past one clips to the top of the ramp.
  norm = Normalize(vmin=0.0, vmax=1.0)

  mesh = None
  for row, controller in enumerate(controllers):
    for column, (x, y, key) in enumerate(GRID_PAIRS):
      ax = axes[row, column]
      despine(ax)
      ax.grid(False)
      data = grids[controller.name][key].data
      xs, ys, values = grid_field(data, x, y, normalised_error(data, scales))
      ax.set_facecolor("#eceae4")
      mesh = ax.pcolormesh(
        _edges(xs),
        _edges(ys),
        np.ma.masked_invalid(values),
        shading="flat",
        cmap=SEQUENTIAL,
        norm=norm,
      )
      _, _, survived = grid_field(data, x, y, data["survived"])
      held = np.nan_to_num(survived, nan=0.0) > 0.5
      ax.contour(xs, ys, held, levels=[0.5], colors=[INK], linewidths=1.2)
      ax.axhline(0.0, color=SURFACE, linewidth=0.6, alpha=0.6)
      ax.axvline(0.0, color=SURFACE, linewidth=0.6, alpha=0.6)
      ax.set_xlabel(f"{AXIS_LABEL[x]} ({AXIS_UNIT[x]})", fontsize=14)
      ax.set_ylabel(f"{AXIS_LABEL[y]} ({AXIS_UNIT[y]})", fontsize=14)
      ax.set_title(f"{AXIS_LABEL[x]} × {AXIS_LABEL[y]}", loc="left", pad=6, color=INK)
    fig.text(
      0.072,
      axes[row, 0].get_position().y1 + 0.28 / height,
      controller.label,
      fontsize=15,
      fontweight="bold",
      color=controller.colour,
    )

  assert mesh is not None
  bar = fig.colorbar(mesh, cax=fig.add_axes((0.918, bottom, 0.014, top - bottom)))
  bar.set_label("Normalised Error", color=INK_2, fontsize=15)
  hide(bar.outline)
  bar.ax.tick_params(colors=MUTED, labelsize=15)
  save(fig, path, tight=False)


# --------------------------------------------------------------------------
# Figure 7: push survival envelope
# --------------------------------------------------------------------------

BATTERY_KEYS = ("push_walk", "push_stand")
BATTERY_TITLE = {"push_walk": "Pushed while walking", "push_stand": "Pushed at a stand"}


def figure_push_envelope(
  controllers: list[Controller],
  batteries: dict[str, dict[str, Battery]],
  path: Path,
) -> None:
  """How hard a shove each direction takes before the robot goes down.

  One closed curve per controller, seen from above with the robot facing up
  the page. The radius is the magnitude at which half the trials in that
  direction end on the floor, interpolated from the survival curve.
  """
  fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.9), subplot_kw={"projection": "polar"})
  ceiling = max(
    float(batteries[controller.name][key].magnitudes.max())
    for key in BATTERY_KEYS
    for controller in controllers
  )

  for ax, key in zip(axes, BATTERY_KEYS, strict=True):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_ylim(0.0, ceiling)
    ax.set_rlabel_position(22.5)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["forwards", "left", "backwards", "right"], fontsize=10)

    for controller in controllers:
      envelope = batteries[controller.name][key].envelope
      angles = np.deg2rad([entry["heading_deg"] for entry in envelope])
      # A direction whose survival never crossed one half is drawn at the edge
      # of the battery and marked open: its envelope is somewhere beyond the
      # magnitudes tested.
      crossed = np.array([entry["crossed"] for entry in envelope])
      radius = np.array(
        [
          entry["critical_delta_v"] if entry["crossed"] else ceiling
          for entry in envelope
        ]
      )
      closed = np.append(angles, angles[:1])
      values = np.append(radius, radius[:1])
      ax.plot(closed, values, color=controller.colour, linewidth=2.0, zorder=3)
      ax.fill(closed, values, color=controller.colour, alpha=0.13, zorder=2)
      ax.plot(
        angles[crossed],
        radius[crossed],
        linestyle="none",
        marker="o",
        markersize=4.0,
        color=controller.colour,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        zorder=4,
      )
      ax.plot(
        angles[~crossed],
        radius[~crossed],
        linestyle="none",
        marker="^",
        markersize=6.0,
        markerfacecolor=SURFACE,
        markeredgecolor=controller.colour,
        markeredgewidth=1.3,
        zorder=4,
      )
    ax.set_title(
      BATTERY_TITLE[key], loc="center", y=-0.20, pad=18, color=INK, fontsize=12
    )

  handles = [
    Line2D([], [], color=controller.colour, linewidth=2.4, label=controller.label)
    for controller in controllers
  ]
  fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=min(len(handles), 3),
    columnspacing=3.0,
    fontsize=11,
  )
  fig.suptitle(
    "Push Survival Envelope",
    x=0.5,
    y=0.995,
    ha="center",
    fontsize=15,
    fontweight="bold",
    color=INK,
  )
  fig.tight_layout(rect=(0, 0.0, 1, 0.90))
  save(fig, path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


@dataclass
class Args:
  input_dir: Path = Path("logs/eval/comparison")
  """Directory ``collect_comparison.sh`` wrote into."""
  output_dir: Path | None = None
  """Where the figures go. Defaults to ``<input-dir>/figures``."""
  controllers: str | None = None
  """Draw only these controllers, in this order: a comma-separated list of the
  ``name=`` fields the collection was given (see its ``controllers.json``).
  Defaults to every controller in the directory."""


def check_inputs(input_dir: Path, controllers: list[Controller]) -> None:
  """Fail before drawing anything, naming every run that is missing."""
  wanted = [(f"profile_{c.name}", "trace.csv") for c in controllers] + [
    (f"{key}_{c.name}", "per_env.csv")
    for c in controllers
    for key in (*(key for _, _, key in GRID_PAIRS), *BATTERY_KEYS)
  ]
  missing = [
    f"{name}/{file}" for name, file in wanted if not (input_dir / name / file).is_file()
  ]
  if missing:
    raise SystemExit(
      "\n".join(
        [f"{len(missing)} run(s) missing from {input_dir}:"]
        + [f"  {name}" for name in missing]
        + [
          "\nIf collect_comparison.sh stopped early, re-run it; to plot what is "
          "here instead, name the complete controllers with --controllers."
        ]
      )
    )


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()
  out = args.output_dir or (args.input_dir / "figures")
  controllers = load_controllers(args.input_dir, args.controllers)
  check_inputs(args.input_dir, controllers)

  grids: dict[str, dict[str, Grid]] = {}
  batteries: dict[str, dict[str, Battery]] = {}
  for controller in controllers:
    directory = args.input_dir
    grids[controller.name] = {
      key: Grid(*load_run(directory / f"{key}_{controller.name}"))
      for _, _, key in GRID_PAIRS
    }
    batteries[controller.name] = {
      key: Battery(*load_run(directory / f"{key}_{controller.name}"))
      for key in BATTERY_KEYS
    }
    trace = load_trace(directory / f"profile_{controller.name}", controller)
    figure_profile(trace, out / f"fig1_profile_{controller.name}")

  figure_normalised_error_plane(
    controllers, grids, out / "fig3c_normalised_error_plane"
  )
  figure_push_envelope(controllers, batteries, out / "fig7_push_envelope")


if __name__ == "__main__":
  main()
