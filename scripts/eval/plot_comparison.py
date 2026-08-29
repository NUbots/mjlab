"""Figures for the walk-controller comparison.

Reads what ``collect_comparison.sh`` wrote and draws it. Nothing is computed
here that a metric could have computed: the per-environment CSVs already carry
the tracking and stability numbers, and this file's job is to choose what to put
next to what.

  uv run python scripts/eval/plot_comparison.py --input-dir logs/eval/comparison

Any number of controllers is drawn, in the order they were collected. Which
ones are in a directory, what to call them and what colour to give them come
from the ``controllers.json`` the collection wrote; a directory collected
before that manifest existed is read from the runs it holds instead.
``--controllers a,b`` narrows and reorders the set, keeping each
controller's colour from the full comparison.

Figures land in ``<input-dir>/figures`` as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import tyro

import mjlab

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Palette. Controllers take categorical slots in the order they were collected;
# where a panel is split by command axis instead, the three axes take slots 1 to
# 3. Chart chrome is the ink scale, never a series colour.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PURPLE = "#7b5bd6"
GOLD = "#b8860b"
RED = "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

PALETTE = (BLUE, ORANGE, AQUA, PURPLE, GOLD, RED)
"""Series colours, handed out in collection order.

Red is last on purpose: it marks a fall everywhere else on these figures, so it
is only spent on a controller once five others have taken a slot.
"""

AXIS_COLOUR = {"vx": BLUE, "vy": ORANGE, "wz": AQUA}
AXIS_LABEL = {"vx": "$v_x$", "vy": "$v_y$", "wz": r"$\omega_z$"}
AXIS_UNIT = {"vx": "m/s", "vy": "m/s", "wz": "rad/s"}

SMOOTH_S = 0.6
"""Window of the moving average drawn over a raw velocity trace, in seconds.

About two gait cycles at the engine's 0.32 s step period. The raw signal swings
by more than the command does within a single step -- the torso sways sideways
and counter-rotates every stride -- so the raw trace shows the gait and the
smoothed one shows the tracking.
"""

# Sequential blue, light to dark: near-zero recedes toward the surface.
SEQUENTIAL = LinearSegmentedColormap.from_list(
  "mjlab_blue",
  ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging blue <-> red about a neutral gray, for a signed error.
DIVERGING = LinearSegmentedColormap.from_list(
  "mjlab_bwr",
  ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a3", "#d03b3b", "#7d1f1f"],
)


def use_house_style() -> None:
  plt.rcParams.update(
    {
      "figure.facecolor": SURFACE,
      "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE,
      "font.family": "sans-serif",
      "font.sans-serif": ["DejaVu Sans"],
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.titleweight": "semibold",
      "axes.titlecolor": INK,
      "axes.labelsize": 9,
      "axes.labelcolor": INK_2,
      "axes.edgecolor": BASELINE,
      "axes.linewidth": 0.8,
      "axes.grid": True,
      "axes.axisbelow": True,
      "grid.color": GRID,
      "grid.linewidth": 0.6,
      "xtick.color": MUTED,
      "ytick.color": MUTED,
      "xtick.labelcolor": INK_2,
      "ytick.labelcolor": INK_2,
      "xtick.labelsize": 8,
      "ytick.labelsize": 8,
      "legend.frameon": False,
      "legend.fontsize": 8,
      "figure.dpi": 130,
    }
  )


def despine(ax) -> None:
  hide(ax.spines["top"])
  hide(ax.spines["right"])


def hide(artist) -> None:
  """Hide one artist.

  Untyped on purpose: matplotlib's stubs do not resolve ``Spine.set_visible``,
  and the alternative is a suppression comment on every call.
  """
  artist.set_visible(False)


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


@dataclass(frozen=True)
class Controller:
  """One controller in a comparison: what to load, and what to call it."""

  name: str
  """Slug the collection tagged its runs with, e.g. ``sweep_vx_<name>``."""
  engine: str
  """``quintic`` or ``rl``."""
  label: str
  """Name shown on the figures."""
  colour: str
  """Series colour."""


def default_label(name: str, engine: str) -> str:
  """The label ``collect_comparison.sh`` would have derived from a name."""
  base = "Quintic walk engine" if engine == "quintic" else "RL policy"
  return base if name == engine else f"{base} ({name})"


def load_controllers(input_dir: Path, only: str | None) -> list[Controller]:
  """The controllers in a collection, in the order they were given to it.

  ``collect_comparison.sh`` writes ``controllers.json`` naming them. A
  directory collected before that manifest existed is read instead from the
  runs it holds, which carry the engine in their own ``summary.json``; the walk
  engine leads, so a two-controller comparison keeps the colours it always had.
  """
  manifest = input_dir / "controllers.json"
  if manifest.is_file():
    with manifest.open() as handle:
      entries = json.load(handle)["controllers"]
  else:
    entries = []
    for directory in sorted(input_dir.glob("sweep_vx_*")):
      summary = directory / "summary.json"
      if not summary.is_file():
        continue
      with summary.open() as handle:
        engine = json.load(handle)["run"]["engine"]
      entries.append({"name": directory.name[len("sweep_vx_") :], "engine": engine})
    entries.sort(key=lambda entry: (entry["engine"] != "quintic", entry["name"]))

  if not entries:
    raise SystemExit(
      f"no controllers found in {input_dir}: it holds neither controllers.json "
      f"nor any sweep_vx_<name>/summary.json. Point --input-dir at a directory "
      f"collect_comparison.sh wrote into."
    )

  # Colours are handed out over the whole directory before any narrowing, so a
  # controller keeps the colour it has in the full comparison however the set is
  # cut down.
  controllers = [
    Controller(
      name=entry["name"],
      engine=entry["engine"],
      label=entry.get("label") or default_label(entry["name"], entry["engine"]),
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
class Sweep:
  """One command sweep or grid, per environment."""

  controller: Controller
  data: dict[str, np.ndarray]
  summary: dict

  @property
  def duration(self) -> float:
    return float(self.summary["run"]["duration_s"])

  @property
  def warmup(self) -> float:
    return float(self.summary["run"].get("warmup_s", 0.0))

  def grouped(self, axis: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Environment indices grouped by the value of one command axis."""
    command = self.data[f"command_{axis}"]
    values = np.unique(command)
    return values, [np.flatnonzero(command == value) for value in values]


def load_sweep(directory: Path, controller: Controller) -> Sweep:
  with (directory / "summary.json").open() as handle:
    summary = json.load(handle)
  return Sweep(controller, read_csv(directory / "per_env.csv"), summary)


@dataclass
class Trace:
  """One profile run: the schedule that was issued and the response to it."""

  controller: Controller
  run: dict
  time: np.ndarray
  command: np.ndarray  # (T, N, 3)
  achieved: np.ndarray  # (T, N, 3)
  upright: np.ndarray  # (T, N)

  @property
  def lanes(self) -> list[dict]:
    return self.run["lanes"]

  def lane_envs(self, name: str) -> np.ndarray:
    return np.array(
      [i for i, lane in enumerate(self.run["lane_of_env"]) if lane == name]
    )

  @property
  def dt(self) -> float:
    return 1.0 / float(self.run["control_hz"])


def load_trace(directory: Path, controller: Controller) -> Trace:
  with (directory / "run.json").open() as handle:
    run = json.load(handle)
  flat = read_csv(directory / "trace.csv")
  num_envs = int(run["num_envs"])
  steps = flat["step"].astype(int)
  num_steps = int(steps.max()) + 1

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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
  """Centred moving average, edges shortened rather than padded."""
  if window <= 1:
    return values
  kernel = np.ones(window) / window
  padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
  return np.convolve(padded, kernel, mode="valid")


# --------------------------------------------------------------------------
# Figure 1: velocity tracking under a moving command
# --------------------------------------------------------------------------


def figure_profile(trace: Trace, path: Path) -> None:
  """DeepWalk Fig. 3, one panel per command schedule.

  Six schedules run in parallel slices of the batch rather than end to end, so
  a fall under one command does not contaminate the next.
  """
  lanes = trace.lanes
  fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex=True)
  window = max(1, int(round(SMOOTH_S / trace.dt)))
  axis_index = {"vx": 0, "vy": 1, "wz": 2}

  # One y-scale for every panel, set by the command rather than by the response.
  # Within a single step the torso sways and counter-rotates by several times
  # what the command asks for; scaling to that would flatten the tracking into a
  # band across the middle, and it would also make the scale depend on the
  # controller, which is the thing being compared. The raw trace is clipped to
  # the frame instead.
  span = max(0.5, 1.7 * float(np.abs(trace.command).max()))

  for ax, lane in zip(axes.T.flatten(), lanes, strict=False):
    despine(ax)
    envs = trace.lane_envs(lane["name"])
    fell_at = None
    for env in envs:
      below = np.flatnonzero(trace.upright[:, env] < 0.5)
      if below.size:
        step = float(trace.time[below[0]])
        fell_at = step if fell_at is None else min(fell_at, step)

    for order, name in enumerate(lane["axes"]):
      column = axis_index[name]
      colour = AXIS_COLOUR[name]
      raw = trace.achieved[:, envs, column]
      # Replicas of a schedule differ only where the controller does: the
      # engine is deterministic, the policy sees noisy observations.
      ax.plot(
        trace.time,
        np.clip(raw.mean(axis=1), -span, span),
        color=colour,
        linewidth=0.5,
        alpha=0.22,
        zorder=2,
      )
      if raw.shape[1] > 1:
        ax.fill_between(
          trace.time,
          np.clip(raw.min(axis=1), -span, span),
          np.clip(raw.max(axis=1), -span, span),
          color=colour,
          alpha=0.10,
          linewidth=0,
          zorder=1,
        )
      ax.plot(
        trace.time,
        moving_average(raw.mean(axis=1), window),
        color=colour,
        linewidth=2.2,
        zorder=4,
        solid_capstyle="round",
      )
      ax.plot(
        trace.time,
        trace.command[:, envs[0], column],
        color=colour,
        linewidth=1.4,
        linestyle=(0, (5, 3)),
        alpha=0.95,
        zorder=3,
      )
      # Direct label, so identity never rests on colour alone. It sits over the
      # first plateau rather than at the right edge, where two axes of a
      # combined schedule both return to zero and the labels would collide.
      smoothed = moving_average(raw.mean(axis=1), window)
      command = trace.command[:, envs[0], column]
      plateau = int(np.argmax(np.abs(command[: command.size // 2])))
      above = order == 0
      ax.annotate(
        AXIS_LABEL[name],
        xy=(trace.time[plateau], smoothed[plateau]),
        xytext=(0, 13 if above else -14),
        textcoords="offset points",
        color=colour,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom" if above else "top",
      )

    if fell_at is not None:
      ax.axvspan(fell_at, trace.time[-1], color=RED, alpha=0.07, zorder=0)
      ax.axvline(fell_at, color=RED, linewidth=1.2, linestyle=":", zorder=5)
      ax.annotate(
        f"fell at {fell_at:.1f} s",
        xy=(fell_at, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -12),
        textcoords="offset points",
        color=RED,
        fontsize=7.5,
        fontweight="bold",
      )

    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.set_title(lane["name"].replace("+", " + "), loc="left", pad=6)
    ax.set_ylim(-span, span)
    ax.margins(x=0.01)

  for ax in axes[-1]:
    ax.set_xlabel("time (s)")
  for ax in axes[:, 0]:
    ax.set_ylabel("velocity (m/s) · yaw rate (rad/s)")

  handles = [
    Line2D([], [], color=INK_2, linewidth=2.0, label="measured (0.6 s mean)"),
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
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=6,
    columnspacing=1.6,
  )
  fig.suptitle(
    f"Velocity tracking under a moving command — {trace.controller.label}",
    x=0.008,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.text(
    0.008,
    0.012,
    "Each panel is an independent robot on the evaluation plant. Single axes first, "
    "then pairs; every schedule visits both signs.",
    fontsize=7.5,
    color=MUTED,
  )
  fig.tight_layout(rect=(0, 0.03, 1, 0.94))
  save(fig, path)


# --------------------------------------------------------------------------
# Figure 2: steady-state tracking
# --------------------------------------------------------------------------


def figure_tracking(
  controllers: list[Controller], sweeps: dict[str, dict[str, Sweep]], path: Path
) -> None:
  """Achieved against commanded on each axis, and the error underneath.

  The identity line is what perfect tracking would draw. Points where the robot
  fell are hollow: it still produced a mean velocity, but over a shorter and
  less meaningful window.
  """
  axes_order = ("vx", "vy", "wz")
  fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))

  for column, axis in enumerate(axes_order):
    top, bottom = axes[0, column], axes[1, column]
    despine(top)
    despine(bottom)
    limits: list[float] = []

    for controller in controllers:
      colour = controller.colour
      sweep = sweeps[controller.name][f"sweep_{axis}"]
      values, groups = sweep.grouped(axis)
      achieved = np.array(
        [np.nanmean(sweep.data[f"achieved_{axis}"][g]) for g in groups]
      )
      spread_lo = np.array(
        [np.nanmin(sweep.data[f"achieved_{axis}"][g]) for g in groups]
      )
      spread_hi = np.array(
        [np.nanmax(sweep.data[f"achieved_{axis}"][g]) for g in groups]
      )
      survived = np.array([sweep.data["survived"][g].mean() for g in groups])
      limits += [float(np.nanmin(values)), float(np.nanmax(values))]

      # The connecting line is drawn only through the commands every robot held
      # for the whole run. A fallen robot still has a mean velocity -- over the
      # seconds before it went down -- and joining those into the curve would
      # draw a trend through robots that were not walking.
      whole = survived >= 0.999
      line = np.where(whole, achieved, np.nan)
      top.fill_between(
        values,
        np.where(whole, spread_lo, np.nan),
        np.where(whole, spread_hi, np.nan),
        color=colour,
        alpha=0.16,
        linewidth=0,
      )
      top.plot(values, line, color=colour, linewidth=2.0, zorder=3)
      top.plot(
        values[whole],
        achieved[whole],
        linestyle="none",
        marker="o",
        markersize=4.0,
        color=colour,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        zorder=4,
        label=controller.label,
      )
      top.plot(
        values[~whole],
        achieved[~whole],
        linestyle="none",
        marker="o",
        markersize=4.0,
        markerfacecolor=SURFACE,
        markeredgecolor=colour,
        markeredgewidth=1.2,
        zorder=4,
      )
      bottom.plot(values, line - values, color=colour, linewidth=2.0, zorder=3)
      bottom.fill_between(
        values,
        np.where(whole, spread_lo - values, np.nan),
        np.where(whole, spread_hi - values, np.nan),
        color=colour,
        alpha=0.16,
        linewidth=0,
      )
      bottom.plot(
        values[~whole],
        (achieved - values)[~whole],
        linestyle="none",
        marker="o",
        markersize=3.4,
        markerfacecolor=SURFACE,
        markeredgecolor=colour,
        markeredgewidth=1.0,
        zorder=4,
      )

    span = (min(limits), max(limits))
    top.plot(span, span, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    top.annotate(
      "perfect tracking",
      xy=(span[1], span[1]),
      xytext=(-4, 6),
      textcoords="offset points",
      ha="right",
      color=MUTED,
      fontsize=7.5,
    )
    bottom.axhline(0.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for ax in (top, bottom):
      ax.axvline(0.0, color=GRID, linewidth=0.8, zorder=0)
      ax.set_xlim(*span)
    unit = AXIS_UNIT[axis]
    top.set_title(f"{AXIS_LABEL[axis]} command sweep", loc="left", pad=6)
    top.set_ylabel(f"achieved ({unit})")
    bottom.set_ylabel(f"error ({unit})")
    bottom.set_xlabel(f"commanded {AXIS_LABEL[axis]} ({unit})")

  handles = [
    Line2D(
      [],
      [],
      color=controller.colour,
      linewidth=2.4,
      marker="o",
      markersize=5,
      markeredgecolor=SURFACE,
      label=controller.label,
    )
    for controller in controllers
  ]
  handles.append(
    Line2D(
      [],
      [],
      color=INK_2,
      linestyle="none",
      marker="o",
      markersize=5,
      markerfacecolor=SURFACE,
      label="at least one robot fell",
    )
  )
  fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=min(len(handles), 4),
    columnspacing=2.0,
  )
  fig.suptitle(
    "Steady-state velocity tracking, one axis at a time",
    x=0.008,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  reference = sweeps[controllers[0].name]["sweep_vx"]
  band = "Band spans the replicas of a command"
  if any(controller.engine == "quintic" for controller in controllers):
    band += "; the engine is deterministic, so its band is a line"
  fig.text(
    0.008,
    0.012,
    f"Evaluation plant, {reference.duration:.0f} s per command with the first "
    f"{reference.warmup:.0f} s discarded. {band}.",
    fontsize=7.5,
    color=MUTED,
  )
  fig.tight_layout(rect=(0, 0.03, 1, 0.92))
  save(fig, path)


# --------------------------------------------------------------------------
# Figures 3 and 4: the command plane
# --------------------------------------------------------------------------


# Fixed vertical margins of the command-plane figure, in inches: the title and
# the legend-height above the panels, the caption below them, and the height one
# controller's row of panels gets. Kept in inches rather than figure fractions so
# adding a controller adds a row instead of shrinking every row.
HEADER_IN = 1.12
FOOTER_IN = 0.61
ROW_IN = 2.74

GRID_PAIRS = (
  ("vx", "vy", "grid_vx_vy"),
  ("vx", "wz", "grid_vx_wz"),
  ("vy", "wz", "grid_vy_wz"),
)


def grid_field(sweep: Sweep, x: str, y: str, field: str) -> tuple[np.ndarray, ...]:
  """One metric reshaped onto the two command axes it was swept over."""
  xs = np.unique(sweep.data[f"command_{x}"])
  ys = np.unique(sweep.data[f"command_{y}"])
  values = np.full((ys.size, xs.size), np.nan)
  x_index = {value: i for i, value in enumerate(xs)}
  y_index = {value: i for i, value in enumerate(ys)}
  for row in range(sweep.data[field].size):
    i = y_index[sweep.data[f"command_{y}"][row]]
    j = x_index[sweep.data[f"command_{x}"][row]]
    values[i, j] = sweep.data[field][row]
  return xs, ys, values


def _pcolor(ax, xs, ys, values, **kwargs):
  """Cell-centred mesh, with the missing cells shown as missing."""
  ax.set_facecolor("#eceae4")
  return ax.pcolormesh(
    _edges(xs), _edges(ys), np.ma.masked_invalid(values), shading="flat", **kwargs
  )


def _edges(values: np.ndarray) -> np.ndarray:
  step = np.diff(values)
  return np.concatenate(
    [[values[0] - step[0] / 2], values[:-1] + step / 2, [values[-1] + step[-1] / 2]]
  )


def figure_command_plane(
  controllers: list[Controller],
  sweeps: dict[str, dict[str, Sweep]],
  path: Path,
  field: str,
  title: str,
  subtitle: str,
) -> None:
  """One metric over three command planes, one row per controller."""
  rows = len(controllers)
  # Placed by hand rather than by tight_layout: the row labels and the colour
  # bar sit at figure coordinates, so the panels have to be somewhere known.
  # The margins are fixed in inches and the panels take what is left, so a
  # three-controller figure is a taller version of a two-controller one rather
  # than the same figure with the panels squeezed.
  height = HEADER_IN + FOOTER_IN + ROW_IN * rows
  fig, axes = plt.subplots(rows, 3, figsize=(12.2, height), squeeze=False)
  top = 1.0 - HEADER_IN / height
  bottom = FOOTER_IN / height
  fig.subplots_adjust(
    left=0.072, right=0.895, top=top, bottom=bottom, wspace=0.40, hspace=0.50
  )
  if field == "tracking_error":
    vmax = 0.0
    for controller in controllers:
      for _, _, key in GRID_PAIRS:
        vmax = max(
          vmax, float(np.nanpercentile(sweeps[controller.name][key].data[field], 98))
        )
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = SEQUENTIAL
    label = "planar velocity error (m/s)"
  else:
    duration = sweeps[controllers[0].name][GRID_PAIRS[0][2]].duration
    norm = Normalize(vmin=0.0, vmax=duration)
    cmap = SEQUENTIAL
    label = "time upright (s)"

  mesh = None
  for row, controller in enumerate(controllers):
    for column, (x, y, key) in enumerate(GRID_PAIRS):
      ax = axes[row, column]
      despine(ax)
      ax.grid(False)
      sweep = sweeps[controller.name][key]
      if field == "time_upright":
        values = np.where(
          np.isnan(sweep.data["fall_time"]),
          sweep.duration,
          sweep.data["fall_time"],
        )
        source = Sweep(
          controller, {**sweep.data, "time_upright": values}, sweep.summary
        )
      else:
        source = sweep
      xs, ys, field_values = grid_field(source, x, y, field)
      mesh = _pcolor(ax, xs, ys, field_values, cmap=cmap, norm=norm)

      # The stability envelope is outlined on both maps. On the error map it
      # says which cells are a measurement of walking: outside the outline the
      # robot fell, and its error is an average over the seconds before it did.
      _, _, survived = grid_field(sweep, x, y, "survived")
      held = np.nan_to_num(survived, nan=0.0) > 0.5
      ax.contour(xs, ys, held, levels=[0.5], colors=[INK], linewidths=1.2)
      if field == "time_upright":
        if held.all():
          # An envelope with no boundary inside the plane draws no outline, so
          # say what the reader is looking at instead of leaving a solid block.
          ax.annotate(
            f"every command held\nfor the full {sweep.duration:.0f} s",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=10,
            fontweight="bold",
            linespacing=1.5,
          )
      ax.axhline(0.0, color=SURFACE, linewidth=0.6, alpha=0.6)
      ax.axvline(0.0, color=SURFACE, linewidth=0.6, alpha=0.6)
      ax.set_xlabel(f"commanded {AXIS_LABEL[x]} ({AXIS_UNIT[x]})")
      ax.set_ylabel(f"commanded {AXIS_LABEL[y]} ({AXIS_UNIT[y]})")
      ax.set_title(f"{AXIS_LABEL[x]} × {AXIS_LABEL[y]}", loc="left", pad=6, color=INK)
    # Read off the row's own panels rather than from a table of positions, so
    # the label follows the panels however many rows there are.
    fig.text(
      0.072,
      axes[row, 0].get_position().y1 + 0.28 / height,
      controller.label,
      fontsize=11,
      fontweight="bold",
      color=controller.colour,
    )

  assert mesh is not None
  bar = fig.colorbar(mesh, cax=fig.add_axes((0.918, bottom, 0.014, top - bottom)))
  bar.set_label(label, color=INK_2, fontsize=8.5)
  hide(bar.outline)
  bar.ax.tick_params(colors=MUTED, labelsize=8)
  fig.suptitle(
    title,
    x=0.008,
    y=1.0 - 0.18 / height,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  fig.text(0.008, 0.13 / height, subtitle, fontsize=7.5, color=MUTED)
  save(fig, path, tight=False)


# --------------------------------------------------------------------------
# Figure 5: stability against time
# --------------------------------------------------------------------------


def survival_curve(
  fall_time: np.ndarray, horizon: float, points: int = 400
) -> tuple[np.ndarray, np.ndarray]:
  """Fraction of the commands still upright, against time."""
  times = np.linspace(0.0, horizon, points)
  falls = fall_time[~np.isnan(fall_time)]
  still_up = np.array(
    [1.0 - (falls <= t).sum() / fall_time.size for t in times], dtype=float
  )
  return times, still_up


def figure_stability(
  controllers: list[Controller], sweeps: dict[str, dict[str, Sweep]], path: Path
) -> None:
  """Three views of stability: over time, over the envelope, and by speed."""
  fig = plt.figure(figsize=(11.5, 7.4))
  spec = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), hspace=0.42, wspace=0.30)
  top_left = fig.add_subplot(spec[0, :2])
  top_right = fig.add_subplot(spec[0, 2])
  bottom = [fig.add_subplot(spec[1, i]) for i in range(3)]
  for ax in (top_left, top_right, *bottom):
    despine(ax)

  horizon = sweeps[controllers[0].name]["grid_vx_wz"].duration

  # (a) Survival over the whole commanded envelope.
  for controller in controllers:
    fall = np.concatenate(
      [sweeps[controller.name][key].data["fall_time"] for _, _, key in GRID_PAIRS]
    )
    times, alive = survival_curve(fall, horizon)
    top_left.plot(
      times,
      100 * alive,
      color=controller.colour,
      linewidth=2.2,
      label=controller.label,
    )
    top_left.annotate(
      f"{100 * alive[-1]:.0f}%",
      xy=(times[-1], 100 * alive[-1]),
      xytext=(5, 0),
      textcoords="offset points",
      color=controller.colour,
      fontsize=9,
      fontweight="bold",
      va="center",
      clip_on=False,
    )
  top_left.set_ylim(0, 103)
  top_left.set_xlim(0, horizon)
  top_left.set_xlabel("time walking (s)")
  top_left.set_ylabel("commands still upright (%)")
  top_left.set_title(
    "a · Survival over the whole commanded envelope", loc="left", pad=6
  )
  top_left.legend(loc="lower left", ncol=1)

  # (b) Survival at the horizon, by controller.
  labels, values, colours = [], [], []
  for controller in controllers:
    survived = np.concatenate(
      [sweeps[controller.name][key].data["survived"] for _, _, key in GRID_PAIRS]
    )
    # The full label, not its head: two policies of one comparison usually
    # differ only in the parenthesis, and trimming it would give the panel two
    # bars called the same thing.
    labels.append(controller.label)
    values.append(100 * float(survived.mean()))
    colours.append(controller.colour)
  # Drawn bottom-up so the bars read in the same order as the legend above.
  bars = top_right.barh(labels[::-1], values[::-1], color=colours[::-1], height=0.5)
  for bar, value in zip(bars, values[::-1], strict=True):
    top_right.annotate(
      f"{value:.0f}%",
      xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
      xytext=(5, 0),
      textcoords="offset points",
      va="center",
      fontsize=10,
      fontweight="bold",
      color=INK,
    )
  top_right.set_xlim(0, 108)
  top_right.set_xlabel(f"commands upright at {horizon:.0f} s (%)")
  top_right.grid(axis="y", visible=False)
  top_right.set_title("b · At the horizon", loc="left", pad=6)
  top_right.tick_params(axis="y", labelsize=8.5 if len(controllers) < 4 else 7.5)

  # (c) Survival split by how fast the robot was asked to go.
  bands = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.65))
  for ax, (low, high) in zip(bottom, bands, strict=True):
    for controller in controllers:
      command = np.concatenate(
        [
          np.hypot(
            sweeps[controller.name][key].data["command_vx"],
            sweeps[controller.name][key].data["command_vy"],
          )
          for _, _, key in GRID_PAIRS
        ]
      )
      fall = np.concatenate(
        [sweeps[controller.name][key].data["fall_time"] for _, _, key in GRID_PAIRS]
      )
      selected = (command >= low) & (command < high)
      if not selected.any():
        continue
      times, alive = survival_curve(fall[selected], horizon)
      ax.plot(times, 100 * alive, color=controller.colour, linewidth=2.0)
    ax.set_ylim(0, 103)
    ax.set_xlim(0, horizon)
    ax.set_xlabel("time walking (s)")
    ax.set_title(
      f"|commanded speed| {low:.2f}–{high:.2f} m/s", loc="left", pad=6, fontsize=9
    )
  bottom[0].set_ylabel("commands still upright (%)")
  bottom[0].annotate(
    "c · Stability depends on how fast it was asked to walk",
    xy=(0, 1.0),
    xycoords="axes fraction",
    xytext=(0, 30),
    textcoords="offset points",
    fontsize=10,
    fontweight="semibold",
    color=INK,
  )

  fig.suptitle(
    "Stability: controller, commanded speed and time",
    x=0.008,
    y=0.985,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  population = sum(
    sweeps[controllers[0].name][key].data["survived"].size for _, _, key in GRID_PAIRS
  )
  fig.text(
    0.008,
    0.012,
    f"Population is every command in the three two-axis grids ({population} per "
    "controller). A command counts as upright until the torso passes 60° from "
    "vertical.",
    fontsize=7.5,
    color=MUTED,
  )
  fig.tight_layout(rect=(0, 0.03, 1, 0.94))
  save(fig, path)


# --------------------------------------------------------------------------
# Figure 6: gait quality against speed
# --------------------------------------------------------------------------


def figure_gait_quality(
  controllers: list[Controller], sweeps: dict[str, dict[str, Sweep]], path: Path
) -> None:
  """Attitude and cadence against forward speed, survivors only."""
  panels = (
    ("rms_roll", "torso roll, RMS (rad)", "a"),
    ("rms_pitch", "torso pitch, RMS (rad)", "b"),
    ("min_upright", "worst uprightness (cos of lean)", "c"),
    ("cadence_hz", "cadence (Hz)", "d"),
  )
  fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
  for ax, (field, label, tag) in zip(axes, panels, strict=True):
    despine(ax)
    for controller in controllers:
      sweep = sweeps[controller.name]["sweep_vx"]
      values, groups = sweep.grouped("vx")
      means, keep = [], []
      for value, group in zip(values, groups, strict=True):
        # A zero command is a robot standing still, not a gait: it has no
        # cadence and almost no roll, and leaving it in puts a spike through
        # the middle of every panel.
        if value == 0.0:
          continue
        upright = sweep.data["survived"][group] > 0.5
        selected = sweep.data[field][group][upright]
        selected = selected[np.isfinite(selected)]
        if selected.size:
          means.append(float(selected.mean()))
          keep.append(value)
      ax.plot(
        keep,
        means,
        color=controller.colour,
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
        label=controller.label,
      )
    ax.set_xlabel("commanded $v_x$ (m/s)")
    ax.set_title(f"{tag} · {label}", loc="left", pad=6, fontsize=9)
  handles = [
    Line2D([], [], color=controller.colour, linewidth=2.4, label=controller.label)
    for controller in controllers
  ]
  fig.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.006, 0.955),
    ncol=min(len(handles), 3),
    columnspacing=2.0,
  )
  fig.suptitle(
    "Gait quality against forward speed, over the robots that stayed up",
    x=0.006,
    y=0.995,
    ha="left",
    fontsize=12,
    fontweight="bold",
    color=INK,
  )
  note = "A command with no survivors is absent rather than plotted at zero."
  engines = {controller.engine for controller in controllers}
  if "rl" in engines:
    note += (
      " A notch at the origin in (a) and (d) is a policy standing still: below "
      "about 0.05 m/s it stops stepping"
    )
    note += ", where the walk engine keeps marching." if "quintic" in engines else "."
  fig.text(0.006, 0.015, note, fontsize=7.5, color=MUTED)
  fig.tight_layout(rect=(0, 0.05, 1, 0.86))
  save(fig, path)


def save(fig, path: Path, tight: bool = True) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  kwargs = {"bbox_inches": "tight"} if tight else {}
  fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
  fig.savefig(path.with_suffix(".pdf"), **kwargs)
  plt.close(fig)
  print(f"wrote {path.with_suffix('.png')}")


@dataclass
class Args:
  input_dir: Path = Path("logs/eval/comparison")
  """Directory ``collect_comparison.sh`` wrote into."""
  output_dir: Path | None = None
  """Where the figures go. Defaults to ``<input-dir>/figures``."""
  controllers: str | None = None
  """Draw only these controllers, in this order: a comma-separated list of
  names, e.g. ``--controllers quintic,history``. The names are the ``name=``
  fields the collection was given, and are listed in its ``controllers.json``.
  Defaults to every controller in the directory, in the order it collected
  them."""


SWEEP_KEYS = (
  "sweep_vx",
  "sweep_vy",
  "sweep_wz",
  "grid_vx_vy",
  "grid_vx_wz",
  "grid_vy_wz",
)


def check_inputs(input_dir: Path, controllers: list[Controller]) -> None:
  """Fail before drawing anything, naming what is missing.

  Every figure but the first puts the controllers side by side, so a directory
  missing any controller's runs cannot be plotted. Checked up front because the
  alternative is half a set of figures and a bare path in a traceback: the usual
  cause is a collection that stopped part way through.
  """
  missing: list[str] = []
  incomplete: list[Controller] = []
  for controller in controllers:
    absent = [
      f"{name}/{wanted}"
      for name, wanted in (
        (f"profile_{controller.name}", "trace.csv"),
        *((f"{key}_{controller.name}", "per_env.csv") for key in SWEEP_KEYS),
      )
      if not (input_dir / name / wanted).is_file()
    ]
    missing += absent
    if absent:
      incomplete.append(controller)
  if not missing:
    return

  lines = [f"{len(missing)} run(s) missing from {input_dir}:"]
  lines += [f"  {name}" for name in missing]
  names = ", ".join(repr(controller.name) for controller in incomplete)
  verb = "has" if len(incomplete) == 1 else "have"
  lines.append(
    f"\nEvery figure but the first compares the controllers side by side, so "
    f"{names} {verb} to be collected too. If collect_comparison.sh stopped "
    f"early, "
    f"check that each engine=rl controller's checkpoint= is the path to a .pt "
    f"file rather than a wandb run id. To plot what is here instead, name the "
    f"complete controllers with --controllers."
  )
  raise SystemExit("\n".join(lines))


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  use_house_style()
  out = args.output_dir or (args.input_dir / "figures")
  controllers = load_controllers(args.input_dir, args.controllers)
  check_inputs(args.input_dir, controllers)

  sweeps: dict[str, dict[str, Sweep]] = {}
  for controller in controllers:
    sweeps[controller.name] = {
      key: load_sweep(args.input_dir / f"{key}_{controller.name}", controller)
      for key in SWEEP_KEYS
    }
    trace = load_trace(args.input_dir / f"profile_{controller.name}", controller)
    figure_profile(trace, out / f"fig1_profile_{controller.name}")

  horizon = sweeps[controllers[0].name]["grid_vx_wz"].duration
  figure_tracking(controllers, sweeps, out / "fig2_tracking_axes")
  figure_command_plane(
    controllers,
    sweeps,
    out / "fig3_tracking_plane",
    field="tracking_error",
    title="Tracking error over the command plane",
    subtitle=(
      "Planar velocity error, |achieved − commanded|. Grey cells fell before "
      "the measurement window opened; the outline encloses the commands the "
      "controller held for the whole run, so only inside it is the error a "
      "measurement of walking."
    ),
  )
  figure_command_plane(
    controllers,
    sweeps,
    out / "fig4_stability_plane",
    field="time_upright",
    title="Stability envelope over the command plane",
    subtitle=(
      f"Seconds upright out of {horizon:.0f}. The outline encloses the commands "
      "the controller held for the whole run."
    ),
  )
  figure_stability(controllers, sweeps, out / "fig5_stability_time")
  figure_gait_quality(controllers, sweeps, out / "fig6_gait_quality")


if __name__ == "__main__":
  main()
