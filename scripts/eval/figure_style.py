"""House style for the evaluation figures.

Shared by every plotting script in this directory so that a figure drawn from a
simulated run and one drawn from a motion-capture log can sit on the same page
without looking like they came from different laboratories. Colours, the
matplotlib defaults, and the two helpers every figure ends with -- despine and
save -- live here and nowhere else.

Nothing in this module knows what is being plotted. The scripts that import it
own the data, the panels and the prose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Palette. Controllers take categorical slots in the order they were collected;
# where a panel is split by command axis instead, the three axes take slots 1 to
# 3. Chart chrome is the ink scale, never a series colour.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
DAQUA = "#0e736f"
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

About one gait cycle at the engine's 0.32 s step period. The raw signal swings
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

def use_nemo_style() -> None:
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
  """Centred moving average, edges shortened rather than padded."""
  if window <= 1:
    return values
  kernel = np.ones(window) / window
  padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
  return np.convolve(padded, kernel, mode="valid")


def save(fig, path: Path, tight: bool = True) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  kwargs = {"bbox_inches": "tight"} if tight else {}
  fig.savefig(path.with_suffix(".png"), dpi=300, **kwargs)
  fig.savefig(path.with_suffix(".pdf"), **kwargs)
  plt.close(fig)
  print(f"wrote {path.with_suffix('.png')}")
