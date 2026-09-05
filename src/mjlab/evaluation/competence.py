"""Per-episode competence over a command x disturbance grid.

The training-time competence tracker
(:mod:`mjlab.tasks.velocity.mdp.competence`) exists to give a curriculum
controller a smoothed population signal, so it EMAs its per-episode statistics
and initialises those EMAs pessimistically. Both are wrong for measurement: an
EMA read offline is a filtered statistic still carrying its initialisation, and
a population mean hides the cells that matter. What is useful is the layer
underneath -- the per-env accumulators collapsed to one number per episode at
reset -- kept disaggregated.

This module is that layer. The definitions are the tracker's, transcribed:

============ ===================================================================
``attain``   Delivered speed projected on the command, as a fraction of it.
             Sampled only on steps with a meaningful command (``|c| >= 0.15``);
             sandbagging reads ~0 without a single fall, and lateral sway is
             orthogonal so it drops out.
``attain_x`` Signed per-axis achieved/commanded, each sample weighted by that
``attain_y`` axis's share of command energy, sampled where ``|c_axis| >= 0.10``.
             Kept separate because "asked for lateral, delivered forward" is a
             distinct failure the scalar hides.
``wobble``   Seconds from the first tilt past 25 degrees to the fall, for
``_lead``    episodes that fell. How much warning the near-miss channel gave
             before the failure it precedes; undefined where nothing fell.
``fell``     Binary, from the ``fell_over`` termination.
``ep_len``   Episode length over the maximum. Survival, which disambiguates a
``_frac``    low ``attain`` caused by early termination from one caused by
             sandbagging.
============ ===================================================================

Two of the tracker's channels are deliberately absent. ``fast_fall_rate`` is a
one-iteration control window and is meaningless offline; ``track_err_norm`` is
legacy and is already out of the tracker's own predicates.

Nothing here reads the environment. The input is
:class:`~mjlab.evaluation.metrics.EvalState` plus the commanded twist, which is
what the quintic and distilled harnesses hand their metrics too, so the same
numbers can be produced for a controller that never sees a
``ManagerBasedRlEnv``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from mjlab.evaluation.metrics import EvalState, upright_from_quat

WOBBLE_GRAVITY_XY = 0.4226
"""Projected-gravity ``xy`` norm above which a step counts as wobbling.

``sin(25 degrees)``. The same literal the training tracker compares against, so
an eval wobble and a training wobble are the same measurement. Well short of
the 50 degrees ``fell_over`` terminates at, which is the point: wobble is the
graded precursor that shows stress in episodes that never terminate.
"""

MIN_COMMAND_NORM = 0.15
"""Commanded planar speed below which ``attain`` is not sampled, in m/s.

Attainment is achieved-over-commanded, so a vanishing command makes it a ratio
of two small numbers. The tracker's answer is not a denominator floor -- which
would cap perfect tracking of a small command -- but to refuse the sample. Cells
under this bar therefore have *no* attainment evidence at all and are reported
undefined rather than zero; they are characterised by ``wobble`` and ``fell``.
"""

FELL_OVER_UPRIGHT = math.cos(math.radians(50.0))
"""Torso up-axis component below which an episode counts as a fall.

``cos(50 degrees)``, which is the ``bad_orientation`` limit the velocity task's
``fell_over`` termination uses. A reinforcement-learned policy's ``fell`` comes
from that termination directly; a scripted engine has no termination manager, so
this reproduces its bound from the same raw state.

Deliberately *not*
:data:`~mjlab.evaluation.metrics.FALL_UPRIGHT_THRESHOLD`, which is
``cos(60 degrees)``. That one dates the moment a robot stopped walking, and 10
degrees of extra grace is the right call when the question is when to stop
averaging. Here the question is whether this episode is one of the ones the
policy was terminated for, and answering it with a looser bound would let a
scripted engine tip further before being counted as fallen than a policy is
allowed to.
"""

MIN_AXIS_COMMAND = 0.10
"""Per-axis commanded speed below which ``attain_x``/``attain_y`` is not
sampled, in m/s. The same refusal as :data:`MIN_COMMAND_NORM`, applied per
axis so a command that is all forward contributes no lateral sample."""


def projected_gravity_xy_norm(quaternion_w: torch.Tensor) -> torch.Tensor:
  """Norm of the body-frame gravity direction's horizontal part.

  The tilt of the torso away from vertical, as a sine: 0 upright, 1 on its
  side. This is what the training tracker thresholds for wobble, there read
  straight off ``projected_gravity_b``; here it comes from the quaternion,
  because that is what an :class:`~mjlab.evaluation.metrics.EvalState` carries
  and every controller in the comparison produces one.

  Body-frame gravity is ``R^T (0, 0, -1)``, i.e. the negated third row of the
  body-to-world rotation, whose horizontal part has norm
  ``2 * hypot(xz - wy, yz + wx)``.
  """
  w = quaternion_w[:, 0]
  x = quaternion_w[:, 1]
  y = quaternion_w[:, 2]
  z = quaternion_w[:, 3]
  return 2.0 * torch.sqrt((x * z - w * y) ** 2 + (y * z + w * x) ** 2)


def episode_end(
  state: EvalState,
  episode_step: torch.Tensor,
  max_episode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Which environments' episodes end on this step, and which of them fell.

  The termination rule of the velocity task, reproduced from raw state for the
  harnesses that have no termination manager: an episode ends when the torso
  tips past :data:`FELL_OVER_UPRIGHT` (a fall) or when it reaches
  ``max_episode_steps`` (a timeout).

  Args:
    state: The robot's state after the step.
    episode_step: Shape ``(N,)`` steps elapsed *including* this one.
    max_episode_steps: Length of a full episode, in control steps.

  Returns:
    ``(done, fell)``, both shape ``(N,)`` bool. ``fell`` implies ``done``; an
    environment that tips on its last step is recorded as a fall rather than a
    timeout, which is the order the termination manager resolves them in.
  """
  fell = upright_from_quat(state.quaternion_w) < FELL_OVER_UPRIGHT
  return fell | (episode_step >= max_episode_steps), fell


@dataclass(frozen=True)
class GridCell:
  """One command x disturbance cell of the evaluation grid."""

  vx: float
  vy: float
  wz: float
  shove: float
  """Magnitude of every shove delivered in this cell, as ``|dv_xy|`` in m/s.
  Zero is the undisturbed row."""

  @property
  def command_norm(self) -> float:
    """Commanded planar speed, in m/s. Below :data:`MIN_COMMAND_NORM` this
    cell has no attainment evidence."""
    return math.hypot(self.vx, self.vy)

  @property
  def attain_defined(self) -> bool:
    return self.command_norm >= MIN_COMMAND_NORM

  @property
  def label(self) -> str:
    return f"vx={self.vx:+.2f} vy={self.vy:+.2f} wz={self.wz:+.2f} dv={self.shove:.2f}"


@dataclass(frozen=True)
class CompetenceGrid:
  """A grid of cells, tiled over the environments of one batch.

  Every environment holds its cell for the whole run: the command is re-pinned
  at each reset, so an environment produces a stream of episodes all belonging
  to one cell, and episodes accumulate per cell until the run has enough of
  them to report a spread.
  """

  cells: tuple[GridCell, ...]
  cell_index: torch.Tensor
  """Shape ``(N,)`` index into :attr:`cells` for each environment."""
  command: torch.Tensor
  """Shape ``(N, 3)`` commanded ``(vx, vy, wz)`` for each environment."""
  shove: torch.Tensor
  """Shape ``(N,)`` shove magnitude for each environment, in m/s."""

  @property
  def num_envs(self) -> int:
    return int(self.cell_index.numel())

  def envs_per_cell(self) -> torch.Tensor:
    """Shape ``(len(cells),)`` count of environments assigned to each cell."""
    return torch.bincount(self.cell_index.cpu(), minlength=len(self.cells))


def build_grid(
  commands: tuple[tuple[float, float, float], ...],
  shoves: tuple[float, ...],
  num_envs: int,
  device: torch.device | str = "cpu",
) -> CompetenceGrid:
  """Cross ``commands`` with ``shoves`` and tile the product over ``num_envs``.

  Tiled rather than blocked: consecutive environments get different cells, so a
  batch truncated by ``num_envs`` not dividing the grid loses at most one
  episode-slot from each of the last few cells instead of dropping whole cells.

  Raises:
    ValueError: If either axis is empty, or the batch is smaller than the grid
      (which would leave cells with no environment at all).
  """
  cells = tuple(
    GridCell(vx=vx, vy=vy, wz=wz, shove=shove)
    for shove in shoves
    for (vx, vy, wz) in commands
  )
  if not cells:
    raise ValueError("grid is empty: both commands and shoves must be non-empty")
  if num_envs < len(cells):
    raise ValueError(
      f"{num_envs} environments cannot cover {len(cells)} cells "
      f"({len(commands)} commands x {len(shoves)} shove bins)"
    )
  index = torch.arange(num_envs, device=device) % len(cells)
  command = torch.tensor(
    [[c.vx, c.vy, c.wz] for c in cells], device=device, dtype=torch.float32
  )[index]
  shove = torch.tensor([c.shove for c in cells], device=device, dtype=torch.float32)[
    index
  ]
  return CompetenceGrid(cells=cells, cell_index=index, command=command, shove=shove)


@dataclass(frozen=True)
class EpisodeTable:
  """One row per completed episode.

  Every field is a shape ``(E,)`` tensor except :attr:`wobble_steps_index`,
  which is ragged -- one variable-length tensor per episode -- because the
  number of wobbly steps is what it reports.

  This is the raw material the figures are built from, and it is what gets
  written to disk. Deliberately not aggregated: the interesting cells are the
  high-variance ones, and a cell's mean cannot show that.
  """

  cell: torch.Tensor
  """Index into the grid's cells."""
  command_vx: torch.Tensor
  command_vy: torch.Tensor
  command_wz: torch.Tensor
  shove: torch.Tensor
  """Shove magnitude in force during the episode, as ``|dv_xy|`` in m/s."""
  attain: torch.Tensor
  """Delivered speed projected on the command, over commanded speed. NaN where
  the command never cleared :data:`MIN_COMMAND_NORM`."""
  attain_x: torch.Tensor
  """Signed achieved/commanded on the forward axis. NaN where unsampled."""
  attain_y: torch.Tensor
  wobble: torch.Tensor
  """Fraction of the episode's steps tilted past 25 degrees.

  Kept as raw material -- it is what ``num_wobble_steps / steps`` comes to --
  but it is no longer the reported wobble channel. Averaged over an episode it
  answers "how much of the run was spent tilted", which mixes a robot that
  wobbled briefly and recovered with one that wobbled briefly and fell, and
  divides both by an episode length that a fall cuts short.
  """
  wobble_lead: torch.Tensor
  """Seconds from the first tilt past 25 degrees to the end of the episode.

  Measured per *fall*, not per episode: NaN wherever the episode timed out,
  because there is no termination to measure to and a robot that wobbled and
  recovered is not a near-miss of anything. Zero where the episode fell without
  a recorded crossing, which means the torso went from under 25 degrees to past
  the 50 the termination fires at inside one control step -- a fall with no
  warning at all, which is the reading that matters and is distinct from not
  having fallen.

  This is what the wobble channel is for: 25 degrees is not a failure, it is
  the precursor to one, and the useful question is how long the precursor ran
  before the failure arrived.
  """
  fell: torch.Tensor
  """1.0 if the episode ended in the ``fell_over`` termination."""
  ep_len_frac: torch.Tensor
  """Episode length over the maximum episode length."""
  shoves_taken: torch.Tensor
  """Shoves actually delivered during the episode. An episode that fell early
  took fewer than a full one, which is why the dose is reported rather than
  assumed."""

  steps: torch.Tensor
  """Steps the episode's averages were taken over.

  The denominator of :attr:`wobble`, so ``num_wobble_steps / steps`` reproduces
  it exactly. One short of the episode's length whenever the episode ended,
  because the step it ended on is excluded from the averages -- see
  :meth:`EpisodeCompetence.record`.
  """
  ep_len: torch.Tensor
  """Length of the episode, in seconds.

  Seconds rather than steps because a step is not comparable across
  controllers: the walk engine runs at 100 Hz and a policy at 50, so the same
  twenty-second episode is 2000 steps for one and 1000 for the other. The step
  count is :attr:`steps` (plus the terminal step), and the fraction of a full
  episode is :attr:`ep_len_frac`.
  """
  num_wobble_steps: torch.Tensor
  """Steps of the episode tilted past 25 degrees, as a count rather than the
  fraction :attr:`wobble` reports."""
  wobble_steps_index: tuple[torch.Tensor, ...]
  """Which steps those were: one tensor per episode of 0-based step indices.

  Ragged, since an episode has as many entries as it had wobbly steps and most
  have none. Indices count control steps from the start of the episode, so they
  are on the same rate-dependent footing as :attr:`steps`; multiply by the
  control period for seconds.
  """

  def column_names(self) -> list[str]:
    return [f.name for f in fields(self)]

  def rows(self) -> list[list[object]]:
    """Per-episode rows, in :meth:`column_names` order.

    The ragged column is rendered as space-separated integers, which keeps one
    row per episode and needs no quoting. It is empty for an episode that never
    wobbled, which is most of them.
    """
    columns: list[list[object]] = []
    for name in self.column_names():
      value = getattr(self, name)
      if name == "wobble_steps_index":
        columns.append([" ".join(str(i) for i in row.tolist()) for row in value])
      else:
        columns.append(list(value.tolist()))
    return [list(row) for row in zip(*columns, strict=True)]

  @property
  def num_episodes(self) -> int:
    return int(self.cell.numel())


class EpisodeCompetence:
  """Accumulates control steps into one :class:`EpisodeTable` row per episode.

  The accumulators are per environment and are zeroed the moment an episode
  closes, so nothing carries across a reset: each row is an independent sample
  from the cell's distribution rather than a point on a trajectory.
  """

  def __init__(
    self,
    grid: CompetenceGrid,
    max_episode_steps: int,
    step_dt: float,
    device: torch.device | str,
  ) -> None:
    """
    Args:
      grid: The cell assignment; supplies the per-environment command.
      max_episode_steps: Denominator for ``ep_len_frac``, i.e. the environment's
        ``max_episode_length``.
      step_dt: Control period in seconds, which turns a step count into the
        ``ep_len`` an engine and a policy can be compared on.
      device: Device the accumulators live on.
    """
    if max_episode_steps <= 0:
      raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")
    self.grid = grid
    self.max_episode_steps = max_episode_steps
    self.step_dt = float(step_dt)
    self.device = torch.device(device)

    n = grid.num_envs
    self.command = grid.command.to(self.device)
    self.shove = grid.shove.to(self.device)
    self.cell_index = grid.cell_index.to(self.device)
    self._cmd_xy = self.command[:, :2]
    self._cmd_sq = (self._cmd_xy * self._cmd_xy).sum(dim=-1)
    # The command is fixed for the run, so the sampling masks and the per-axis
    # energy weights are too. Computing them once is not an optimisation: it is
    # the statement that a cell either has attainment evidence or has none.
    self._meaningful = self._cmd_sq >= MIN_COMMAND_NORM**2
    self._axis_weight = self._cmd_xy**2 / self._cmd_sq.clamp(min=1e-6).unsqueeze(-1)
    self._axis_sampled = self._cmd_xy.abs() >= MIN_AXIS_COMMAND

    zeros = torch.zeros(n, device=self.device)
    self.episode_step = torch.zeros(n, dtype=torch.long, device=self.device)
    """Shape ``(N,)`` control steps elapsed in each environment's current
    episode. Read by the shove driver to place its onsets."""
    self._attain_sum = zeros.clone()
    self._attain_weight = zeros.clone()
    self._attain_axis_sum = torch.zeros(n, 2, device=self.device)
    self._attain_axis_weight = torch.zeros(n, 2, device=self.device)
    self._wobble_sum = zeros.clone()
    self._sample_steps = zeros.clone()
    self._shoves = zeros.clone()
    # Which steps of the current episode wobbled, as a bitmap rather than a
    # growing list per environment: one bool per env per step is a few megabytes
    # at any batch this pipeline runs, and it makes the per-episode extraction
    # one masked nonzero instead of a Python append on every step.
    self._wobble_mask = torch.zeros(
      n, max_episode_steps, dtype=torch.bool, device=self.device
    )
    self._rows: list[dict[str, torch.Tensor]] = []
    # Kept beside ``_rows`` and appended in the same breath, because the ragged
    # column has to stay aligned with the columnar ones and nothing else
    # enforces that.
    self._wobble_indices: list[torch.Tensor] = []
    self._completed = torch.zeros(len(grid.cells), dtype=torch.long, device=self.device)

  def note_shoves(self, env_ids: torch.Tensor) -> None:
    """Count a delivered shove against the current episode of each env."""
    self._shoves[env_ids] += 1.0

  def record(self, state: EvalState, done: torch.Tensor, fell: torch.Tensor) -> None:
    """Accumulate one control step, closing out any episode that ended on it.

    Args:
      state: The robot's state after the step.
      done: Shape ``(N,)`` bool, environments whose episode ended on this step.
      fell: Shape ``(N,)`` bool, of those, the ones that ended by falling.

    The environment auto-resets inside its own ``step``, so by the time a
    harness can read the state back, a ``done`` environment is already standing
    at its reset pose. That one terminal sample is therefore excluded from the
    averages -- it belongs to the next episode, not this one. The episode
    *length* still counts it, because the step was taken. The loss is one
    sample in several hundred, and it is the one sample whose exclusion is
    conservative: an episode about to trip the 50-degree fall termination has
    already spent many steps past the 25-degree wobble bar.
    """
    self.episode_step += 1
    live = ~done
    weight = live.float()

    tilt = projected_gravity_xy_norm(state.quaternion_w)
    wobbling = (tilt > WOBBLE_GRAVITY_XY) & live
    self._wobble_sum += wobbling.float()
    self._sample_steps += weight
    # 0-based within the episode, and episode_step was incremented above, so
    # the step just taken is the one before it. Clamped only so a caller that
    # runs an episode past its stated length cannot index out of the bitmap.
    index = (self.episode_step - 1).clamp(0, self.max_episode_steps - 1)
    self._wobble_mask[wobbling, index[wobbling]] = True

    vel_xy = state.lin_vel_b[:, :2]
    attain = (vel_xy * self._cmd_xy).sum(dim=-1) / self._cmd_sq.clamp(min=1e-6)
    sampled = weight * self._meaningful.float()
    self._attain_sum += sampled * attain
    self._attain_weight += sampled

    # Per-axis: achieved/commanded, signed, so backpedalling reads negative.
    # The clamp only guards the masked-out lanes; where the mask holds,
    # |c_axis| >= MIN_AXIS_COMMAND makes the division well defined.
    axis_attain = vel_xy / torch.where(
      self._axis_sampled, self._cmd_xy, torch.ones_like(self._cmd_xy)
    )
    axis_weight = self._axis_weight * self._axis_sampled.float() * weight.unsqueeze(-1)
    self._attain_axis_sum += axis_weight * axis_attain
    self._attain_axis_weight += axis_weight

    if done.any():
      self._close(done, fell)

  def _close(self, done: torch.Tensor, fell: torch.Tensor) -> None:
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    nan = torch.full((len(ids),), float("nan"), device=self.device)

    def ratio(sums: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
      w = weights[ids]
      return torch.where(w > 0, sums[ids] / w.clamp(min=1e-6), nan)

    axis = torch.where(
      self._attain_axis_weight[ids] > 0,
      self._attain_axis_sum[ids] / self._attain_axis_weight[ids].clamp(min=1e-6),
      nan.unsqueeze(-1),
    )
    steps = self._sample_steps[ids]
    self._rows.append(
      {
        "cell": self.cell_index[ids],
        "command_vx": self.command[ids, 0],
        "command_vy": self.command[ids, 1],
        "command_wz": self.command[ids, 2],
        "shove": self.shove[ids],
        "attain": ratio(self._attain_sum, self._attain_weight),
        "attain_x": axis[:, 0],
        "attain_y": axis[:, 1],
        "wobble": torch.where(
          steps > 0, self._wobble_sum[ids] / steps.clamp(min=1), nan
        ),
        "fell": fell[ids].float(),
        "ep_len_frac": self.episode_step[ids].float() / self.max_episode_steps,
        "shoves_taken": self._shoves[ids],
        "steps": steps,
        "ep_len": self.episode_step[ids].float() * self.step_dt,
        "num_wobble_steps": self._wobble_sum[ids],
        "wobble_lead": self._wobble_lead(ids, fell[ids]),
      }
    )
    # One ragged entry per closing episode, in the same order as the columns
    # above: nonzero returns row-major, so splitting by the per-row counts
    # recovers each episode's indices without a loop over environments.
    closing = self._wobble_mask[ids]
    counts = closing.sum(dim=1)
    columns = closing.nonzero(as_tuple=False)[:, 1].to(torch.int32)
    self._wobble_indices.extend(torch.split(columns, counts.tolist()))
    self._completed += torch.bincount(
      self.cell_index[ids], minlength=len(self.grid.cells)
    )

    self.episode_step[ids] = 0
    self._attain_sum[ids] = 0.0
    self._attain_weight[ids] = 0.0
    self._attain_axis_sum[ids] = 0.0
    self._attain_axis_weight[ids] = 0.0
    self._wobble_sum[ids] = 0.0
    self._sample_steps[ids] = 0.0
    self._shoves[ids] = 0.0
    self._wobble_mask[ids] = False

  def _wobble_lead(self, ids: torch.Tensor, fell: torch.Tensor) -> torch.Tensor:
    """Seconds from the first 25-degree crossing to the end of the episode.

    Only for the episodes that fell; the rest get NaN, because a wobble that
    was recovered from is not the precursor to anything and has no termination
    to be measured against.

    The first crossing is the first set bit of the episode's wobble bitmap.
    ``argmax`` on a boolean row returns the first maximum, which is the first
    ``True``; a row with no crossing returns 0, so it is masked out by the
    count rather than trusted. A fallen episode with no recorded crossing gets
    zero: the terminal step is not sampled, so a torso that went from under 25
    degrees to past 50 inside one step leaves no bit set, and no warning is
    exactly what happened. For the same reason the smallest non-zero lead is
    two control periods -- the last step that can carry a crossing, plus the
    unsampled step the episode ended on.
    """
    closing = self._wobble_mask[ids]
    first = closing.float().argmax(dim=1)
    crossed = closing.any(dim=1)
    # Measured from the start of the crossing step to the end of the episode,
    # not to the last step sampled. The two differ by one control period, and
    # only the first is rate-invariant: the same fall recorded at 50 Hz and at
    # 100 Hz has to come to the same number of seconds, or an engine's warning
    # time and a policy's are not comparable.
    lead = (self.episode_step[ids] - first).clamp(min=0).float() * self.step_dt
    lead = torch.where(crossed, lead, torch.zeros_like(lead))
    return torch.where(fell, lead, torch.full_like(lead, float("nan")))

  @property
  def completed_per_cell(self) -> torch.Tensor:
    """Shape ``(len(cells),)`` episodes closed so far in each cell."""
    return self._completed

  @property
  def min_completed(self) -> int:
    """Episodes closed in the least-sampled cell. The run's stopping test."""
    return int(self._completed.min())

  def table(self) -> EpisodeTable:
    """Concatenate the closed episodes. Episodes still in flight are dropped.

    Dropping them is the point: an episode truncated by the end of the run has
    a censored length and a partial dose of shoves, and counting it would pull
    ``ep_len_frac`` down for reasons that have nothing to do with the robot.
    """
    names = [f.name for f in fields(EpisodeTable) if f.name != "wobble_steps_index"]
    if not self._rows:
      empty = {name: torch.zeros(0, device=self.device) for name in names}
      return EpisodeTable(**empty, wobble_steps_index=())
    columns = {
      name: torch.cat([row[name].float() for row in self._rows]) for name in names
    }
    assert len(self._wobble_indices) == len(columns["cell"])
    return EpisodeTable(**columns, wobble_steps_index=tuple(self._wobble_indices))


@dataclass(frozen=True)
class ShoveCfg:
  """Timing of the deterministic shove train.

  Every episode in a cell takes the same shoves at the same times, so the cell
  is a disturbance *regime* -- "can it keep delivering under this much hit" --
  rather than a single event whose outcome depends on where in the episode it
  landed.
  """

  settle: float = 3.0
  """Seconds of undisturbed walking before the first shove, so the robot is at
  steady state rather than still accelerating out of its reset pose."""
  period: float = 4.0
  """Seconds between shoves. Comfortably longer than the recovery the training
  tracker measures, so one event is over before the next arrives."""
  tail: float = 2.0
  """Seconds at the end of the episode with no shove, so the last event has
  room to resolve inside the episode that owns it."""

  def onsets(self, dt: float, max_episode_steps: int) -> tuple[int, ...]:
    """Episode step indices at which a shove lands."""
    first = int(round(self.settle / dt))
    stride = max(1, int(round(self.period / dt)))
    last = max_episode_steps - int(round(self.tail / dt))
    return tuple(range(first, last, stride))


DEFAULT_SHOVE = ShoveCfg()
"""The shove timing every run uses unless told otherwise. A module singleton
because :class:`ShoveCfg` is frozen, so one instance is safely shared."""


class ShoveDriver:
  """Adds a planar velocity impulse of a fixed magnitude at fixed times.

  The disturbance is the training push
  (:func:`~mjlab.tasks.velocity.mdp.competence.push_cohort_by_setting_velocity`)
  driven deterministically instead of sampled: there the world-frame ``dv`` is
  drawn from a box and the resulting ``|dv_xy|`` is merely *observed*, so the
  survival frontier has to bin it after the fact. Here the magnitude is the
  cell, and only the heading is drawn -- which marginalises direction rather
  than confounding it with magnitude.

  Planar only. The training event also kicks roll and pitch; including those
  would mean the cell's ``|dv_xy|`` no longer described the whole disturbance.
  """

  def __init__(
    self,
    magnitude: torch.Tensor,
    robot,
    dt: float,
    max_episode_steps: int,
    cfg: ShoveCfg = DEFAULT_SHOVE,
    generator: torch.Generator | None = None,
  ) -> None:
    """
    Args:
      magnitude: Shape ``(N,)`` ``|dv_xy|`` for each environment, in m/s.
      robot: The :class:`~mjlab.entity.Entity` to shove.
      dt: Control period, in seconds.
      max_episode_steps: The environment's ``max_episode_length``.
      cfg: Onset timing.
      generator: Seeded source for the headings, so a run is reproducible.
    """
    self.magnitude = magnitude
    self.robot = robot
    self.cfg = cfg
    self.generator = generator
    self.onsets = cfg.onsets(dt, max_episode_steps)
    if not self.onsets:
      raise ValueError(
        f"no shove fits in an episode of {max_episode_steps} steps with "
        f"settle={cfg.settle}s and tail={cfg.tail}s"
      )
    self._onset_steps = torch.tensor(
      self.onsets, device=magnitude.device, dtype=torch.long
    )
    self.delivered = 0
    """Shove events applied so far, summed over environments."""

  def apply(self, episode_step: torch.Tensor) -> torch.Tensor:
    """Shove every environment whose episode has reached an onset.

    Args:
      episode_step: Shape ``(N,)`` steps elapsed in each environment's episode,
        read before the step this call precedes.

    Returns:
      The environment indices shoved, for the caller to count against the
      episodes that took them.
    """
    due = (episode_step.unsqueeze(-1) == self._onset_steps).any(dim=-1)
    due &= self.magnitude > 0.0
    ids = due.nonzero(as_tuple=False).squeeze(-1)
    if len(ids) == 0:
      return ids
    heading = torch.rand(
      len(ids), device=self.magnitude.device, generator=self.generator
    ) * (2.0 * math.pi)
    delta = torch.zeros(len(ids), 6, device=self.magnitude.device)
    delta[:, 0] = self.magnitude[ids] * torch.cos(heading)
    delta[:, 1] = self.magnitude[ids] * torch.sin(heading)
    velocity = self.robot.data.root_link_vel_w[ids] + delta
    self.robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
    self.delivered += len(ids)
    return ids


def _quantiles(values: torch.Tensor) -> dict[str, float]:
  """Median and interquartile range of a 1-D tensor, NaNs dropped.

  Quartiles rather than a standard deviation: ``attain`` is bounded below by
  what a fallen robot does and ``wobble`` is a fraction, so both are skewed, and
  a mean plus a symmetric spread describes neither.
  """
  finite = values[values.isfinite()]
  if finite.numel() == 0:
    nan = float("nan")
    return {"n": 0, "median": nan, "q25": nan, "q75": nan, "iqr": nan}
  q = torch.quantile(finite, torch.tensor([0.25, 0.5, 0.75], device=finite.device))
  return {
    "n": int(finite.numel()),
    "median": float(q[1]),
    "q25": float(q[0]),
    "q75": float(q[2]),
    "iqr": float(q[2] - q[0]),
  }


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
  """Wilson score interval for a binomial rate.

  The normal approximation is useless at the ends of the grid, which is exactly
  where the fall rate lives: 0 falls in 40 episodes and 0 in 4 are different
  claims, and a +/- 0 interval says they are not.
  """
  if trials == 0:
    return (float("nan"), float("nan"))
  p = successes / trials
  denominator = 1.0 + z * z / trials
  centre = (p + z * z / (2 * trials)) / denominator
  spread = (
    z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
  )
  return (max(0.0, centre - spread), min(1.0, centre + spread))


CELL_METRICS: tuple[str, ...] = (
  "attain",
  "attain_x",
  "attain_y",
  "wobble_lead",
  "wobble",
  "ep_len_frac",
)
"""Continuous per-episode quantities summarised by quartiles. ``fell`` is
binary and gets a rate with a binomial interval instead.

``wobble_lead`` is quartiled over the episodes that fell, which is the only
place it is defined, so its ``n`` is the cell's fall count rather than its
episode count. The per-episode ``wobble`` fraction is still summarised -- it
costs nothing and the raw columns are in ``episodes.csv`` anyway -- but nothing
draws it."""


def summarise_cells(grid: CompetenceGrid, table: EpisodeTable) -> list[dict]:
  """Reduce the episode table to one record per cell.

  A cell below :data:`MIN_COMMAND_NORM` reports ``attain_defined: false`` and
  NaN attainment quartiles. That is not a gap in the data: no attainment sample
  was ever taken there, and reporting a zero would be inventing a measurement
  of sandbagging where the instrument was switched off. Such cells are read
  through ``wobble`` and ``fell`` alone.
  """
  cell_of = table.cell.long()
  records = []
  for index, cell in enumerate(grid.cells):
    rows = cell_of == index
    episodes = int(rows.sum())
    fell = table.fell[rows]
    falls = int(fell.sum()) if episodes else 0
    low, high = _wilson(falls, episodes)
    record: dict = {
      "cell": index,
      "vx": cell.vx,
      "vy": cell.vy,
      "wz": cell.wz,
      "shove": cell.shove,
      "command_norm": cell.command_norm,
      "attain_defined": cell.attain_defined,
      "episodes": episodes,
      "fell_rate": float(fell.mean()) if episodes else float("nan"),
      "fell_ci_low": low,
      "fell_ci_high": high,
      "shoves_taken": _quantiles(table.shoves_taken[rows]),
    }
    for name in CELL_METRICS:
      record[name] = _quantiles(getattr(table, name)[rows])
    records.append(record)
  return records


def format_grid_summary(grid: CompetenceGrid, records: list[dict]) -> str:
  """One-screen rendering of the grid: one line per cell."""
  header = (
    f"{'vx':>6} {'vy':>6} {'wz':>6} {'dv':>5} {'eps':>5} "
    f"{'attain':>18} {'wobble lead (s)':>18} {'fell':>17} {'ep_len':>18}"
  )
  lines = [
    f"cells {len(grid.cells)}, episodes {sum(r['episodes'] for r in records)}",
    header,
    "-" * len(header),
  ]

  def band(stat: dict) -> str:
    if not stat["n"]:
      return f"{'undefined':>18}"
    return f"{stat['median']:>7.3f} [{stat['q25']:.2f},{stat['q75']:.2f}]"

  for record in records:
    fell = (
      f"{record['fell_rate']:>6.2f} [{record['fell_ci_low']:.2f},"
      f"{record['fell_ci_high']:.2f}]"
    )
    lines.append(
      f"{record['vx']:>6.2f} {record['vy']:>6.2f} {record['wz']:>6.2f} "
      f"{record['shove']:>5.2f} {record['episodes']:>5d} "
      f"{band(record['attain'])} {band(record['wobble_lead'])} {fell} "
      f"{band(record['ep_len_frac'])}"
    )
  return "\n".join(lines)


def write_episodes_csv(path: Path, table: EpisodeTable) -> None:
  """Write one row per episode. This is the file the plotters read."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["episode"] + table.column_names())
    for index, row in enumerate(table.rows()):
      writer.writerow([index] + row)


def save_grid_run(
  output_dir: Path,
  run: dict,
  grid: CompetenceGrid,
  table: EpisodeTable,
) -> dict:
  """Write ``episodes.csv`` and ``cells.json``, and return the summary."""
  records = summarise_cells(grid, table)
  summary = {
    "run": run,
    "num_cells": len(grid.cells),
    "num_episodes": table.num_episodes,
    "cells": records,
  }
  write_episodes_csv(output_dir / "episodes.csv", table)
  path = output_dir / "cells.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=False)
    handle.write("\n")
  return summary
