"""Edge-of-competence curriculum: tracker, controller, and adaptive terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

_NUM_STEPS_PER_ENV = 24
_MIN_CMD_NORM = 0.2

# Command velocity ranges by competence level (doc 13).
COMMAND_LEVEL_TABLE: list[dict[str, tuple[float, float]]] = [
  {"lin_vel_x": (-0.20, 0.20), "lin_vel_y": (-0.05, 0.05), "ang_vel_z": (-0.10, 0.10)},
  {"lin_vel_x": (-0.30, 0.30), "lin_vel_y": (-0.10, 0.10), "ang_vel_z": (-0.25, 0.25)},
  {"lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.20, 0.20), "ang_vel_z": (-0.40, 0.40)},
  {"lin_vel_x": (-0.50, 0.50), "lin_vel_y": (-0.30, 0.30), "ang_vel_z": (-0.50, 0.50)},
  {"lin_vel_x": (-0.60, 0.60), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-0.65, 0.65)},
  {"lin_vel_x": (-0.75, 0.75), "lin_vel_y": (-0.45, 0.45), "ang_vel_z": (-0.80, 0.80)},
]

PUSH_LEVEL_SCALES: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

_PUSH_VELOCITY_RANGE_BASE: dict[str, tuple[float, float]] = {
  "x": (-0.2, 0.4),
  "y": (-0.2, 0.2),
  "z": (-0.0, 0.0),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": (-0.0, 0.0),
}


class CompetenceControllerState(TypedDict):
  level: int
  last_change_iter: int
  promote_streak: int
  frozen: bool


@dataclass
class CompetenceThresholds:
  """Gating thresholds.

  Falls are a TRAILING indicator: over-challenged policies first retreat
  into conservative stability (stand/shuffle, tracking sacrificed) and only
  fall at terminal collapse (v20 pair 1: full ended ep_len 204 after a
  whole mid-run of low fall rates). Attainment (commanded-direction speed
  achieved / commanded speed) is the LEADING indicator — sandbagging reads
  ~0 while never falling — and wobble fraction (steps with tilt > ~25°) is
  the graded near-fall precursor. Falls remain as the safety net only.
  """

  promote_track_err: float = 0.25  # legacy, no longer in the predicate
  demote_track_err: float = 0.45  # legacy, no longer in the predicate
  promote_attain: float = 0.75
  demote_attain: float = 0.5
  promote_wobble: float = 0.10
  demote_wobble: float = 0.25
  promote_fell: float = 0.3
  demote_fell: float = 0.35
  cooldown_iters: int = 150
  promote_streak_required: int = 3
  stability_ep_len_frac: float = 0.8
  stability_fell: float = 0.5
  stability_attain: float = 0.6
  # Fast demote channel (v23 postmortem): the per-env EMAs update only at
  # episode ends, so at healthy episode lengths (~900 steps) a crash at the
  # top of the ladder is invisible for ~200 iterations — long enough for
  # the -10 fall terminations to dominate the gradient and shatter the
  # policy (v23: L5 -> fell 10.7 -> demoted too late -> attain -0.68,
  # unrecoverable). The windowed population fall rate crosses this bar
  # within ~5 iterations of a real crash. Healthy L5 operation measured
  # ~0.26 falls/episode-end (v23 iter 1815), so 0.5 clears the working
  # band while catching the spiral at its front edge.
  demote_fast_fell: float = 0.5
  # Extra promote caution on the top rungs (L4 -> L5 was v23's cliff).
  top_streak_required: int = 5
  top_level_start: int = 4


class CompetenceController:
  """Hysteresis + cooldown level controller driven by population competence."""

  def __init__(
    self,
    *,
    l_max: int,
    thresholds: CompetenceThresholds,
    gate_on_track_err: bool = True,
  ):
    self.l_max = l_max
    self.thresholds = thresholds
    self.gate_on_track_err = gate_on_track_err
    self.level = 0
    self.last_change_iter = -thresholds.cooldown_iters
    self.promote_streak = 0
    self.frozen = False

  def state_dict(self) -> CompetenceControllerState:
    return {
      "level": self.level,
      "last_change_iter": self.last_change_iter,
      "promote_streak": self.promote_streak,
      "frozen": self.frozen,
    }

  def load_state_dict(self, state: CompetenceControllerState) -> None:
    self.level = int(state["level"])
    self.last_change_iter = int(state["last_change_iter"])
    self.promote_streak = int(state["promote_streak"])
    self.frozen = bool(state["frozen"])

  def _current_iter(self, common_step_counter: int) -> int:
    return common_step_counter // _NUM_STEPS_PER_ENV

  def _cooldown_elapsed(self, common_step_counter: int) -> bool:
    cur = self._current_iter(common_step_counter)
    return cur - self.last_change_iter >= self.thresholds.cooldown_iters

  def update(
    self,
    *,
    track_err_norm: float,
    fell_ema: float,
    ep_len_frac: float,
    common_step_counter: int,
    attain_ema: float = 0.0,
    wobble_ema: float = 1.0,
    fast_fall_rate: float = 0.0,
  ) -> str | None:
    """Return ``promote``, ``demote``, or None.

    Unified predicate for all axes (the old fell-only push gating let the
    push axis outrun everything): promote needs the task actually being
    performed (attainment) with low near-fall wobble and low falls; demote
    fires on sandbagging (attainment collapse), sustained wobble, or falls.
    ``track_err_norm`` is retained for logging/compat but no longer gates.
    """
    del track_err_norm
    t = self.thresholds
    promote_ok = (
      attain_ema > t.promote_attain
      and wobble_ema < t.promote_wobble
      and fell_ema < t.promote_fell
    )
    demote_bad = (
      attain_ema < t.demote_attain
      or wobble_ema > t.demote_wobble
      or fell_ema > t.demote_fell
      or fast_fall_rate > t.demote_fast_fell
    )

    if demote_bad and self.level > 0:
      self.level -= 1
      self.last_change_iter = self._current_iter(common_step_counter)
      self.promote_streak = 0
      return "demote"

    if not promote_ok:
      self.promote_streak = 0
      return None

    self.promote_streak += 1
    streak_required = (
      t.top_streak_required
      if self.level + 1 >= t.top_level_start
      else t.promote_streak_required
    )
    if (
      self.promote_streak >= streak_required
      and self.level < self.l_max
      and self._cooldown_elapsed(common_step_counter)
    ):
      self.level += 1
      self.last_change_iter = self._current_iter(common_step_counter)
      self.promote_streak = 0
      return "promote"
    return None

  def stability_ok(
    self, *, fell_ema: float, ep_len_frac: float, attain_ema: float = 1.0
  ) -> bool:
    return (
      ep_len_frac > self.thresholds.stability_ep_len_frac
      and fell_ema < self.thresholds.stability_fell
      and attain_ema > self.thresholds.stability_attain
    )


class CompetenceTracker:
  """Per-env episode stats at reset, EMA'd for population competence signals."""

  def __init__(self, env: ManagerBasedRlEnv, *, ema_alpha: float = 0.1):
    self.device = env.device
    self.num_envs = env.num_envs
    self.ema_alpha = ema_alpha
    n = env.num_envs
    self._track_sum = torch.zeros(n, device=self.device)
    self._track_weight = torch.zeros(n, device=self.device)
    self._attain_sum = torch.zeros(n, device=self.device)
    self._attain_weight = torch.zeros(n, device=self.device)
    self._attain_x_sum = torch.zeros(n, device=self.device)
    self._attain_x_weight = torch.zeros(n, device=self.device)
    self._attain_y_sum = torch.zeros(n, device=self.device)
    self._attain_y_weight = torch.zeros(n, device=self.device)
    self._wobble_sum = torch.zeros(n, device=self.device)
    self._step_count = torch.zeros(n, device=self.device)
    # Pessimistic init: a fresh policy must EARN competence. Zero-init would
    # read as "perfectly competent" and allow spurious promotions in the
    # first iterations before the EMAs warm up.
    self.track_err_ema = torch.ones(n, device=self.device)
    self.fell_ema = torch.ones(n, device=self.device)
    self.ep_len_frac_ema = torch.zeros(n, device=self.device)
    self.attain_ema = torch.zeros(n, device=self.device)
    self.attain_x_ema = torch.zeros(n, device=self.device)
    self.attain_y_ema = torch.zeros(n, device=self.device)
    self.wobble_ema = torch.ones(n, device=self.device)
    self._finalized_step = -1
    # Fast population fall-rate channel: falls / episode-ends accumulated
    # since the last refresh, smoothed with a per-iteration EMA. Pessimistic
    # init like the slow EMAs; refreshed at most once per training iteration
    # and only once enough episodes ended for the ratio to be meaningful.
    self._win_fell = 0.0
    self._win_done = 0.0
    self._fast_next_step = 0
    self.fast_fall_rate = 1.0
    self.fast_alpha = 0.2
    # Push-cohort stratification (user design 2026-07-05): a fixed slice of
    # env indices receives pushes; the rest train push-free. Attribution is
    # by membership - no recovery-horizon guess - and the clean cohort
    # matches the mostly-push-free deployment distribution. The policy
    # cannot observe membership. Fraction set via the pushed-cohort event
    # wrapper / diagnostics term; 1.0 (all pushed) is the legacy behavior.
    self.push_cohort_frac = 1.0
    self.push_cohort = torch.ones(n, dtype=torch.bool, device=self.device)
    self.last_push_step = torch.full((n,), -(10**9), device=self.device)
    self._win_fell_clean = 0.0
    self._win_done_clean = 0.0
    self._win_fell_pushed = 0.0
    self._win_done_pushed = 0.0
    self.fast_fall_clean = 1.0
    self.fast_fall_pushed = 1.0
    # Time-from-push-to-fall histogram (seconds bins: see edges) - answers
    # "how long does recovery take" empirically.
    # 0.5 s bins over [0, 16 s); quantiles (t50/t75) are the outputs.
    self.n_dt_bins = 32
    self.dt_bin_width = 0.5
    self.push_fall_dt_counts = torch.zeros(self.n_dt_bins, device=self.device)
    # Frontier estimator (clean cohort only): exposure steps and falls
    # densely binned (R18: bins are the sufficient statistic; the
    # curriculum consumes interpolated level-crossings/quantiles, which
    # integrate across bins and are robust to fine binning).
    self.n_cmd_buckets = 32
    self._bucket_steps = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._bucket_falls = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.bucket_hazard = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._cur_bucket = torch.full((n,), -1, dtype=torch.long, device=self.device)
    self._bucket_next_step = 0
    # Mahalanobis-radius buckets (doc 15 R11): rho normalizes (vx, vy, wz)
    # by the current per-axis maxima, so hazards are measured in the
    # geometry that respects axis coupling — under box sampling the high
    # bins ARE the corners (rho spans up to ~1.7), under ellipsoid
    # sampling rho <= 1 by construction. 8 bins of width 0.2 over [0, 1.6).
    self._rho_steps = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._rho_falls = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.rho_hazard = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._cur_rho_bucket = torch.full((n,), -1, dtype=torch.long, device=self.device)
    try:
      self._step_dt = float(env.step_dt)
    except (TypeError, ValueError):
      self._step_dt = 0.02
    # Swing peak height population EMA, ingested from the per-step
    # Metrics/peak_height_mean the feet_swing_height reward publishes
    # (noisy per step - few landings per step - so a slow EMA), plus a
    # slowly-decaying trailing max used as the style-regression reference
    # by the Lagrangian penalty gates (doc 15 R10).
    self.peak_height_ema = 0.0
    self.peak_height_trailing_max = 0.0

  def set_push_cohort(self, frac: float) -> None:
    self.push_cohort_frac = frac
    n_pushed = int(round(frac * self.num_envs))
    self.push_cohort = torch.arange(self.num_envs, device=self.device) < n_pushed

  def record_push(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    self.last_push_step[env_ids] = float(env.common_step_counter)

  def record_peak_height(self, env: ManagerBasedRlEnv) -> None:
    log = env.extras.get("log", {}) if isinstance(env.extras, dict) else {}
    val = log.get("Metrics/peak_height_mean")
    if val is None:
      return
    v = float(val)
    if v <= 0.0:
      return
    self.peak_height_ema += 0.01 * (v - self.peak_height_ema)
    # Sticky short-term, adaptive long-term: decays with ~1400-iter
    # half-life so a config-level shift in healthy swing height does not
    # permanently pin the reference.
    self.peak_height_trailing_max = max(
      self.peak_height_trailing_max * 0.99998, self.peak_height_ema
    )

  def state_dict(self) -> dict[str, Any]:
    return {
      "track_err_ema": self.track_err_ema.cpu(),
      "fell_ema": self.fell_ema.cpu(),
      "ep_len_frac_ema": self.ep_len_frac_ema.cpu(),
      "attain_ema": self.attain_ema.cpu(),
      "wobble_ema": self.wobble_ema.cpu(),
      "fast_fall_rate": self.fast_fall_rate,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.track_err_ema = state["track_err_ema"].to(self.device)
    self.fell_ema = state["fell_ema"].to(self.device)
    self.ep_len_frac_ema = state["ep_len_frac_ema"].to(self.device)
    if "attain_ema" in state:
      self.attain_ema = state["attain_ema"].to(self.device)
      self.wobble_ema = state["wobble_ema"].to(self.device)
    if "fast_fall_rate" in state:
      self.fast_fall_rate = float(state["fast_fall_rate"])

  def record_step(self, env: ManagerBasedRlEnv) -> None:
    term = env.command_manager.get_term("twist")
    if term is None:
      return
    command_term = cast("UniformVelocityCommand", term)
    # Wobble (near-fall precursor): fraction of steps with tilt > ~25deg,
    # counted for ALL envs (a tilting stander is also incompetent).
    grav_xy = command_term.robot.data.projected_gravity_b[:, :2]
    tilted = (torch.norm(grav_xy, dim=-1) > 0.4226).float()  # sin(25 deg)
    self._wobble_sum += tilted
    self._step_count += 1.0

    standing = command_term.is_standing_env
    walking = ~standing
    if not walking.any():
      return
    cmd_xy = command_term.vel_command_b[:, :2]
    vel_xy = command_term.robot.data.root_link_lin_vel_b[:, :2]
    err_norm = torch.norm(cmd_xy - vel_xy, dim=-1)
    cmd_norm = torch.clamp(torch.norm(cmd_xy, dim=-1), min=_MIN_CMD_NORM)
    normalized = err_norm / cmd_norm
    self._track_sum[walking] += normalized[walking]
    self._track_weight[walking] += 1.0
    # Attainment: achieved velocity projected onto the commanded direction,
    # as a TRUE fraction of commanded speed, measured only on steps with a
    # meaningful command (|c| >= 0.15). Sandbagging reads ~0 without a
    # single fall; lateral sway is orthogonal and drops out. The earlier
    # floor-SQUARED denominator capped perfect tracking of a 0.1 m/s
    # command at 0.25 and froze level-0 promotion (v20 pair 3: attain
    # 0.17-0.25 while every other gate passed); the meaningful-command
    # filter replaces the floor's anti-gaming role.
    cmd_sq = (cmd_xy * cmd_xy).sum(dim=-1)
    meaningful = walking & (cmd_sq >= 0.15 * 0.15)
    attain = (vel_xy * cmd_xy).sum(dim=-1) / cmd_sq.clamp(min=1e-6)
    self._attain_sum[meaningful] += attain[meaningful]
    self._attain_weight[meaningful] += 1.0

    # Per-axis attainment (doc 15 R12): the level table's axis RATIOS are
    # vibes-based; these measure achieved/commanded per direction, each
    # sample weighted by that axis's share of command energy, so "when
    # commands ask for x, do we deliver x" is answered independently of y.
    for axis, sums, weights in (
      (0, self._attain_x_sum, self._attain_x_weight),
      (1, self._attain_y_sum, self._attain_y_weight),
    ):
      c = cmd_xy[:, axis]
      w = (c * c) / cmd_sq.clamp(min=1e-6)
      m = meaningful & (c.abs() >= 0.10)
      # The mask guarantees |c| >= 0.10 so the division is well-defined;
      # achieved/commanded is signed, so backpedaling reads negative.
      ax = vel_xy[:, axis][m] / c[m]
      sums[m] += ax * w[m]
      weights[m] += w[m]

    # Frontier estimator exposure (clean cohort only): bucket walking steps
    # by commanded speed so falls can be attributed to the speed at which
    # they happened (binned Bernoulli estimate of P(fall | speed)).
    bucket = (
      (torch.sqrt(cmd_sq) / (0.8 / self.n_cmd_buckets))
      .long()
      .clamp(0, self.n_cmd_buckets - 1)
    )
    self._cur_bucket = torch.where(walking, bucket, torch.full_like(bucket, -1))
    expose = walking & ~self.push_cohort
    if expose.any():
      self._bucket_steps.scatter_add_(
        0, bucket[expose], torch.ones(int(expose.sum()), device=self.device)
      )

    # Mahalanobis-radius exposure: normalize by the CURRENT per-axis
    # maxima from the live command cfg (they move with the curriculum).
    ranges = command_term.cfg.ranges
    rx = max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]), 1e-6)
    ry = max(abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]), 1e-6)
    rw = max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]), 1e-6)
    wz = command_term.vel_command_b[:, 2]
    rho = torch.sqrt(
      (cmd_xy[:, 0] / rx) ** 2 + (cmd_xy[:, 1] / ry) ** 2 + (wz / rw) ** 2
    )
    rho_bucket = (
      (rho / (1.6 / self.n_cmd_buckets)).long().clamp(0, self.n_cmd_buckets - 1)
    )
    self._cur_rho_bucket = torch.where(
      walking, rho_bucket, torch.full_like(rho_bucket, -1)
    )
    if expose.any():
      self._rho_steps.scatter_add_(
        0, rho_bucket[expose], torch.ones(int(expose.sum()), device=self.device)
      )

  def finalize_episodes(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> None:
    if env.common_step_counter == self._finalized_step:
      return
    self._finalized_step = env.common_step_counter
    if isinstance(env_ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids.to(self.device)

    ep_len = env.episode_length_buf[ids].float()
    max_len = float(env.max_episode_length)
    ep_frac = ep_len / max_len

    fell = torch.zeros(len(ids), device=self.device)
    if "fell_over" in env.termination_manager.active_terms:
      fell = env.termination_manager.get_term("fell_over")[ids].float()

    track_err = torch.zeros(len(ids), device=self.device)
    has_weight = self._track_weight[ids] > 0
    track_err[has_weight] = (
      self._track_sum[ids][has_weight] / self._track_weight[ids][has_weight]
    )

    alpha = self.ema_alpha
    # Envs with no tracking weight this episode (e.g. standing envs) keep
    # their previous EMA. All operands are full [len(ids)]-shaped so the
    # where() is well-formed regardless of the has_weight mix.
    updated_track = alpha * track_err + (1.0 - alpha) * self.track_err_ema[ids]
    self.track_err_ema[ids] = torch.where(
      has_weight, updated_track, self.track_err_ema[ids]
    )

    attain = torch.zeros(len(ids), device=self.device)
    has_attain = self._attain_weight[ids] > 0
    attain[has_attain] = (
      self._attain_sum[ids][has_attain] / self._attain_weight[ids][has_attain]
    )
    updated_attain = alpha * attain + (1.0 - alpha) * self.attain_ema[ids]
    self.attain_ema[ids] = torch.where(has_attain, updated_attain, self.attain_ema[ids])

    for sums, weights, ema in (
      (self._attain_x_sum, self._attain_x_weight, self.attain_x_ema),
      (self._attain_y_sum, self._attain_y_weight, self.attain_y_ema),
    ):
      val = torch.zeros(len(ids), device=self.device)
      has = weights[ids] > 0
      val[has] = sums[ids][has] / weights[ids][has]
      upd = alpha * val + (1.0 - alpha) * ema[ids]
      ema[ids] = torch.where(has, upd, ema[ids])

    steps = self._step_count[ids].clamp(min=1.0)
    wobble = self._wobble_sum[ids] / steps
    has_steps = self._step_count[ids] > 0
    updated_wobble = alpha * wobble + (1.0 - alpha) * self.wobble_ema[ids]
    self.wobble_ema[ids] = torch.where(has_steps, updated_wobble, self.wobble_ema[ids])
    self.fell_ema[ids] = alpha * fell + (1.0 - alpha) * self.fell_ema[ids]
    self.ep_len_frac_ema[ids] = (
      alpha * ep_frac + (1.0 - alpha) * self.ep_len_frac_ema[ids]
    )

    self._track_sum[ids] = 0.0
    self._track_weight[ids] = 0.0
    self._attain_sum[ids] = 0.0
    self._attain_weight[ids] = 0.0
    self._attain_x_sum[ids] = 0.0
    self._attain_x_weight[ids] = 0.0
    self._attain_y_sum[ids] = 0.0
    self._attain_y_weight[ids] = 0.0
    self._wobble_sum[ids] = 0.0
    self._step_count[ids] = 0.0

    self._win_fell += float(fell.sum().item())
    self._win_done += float(len(ids))

    # Cohort-stratified windows: attribution by membership, not by any
    # recovery-horizon guess (user design 2026-07-05).
    pushed = self.push_cohort[ids]
    self._win_fell_pushed += float(fell[pushed].sum().item())
    self._win_done_pushed += float(int(pushed.sum()))
    clean = ~pushed
    self._win_fell_clean += float(fell[clean].sum().item())
    self._win_done_clean += float(int(clean.sum()))

    # Time-from-push-to-fall histogram (pushed cohort): answers "how long
    # does recovery take" empirically instead of assuming a horizon.
    fell_pushed = pushed & (fell > 0)
    if fell_pushed.any():
      dt_s = (
        env.common_step_counter - self.last_push_step[ids][fell_pushed]
      ) * self._step_dt
      bins = (dt_s / self.dt_bin_width).long().clamp(0, self.n_dt_bins - 1)
      self.push_fall_dt_counts.scatter_add_(
        0, bins, torch.ones(len(dt_s), device=self.device)
      )

    # Frontier falls (clean cohort): charge the fall to the speed bucket
    # the env was walking in when it fell.
    fell_clean_bucketed = clean & (fell > 0) & (self._cur_bucket[ids] >= 0)
    if fell_clean_bucketed.any():
      fb = self._cur_bucket[ids][fell_clean_bucketed]
      self._bucket_falls.scatter_add_(0, fb, torch.ones(len(fb), device=self.device))
    fell_rho = clean & (fell > 0) & (self._cur_rho_bucket[ids] >= 0)
    if fell_rho.any():
      rb = self._cur_rho_bucket[ids][fell_rho]
      self._rho_falls.scatter_add_(0, rb, torch.ones(len(rb), device=self.device))

    if env.common_step_counter >= self._fast_next_step:
      # Require ~0.2% of envs to have finished an episode before trusting
      # the ratio; below that, keep accumulating into the next window.
      if self._win_done >= max(4.0, 0.002 * self.num_envs):
        rate = self._win_fell / self._win_done
        a = self.fast_alpha
        self.fast_fall_rate = a * rate + (1.0 - a) * self.fast_fall_rate
        self._win_fell = 0.0
        self._win_done = 0.0
        self._fast_next_step = env.common_step_counter + _NUM_STEPS_PER_ENV
        if self._win_done_clean >= 4.0:
          r = self._win_fell_clean / self._win_done_clean
          self.fast_fall_clean = a * r + (1.0 - a) * self.fast_fall_clean
          self._win_fell_clean = 0.0
          self._win_done_clean = 0.0
        if self._win_done_pushed >= 4.0:
          r = self._win_fell_pushed / self._win_done_pushed
          self.fast_fall_pushed = a * r + (1.0 - a) * self.fast_fall_pushed
          self._win_fell_pushed = 0.0
          self._win_done_pushed = 0.0

    # Bucket hazards refresh on a slower cadence (falls per bucket are
    # sparse): ~50 iterations accumulates thousands of exposure steps per
    # bucket at production env counts.
    if env.common_step_counter >= self._bucket_next_step:
      if self._bucket_steps.sum() >= 100.0:
        hazard = self._bucket_falls / self._bucket_steps.clamp(min=1.0)
        self.bucket_hazard = 0.3 * hazard + 0.7 * self.bucket_hazard
        self._bucket_steps.zero_()
        self._bucket_falls.zero_()
        rho_hazard = self._rho_falls / self._rho_steps.clamp(min=1.0)
        self.rho_hazard = 0.3 * rho_hazard + 0.7 * self.rho_hazard
        self._rho_steps.zero_()
        self._rho_falls.zero_()
        self._bucket_next_step = env.common_step_counter + 50 * _NUM_STEPS_PER_ENV

  def population_means(self) -> dict[str, float]:
    return {
      "track_err_norm": self.track_err_ema.mean().item(),
      "fell_ema": self.fell_ema.mean().item(),
      "ep_len_frac": self.ep_len_frac_ema.mean().item(),
      "attain": self.attain_ema.mean().item(),
      "wobble": self.wobble_ema.mean().item(),
      "fast_fall_rate": self.fast_fall_rate,
    }

  def stratified_means(self) -> dict[str, float]:
    """Per-cohort aggregation of the per-env EMAs plus the fast splits.

    With cohort membership fixed by env index, stratification is pure
    aggregation - the accumulation machinery is untouched. Falls back to
    global values when a cohort is empty (legacy all-pushed configs).
    """
    clean = ~self.push_cohort
    out: dict[str, float] = {}
    for name, ema in (
      ("attain", self.attain_ema),
      ("attain_x", self.attain_x_ema),
      ("attain_y", self.attain_y_ema),
      ("wobble", self.wobble_ema),
      ("fell_ema", self.fell_ema),
      ("track_err_norm", self.track_err_ema),
    ):
      out[f"clean_{name}"] = (
        ema[clean].mean().item() if clean.any() else ema.mean().item()
      )
      out[f"pushed_{name}"] = (
        ema[self.push_cohort].mean().item()
        if self.push_cohort.any()
        else ema.mean().item()
      )
    out["fast_fall_clean"] = self.fast_fall_clean
    out["fast_fall_pushed"] = self.fast_fall_pushed
    # Excess fall rate attributable to pushes (difference-in-rates; both
    # cohorts share the command sampler, so command difficulty cancels).
    out["push_excess_fall"] = self.fast_fall_pushed - self.fast_fall_clean
    return out


def get_competence_tracker(env: ManagerBasedRlEnv) -> CompetenceTracker:
  tracker = getattr(env, "_competence_tracker", None)
  if tracker is None:
    tracker = CompetenceTracker(env)
    env._competence_tracker = tracker
  return tracker


def competence_tracker_step(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  del env_ids
  tracker = getattr(env, "_competence_tracker", None)
  if tracker is not None:
    tracker.record_step(env)
    tracker.record_peak_height(env)


def push_cohort_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  cohort_frac: float = 1.0,
) -> None:
  """Cohort-filtered push: only env indices < cohort_frac * num_envs are
  pushed; the rest train push-free (deployment-matched) and serve as the
  uncontaminated baseline for tracking competence. Also stamps push times
  on the tracker for the recovery-time histogram."""
  from mjlab.envs.mdp.events import push_by_setting_velocity

  tracker = get_competence_tracker(env)
  if tracker.push_cohort_frac != cohort_frac:
    tracker.set_push_cohort(cohort_frac)
  mask = tracker.push_cohort[env_ids]
  pushed_ids = env_ids[mask]
  if len(pushed_ids) == 0:
    return
  tracker.record_push(env, pushed_ids)
  push_by_setting_velocity(env, pushed_ids, velocity_range)


class joule_lambda_shadow:
  """Log-only pilot of the inverted-Lagrangian energy multiplier (R10).

  Design (user, 2026-07-05): minimize energy SUBJECT TO competence - the
  multiplier rises only while every style/competence gate holds, and
  retreats multiplicatively the moment the squeeze starts eating them.
  The two historical failure modes of the joule penalty are the two
  gates, so the live version cannot reproduce them silently:
    - foot-lift collapse: peak_height EMA >= peak_floor_frac of its
      trailing max (v16c dragged feet; v25-slow slid 0.0128 -> 0.0072);
    - sandbagging: clean-cohort attainment >= attain_floor (attainment
      was built to make command-avoidance visible; the cohort split
      decontaminated it);
    - plus the shared governor: clean fast fall rate < fall_bar.
  eta is sized for timescale separation (full ramp over ~1000 iters -
  the disease-#2 lesson: the policy must track the objective
  quasi-statically). This SHADOW version computes and logs the lambda
  trajectory alongside the staged weight actually in charge; it changes
  nothing. Validation: lambda climbs while style is healthy, freezes on
  a peak-height dip, retreats before a T4-style decay would fire.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._tracker = get_competence_tracker(env)
    self._reward_name: str = cfg.params.get("reward_name", "joule_heating")
    self._lambda_cap: float = cfg.params.get("lambda_cap", 2e-5)
    self._eta: float = self._lambda_cap / cfg.params.get("ramp_iters", 1000)
    self._retreat: float = cfg.params.get("retreat", 0.8)
    self._peak_floor_frac: float = cfg.params.get("peak_floor_frac", 0.85)
    self._peak_retreat_frac: float = cfg.params.get("peak_retreat_frac", 0.70)
    self._attain_floor: float = cfg.params.get("attain_floor", 0.50)
    self._fall_bar: float = cfg.params.get("fall_bar", 0.20)
    # Live mode (R10 phase 2): the multiplier REPLACES the staged joule
    # weight (reward weight = -lambda). Flip only after a full shadow run
    # validates the trajectory; the staged curriculum for this reward
    # must be disabled when live or the two would fight.
    self._apply_live: bool = bool(cfg.params.get("apply_live", False))
    self._lam = 0.0
    self._last_iter = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str = "joule_heating",
    lambda_cap: float = 2e-5,
    ramp_iters: int = 1000,
    retreat: float = 0.8,
    peak_floor_frac: float = 0.85,
    peak_retreat_frac: float = 0.70,
    attain_floor: float = 0.50,
    fall_bar: float = 0.20,
    apply_live: bool = False,
  ) -> dict[str, torch.Tensor]:
    del (reward_name, lambda_cap, ramp_iters, retreat, peak_floor_frac)
    del (peak_retreat_frac, attain_floor, fall_bar, apply_live)
    self._tracker.finalize_episodes(env, env_ids)
    t = self._tracker
    strat = t.stratified_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    peak_ref = max(t.peak_height_trailing_max, 1e-6)
    peak_frac = t.peak_height_ema / peak_ref
    style_broken = peak_frac < self._peak_retreat_frac
    gates_ok = (
      peak_frac >= self._peak_floor_frac
      and strat["clean_attain"] >= self._attain_floor
      and strat["fast_fall_clean"] < self._fall_bar
    )
    if cur_iter != self._last_iter:
      self._last_iter = cur_iter
      if style_broken:
        self._lam *= self._retreat
      elif gates_ok:
        self._lam = min(self._lam + self._eta, self._lambda_cap)
      # else: hold.
    if self._apply_live:
      try:
        env.reward_manager.get_term_cfg(self._reward_name).weight = -self._lam
      except (KeyError, AttributeError):
        pass
    try:
      live_weight = float(env.reward_manager.get_term_cfg(self._reward_name).weight)
    except (KeyError, AttributeError, TypeError, ValueError):
      live_weight = 0.0
    return {
      "lambda": torch.tensor(self._lam),
      "live_weight": torch.tensor(live_weight),
      "peak_frac": torch.tensor(peak_frac),
      "peak_ema": torch.tensor(t.peak_height_ema),
      "gates_ok": torch.tensor(1.0 if gates_ok else 0.0),
      "style_broken": torch.tensor(1.0 if style_broken else 0.0),
    }


def _interp_crossing(hazards: torch.Tensor, bin_width: float, bar: float) -> float:
  """First up-crossing of the hazard bar, linearly interpolated (R18).

  Light 3-bin smoothing first: quantiles/crossings integrate noise, but
  a single hot fine bin should not snap the frontier. Returns the
  bin-center-based crossing position; the full range if never crossed.
  """
  n = len(hazards)
  sm = hazards.clone()
  if n >= 3:
    sm[1:-1] = (hazards[:-2] + hazards[1:-1] + hazards[2:]) / 3.0
  prev_c = 0.5 * bin_width
  prev_h = float(sm[0])
  if prev_h > bar:
    return prev_c
  for i in range(1, n):
    c = (i + 0.5) * bin_width
    h = float(sm[i])
    if h > bar:
      frac = (bar - prev_h) / max(h - prev_h, 1e-12)
      return prev_c + frac * (c - prev_c)
    prev_c, prev_h = c, h
  return n * bin_width


def _binned_quantile(counts: torch.Tensor, bin_width: float, q: float) -> float:
  """Interpolated quantile of a binned distribution (R18)."""
  total = float(counts.sum())
  if total <= 0:
    return 0.0
  target = q * total
  cum = 0.0
  for i in range(len(counts)):
    c = float(counts[i])
    if cum + c >= target:
      frac = (target - cum) / max(c, 1e-12)
      return (i + frac) * bin_width
    cum += c
  return len(counts) * bin_width


class competence_diagnostics:
  """Log-only curriculum term: cohort-stratified competence + frontier.

  Publishes the decoupled signals (clean vs pushed attain/wobble/falls,
  difference-in-rates push effect, per-speed-bucket fall hazards and the
  estimated frontier speed, push-to-fall timing histogram) so they can be
  validated against the coupled controllers before any of them drive
  difficulty. Phase 1 of the frontier-estimator plan (doc 15 R9).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._tracker = get_competence_tracker(env)
    self._last_hist_iter = -1
    cohort_frac = cfg.params.get("cohort_frac", 1.0)
    if self._tracker.push_cohort_frac != cohort_frac:
      self._tracker.set_push_cohort(cohort_frac)
    self._hazard_bar: float = cfg.params.get("frontier_hazard_bar", 5e-4)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    cohort_frac: float = 1.0,
    frontier_hazard_bar: float = 5e-4,
  ) -> dict[str, torch.Tensor]:
    del cohort_frac, frontier_hazard_bar
    self._tracker.finalize_episodes(env, env_ids)
    out = {k: torch.tensor(v) for k, v in self._tracker.stratified_means().items()}
    t = self._tracker
    speed_bw = 0.8 / t.n_cmd_buckets
    rho_bw = 1.6 / t.n_cmd_buckets
    # R18: the statistics ARE the metrics; dense bins feed the
    # interpolated crossings/quantiles and the histogram views only.
    out["frontier_speed"] = torch.tensor(
      _interp_crossing(t.bucket_hazard, speed_bw, self._hazard_bar)
    )
    out["frontier_rho"] = torch.tensor(
      _interp_crossing(t.rho_hazard, rho_bw, self._hazard_bar)
    )
    out["push_fall_t50"] = torch.tensor(
      _binned_quantile(t.push_fall_dt_counts, t.dt_bin_width, 0.50)
    )
    out["push_fall_t75"] = torch.tensor(
      _binned_quantile(t.push_fall_dt_counts, t.dt_bin_width, 0.75)
    )
    self._log_histograms(env)
    return out

  def _log_histograms(self, env: ManagerBasedRlEnv) -> None:
    """W&B heatmap views of the bucketed profiles (user request): the
    per-bucket scalars stay (monitor triggers and API pulls read them);
    these are the human-readable evolution-over-time renderings. Logged
    directly with commit=False so they attach to the logger's next
    committed step; inherently rank-0-only (only that process owns a
    W&B run). Cadence matches the hazard refresh (~50 iters).
    """
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter - self._last_hist_iter < 50:
      return
    self._last_hist_iter = cur_iter
    try:
      import numpy as np
      import wandb

      if wandb.run is None:
        return
      t = self._tracker
      n = t.n_cmd_buckets
      wandb.log(
        {
          "frontier/hazard_by_speed": wandb.Histogram(
            np_histogram=(
              t.bucket_hazard.cpu().numpy(),
              np.linspace(0.0, 0.8, n + 1),
            )
          ),
          "frontier/hazard_by_rho": wandb.Histogram(
            np_histogram=(
              t.rho_hazard.cpu().numpy(),
              np.linspace(0.0, 1.6, n + 1),
            )
          ),
          "frontier/push_fall_dt": wandb.Histogram(
            np_histogram=(
              t.push_fall_dt_counts.cpu().numpy(),
              np.linspace(0.0, t.n_dt_bins * t.dt_bin_width, t.n_dt_bins + 1),
            )
          ),
        },
        commit=False,
      )
    except Exception:
      # Visualization must never take down training.
      return


def _scale_push_velocity_range(scale: float) -> dict[str, tuple[float, float]]:
  return {
    axis: (lo * scale, hi * scale)
    for axis, (lo, hi) in _PUSH_VELOCITY_RANGE_BASE.items()
  }


def _apply_command_level(
  env: ManagerBasedRlEnv, command_name: str, level: int
) -> dict[str, torch.Tensor]:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = command_term.cfg
  assert isinstance(cfg, UniformVelocityCommandCfg)
  table = COMMAND_LEVEL_TABLE[level]
  cfg.ranges.lin_vel_x = table["lin_vel_x"]
  cfg.ranges.lin_vel_y = table["lin_vel_y"]
  cfg.ranges.ang_vel_z = table["ang_vel_z"]
  return {
    "level": torch.tensor(level),
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def _apply_push_level(
  env: ManagerBasedRlEnv, event_name: str, level: int
) -> dict[str, torch.Tensor]:
  term_cfg = env.event_manager.get_term_cfg(event_name)
  scale = PUSH_LEVEL_SCALES[level]
  term_cfg.params["velocity_range"] = _scale_push_velocity_range(scale)
  velocity_range = term_cfg.params["velocity_range"]
  return {
    "level": torch.tensor(level),
    **{
      f"push_{axis}_{bound}": torch.tensor(velocity_range[axis][i])
      for axis in velocity_range
      for i, bound in enumerate(("min", "max"))
    },
  }


class adaptive_command_level:
  """Widen velocity commands when population tracking competence is high."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    l_max: int = cfg.params.get("l_max", 3)
    thresholds = CompetenceThresholds(
      promote_track_err=cfg.params.get("promote_track_err", 0.25),
      demote_track_err=cfg.params.get("demote_track_err", 0.45),
      promote_attain=cfg.params.get("promote_attain", 0.75),
      demote_attain=cfg.params.get("demote_attain", 0.5),
      promote_wobble=cfg.params.get("promote_wobble", 0.10),
      demote_wobble=cfg.params.get("demote_wobble", 0.25),
      promote_fell=cfg.params.get("promote_fell", 0.3),
      demote_fell=cfg.params.get("demote_fell", 0.35),
      cooldown_iters=cfg.params.get("cooldown_iters", 150),
      demote_fast_fell=cfg.params.get("demote_fast_fell", 0.5),
      top_streak_required=cfg.params.get("top_streak_required", 5),
    )
    self._controller = CompetenceController(l_max=l_max, thresholds=thresholds)
    self._tracker = get_competence_tracker(env)
    self._last_check_iter = -1
    _apply_command_level(env, self._command_name, 0)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    l_max: int = 3,
    promote_track_err: float = 0.25,
    demote_track_err: float = 0.45,
    promote_attain: float = 0.75,
    demote_attain: float = 0.5,
    promote_wobble: float = 0.10,
    demote_wobble: float = 0.25,
    promote_fell: float = 0.3,
    demote_fell: float = 0.35,
    cooldown_iters: int = 150,
    demote_fast_fell: float = 0.5,
    top_streak_required: int = 5,
  ) -> dict[str, torch.Tensor]:
    del (
      command_name,
      l_max,
      promote_track_err,
      demote_track_err,
      promote_attain,
      demote_attain,
      promote_wobble,
      demote_wobble,
      promote_fell,
      demote_fell,
      cooldown_iters,
      demote_fast_fell,
      top_streak_required,
    )
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter != self._last_check_iter:
      self._last_check_iter = cur_iter
      self._controller.update(
        track_err_norm=stats["track_err_norm"],
        fell_ema=stats["fell_ema"],
        ep_len_frac=stats["ep_len_frac"],
        common_step_counter=env.common_step_counter,
        attain_ema=stats["attain"],
        wobble_ema=stats["wobble"],
        fast_fall_rate=stats["fast_fall_rate"],
      )
    snapshot = _apply_command_level(env, self._command_name, self._controller.level)
    snapshot.update({f"competence_{k}": torch.tensor(v) for k, v in stats.items()})
    return snapshot


class adaptive_push_level:
  """ADR-lite push magnitude curriculum gated on fell_ema."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_name: str = cfg.params["event_name"]
    l_max: int = cfg.params.get("l_max", 3)
    thresholds = CompetenceThresholds(
      promote_track_err=cfg.params.get("promote_track_err", 0.25),
      demote_track_err=cfg.params.get("demote_track_err", 0.45),
      promote_attain=cfg.params.get("promote_attain", 0.75),
      demote_attain=cfg.params.get("demote_attain", 0.5),
      promote_wobble=cfg.params.get("promote_wobble", 0.10),
      demote_wobble=cfg.params.get("demote_wobble", 0.25),
      promote_fell=cfg.params.get("promote_fell", 0.3),
      demote_fell=cfg.params.get("demote_fell", 0.35),
      cooldown_iters=cfg.params.get("cooldown_iters", 150),
      demote_fast_fell=cfg.params.get("demote_fast_fell", 0.5),
      top_streak_required=cfg.params.get("top_streak_required", 5),
    )
    self._controller = CompetenceController(
      l_max=l_max, thresholds=thresholds, gate_on_track_err=False
    )
    self._tracker = get_competence_tracker(env)
    start_level: int = cfg.params.get("start_level", 1)
    self._controller.level = start_level
    self._last_check_iter = -1
    _apply_push_level(env, self._event_name, start_level)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    l_max: int = 3,
    start_level: int = 1,
    promote_track_err: float = 0.25,
    demote_track_err: float = 0.45,
    promote_attain: float = 0.75,
    demote_attain: float = 0.5,
    promote_wobble: float = 0.10,
    demote_wobble: float = 0.25,
    promote_fell: float = 0.3,
    demote_fell: float = 0.35,
    cooldown_iters: int = 150,
    demote_fast_fell: float = 0.5,
    top_streak_required: int = 5,
  ) -> dict[str, torch.Tensor]:
    del (
      event_name,
      l_max,
      start_level,
      promote_track_err,
      demote_track_err,
      promote_attain,
      demote_attain,
      promote_wobble,
      demote_wobble,
      promote_fell,
      demote_fell,
      cooldown_iters,
      demote_fast_fell,
      top_streak_required,
    )
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter != self._last_check_iter:
      self._last_check_iter = cur_iter
      self._controller.update(
        track_err_norm=stats["track_err_norm"],
        fell_ema=stats["fell_ema"],
        ep_len_frac=stats["ep_len_frac"],
        common_step_counter=env.common_step_counter,
        attain_ema=stats["attain"],
        wobble_ema=stats["wobble"],
        fast_fall_rate=stats["fast_fall_rate"],
      )
    snapshot = _apply_push_level(env, self._event_name, self._controller.level)
    snapshot.update({f"competence_{k}": torch.tensor(v) for k, v in stats.items()})
    return snapshot


def _lerp_range(
  lo: tuple[float, float], hi: tuple[float, float], d: float
) -> tuple[float, float]:
  return (lo[0] + (hi[0] - lo[0]) * d, lo[1] + (hi[1] - lo[1]) * d)


@dataclass
class AimdParams:
  """AIMD (TCP-style) continuous difficulty control law.

  Every constant is calibrated against measured v20-v25 data; see doc 15
  R8. The congestion signal is the fast windowed fall rate (detection
  latency ~11 iters at its 0.2/iter EMA), so the additive rate is chosen
  so the controller cannot travel meaningfully past capacity within the
  detection lag: 0.002/iter x 11 iters = 0.022 in d = ~0.01 m/s of
  lin_vel_x (lerp span 0.5). Congestion cut 0.7 takes the burn zone
  (d~0.85, +-0.68 m/s) to +-0.55 in one event - below every observed
  ignition - instead of the level-cascade's trip to baby commands.
  Bars: 0.35 proved recoverable twice (v25-push bounces), 0.5 proved
  fatal (v24); healthy band is ~0.26. Increase gates hold difficulty
  while stressed and are feasibility-checked per the R3 standing rule
  (worst observed healthy values: fast 0.02-0.15, wobble 0.011-0.02,
  attain ceiling 0.543): they gate the RATE, not a promote event, so an
  infeasible gate freezes d rather than blocking a jump.
  """

  alpha: float = 0.002  # additive increase per healthy iteration
  ssthresh_alpha_div: float = 3.0  # probe rate divisor above ssthresh
  beta: float = 0.7  # multiplicative decrease per congestion event
  beta_emergency: float = 0.5  # harder cut while in emergency
  # Continuous per-iteration decay while the signal stays above the bar
  # (arrest mode, v27 postmortem): halves d in ~10 iterations, matching
  # the discrete cascade's aggression instead of waiting out refractories
  # during an active ignition.
  beta_arrest: float = 0.93
  congest_bar: float = 0.35  # fast fall rate: congestion event
  emergency_bar: float = 0.55  # fast fall rate: spiral in progress
  refractory_iters: int = 15  # one event = one cut (detection lag + margin)
  backoff_window_iters: int = 100  # repeat congestion inside this doubles
  refractory_max_iters: int = 60  # cap for doubled refractory
  backoff_reset_iters: int = 300  # clean streak restoring base refractory
  ssthresh_factor: float = 0.85  # high-water mark below the congestion d
  gate_fast: float = 0.20  # below healthy-band ceiling (0.26) w/ margin
  gate_wobble: float = 0.10  # healthy 0.011-0.02; 5x margin
  gate_attain: float = 0.40  # worst ceiling 0.543; 26% margin


class AimdDifficultyState(TypedDict):
  d: float
  ssthresh: float
  refractory: int
  last_cut_iter: int
  last_congest_iter: int


class AimdController:
  """Additive-increase / multiplicative-decrease difficulty controller.

  Chiu-Jain: AIMD is the control law that converges to a stable sawtooth
  around an unknown capacity under binary congestion feedback. Difficulty
  is a single scalar d in [0, 1] so command width and push magnitude move
  together (v25-push showed independent ladders co-promote into the same
  wall).
  """

  def __init__(self, params: AimdParams):
    self.p = params
    self.d = 0.0
    self.ssthresh = 1.0
    self.refractory = params.refractory_iters
    self.last_cut_iter = -(10**9)
    self.last_congest_iter = -(10**9)

  def state_dict(self) -> AimdDifficultyState:
    return {
      "d": self.d,
      "ssthresh": self.ssthresh,
      "refractory": self.refractory,
      "last_cut_iter": self.last_cut_iter,
      "last_congest_iter": self.last_congest_iter,
    }

  def load_state_dict(self, state: AimdDifficultyState) -> None:
    self.d = float(state["d"])
    self.ssthresh = float(state["ssthresh"])
    self.refractory = int(state["refractory"])
    self.last_cut_iter = int(state["last_cut_iter"])
    self.last_congest_iter = int(state["last_congest_iter"])

  def update(
    self,
    *,
    cur_iter: int,
    fast_fall_rate: float,
    wobble_ema: float,
    attain_ema: float,
  ) -> str | None:
    p = self.p
    in_refractory = cur_iter - self.last_cut_iter < self.refractory

    if fast_fall_rate >= p.congest_bar:
      # Nothing to cut near d=0 (early-training chaos under the pessimistic
      # EMA init): do no bookkeeping, or the meaningless "cuts" pin
      # ssthresh at its floor and the whole climb runs at probe rate
      # (observed on v26: alpha/3 from iter ~550).
      if self.d < 0.05:
        return None
      if in_refractory:
        # ARREST MODE (v27 postmortem): the refractory gates re-entry, not
        # survival. One-cut-per-refractory walked v27 from d=1.0 to 0.28
        # over ~100+ iterations of sustained super-bar fall rate while the
        # -10s poisoned the batch; the discrete cascade this replaced did
        # its full descent in 5. While the signal STAYS above the bar,
        # keep cutting every iteration at beta_arrest (halving in ~10).
        self.d *= p.beta_arrest
        return "arrest"
      # Exponential backoff at persistent walls (TCP RTO analog): repeat
      # congestion shortly after the last one doubles the refractory.
      if cur_iter - self.last_congest_iter <= p.backoff_window_iters:
        self.refractory = min(self.refractory * 2, p.refractory_max_iters)
      elif cur_iter - self.last_congest_iter >= p.backoff_reset_iters:
        self.refractory = p.refractory_iters
      beta = p.beta_emergency if fast_fall_rate >= p.emergency_bar else p.beta
      self.ssthresh = max(self.d * p.ssthresh_factor, 0.05)
      self.d *= beta
      self.last_cut_iter = cur_iter
      self.last_congest_iter = cur_iter
      return "cut"

    if in_refractory:
      return None
    healthy = (
      fast_fall_rate < p.gate_fast
      and wobble_ema < p.gate_wobble
      and attain_ema > p.gate_attain
    )
    if not healthy:
      return None
    alpha = p.alpha
    if self.d >= self.ssthresh:
      alpha /= p.ssthresh_alpha_div
    self.d = min(self.d + alpha, 1.0)
    return "increase"


class aimd_difficulty:
  """Continuous TCP-style difficulty, split per axis (v28 postmortem).

  v28 died of a cohort-blind governor: the pushed cohort burned above the
  congestion bar for ~1000 iterations (fast_fall_pushed 0.34->0.62 from
  iter ~970) while the single controller read the blended population rate
  (~0.15, diluted by the healthy 70% clean cohort) - a third of every
  batch was -10 poison, invisible to the one signal consumed, and the
  single scalar could not have eased pushes without easing commands even
  if it had seen it. Control now matches the stratified measurement:

  - d_cmd (command ellipsoid scale): congestion on fast_fall_clean.
  - d_push (push magnitude): congestion on the EXCESS rate
    (fast_fall_pushed - fast_fall_clean), so it responds to push-specific
    damage without double-charging when everything burns. Replay: at
    v28's iter 970 the excess was 0.30 -> pushes would have cut 2.0x ->
    1.4x right at burn onset while commands held.
  - Population emergency (>= emergency_bar) arrests BOTH axes.

  Falls back to blended signals when the cohort split is off
  (PUSH_COHORT_FRAC=1: stratified means degrade to global means).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._event_name: str = cfg.params["event_name"]
    base = dict(
      alpha=cfg.params.get("alpha", 0.002),
      beta=cfg.params.get("beta", 0.7),
      emergency_bar=cfg.params.get("emergency_bar", 0.55),
      gate_attain=cfg.params.get("gate_attain", 0.40),
      beta_arrest=cfg.params.get("beta_arrest", 0.93),
    )
    self._cmd_params = AimdParams(
      congest_bar=cfg.params.get("congest_bar", 0.35), **base
    )
    # Push axis: bar 0.30 on the excess rate (healthy excess measured
    # 0.15-0.18 at moderate push scales; 0.30 was the v28 burn onset).
    # gate_fast 0.15 holds push growth while excess is elevated.
    self._push_params = AimdParams(
      congest_bar=cfg.params.get("push_congest_bar", 0.30),
      gate_fast=cfg.params.get("push_gate_excess", 0.15),
      **base,
    )
    # Envelope extension (v27 postmortem): capacity turned out to be AT or
    # above the L5 table, so d parked at the cap, the sawtooth never
    # engaged, and saturation churn returned on schedule. The lerp target
    # is scaled past the table so true capacity always sits BELOW the cap
    # and AIMD oscillates at capacity instead of at an artificial ceiling.
    # Safe only in combination with arrest mode + ellipsoid geometry.
    self._envelope_scale: float = cfg.params.get("envelope_scale", 1.0)
    # Attainment-slide congestion (v29 postmortem, R13): under ellipsoid
    # geometry, commands beyond capability UNDER-TRACK instead of causing
    # falls, so fall-based congestion never binds on the command axis at
    # high speed - d_cmd parks at the cap and saturation churn returns on
    # the usual ~800-iter fuse (v29: parked at 1045, attain slid
    # 0.712->0.66 from 1520, fell-creep from 1615, dead 2106). The churn
    # signature IS the attain slide, so it becomes the command axis's
    # second congestion signal: attain below attain_slide_frac of its
    # trailing max fires one cut per refractory. Replay: bar 0.95 fires
    # at ~iter 1810 (wd 1.39, fast_clean 0.13 - fully recoverable);
    # healthy dP-cut wobbles (1.4% dips) stay well inside the band.
    self._attain_slide_frac: float = cfg.params.get("attain_slide_frac", 0.95)
    # Band controller + attained floor (R19, user design after v35/v36
    # falsified the penalty theory): the open-loop climb overshot
    # demonstrated capability by 40-50% in every dead run (vmax 0.98 at
    # attained ~0.64-0.70), and the arrest cascade then cut commands FAR
    # below attainment (0.98 -> 0.24 vs attained 0.66) - which is when
    # falls actually exploded (0.14 -> 0.6): the descent was iatrogenic.
    # Now: climb only while clean attain > band_hi, glide down at a
    # bounded slew when attain < band_lo (no halving cascades), and
    # NEVER command below floor_frac of the best recently attained
    # speed - fall cuts and emergencies clamp to the floor; the watchdog
    # remains the catastrophe escape.
    self._band_hi: float = cfg.params.get("attain_band_hi", 0.66)
    self._band_lo: float = cfg.params.get("attain_band_lo", 0.60)
    self._glide_mult: float = cfg.params.get("glide_mult", 2.0)
    self._floor_frac: float = cfg.params.get("floor_frac", 0.95)
    self._attained_best_v = 0.0
    self._attain_max = 0.0
    # Landing anneal (R14): task-side relief cannot stop churn once begun
    # (five demonstrations, v24-v30); the only lever left for runs longer
    # than ~1800 iters is optimizer-side. Once the run is AT CAPACITY
    # (d_cmd >= at_capacity_d sustained) AND PLATEAUED (attain within 2%
    # of its trailing max sustained), there is nothing left to learn at
    # this difficulty - so the landing factor decays monotonically and
    # the runner scales desired_kl by it (the adaptive schedule then
    # walks the LR to its floor): convergence instead of churn fuel.
    # Churn onset despite the anneal (attain slide) hard-freezes faster.
    self._landing_enabled: bool = bool(cfg.params.get("landing_anneal", False))
    self._landing_factor = 1.0
    self._at_capacity_since = -1
    self._plateau_since = -1
    self._cmd_ctrl = AimdController(self._cmd_params)
    self._push_ctrl = AimdController(self._push_params)
    self._tracker = get_competence_tracker(env)
    self._last_check_iter = -1
    self._apply(env, 0.0, 0.0)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def _apply(
    self, env: ManagerBasedRlEnv, d_cmd: float, d_push: float
  ) -> dict[str, torch.Tensor]:
    lo, base_hi = COMMAND_LEVEL_TABLE[0], COMMAND_LEVEL_TABLE[-1]
    es = self._envelope_scale
    hi = {axis: (rng[0] * es, rng[1] * es) for axis, rng in base_hi.items()}
    command_term = env.command_manager.get_term(self._command_name)
    assert command_term is not None
    ccfg = command_term.cfg
    assert isinstance(ccfg, UniformVelocityCommandCfg)
    ccfg.ranges.lin_vel_x = _lerp_range(lo["lin_vel_x"], hi["lin_vel_x"], d_cmd)
    ccfg.ranges.lin_vel_y = _lerp_range(lo["lin_vel_y"], hi["lin_vel_y"], d_cmd)
    ccfg.ranges.ang_vel_z = _lerp_range(lo["ang_vel_z"], hi["ang_vel_z"], d_cmd)

    push_scale = (
      PUSH_LEVEL_SCALES[0] + (PUSH_LEVEL_SCALES[-1] - PUSH_LEVEL_SCALES[0]) * d_push
    )
    ecfg = env.event_manager.get_term_cfg(self._event_name)
    ecfg.params["velocity_range"] = _scale_push_velocity_range(push_scale)

    return {
      "difficulty": torch.tensor(d_cmd),
      "difficulty_push": torch.tensor(d_push),
      "ssthresh": torch.tensor(self._cmd_ctrl.ssthresh),
      "ssthresh_push": torch.tensor(self._push_ctrl.ssthresh),
      "lin_vel_x_max": torch.tensor(ccfg.ranges.lin_vel_x[1]),
      "ang_vel_z_max": torch.tensor(ccfg.ranges.ang_vel_z[1]),
      "push_scale": torch.tensor(push_scale),
      "attain_trailing_max": torch.tensor(self._attain_max),
      "landing_factor": torch.tensor(self._landing_factor),
    }

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    event_name: str,
    alpha: float = 0.002,
    beta: float = 0.7,
    congest_bar: float = 0.35,
    emergency_bar: float = 0.55,
    gate_attain: float = 0.40,
    beta_arrest: float = 0.93,
    envelope_scale: float = 1.0,
    push_congest_bar: float = 0.30,
    push_gate_excess: float = 0.15,
    attain_slide_frac: float = 0.95,
    landing_anneal: bool = False,
    attain_band_hi: float = 0.66,
    attain_band_lo: float = 0.60,
    glide_mult: float = 2.0,
    floor_frac: float = 0.95,
  ) -> dict[str, torch.Tensor]:
    del (command_name, event_name, alpha, beta, congest_bar, emergency_bar)
    del (gate_attain, beta_arrest, envelope_scale)
    del (push_congest_bar, push_gate_excess, attain_slide_frac, landing_anneal)
    del (attain_band_hi, attain_band_lo, glide_mult, floor_frac)
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    strat = self._tracker.stratified_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter != self._last_check_iter:
      self._last_check_iter = cur_iter
      pop_fast = stats["fast_fall_rate"]
      emergency = pop_fast >= self._cmd_params.emergency_bar
      excess = max(0.0, strat["fast_fall_pushed"] - strat["fast_fall_clean"])
      att = strat["clean_attain"]
      # Sticky short-term, adaptive long-term (~14k-iter half-life).
      self._attain_max = max(self._attain_max * 0.99995, att)
      attain_slide = (
        self._attain_max > 0.5 and att < self._attain_slide_frac * self._attain_max
      )
      # Attained-speed memory (slow decay) and its floor in d-units (R19).
      lo_v = COMMAND_LEVEL_TABLE[0]["lin_vel_x"][1]
      hi_v = COMMAND_LEVEL_TABLE[-1]["lin_vel_x"][1] * self._envelope_scale
      vmax_now = lo_v + (hi_v - lo_v) * self._cmd_ctrl.d
      self._attained_best_v = max(self._attained_best_v * 0.999, att * vmax_now)
      floor_d = max(
        0.0, (self._floor_frac * self._attained_best_v - lo_v) / (hi_v - lo_v)
      )
      cmd_signal = max(strat["fast_fall_clean"], pop_fast if emergency else 0.0)
      if self._landing_enabled:
        if self._cmd_ctrl.d >= 0.95:
          if self._at_capacity_since < 0:
            self._at_capacity_since = cur_iter
        else:
          self._at_capacity_since = -1
        plateaued = self._attain_max > 0.5 and att >= 0.98 * self._attain_max
        if plateaued:
          if self._plateau_since < 0:
            self._plateau_since = cur_iter
        else:
          self._plateau_since = -1
        ready = (
          self._at_capacity_since >= 0
          and cur_iter - self._at_capacity_since >= 200
          and self._plateau_since >= 0
          and cur_iter - self._plateau_since >= 150
        )
        if attain_slide:
          self._landing_factor = max(self._landing_factor * 0.9, 0.02)
        elif ready:
          self._landing_factor = max(self._landing_factor * 0.995, 0.02)
        env._landing_factor = self._landing_factor
      # Fall cuts keep the AIMD machinery; attain_ema=0 blocks its own
      # additive path (the band below owns all upward movement).
      self._cmd_ctrl.update(
        cur_iter=cur_iter,
        fast_fall_rate=cmd_signal,
        wobble_ema=strat["clean_wobble"],
        attain_ema=0.0,
      )
      if cmd_signal < self._cmd_params.congest_bar:
        if att > self._band_hi and strat["clean_wobble"] < self._cmd_params.gate_wobble:
          self._cmd_ctrl.d = min(self._cmd_ctrl.d + self._cmd_params.alpha, 1.0)
        elif att < self._band_lo:
          self._cmd_ctrl.d -= self._glide_mult * self._cmd_params.alpha
      self._cmd_ctrl.d = max(self._cmd_ctrl.d, floor_d)
      self._push_ctrl.update(
        cur_iter=cur_iter,
        fast_fall_rate=max(excess, pop_fast if emergency else 0.0),
        wobble_ema=strat["pushed_wobble"],
        attain_ema=strat["clean_attain"],
      )
    snapshot = self._apply(env, self._cmd_ctrl.d, self._push_ctrl.d)
    snapshot.update({f"competence_{k}": torch.tensor(v) for k, v in stats.items()})
    return snapshot


class track_reward_watchdog:
  """Fail fast when a once-good policy rots (user rule, 2026-07-05).

  Signal: the exact quantity logged as Episode_Reward/<reward_name>
  (episodic sums are read BEFORE the reward manager zeroes them - the
  curriculum manager runs first in _reset_idx), smoothed with a 0.05/iter
  EMA (~20-iter response). Arms once the EMA exceeds arm_above (every
  healthy corrected-physics run passed 2.0 by iter ~800; observed
  2.26-2.43); fires when the EMA stays below fail_below for
  fail_persist_iters consecutive iterations. Persistence is calibrated
  against the v25-push bounce traces: transient dips reach ~0.85 for
  <=45 iters and the EMA barely crosses 1.0, while v25-slow's rot (a
  600-iter slide) exceeds any plausible persistence bar. Firing raises
  RuntimeError: the pod fails, torchrun tears down the gang, and the job
  fails fast instead of burning GPU-hours on an unrecoverable policy.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del env
    self._reward_name: str = cfg.params.get("reward_name", "track_linear_velocity")
    self._arm_above: float = cfg.params.get("arm_above", 2.0)
    self._fail_below: float = cfg.params.get("fail_below", 1.0)
    self._persist_iters: int = cfg.params.get("fail_persist_iters", 60)
    self._ema_alpha: float = cfg.params.get("ema_alpha", 0.05)
    self._ema: float = 0.0
    self._armed = False
    self._below_count = 0
    self._last_iter = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str = "track_linear_velocity",
    arm_above: float = 2.0,
    fail_below: float = 1.0,
    fail_persist_iters: int = 60,
    ema_alpha: float = 0.05,
  ) -> dict[str, torch.Tensor]:
    del (reward_name, arm_above, fail_below, fail_persist_iters, ema_alpha)
    sums = env.reward_manager._episode_sums.get(self._reward_name)
    if sums is not None and not isinstance(env_ids, slice) and len(env_ids) > 0:
      window = sums[env_ids].mean().item() / env.max_episode_length_s
      self._ema += self._ema_alpha * (window - self._ema)

    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter != self._last_iter:
      self._last_iter = cur_iter
      if self._ema > self._arm_above:
        self._armed = True
      if self._armed and self._ema < self._fail_below:
        self._below_count += 1
      else:
        self._below_count = 0
      if self._below_count >= self._persist_iters:
        raise RuntimeError(
          f"track_reward_watchdog: {self._reward_name} EMA "
          f"{self._ema:.3f} below {self._fail_below} for "
          f"{self._below_count} iters after peaking above "
          f"{self._arm_above} - policy is rotting, failing fast."
        )
    return {
      "ema": torch.tensor(self._ema),
      "armed": torch.tensor(1.0 if self._armed else 0.0),
      "below_count": torch.tensor(float(self._below_count)),
    }


class staged_on_competence:
  """Ramp penalty weights when stability competence holds; back off on loss.

  Promote (next stage) requires stability competence AND the cooldown.
  Demote (previous stage) fires when stability is badly lost
  (``fell_ema > demote_fell``), also cooldown-limited so a single demotion
  gets time to take effect before the next. This is the disease-#2
  countermeasure (doc 14): a pure freeze cannot recover a policy that is
  already sliding down the penalty gradient — releasing penalty pressure
  restores the basin the gait was learned in and gives the policy a path
  back. Between the demote and re-promote thresholds lies the freeze band.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    from mjlab.envs.mdp.curriculums import (
      RewardCurriculumStage,
      _apply_stages,
      _validate_stages,
    )

    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    self._tracker = get_competence_tracker(env)
    self._controller = CompetenceController(
      l_max=len(stages) - 1,
      thresholds=CompetenceThresholds(
        cooldown_iters=cfg.params.get("cooldown_iters", 150),
        demote_fell=cfg.params.get("demote_fell", 0.35),
        demote_attain=cfg.params.get("demote_attain", 0.5),
        demote_wobble=cfg.params.get("demote_wobble", 0.25),
        demote_fast_fell=cfg.params.get("demote_fast_fell", 0.5),
      ),
    )
    self._stage_idx = 0
    self._last_change_iter = -cfg.params.get("cooldown_iters", 50)
    _validate_stages(self._term_cfg, reward_name, self._stages)
    _apply_stages(self._term_cfg, self._stages[0]["step"], self._stages)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[dict[str, float]],
    cooldown_iters: int = 150,
    demote_fell: float = 0.35,
    promote_track_err: float = 0.25,
    demote_track_err: float = 0.45,
    promote_attain: float = 0.75,
    demote_attain: float = 0.5,
    promote_wobble: float = 0.10,
    demote_wobble: float = 0.25,
    promote_fell: float = 0.3,
    demote_fast_fell: float = 0.5,
    top_streak_required: int = 5,
  ) -> dict[str, torch.Tensor]:
    # The curriculum manager passes ALL cfg params as kwargs
    # (curriculum_manager.py: ``func(env, env_ids, **term_cfg.params)``), so
    # this signature must accept the full threshold set even though the
    # values are consumed in __init__ via the controller.
    from mjlab.envs.mdp.curriculums import _apply_stages

    del (
      reward_name,
      stages,
      cooldown_iters,
      demote_fell,
      promote_track_err,
      demote_track_err,
      promote_attain,
      demote_attain,
      promote_wobble,
      demote_wobble,
      promote_fell,
      demote_fast_fell,
      top_streak_required,
    )
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    cooldown = self._controller.thresholds.cooldown_iters
    cooldown_elapsed = cur_iter - self._last_change_iter >= cooldown
    stable = self._controller.stability_ok(
      fell_ema=stats["fell_ema"],
      ep_len_frac=stats["ep_len_frac"],
      attain_ema=stats["attain"],
    )
    t = self._controller.thresholds
    badly_lost = (
      stats["fell_ema"] > t.demote_fell
      or stats["attain"] < t.demote_attain
      or stats["wobble"] > t.demote_wobble
      or stats["fast_fall_rate"] > t.demote_fast_fell
    )

    if badly_lost and self._stage_idx > 0 and cooldown_elapsed:
      self._stage_idx -= 1
      self._last_change_iter = cur_iter
    elif stable and self._stage_idx < len(self._stages) - 1 and cooldown_elapsed:
      self._stage_idx += 1
      self._last_change_iter = cur_iter

    snapshot = _apply_stages(
      self._term_cfg, self._stages[self._stage_idx]["step"], self._stages
    )
    snapshot["stage_idx"] = torch.tensor(self._stage_idx)
    snapshot["competence_fell_ema"] = torch.tensor(stats["fell_ema"])
    snapshot["competence_ep_len_frac"] = torch.tensor(stats["ep_len_frac"])
    return snapshot
