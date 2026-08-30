"""Competence-gated reward weights and frontier diagnostics.

Two cooperating pieces share one :class:`CompetenceTracker`:

* :class:`staged_on_competence` — a curriculum term that advances (or
  retreats) a penalty term's weight along a stage ladder based on whether
  the population is currently demonstrating stability competence.
* :class:`competence_diagnostics` — a log-only curriculum term publishing
  cohort-stratified competence signals and the frontier estimates (fall
  hazard and attainment as functions of commanded speed, push-survival by
  shove magnitude, push-to-fall timing).

Every threshold is a hardcoded constant or a dataclass default; there are
no runtime knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab.envs.mdp.curriculums import (
  RewardCurriculumStage,
  _apply_stages,
  _validate_stages,
)
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

_NUM_STEPS_PER_ENV = 24
_MIN_CMD_NORM = 0.2

# Fraction of env indices that receive pushes. The remaining ("clean")
# cohort trains push-free: it is the uncontaminated tracking-competence
# baseline and matches the mostly-push-free deployment distribution. The
# frontier estimator draws its exposure and falls exclusively from the
# clean cohort, so this must be < 1.0 for the frontier to have data.
PUSH_COHORT_FRAC = 0.3

# Fall-hazard level whose first up-crossing defines ``frontier_speed``.
FRONTIER_HAZARD_BAR = 5e-4

# Bootstrap push-observation window; the tracker adapts it to the measured
# push->fall t75 once the delay histogram has evidence.
PUSH_OBS_WINDOW_S = 6.0

# Attainment bar for the capability curve's interpolated crossing.
ATTAINED_FRONTIER_BAR = 0.60

# Upper commanded speed of the "core band" the gate's attainment reads. A
# gate that reads attainment over the WHOLE command box is confounded with
# how ambitious the box is: a limit-pushing curriculum commands speeds
# beyond the frontier on purpose, and every such command drags the mean
# down no matter how well the policy walks, so the gate never promotes.
# Restricting the gate's view to commands comfortably inside any plausible
# frontier separates "is the policy actually trying" from "how far past the
# frontier are we commanding". Low enough to be reachable, high enough to
# require a real gait rather than a shuffle.
CORE_CMD_SPEED = 0.6


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
  fall at terminal collapse. Attainment (commanded-direction speed achieved
  / commanded speed) is the LEADING indicator — sandbagging reads ~0 while
  never falling — and wobble fraction (steps with tilt > ~25 deg) is the
  graded near-fall precursor. Falls remain as the safety net only.

  The attainment bars are read against ``attain_core_ema`` (commands up to
  ``CORE_CMD_SPEED``), not the full-box mean: the full-box mean is capped by
  however far past the frontier the command curriculum reaches, so bars set
  against it are really bars on the command range.
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
  # Fast demote channel: the per-env EMAs update only at episode ends, so
  # at healthy episode lengths (~900 steps) a crash at the top of the
  # ladder is invisible for ~200 iterations — long enough for the fall
  # terminations to dominate the gradient and shatter the policy. The
  # windowed population fall rate crosses this bar within ~5 iterations of
  # a real crash; healthy top-rung operation measures ~0.26
  # falls/episode-end, so 0.5 clears the working band while catching the
  # spiral at its front edge.
  demote_fast_fell: float = 0.5
  # Extra promote caution on the top rungs.
  top_streak_required: int = 5
  top_level_start: int = 4


class CompetenceController:
  """Hysteresis + cooldown level controller driven by population competence."""

  def __init__(self, *, l_max: int, thresholds: CompetenceThresholds):
    self.l_max = l_max
    self.thresholds = thresholds
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
    fell_ema: float,
    common_step_counter: int,
    attain_ema: float = 0.0,
    wobble_ema: float = 1.0,
    fast_fall_rate: float = 0.0,
  ) -> str | None:
    """Return ``promote``, ``demote``, or None.

    Promote needs the task actually being performed (attainment) with low
    near-fall wobble and low falls; demote fires on sandbagging (attainment
    collapse), sustained wobble, or falls.
    """
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
    # Sandbag detector for the gate: attainment over the core command band
    # only (see CORE_CMD_SPEED). The full-box ``attain_ema`` stays the
    # headline number and the degradation signal.
    self._attain_core_sum = torch.zeros(n, device=self.device)
    self._attain_core_weight = torch.zeros(n, device=self.device)
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
    self.attain_core_ema = torch.zeros(n, device=self.device)
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
    # Push-cohort stratification: a fixed slice of env indices receives
    # pushes; the rest train push-free. Attribution is by membership — no
    # recovery-horizon guess — and the clean cohort matches the mostly
    # push-free deployment distribution. The policy cannot observe
    # membership. 1.0 (all pushed) leaves the clean cohort empty.
    self.push_cohort_frac = 1.0
    self.push_cohort = torch.ones(n, dtype=torch.bool, device=self.device)
    self.last_push_step = torch.full((n,), -(10**9), device=self.device)
    self._win_fell_clean = 0.0
    self._win_done_clean = 0.0
    self._win_fell_pushed = 0.0
    self._win_done_pushed = 0.0
    self.fast_fall_clean = 1.0
    self.fast_fall_pushed = 1.0
    # Time-from-push-to-fall histogram: answers "how long does recovery
    # take" empirically. 0.5 s bins over [0, 20 s), sized to the training
    # episode so the instrument out-ranges every dt attribution can
    # produce. Windowed like the survival bins: falls scatter into
    # ``_push_fall_dt_win``; at the hazard-refresh cadence a window holding
    # >= 50 events folds into the EMA'd ``push_fall_dt_counts`` and clears,
    # while sparse windows keep accumulating so evidence is never
    # discarded. The EMA is what t50/t75, the adaptive observation window,
    # and the histogram read — the CURRENT policy's recovery profile, not
    # the run-lifetime average.
    self.n_dt_bins = 40
    self.dt_bin_width = 0.5
    self.push_fall_dt_counts = torch.zeros(self.n_dt_bins, device=self.device)
    self._push_fall_dt_win = torch.zeros(self.n_dt_bins, device=self.device)
    # Push survival frontier: per-event outcomes binned by shove magnitude
    # |dv_xy|. A push survives if no fall arrives before the next push or
    # timeout; a fall charges the pending push's bin.
    self.n_push_bins = 40
    self.push_bin_width = 0.05  # covers |dv| to 2.0 m/s
    # Observation window: survival credit requires this many seconds of
    # clean observation after the push. Anything ambiguous is CENSORED: an
    # intervening push inside the window discards the earlier event
    # (hard-then-easy must not launder the hard one), and a timeout inside
    # the window is observation ending, not survival. Adaptive: tracks the
    # live t75 of the measured push->fall delay distribution once >= 50
    # fall events exist (the histogram is measured independently of the
    # attribution window, so this is a clean closed loop). The configured
    # value is only the bootstrap; clamped to [2, 12] s so a stretched tail
    # cannot censor everything at 3-10 s push intervals.
    self.push_obs_window_s = PUSH_OBS_WINDOW_S
    self._pending_push_mag = torch.full((n,), -1.0, device=self.device)
    self._pending_push_step = torch.full((n,), -1.0, device=self.device)
    self._push_bin_survive = torch.zeros(self.n_push_bins, device=self.device)
    self._push_bin_fall = torch.zeros(self.n_push_bins, device=self.device)
    self.push_survival = torch.ones(self.n_push_bins, device=self.device)
    self.push_survival_weight = torch.zeros(self.n_push_bins, device=self.device)
    # Frontier estimator (clean cohort only): exposure steps and falls
    # densely binned — bins are the sufficient statistic, and the readouts
    # are interpolated level-crossings/quantiles, which integrate across
    # bins and are robust to fine binning.
    self.n_cmd_buckets = 64
    self.speed_bin_width = 0.05  # covers commanded speeds to 3.20 m/s
    self._bucket_steps = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._bucket_falls = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.bucket_hazard = torch.zeros(self.n_cmd_buckets, device=self.device)
    # Persistent per-bin exposure EMA (folded with the hazard): lets the
    # frontier readouts clamp to "as far as we have actually sampled"
    # instead of returning the instrument end when no hazard crossing
    # exists (a near-zero fall rate would otherwise read the full
    # instrument range as if it were a capability claim).
    self.bucket_exposure = torch.zeros(self.n_cmd_buckets, device=self.device)
    # Flight-by-speed (walk->run boundary): per-bin counts of TRUE-flight
    # steps (all feet airborne AND upright) over eligible exposure, so the
    # boundary crossing reads as a fraction-of-steps-in-flight curve
    # against commanded speed. Two contamination gates: frames within 1 s
    # of a push are excluded entirely (a shove can toss the robot airborne
    # with no gait flight due), and the upright gate (same 25 deg tilt
    # bound as the wobble metric) keeps falls/tumbles from counting as air
    # time. Cumulative counts; evidence-masked at histogram-log time.
    self.flight_steps_by_speed = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.flight_exposure_by_speed = torch.zeros(self.n_cmd_buckets, device=self.device)
    # Attainment conditional on commanded speed: the capability curve.
    # Windowed sums -> EMA'd per-bin curve at the hazard cadence; the
    # interpolated bar-crossing of this curve (attained_frontier) says
    # WHERE failure lives, which the population-mean attain cannot.
    self._attain_bin_sum = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._attain_bin_weight = torch.zeros(self.n_cmd_buckets, device=self.device)
    # Survivor conditioning: attainment credit toward the capability curve
    # buffers PER EPISODE and folds into the window bins only when the
    # episode ends by TIMEOUT — a lunge that attains a speed and then falls
    # contributes nothing, so the frontier cannot ratchet on stunts.
    # Duration weighting is inherent (per-step samples). The
    # population-mean attain EMA deliberately still counts all episodes: it
    # is the degradation/sandbag detector and must not freeze during
    # crashes.
    self._attain_ep_sum = torch.zeros((n, self.n_cmd_buckets), device=self.device)
    self._attain_ep_weight = torch.zeros((n, self.n_cmd_buckets), device=self.device)
    # Attainment must certify MAINTAINED speed, not touched speed. Two
    # guards, both censoring (evidence discarded, not counted against):
    # (1) settle exclusion — steps count toward a speed bin only after the
    # command has been held ``attain_settle_s``, so the acceleration
    # transient after a resample (or a shove-induced surge) never mints
    # credit; (2) minimum dwell — an episode's evidence for a bin folds
    # into the frontier histogram only if it accumulated
    # ``attain_min_dwell_s`` of post-settle time, so a high-speed command
    # landing just before timeout is censored like an end-of-episode push.
    # Segments between resamples run 5-10 s, so honest evidence passes
    # both bars comfortably.
    self.attain_settle_s = 0.75
    self.attain_min_dwell_s = 3.0
    self._bin_dwell = torch.zeros(n, device=self.device)
    self.attain_by_speed = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.attain_by_speed_weight = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._cur_bucket = torch.full((n,), -1, dtype=torch.long, device=self.device)
    self._bucket_next_step = 0
    # Mahalanobis-radius buckets: rho normalizes (vx, vy, wz) by the
    # current per-axis maxima, so hazards are measured in the geometry that
    # respects axis coupling — under box sampling the high bins ARE the
    # corners (rho spans up to ~1.7).
    self._rho_steps = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._rho_falls = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.rho_hazard = torch.zeros(self.n_cmd_buckets, device=self.device)
    self.rho_exposure = torch.zeros(self.n_cmd_buckets, device=self.device)
    self._cur_rho_bucket = torch.full((n,), -1, dtype=torch.long, device=self.device)
    try:
      self._step_dt = float(env.step_dt)
    except (TypeError, ValueError):
      self._step_dt = 0.02

  def set_push_cohort(self, frac: float) -> None:
    self.push_cohort_frac = frac
    n_pushed = int(round(frac * self.num_envs))
    self.push_cohort = torch.arange(self.num_envs, device=self.device) < n_pushed

  def record_push(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    magnitudes: torch.Tensor | None = None,
  ) -> None:
    if magnitudes is not None:
      # Settle the previous pending push: SURVIVAL only if its full
      # observation window elapsed cleanly before this new push; otherwise
      # the earlier event is censored — no credit, so a hard push followed
      # quickly by an easy one is never laundered.
      window_steps = self.push_obs_window_s / self._step_dt
      pending = self._pending_push_mag[env_ids]
      elapsed = env.common_step_counter - self._pending_push_step[env_ids]
      survived = (pending >= 0) & (elapsed >= window_steps)
      if survived.any():
        bins = (
          (pending[survived] / self.push_bin_width)
          .long()
          .clamp(0, self.n_push_bins - 1)
        )
        self._push_bin_survive.scatter_add_(
          0, bins, torch.ones(len(bins), device=self.device)
        )
      self._pending_push_mag[env_ids] = magnitudes
      self._pending_push_step[env_ids] = float(env.common_step_counter)
    self.last_push_step[env_ids] = float(env.common_step_counter)

  def state_dict(self) -> dict[str, Any]:
    return {
      "track_err_ema": self.track_err_ema.cpu(),
      "fell_ema": self.fell_ema.cpu(),
      "ep_len_frac_ema": self.ep_len_frac_ema.cpu(),
      "attain_ema": self.attain_ema.cpu(),
      "attain_core_ema": self.attain_core_ema.cpu(),
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
    if "attain_core_ema" in state:
      self.attain_core_ema = state["attain_core_ema"].to(self.device)
    if "fast_fall_rate" in state:
      self.fast_fall_rate = float(state["fast_fall_rate"])

  def record_step(self, env: ManagerBasedRlEnv) -> None:
    term = env.command_manager.get_term("twist")
    if term is None:
      return
    command_term = cast("UniformVelocityCommand", term)
    # Wobble (near-fall precursor): fraction of steps with tilt > ~25 deg,
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
    # single fall; lateral sway is orthogonal and drops out. The
    # meaningful-command filter plays the anti-gaming role a denominator
    # floor otherwise would, without capping perfect tracking of small
    # commands.
    cmd_sq = (cmd_xy * cmd_xy).sum(dim=-1)
    meaningful = walking & (cmd_sq >= 0.15 * 0.15)
    attain = (vel_xy * cmd_xy).sum(dim=-1) / cmd_sq.clamp(min=1e-6)
    self._attain_sum[meaningful] += attain[meaningful]
    self._attain_weight[meaningful] += 1.0
    core = meaningful & (cmd_sq <= CORE_CMD_SPEED * CORE_CMD_SPEED)
    self._attain_core_sum[core] += attain[core]
    self._attain_core_weight[core] += 1.0

    # Per-axis attainment: achieved/commanded per direction, each sample
    # weighted by that axis's share of command energy, so "when commands
    # ask for x, do we deliver x" is answered independently of y.
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
    cmd_speed = torch.sqrt(cmd_sq)
    bucket = (cmd_speed / self.speed_bin_width).long().clamp(0, self.n_cmd_buckets - 1)
    new_bucket = torch.where(walking, bucket, torch.full_like(bucket, -1))
    same_bin = new_bucket == self._cur_bucket
    self._bin_dwell = torch.where(
      same_bin, self._bin_dwell + 1.0, torch.zeros_like(self._bin_dwell)
    )
    self._cur_bucket = new_bucket
    expose = walking & ~self.push_cohort
    if expose.any():
      self._bucket_steps.scatter_add_(
        0, bucket[expose], torch.ones(int(expose.sum()), device=self.device)
      )

    # Flight accounting (see buffer docstring): eligible = walking and
    # outside the 1 s post-push window; flight = eligible + all feet
    # airborne + upright.
    try:
      contact = env.scene["feet_ground_contact"]
      found = contact.data.found
    except KeyError:
      found = None
    if isinstance(found, torch.Tensor):
      airborne = (found == 0).all(dim=1)
      upright = torch.norm(grav_xy, dim=-1) <= 0.4226  # sin(25 deg)
      no_recent_push = (
        float(env.common_step_counter) - self.last_push_step
      ) > 1.0 / self._step_dt
      settled = env.episode_length_buf > 25  # spawn drop-in exclusion
      eligible = walking & no_recent_push & settled
      if eligible.any():
        self.flight_exposure_by_speed.scatter_add_(
          0, bucket[eligible], torch.ones(int(eligible.sum()), device=self.device)
        )
        fly = eligible & airborne & upright
        if fly.any():
          self.flight_steps_by_speed.scatter_add_(
            0, bucket[fly], torch.ones(int(fly.sum()), device=self.device)
          )
    settled = self._bin_dwell >= self.attain_settle_s / self._step_dt
    m_attain = meaningful & ~self.push_cohort & settled
    if m_attain.any():
      env_idx = torch.arange(self.num_envs, device=self.device)[m_attain]
      self._attain_ep_sum.index_put_(
        (env_idx, bucket[m_attain]), attain[m_attain], accumulate=True
      )
      self._attain_ep_weight.index_put_(
        (env_idx, bucket[m_attain]),
        torch.ones(len(env_idx), device=self.device),
        accumulate=True,
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
      (self._attain_core_sum, self._attain_core_weight, self.attain_core_ema),
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
    self._attain_core_sum[ids] = 0.0
    self._attain_core_weight[ids] = 0.0
    self._wobble_sum[ids] = 0.0
    self._step_count[ids] = 0.0

    # Survivor conditioning: only timed-out episodes deposit their buffered
    # attainment into the capability-curve window.
    survivors = ids[fell < 0.5]
    if len(survivors) > 0:
      ep_w = self._attain_ep_weight[survivors]
      held = ep_w >= self.attain_min_dwell_s / self._step_dt
      self._attain_bin_sum += (self._attain_ep_sum[survivors] * held).sum(dim=0)
      self._attain_bin_weight += (ep_w * held).sum(dim=0)
    self._attain_ep_sum[ids] = 0.0
    self._attain_ep_weight[ids] = 0.0
    self._bin_dwell[ids] = 0.0

    self._win_fell += float(fell.sum().item())
    self._win_done += float(len(ids))

    # Cohort-stratified windows: attribution by membership, not by any
    # recovery-horizon guess.
    pushed = self.push_cohort[ids]
    self._win_fell_pushed += float(fell[pushed].sum().item())
    self._win_done_pushed += float(int(pushed.sum()))
    clean = ~pushed
    self._win_fell_clean += float(fell[clean].sum().item())
    self._win_done_clean += float(int(clean.sum()))

    # Settle pending pushes at episode end (censoring-aware):
    #   fall within the window  -> failure for the pending bin;
    #   fall beyond the window  -> the push already survived its window;
    #   timeout, window elapsed -> survival;
    #   timeout, window unfinished -> CENSORED (observed, not survived).
    window_steps = self.push_obs_window_s / self._step_dt
    pend = self._pending_push_mag[ids]
    pend_elapsed = env.common_step_counter - self._pending_push_step[ids]
    has_pend = pend >= 0
    if has_pend.any():
      pbins = (
        (pend[has_pend] / self.push_bin_width).long().clamp(0, self.n_push_bins - 1)
      )
      fell_p = fell[has_pend] > 0.5
      in_window = pend_elapsed[has_pend] < window_steps
      failure = fell_p & in_window
      survival = ~in_window  # window completed cleanly before episode end
      if failure.any():
        self._push_bin_fall.scatter_add_(
          0, pbins[failure], torch.ones(int(failure.sum()), device=self.device)
        )
      if survival.any():
        self._push_bin_survive.scatter_add_(
          0,
          pbins[survival],
          torch.ones(int(survival.sum()), device=self.device),
        )
      # fell & beyond-window -> survived; timed-out & in-window -> censored.
    self._pending_push_mag[ids] = -1.0
    self._pending_push_step[ids] = -1.0

    # Time-from-push-to-fall histogram (pushed cohort): answers "how long
    # does recovery take" empirically instead of assuming a horizon.
    # Attribution requires an actual push THIS episode (last_push_step is
    # cleared at episode end below — a stamp from a previous life is not a
    # push), and dts beyond the bin range are dropped, not clamped: a fall
    # 20+ s after the last push is a walking failure, not a recovery
    # outcome.
    fell_pushed = pushed & (fell > 0) & (self.last_push_step[ids] >= 0.0)
    if fell_pushed.any():
      dt_s = (
        env.common_step_counter - self.last_push_step[ids][fell_pushed]
      ) * self._step_dt
      in_range = dt_s < self.n_dt_bins * self.dt_bin_width
      if in_range.any():
        bins = (dt_s[in_range] / self.dt_bin_width).long()
        self._push_fall_dt_win.scatter_add_(
          0, bins, torch.ones(int(in_range.sum()), device=self.device)
        )
    # Pushes do not carry across resets: the next episode starts unpushed.
    self.last_push_step[ids] = -(10**9)

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
        self.bucket_exposure = 0.3 * self._bucket_steps + 0.7 * self.bucket_exposure
        self._bucket_steps.zero_()
        self._bucket_falls.zero_()
        rho_hazard = self._rho_falls / self._rho_steps.clamp(min=1.0)
        self.rho_hazard = 0.3 * rho_hazard + 0.7 * self.rho_hazard
        self.rho_exposure = 0.3 * self._rho_steps + 0.7 * self.rho_exposure
        self._rho_steps.zero_()
        self._rho_falls.zero_()
        if self.fast_fall_clean < 0.15:
          has_w = self._attain_bin_weight > 8.0
          window_attain = self._attain_bin_sum / self._attain_bin_weight.clamp(min=1e-6)
          self.attain_by_speed = torch.where(
            has_w,
            0.3 * window_attain + 0.7 * self.attain_by_speed,
            self.attain_by_speed,
          )
          self.attain_by_speed_weight = (
            0.3 * self._attain_bin_weight + 0.7 * self.attain_by_speed_weight
          )
        else:
          # Survivor-bias guard: during population decline the survivors
          # are the elite, so folding their windows RAISES the frontier
          # exactly when the truth is falling. While the clean fall rate is
          # unhealthy, bin confidence decays instead — the frontier retreats
          # with the population, never polls only the living.
          self.attain_by_speed_weight *= 0.5
        self._attain_bin_sum.zero_()
        self._attain_bin_weight.zero_()
        if float(self._push_fall_dt_win.sum()) >= 50.0:
          self.push_fall_dt_counts = (
            0.3 * self._push_fall_dt_win + 0.7 * self.push_fall_dt_counts
          )
          self._push_fall_dt_win.zero_()
          t75 = _binned_quantile(self.push_fall_dt_counts, self.dt_bin_width, 0.75)
          self.push_obs_window_s = min(max(t75, 2.0), 12.0)
        events = self._push_bin_survive + self._push_bin_fall
        has_ev = events > 8.0
        win_surv = self._push_bin_survive / events.clamp(min=1e-6)
        self.push_survival = torch.where(
          has_ev, 0.3 * win_surv + 0.7 * self.push_survival, self.push_survival
        )
        self.push_survival_weight = 0.3 * events + 0.7 * self.push_survival_weight
        self._push_bin_survive.zero_()
        self._push_bin_fall.zero_()
        self._bucket_next_step = env.common_step_counter + 50 * _NUM_STEPS_PER_ENV

  def population_means(self) -> dict[str, float]:
    return {
      "track_err_norm": self.track_err_ema.mean().item(),
      "fell_ema": self.fell_ema.mean().item(),
      "ep_len_frac": self.ep_len_frac_ema.mean().item(),
      "attain": self.attain_ema.mean().item(),
      "attain_core": self.attain_core_ema.mean().item(),
      "wobble": self.wobble_ema.mean().item(),
      "fast_fall_rate": self.fast_fall_rate,
    }

  def stratified_means(self) -> dict[str, float]:
    """Per-cohort aggregation of the per-env EMAs plus the fast splits.

    With cohort membership fixed by env index, stratification is pure
    aggregation — the accumulation machinery is untouched. Falls back to
    global values when a cohort is empty.
    """
    clean = ~self.push_cohort
    out: dict[str, float] = {}
    for name, ema in (
      ("attain", self.attain_ema),
      ("attain_core", self.attain_core_ema),
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
  """Fetch the env's tracker, creating it on first use.

  The tracker is stashed as a dynamic attribute on the env so the step
  event, the gating terms and the diagnostics all share one instance
  regardless of which of them the managers construct first.
  """
  tracker = getattr(env, "_competence_tracker", None)
  if tracker is None:
    tracker = CompetenceTracker(env)
    setattr(env, "_competence_tracker", tracker)  # noqa: B010
  return tracker


def competence_tracker_step(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  del env_ids
  tracker = getattr(env, "_competence_tracker", None)
  if tracker is not None:
    tracker.record_step(env)


def push_cohort_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  cohort_frac: float = PUSH_COHORT_FRAC,
) -> None:
  """Cohort-filtered push: only env indices < ``cohort_frac * num_envs``
  are pushed; the rest train push-free (deployment-matched) and serve as
  the uncontaminated baseline for tracking competence. Samples the shove
  itself (rather than delegating) so the per-event |dv_xy| magnitude is
  known for the survival frontier, and settles the PREVIOUS pending push
  for these envs as survived (no fall intervened — horizon-free event
  outcomes)."""
  tracker = get_competence_tracker(env)
  if tracker.push_cohort_frac != cohort_frac:
    tracker.set_push_cohort(cohort_frac)
  mask = tracker.push_cohort[env_ids]
  pushed_ids = env_ids[mask]
  if len(pushed_ids) == 0:
    return
  asset = env.scene["robot"]
  vel_w = asset.data.root_link_vel_w[pushed_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  delta = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  magnitudes = torch.norm(delta[:, :2], dim=-1)
  tracker.record_push(env, pushed_ids, magnitudes)
  asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=pushed_ids)


def _evidence_masked(
  values: torch.Tensor,
  evidence: torch.Tensor,
  floor: float = 1.0,
) -> torch.Tensor:
  """Zero out bins without evidence for the histogram views.

  The stratified buffers hold priors or noise where nothing was measured:
  ``push_survival`` keeps its optimistic 1.0 prior in every dv bin never
  shoved (rendered as a plateau of fake max-survival above the real data),
  and the hazard curves show falls/steps spikes from bins with a handful of
  steps (1 fall in 2 steps renders as hazard 0.5 next to well-sampled bins
  at ~1e-4). The scalar readouts already refuse to let such bins testify
  (``_interp_crossing`` exposure clamp, ``_attained_frontier`` min_weight);
  this applies the same rule to the histogram renderings. Zero means "no
  data here", and the paired evidence histograms disambiguate where needed.
  """
  return torch.where(evidence > floor, values, torch.zeros_like(values))


def _interp_crossing(
  hazards: torch.Tensor,
  bin_width: float,
  bar: float,
  exposure: torch.Tensor | None = None,
) -> float:
  """First up-crossing of the hazard bar, linearly interpolated.

  Light 3-bin smoothing first: quantiles/crossings integrate noise, but a
  single hot fine bin should not snap the frontier. Returns the
  bin-center-based crossing position; if never crossed, the readout is
  clamped to the highest bin with ``exposure`` evidence (when given):
  "clean as far as we have actually sampled". Without the clamp a near-zero
  fall rate renders the full instrument range as if it were a capability
  claim — unvisited bins hold hazard 0 and cannot testify.
  """
  n = len(hazards)
  sm = hazards.clone()
  if n >= 3:
    sm[1:-1] = (hazards[:-2] + hazards[1:-1] + hazards[2:]) / 3.0
  crossing = float(n * bin_width)
  prev_c = 0.5 * bin_width
  prev_h = float(sm[0])
  if prev_h > bar:
    crossing = prev_c
  else:
    for i in range(1, n):
      c = (i + 0.5) * bin_width
      h = float(sm[i])
      if h > bar:
        frac = (bar - prev_h) / max(h - prev_h, 1e-12)
        crossing = prev_c + frac * (c - prev_c)
        break
      prev_c, prev_h = c, h
  if exposure is not None:
    seen = torch.nonzero(exposure > 1.0)
    limit = 0.0 if len(seen) == 0 else (float(seen[-1]) + 0.5) * bin_width
    crossing = min(crossing, limit)
  return crossing


def _attained_frontier(
  attain_by_speed: torch.Tensor,
  weights: torch.Tensor,
  bin_width: float,
  bar: float,
  min_weight: float = 8.0,
  abs_tol: float = 0.08,
) -> float:
  """Highest interpolated speed at which the conditional attainment curve
  still clears the (graded) bar.

  The effective per-bin bar is ``min(bar, 1 - abs_tol/v)``: attainment is a
  FRACTIONAL measure, so a fixed absolute wobble of ~abs_tol m/s must not
  fail low-speed bins that a policy tracks as well as physics allows. Scans
  downward from the fastest bin with real exposure; returns 0 when no bin
  with data clears its bar."""

  def bin_bar(i: int) -> float:
    v = (i + 0.5) * bin_width
    return min(bar, 1.0 - abs_tol / max(v, 1e-6))

  n = len(attain_by_speed)
  hi = -1
  for i in range(n - 1, -1, -1):
    if float(weights[i]) >= min_weight:
      hi = i
      break
  if hi < 0:
    return 0.0
  for i in range(hi, -1, -1):
    if float(weights[i]) < min_weight:
      continue
    a = float(attain_by_speed[i])
    b = bin_bar(i)
    if a >= b:
      c = (i + 0.5) * bin_width
      if i < hi and float(weights[i + 1]) >= min_weight:
        a_next = float(attain_by_speed[i + 1])
        b_next = bin_bar(i + 1)
        if a_next < b_next and a - b > a_next - b_next:
          margin, margin_next = a - b, a_next - b_next
          frac = margin / max(margin - margin_next, 1e-12)
          return c + min(frac, 1.0) * bin_width
      return c
  return 0.0


def _binned_quantile(counts: torch.Tensor, bin_width: float, q: float) -> float:
  """Interpolated quantile of a binned distribution."""
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
  estimated frontier speed, push-to-fall timing histogram) alongside the
  competence-gated reward weights they explain.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._tracker = get_competence_tracker(env)
    self._last_hist_iter = -1
    if self._tracker.push_cohort_frac != PUSH_COHORT_FRAC:
      self._tracker.set_push_cohort(PUSH_COHORT_FRAC)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    self._tracker.finalize_episodes(env, env_ids)
    out = {k: torch.tensor(v) for k, v in self._tracker.stratified_means().items()}
    t = self._tracker
    speed_bw = t.speed_bin_width
    rho_bw = 1.6 / t.n_cmd_buckets
    # The statistics ARE the metrics; dense bins feed the interpolated
    # crossings/quantiles and the histogram views only.
    out["frontier_speed"] = torch.tensor(
      _interp_crossing(
        t.bucket_hazard,
        speed_bw,
        FRONTIER_HAZARD_BAR,
        exposure=t.bucket_exposure + t._bucket_steps,
      )
    )
    out["frontier_rho"] = torch.tensor(
      _interp_crossing(
        t.rho_hazard,
        rho_bw,
        FRONTIER_HAZARD_BAR,
        exposure=t.rho_exposure + t._rho_steps,
      )
    )
    out["attained_frontier"] = torch.tensor(
      _attained_frontier(
        t.attain_by_speed,
        t.attain_by_speed_weight,
        t.speed_bin_width,
        ATTAINED_FRONTIER_BAR,
      )
    )
    out["push_obs_window"] = torch.tensor(t.push_obs_window_s)
    out["push_fall_t50"] = torch.tensor(
      _binned_quantile(t.push_fall_dt_counts, t.dt_bin_width, 0.50)
    )
    out["push_fall_t75"] = torch.tensor(
      _binned_quantile(t.push_fall_dt_counts, t.dt_bin_width, 0.75)
    )
    # Per-env health bimodality: global degradation moves everyone together
    # (unimodal); per-env state corruption grows a BROKEN subpopulation
    # while the rest stay pristine (bimodal).
    out["envs_broken_frac"] = (t.fell_ema > 0.9).float().mean().cpu()
    out["envs_healthy_frac"] = (t.fell_ema < 0.1).float().mean().cpu()
    self._log_histograms(env)
    return out

  def _log_histograms(self, env: ManagerBasedRlEnv) -> None:
    """W&B heatmap views of the bucketed profiles: the per-bucket scalars
    stay (monitor triggers and API pulls read them); these are the
    human-readable evolution-over-time renderings. Logged directly with
    ``commit=False`` so they attach to the logger's next committed step;
    inherently rank-0-only (only that process owns a W&B run). Cadence
    matches the hazard refresh (~50 iters).
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
      # Evidence-masked views: bins without measurements render as zero
      # instead of leaking priors (push_survival's 1.0) or falls/steps
      # noise spikes from near-empty bins (see _evidence_masked).
      hazard_speed = _evidence_masked(
        t.bucket_hazard, t.bucket_exposure + t._bucket_steps
      )
      hazard_rho = _evidence_masked(t.rho_hazard, t.rho_exposure + t._rho_steps)
      attain_speed = _evidence_masked(t.attain_by_speed, t.attain_by_speed_weight)
      flight_speed = _evidence_masked(
        t.flight_steps_by_speed / t.flight_exposure_by_speed.clamp(min=1.0),
        t.flight_exposure_by_speed,
        floor=200.0,  # fractions from <200 frames are noise, render 0
      )
      push_surv = _evidence_masked(t.push_survival, t.push_survival_weight)
      wandb.log(
        {
          "frontier/hazard_by_speed": wandb.Histogram(
            np_histogram=(
              hazard_speed.cpu().numpy(),
              np.linspace(0.0, t.speed_bin_width * n, n + 1),
            )
          ),
          "frontier/attain_by_speed": wandb.Histogram(
            np_histogram=(
              attain_speed.cpu().numpy(),
              np.linspace(0.0, t.speed_bin_width * n, n + 1),
            )
          ),
          "frontier/flight_by_speed": wandb.Histogram(
            np_histogram=(
              flight_speed.cpu().numpy(),
              np.linspace(0.0, t.speed_bin_width * n, n + 1),
            )
          ),
          "frontier/hazard_by_rho": wandb.Histogram(
            np_histogram=(
              hazard_rho.cpu().numpy(),
              np.linspace(0.0, 1.6, n + 1),
            )
          ),
          "frontier/push_fall_dt": wandb.Histogram(
            np_histogram=(
              t.push_fall_dt_counts.cpu().numpy(),
              np.linspace(0.0, t.n_dt_bins * t.dt_bin_width, t.n_dt_bins + 1),
            )
          ),
          # Survival frontier: survival rate per shove |dv_xy| bin plus its
          # evidence weight. Only the rendering is masked (a wall of prior
          # 1.0s above the delivered dv range reads as fake demonstrated
          # competence).
          "frontier/push_survival_by_dv": wandb.Histogram(
            np_histogram=(
              push_surv.cpu().numpy(),
              np.linspace(0.0, t.n_push_bins * t.push_bin_width, t.n_push_bins + 1),
            )
          ),
          "frontier/push_events_by_dv": wandb.Histogram(
            np_histogram=(
              t.push_survival_weight.cpu().numpy(),
              np.linspace(0.0, t.n_push_bins * t.push_bin_width, t.n_push_bins + 1),
            )
          ),
        },
        commit=False,
      )
    except Exception:
      # Visualization must never take down training.
      return


class staged_on_competence:
  """Ramp penalty weights when stability competence holds; back off on loss.

  Promote (next stage) requires stability competence AND the cooldown.
  Demote (previous stage) fires when stability is badly lost, also
  cooldown-limited so a single demotion gets time to take effect before the
  next. A pure freeze cannot recover a policy that is already sliding down
  the penalty gradient — releasing penalty pressure restores the basin the
  gait was learned in and gives the policy a path back. Between the demote
  and re-promote thresholds lies the freeze band.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    self._tracker = get_competence_tracker(env)
    self._controller = CompetenceController(
      l_max=len(stages) - 1, thresholds=CompetenceThresholds()
    )
    self._stage_idx = 0
    self._last_change_iter = -self._controller.thresholds.cooldown_iters
    _validate_stages(self._term_cfg, reward_name, self._stages)
    _apply_stages(self._term_cfg, self._stages[0]["step"], self._stages)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    # The curriculum manager passes all cfg params as kwargs
    # (``func(env, env_ids, **term_cfg.params)``), so this signature must
    # accept them even though they are consumed in __init__.
    del reward_name, stages
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    cooldown = self._controller.thresholds.cooldown_iters
    cooldown_elapsed = cur_iter - self._last_change_iter >= cooldown
    stable = self._controller.stability_ok(
      fell_ema=stats["fell_ema"],
      ep_len_frac=stats["ep_len_frac"],
      attain_ema=stats["attain_core"],
    )
    t = self._controller.thresholds
    badly_lost = (
      stats["fell_ema"] > t.demote_fell
      or stats["attain_core"] < t.demote_attain
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
    snapshot["competence_attain_core"] = torch.tensor(stats["attain_core"])
    return snapshot
