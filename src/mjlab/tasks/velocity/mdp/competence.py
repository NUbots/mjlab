"""Edge-of-competence curriculum: tracker, controller, and adaptive terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

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
    self._wobble_sum = torch.zeros(n, device=self.device)
    self._step_count = torch.zeros(n, device=self.device)
    # Pessimistic init: a fresh policy must EARN competence. Zero-init would
    # read as "perfectly competent" and allow spurious promotions in the
    # first iterations before the EMAs warm up.
    self.track_err_ema = torch.ones(n, device=self.device)
    self.fell_ema = torch.ones(n, device=self.device)
    self.ep_len_frac_ema = torch.zeros(n, device=self.device)
    self.attain_ema = torch.zeros(n, device=self.device)
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
    command_term = env.command_manager.get_term("twist")
    if command_term is None:
      return
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
    self._wobble_sum[ids] = 0.0
    self._step_count[ids] = 0.0

    self._win_fell += float(fell.sum().item())
    self._win_done += float(len(ids))
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

  def population_means(self) -> dict[str, float]:
    return {
      "track_err_norm": self.track_err_ema.mean().item(),
      "fell_ema": self.fell_ema.mean().item(),
      "ep_len_frac": self.ep_len_frac_ema.mean().item(),
      "attain": self.attain_ema.mean().item(),
      "wobble": self.wobble_ema.mean().item(),
      "fast_fall_rate": self.fast_fall_rate,
    }


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
      if in_refractory:
        return None
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
  """Continuous TCP-style difficulty: one scalar drives commands + pushes."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._event_name: str = cfg.params["event_name"]
    self._params = AimdParams(
      alpha=cfg.params.get("alpha", 0.002),
      beta=cfg.params.get("beta", 0.7),
      congest_bar=cfg.params.get("congest_bar", 0.35),
      emergency_bar=cfg.params.get("emergency_bar", 0.55),
      gate_attain=cfg.params.get("gate_attain", 0.40),
    )
    self._controller = AimdController(self._params)
    self._tracker = get_competence_tracker(env)
    self._last_check_iter = -1
    self._apply(env, 0.0)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def _apply(self, env: ManagerBasedRlEnv, d: float) -> dict[str, torch.Tensor]:
    lo, hi = COMMAND_LEVEL_TABLE[0], COMMAND_LEVEL_TABLE[-1]
    command_term = env.command_manager.get_term(self._command_name)
    assert command_term is not None
    ccfg = command_term.cfg
    assert isinstance(ccfg, UniformVelocityCommandCfg)
    ccfg.ranges.lin_vel_x = _lerp_range(lo["lin_vel_x"], hi["lin_vel_x"], d)
    ccfg.ranges.lin_vel_y = _lerp_range(lo["lin_vel_y"], hi["lin_vel_y"], d)
    ccfg.ranges.ang_vel_z = _lerp_range(lo["ang_vel_z"], hi["ang_vel_z"], d)

    push_scale = (
      PUSH_LEVEL_SCALES[0] + (PUSH_LEVEL_SCALES[-1] - PUSH_LEVEL_SCALES[0]) * d
    )
    ecfg = env.event_manager.get_term_cfg(self._event_name)
    ecfg.params["velocity_range"] = _scale_push_velocity_range(push_scale)

    return {
      "difficulty": torch.tensor(d),
      "ssthresh": torch.tensor(self._controller.ssthresh),
      "lin_vel_x_max": torch.tensor(ccfg.ranges.lin_vel_x[1]),
      "ang_vel_z_max": torch.tensor(ccfg.ranges.ang_vel_z[1]),
      "push_scale": torch.tensor(push_scale),
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
  ) -> dict[str, torch.Tensor]:
    del (command_name, event_name, alpha, beta, congest_bar, emergency_bar)
    del gate_attain
    self._tracker.finalize_episodes(env, env_ids)
    stats = self._tracker.population_means()
    cur_iter = env.common_step_counter // _NUM_STEPS_PER_ENV
    if cur_iter != self._last_check_iter:
      self._last_check_iter = cur_iter
      self._controller.update(
        cur_iter=cur_iter,
        fast_fall_rate=stats["fast_fall_rate"],
        wobble_ema=stats["wobble"],
        attain_ema=stats["attain"],
      )
    snapshot = self._apply(env, self._controller.d)
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
