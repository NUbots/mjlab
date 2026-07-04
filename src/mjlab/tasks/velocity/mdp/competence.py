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
    if (
      self.promote_streak >= self.thresholds.promote_streak_required
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

  def state_dict(self) -> dict[str, Any]:
    return {
      "track_err_ema": self.track_err_ema.cpu(),
      "fell_ema": self.fell_ema.cpu(),
      "ep_len_frac_ema": self.ep_len_frac_ema.cpu(),
      "attain_ema": self.attain_ema.cpu(),
      "wobble_ema": self.wobble_ema.cpu(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.track_err_ema = state["track_err_ema"].to(self.device)
    self.fell_ema = state["fell_ema"].to(self.device)
    self.ep_len_frac_ema = state["ep_len_frac_ema"].to(self.device)
    if "attain_ema" in state:
      self.attain_ema = state["attain_ema"].to(self.device)
      self.wobble_ema = state["wobble_ema"].to(self.device)

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

  def population_means(self) -> dict[str, float]:
    return {
      "track_err_norm": self.track_err_ema.mean().item(),
      "fell_ema": self.fell_ema.mean().item(),
      "ep_len_frac": self.ep_len_frac_ema.mean().item(),
      "attain": self.attain_ema.mean().item(),
      "wobble": self.wobble_ema.mean().item(),
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
      )
    snapshot = _apply_push_level(env, self._event_name, self._controller.level)
    snapshot.update({f"competence_{k}": torch.tensor(v) for k, v in stats.items()})
    return snapshot


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
