"""Tests for edge-of-competence curriculum (doc 13)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg
from mjlab.tasks.velocity.mdp.competence import (
  COMMAND_LEVEL_TABLE,
  PUSH_LEVEL_SCALES,
  CompetenceController,
  CompetenceThresholds,
  CompetenceTracker,
  _apply_command_level,
  _apply_push_level,
  _scale_push_velocity_range,
  aimd_difficulty,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

_NUM_STEPS_PER_ENV = 24


def _thresholds(**overrides: float | int) -> CompetenceThresholds:
  base = CompetenceThresholds(cooldown_iters=2, promote_streak_required=2)
  for key, value in overrides.items():
    setattr(base, key, value)
  return base


def test_controller_promotes_after_cooldown_and_streak() -> None:
  ctrl = CompetenceController(l_max=3, thresholds=_thresholds())
  step = 2 * _NUM_STEPS_PER_ENV
  assert (
    ctrl.update(
      track_err_norm=0.1,
      fell_ema=0.1,
      ep_len_frac=0.9,
      common_step_counter=step,
      attain_ema=0.9,
      wobble_ema=0.01,
    )
    is None
  )
  step += 2 * _NUM_STEPS_PER_ENV
  assert (
    ctrl.update(
      track_err_norm=0.1,
      fell_ema=0.1,
      ep_len_frac=0.9,
      common_step_counter=step,
      attain_ema=0.9,
      wobble_ema=0.01,
    )
    == "promote"
  )
  assert ctrl.level == 1


def test_controller_demotes_immediately() -> None:
  ctrl = CompetenceController(l_max=3, thresholds=_thresholds())
  ctrl.level = 2
  assert (
    ctrl.update(
      track_err_norm=0.5,
      fell_ema=0.1,
      ep_len_frac=0.9,
      common_step_counter=100 * _NUM_STEPS_PER_ENV,
      attain_ema=0.3,
      wobble_ema=0.01,
    )
    == "demote"
  )
  assert ctrl.level == 1


def test_controller_hysteresis_holds_level_in_band() -> None:
  ctrl = CompetenceController(l_max=3, thresholds=_thresholds())
  ctrl.level = 1
  assert (
    ctrl.update(
      track_err_norm=0.30,
      fell_ema=0.32,
      ep_len_frac=0.9,
      common_step_counter=50 * _NUM_STEPS_PER_ENV,
      attain_ema=0.65,
      wobble_ema=0.08,
    )
    is None
  )
  assert ctrl.level == 1


def test_controller_state_round_trip() -> None:
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 3
  ctrl.last_change_iter = 42
  ctrl.promote_streak = 1
  ctrl.frozen = True
  restored = CompetenceController(l_max=5, thresholds=_thresholds())
  restored.load_state_dict(ctrl.state_dict())
  assert restored.state_dict() == ctrl.state_dict()


def test_command_levels_match_doc13_table() -> None:
  assert COMMAND_LEVEL_TABLE[3]["lin_vel_x"] == (-0.50, 0.50)
  assert COMMAND_LEVEL_TABLE[5]["ang_vel_z"] == (-0.80, 0.80)


def test_apply_command_level_sets_ranges() -> None:
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

  ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    ranges=ranges,
  )
  env = MagicMock()
  term = MagicMock()
  term.cfg = cfg
  env.command_manager.get_term.return_value = term
  _apply_command_level(env, "twist", 3)
  assert ranges.lin_vel_x == (-0.50, 0.50)
  assert ranges.lin_vel_y == (-0.30, 0.30)


def test_push_level_scales_match_base_range() -> None:
  base = _scale_push_velocity_range(1.0)
  scaled = _scale_push_velocity_range(PUSH_LEVEL_SCALES[3])
  assert scaled["x"] == pytest.approx((base["x"][0] * 1.5, base["x"][1] * 1.5))


def test_apply_push_level_updates_event_params() -> None:
  env = MagicMock()
  term_cfg = MagicMock()
  term_cfg.params = {"velocity_range": _scale_push_velocity_range(1.0)}
  env.event_manager.get_term_cfg.return_value = term_cfg
  _apply_push_level(env, "push_robot", 5)
  assert term_cfg.params["velocity_range"]["x"] == pytest.approx((-0.4, 0.8))


def test_adaptive_command_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ADAPTIVE_COMMANDS", "1")
  monkeypatch.setenv("ADAPTIVE_CMD_LMAX", "3")
  cfg = nubots_nugus_flat_env_cfg()
  assert "adaptive_command_level" in cfg.curriculum
  assert "command_vel" not in cfg.curriculum
  assert "competence_tracker" in cfg.events


def test_adaptive_full_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ADAPTIVE_COMMANDS", "1")
  monkeypatch.setenv("ADAPTIVE_PUSHES", "1")
  monkeypatch.setenv("PENALTY_GATE", "competence")
  cfg = nubots_nugus_flat_env_cfg()
  assert "adaptive_push_level" in cfg.curriculum
  assert "joule_heating_competence" in cfg.curriculum
  assert "joint_acc_l2_competence" in cfg.curriculum
  assert "joule_heating_warmup" not in cfg.curriculum


def test_v20_control_keeps_time_penalty_warmup(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("PHASE_C_WARMUP", "1")
  cfg = nubots_nugus_flat_env_cfg()
  assert "joule_heating_warmup" in cfg.curriculum
  assert "adaptive_command_level" not in cfg.curriculum


def test_tracker_excludes_standing_from_track_err() -> None:
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = 2
  env.episode_length_buf = torch.tensor([100, 100])
  env.max_episode_length = 1000
  env.common_step_counter = 24
  env.termination_manager.active_terms = ["fell_over"]
  env.termination_manager.get_term.return_value = torch.tensor([False, True])

  command_term = MagicMock()
  command_term.is_standing_env = torch.tensor([True, False])
  command_term.vel_command_b = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
  command_term.robot.data.root_link_lin_vel_b = torch.tensor([[0.0, 0.0], [0.4, 0.0]])
  command_term.robot.data.projected_gravity_b = torch.tensor(
    [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
  )
  command_term.cfg.ranges.lin_vel_x = (-0.75, 0.75)
  command_term.cfg.ranges.lin_vel_y = (-0.45, 0.45)
  command_term.cfg.ranges.ang_vel_z = (-0.80, 0.80)
  env.command_manager.get_term.return_value = command_term

  tracker = CompetenceTracker(env)
  tracker.record_step(env)
  assert tracker._track_weight[0].item() == 0.0
  assert tracker._track_weight[1].item() == 1.0
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.fell_ema[1].item() > tracker.fell_ema[0].item()


def _mock_tracked_env(
  *, fell: bool, ep_len: int = 950, step_counter: int = 24
) -> MagicMock:
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = 2
  env.episode_length_buf = torch.tensor([ep_len, ep_len])
  env.max_episode_length = 1000
  env.common_step_counter = step_counter
  env.termination_manager.active_terms = ["fell_over"]
  env.termination_manager.get_term.return_value = torch.tensor([fell, fell])
  command_term = MagicMock()
  command_term.is_standing_env = torch.tensor([False, False])
  command_term.vel_command_b = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
  command_term.robot.data.root_link_lin_vel_b = torch.tensor([[0.45, 0.0], [0.45, 0.0]])
  command_term.robot.data.projected_gravity_b = torch.tensor(
    [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
  )
  command_term.cfg.ranges.lin_vel_x = (-0.75, 0.75)
  command_term.cfg.ranges.lin_vel_y = (-0.45, 0.45)
  command_term.cfg.ranges.ang_vel_z = (-0.80, 0.80)
  env.command_manager.get_term.return_value = command_term
  return env


def test_tracker_pessimistic_init() -> None:
  """A fresh tracker must read as incompetent, not perfect (doc 14)."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  stats = tracker.population_means()
  assert stats["fell_ema"] == pytest.approx(1.0)
  assert stats["track_err_norm"] == pytest.approx(1.0)
  assert stats["ep_len_frac"] == pytest.approx(0.0)


def test_tracker_finalize_all_weightless_envs() -> None:
  """Resetting only standing envs (zero track weight) must not crash."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  # No record_step calls at all -> both envs weightless.
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  # track_err EMA unchanged from init; fell/ep_len EMAs still update.
  assert tracker.track_err_ema[0].item() == pytest.approx(1.0)
  assert tracker.fell_ema[0].item() == pytest.approx(0.9)


def test_curriculum_terms_accept_manager_kwargs() -> None:
  """The curriculum manager calls ``func(env, env_ids, **params)``; every
  key wired in env_cfgs must be accepted by the term's __call__ signature."""
  import inspect

  from mjlab.tasks.velocity.config.nugus.env_cfgs import (
    _competence_threshold_params,
  )
  from mjlab.tasks.velocity.mdp.competence import (
    adaptive_command_level,
    adaptive_push_level,
    staged_on_competence,
  )

  threshold_keys = set(_competence_threshold_params())
  for cls, extra in (
    (adaptive_command_level, {"command_name", "l_max"}),
    (adaptive_push_level, {"event_name", "l_max", "start_level"}),
    (staged_on_competence, {"reward_name", "stages"}),
  ):
    sig_params = set(inspect.signature(cls.__call__).parameters)
    missing = (threshold_keys | extra) - sig_params
    assert not missing, f"{cls.__name__}.__call__ missing params: {missing}"


def test_staged_on_competence_demotes_on_instability() -> None:
  """Penalty gate must BACK OFF a stage when stability is badly lost
  (disease #2, doc 14) - a freeze alone cannot recover a sliding policy."""
  from mjlab.tasks.velocity.mdp.competence import (
    get_competence_tracker,
    staged_on_competence,
  )

  stages = [{"step": i, "weight": -0.001 * i} for i in range(5)]
  env = _mock_tracked_env(fell=False)
  # MagicMock auto-creates attributes, so get_competence_tracker(env) would
  # return a mock tracker; install a real one explicitly.
  env._competence_tracker = CompetenceTracker(env)
  term_cfg_mock = MagicMock()
  env.reward_manager.get_term_cfg.return_value = term_cfg_mock

  cfg = MagicMock()
  cfg.params = {
    "reward_name": "torque_rate",
    "stages": stages,
    "cooldown_iters": 1,
    "demote_fell": 1.0,
  }
  term = staged_on_competence(cfg, env)
  tracker = get_competence_tracker(env)

  # Drive to stage 2 via two stable windows. The fast fall channel starts
  # at its pessimistic init (1.0) and must be marked healthy like the slow
  # EMAs or it vetoes every promotion.
  tracker.fast_fall_rate = 0.05
  for it in (10, 20):
    tracker.fell_ema[:] = 0.1
    tracker.ep_len_frac_ema[:] = 0.95
    tracker.attain_ema[:] = 0.9
    tracker.wobble_ema[:] = 0.02
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    snap = term(env, torch.tensor([0, 1]), "torque_rate", stages)
  assert snap["stage_idx"].item() == 2

  # Stability badly lost -> demote one stage. Falls stay LOW — the loss
  # shows up as sandbagging (attainment collapse), per the user's
  # observation that the policy prefers a stable stand to risking falls.
  tracker.fell_ema[:] = 0.2
  tracker.attain_ema[:] = 0.25
  tracker.ep_len_frac_ema[:] = 0.9
  env.common_step_counter = 40 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  snap = term(env, torch.tensor([0, 1]), "torque_rate", stages)
  assert snap["stage_idx"].item() == 1

  # Freeze band (between thresholds): holds, no further demote.
  tracker.fell_ema[:] = 0.2
  tracker.attain_ema[:] = 0.55
  tracker.wobble_ema[:] = 0.08
  env.common_step_counter = 80 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  snap = term(env, torch.tensor([0, 1]), "torque_rate", stages)
  assert snap["stage_idx"].item() == 1


def test_controller_sandbag_demotes_without_falls() -> None:
  """Conservative retreat (high ep_len, low falls, near-zero attainment)
  must DEMOTE — falls are the trailing indicator (user observation
  2026-07-04: the policy prefers a stable stand over risking a fall, so
  fall-gated demotion never fires until terminal collapse)."""
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 4
  result = ctrl.update(
    track_err_norm=1.0,
    fell_ema=0.05,
    ep_len_frac=0.98,
    common_step_counter=200 * _NUM_STEPS_PER_ENV,
    attain_ema=0.2,
    wobble_ema=0.02,
  )
  assert result == "demote"
  assert ctrl.level == 3


def test_controller_wobble_demotes_before_falls() -> None:
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 2
  result = ctrl.update(
    track_err_norm=0.3,
    fell_ema=0.1,
    ep_len_frac=0.9,
    common_step_counter=300 * _NUM_STEPS_PER_ENV,
    attain_ema=0.9,
    wobble_ema=0.4,
  )
  assert result == "demote"


def test_attainment_true_fraction_at_small_commands() -> None:
  """Perfect tracking of a 0.18 m/s command must read attain ~1.0 — the
  floor-squared form capped it at |c|^2/0.04 and froze level-0 promotion
  (v20 pair 3: attain 0.17-0.25 while every other gate passed)."""
  env = _mock_tracked_env(fell=False)
  ct = env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.18, 0.0, 0.0], [0.18, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.18, 0.0], [0.18, 0.0]])
  tracker = CompetenceTracker(env)
  for _ in range(10):
    tracker.record_step(env)
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  # EMA from pessimistic 0 with alpha 0.1 and a perfect episode -> 0.1.
  assert tracker.attain_ema[0].item() == pytest.approx(0.1, abs=0.01)
  # Commands below the 0.15 filter contribute no attainment weight.
  ct.vel_command_b = torch.tensor([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
  tracker2 = CompetenceTracker(env)
  tracker2.record_step(env)
  assert tracker2._attain_weight.sum().item() == 0.0


def test_fast_fall_rate_demotes_despite_stale_slow_emas() -> None:
  """v23 postmortem: a top-rung crash must demote on the fast channel alone.

  During the v23 collapse the slow per-env EMAs looked healthy for ~200
  iterations (long episodes -> rare episode-end updates) while the policy
  shattered under -10 fall gradients. The windowed fall rate must cascade
  the level down even when every slow EMA still reads healthy.
  """
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 5
  step = 1000 * _NUM_STEPS_PER_ENV
  for expected_level in (4, 3, 2, 1, 0):
    assert (
      ctrl.update(
        track_err_norm=0.1,
        fell_ema=0.1,
        ep_len_frac=0.9,
        common_step_counter=step,
        attain_ema=0.9,
        wobble_ema=0.01,
        fast_fall_rate=0.8,
      )
      == "demote"
    )
    assert ctrl.level == expected_level
    step += _NUM_STEPS_PER_ENV


def test_fast_fall_rate_clears_healthy_top_rung_band() -> None:
  """Healthy L5 operation measured ~0.26 falls/episode-end (v23 iter 1815);
  the 0.5 bar must not demote a working hard-envelope policy."""
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 5
  result = ctrl.update(
    track_err_norm=0.1,
    fell_ema=0.28,
    ep_len_frac=0.85,
    common_step_counter=1000 * _NUM_STEPS_PER_ENV,
    attain_ema=0.7,
    wobble_ema=0.08,
    fast_fall_rate=0.30,
  )
  assert result is None
  assert ctrl.level == 5


def test_top_rung_requires_longer_streak() -> None:
  """Promotion INTO levels >= top_level_start needs the longer streak."""
  th = _thresholds(top_streak_required=4, top_level_start=3)
  ctrl = CompetenceController(l_max=5, thresholds=th)
  ctrl.level = 2
  step = 1000 * _NUM_STEPS_PER_ENV

  def healthy_update() -> str | None:
    nonlocal step
    step += 2 * _NUM_STEPS_PER_ENV
    return ctrl.update(
      track_err_norm=0.1,
      fell_ema=0.1,
      ep_len_frac=0.9,
      common_step_counter=step,
      attain_ema=0.9,
      wobble_ema=0.01,
    )

  # Base streak (2) is NOT enough to enter level 3 (top rung starts there).
  assert healthy_update() is None
  assert healthy_update() is None
  assert ctrl.level == 2
  # The longer streak (4) is.
  assert healthy_update() is None
  assert healthy_update() == "promote"
  assert ctrl.level == 3


def test_tracker_fast_window_tracks_crash() -> None:
  """The fast rate must fall during health and alarm within ~10 iterations
  of a total crash (vs ~200 for the per-env EMAs)."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  assert tracker.population_means()["fast_fall_rate"] == pytest.approx(1.0)

  step = 24
  for _ in range(30):
    env.common_step_counter = step
    tracker.finalize_episodes(env, torch.tensor([0, 1]))
    step += 24
  healthy = tracker.population_means()["fast_fall_rate"]
  assert healthy < 0.05

  crash_env = _mock_tracked_env(fell=True, ep_len=200)
  for _ in range(10):
    crash_env.common_step_counter = step
    tracker.finalize_episodes(crash_env, torch.tensor([0, 1]))
    step += 24
  assert tracker.population_means()["fast_fall_rate"] > 0.5


def test_aimd_rises_at_alpha_and_gates_on_stress() -> None:
  """Healthy iterations add exactly alpha; any unhealthy gate freezes d."""
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  for i in range(50):
    c.update(cur_iter=i, fast_fall_rate=0.05, wobble_ema=0.02, attain_ema=0.6)
  assert c.d == pytest.approx(50 * 0.002)
  # Stressed-but-not-congested: hold, don't cut.
  d = c.d
  assert (
    c.update(cur_iter=50, fast_fall_rate=0.28, wobble_ema=0.02, attain_ema=0.6) is None
  )
  assert c.d == d
  # Attainment gate below bar: hold (rate gate, not a promote event).
  assert (
    c.update(cur_iter=51, fast_fall_rate=0.05, wobble_ema=0.02, attain_ema=0.3) is None
  )
  assert c.d == d


def test_aimd_congestion_cut_refractory_and_backoff() -> None:
  """One event = one 0.7x cut; repeats inside the window double refractory."""
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  c.d = 0.8
  assert (
    c.update(cur_iter=1000, fast_fall_rate=0.4, wobble_ema=0.1, attain_ema=0.5) == "cut"
  )
  assert c.d == pytest.approx(0.8 * 0.7)
  assert c.ssthresh == pytest.approx(0.8 * 0.85)
  # Inside the refractory a sustained signal keeps decaying d (arrest
  # mode, v27 postmortem) without event bookkeeping.
  d_before = c.d
  assert (
    c.update(cur_iter=1005, fast_fall_rate=0.4, wobble_ema=0.1, attain_ema=0.5)
    == "arrest"
  )
  assert c.d == pytest.approx(d_before * 0.93)
  assert c.ssthresh == pytest.approx(0.8 * 0.85)
  # After refractory, a repeat congestion doubles the refractory (backoff).
  assert (
    c.update(cur_iter=1016, fast_fall_rate=0.4, wobble_ema=0.1, attain_ema=0.5) == "cut"
  )
  assert c.refractory == 30


def test_aimd_emergency_cuts_harder() -> None:
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  c.d = 0.8
  c.update(cur_iter=500, fast_fall_rate=0.6, wobble_ema=0.3, attain_ema=0.1)
  assert c.d == pytest.approx(0.8 * 0.5)


def test_aimd_slow_probe_above_ssthresh() -> None:
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  c.d = 0.6
  c.ssthresh = 0.5
  c.update(cur_iter=2000, fast_fall_rate=0.05, wobble_ema=0.02, attain_ema=0.6)
  assert c.d == pytest.approx(0.6 + 0.002 / 3.0)


def test_aimd_lerp_endpoints_match_tables() -> None:
  """d=0 must reproduce level 0; d=1 must reproduce the full envelope."""
  from mjlab.tasks.velocity.mdp.competence import _lerp_range

  lo, hi = COMMAND_LEVEL_TABLE[0], COMMAND_LEVEL_TABLE[-1]
  assert _lerp_range(lo["lin_vel_x"], hi["lin_vel_x"], 0.0) == lo["lin_vel_x"]
  assert _lerp_range(lo["lin_vel_x"], hi["lin_vel_x"], 1.0) == hi["lin_vel_x"]
  mid = _lerp_range(lo["lin_vel_x"], hi["lin_vel_x"], 0.5)
  assert mid == pytest.approx((-0.475, 0.475))


def test_watchdog_arms_and_fires_on_sustained_rot() -> None:
  """Arms above 2.0; fires only after the persistence bar, so v25-push
  style bounce transients (<=45 iters) never kill a healthy run."""
  from mjlab.tasks.velocity.mdp.competence import track_reward_watchdog

  cfg = MagicMock()
  cfg.params = {"fail_persist_iters": 60, "ema_alpha": 1.0}
  wd = track_reward_watchdog(cfg, MagicMock())

  def step(env_step: int, value: float) -> None:
    env = MagicMock()
    env.max_episode_length_s = 20.0
    env.common_step_counter = env_step * _NUM_STEPS_PER_ENV
    env.reward_manager._episode_sums = {
      "track_linear_velocity": torch.tensor([value * 20.0, value * 20.0])
    }
    wd(env, torch.tensor([0, 1]))

  step(1, 2.5)  # arms
  assert wd._armed
  # Transient dip (40 iters below 1.0): no fire.
  for i in range(2, 42):
    step(i, 0.8)
  # Recovery resets the counter.
  step(42, 1.5)
  assert wd._below_count == 0
  # Sustained rot: fires at the 60th consecutive iteration below 1.0.
  with pytest.raises(RuntimeError, match="failing fast"):
    for i in range(43, 43 + 61):
      step(i, 0.6)


def test_aimd_and_watchdog_accept_manager_kwargs() -> None:
  import inspect

  from mjlab.tasks.velocity.mdp.competence import (
    aimd_difficulty,
    track_reward_watchdog,
  )

  aimd_params = {
    "command_name",
    "event_name",
    "alpha",
    "beta",
    "congest_bar",
    "emergency_bar",
    "gate_attain",
  }
  sig = set(inspect.signature(aimd_difficulty.__call__).parameters)
  assert not (aimd_params - sig)
  wd_params = {
    "reward_name",
    "arm_above",
    "fail_below",
    "fail_persist_iters",
  }
  sig = set(inspect.signature(track_reward_watchdog.__call__).parameters)
  assert not (wd_params - sig)


def test_cohort_stratified_attribution() -> None:
  """Falls in the pushed cohort must not contaminate clean-cohort stats."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.5)  # env 0 pushed, env 1 clean
  assert tracker.push_cohort.tolist() == [True, False]

  # Env 0 falls (pushed), env 1 healthy: window splits must diverge.
  env.termination_manager.get_term.return_value = torch.tensor([True, False])
  step = 24
  for _ in range(60):
    env.common_step_counter = step
    tracker.finalize_episodes(env, torch.tensor([0, 1]))
    step += 24
  strat = tracker.stratified_means()
  assert strat["fast_fall_pushed"] > 0.9
  assert strat["fast_fall_clean"] < 0.1
  assert strat["push_excess_fall"] > 0.8
  # Per-env EMA stratification follows membership.
  assert strat["pushed_fell_ema"] > 0.9
  assert strat["clean_fell_ema"] < 0.1


def test_push_fall_dt_histogram() -> None:
  """Push-to-fall timing lands in the right bin (recovery-time question)."""
  env = _mock_tracked_env(fell=True)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  env.common_step_counter = 1000
  tracker.record_push(env, torch.tensor([0, 1]))
  # Fall 0.72 s later (36 steps at dt=0.02): second bin (0.5-1.0 s).
  env.common_step_counter = 1036
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  counts = tracker._push_fall_dt_win
  assert counts[1].item() == pytest.approx(2.0)
  assert counts.sum().item() == pytest.approx(2.0)


def test_push_fall_dt_window_folds_into_ema() -> None:
  """The dt histogram is windowed like the survival bins: a window with
  >= 50 events folds 0.3/0.7 into the EMA and clears; sparse windows
  keep accumulating. The EMA tracks the CURRENT policy's recovery
  profile instead of letting early-training falls weigh forever."""
  env = _mock_tracked_env(fell=True)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  tracker._push_fall_dt_win[2] = 60.0  # 60 falls at ~1-1.5 s (early flail)
  env.common_step_counter = 50 * _NUM_STEPS_PER_ENV
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0  # satisfy the refresh precondition
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.push_fall_dt_counts[2].item() == pytest.approx(18.0)  # 0.3*60
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(0.0)
  # A sparse window (< 50 events) neither folds nor discards.
  tracker._push_fall_dt_win[8] = 10.0
  env.common_step_counter += 50 * _NUM_STEPS_PER_ENV
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0
  tracker._finalized_step = -1
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.push_fall_dt_counts[8].item() == pytest.approx(0.0)
  assert tracker._push_fall_dt_win[8].item() == pytest.approx(10.0)
  # Once the window reaches 50, the fold shifts the EMA toward the new
  # shape: recovery now takes ~4-4.5 s, and the old 1-1.5 s mode decays.
  tracker._push_fall_dt_win[8] = 50.0
  env.common_step_counter += 50 * _NUM_STEPS_PER_ENV
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0
  tracker._finalized_step = -1
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.push_fall_dt_counts[8].item() == pytest.approx(15.0)  # 0.3*50
  assert tracker.push_fall_dt_counts[2].item() == pytest.approx(12.6)  # 0.7*18


def test_push_fall_dt_requires_push_this_episode() -> None:
  """Falls with no push in the CURRENT episode never enter the histogram:
  not at run start (init sentinel), not via a stamp from a previous
  episode, and never via clamping into the top bin (the phantom spike
  at 15.5-16 s)."""
  env = _mock_tracked_env(fell=True)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  # Fall with no push ever: nothing recorded.
  env.common_step_counter = 500
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(0.0)
  # Push, episode ends WITHOUT a fall, next episode falls unpushed:
  # the stale stamp must not attribute the new fall to the old push.
  env.common_step_counter = 1000
  tracker.record_push(env, torch.tensor([0, 1]))
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  env.common_step_counter = 1100
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 1200
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(0.0)
  # The bins cover a full 20 s episode, so a late-episode fall after an
  # early push IS a genuine datum (17 s -> bin 34)...
  env.common_step_counter = 2000
  tracker.record_push(env, torch.tensor([0, 1]))
  env.common_step_counter = 2000 + int(17.0 / 0.02)
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win[34].item() == pytest.approx(2.0)
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(2.0)
  # ...while an impossible dt (> episode length; cannot occur with
  # per-episode attribution) is dropped, never clamped into the top bin.
  env.common_step_counter = 4000
  tracker.record_push(env, torch.tensor([0, 1]))
  env.common_step_counter = 4000 + int(21.0 / 0.02)
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(2.0)


def test_frontier_buckets_charge_falls_to_speed() -> None:
  """Clean-cohort falls are charged to the commanded-speed bucket."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.5)  # env 1 is clean
  # Command 0.55 m/s -> bin 11 of 64 (0.05 m/s bins, R31 range).
  ct = env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.55, 0.0, 0.0], [0.55, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
  tracker.record_step(env)
  assert tracker._cur_bucket.tolist() == [11, 11]
  # Only the clean env's exposure counts.
  assert tracker._bucket_steps[11].item() == pytest.approx(1.0)
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  # Only env 1 (clean) charges the bin; env 0's fall is push-cohort.
  assert tracker._bucket_falls[11].item() == pytest.approx(1.0)
  assert tracker._bucket_falls.sum().item() == pytest.approx(1.0)


def test_push_cohort_event_filters_and_stamps(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from mjlab.tasks.velocity.mdp import competence as comp

  env = _mock_tracked_env(fell=False)
  env._competence_tracker = CompetenceTracker(env)
  robot = MagicMock()
  robot.data.root_link_vel_w = torch.zeros(2, 6)
  writes: list[tuple[torch.Tensor, torch.Tensor]] = []
  robot.write_root_link_velocity_to_sim = lambda vel, env_ids: writes.append(
    (vel, env_ids)
  )
  env.scene = {"robot": robot}
  env.common_step_counter = 240
  comp.push_cohort_by_setting_velocity(
    env, torch.tensor([0, 1]), {"x": (0.3, 0.3), "y": (0.0, 0.0)}, cohort_frac=0.5
  )
  # Only env 0 (cohort) is pushed, with the sampled magnitude recorded.
  assert len(writes) == 1
  assert writes[0][1].tolist() == [0]
  tracker = env._competence_tracker
  assert tracker.last_push_step[0].item() == pytest.approx(240.0)
  assert tracker.last_push_step[1].item() < 0
  assert tracker._pending_push_mag[0].item() == pytest.approx(0.3, rel=1e-5)
  assert tracker._pending_push_mag[1].item() == pytest.approx(-1.0)


def test_aimd_ignores_congestion_at_zero_difficulty() -> None:
  """Early-chaos regression (v26): bogus cuts at d=0 must not pin ssthresh."""
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  assert (
    c.update(cur_iter=5, fast_fall_rate=1.0, wobble_ema=1.0, attain_ema=0.0) is None
  )
  assert c.ssthresh == pytest.approx(1.0)
  assert c.last_cut_iter < 0


def test_push_cohort_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ADAPTIVE_COMMANDS", "1")
  monkeypatch.setenv("PUSH_COHORT_FRAC", "0.3")
  from mjlab.tasks.velocity import mdp

  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.events["push_robot"].func is mdp.push_cohort_by_setting_velocity
  assert cfg.events["push_robot"].params["cohort_frac"] == pytest.approx(0.3)
  assert "competence_diagnostics" in cfg.curriculum


def _shadow_env(*, peak: float, step: int) -> MagicMock:
  env = _mock_tracked_env(fell=False, step_counter=step)
  env.extras = {"log": {"Metrics/peak_height_mean": torch.tensor(peak)}}
  env.reward_manager.get_term_cfg.return_value = MagicMock(weight=-1e-5)
  return env


def test_joule_shadow_climbs_when_healthy_and_retreats_on_style_collapse() -> None:
  """Feasibility replays: a healthy plateau must climb lambda; a
  v25-slow-style swing-height slide (0.0128 -> 0.0072, 44% drop) must
  freeze at 85% and retreat before 60% decay (ahead of the T4 monitor)."""
  from mjlab.tasks.velocity.mdp.competence import joule_lambda_shadow

  cfg = MagicMock()
  cfg.params = {"lambda_cap": 2e-5, "ramp_iters": 100}
  env0 = _shadow_env(peak=0.0128, step=0)
  env0._competence_tracker = CompetenceTracker(env0)
  term = joule_lambda_shadow(cfg, env0)
  tracker = env0._competence_tracker
  tracker.set_push_cohort(0.5)
  # Mark competence healthy (the gates read the clean cohort).
  tracker.attain_ema[:] = 0.7
  tracker.fast_fall_clean = 0.05

  def run_iters(n: int, peak: float, start: int) -> int:
    step = start
    for _ in range(n):
      env = _shadow_env(peak=peak, step=step)
      env._competence_tracker = tracker
      for _ in range(30):  # settle the slow peak EMA at this level
        tracker.record_peak_height(env)
      tracker._finalized_step = -1
      term(env, torch.tensor([0, 1]))
      step += _NUM_STEPS_PER_ENV
    return step

  # Warm up the EMA to the healthy level, then climb.
  for _ in range(200):
    tracker.record_peak_height(_shadow_env(peak=0.0128, step=0))
  step = run_iters(50, peak=0.0128, start=24)
  lam_healthy = term._lam
  assert lam_healthy == pytest.approx(50 * 2e-7, rel=0.05)

  # Style slide: EMA sinks toward 0.0072 while trailing max holds ~0.0128.
  step = run_iters(60, peak=0.0072, start=step)
  assert term._lam < lam_healthy  # retreat fired (peak_frac < 0.70)
  # And it fired hard: multiplicative unwind, not a freeze.
  assert term._lam < 0.5 * lam_healthy


def test_joule_shadow_freezes_on_sandbagging() -> None:
  from mjlab.tasks.velocity.mdp.competence import joule_lambda_shadow

  cfg = MagicMock()
  cfg.params = {"lambda_cap": 2e-5, "ramp_iters": 100}
  env0 = _shadow_env(peak=0.012, step=0)
  env0._competence_tracker = CompetenceTracker(env0)
  term = joule_lambda_shadow(cfg, env0)
  tracker = env0._competence_tracker
  tracker.set_push_cohort(0.5)
  tracker.fast_fall_clean = 0.05
  for _ in range(200):
    tracker.record_peak_height(_shadow_env(peak=0.012, step=0))
  # Sandbagging: clean attain below floor -> hold, not retreat.
  tracker.attain_ema[:] = 0.3
  env = _shadow_env(peak=0.012, step=24)
  env._competence_tracker = tracker
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]))
  assert out["gates_ok"].item() == 0.0
  assert out["style_broken"].item() == 0.0
  assert term._lam == pytest.approx(0.0)


def _ellipsoid_command_term() -> "UniformVelocityCommand":
  """Build a real UniformVelocityCommand with mocked env internals."""
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.75, 0.75),
      lin_vel_y=(-0.45, 0.45),
      ang_vel_z=(-0.80, 0.80),
    ),
    command_geometry="ellipsoid",
  )
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = 512
  term = UniformVelocityCommand.__new__(UniformVelocityCommand)
  term.cfg = cfg
  term._env = env
  term.vel_command_b = torch.zeros(512, 3)
  term.vel_command_w = torch.zeros(512, 3)
  term.heading_target = torch.zeros(512)
  term.is_heading_env = torch.zeros(512, dtype=torch.bool)
  term.is_standing_env = torch.zeros(512, dtype=torch.bool)
  term.has_stop_tail = torch.zeros(512, dtype=torch.bool)
  term.is_world_env = torch.zeros(512, dtype=torch.bool)
  term.is_forward_env = torch.zeros(512, dtype=torch.bool)
  term.robot = MagicMock()
  return term


def test_ellipsoid_geometry_removes_corners() -> None:
  """No sample may exceed Mahalanobis radius 1; axis maxima stay in play."""
  torch.manual_seed(0)
  term = _ellipsoid_command_term()
  ids = torch.arange(512)
  term._resample_command(ids)
  cmd = term.vel_command_b
  rho = torch.sqrt(
    (cmd[:, 0] / 0.75) ** 2 + (cmd[:, 1] / 0.45) ** 2 + (cmd[:, 2] / 0.80) ** 2
  )
  assert rho.max().item() <= 1.0 + 1e-5
  # The ellipsoid must still reach near the axis maxima (radius use > 0.95
  # somewhere in 512 draws - the surface is populated, not shrunk).
  assert rho.max().item() > 0.90
  # And direction is preserved for projected samples: no axis sign flips
  # relative to a fresh box draw is impossible to check directly, but the
  # distribution must keep substantial spread in every axis.
  assert cmd[:, 0].abs().max().item() > 0.55
  assert cmd[:, 2].abs().max().item() > 0.55


def test_box_geometry_unchanged_default() -> None:
  torch.manual_seed(0)
  term = _ellipsoid_command_term()
  term.cfg.command_geometry = "box"
  ids = torch.arange(512)
  term._resample_command(ids)
  cmd = term.vel_command_b
  rho = torch.sqrt(
    (cmd[:, 0] / 0.75) ** 2 + (cmd[:, 1] / 0.45) ** 2 + (cmd[:, 2] / 0.80) ** 2
  )
  # Box sampling routinely exceeds rho 1 (the corners exist).
  assert rho.max().item() > 1.2


def test_rho_buckets_attribute_corner_falls() -> None:
  """Under box geometry a corner command lands in a high rho bucket."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.5)
  ct = env.command_manager.get_term.return_value
  ranges = MagicMock()
  ranges.lin_vel_x = (-0.75, 0.75)
  ranges.lin_vel_y = (-0.45, 0.45)
  ranges.ang_vel_z = (-0.80, 0.80)
  ct.cfg.ranges = ranges
  # Corner command: all axes near max -> rho ~ sqrt(3) ~ 1.69 -> top bin
  # (clamped) of 64 x 0.05-wide bins (R18/R31).
  ct.vel_command_b = torch.tensor([[0.74, 0.44, 0.79], [0.74, 0.44, 0.79]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.3, 0.1], [0.3, 0.1]])
  tracker.record_step(env)
  assert tracker._cur_rho_bucket.tolist() == [63, 63]
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._rho_falls[63].item() == pytest.approx(1.0)  # clean env only


def test_aimd_arrest_mode_cuts_every_iteration_above_bar() -> None:
  """v27 postmortem: one-cut-per-refractory walked d from 1.0 to 0.28 over
  ~100+ iterations of sustained ignition. Under arrest mode the descent
  must match the discrete cascade's aggression (halving in ~10 iters)."""
  from mjlab.tasks.velocity.mdp.competence import AimdController, AimdParams

  c = AimdController(AimdParams())
  c.d = 1.0
  it = 1000
  assert (
    c.update(cur_iter=it, fast_fall_rate=0.5, wobble_ema=0.2, attain_ema=0.5) == "cut"
  )
  d_after_cut = c.d
  assert d_after_cut == pytest.approx(0.7)
  # Signal stays above the bar: every subsequent iteration decays d.
  for k in range(1, 11):
    r = c.update(cur_iter=it + k, fast_fall_rate=0.5, wobble_ema=0.2, attain_ema=0.5)
    assert r == "arrest"
  assert c.d == pytest.approx(0.7 * 0.93**10, rel=1e-4)
  assert c.d < 0.36  # halved within 10 iterations of sustained signal
  # Signal clears: increases stay refractory-gated (no instant rebound).
  assert (
    c.update(cur_iter=it + 11, fast_fall_rate=0.05, wobble_ema=0.02, attain_ema=0.7)
    is None
  )


def test_aimd_envelope_scale_extends_lerp_target() -> None:
  """envelope_scale=1.3 puts the d=1.0 target beyond the L5 table so the
  sawtooth binds at capacity, not at an artificial cap."""
  from mjlab.tasks.velocity.mdp.competence import aimd_difficulty

  cfg = MagicMock()
  cfg.params = {
    "command_name": "twist",
    "event_name": "push_robot",
    "envelope_scale": 1.3,
  }
  env = _mock_tracked_env(fell=False)
  env._competence_tracker = CompetenceTracker(env)
  ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
  )
  ccfg = UniformVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(3.0, 8.0), ranges=ranges
  )
  cterm = MagicMock()
  cterm.cfg = ccfg
  env.command_manager.get_term.return_value = cterm
  ecfg = MagicMock()
  ecfg.params = {}
  env.event_manager.get_term_cfg.return_value = ecfg
  term = aimd_difficulty(cfg, env)
  term._apply(env, 1.0, 0.0)
  assert ranges.lin_vel_x == pytest.approx((-0.975, 0.975))
  assert ranges.ang_vel_z == pytest.approx((-1.04, 1.04))
  term._apply(env, 0.0, 0.0)
  assert ranges.lin_vel_x == pytest.approx((-0.20, 0.20))


def test_per_axis_attainment_separates_directions() -> None:
  """A robot that delivers x but not y must read high attain_x, low
  attain_y (doc 15 R12: the table's axis ratios are vibes; measure them)."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  ct = env.command_manager.get_term.return_value
  # Commanded diagonal; delivered: full x, none of y.
  ct.vel_command_b = torch.tensor([[0.4, 0.3, 0.0], [0.4, 0.3, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.4, 0.0], [0.4, 0.0]])
  for _ in range(5):
    tracker.record_step(env)
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  strat = tracker.stratified_means()
  assert strat["clean_attain_x"] > 0.05
  assert abs(strat["clean_attain_y"]) < 0.01
  # EMA direction check on raw tensors (alpha=0.1, one episode).
  assert tracker.attain_x_ema[0].item() == pytest.approx(0.1, rel=0.01)
  assert tracker.attain_y_ema[0].item() == pytest.approx(0.0, abs=1e-6)


def _split_aimd_term() -> "tuple[aimd_difficulty, MagicMock]":
  cfg = MagicMock()
  cfg.params = {"command_name": "twist", "event_name": "push_robot"}
  env = _mock_tracked_env(fell=False)
  env._competence_tracker = CompetenceTracker(env)
  env._competence_tracker.set_push_cohort(0.5)
  ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
  )
  ccfg = UniformVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(3.0, 8.0), ranges=ranges
  )
  cterm = env.command_manager.get_term.return_value
  cterm.cfg = ccfg
  ecfg = MagicMock()
  ecfg.params = {}
  env.event_manager.get_term_cfg.return_value = ecfg
  return aimd_difficulty(cfg, env), env


def test_split_aimd_push_burn_cuts_pushes_not_commands() -> None:
  """The v28 counterexample: pushed cohort burning (excess 0.30) while the
  clean cohort is healthy must cut d_push and leave d_cmd climbing."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  term._cmd_ctrl.d = 0.9
  term._push_ctrl.d = 0.9
  tracker.attain_ema[:] = 0.6
  tracker.wobble_ema[:] = 0.02
  tracker.fast_fall_rate = 0.15  # blended: what fooled v28's controller
  tracker.fast_fall_clean = 0.04
  tracker.fast_fall_pushed = 0.34
  env.common_step_counter = 1000 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["difficulty_push"].item() == pytest.approx(0.9 * 0.7)
  assert out["difficulty"].item() >= 0.89  # cmd axis unharmed (held, fp32)


def test_split_aimd_clean_burn_cuts_commands_not_pushes() -> None:
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  term._cmd_ctrl.d = 0.9
  term._push_ctrl.d = 0.9
  tracker.attain_ema[:] = 0.6
  tracker.wobble_ema[:] = 0.02
  tracker.fast_fall_rate = 0.42
  tracker.fast_fall_clean = 0.40
  tracker.fast_fall_pushed = 0.45  # excess only 0.05: pushes not to blame
  env.common_step_counter = 1000 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["difficulty"].item() == pytest.approx(0.9 * 0.7)
  assert out["difficulty_push"].item() >= 0.9


def test_split_aimd_population_emergency_arrests_both() -> None:
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  term._cmd_ctrl.d = 0.9
  term._push_ctrl.d = 0.9
  tracker.attain_ema[:] = 0.3
  tracker.wobble_ema[:] = 0.4
  tracker.fast_fall_rate = 0.60
  tracker.fast_fall_clean = 0.30  # each below its own bar...
  tracker.fast_fall_pushed = 0.50  # (excess 0.20 below push bar too)
  env.common_step_counter = 1000 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  # ...but the population emergency cuts both anyway.
  assert out["difficulty"].item() < 0.9
  assert out["difficulty_push"].item() < 0.9


def test_joule_lambda_live_applies_weight() -> None:
  """Live mode writes -lambda into the reward weight; shadow does not."""
  from mjlab.tasks.velocity.mdp.competence import joule_lambda_shadow

  for live, expect_written in ((False, False), (True, True)):
    cfg = MagicMock()
    cfg.params = {"lambda_cap": 2e-5, "ramp_iters": 10, "apply_live": live}
    env0 = _shadow_env(peak=0.012, step=0)
    env0._competence_tracker = CompetenceTracker(env0)
    term = joule_lambda_shadow(cfg, env0)
    tracker = env0._competence_tracker
    tracker.set_push_cohort(0.5)
    tracker.attain_ema[:] = 0.7
    tracker.fast_fall_clean = 0.05
    for _ in range(200):
      tracker.record_peak_height(_shadow_env(peak=0.012, step=0))
    env = _shadow_env(peak=0.012, step=24)
    env._competence_tracker = tracker
    term_cfg = MagicMock()
    term_cfg.weight = -1e-5
    env.reward_manager.get_term_cfg.return_value = term_cfg
    tracker._finalized_step = -1
    term(env, torch.tensor([0, 1]))
    if expect_written:
      assert term_cfg.weight == pytest.approx(-term._lam)
      assert term._lam > 0
    else:
      assert term_cfg.weight == pytest.approx(-1e-5)


def test_landing_anneal_triggers_at_capacity_plateau() -> None:
  """v30 replay: at capacity from ~1100, plateaued from ~1400 -> the
  landing factor must begin decaying around 1600 and be well below 1 by
  the historical ignition window (~1750), monotone throughout."""
  term, env = _split_aimd_term()
  term._landing_enabled = True
  tracker = env._competence_tracker
  term._cmd_ctrl.d = 1.0
  term._push_ctrl.d = 0.6
  tracker.wobble_ema[:] = 0.02
  tracker.fast_fall_rate = 0.10
  tracker.fast_fall_clean = 0.08
  tracker.fast_fall_pushed = 0.28

  factors = {}
  for it in range(1100, 1800):
    tracker.attain_ema[:] = 0.615  # flat plateau
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
    factors[it] = out["landing_factor"].item()

  # Conditions age from 1100: capacity needs 200 iters, plateau 150.
  assert factors[1250] == pytest.approx(1.0)
  assert factors[1299] == pytest.approx(1.0)
  assert factors[1350] < 1.0  # anneal underway after both aged
  assert factors[1500] < 0.5  # 0.995^200
  assert factors[1750] < 0.15  # well down before the ignition window
  vals = [factors[k] for k in sorted(factors)]
  assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:], strict=False))  # monotone
  assert env._landing_factor == pytest.approx(vals[-1])


def test_landing_anneal_off_by_default() -> None:
  term, env = _split_aimd_term()
  term._cmd_ctrl.d = 1.0
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.615
  tracker.wobble_ema[:] = 0.02
  tracker.fast_fall_rate = 0.10
  tracker.fast_fall_clean = 0.08
  tracker.fast_fall_pushed = 0.28
  for it in range(1100, 1800, 50):
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["landing_factor"].item() == pytest.approx(1.0)


def test_runner_wrapper_scales_desired_kl() -> None:
  from mjlab.rl.runner import MjlabOnPolicyRunner

  runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
  runner.cfg = {}
  alg = MagicMock()
  alg.desired_kl = 0.01
  calls = []
  alg.update = lambda *a, **k: calls.append(alg.desired_kl)
  runner.alg = alg
  inner_env = MagicMock()
  inner_env._landing_factor = 1.0
  wrapper_env = MagicMock()
  wrapper_env.unwrapped = inner_env
  runner.env = wrapper_env
  runner._install_landing_anneal()

  alg.update()
  inner_env._landing_factor = 0.25
  alg.update()
  inner_env._landing_factor = 0.001  # floor clamps
  alg.update()
  assert calls[0] == pytest.approx(0.01)
  assert calls[1] == pytest.approx(0.0025)
  assert calls[2] == pytest.approx(2e-4)


def test_obs_norm_freeze_sets_until_and_stops_updates() -> None:
  """R15: the freeze must set `until` on every EmpiricalNormalization in
  actor/critic, and a normalizer past its budget must stop learning."""
  from rsl_rl.modules import EmpiricalNormalization

  from mjlab.rl.runner import MjlabOnPolicyRunner

  runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
  runner.cfg = {"num_steps_per_env": 24}
  env = MagicMock()
  env.num_envs = 100
  runner.env = env
  norm_a = EmpiricalNormalization(4)
  norm_c = EmpiricalNormalization(4)
  actor = torch.nn.Sequential(norm_a)
  critic = torch.nn.Sequential(norm_c)
  alg = MagicMock()
  alg.actor = actor
  alg.critic = critic
  runner.alg = alg

  runner._apply_obs_norm_freeze({"obs_norm_freeze_iters": 500})
  assert norm_a.until == 500 * 24 * 100
  assert norm_c.until == 500 * 24 * 100

  # Past the budget, update() is a no-op.
  norm_a.count.fill_(norm_a.until + 1)
  mean_before = norm_a._mean.clone()
  norm_a.train()
  norm_a.update(torch.randn(64, 4) + 5.0)
  assert torch.allclose(norm_a._mean, mean_before)

  # Zero/absent knob leaves normalizers unfrozen (legacy).
  norm_b = EmpiricalNormalization(4)
  alg.actor = torch.nn.Sequential(norm_b)
  runner._apply_obs_norm_freeze({})
  assert norm_b.until is None


def test_freeze_policy_probe_noops_optimizer_after_threshold() -> None:
  """R16 probe: after N updates the optimizer step must become a no-op
  while update() itself keeps running (losses/logging intact)."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
  runner.cfg = {"freeze_policy_after": 3}
  param = torch.nn.Parameter(torch.zeros(2))
  opt = torch.optim.SGD([param], lr=1.0)
  alg = MagicMock()
  alg.desired_kl = 0.01
  alg.optimizer = opt

  def fake_update(*a, **k):
    param.grad = torch.ones_like(param)
    opt.step()
    return {}

  alg.update = fake_update
  env = MagicMock()
  env.unwrapped = MagicMock()
  env.unwrapped._landing_factor = 1.0
  runner.env = env
  runner.alg = alg
  runner._install_landing_anneal()

  alg.update()
  alg.update()
  assert param[0].item() == pytest.approx(-2.0)  # two real steps
  alg.update()  # threshold reached: frozen from this update on
  after_third = param[0].item()
  alg.update()
  alg.update()
  assert param[0].item() == pytest.approx(after_third)  # bit-frozen


def test_interp_crossing_and_binned_quantile() -> None:
  """R18: the curriculum consumes interpolated statistics, not buckets."""
  from mjlab.tasks.velocity.mdp.competence import (
    _binned_quantile,
    _interp_crossing,
  )

  # Linear hazard ramp: h(bin i) = i * 1e-4 over 0.025-wide bins.
  hazards = torch.arange(32, dtype=torch.float32) * 1e-4
  bar = 5e-4
  x = _interp_crossing(hazards, 0.025, bar)
  # Crossing near bin 5 (h=5e-4), interpolated - continuous, not a
  # 0.1-quantized stair step.
  assert 0.10 < x < 0.16
  # Never crossed, no exposure info: returns full range.
  assert _interp_crossing(torch.zeros(32), 0.025, bar) == pytest.approx(0.8)
  # Never crossed WITH exposure: clamps to the highest sampled bin -
  # "clean as far as we have sampled", not an instrument-range
  # capability claim (frontier_speed read 3.2 m/s at near-zero fall
  # rates before this).
  exposure = torch.zeros(32)
  exposure[:20] = 100.0
  x = _interp_crossing(torch.zeros(32), 0.025, bar, exposure=exposure)
  assert x == pytest.approx((19 + 0.5) * 0.025)
  # No exposure anywhere: nothing demonstrated.
  assert _interp_crossing(torch.zeros(32), 0.025, bar, exposure=torch.zeros(32)) == 0.0
  # A genuine crossing below the exposure limit is unaffected.
  x = _interp_crossing(hazards, 0.025, bar, exposure=exposure)
  assert 0.10 < x < 0.16

  # Quantiles of a binned distribution: uniform mass in bins 0..3.
  counts = torch.tensor([10.0, 10.0, 10.0, 10.0] + [0.0] * 28)
  assert _binned_quantile(counts, 0.5, 0.50) == pytest.approx(1.0)
  assert _binned_quantile(counts, 0.5, 0.75) == pytest.approx(1.5)
  assert _binned_quantile(torch.zeros(32), 0.5, 0.75) == pytest.approx(0.0)


def test_frontier_controller_climbs_glides_and_floors() -> None:
  """R20: difficulty slews toward attained_frontier x headroom, glides
  (never cascades) when the frontier retreats, and fall cuts cannot
  pierce the attained floor."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.wobble_ema[:] = 0.02
  tracker.attain_ema[:] = 0.70

  def set_frontier(v: float) -> None:
    tracker.attain_by_speed.zero_()
    tracker.attain_by_speed_weight.zero_()
    n = int(v / tracker.speed_bin_width)
    tracker.attain_by_speed[:n] = 0.75
    tracker.attain_by_speed_weight[:n] = 100.0

  def step(it: int, f_clean: float = 0.0) -> float:
    tracker.fast_fall_clean = f_clean
    tracker.fast_fall_rate = f_clean
    tracker.fast_fall_pushed = 0.20
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
    return out["difficulty"].item()

  lo_v, hi_v = 0.20, 0.75 * term._envelope_scale

  # Frontier at 0.50 m/s: target = 0.575, so from d=0.5 it climbs at alpha.
  set_frontier(0.50)
  term._cmd_ctrl.d = 0.5
  d1 = step(1000)
  assert d1 == pytest.approx(0.5 + 0.002)
  # Converges to the target, not the cap: run it forward.
  for it in range(1001, 1150):
    d_eq = step(it)
  frontier_v = term._attained_best_v
  target_d = (frontier_v * term._headroom - lo_v) / (hi_v - lo_v)
  assert d_eq == pytest.approx(target_d, abs=0.01)
  assert d_eq < 1.0  # emphatically not parked at the cap

  # Frontier retreats: glide down at bounded slew, stopping at the floor
  # from the (slowly decaying) attained memory - not a halving cascade.
  set_frontier(0.30)
  d_prev = d_eq
  d_next = step(1150)
  assert d_next == pytest.approx(d_prev - 2 * 0.002)
  for it in range(1151, 1400):
    d_glide = step(it)
  floor_d = (0.95 * term._attained_best_v - lo_v) / (hi_v - lo_v)
  assert d_glide == pytest.approx(floor_d, abs=0.01)
  assert d_glide > 0.2  # held well above zero (floor decays ~0.999/iter)

  # R27: under a SUSTAINED crash signal the floor releases (0.99/iter,
  # no frontier re-ratchet) and difficulty follows capability down -
  # the v38/v39 cage is the counterexample this encodes. The watchdog,
  # not the floor, is the answer to a true catastrophe.
  for it in range(1400, 1470):
    d_crash = step(it, f_clean=0.60)
  assert term._attained_best_v < 0.30
  assert d_crash < 0.15


def test_survivor_conditioning_gates_capability_credit() -> None:
  """R21: a sprint that ends in a fall contributes nothing to attain(v);
  a timed-out episode deposits its duration-weighted samples."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.0)  # both envs clean... 0 pushed
  ct = env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.45, 0.0], [0.45, 0.0]])
  for _ in range(3):  # duration weighting: three steps buffered
    tracker.record_step(env)
  b = int(0.5 / tracker.speed_bin_width)

  # Episode ends in a FALL: buffered credit is discarded.
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == pytest.approx(0.0)
  assert tracker._attain_ep_weight.sum().item() == pytest.approx(0.0)

  # Same steps, episode TIMES OUT: credit lands, weighted by duration.
  for _ in range(3):
    tracker.record_step(env)
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  env.common_step_counter = 96
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == pytest.approx(6.0)  # 2 envs x 3 steps
  assert tracker._attain_bin_sum[b].item() == pytest.approx(6.0 * 0.9, rel=1e-3)


def test_graded_bar_and_bootstrap_fix_cold_start() -> None:
  """R22: v37 replay — a 0.64 attainment ceiling at 0.2 m/s must qualify
  under the graded bar, and with no qualified bins the controller crawls
  instead of parking."""
  from mjlab.tasks.velocity.mdp.competence import _attained_frontier

  # Graded bar: bin at ~0.18 m/s has bar 1-0.08/0.18 ~ 0.56, so the
  # measured 0.64 ceiling QUALIFIES (flat 0.60 bar would too here, but
  # at 0.14 the graded bar ~0.43 rescues bins a flat bar rejects).
  curve = torch.zeros(32)
  w = torch.zeros(32)
  curve[3] = 0.50  # ~0.14 m/s at 0.55 attain: graded bar ~0.43 passes
  w[3] = 100.0
  curve[4] = 0.64  # ~0.18 m/s: passes graded bar ~0.56
  w[4] = 100.0
  f = _attained_frontier(curve, w, 0.04, 0.60)
  assert f > 0.15  # frontier engaged at cold-start speeds

  # Bootstrap crawl: zero frontier + healthy -> d climbs at alpha.
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.55
  tracker.wobble_ema[:] = 0.02
  tracker.fast_fall_clean = 0.05
  tracker.fast_fall_rate = 0.05
  tracker.fast_fall_pushed = 0.15
  term._cmd_ctrl.d = 0.0
  env.common_step_counter = 100 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["difficulty"].item() == pytest.approx(0.002)
  # Unhealthy: crawl pauses.
  tracker.fast_fall_clean = 0.30
  tracker.fast_fall_rate = 0.30
  env.common_step_counter = 101 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["difficulty"].item() == pytest.approx(0.002)


def test_push_survival_censoring() -> None:
  """R24 (user scenarios): a hard push followed quickly by an easy push
  then a fall must NOT credit the hard push (censored) and must charge
  the easy one; a push shortly before timeout is censored, not survived;
  a full clean window earns survival."""
  env = _mock_tracked_env(fell=False)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  w_steps = int(tracker.push_obs_window_s / 0.02)  # 300 steps
  b_hard = int(0.80 / tracker.push_bin_width)
  b_easy = int(0.20 / tracker.push_bin_width)

  # Hard push, easy push 3 s later (inside window), fall 1 s after that.
  env.common_step_counter = 1000
  tracker.record_push(env, torch.tensor([0]), torch.tensor([0.80]))
  env.common_step_counter = 1150  # 3 s later - hard window unfinished
  tracker.record_push(env, torch.tensor([0]), torch.tensor([0.20]))
  assert tracker._push_bin_survive[b_hard].item() == pytest.approx(0.0)  # censored
  env.termination_manager.get_term.return_value = torch.tensor([True, False])
  env.common_step_counter = 1200  # 1 s after easy push: in-window fall
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_bin_fall[b_easy].item() == pytest.approx(1.0)
  assert tracker._push_bin_fall[b_hard].item() == pytest.approx(0.0)
  assert tracker._push_bin_survive[b_hard].item() == pytest.approx(0.0)

  # Push just before timeout: censored, not survived.
  env.common_step_counter = 2000
  tracker.record_push(env, torch.tensor([1]), torch.tensor([0.80]))
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  env.common_step_counter = 2000 + w_steps // 3  # window unfinished
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_bin_survive[b_hard].item() == pytest.approx(0.0)
  assert tracker._push_bin_fall[b_hard].item() == pytest.approx(0.0)

  # Full clean window before the next push: survival.
  env.common_step_counter = 3000
  tracker.record_push(env, torch.tensor([0]), torch.tensor([0.80]))
  env.common_step_counter = 3000 + w_steps + 10
  tracker.record_push(env, torch.tensor([0]), torch.tensor([0.20]))
  assert tracker._push_bin_survive[b_hard].item() == pytest.approx(1.0)
  # And a fall BEYOND the window does not retract it.
  env.termination_manager.get_term.return_value = torch.tensor([True, False])
  env.common_step_counter = 3000 + 2 * w_steps + 50
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_bin_fall[b_easy].item() == pytest.approx(1.0)  # unchanged
  assert tracker._push_bin_survive[b_easy].item() == pytest.approx(1.0)


def test_adaptive_obs_window_tracks_t75() -> None:
  """R25: the observation window follows the measured push->fall t75
  (clamped), and holds the bootstrap value until evidence accumulates."""
  env = _mock_tracked_env(fell=True, ep_len=200)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  assert tracker.push_obs_window_s == pytest.approx(6.0)  # bootstrap

  # 60 falls at ~3.6 s and ~4.4 s after the push -> t75 ~ 4.3 s.
  step = 1000
  for _k in range(30):
    for dt_steps in (180, 220):
      env.common_step_counter = step
      tracker.record_push(env, torch.tensor([0, 1]), torch.tensor([0.3, 0.3]))
      env.termination_manager.get_term.return_value = torch.tensor([True, True])
      env.common_step_counter = step + dt_steps
      tracker._finalized_step = -1
      tracker.finalize_episodes(env, torch.tensor([0, 1]))
      step += 2000
  # Trigger the cadence refresh.
  env.common_step_counter = step + 50 * _NUM_STEPS_PER_ENV
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0  # satisfy the refresh precondition
  tracker._finalized_step = -1
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert 3.5 < tracker.push_obs_window_s < 5.0  # adapted to measured t75


def test_frontier_retreats_during_population_decline() -> None:
  """R26 (v38 replay): unhealthy windows must decay frontier confidence,
  not poll the surviving elite."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.attain_by_speed[:16] = 0.75
  tracker.attain_by_speed_weight[:16] = 100.0
  tracker._attain_bin_sum[:16] = 90.0 * 0.8  # elite survivors still great
  tracker._attain_bin_weight[:16] = 90.0
  tracker.fast_fall_clean = 0.50  # population dying
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0
  env.common_step_counter = 5000
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracker._finalized_step = -1
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  # Curve values unchanged (no fold), confidence decayed.
  assert tracker.attain_by_speed[8].item() == pytest.approx(0.75)
  assert tracker.attain_by_speed_weight[8].item() == pytest.approx(50.0)


def test_floor_memory_releases_during_crash() -> None:
  """R26: sustained crash signal must deflate the attained memory fast
  so the floor follows capability down instead of caging the run."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.6
  tracker.wobble_ema[:] = 0.02
  term._attained_best_v = 0.68
  term._cmd_ctrl.d = 0.5
  tracker.attain_by_speed.zero_()
  tracker.attain_by_speed_weight.zero_()  # frontier reads 0

  def step(it: int, f: float) -> None:
    tracker.fast_fall_clean = f
    tracker.fast_fall_rate = f
    tracker.fast_fall_pushed = f
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    term(env, torch.tensor([0, 1]), "twist", "push_robot")

  step(1000, 0.05)
  assert term._attained_best_v == pytest.approx(0.68 * 0.999)
  before = term._attained_best_v
  for it in range(1001, 1101):
    step(it, 0.60)  # crash: fast release
  assert term._attained_best_v < before * 0.45  # 0.99^100 ~ 0.366


def test_release_is_not_reratcheted_by_stale_frontier() -> None:
  """R27 (v39 replay): during a crash the floor memory must decay even
  while the (stale) frontier still reads high."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.6
  tracker.wobble_ema[:] = 0.02
  # Stale-high frontier: curve says 0.67 with plenty of weight.
  n = int(0.68 / tracker.speed_bin_width)
  tracker.attain_by_speed[:n] = 0.75
  tracker.attain_by_speed_weight[:n] = 100.0
  term._attained_best_v = 0.673
  term._cmd_ctrl.d = 0.574
  for it in range(2000, 2100):
    tracker.fast_fall_clean = 0.50
    tracker.fast_fall_rate = 0.50
    tracker.fast_fall_pushed = 0.50
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert term._attained_best_v < 0.673 * 0.45  # released despite frontier
  assert out["difficulty"].item() < 0.30  # cage broken: d followed down


def test_stress_scaled_headroom_closes_the_valve() -> None:
  """R27: the beyond-frontier margin shrinks to zero as stress rises."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.7
  tracker.wobble_ema[:] = 0.02
  n = int(0.52 / tracker.speed_bin_width)
  tracker.attain_by_speed[:n] = 0.75
  tracker.attain_by_speed_weight[:n] = 100.0
  lo_v, hi_v = 0.20, 0.75 * term._envelope_scale

  def eq_target(f_clean: float) -> float:
    term._cmd_ctrl.d = 0.2
    tracker.fast_fall_clean = f_clean
    tracker.fast_fall_rate = f_clean
    tracker.fast_fall_pushed = f_clean
    d = 0.2
    for it in range(3000, 3400):
      env.common_step_counter = it * _NUM_STEPS_PER_ENV
      tracker._finalized_step = -1
      out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
      d = out["difficulty"].item()
    return lo_v + (hi_v - lo_v) * d

  v_relaxed = eq_target(0.01)  # ~ frontier x 1.15
  v_stressed = eq_target(0.13)  # near the gate: margin ~ gone
  assert v_relaxed > v_stressed
  assert v_stressed < 0.56  # ~ frontier x <1.02 (floor may hold it near)


def test_envelope_auto_extends_at_the_cap() -> None:
  """R30: at the cap, healthy, frontier at the wall -> envelope grows 5%
  with commanded max continuous; unhealthy or distant frontier -> no
  extension."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.8
  tracker.wobble_ema[:] = 0.02
  # Frontier right at the wall: fill bins to ~ vmax (envelope 1.0: 0.75).
  n = int(0.78 / tracker.speed_bin_width)
  tracker.attain_by_speed[:n] = 0.8
  tracker.attain_by_speed_weight[:n] = 100.0
  term._cmd_ctrl.d = 1.0

  def step(it: int, f: float) -> dict:
    tracker.fast_fall_clean = f
    tracker.fast_fall_rate = f
    tracker.fast_fall_pushed = f
    env.common_step_counter = it * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    return term(env, torch.tensor([0, 1]), "twist", "push_robot")

  out0 = step(1000, 0.05)
  assert out0["envelope_scale"].item() == pytest.approx(1.05)
  # Commanded max continuous: d rescaled below 1.0, vmax unchanged.
  vmax = out0["lin_vel_x_max"].item()
  assert vmax == pytest.approx(0.75, abs=0.02)
  assert out0["difficulty"].item() < 1.0

  # Unhealthy: no further extension even if d returns to the cap.
  term._cmd_ctrl.d = 1.0
  out1 = step(1001, 0.30)
  assert out1["envelope_scale"].item() == pytest.approx(1.05)


def test_frontier_instrument_outranges_envelope_cap() -> None:
  """R31: the attain histogram must read past anything the controller
  can command (envelope sanity cap 4x table = 3.0 m/s), and the push
  histogram past any sampled |dv|; a saturated top bin silently caps
  the auto-extension gate (v45: frontier pinned at 1.260 = old
  32x0.04 top-bin center while commands sat at 1.389)."""
  tracker = CompetenceTracker(_mock_tracked_env(fell=False))
  vmax_possible = COMMAND_LEVEL_TABLE[-1]["lin_vel_x"][1] * 4.0
  assert tracker.n_cmd_buckets * tracker.speed_bin_width > vmax_possible
  assert tracker.n_push_bins * tracker.push_bin_width >= 2.0


def test_extension_requires_strict_attainment() -> None:
  """R32: top bins at 0.70 attain pass the lax climb bar (0.60) but must
  NOT extend the envelope (strict bar 0.80): 'surviving' a speed is not
  'stably hitting' it (v45 outpacing)."""
  term, env = _split_aimd_term()
  tracker = env._competence_tracker
  tracker.attain_ema[:] = 0.8
  tracker.wobble_ema[:] = 0.02
  n = int(0.78 / tracker.speed_bin_width)
  tracker.attain_by_speed[:n] = 0.70
  tracker.attain_by_speed_weight[:n] = 100.0
  tracker.fast_fall_clean = 0.05
  tracker.fast_fall_rate = 0.05
  tracker.fast_fall_pushed = 0.05
  term._cmd_ctrl.d = 1.0
  env.common_step_counter = 1000 * _NUM_STEPS_PER_ENV
  tracker._finalized_step = -1
  out = term(env, torch.tensor([0, 1]), "twist", "push_robot")
  assert out["envelope_scale"].item() == pytest.approx(1.0)
  # Lax frontier sees the wall; strict does not.
  assert out["attained_frontier_v"].item() > 0.5
  assert out["strict_frontier_v"].item() < 0.5


def _walking_env_for_attain() -> MagicMock:
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = 2
  env.episode_length_buf = torch.tensor([100, 100])
  env.max_episode_length = 1000
  env.common_step_counter = 24
  env.termination_manager.active_terms = ["fell_over"]
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  ct = MagicMock()
  ct.is_standing_env = torch.tensor([False, False])
  ct.vel_command_b = torch.tensor([[0.55, 0.0, 0.0], [0.55, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
  ct.robot.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
  ct.cfg.ranges.lin_vel_x = (-0.75, 0.75)
  ct.cfg.ranges.lin_vel_y = (-0.45, 0.45)
  ct.cfg.ranges.ang_vel_z = (-0.80, 0.80)
  env.command_manager.get_term.return_value = ct
  return env


def test_attain_credit_requires_settling() -> None:
  """R33 guard 1: no frontier credit until the command has been held
  attain_settle_s - the acceleration transient measures 'reached', not
  'maintained'."""
  env = _walking_env_for_attain()
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.0)  # all envs clean
  tracker._step_dt = 1.0  # settle = 0.75 -> settled from dwell 1
  b = int(0.55 / tracker.speed_bin_width)
  tracker.record_step(env)  # dwell 0 (bin just entered): excluded
  assert tracker._attain_ep_weight[0, b].item() == 0.0
  tracker.record_step(env)  # dwell 1 >= 0.75: counts
  tracker.record_step(env)
  assert tracker._attain_ep_weight[0, b].item() == pytest.approx(2.0)


def test_attain_fold_censors_short_segments() -> None:
  """R33 guard 2: a survivor episode's bin evidence folds only if it
  held the bin >= attain_min_dwell_s post-settle - a high-speed command
  landing just before timeout is censored, not credited."""
  env = _walking_env_for_attain()
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.0)  # all envs clean
  tracker._step_dt = 1.0  # min dwell = 3 steps
  b = int(0.55 / tracker.speed_bin_width)
  for _ in range(3):  # 2 settled steps of credit: below the dwell bar
    tracker.record_step(env)
  env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == 0.0  # censored
  for _ in range(5):  # 4 settled steps: clears the bar
    tracker.record_step(env)
  env.common_step_counter = 48  # new step so the finalize guard passes
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() >= 3.0  # folded
