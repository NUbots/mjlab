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
  # Refractory: same signal a few iters later must not cut again.
  assert (
    c.update(cur_iter=1005, fast_fall_rate=0.4, wobble_ema=0.1, attain_ema=0.5) is None
  )
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
  counts = tracker.push_fall_dt_counts
  assert counts[1].item() == pytest.approx(2.0)
  assert counts.sum().item() == pytest.approx(2.0)


def test_frontier_buckets_charge_falls_to_speed() -> None:
  """Clean-cohort falls are charged to the commanded-speed bucket."""
  env = _mock_tracked_env(fell=False)
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(0.5)  # env 1 is clean
  # Command 0.55 m/s -> bucket 5.
  ct = env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.55, 0.0, 0.0], [0.55, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
  tracker.record_step(env)
  assert tracker._cur_bucket.tolist() == [5, 5]
  # Only the clean env's exposure counts.
  assert tracker._bucket_steps[5].item() == pytest.approx(1.0)
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  # Only env 1 (clean) charges the bucket; env 0's fall is push-cohort.
  assert tracker._bucket_falls[5].item() == pytest.approx(1.0)
  assert tracker._bucket_falls.sum().item() == pytest.approx(1.0)


def test_push_cohort_event_filters_and_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
  from mjlab.tasks.velocity.mdp import competence as comp

  env = _mock_tracked_env(fell=False)
  env._competence_tracker = CompetenceTracker(env)
  pushed_ids: list[torch.Tensor] = []
  monkeypatch.setattr(
    "mjlab.envs.mdp.events.push_by_setting_velocity",
    lambda e, ids, vr: pushed_ids.append(ids),
  )
  env.common_step_counter = 240
  comp.push_cohort_by_setting_velocity(
    env, torch.tensor([0, 1]), {"x": (-0.2, 0.4)}, cohort_frac=0.5
  )
  assert len(pushed_ids) == 1
  assert pushed_ids[0].tolist() == [0]
  assert env._competence_tracker.last_push_step[0].item() == pytest.approx(240.0)
  assert env._competence_tracker.last_push_step[1].item() < 0


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
  # Corner command: all axes near max -> rho ~ sqrt(3) ~ 1.69 -> bucket 7.
  ct.vel_command_b = torch.tensor([[0.74, 0.44, 0.79], [0.74, 0.44, 0.79]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.3, 0.1], [0.3, 0.1]])
  tracker.record_step(env)
  assert tracker._cur_rho_bucket.tolist() == [7, 7]
  env.termination_manager.get_term.return_value = torch.tensor([True, True])
  env.common_step_counter = 48
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._rho_falls[7].item() == pytest.approx(1.0)  # clean env only
