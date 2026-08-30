"""Tests for the competence-gated reward curriculum and its diagnostics."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs.mdp.curriculums import RewardCurriculumStage
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg
from mjlab.tasks.velocity.mdp.competence import (
  CompetenceController,
  CompetenceThresholds,
  CompetenceTracker,
  _attained_frontier,
  _binned_quantile,
  _evidence_masked,
  _interp_crossing,
  get_competence_tracker,
  push_cohort_by_setting_velocity,
  staged_on_competence,
)

_NUM_STEPS_PER_ENV = 24


def _thresholds(**overrides: float | int) -> CompetenceThresholds:
  base = CompetenceThresholds(cooldown_iters=2, promote_streak_required=2)
  for key, value in overrides.items():
    setattr(base, key, value)
  return base


@pytest.fixture
def tracked_env() -> MagicMock:
  """Two walking envs, tracking 0.5 m/s at 0.45 m/s, neither falling."""
  return _mock_tracked_env(fell=False)


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


##
# Controller: hysteresis, cooldown, streaks.
##


def test_controller_promotes_after_cooldown_and_streak() -> None:
  ctrl = CompetenceController(l_max=3, thresholds=_thresholds())
  step = 2 * _NUM_STEPS_PER_ENV
  assert (
    ctrl.update(fell_ema=0.1, common_step_counter=step, attain_ema=0.9, wobble_ema=0.01)
    is None
  )
  step += 2 * _NUM_STEPS_PER_ENV
  assert (
    ctrl.update(fell_ema=0.1, common_step_counter=step, attain_ema=0.9, wobble_ema=0.01)
    == "promote"
  )
  assert ctrl.level == 1


def test_controller_demotes_immediately() -> None:
  ctrl = CompetenceController(l_max=3, thresholds=_thresholds())
  ctrl.level = 2
  assert (
    ctrl.update(
      fell_ema=0.1,
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
      fell_ema=0.32,
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


def test_controller_sandbag_demotes_without_falls() -> None:
  """Conservative retreat (high ep_len, low falls, near-zero attainment)
  must DEMOTE — falls are the trailing indicator, so fall-gated demotion
  never fires until terminal collapse."""
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 4
  result = ctrl.update(
    fell_ema=0.05,
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
    fell_ema=0.1,
    common_step_counter=300 * _NUM_STEPS_PER_ENV,
    attain_ema=0.9,
    wobble_ema=0.4,
  )
  assert result == "demote"


def test_fast_fall_rate_demotes_despite_stale_slow_emas() -> None:
  """A top-rung crash must cascade the level down on the fast channel
  alone: the slow per-env EMAs stay healthy for ~200 iterations (long
  episodes -> rare episode-end updates) while the policy shatters."""
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 5
  step = 1000 * _NUM_STEPS_PER_ENV
  for expected_level in (4, 3, 2, 1, 0):
    assert (
      ctrl.update(
        fell_ema=0.1,
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
  """Healthy top-rung operation measures ~0.26 falls/episode-end; the 0.5
  bar must not demote a working hard-envelope policy."""
  ctrl = CompetenceController(l_max=5, thresholds=_thresholds())
  ctrl.level = 5
  result = ctrl.update(
    fell_ema=0.28,
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
      fell_ema=0.1, common_step_counter=step, attain_ema=0.9, wobble_ema=0.01
    )

  # Base streak (2) is NOT enough to enter level 3 (top rung starts there).
  assert healthy_update() is None
  assert healthy_update() is None
  assert ctrl.level == 2
  # The longer streak (4) is.
  assert healthy_update() is None
  assert healthy_update() == "promote"
  assert ctrl.level == 3


##
# Tracker: episode-stat bookkeeping.
##


def test_tracker_excludes_standing_from_track_err() -> None:
  env = _mock_tracked_env(fell=False)
  env.termination_manager.get_term.return_value = torch.tensor([False, True])
  ct = env.command_manager.get_term.return_value
  ct.is_standing_env = torch.tensor([True, False])
  ct.vel_command_b = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.0, 0.0], [0.4, 0.0]])

  tracker = CompetenceTracker(env)
  tracker.record_step(env)
  assert tracker._track_weight[0].item() == 0.0
  assert tracker._track_weight[1].item() == 1.0
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.fell_ema[1].item() > tracker.fell_ema[0].item()


def test_tracker_pessimistic_init(tracked_env: MagicMock) -> None:
  """A fresh tracker must read as incompetent, not perfect."""
  tracker = CompetenceTracker(tracked_env)
  stats = tracker.population_means()
  assert stats["fell_ema"] == pytest.approx(1.0)
  assert stats["track_err_norm"] == pytest.approx(1.0)
  assert stats["ep_len_frac"] == pytest.approx(0.0)


def test_tracker_finalize_all_weightless_envs(tracked_env: MagicMock) -> None:
  """Resetting only standing envs (zero track weight) must not crash."""
  tracker = CompetenceTracker(tracked_env)
  # No record_step calls at all -> both envs weightless.
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  # track_err EMA unchanged from init; fell/ep_len EMAs still update.
  assert tracker.track_err_ema[0].item() == pytest.approx(1.0)
  assert tracker.fell_ema[0].item() == pytest.approx(0.9)


def test_attainment_true_fraction_at_small_commands(tracked_env: MagicMock) -> None:
  """Perfect tracking of a 0.18 m/s command must read attain ~1.0, and
  commands below the meaningful-command filter must contribute nothing."""
  ct = tracked_env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.18, 0.0, 0.0], [0.18, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.18, 0.0], [0.18, 0.0]])
  tracker = CompetenceTracker(tracked_env)
  for _ in range(10):
    tracker.record_step(tracked_env)
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  # EMA from pessimistic 0 with alpha 0.1 and a perfect episode -> 0.1.
  assert tracker.attain_ema[0].item() == pytest.approx(0.1, abs=0.01)
  ct.vel_command_b = torch.tensor([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
  tracker2 = CompetenceTracker(tracked_env)
  tracker2.record_step(tracked_env)
  assert tracker2._attain_weight.sum().item() == 0.0


def test_core_band_attainment_ignores_commands_past_the_frontier(
  tracked_env: MagicMock,
) -> None:
  """The gate's attainment must not be diluted by commands the curriculum
  issues precisely because they are out of reach. Env 0 is commanded inside
  the core band, env 1 well past it; both deliver half. The full-box mean
  sees both, the gate's core-band mean sees only env 0."""
  env = tracked_env
  command_term = env.command_manager.get_term.return_value
  command_term.vel_command_b = torch.tensor([[0.5, 0.0, 0.0], [1.4, 0.0, 0.0]])
  command_term.robot.data.root_link_lin_vel_b = torch.tensor([[0.25, 0.0], [0.7, 0.0]])
  tracker = CompetenceTracker(env)

  tracker.record_step(env)

  torch.testing.assert_close(tracker._attain_weight, torch.tensor([1.0, 1.0]))
  torch.testing.assert_close(tracker._attain_core_weight, torch.tensor([1.0, 0.0]))
  torch.testing.assert_close(tracker._attain_core_sum, torch.tensor([0.5, 0.0]))

  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  stats = tracker.population_means()
  # Only the in-band env moves the gate's signal off its pessimistic init.
  assert stats["attain_core"] == pytest.approx(0.5 * tracker.ema_alpha / 2)


def test_per_axis_attainment_separates_directions(tracked_env: MagicMock) -> None:
  """A robot that delivers x but not y must read high attain_x, low
  attain_y."""
  tracker = CompetenceTracker(tracked_env)
  ct = tracked_env.command_manager.get_term.return_value
  # Commanded diagonal; delivered: full x, none of y.
  ct.vel_command_b = torch.tensor([[0.4, 0.3, 0.0], [0.4, 0.3, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.4, 0.0], [0.4, 0.0]])
  for _ in range(5):
    tracker.record_step(tracked_env)
  tracked_env.common_step_counter = 48
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  strat = tracker.stratified_means()
  assert strat["clean_attain_x"] > 0.05
  assert abs(strat["clean_attain_y"]) < 0.01
  # EMA direction check on raw tensors (alpha=0.1, one episode).
  assert tracker.attain_x_ema[0].item() == pytest.approx(0.1, rel=0.01)
  assert tracker.attain_y_ema[0].item() == pytest.approx(0.0, abs=1e-6)


def test_tracker_fast_window_tracks_crash(tracked_env: MagicMock) -> None:
  """The fast rate must fall during health and alarm within ~10 iterations
  of a total crash (vs ~200 for the per-env EMAs)."""
  tracker = CompetenceTracker(tracked_env)
  assert tracker.population_means()["fast_fall_rate"] == pytest.approx(1.0)

  step = 24
  for _ in range(30):
    tracked_env.common_step_counter = step
    tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
    step += 24
  assert tracker.population_means()["fast_fall_rate"] < 0.05

  crash_env = _mock_tracked_env(fell=True, ep_len=200)
  for _ in range(10):
    crash_env.common_step_counter = step
    tracker.finalize_episodes(crash_env, torch.tensor([0, 1]))
    step += 24
  assert tracker.population_means()["fast_fall_rate"] > 0.5


def test_cohort_stratified_attribution(tracked_env: MagicMock) -> None:
  """Falls in the pushed cohort must not contaminate clean-cohort stats."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.5)  # env 0 pushed, env 1 clean
  assert tracker.push_cohort.tolist() == [True, False]

  tracked_env.termination_manager.get_term.return_value = torch.tensor([True, False])
  step = 24
  for _ in range(60):
    tracked_env.common_step_counter = step
    tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
    step += 24
  strat = tracker.stratified_means()
  assert strat["fast_fall_pushed"] > 0.9
  assert strat["fast_fall_clean"] < 0.1
  assert strat["push_excess_fall"] > 0.8
  # Per-env EMA stratification follows membership.
  assert strat["pushed_fell_ema"] > 0.9
  assert strat["clean_fell_ema"] < 0.1


##
# Frontier estimator: exposure, censoring, and the interpolated readouts.
##


def test_frontier_buckets_charge_falls_to_speed(tracked_env: MagicMock) -> None:
  """Clean-cohort falls are charged to the commanded-speed bucket."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.5)  # env 1 is clean
  # Command 0.55 m/s -> bin 11 of 64 (0.05 m/s bins).
  ct = tracked_env.command_manager.get_term.return_value
  ct.vel_command_b = torch.tensor([[0.55, 0.0, 0.0], [0.55, 0.0, 0.0]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
  tracker.record_step(tracked_env)
  assert tracker._cur_bucket.tolist() == [11, 11]
  # Only the clean env's exposure counts.
  assert tracker._bucket_steps[11].item() == pytest.approx(1.0)
  tracked_env.termination_manager.get_term.return_value = torch.tensor([True, True])
  tracked_env.common_step_counter = 48
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  # Only env 1 (clean) charges the bin; env 0's fall is push-cohort.
  assert tracker._bucket_falls[11].item() == pytest.approx(1.0)
  assert tracker._bucket_falls.sum().item() == pytest.approx(1.0)


def test_rho_buckets_attribute_corner_falls(tracked_env: MagicMock) -> None:
  """A corner command lands in a high Mahalanobis-radius bucket."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.5)
  ct = tracked_env.command_manager.get_term.return_value
  # Corner command: all axes near max -> rho ~ sqrt(3) ~ 1.69 -> top bin.
  ct.vel_command_b = torch.tensor([[0.74, 0.44, 0.79], [0.74, 0.44, 0.79]])
  ct.robot.data.root_link_lin_vel_b = torch.tensor([[0.3, 0.1], [0.3, 0.1]])
  tracker.record_step(tracked_env)
  assert tracker._cur_rho_bucket.tolist() == [63, 63]
  tracked_env.termination_manager.get_term.return_value = torch.tensor([True, True])
  tracked_env.common_step_counter = 48
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  assert tracker._rho_falls[63].item() == pytest.approx(1.0)  # clean env only


def test_survivor_conditioning_gates_capability_credit(tracked_env: MagicMock) -> None:
  """A sprint that ends in a fall contributes nothing to attain(v); a
  timed-out episode deposits its duration-weighted samples."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.0)  # both envs clean
  b = int(0.5 / tracker.speed_bin_width)
  for _ in range(3):  # duration weighting: three steps buffered
    tracker.record_step(tracked_env)

  # Episode ends in a FALL: buffered credit is discarded.
  tracked_env.termination_manager.get_term.return_value = torch.tensor([True, True])
  tracked_env.common_step_counter = 48
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == pytest.approx(0.0)
  assert tracker._attain_ep_weight.sum().item() == pytest.approx(0.0)

  # Same steps, episode TIMES OUT: credit lands, weighted by duration.
  for _ in range(3):
    tracker.record_step(tracked_env)
  tracked_env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracked_env.common_step_counter = 96
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == pytest.approx(6.0)  # 2 envs x 3
  assert tracker._attain_bin_sum[b].item() == pytest.approx(6.0 * 0.9, rel=1e-3)


def test_attain_credit_requires_settling(tracked_env: MagicMock) -> None:
  """No frontier credit until the command has been held
  ``attain_settle_s`` — the acceleration transient measures 'reached', not
  'maintained'."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.0)  # all envs clean
  tracker._step_dt = 1.0  # settle = 0.75 -> settled from dwell 1
  b = int(0.5 / tracker.speed_bin_width)
  tracker.record_step(tracked_env)  # dwell 0 (bin just entered): excluded
  assert tracker._attain_ep_weight[0, b].item() == 0.0
  tracker.record_step(tracked_env)  # dwell 1 >= 0.75: counts
  tracker.record_step(tracked_env)
  assert tracker._attain_ep_weight[0, b].item() == pytest.approx(2.0)


def test_attain_fold_censors_short_segments(tracked_env: MagicMock) -> None:
  """A survivor episode's bin evidence folds only if it held the bin >=
  ``attain_min_dwell_s`` post-settle — a high-speed command landing just
  before timeout is censored, not credited."""
  tracker = CompetenceTracker(tracked_env)
  tracker.set_push_cohort(0.0)  # all envs clean
  tracker._step_dt = 1.0  # min dwell = 3 steps
  b = int(0.5 / tracker.speed_bin_width)
  for _ in range(3):  # 2 settled steps of credit: below the dwell bar
    tracker.record_step(tracked_env)
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() == 0.0  # censored
  for _ in range(5):  # 4 settled steps: clears the bar
    tracker.record_step(tracked_env)
  tracked_env.common_step_counter = 48  # new step so the finalize guard passes
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  assert tracker._attain_bin_weight[b].item() >= 3.0  # folded


def test_frontier_retreats_during_population_decline(tracked_env: MagicMock) -> None:
  """Unhealthy windows must decay frontier confidence rather than poll the
  surviving elite (whose windows would RAISE the frontier as the
  population dies)."""
  tracker = CompetenceTracker(tracked_env)
  tracker.attain_by_speed[:16] = 0.75
  tracker.attain_by_speed_weight[:16] = 100.0
  tracker._attain_bin_sum[:16] = 90.0 * 0.8  # elite survivors still great
  tracker._attain_bin_weight[:16] = 90.0
  tracker.fast_fall_clean = 0.50  # population dying
  tracker._bucket_next_step = 0
  tracker._bucket_steps[0] = 200.0
  tracked_env.common_step_counter = 5000
  tracked_env.termination_manager.get_term.return_value = torch.tensor([False, False])
  tracker.finalize_episodes(tracked_env, torch.tensor([0, 1]))
  # Curve values unchanged (no fold), confidence decayed.
  assert tracker.attain_by_speed[8].item() == pytest.approx(0.75)
  assert tracker.attain_by_speed_weight[8].item() == pytest.approx(50.0)


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
  """The dt histogram is windowed like the survival bins: a window with >=
  50 events folds 0.3/0.7 into the EMA and clears; sparse windows keep
  accumulating."""
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
  # shape and the old mode decays.
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
  episode, and never via clamping into the top bin."""
  env = _mock_tracked_env(fell=True)
  env.step_dt = 0.02
  tracker = CompetenceTracker(env)
  tracker.set_push_cohort(1.0)
  # Fall with no push ever: nothing recorded.
  env.common_step_counter = 500
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(0.0)
  # Push, episode ends WITHOUT a fall, next episode falls unpushed: the
  # stale stamp must not attribute the new fall to the old push.
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
  # ...while an impossible dt (> episode length) is dropped, never clamped
  # into the top bin.
  env.common_step_counter = 4000
  tracker.record_push(env, torch.tensor([0, 1]))
  env.common_step_counter = 4000 + int(21.0 / 0.02)
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker._push_fall_dt_win.sum().item() == pytest.approx(2.0)


def test_push_survival_censoring(tracked_env: MagicMock) -> None:
  """A hard push followed quickly by an easy push then a fall must NOT
  credit the hard push (censored) and must charge the easy one; a push
  shortly before timeout is censored, not survived; a full clean window
  earns survival."""
  env = tracked_env
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
  """The observation window follows the measured push->fall t75 (clamped),
  and holds the bootstrap value until evidence accumulates."""
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


def test_push_cohort_event_filters_and_stamps(tracked_env: MagicMock) -> None:
  tracked_env._competence_tracker = CompetenceTracker(tracked_env)
  robot = MagicMock()
  robot.data.root_link_vel_w = torch.zeros(2, 6)
  writes: list[tuple[torch.Tensor, torch.Tensor]] = []
  robot.write_root_link_velocity_to_sim = lambda vel, env_ids: writes.append(
    (vel, env_ids)
  )
  tracked_env.scene = {"robot": robot}
  tracked_env.common_step_counter = 240
  push_cohort_by_setting_velocity(
    tracked_env,
    torch.tensor([0, 1]),
    {"x": (0.3, 0.3), "y": (0.0, 0.0)},
    cohort_frac=0.5,
  )
  # Only env 0 (cohort) is pushed, with the sampled magnitude recorded.
  assert len(writes) == 1
  assert writes[0][1].tolist() == [0]
  tracker = tracked_env._competence_tracker
  assert tracker.last_push_step[0].item() == pytest.approx(240.0)
  assert tracker.last_push_step[1].item() < 0
  assert tracker._pending_push_mag[0].item() == pytest.approx(0.3, rel=1e-5)
  assert tracker._pending_push_mag[1].item() == pytest.approx(-1.0)


def test_interp_crossing_and_binned_quantile() -> None:
  """The diagnostics consume interpolated statistics, not raw buckets."""
  # Linear hazard ramp: h(bin i) = i * 1e-4 over 0.025-wide bins.
  hazards = torch.arange(32, dtype=torch.float32) * 1e-4
  bar = 5e-4
  x = _interp_crossing(hazards, 0.025, bar)
  # Crossing near bin 5 (h=5e-4), interpolated — continuous, not a
  # quantized stair step.
  assert 0.10 < x < 0.16
  # Never crossed, no exposure info: returns full range.
  assert _interp_crossing(torch.zeros(32), 0.025, bar) == pytest.approx(0.8)
  # Never crossed WITH exposure: clamps to the highest sampled bin —
  # "clean as far as we have sampled", not an instrument-range capability
  # claim.
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


def test_attained_frontier_graded_bar() -> None:
  """The graded per-bin bar (min(bar, 1 - abs_tol/v)) must not fail
  low-speed bins that a policy tracks as well as physics allows."""
  curve = torch.zeros(32)
  w = torch.zeros(32)
  curve[3] = 0.50  # ~0.14 m/s at 0.50 attain: graded bar ~0.43 passes
  w[3] = 100.0
  curve[4] = 0.64  # ~0.18 m/s: passes graded bar ~0.56
  w[4] = 100.0
  assert _attained_frontier(curve, w, 0.04, 0.60) > 0.15
  # No bin carries enough weight to testify -> nothing demonstrated.
  assert _attained_frontier(curve, torch.zeros(32), 0.04, 0.60) == 0.0


def test_evidence_masked_histogram_views() -> None:
  """Histogram views must not render priors/noise from unmeasured bins."""
  # push_survival-style buffer: 1.0 prior everywhere, evidence in bins 0..3
  # only. The prior wall above the data must render as zero.
  survival = torch.ones(8)
  survival[2] = 0.6
  weight = torch.tensor([5.0, 4.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0])
  masked = _evidence_masked(survival, weight)
  assert torch.equal(masked[:4], survival[:4])
  assert torch.all(masked[4:] == 0.0)

  # hazard-style buffer: a near-empty bin (1 fall / 2 steps) spikes to 0.5
  # next to well-sampled bins at ~1e-4; the spike must not render.
  hazard = torch.tensor([1e-4, 2e-4, 0.5, 0.0])
  exposure = torch.tensor([5000.0, 5000.0, 0.6, 0.0])
  masked = _evidence_masked(hazard, exposure)
  assert torch.equal(masked, torch.tensor([1e-4, 2e-4, 0.0, 0.0]))


##
# staged_on_competence: promote / demote / apply.
##


def test_staged_on_competence_promotes_demotes_and_freezes(
  tracked_env: MagicMock,
) -> None:
  """The penalty gate must advance under demonstrated stability and BACK
  OFF a stage when it is badly lost — a freeze alone cannot recover a
  policy already sliding down the penalty gradient."""
  stages: list[RewardCurriculumStage] = [
    {"step": i, "weight": -0.001 * i} for i in range(5)
  ]
  env = tracked_env
  # MagicMock auto-creates attributes, so get_competence_tracker(env) would
  # return a mock tracker; install a real one explicitly.
  env._competence_tracker = CompetenceTracker(env)
  env.reward_manager.get_term_cfg.return_value = MagicMock()

  cfg = MagicMock()
  cfg.params = {"reward_name": "torque_rate", "stages": stages}
  term = staged_on_competence(cfg, env)
  tracker = get_competence_tracker(env)

  def run(iteration: int) -> dict[str, torch.Tensor]:
    env.common_step_counter = iteration * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    return term(env, torch.tensor([0, 1]), "torque_rate", stages)

  # Drive to stage 2 via two stable windows spaced by the 150-iteration
  # cooldown. The fast fall channel starts at its pessimistic init (1.0)
  # and must be marked healthy like the slow EMAs or it vetoes promotion.
  tracker.fast_fall_rate = 0.05

  def healthy_run(iteration: int) -> dict[str, torch.Tensor]:
    tracker.fell_ema[:] = 0.1
    tracker.ep_len_frac_ema[:] = 0.95
    # Core-band attainment is what the gate reads. The full-box mean is
    # held BELOW every bar throughout: a limit-pushing command curriculum
    # keeps it there permanently, and the gate must not stall on it.
    tracker.attain_core_ema[:] = 0.9
    tracker.attain_ema[:] = 0.35
    tracker.wobble_ema[:] = 0.02
    return run(iteration)

  healthy_run(200)
  snap = healthy_run(400)
  assert snap["stage_idx"].item() == 2
  assert snap["weight"].item() == pytest.approx(-0.002)

  # Stability badly lost -> demote one stage. Falls stay LOW: the loss
  # shows up as sandbagging (attainment collapse), because the policy
  # prefers a stable stand over risking falls.
  tracker.fell_ema[:] = 0.2
  tracker.attain_core_ema[:] = 0.25
  tracker.ep_len_frac_ema[:] = 0.9
  snap = run(600)
  assert snap["stage_idx"].item() == 1
  assert snap["weight"].item() == pytest.approx(-0.001)

  # Freeze band (between thresholds): holds, no further demote.
  tracker.fell_ema[:] = 0.2
  tracker.attain_core_ema[:] = 0.55
  tracker.wobble_ema[:] = 0.08
  snap = run(800)
  assert snap["stage_idx"].item() == 1


def test_staged_on_competence_respects_cooldown(tracked_env: MagicMock) -> None:
  """Two stable evaluations inside one cooldown window promote once."""
  stages: list[RewardCurriculumStage] = [
    {"step": i, "weight": -0.001 * i} for i in range(5)
  ]
  env = tracked_env
  env._competence_tracker = CompetenceTracker(env)
  env.reward_manager.get_term_cfg.return_value = MagicMock()
  cfg = MagicMock()
  cfg.params = {"reward_name": "torque_rate", "stages": stages}
  term = staged_on_competence(cfg, env)
  tracker = get_competence_tracker(env)
  tracker.fast_fall_rate = 0.05
  tracker.fell_ema[:] = 0.1
  tracker.ep_len_frac_ema[:] = 0.95
  tracker.attain_core_ema[:] = 0.9
  tracker.wobble_ema[:] = 0.02

  def run(iteration: int) -> dict[str, torch.Tensor]:
    env.common_step_counter = iteration * _NUM_STEPS_PER_ENV
    tracker._finalized_step = -1
    return term(env, torch.tensor([0, 1]), "torque_rate", stages)

  run(200)
  run(210)
  assert run(220)["stage_idx"].item() == 1


##
# Wiring: the NUgus config must install the feature and pass only params
# the term signatures accept.
##


def test_competence_wiring_is_always_on() -> None:
  from mjlab.tasks.velocity import mdp

  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.events["competence_tracker"].func is mdp.competence_tracker_step
  assert cfg.events["push_robot"].func is mdp.push_cohort_by_setting_velocity
  assert "competence_diagnostics" in cfg.curriculum
  for reward_name in (
    "joule_heating",
    "joint_acc_l2",
    "torque_rate",
    "soft_landing",
  ):
    term = cfg.curriculum[f"{reward_name}_competence"]
    assert term.func is mdp.staged_on_competence
    assert term.params["reward_name"] == reward_name
    stages = term.params["stages"]
    # Ramp starts at zero pressure and reaches the term's peak weight.
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] < 0.0
    assert reward_name in cfg.rewards


def test_curriculum_terms_accept_manager_kwargs() -> None:
  """The curriculum manager calls ``func(env, env_ids, **params)``, so
  every key wired in env_cfgs must be accepted by the term's signature."""
  cfg = nubots_nugus_flat_env_cfg()
  for name, term in cfg.curriculum.items():
    if "competence" not in name:
      continue
    sig = set(inspect.signature(term.func.__call__).parameters)
    missing = set(term.params) - sig
    assert not missing, f"{name} passes params the term rejects: {missing}"
