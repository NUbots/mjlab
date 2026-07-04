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
      fell_ema=0.4,
      ep_len_frac=0.9,
      common_step_counter=50 * _NUM_STEPS_PER_ENV,
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
  command_term.vel_command_b = torch.tensor([[0.0, 0.0], [0.5, 0.0]])
  command_term.robot.data.root_link_lin_vel_b = torch.tensor([[0.0, 0.0], [0.4, 0.0]])
  env.command_manager.get_term.return_value = command_term

  tracker = CompetenceTracker(env)
  tracker.record_step(env)
  assert tracker._track_weight[0].item() == 0.0
  assert tracker._track_weight[1].item() == 1.0
  tracker.finalize_episodes(env, torch.tensor([0, 1]))
  assert tracker.fell_ema[1].item() > tracker.fell_ema[0].item()
