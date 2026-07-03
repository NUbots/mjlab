"""Tests for v10 Nugus pipeline knobs: CRITIC_HEIGHT_SCAN and hard_continue."""

from __future__ import annotations

import os
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.tasks.velocity.config.nugus.env_cfgs import (
  _hard_continue_velocity_stages,
  nubots_nugus_flat_env_cfg,
  nubots_nugus_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp.curriculums import (
  PushRobotStage,
  VelocityStage,
  commands_vel,
  push_robot_curriculum,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_NUM_STEPS_PER_ENV = 24
_PHASE_ITERATIONS = 2000
_CONT_BASE = _PHASE_ITERATIONS * _NUM_STEPS_PER_ENV


@pytest.fixture(autouse=True)
def _clear_nugus_env(monkeypatch: pytest.MonkeyPatch) -> None:
  for key in (
    "CRITIC_HEIGHT_SCAN",
    "TRAINING_REGIME",
    "RESUME",
    "CONT_BASE_STEP",
    "HARD_COMPONENTS",
    "PHASE_ITERATIONS",
    "MAX_ITERATIONS",
    "UPRIGHT_W",
    "PROGRESS_BACKSLIDE_W",
    "SWING_TARGET_HEIGHT",
    "FLATTEN_PHASE_C",
    "LINK_MASS_SCALE_MIN",
    "LINK_MASS_SCALE_MAX",
    "PAYLOAD_KG_MIN",
    "PAYLOAD_KG_MAX",
  ):
    monkeypatch.delenv(key, raising=False)


def test_progress_backslide_default_weight_enabled() -> None:
  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.rewards["command_progress_backslide"].weight == -0.5


def test_progress_backslide_weight_env_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("PROGRESS_BACKSLIDE_W", "-1.25")
  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.rewards["command_progress_backslide"].weight == -1.25


def test_critic_height_scan_off_flat_critic_lacks_height_scan() -> None:
  cfg = nubots_nugus_flat_env_cfg()
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms
  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  assert "terrain_scan" not in sensor_names


def test_critic_height_scan_on_retains_height_scan_and_terrain_scan() -> None:
  os.environ["CRITIC_HEIGHT_SCAN"] = "true"
  flat_cfg = nubots_nugus_flat_env_cfg()
  rough_cfg = nubots_nugus_rough_env_cfg()
  for cfg in (flat_cfg, rough_cfg):
    assert "height_scan" not in cfg.observations["actor"].terms
    assert "height_scan" in cfg.observations["critic"].terms
  flat_sensor_names = {sensor.name for sensor in flat_cfg.scene.sensors or ()}
  assert "terrain_scan" in flat_sensor_names


def test_hard_continue_stages_anchor_at_resume_base(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("RESUME", "true")
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))
  cfg = nubots_nugus_flat_env_cfg()
  velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  push_stages = cfg.curriculum["push_robot_ramp"].params["push_stages"]
  assert velocity_stages[0]["step"] == _CONT_BASE
  assert velocity_stages[-1]["step"] == _CONT_BASE + 1000 * _NUM_STEPS_PER_ENV
  assert push_stages[0]["step"] == _CONT_BASE
  assert push_stages[-1]["step"] == _CONT_BASE + 1000 * _NUM_STEPS_PER_ENV
  assert velocity_stages[-1]["lin_vel_x"] == (-0.75, 0.75)
  assert velocity_stages[0]["lin_vel_x"] == (-0.5, 0.5)
  assert velocity_stages[0]["lin_vel_y"] == (-0.3, 0.3)
  assert velocity_stages[0]["ang_vel_z"] == (-0.5, 0.5)
  upright_stages = cfg.curriculum["upright_ramp"].params["stages"]
  assert upright_stages[0]["step"] == _CONT_BASE
  assert upright_stages[-1]["weight"] == 0.25
  assert upright_stages[-1]["params"]["std"] == 0.35


def test_hard_continue_cont_base_override(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("RESUME", "true")
  monkeypatch.setenv("CONT_BASE_STEP", "12345")
  cfg = nubots_nugus_flat_env_cfg()
  velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages[0]["step"] == 12345


def test_hard_continue_fresh_base_hard_merges_command_ramp_at_cont_base(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Fresh base→hard keeps base command_vel stages and appends hard ramp."""
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("RESUME", "false")
  monkeypatch.setenv("CONT_BASE_STEP", str(_CONT_BASE))
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))
  cfg = nubots_nugus_flat_env_cfg()
  base_stages = (
    make_velocity_env_cfg().curriculum["command_vel"].params["velocity_stages"]
  )
  velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages[: len(base_stages)] == base_stages
  assert velocity_stages[len(base_stages)]["step"] == _CONT_BASE
  assert "push_robot_ramp" in cfg.curriculum
  push_stages = cfg.curriculum["push_robot_ramp"].params["push_stages"]
  assert push_stages[0]["step"] == _CONT_BASE


def test_base_then_hard_alias_matches_hard_continue(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "base_then_hard")
  monkeypatch.setenv("RESUME", "false")
  monkeypatch.setenv("CONT_BASE_STEP", str(_CONT_BASE))
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))
  cfg = nubots_nugus_flat_env_cfg()
  assert "push_robot_ramp" in cfg.curriculum


def test_hard_continue_absent_in_base_regime(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "base")
  cfg = nubots_nugus_flat_env_cfg()
  assert "push_robot_ramp" not in cfg.curriculum


def test_push_robot_curriculum_applies_staged_velocity_range() -> None:
  push_stages = [
    {
      "step": 0,
      "params": {
        "velocity_range": {
          "x": (-0.2, 0.4),
          "y": (-0.2, 0.2),
          "z": (-0.0, 0.0),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.0, 0.0),
        },
      },
    },
    {
      "step": 6000,
      "params": {
        "velocity_range": {
          "x": (-0.4, 0.8),
          "y": (-0.4, 0.4),
          "z": (-0.0, 0.0),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-0.0, 0.0),
        },
      },
    },
  ]
  term_cfg = MagicMock()
  term_cfg.params = {
    "velocity_range": {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    },
  }
  env = MagicMock()
  env.common_step_counter = 6000
  env.event_manager.get_term_cfg = MagicMock(return_value=term_cfg)
  push_robot_curriculum(
    env,
    env_ids=torch.tensor([0]),
    event_name="push_robot",
    push_stages=cast(list[PushRobotStage], push_stages),
  )
  assert term_cfg.params["velocity_range"]["x"] == (-0.4, 0.8)


def test_hard_continue_stage0_matches_v9_terminal_at_cont_base(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """At cont_base, hard_continue stage 0 should match base curriculum terminal."""
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("RESUME", "true")
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))

  hard_stages = _hard_continue_velocity_stages(_CONT_BASE)
  base_stages = (
    make_velocity_env_cfg().curriculum["command_vel"].params["velocity_stages"]
  )

  hard_env, hard_term = _make_command_curriculum_env(_CONT_BASE)
  commands_vel(
    hard_env,
    env_ids=torch.tensor([0]),
    command_name="twist",
    velocity_stages=cast(list[VelocityStage], hard_stages),
  )

  base_env, base_term = _make_command_curriculum_env(_CONT_BASE)
  commands_vel(
    base_env,
    env_ids=torch.tensor([0]),
    command_name="twist",
    velocity_stages=cast(list[VelocityStage], base_stages),
  )

  assert hard_term.cfg.ranges.lin_vel_x == base_term.cfg.ranges.lin_vel_x
  assert hard_term.cfg.ranges.lin_vel_y == base_term.cfg.ranges.lin_vel_y
  assert hard_term.cfg.ranges.ang_vel_z == base_term.cfg.ranges.ang_vel_z
  assert hard_term.cfg.ranges.lin_vel_x == (-0.5, 0.5)
  assert hard_term.cfg.ranges.lin_vel_y == (-0.3, 0.3)
  assert hard_term.cfg.ranges.ang_vel_z == (-0.5, 0.5)


def _make_command_curriculum_env(step_counter: int) -> tuple[MagicMock, MagicMock]:
  ranges = MagicMock()
  ranges.lin_vel_x = (0.0, 0.0)
  ranges.lin_vel_y = (0.0, 0.0)
  ranges.ang_vel_z = (0.0, 0.0)
  term = MagicMock()
  term.cfg.ranges = ranges
  env = MagicMock()
  env.common_step_counter = step_counter
  env.command_manager.get_term = MagicMock(return_value=term)
  return env, term


def test_hard_components_commands_only_adds_command_ramp(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("CONT_BASE_STEP", str(_CONT_BASE))
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))
  monkeypatch.setenv("HARD_COMPONENTS", "commands")
  cfg = nubots_nugus_flat_env_cfg()
  assert "push_robot_ramp" not in cfg.curriculum
  assert "upright_ramp" not in cfg.curriculum
  assert "joule_heating_rampup" not in cfg.curriculum
  base_stages = (
    make_velocity_env_cfg().curriculum["command_vel"].params["velocity_stages"]
  )
  velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages[: len(base_stages)] == base_stages
  assert velocity_stages[len(base_stages)]["step"] == _CONT_BASE
  assert velocity_stages[-1]["lin_vel_x"] == (-0.75, 0.75)


def test_hard_components_phasec_off_skips_phase_c_ramps(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "hard_continue")
  monkeypatch.setenv("CONT_BASE_STEP", str(_CONT_BASE))
  monkeypatch.setenv("PHASE_ITERATIONS", str(_PHASE_ITERATIONS))
  monkeypatch.setenv("HARD_COMPONENTS", "commands,pushes,upright")
  cfg = nubots_nugus_flat_env_cfg()
  assert "joule_heating_rampup" not in cfg.curriculum
  assert cfg.rewards["joule_heating"].weight == 0.0


def test_hard_from_start_applies_static_final_values(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "hard_from_start")
  monkeypatch.setenv("UPRIGHT_W", "0.25")
  monkeypatch.setenv("JOULE_W", "1e-5")
  cfg = nubots_nugus_flat_env_cfg()
  assert "push_robot_ramp" not in cfg.curriculum
  assert "upright_ramp" not in cfg.curriculum
  assert "joule_heating_rampup" not in cfg.curriculum
  velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages == [
    {
      "step": 0,
      "lin_vel_x": (-0.75, 0.75),
      "lin_vel_y": (-0.45, 0.45),
      "ang_vel_z": (-0.80, 0.80),
    }
  ]
  assert cfg.events["push_robot"].params["velocity_range"]["x"] == (-0.4, 0.8)
  assert cfg.rewards["upright"].weight == 0.25
  assert cfg.rewards["upright"].params["std"] == 0.35
  assert cfg.rewards["joule_heating"].weight == -1e-5
  assert cfg.rewards["joint_acc_l2"].weight == -1e-4
  assert cfg.rewards["foot_swing_height"].weight == 0.75
  assert "clock_anneal" in cfg.curriculum


def test_flatten_phase_c_skips_phase_c_curriculum(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("TRAINING_REGIME", "base")
  monkeypatch.setenv("FLATTEN_PHASE_C", "1")
  monkeypatch.setenv("JOULE_W", "1e-5")
  cfg = nubots_nugus_flat_env_cfg()
  assert "joule_heating_rampup" not in cfg.curriculum
  assert cfg.rewards["joule_heating"].weight == -1e-5
  assert cfg.rewards["joint_acc_l2"].weight == -1e-4
  assert cfg.rewards["torque_rate"].weight == -1e-3
  assert cfg.rewards["soft_landing"].weight == -0.01
  assert cfg.rewards["base_height"].weight == 0.3


def test_swing_target_height_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SWING_TARGET_HEIGHT", "0.05")
  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.05
  assert cfg.rewards["foot_swing_height_landing"].params["target_height"] == 0.05


def test_v16b_dr_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("LINK_MASS_SCALE_MIN", "0.90")
  monkeypatch.setenv("LINK_MASS_SCALE_MAX", "1.10")
  monkeypatch.setenv("PAYLOAD_KG_MIN", "-0.2")
  monkeypatch.setenv("PAYLOAD_KG_MAX", "0.2")
  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.events["link_mass"].params["ranges"] == (0.90, 1.10)
  assert cfg.events["payload"].params["ranges"] == (-0.2, 0.2)


def test_vel_sat_frac_event_registered() -> None:
  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.events["vel_sat_frac"].mode == "step"
