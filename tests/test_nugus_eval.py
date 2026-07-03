"""Tests for the NUgus fixed evaluation harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_nugus_eval():
  module_name = "nugus_eval"
  spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / "nugus_eval.py")
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def nugus_eval():
  return _load_nugus_eval()


@pytest.fixture(autouse=True)
def _clear_nugus_env(monkeypatch: pytest.MonkeyPatch) -> None:
  for key in (
    "CRITIC_HEIGHT_SCAN",
    "TRAINING_REGIME",
    "RESUME",
    "CONT_BASE_STEP",
    "PHASE_ITERATIONS",
    "MAX_ITERATIONS",
    "MJLAB_VARIANT",
    "SEED",
  ):
    monkeypatch.delenv(key, raising=False)


def test_command_label_formats():
  mod = _load_nugus_eval()
  assert mod.command_label((0.3, 0.0, 0.0)) == "0p3_0_0"
  assert mod.command_label((0.0, -0.3, 0.0)) == "0_m0p3_0"
  assert mod.command_label((0.5, 0.3, 0.5)) == "0p5_0p3_0p5"


def test_env_group_ids_assigns_command_blocks():
  mod = _load_nugus_eval()
  groups = mod.env_group_ids(768, envs_per_command=256, device="cpu")
  assert groups[:256].eq(0).all()
  assert groups[256:512].eq(1).all()
  assert groups[512:768].eq(2).all()


def test_build_eval_env_cfg_keeps_pushes_and_disables_resampling(nugus_eval):
  cfg = nugus_eval.build_eval_env_cfg(seed=7, episode_length_s=30.0, num_envs=2560)
  assert cfg.seed == 7
  assert cfg.episode_length_s == 30.0
  assert cfg.scene.num_envs == 2560
  assert "push_robot" in cfg.events
  assert cfg.curriculum == {}
  twist = cfg.commands["twist"]
  assert twist.resampling_time_range == (1e9, 1e9)
  assert twist.rel_stop_envs == 0.0


def test_build_eval_env_cfg_rejects_too_few_envs(nugus_eval):
  with pytest.raises(ValueError, match="below"):
    nugus_eval.build_eval_env_cfg(num_envs=128, envs_per_command=256)


def test_apply_fixed_commands_pins_grid(nugus_eval):
  from unittest.mock import MagicMock

  from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
  )

  twist_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
    ),
  )
  env = MagicMock()
  env.device = torch.device("cpu")
  env.num_envs = 2560
  env.scene = {"robot": MagicMock()}
  twist = UniformVelocityCommand(twist_cfg, env)

  nugus_eval.apply_fixed_commands(twist, envs_per_command=256)
  assert torch.allclose(twist.vel_command_b[:256, 0], torch.full((256,), 0.3))
  assert torch.allclose(twist.vel_command_b[256:512, 0], torch.full((256,), 0.5))
  assert twist.is_standing_env[:256].eq(False).all()
  standing_block = twist.vel_command_b[8 * 256 : 9 * 256]
  assert torch.allclose(standing_block, torch.zeros_like(standing_block))
  assert twist.is_standing_env[8 * 256 : 9 * 256].all()
  combo_block = twist.vel_command_b[9 * 256 : 10 * 256]
  assert torch.allclose(combo_block[:, 0], torch.full((256,), 0.5))
  assert torch.allclose(combo_block[:, 1], torch.full((256,), 0.3))
  assert torch.allclose(combo_block[:, 2], torch.full((256,), 0.5))


def test_accumulator_to_metrics_computes_rmse_and_falls_per_min(nugus_eval):
  acc = nugus_eval._StepAccumulator(
    lin_sq_sum=2.0,
    lin_count=2,
    ang_sq_sum=0.5,
    ang_count=2,
    slip_sum=0.4,
    slip_count=2,
    swing_err_sum=0.02,
    swing_count=1,
    fall_count=2,
    ep_len_sum_s=120.0,
    ep_count=4,
  )
  metrics = nugus_eval._accumulator_to_metrics(acc)
  assert metrics["eval/lin_vel_rmse"] == pytest.approx(1.0)
  assert metrics["eval/ang_vel_rmse"] == pytest.approx(0.5)
  assert metrics["eval/falls_per_min"] == pytest.approx(1.0)
  assert metrics["eval/mean_ep_len_s"] == pytest.approx(30.0)
  assert metrics["eval/slip_vel"] == pytest.approx(0.2)
  assert metrics["eval/swing_height_err"] == pytest.approx(0.02)


def test_eval_metrics_state_flattens_per_command_keys(nugus_eval):
  state = nugus_eval.EvalMetricsState()
  state.per_command[0].lin_sq_sum = 1.0
  state.per_command[0].lin_count = 1
  flat = state.to_flat_dict()
  label = nugus_eval.command_label(nugus_eval.COMMAND_GRID[0])
  assert flat[f"eval/lin_vel_rmse/cmd_{label}"] == pytest.approx(1.0)
