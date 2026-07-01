"""Tests for policy-owned gait phase (PhaseDeltaAction, nominal cost, wiring)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch

from mjlab.envs.mdp.actions.phase_delta import PhaseDeltaAction, PhaseDeltaActionCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.observations import gait_clock
from mjlab.tasks.velocity.mdp.rewards import (
  feet_swing_height_clock,
  phase_delta_nominal_cost,
)


def _make_phase_env(
  *,
  num_envs: int = 2,
  step_dt: float = 0.1,
  episode_steps: int | list[int] = 0,
  command: torch.Tensor | None = None,
) -> MagicMock:
  env = MagicMock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.step_dt = step_dt
  if isinstance(episode_steps, int):
    env.episode_length_buf = torch.full((num_envs,), episode_steps, dtype=torch.long)
  else:
    env.episode_length_buf = torch.tensor(episode_steps, dtype=torch.long)
  env.extras = {"log": {}}
  env.scene = MagicMock()
  env.scene.__getitem__ = MagicMock(return_value=MagicMock())
  if command is None:
    command = torch.tensor([[0.5, 0.0, 0.0]] * num_envs)
  env.command_manager.get_command = MagicMock(return_value=command)
  return env


def _make_phase_action(
  env: MagicMock,
  *,
  period: float = 0.7,
  command_threshold: float = 0.05,
) -> PhaseDeltaAction:
  cfg = PhaseDeltaActionCfg(
    entity_name="robot",
    period=period,
    command_name="twist",
    command_threshold=command_threshold,
  )
  action = PhaseDeltaAction(cfg, env)
  env.action_manager.get_term = MagicMock(return_value=action)
  return action


def test_phase_delta_accumulates_and_wraps():
  env = _make_phase_env(num_envs=1)
  action = _make_phase_action(env, period=0.7)
  # raw=1 -> delta = 0.1/0.7; five steps -> phase ~ 5/7
  for _ in range(5):
    action.process_actions(torch.ones(1, 1))
  expected = (5 * 0.1 / 0.7) % 1.0
  assert math.isclose(action.policy_phase[0].item(), expected, abs_tol=1e-6)


def test_phase_delta_resets_on_episode_reset():
  env = _make_phase_env(num_envs=2)
  action = _make_phase_action(env)
  action.process_actions(torch.ones(2, 1))
  assert action.policy_phase[0].item() > 0.0
  action.reset(env_ids=torch.tensor([0]))
  assert action.policy_phase[0].item() == 0.0
  assert action.policy_phase[1].item() > 0.0


def test_phase_delta_standing_gate_zeros_phase():
  env = _make_phase_env(
    num_envs=2,
    command=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
  )
  action = _make_phase_action(env)
  action.process_actions(torch.ones(2, 1))
  assert action.policy_phase[0].item() > 0.0
  assert action.policy_phase[1].item() == 0.0
  assert action.last_delta[1].item() == 0.0


def test_phase_delta_metrics_nominal_ratio():
  env = _make_phase_env(num_envs=1)
  action = _make_phase_action(env, period=0.7)
  action.process_actions(torch.ones(1, 1))
  ratio = env.extras["log"]["Metrics/phase_delta_nominal_ratio_mean"]
  assert math.isclose(ratio.item(), 1.0, rel_tol=1e-5)
  period_eff = env.extras["log"]["Metrics/phase_period_effective_mean"]
  assert math.isclose(period_eff.item(), 0.7, rel_tol=1e-5)


def test_phase_delta_nominal_cost_at_and_off_nominal():
  env = _make_phase_env(num_envs=2)
  action = _make_phase_action(env, period=0.7)
  action.process_actions(torch.tensor([[1.0], [2.0]]))
  cost = phase_delta_nominal_cost(env)
  assert math.isclose(cost[0].item(), 0.0, abs_tol=1e-6)
  assert math.isclose(cost[1].item(), 1.0, abs_tol=1e-6)
  assert "Metrics/phase_delta_nominal_error_mean" in env.extras["log"]


def test_gait_clock_uses_policy_phase():
  env = _make_phase_env(num_envs=1)
  action = _make_phase_action(env, period=0.8)
  action._policy_phase[:] = 0.25
  clock = gait_clock(
    env,
    period=0.8,
    command_name="twist",
    phase_source="policy",
  )
  assert math.isclose(clock[0, 0].item(), 1.0, abs_tol=1e-6)
  assert math.isclose(clock[0, 1].item(), 0.0, abs_tol=1e-6)


def test_swing_height_clock_uses_policy_phase_not_time():
  env = _make_phase_env(num_envs=1, episode_steps=100)
  action = _make_phase_action(env, period=0.8)
  action._policy_phase[:] = 0.0

  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.data.heights = torch.zeros(1, 2)
  env.scene.__getitem__ = MagicMock(return_value=height_sensor)

  reward_at_policy = feet_swing_height_clock(
    env,
    height_sensor_name="foot_height_scan",
    target_height=0.08,
    period=0.8,
    swing_ratio=0.45,
    std=0.05,
    command_name="twist",
    phase_source="policy",
  )

  action._policy_phase[:] = 0.25
  reward_shifted = feet_swing_height_clock(
    env,
    height_sensor_name="foot_height_scan",
    target_height=0.08,
    period=0.8,
    swing_ratio=0.45,
    std=0.05,
    command_name="twist",
    phase_source="policy",
  )

  # Episode time would give phase 100*0.1/0.8 = 12.5 mod 1 = 0.5; policy phase
  # 0.0 vs 0.25 should produce different rewards despite high episode step.
  assert not math.isclose(reward_at_policy.item(), reward_shifted.item(), abs_tol=1e-6)


_NUM_STEPS_PER_ENV = 24


def test_clock_learned_phase_delta_nominal_curriculum(monkeypatch):
  monkeypatch.setenv("MJLAB_VARIANT", "clock_learned")
  monkeypatch.setenv("MAX_ITERATIONS", "20000")
  monkeypatch.setenv("PHASE_ITERATIONS", "20000")

  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_rough_env_cfg

  cfg = nubots_nugus_rough_env_cfg()
  assert cfg.rewards["phase_delta_nominal"].weight == -5.0

  stages = cfg.curriculum["phase_delta_nominal_anneal"].params["stages"]
  assert stages == [
    {"step": 0, "weight": -5.0},
    {"step": 2400, "weight": -0.5},
    {"step": 120000, "weight": -0.2},
    {"step": 288000, "weight": -0.05},
    {"step": 408000, "weight": 0.0},
  ]


def test_clock_learned_phase_delta_nominal_curriculum_short_run(monkeypatch):
  """Grid-search runs use 2k iters with a 1k-iter strong phase."""
  monkeypatch.setenv("MJLAB_VARIANT", "clock_learned")
  monkeypatch.setenv("MAX_ITERATIONS", "2000")
  monkeypatch.setenv("PHASE_ITERATIONS", "2000")
  monkeypatch.setenv("PHASE_C_FRAC", "0.5")
  monkeypatch.setenv("PHASE_DELTA_STRONG_ITERS", "1000")

  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_rough_env_cfg

  cfg = nubots_nugus_rough_env_cfg()
  stages = cfg.curriculum["phase_delta_nominal_anneal"].params["stages"]
  assert stages == [
    {"step": 0, "weight": -5.0},
    {"step": 24000, "weight": -0.05},
    {"step": 40800, "weight": 0.0},
  ]


def test_clock_learned_phase_delta_strong_env_knobs(monkeypatch):
  monkeypatch.setenv("MJLAB_VARIANT", "clock_learned")
  monkeypatch.setenv("PHASE_DELTA_STRONG_W", "-8.0")
  monkeypatch.setenv("PHASE_DELTA_STRONG_ITERS", "50")

  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_rough_env_cfg

  cfg = nubots_nugus_rough_env_cfg()
  assert cfg.rewards["phase_delta_nominal"].weight == -8.0

  stages = cfg.curriculum["phase_delta_nominal_anneal"].params["stages"]
  assert stages[0] == {"step": 0, "weight": -8.0}
  assert stages[1] == {"step": 1200, "weight": -0.5}


def test_nugus_upright_weight_env_knob(monkeypatch):
  monkeypatch.setenv("UPRIGHT_W", "0.5")

  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_rough_env_cfg

  cfg = nubots_nugus_rough_env_cfg()
  assert cfg.rewards["upright"].weight == 0.5


def test_nugus_empty_env_vars_use_defaults(monkeypatch):
  """Blank K8s env entries must not override factory defaults."""
  monkeypatch.setenv("MJLAB_VARIANT", "clock_learned")
  for name in (
    "PHASE_DELTA_STRONG_W",
    "PHASE_DELTA_STRONG_ITERS",
    "UPRIGHT_W",
    "JOULE_W",
    "MAX_ITERATIONS",
    "PHASE_ITERATIONS",
    "SILENCE_CLOCK",
    "CURRENT_OBS",
  ):
    monkeypatch.setenv(name, "")

  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_rough_env_cfg

  cfg = nubots_nugus_rough_env_cfg()
  assert cfg.rewards["phase_delta_nominal"].weight == -5.0
  assert cfg.rewards["upright"].weight == 1.0
