"""Scripted saccadic head action and physical joule energy term."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mjlab.envs.mdp.actions.scripted_head import (
  ScriptedHeadAction,
  ScriptedHeadActionCfg,
)


def _make_head_env(num_envs=4, step_dt=0.02):
  env = MagicMock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.step_dt = step_dt
  env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
  entity = MagicMock()
  entity.find_joints = MagicMock(return_value=([12, 13], ["neck_yaw", "head_pitch"]))
  entity.data.default_joint_pos = torch.zeros(num_envs, 20)
  limits = torch.zeros(num_envs, 20, 2)
  limits[..., 0] = -0.5
  limits[..., 1] = 0.5
  entity.data.soft_joint_pos_limits = limits
  entity.set_joint_position_target = MagicMock()
  env.scene = MagicMock()
  env.scene.__getitem__ = MagicMock(return_value=entity)
  return env, entity


def _build(env, **kwargs):
  cfg = ScriptedHeadActionCfg(
    entity_name="robot", joint_names=("neck_yaw", "head_pitch"), **kwargs
  )
  return ScriptedHeadAction(cfg, env)


def test_scripted_head_zero_action_dim():
  env, _ = _make_head_env()
  term = _build(env)
  assert term.action_dim == 0
  assert term.raw_action.shape == (4, 0)
  # process_actions accepts an empty slice without error.
  term.process_actions(torch.zeros(4, 0))


def test_scripted_head_targets_within_limits():
  env, entity = _make_head_env()
  term = _build(env, target_frac=1.0)
  term.apply_actions()
  target = entity.set_joint_position_target.call_args[0][0]
  assert torch.all(target >= -0.5) and torch.all(target <= 0.5)
  assert target.shape == (4, 2)


def test_scripted_head_saccades_hold_then_jump():
  # Dwell fixed at 5 steps (0.1 s at dt=0.02): target holds, then jumps.
  env, entity = _make_head_env()
  term = _build(env, dwell_range_s=(0.1, 0.1))
  seen = []
  for step in range(20):
    env.episode_length_buf = torch.full((4,), step, dtype=torch.long)
    term.apply_actions()
    seen.append(term._target[0].clone())
  T = torch.stack(seen)
  changes = int(((T[1:] - T[:-1]).abs().sum(dim=1) > 1e-6).sum())
  # 20 steps / 5-step dwell => ~3-4 saccades; certainly holds (not every step).
  assert 2 <= changes <= 5


def test_scripted_head_resamples_on_reset():
  env, _ = _make_head_env()
  term = _build(env, dwell_range_s=(10.0, 10.0))  # long dwell: stable target
  before = term._target.clone()
  term.reset(torch.tensor([0, 1]))
  # Reset envs get fresh targets; untouched envs keep theirs.
  assert not torch.equal(term._target[0], before[0]) or not torch.equal(
    term._target[1], before[1]
  )
  assert torch.equal(term._target[2], before[2])
  assert torch.equal(term._target[3], before[3])


def test_joule_electrical_upweights_small_kt():
  # (tau/Kt)^2 with arm Kt < leg Kt gives arms a larger per-Nm penalty.
  from unittest.mock import patch

  env = MagicMock()
  env.device = "cpu"
  asset = MagicMock()
  # two actuators: idx0 "leg" (Kt 2.68), idx1 "shoulder" (Kt 1.5); equal torque.
  asset.data.actuator_force = torch.tensor([[2.0, 2.0]])
  env.scene.__getitem__ = MagicMock(return_value=asset)
  cfg = MagicMock()
  cfg.name = "robot"
  cfg.actuator_ids = [0, 1]

  kt = {r"shoulder": 1.5, "default": 2.68}
  from mjlab.tasks.velocity.mdp.rewards import joule_heating_electrical

  with patch(
    "mjlab.tasks.velocity.mdp.observations._build_kt_tensor",
    return_value=torch.tensor([[2.68, 1.5]]),
  ):
    out = joule_heating_electrical(env, kt=kt, asset_cfg=cfg)
  expected = (2.0 / 2.68) ** 2 + (2.0 / 1.5) ** 2
  assert torch.allclose(out, torch.tensor([expected]), atol=1e-5)
