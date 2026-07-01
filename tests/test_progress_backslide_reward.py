"""Tests for command_progress_backslide anti-rocking reward."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import command_progress_backslide


def _make_backslide_env(
  *,
  num_envs: int,
  pos_xy: torch.Tensor,
  command: torch.Tensor,
  command_counter: torch.Tensor | None = None,
):
  asset = MagicMock()
  type(asset).data = PropertyMock(
    return_value=SimpleNamespace(
      root_link_pos_w=torch.cat([pos_xy, torch.zeros(num_envs, 1)], dim=1)
    )
  )

  cmd_term = MagicMock()
  cmd_term.command_counter = (
    command_counter
    if command_counter is not None
    else torch.ones(num_envs, dtype=torch.long)
  )

  env = MagicMock()
  env.device = torch.device("cpu")
  env.num_envs = num_envs
  env.scene.__getitem__ = MagicMock(return_value=asset)
  env.command_manager.get_command = MagicMock(return_value=command)
  env.command_manager.get_term = MagicMock(return_value=cmd_term)
  return env, cmd_term


def _make_term(env, num_envs: int = 1) -> command_progress_backslide:
  cfg = RewardTermCfg(
    func=command_progress_backslide,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "twist",
    },
  )
  mock_env = MagicMock()
  mock_env.num_envs = num_envs
  mock_env.device = torch.device("cpu")
  return command_progress_backslide(cfg, mock_env)


def test_backslide_zero_for_zero_command():
  env, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=torch.tensor([[0.0, 0.0]]),
    command=torch.tensor([[0.0, 0.0, 0.0]]),
  )
  term = _make_term(env)
  cost = term(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    stall_steps=10,
    stall_penalty=2.0,
  )
  assert cost[0].item() == pytest.approx(0.0)


def test_backslide_zero_for_near_zero_command():
  env, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=torch.tensor([[0.0, 0.0]]),
    command=torch.tensor([[0.02, 0.01, 0.0]]),
  )
  term = _make_term(env)
  cost = term(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    stall_steps=5,
    stall_penalty=3.0,
  )
  assert cost[0].item() == pytest.approx(0.0)


def test_monotonic_forward_no_backslide_penalty():
  term = _make_term(MagicMock(), num_envs=1)
  origin = torch.tensor([[0.0, 0.0]])
  command = torch.tensor([[0.5, 0.0, 0.0]])
  env, cmd_term = _make_backslide_env(
    num_envs=1, pos_xy=origin, command=command, command_counter=torch.tensor([1])
  )
  cost0 = term(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    deadband=0.02,
  )
  advanced = torch.tensor([[0.2, 0.0]])
  env, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=advanced,
    command=command,
    command_counter=torch.tensor([1]),
  )
  term._prev_command_counter = torch.tensor([1])
  term._segment_active = torch.tensor([True])
  term._origin_xy = origin.clone()
  cost1 = term(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    deadband=0.02,
  )
  assert cost0[0].item() == pytest.approx(0.0)
  assert cost1[0].item() == pytest.approx(0.0)


def test_backslide_penalizes_large_reverse_from_peak():
  term = _make_term(MagicMock(), num_envs=1)
  origin = torch.tensor([[0.0, 0.0]])
  command = torch.tensor([[1.0, 0.0, 0.0]])
  peak_pos = torch.tensor([[0.3, 0.0]])
  back_pos = torch.tensor([[0.05, 0.0]])

  env_peak, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=peak_pos,
    command=command,
    command_counter=torch.tensor([1]),
  )
  term._prev_command_counter = torch.tensor([1])
  term._segment_active = torch.tensor([True])
  term._origin_xy = origin.clone()
  term(env_peak, asset_cfg=SceneEntityCfg("robot"), command_name="twist", deadband=0.02)

  env_back, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=back_pos,
    command=command,
    command_counter=torch.tensor([1]),
  )
  cost = term(
    env_back,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    deadband=0.02,
  )
  assert cost[0].item() > 0.0


def test_backslide_resets_on_command_resample():
  term = _make_term(MagicMock(), num_envs=1)
  origin = torch.tensor([[0.0, 0.0]])
  command = torch.tensor([[1.0, 0.0, 0.0]])
  back_pos = torch.tensor([[0.05, 0.0]])

  env, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=back_pos,
    command=command,
    command_counter=torch.tensor([1]),
  )
  term._prev_command_counter = torch.tensor([1])
  term._segment_active = torch.tensor([True])
  term._origin_xy = origin.clone()
  term._s_max = torch.tensor([0.3])
  term.reset(torch.tensor([0]))

  env_resample, _ = _make_backslide_env(
    num_envs=1,
    pos_xy=back_pos,
    command=command,
    command_counter=torch.tensor([2]),
  )
  cost = term(
    env_resample,
    asset_cfg=SceneEntityCfg("robot"),
    command_name="twist",
    command_threshold=0.05,
    deadband=0.02,
  )
  assert cost[0].item() == pytest.approx(0.0)
