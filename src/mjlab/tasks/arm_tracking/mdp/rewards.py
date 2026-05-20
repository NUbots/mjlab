"""Rewards for arm tracking task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.arm_tracking.mdp.commands import SinusoidalJointCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _get_sin_cmd(env: ManagerBasedRlEnv, command_name: str) -> SinusoidalJointCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, SinusoidalJointCommand)
  return term


def joint_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Exponential reward for tracking sinusoidal joint targets."""
  cmd = _get_sin_cmd(env, command_name)
  target = cmd._target_pos
  actual = env.scene["robot"].data.joint_pos[:, cmd._all_joint_ids]
  error = torch.sum(torch.square(actual - target), dim=-1)
  return torch.exp(-error / (std**2))


def joint_velocity_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward for matching expected joint velocities from the sinusoid."""
  cmd = _get_sin_cmd(env, command_name)
  t = cmd._elapsed.unsqueeze(-1)
  target_vel = (
    cmd._amplitude
    * 2
    * math.pi
    * cmd._frequency
    * torch.cos(2 * math.pi * cmd._frequency * t + cmd._phase)
  )
  actual_vel = env.scene["robot"].data.joint_vel[:, cmd._all_joint_ids]
  error = torch.sum(torch.square(actual_vel - target_vel), dim=-1)
  return torch.exp(-error / (std**2))
