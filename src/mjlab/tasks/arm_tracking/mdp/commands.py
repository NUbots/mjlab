"""Sinusoidal joint position command for motor response testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class SinusoidalJointCommand(CommandTerm):
  """Generates sinusoidal target positions for specified joint pairs.

  The command (observation) tensor layout is::

      [frequency, amplitude, pair_select_0, pair_select_1, ...]

  The policy observes these parameters and must learn to produce the
  corresponding sinusoidal joint motions. Target positions are computed
  internally for the reward but are not observed by the policy.

  On deployment, set the command vector to control which joints move
  and how fast/large the oscillations are.
  """

  cfg: SinusoidalJointCommandCfg

  def __init__(self, cfg: SinusoidalJointCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    joint_names = self.robot.joint_names
    self._joint_pair_names: list[str] = []
    self._joint_pair_ids: list[tuple[int, int]] = []
    for pair_label, (left_pat, right_pat) in cfg.joint_pairs.items():
      left_id = right_id = None
      for i, name in enumerate(joint_names):
        if left_pat in name:
          left_id = i
        if right_pat in name:
          right_id = i
      if left_id is None or right_id is None:
        raise ValueError(
          f"Joint pair {pair_label!r} not found: "
          f"left={left_pat!r} right={right_pat!r} "
          f"in {joint_names}"
        )
      self._joint_pair_names.append(pair_label)
      self._joint_pair_ids.append((left_id, right_id))

    self._n_pairs = len(self._joint_pair_ids)

    self._all_joint_ids: list[int] = []
    for left_id, right_id in self._joint_pair_ids:
      if left_id not in self._all_joint_ids:
        self._all_joint_ids.append(left_id)
      if right_id not in self._all_joint_ids:
        self._all_joint_ids.append(right_id)

    self._n_joints = len(self._all_joint_ids)
    default_pos = self.robot.data.default_joint_pos
    self._default_pos = default_pos[:, self._all_joint_ids]

    self._frequency = torch.zeros(self.num_envs, self._n_joints, device=self.device)
    self._amplitude = torch.zeros_like(self._frequency)
    self._phase = torch.zeros_like(self._frequency)
    self._elapsed = torch.zeros(self.num_envs, device=self.device)
    self._target_pos = self._default_pos.clone()

    # command = [freq, amp, pair_select_0, pair_select_1, ...]
    cmd_dim = 2 + self._n_pairs
    self._command = torch.zeros(self.num_envs, cmd_dim, device=self.device)

    self.metrics["tracking_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _pair_local_ids(self, pair_idx: int) -> tuple[int, int]:
    left_global, right_global = self._joint_pair_ids[pair_idx]
    return (
      self._all_joint_ids.index(left_global),
      self._all_joint_ids.index(right_global),
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    freq_lo, freq_hi = self.cfg.frequency_range
    amp_lo, amp_hi = self.cfg.amplitude_range

    if self.cfg.fixed_pair_index is not None:
      pair_indices = torch.full((n,), self.cfg.fixed_pair_index, device=self.device)
    else:
      pair_indices = torch.randint(0, self._n_pairs, (n,), device=self.device)

    pair_select = torch.zeros(n, self._n_pairs, device=self.device)
    pair_select.scatter_(1, pair_indices.unsqueeze(1), 1.0)

    self._frequency[env_ids] = 0.0
    self._amplitude[env_ids] = 0.0
    self._phase[env_ids] = 0.0

    for env_local, env_global in enumerate(env_ids):
      pi = pair_indices[env_local].item()
      left_local, right_local = self._pair_local_ids(int(pi))

      if self.cfg.fixed_frequency is not None:
        f = self.cfg.fixed_frequency
      else:
        f = torch.rand(1, device=self.device).item() * (freq_hi - freq_lo) + freq_lo
      if self.cfg.fixed_amplitude is not None:
        a = self.cfg.fixed_amplitude
      else:
        a = torch.rand(1, device=self.device).item() * (amp_hi - amp_lo) + amp_lo

      self._frequency[env_global, left_local] = f
      self._frequency[env_global, right_local] = f
      self._amplitude[env_global, left_local] = a
      self._amplitude[env_global, right_local] = a

      self._command[env_global, 0] = f
      self._command[env_global, 1] = a
      self._command[env_global, 2:] = pair_select[env_local]

    self._elapsed[env_ids] = 0.0

  def _update_command(self) -> None:
    self._elapsed += self._env.step_dt
    t = self._elapsed.unsqueeze(-1)
    self._target_pos = self._default_pos + self._amplitude * torch.sin(
      2 * math.pi * self._frequency * t + self._phase
    )

  def _update_metrics(self) -> None:
    actual = self.robot.data.joint_pos[:, self._all_joint_ids]
    err = torch.mean(torch.abs(actual - self._target_pos), dim=-1)
    self.metrics["tracking_error"] = err


@dataclass(kw_only=True)
class SinusoidalJointCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  joint_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
  frequency_range: tuple[float, float] = (0.5, 2.0)
  amplitude_range: tuple[float, float] = (0.1, 0.4)
  fixed_frequency: float | None = None
  fixed_amplitude: float | None = None
  fixed_pair_index: int | None = None

  def build(self, env: ManagerBasedRlEnv) -> SinusoidalJointCommand:
    return SinusoidalJointCommand(self, env)
