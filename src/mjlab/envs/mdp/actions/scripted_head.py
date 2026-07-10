"""Scripted saccadic head/neck motion driven off-policy (deployment fidelity).

On the real robot the head is the vision system's actuator, not the walk
policy's: it runs scan/tracking patterns the locomotion controller never
commands. Training the walk policy to CONTROL the head is both unrealistic
and wasteful (the head is light and unloaded, so its action dims just park
exploration noise -> flail, doc 15 R37). This action term removes the head
from the policy: it consumes zero policy action dims and instead drives the
neck/head joints saccadically -- dwell at a fixation point, then step the
target to a new random point so the servo PD produces a fast slew -- as a
per-env-randomized disturbance the policy must stay upright through. The
saccades matter: their fast slews impart angular-momentum impulses a smooth
scan would not, which is the realistic head-motion disturbance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ScriptedHeadActionCfg(ActionTermCfg):
  """Configuration for off-policy scripted saccadic head/neck motion.

  The head holds a fixation target (a step function) for a per-env-random
  dwell, then jumps every listed joint to a new random fixation; the servo
  PD produces the saccade slew. Targets span a fraction of each joint's soft
  range about its default pose. Dwell and targets are resampled per-env on
  reset and at every saccade so the motion is unpredictable and the policy
  must be robust to it rather than memorize it.
  """

  joint_names: tuple[str, ...]
  """Head/neck joints to drive (e.g. ``("neck_yaw", "head_pitch")``)."""

  target_frac: float = 0.85
  """Fixation targets span +/- this fraction of each joint's half-range."""

  dwell_range_s: tuple[float, float] = (0.3, 1.2)
  """Per-env hold time between saccades (seconds)."""

  def build(self, env: ManagerBasedRlEnv) -> ScriptedHeadAction:
    return ScriptedHeadAction(self, env)


class ScriptedHeadAction(ActionTerm):
  """Drive head joints saccadically; zero policy action dims."""

  cfg: ScriptedHeadActionCfg

  def __init__(self, cfg: ScriptedHeadActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    joint_ids, _ = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_joints = len(joint_ids)

    self._center = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids, :]  # [B,J,2]
    self._lower = limits[..., 0]
    self._upper = limits[..., 1]
    self._half_range = 0.5 * (self._upper - self._lower)

    shape = (self.num_envs, self._num_joints)
    self._target = torch.zeros(shape, device=self.device)
    self._next_saccade = torch.zeros(self.num_envs, device=self.device)  # [B], seconds
    self._empty = torch.zeros(self.num_envs, 0, device=self.device)
    self._resample(slice(None))

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._empty

  def _env_time(self) -> torch.Tensor:
    return self._env.episode_length_buf * self._env.step_dt  # [B]

  def _resample(self, env_ids: torch.Tensor | slice) -> None:
    """Pick new fixation targets and next-saccade times for ``env_ids``."""
    if isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0:
      return
    n = self.num_envs if isinstance(env_ids, slice) else int(env_ids.numel())
    j = self._num_joints
    # Fixation targets: uniform in +/- target_frac * half_range about center.
    frac = (torch.rand(n, j, device=self.device) * 2.0 - 1.0) * self.cfg.target_frac
    self._target[env_ids] = torch.clamp(
      self._center[env_ids] + frac * self._half_range[env_ids],
      self._lower[env_ids],
      self._upper[env_ids],
    )
    lo, hi = self.cfg.dwell_range_s
    dwell = torch.rand(n, device=self.device) * (hi - lo) + lo
    # Schedule relative to current env time (0 for freshly reset envs).
    base = self._env_time() if isinstance(env_ids, slice) else self._env_time()[env_ids]
    self._next_saccade[env_ids] = base + dwell

  def process_actions(self, actions: torch.Tensor) -> None:
    """No policy input: the head is driven entirely by the script."""
    del actions

  def apply_actions(self) -> None:
    due = self._env_time() >= self._next_saccade  # [B]
    if bool(due.any()):
      self._resample(due.nonzero(as_tuple=False).flatten())
    self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._resample(slice(None) if env_ids is None else env_ids)
