"""Policy-owned gait phase via per-step phase deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_PHASE_DELTA_EPS = 1e-8


def get_phase_delta_action(
  env: ManagerBasedRlEnv,
  term_name: str = "phase_delta",
) -> PhaseDeltaAction:
  """Return the :class:`PhaseDeltaAction` term wired on ``env``."""
  term = env.action_manager.get_term(term_name)
  if not isinstance(term, PhaseDeltaAction):
    raise TypeError(
      f"Action term {term_name!r} is {type(term).__name__}, expected PhaseDeltaAction."
    )
  return term


def _command_active_mask(
  env: ManagerBasedRlEnv,
  command_name: str | None,
  command_threshold: float,
) -> torch.Tensor | None:
  """Return a [B] float mask (1=walking, 0=standing) or None if ungated."""
  if command_name is None:
    return None
  command = env.command_manager.get_command(command_name)
  if command is None:
    return None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  return (total_command > command_threshold).float()


@dataclass(kw_only=True)
class PhaseDeltaActionCfg(ActionTermCfg):
  """Configuration for policy-owned gait phase accumulation."""

  period: float
  """Full gait-cycle duration (seconds). Delta scale is ``step_dt / period``."""

  command_name: str | None = "twist"
  """When set, standing envs freeze phase at zero."""

  command_threshold: float = 0.05
  """Command magnitude below which phase is gated off."""

  scale: float | None = None
  """Override delta scale; defaults to ``step_dt / period`` each step."""

  def build(self, env: ManagerBasedRlEnv) -> PhaseDeltaAction:
    return PhaseDeltaAction(self, env)


class PhaseDeltaAction(ActionTerm):
  """Accumulate a per-env gait phase from policy phase-delta actions.

  Each policy step advances ``policy_phase`` by ``scale * raw_action`` (wrapped
  mod 1). When the twist command is below threshold, delta is zero and phase
  resets to zero so standing presents a distinct signal from walking.
  """

  cfg: PhaseDeltaActionCfg

  def __init__(self, cfg: PhaseDeltaActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._action_dim = 1
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._policy_phase = torch.zeros(self.num_envs, device=self.device)
    self._last_delta = torch.zeros(self.num_envs, device=self.device)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def policy_phase(self) -> torch.Tensor:
    """Per-env normalized gait phase in ``[0, 1)``."""
    return self._policy_phase

  @property
  def last_delta(self) -> torch.Tensor:
    """Scaled phase increment from the most recent policy step."""
    return self._last_delta

  def _delta_scale(self) -> float:
    if self.cfg.scale is not None:
      return float(self.cfg.scale)
    return self._env.step_dt / self.cfg.period

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    raw = actions.squeeze(-1)
    scale = self._delta_scale()
    delta = scale * raw
    active = _command_active_mask(
      self._env, self.cfg.command_name, self.cfg.command_threshold
    )
    if active is not None:
      delta = delta * active
      walking = active > 0
      self._policy_phase = torch.where(
        walking,
        torch.remainder(self._policy_phase + delta, 1.0),
        torch.zeros_like(self._policy_phase),
      )
    else:
      self._policy_phase = torch.remainder(self._policy_phase + delta, 1.0)
    self._last_delta = delta
    self._log_metrics(delta, raw, scale, active)

  def _log_metrics(
    self,
    delta: torch.Tensor,
    raw: torch.Tensor,
    scale: float,
    active: torch.Tensor | None,
  ) -> None:
    nominal_delta = self._env.step_dt / self.cfg.period
    if active is not None:
      walk = active > 0
      if not walk.any():
        return
      delta_w = delta[walk]
      raw_w = raw[walk]
    else:
      delta_w = delta
      raw_w = raw

    log = self._env.extras.setdefault("log", {})
    log["Metrics/phase_delta_mean"] = delta_w.mean()
    log["Metrics/phase_delta_raw_mean"] = raw_w.mean()
    clamped = torch.clamp(delta_w.abs(), min=_PHASE_DELTA_EPS)
    log["Metrics/phase_period_effective_mean"] = self._env.step_dt / clamped.mean()
    log["Metrics/phase_delta_nominal_ratio_mean"] = (delta_w / nominal_delta).mean()

  def apply_actions(self) -> None:
    """Phase is state-only; nothing to write to the simulation."""

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._policy_phase[env_ids] = 0.0
    self._last_delta[env_ids] = 0.0
