from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


# Stage schemas.


class _RewardCurriculumStageOptional(TypedDict, total=False):
  weight: float
  params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
  step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]
  time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
  step: int


# Shared engine.  Stage dicts are passed directly from the public TypedDict
# schemas.  Any key that isn't "step" or "params" is treated as a top-level
# field on the target term config (e.g. "weight" on RewardTermCfg).

_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
  term_cfg: Any,
  term_name: str,
  stages: Sequence[Any],
) -> None:
  """Validate stage ordering, field existence, and param keys."""
  for i in range(1, len(stages)):
    if stages[i]["step"] < stages[i - 1]["step"]:
      raise ValueError(
        f"Curriculum stages must be in nondecreasing step order,"
        f" but stage {i} has step"
        f" {stages[i]['step']} < {stages[i - 1]['step']}."
      )
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
        raise AttributeError(
          f"Field '{key}' does not exist on the resolved term config for '{term_name}'."
        )
  for stage in stages:
    unknown = stage.get("params", {}).keys() - term_cfg.params.keys()
    if unknown:
      raise KeyError(
        f"Stage at step {stage['step']} sets unknown param(s)"
        f" {unknown} on term '{term_name}'. Check for typos."
      )


def _apply_stages(
  term_cfg: Any,
  step_counter: int,
  stages: Sequence[Any],
) -> dict[str, torch.Tensor]:
  """Apply staged updates and return a logging snapshot."""
  for stage in stages:
    if step_counter >= stage["step"]:
      for key, value in stage.items():
        if key not in _RESERVED_KEYS:
          setattr(term_cfg, key, value)
      if "params" in stage:
        term_cfg.params.update(stage["params"])
  # Only log values that stages actually reference.
  logged_fields: set[str] = set()
  logged_params: set[str] = set()
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS:
        logged_fields.add(key)
    for key in stage.get("params", {}):
      logged_params.add(key)
  result: dict[str, torch.Tensor] = {}
  for key in logged_fields:
    value = getattr(term_cfg, key)
    if isinstance(value, (int, float, bool)):
      result[key] = torch.tensor(value)
    elif isinstance(value, torch.Tensor):
      result[key] = value
  for key in logged_params:
    v = term_cfg.params[key]
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  return result


# Public wrappers.


class reward_curriculum:
  """Update a reward term's weight and/or params based on training steps.

  Each stage specifies a ``step`` threshold and optionally a ``weight``
  and/or ``params`` dict.  When ``env.common_step_counter`` reaches a
  stage's ``step``, the corresponding values are applied.  Later stages
  take precedence when multiple thresholds are reached.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 12000, "weight": -0.1},
          {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    _validate_stages(self._term_cfg, reward_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


def _read_plateau_metric(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  metric_key: str | None,
  metric_reward_name: str | None,
) -> float:
  """Read a scalar training metric for plateau detection.

  Prefers ``metric_key`` in ``env.extras["log"]`` (e.g. reward-side
  ``Metrics/*`` entries written during the preceding step).  Falls back to
  the mean episodic sum in ``reward_manager._episode_sums`` for
  ``metric_reward_name`` over the resetting envs.
  """
  if metric_key is not None:
    log = env.extras.get("log", {})
    if metric_key not in log:
      raise KeyError(
        f"Plateau metric key '{metric_key}' not found in env.extras['log']."
      )
    value = log[metric_key]
    if isinstance(value, torch.Tensor):
      return value.mean().item()
    return float(value)
  assert metric_reward_name is not None
  episode_sums = env.reward_manager._episode_sums[metric_reward_name]
  if isinstance(env_ids, slice):
    batch = episode_sums
  else:
    batch = episode_sums[env_ids]
  return batch.mean().item()


class staged_on_plateau:
  """Advance reward-term stages when a tracked metric plateaus.

  Like ``reward_curriculum``, each stage sets ``weight`` and/or ``params`` on
  a target reward term.  Stages are ordered by their ``step`` field (used only
  for validation and cumulative application, not as a trigger threshold).

  The curriculum runs on env reset (see ``ManagerBasedRlEnv._reset_idx``),
  before ``extras['log']`` is cleared.  Track progress via:

  - ``metric_key``: a scalar in ``env.extras['log']`` from the preceding step
    (e.g. ``Metrics/peak_height_mean`` written by a reward term), or
  - ``metric_reward_name``: mean episodic sum from ``reward_manager`` for the
    envs being reset (available because reward reset runs after curriculum).

  Each evaluation updates an EMA of the metric.  When the EMA fails to beat the
  running best by ``improvement_threshold`` for ``patience`` consecutive
  evaluations *after* ``min_steps_per_stage`` env steps in the current stage,
  the next stage is applied and patience resets.

  Example::

    CurriculumTermCfg(
      func=mdp.staged_on_plateau,
      params={
        "reward_name": "foot_swing_height",
        "metric_key": "Metrics/peak_height_mean",
        "patience": 20,
        "min_steps_per_stage": 4000 * 24,
        "improvement_threshold": 0.005,
        "stages": [
          {"step": 0, "weight": 0.75},
          {"step": 1, "weight": 0.4},
          {"step": 2, "weight": 0.1},
          {"step": 3, "weight": 0.0},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    metric_key: str | None = cfg.params.get("metric_key")
    metric_reward_name: str | None = cfg.params.get("metric_reward_name")
    if (metric_key is None) == (metric_reward_name is None):
      raise ValueError(
        "staged_on_plateau requires exactly one of 'metric_key' or "
        "'metric_reward_name'."
      )

    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    self._metric_key = metric_key
    self._metric_reward_name = metric_reward_name
    self._patience: int = cfg.params.get("patience", 10)
    self._min_steps_per_stage: int = cfg.params.get("min_steps_per_stage", 0)
    self._improvement_threshold: float = cfg.params.get("improvement_threshold", 0.01)
    self._ema_alpha: float = cfg.params.get("ema_alpha", 0.1)

    self._stage_idx = 0
    self._stage_start_step = env.common_step_counter
    self._best = float("-inf")
    self._ema = 0.0
    self._patience_count = 0
    self._initialized = False

    _validate_stages(self._term_cfg, reward_name, self._stages)
    _apply_stages(self._term_cfg, self._stages[0]["step"], self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
    metric_key: str | None = None,
    metric_reward_name: str | None = None,
    patience: int = 10,
    min_steps_per_stage: int = 0,
    improvement_threshold: float = 0.01,
    ema_alpha: float = 0.1,
  ) -> dict[str, torch.Tensor]:
    del (
      reward_name,
      stages,
      metric_key,
      metric_reward_name,
      patience,
      min_steps_per_stage,
      improvement_threshold,
      ema_alpha,
    )

    metric = _read_plateau_metric(
      env,
      env_ids,
      metric_key=self._metric_key,
      metric_reward_name=self._metric_reward_name,
    )
    if not self._initialized:
      self._ema = metric
      self._best = metric
      self._initialized = True
    else:
      self._ema = self._ema_alpha * metric + (1.0 - self._ema_alpha) * self._ema

    steps_in_stage = env.common_step_counter - self._stage_start_step
    if steps_in_stage >= self._min_steps_per_stage:
      if self._ema > self._best + self._improvement_threshold:
        self._best = self._ema
        self._patience_count = 0
      else:
        self._patience_count += 1

      if (
        self._patience_count >= self._patience
        and self._stage_idx < len(self._stages) - 1
      ):
        self._stage_idx += 1
        self._stage_start_step = env.common_step_counter
        self._patience_count = 0
        self._best = self._ema
        _apply_stages(
          self._term_cfg,
          self._stages[self._stage_idx]["step"],
          self._stages,
        )

    snapshot = _apply_stages(
      self._term_cfg,
      self._stages[self._stage_idx]["step"],
      self._stages,
    )
    snapshot["stage_idx"] = torch.tensor(self._stage_idx)
    snapshot["metric_ema"] = torch.tensor(self._ema)
    snapshot["metric_best"] = torch.tensor(self._best)
    snapshot["plateau_count"] = torch.tensor(self._patience_count)
    return snapshot


class termination_curriculum:
  """Update a termination term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict.  When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied.  Later stages take precedence.

  Example::

    CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 12000, "params": {"threshold": 1000.0}},
          {"step": 24000, "params": {"threshold": 700.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    termination_name: str = cfg.params["termination_name"]
    stages: list[TerminationCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
    self._stages = stages
    _validate_stages(self._term_cfg, termination_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    termination_name: str,
    stages: list[TerminationCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, termination_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)
