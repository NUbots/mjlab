"""Dribbling task curricula."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.tasks.dribbling.mdp.ball_command import BallVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class _BallCommandStageOptional(TypedDict, total=False):
  speed_range: tuple[float, float]
  resampling_time_range: tuple[float, float]


class BallCommandStage(_BallCommandStageOptional):
  step: int


def ball_command_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[BallCommandStage],
) -> dict[str, torch.Tensor]:
  """Step the ball command's speed range and turn frequency over training.

  JOGA Table VII: dribbling first learns with infrequent turns (a new command
  every 7 s), then the resampling interval is shortened and widened.
  """
  del env_ids  # Unused.
  term = env.command_manager.get_term(command_name)
  assert term is not None
  cfg = cast(BallVelocityCommandCfg, term.cfg)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "speed_range" in stage:
        cfg.speed_range = stage["speed_range"]
      if "resampling_time_range" in stage:
        cfg.resampling_time_range = stage["resampling_time_range"]
  return {
    "speed_max": torch.tensor(cfg.speed_range[1]),
    "resample_min_s": torch.tensor(cfg.resampling_time_range[0]),
    "resample_max_s": torch.tensor(cfg.resampling_time_range[1]),
  }
