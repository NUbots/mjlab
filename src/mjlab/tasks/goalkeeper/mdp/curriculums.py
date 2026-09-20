"""Goalkeeper curricula."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def shot_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  advance_at: float = 0.6,
  min_shots: int = 400,
) -> dict[str, torch.Tensor]:
  """Move the keeper up a drill once it is actually saving the one it is on.

  Advancing on a step count, as the first version did, moves the keeper on whether or
  not it has learned anything; advancing on the save rate means a level that is not
  working holds it until it does. The running save rate is reset on promotion so the
  next level has to earn its own evidence, which also stops several levels being
  cleared at once on the strength of one easy drill.
  """
  del env_ids  # Applies to the whole batch.
  command_term = env.command_manager.get_term(command_name)
  shot = cast(ShotCommand, command_term)

  ready = (
    float(shot.recent_save_rate) > advance_at
    and shot.shots_since_level >= min_shots
    and shot.level < len(shot.levels) - 1
  )
  if ready:
    shot.level += 1
    shot.shots_since_level = 0
    shot.recent_save_rate.zero_()

  return {
    "level": torch.tensor(float(shot.level)),
    "recent_save_rate": shot.recent_save_rate.detach().clone().cpu(),
    "crossing_max": torch.tensor(float(shot.levels[shot.level]["crossing"][1])),
  }
