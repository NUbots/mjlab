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


def relax_upright(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reward_name: str = "upright",
  weights: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25),
) -> dict[str, torch.Tensor]:
  """Loosen the upright bonus as the keeper moves up the drills.

  The upright term is scaffolding. It was added because at the start of training the
  action-smoothness penalties made falling over the quickest way to stop paying them,
  and episodes collapsed to about 7 steps. Once the keeper can stand, holding it
  rigidly upright only rules out the motions a keeper needs: leaning out over a foot,
  dropping the hips, reaching across. So it is strong while the keeper is learning to
  stand on drill 1 and is eased off level by level after that.

  Tied to the drill rather than to a step count for the same reason the drills are:
  the keeper should only be given the freedom once it has shown it can stand.
  """
  del env_ids  # Applies to the whole batch.
  shot = cast(ShotCommand, env.command_manager.get_term(command_name))
  term_cfg = env.reward_manager.get_term_cfg(reward_name)
  term_cfg.weight = float(weights[min(shot.level, len(weights) - 1)])
  return {"upright_weight": torch.tensor(term_cfg.weight)}
