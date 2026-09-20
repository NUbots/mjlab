"""Goalkeeper curricula."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ShotStage(TypedDict, total=False):
  """One stage of the shot envelope, applied once ``step`` env steps have passed."""

  step: int
  crossing: tuple[float, float]
  speed: tuple[float, float]
  distance: tuple[float, float]


def shot_envelope(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  shot_stages: list[ShotStage],
) -> dict[str, torch.Tensor]:
  """Widen the shots the goalie faces as it learns.

  A shot crossing a metre away cannot be reached from a standing start in the time
  available, so early on it is a reward the policy cannot earn no matter what it does,
  and the gradient it contributes is noise. Starting with shots that come close to the
  goalie and widening from there gives it something to chase from the first iteration.
  """
  del env_ids  # Applied to the whole batch.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(ShotCommandCfg, command_term.cfg)
  for stage in shot_stages:
    if env.common_step_counter >= stage["step"]:
      if stage.get("crossing") is not None:
        cfg.crossing = stage["crossing"]
      if stage.get("speed") is not None:
        cfg.speed = stage["speed"]
      if stage.get("distance") is not None:
        cfg.distance = stage["distance"]
  return {
    "crossing_max": torch.tensor(cfg.crossing[1]),
    "speed_max": torch.tensor(cfg.speed[1]),
  }
