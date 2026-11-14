"""Dribbling task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_out_of_reach(env: ManagerBasedRlEnv, max_distance: float) -> torch.Tensor:
  """The ball has got away: end the episode rather than train a long chase."""
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  rel = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return torch.norm(rel, dim=-1) > max_distance
