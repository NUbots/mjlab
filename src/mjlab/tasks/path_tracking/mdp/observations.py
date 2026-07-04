from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.path_tracking.mdp.path_command import PathCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def path_waypoints(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Lookahead path waypoints relative to the robot's yaw frame.

  Returns [B, K*4] with (dx, dy, cos(dheading), sin(dheading)) per
  waypoint. This is the deployment-safe path interface: the onboard walk
  path planner can produce exactly these quantities from its planned path
  and odometry, so the actor may observe them (unlike world-frame state or
  the desired twist, which only rewards/critic may use).
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, PathCommand), (
    f"path_waypoints requires a PathCommand term, got {type(term).__name__}"
  )
  return term.waypoints
