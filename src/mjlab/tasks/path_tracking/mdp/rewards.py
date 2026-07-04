"""Rewards specific to the path tracking task.

The velocity rewards only track the *derived twist*; these add direct
pressure to stay on the generated path itself. The path is
time-parameterized — its reference pose advances at the planned speed
whether or not the robot follows — so tracking the current reference pose
rewards the robot for keeping pace and reaching the path's end on
schedule, and for aligning its heading with the path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.path_tracking.mdp.path_command import PathCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _path_command(env: ManagerBasedRlEnv, command_name: str) -> PathCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, PathCommand), (
    f"expected a PathCommand term, got {type(term).__name__}"
  )
  return term


def track_path_position(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  """Reward for keeping up with the moving reference pose along the path.

  The reference pose advances at the planned pace regardless of the robot,
  so a robot that lags — the "can't keep up" failure mode — sees its
  distance to the current reference grow and this reward decay. Scored with
  the same exponential kernel as the velocity trackers, so keeping pace (and
  thus reaching the path's end on schedule) is directly rewarded.
  """
  term = _path_command(env, command_name)
  return torch.exp(-torch.square(term.pos_error) / std**2)


def track_path_heading(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  """Reward for aligning the robot's heading with the path reference heading.

  Complements :func:`track_path_position`: staying on the path in position
  does not by itself force the yaw to match the reference, so this term adds
  the pressure that tightens heading alignment.
  """
  term = _path_command(env, command_name)
  return torch.exp(-torch.square(term.heading_error) / std**2)
