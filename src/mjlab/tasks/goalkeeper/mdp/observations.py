"""Goalkeeper observations.

The actor sees only what the robot will have: its own proprioception and the block
command. Everything here is for the critic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.goalkeeper.mdp.shot_command import ShotCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def true_ball_state(
  env: ManagerBasedRlEnv,
  command_name: str = "shot",
) -> torch.Tensor:
  """Privileged ball state in the goalie's frame: position, velocity, crossing, time.

  The actor's command comes from a simulated estimate that is late and slow to pick up
  a kick. Giving the critic the truth lets it value states the actor cannot yet
  distinguish, which is the usual asymmetric-actor-critic setup.
  """
  shot = env.command_manager.get_term(command_name)
  assert isinstance(shot, ShotCommand)
  pos_r, vel_r = shot.to_robot_frame(
    shot.ball.data.root_link_pos_w, shot.ball.data.root_link_lin_vel_w[:, :2]
  )
  return torch.cat(
    [
      pos_r,
      vel_r,
      shot.true_crossing.unsqueeze(-1),
      shot.true_time_to_cross.unsqueeze(-1),
      shot.on_target.float().unsqueeze(-1),
    ],
    dim=-1,
  )
