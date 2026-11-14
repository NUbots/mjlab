from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def drop_command_to_zero(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  hold_range_s: tuple[float, float],
  command_threshold: float = 0.05,
) -> None:
  """Cut a moving command to zero instantly and hold it there.

  Use with ``mode="interval"``. Of the selected envs, those currently commanded
  to move (``|v_xy| + |w_z| > command_threshold``) are switched to standing in
  the same step, with no ramp, so the policy has to learn an abrupt stop. The
  env is flagged as a standing env, which also overrides the heading and
  world-frame controllers, and its resample timer is set to a hold time drawn
  from ``hold_range_s``. The zero command therefore lasts long enough for the
  stop to be rewarded before normal resampling resumes.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, UniformVelocityCommand)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  command = term.vel_command_b[env_ids]
  moving = (
    torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  ) > command_threshold
  drop_ids = env_ids[moving & ~term.is_standing_env[env_ids]]
  if len(drop_ids) == 0:
    return

  term.is_standing_env[drop_ids] = True
  term.vel_command_b[drop_ids] = 0.0
  term.vel_command_w[drop_ids] = 0.0
  term.time_left[drop_ids] = torch.empty(len(drop_ids), device=env.device).uniform_(
    *hold_range_s
  )
