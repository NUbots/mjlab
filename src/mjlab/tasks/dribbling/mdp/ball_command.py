"""Commanded ball velocity for the dribbling task.

Following JOGA (Appendix B.1), the operator commands where the *ball* should
go, as a planar world-frame velocity, and the policy decides how the body gets
it there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BallVelocityCommand(CommandTerm):
  """World-frame planar ball velocity command, piecewise constant in time.

  The command tensor is ``[vx_w, vy_w, 0]``. The trailing zero keeps it
  shape-compatible with the velocity task's twist command, so the gait
  regularizers gated on ``|v_xy| + |w_z|`` work unchanged.
  """

  cfg: BallVelocityCommandCfg

  def __init__(self, cfg: BallVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_name]
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    # True for the first command of an episode, which is aimed relative to
    # the robot's heading rather than uniformly over the circle.
    self._first = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["error_ball_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ball_speed_along_cmd"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_w

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if env_ids is None:
      env_ids = slice(None)
    self._first[env_ids] = True
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    ball_vel = self.ball.data.root_link_lin_vel_w[:, :2]
    cmd = self.vel_command_w[:, :2]
    self.metrics["error_ball_vel"] += torch.norm(cmd - ball_vel, dim=-1) / max_step
    cmd_dir = cmd / torch.norm(cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    self.metrics["ball_speed_along_cmd"] += (
      torch.sum(ball_vel * cmd_dir, dim=-1) / max_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    speed = r.uniform_(*self.cfg.speed_range).clone()
    # Subsequent commands: any world direction (sharp turns included).
    direction = r.uniform_(-math.pi, math.pi).clone()
    # First command of the episode: within a cone around the robot's heading,
    # so early training sees the ball roughly between robot and target.
    first = self._first[env_ids]
    # xquat is stale right after a reset (kinematics not yet forwarded), so
    # read the root orientation from qpos, which the reset event just wrote.
    data = self.robot.data
    quat = data.data.qpos[:, data.indexing.free_joint_q_adr[3:7]][env_ids]
    yaw = torch.atan2(
      2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
      1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    cone = r.uniform_(*self.cfg.initial_heading_range).clone()
    direction = torch.where(first, yaw + cone, direction)
    self._first[env_ids] = False

    standing = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs
    speed = torch.where(standing, torch.zeros_like(speed), speed)

    self.vel_command_w[env_ids, 0] = speed * torch.cos(direction)
    self.vel_command_w[env_ids, 1] = speed * torch.sin(direction)
    self.vel_command_w[env_ids, 2] = 0.0

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids  # Piecewise constant: nothing to advance between resamples.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    cmds = self.vel_command_w.cpu().numpy()
    ball_pos = self.ball.data.root_link_pos_w.cpu().numpy()
    ball_vel = self.ball.data.root_link_lin_vel_w.cpu().numpy()
    lift = np.array([0.0, 0.0, 0.25])
    for i in env_indices:
      if np.linalg.norm(ball_pos[i]) < 1e-6:
        continue
      start = ball_pos[i] + lift
      # Commanded ball velocity (green) and actual ball velocity (blue).
      visualizer.add_arrow(
        start,
        start + np.array([cmds[i, 0], cmds[i, 1], 0.0]) * self.cfg.viz_scale,
        color=(0.1, 0.8, 0.1, 0.8),
        width=0.02,
      )
      visualizer.add_arrow(
        start,
        start + np.array([ball_vel[i, 0], ball_vel[i, 1], 0.0]) * self.cfg.viz_scale,
        color=(0.1, 0.3, 0.9, 0.8),
        width=0.02,
      )


@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_name: str = "ball"
  speed_range: tuple[float, float] = (0.3, 1.0)
  """Commanded ball speed (m/s)."""
  initial_heading_range: tuple[float, float] = (-math.pi, math.pi)
  """Direction of the episode's first command, relative to the robot's yaw."""
  rel_standing_envs: float = 0.05
  """Fraction of commands that ask for the ball to be held still."""
  viz_scale: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> BallVelocityCommand:
    return BallVelocityCommand(self, env)
