"""Commanded ball exit velocity for the kicking task, with strike tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.dribbling.mdp.ball_command import (
  BallVelocityCommand,
  BallVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


class KickCommand(BallVelocityCommand):
  """One commanded ball exit velocity per episode, plus the strike state.

  The kick rewards (JOGA Table V) are phased: ``I_pre`` before the strike,
  ``I_win`` for a short window after it, and "after" from the strike onward.
  This term owns that state. The strike is the first step the ball moves
  faster than ``strike_speed`` while within ``strike_radius`` of the robot;
  it latches for the rest of the episode, so a weak first touch is a spent
  kick rather than something to try again.

  A strike made with the shin or knee earns only ``leg_strike_scale`` of the
  window rewards: at full pay the policy learns to knee the ball, the easiest
  way to move it at speed; at zero it never finds out that moving the ball
  pays and stands beside it.
  """

  cfg: KickCommandCfg

  def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.struck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.steps_since_strike = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.peak_speed = torch.zeros(self.num_envs, device=self.device)
    self.leg_strike = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["leg_strike"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["struck"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["peak_speed_ratio"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["peak_heading_error_deg"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def pre_strike(self) -> torch.Tensor:
    return ~self.struck

  @property
  def in_window(self) -> torch.Tensor:
    window_steps = int(round(self.cfg.window_s / self._env.step_dt))
    return self.struck & (self.steps_since_strike < window_steps)

  @property
  def window_weight(self) -> torch.Tensor:
    """1 inside the reward window, scaled down for a kick made with the leg."""
    scale = torch.where(self.leg_strike, self.cfg.leg_strike_scale, 1.0)
    return self.in_window.float() * scale

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if env_ids is None:
      env_ids = slice(None)
    extras = super().reset(env_ids)
    self.struck[env_ids] = False
    self.steps_since_strike[env_ids] = 0
    self.peak_speed[env_ids] = 0.0
    self.leg_strike[env_ids] = False
    return extras

  def _update_metrics(self) -> None:
    ball_vel = self.ball.data.root_link_lin_vel_w[:, :2]
    speed = torch.norm(ball_vel, dim=-1)
    rel = self.ball.data.root_link_pos_w[:, :2] - self.robot.data.root_link_pos_w[:, :2]
    near = torch.norm(rel, dim=-1) < self.cfg.strike_radius
    new_strike = ~self.struck & near & (speed > self.cfg.strike_speed)
    self.steps_since_strike = torch.where(
      self.struck, self.steps_since_strike + 1, self.steps_since_strike
    )
    self.struck |= new_strike
    if self.cfg.leg_contact_sensor is not None:
      # Any leg contact up to the end of the window spoils the kick. The
      # substep history catches a contact that ends between control steps.
      sensor: ContactSensor = self._env.scene[self.cfg.leg_contact_sensor]
      assert sensor.data.force_history is not None
      force = torch.norm(sensor.data.force_history, dim=-1)  # [B, N, H]
      leg_hit = force.flatten(1).max(dim=-1).values > self.cfg.leg_force_threshold
      window_steps = int(round(self.cfg.window_s / self._env.step_dt))
      open_ = ~self.struck | (self.steps_since_strike < window_steps)
      self.leg_strike |= leg_hit & open_
    self.metrics["leg_strike"] = self.leg_strike.float()

    cmd = self.vel_command_w[:, :2]
    cmd_speed = torch.norm(cmd, dim=-1).clamp(min=1e-6)
    # Speed and heading error at the ball's peak speed (JOGA Sec. III-C).
    is_peak = self.struck & (speed > self.peak_speed)
    self.peak_speed = torch.where(is_peak, speed, self.peak_speed)
    cos = torch.sum(ball_vel * cmd, dim=-1) / (speed.clamp(min=1e-6) * cmd_speed)
    heading_err = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
    self.metrics["struck"] = self.struck.float()
    self.metrics["peak_speed_ratio"] = self.peak_speed / cmd_speed
    self.metrics["peak_heading_error_deg"] = torch.where(
      is_peak, heading_err, self.metrics["peak_heading_error_deg"]
    )


@dataclass(kw_only=True)
class KickCommandCfg(BallVelocityCommandCfg):
  strike_speed: float = 0.5
  """Ball speed (m/s) above which the ball counts as struck."""
  strike_radius: float = 0.6
  """The ball must be this close (m) to the robot for a strike to register."""
  window_s: float = 0.4
  """Length of the post-strike window in which exit velocity is rewarded."""
  leg_contact_sensor: str | None = None
  """Contact sensor (with ``history_length`` set) for shin/knee on the ball."""
  leg_strike_scale: float = 0.25
  """Share of the window rewards kept by a kick made with the leg."""
  leg_force_threshold: float = 5.0
  """Leg-ball force (N) above which the kick counts as made with the leg."""

  def build(self, env: ManagerBasedRlEnv) -> KickCommand:
    return KickCommand(self, env)
