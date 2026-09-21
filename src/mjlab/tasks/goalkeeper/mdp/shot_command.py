"""Shot command: places the ball, kicks it, and reports the block command.

The command the policy sees is the one the robot's behaviour system will send it
(``message::skill::Block``): ``[active, dy, t, v]``, where ``dy`` is where the ball is
predicted to cross the goalie's own lateral line, ``t`` is how long until it does and
``v`` is the ball's speed. See ``BLOCK_POLICY_CONTRACT.md`` in the goalie plan.

The command is computed from a *simulated ball estimate*, not from the ball's true
state: on the robot it comes from vision and the ball UKF, which are late, noisy and
slow to pick up a kick. The model here reproduces what that chain measured in NUSim
(``ukf-validation.md``): about 30 Hz updates, 0.04 s of effective lag, position noise
that grows with range, and a velocity estimate that needs about 0.3 s to catch up with
a kick. Rewards, in contrast, are computed from the true ball, so the policy is paid
for actually blocking it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ShotLevel(TypedDict, total=False):
  """One drill in the curriculum: the shots a level serves up.

  Levels work the way soccer training does. The keeper starts on a ball rolled
  straight at it and has to stop it, then the ball starts arriving to one side, then
  faster and wider. A level is only left behind once the keeper is actually saving
  them, and later levels keep serving a share of earlier ones so it does not forget
  how to deal with the simple ones.
  """

  name: str
  crossing: tuple[float, float]
  speed: tuple[float, float]
  distance: tuple[float, float]


class ShotCommand(CommandTerm):
  """Drives one shot per resampling period and publishes the block command."""

  cfg: ShotCommandCfg

  def __init__(self, cfg: ShotCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_name]

    zeros = lambda *shape: torch.zeros(*shape, device=self.device)  # noqa: E731

    # Command published to the policy: [active, dy, t, v].
    self.command_buf = zeros(self.num_envs, 4)

    # Shot bookkeeping.
    self.time_since_resample = zeros(self.num_envs)
    self.kick_delay = zeros(self.num_envs)
    self.kicked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.touched = torch.zeros_like(self.kicked)
    self.finished = torch.zeros_like(self.kicked)
    # The velocity written at the kick only shows up in the entity's data after the
    # next forward(), so a shot is only judged to have stopped once it has been seen
    # moving. Without this every shot is "over" on the step it is kicked.
    self.was_moving = torch.zeros_like(self.kicked)
    self.shot_speed = zeros(self.num_envs)
    self.shot_direction = zeros(self.num_envs, 2)

    # How much of the danger has been taken out of the ball: 0 while it is still
    # heading inside the posts, 1 once it can no longer reach them at all, graded in
    # between by how far outside it is now projected to pass. Rewards read this to pay
    # for the outcome of a touch rather than the touch itself.
    self.defused = zeros(self.num_envs)

    # Where the ball will come to rest if left alone, as metres up the field from the
    # goalie's start (the goal is behind, at negative x). A save only keeps the ball
    # out for now; clearing it upfield is what stops it coming straight back.
    self.rest_x = zeros(self.num_envs)
    self.cleared_now = zeros(self.num_envs)

    # True crossing of the goalie's lateral line, in the goalie's frame, and whether
    # the ball is on its way there. Rewards read these.
    self.true_crossing = zeros(self.num_envs)
    self.true_time_to_cross = zeros(self.num_envs)
    self.ball_offset = zeros(self.num_envs)
    self.on_target = torch.zeros_like(self.kicked)

    # Whether the shot, left alone, would have gone in. Only those shots are a
    # goalkeeping test, so they are the ones the save rate is measured over.
    self.on_target_shot = torch.zeros_like(self.kicked)

    # Events for the sparse rewards, true for the single step they happen on.
    self.touched_now = torch.zeros_like(self.kicked)
    self.saved_now = torch.zeros_like(self.kicked)
    self.scored_now = torch.zeros_like(self.kicked)

    # Simulated estimate state.
    self.est_pos = zeros(self.num_envs, 2)
    self.est_vel = zeros(self.num_envs, 2)
    self.estimate_timer = zeros(self.num_envs)
    depth = max(
      1,
      int(
        round(
          max(cfg.estimate_latency_s, cfg.velocity_dead_time_s) * cfg.estimate_rate_hz
        )
      )
      + 1,
    )
    self.history_pos = zeros(self.num_envs, depth, 2)
    self.history_vel = zeros(self.num_envs, depth, 2)
    self.history_index = 0

    # Counted over the episode, so the metrics below are per-shot rates. "Did this
    # episode ever block" cannot tell one shot in four from four.
    self.shots_finished = zeros(self.num_envs)
    self.shots_on_target = zeros(self.num_envs)
    self.shots_saved = zeros(self.num_envs)
    self.shots_touched = zeros(self.num_envs)
    self.shots_conceded = zeros(self.num_envs)
    self.shots_cleared = zeros(self.num_envs)
    self.clearance_total = zeros(self.num_envs)

    self.metrics["save_rate"] = zeros(self.num_envs)
    self.metrics["concede_rate"] = zeros(self.num_envs)
    self.metrics["touch_rate"] = zeros(self.num_envs)
    self.metrics["shots_on_target"] = zeros(self.num_envs)
    self.metrics["command_dy_error"] = zeros(self.num_envs)
    self.metrics["clear_rate"] = zeros(self.num_envs)
    self.metrics["clearance"] = zeros(self.num_envs)

    # Curriculum: which drill the keeper is on, and how it is doing on it. The save
    # rate is a running average over resolved shots rather than a per-episode figure,
    # so the curriculum can advance on evidence instead of on a step count.
    levels = cfg.levels or (
      {
        "name": "full",
        "crossing": (-0.8, 0.8),
        "speed": (1.5, 4.0),
        "distance": (2.0, 4.5),
      },
    )
    self.levels: tuple[ShotLevel, ...] = tuple(levels)
    self.level = int(min(cfg.start_level, len(self.levels) - 1))
    self.recent_save_rate = torch.zeros((), device=self.device)
    self.shots_since_level = 0

    def to_range(key: str) -> torch.Tensor:
      return torch.tensor(
        [list(level[key]) for level in self.levels], device=self.device
      )

    self._level_crossing = to_range("crossing")
    self._level_speed = to_range("speed")
    self._level_distance = to_range("distance")

    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    """``[active, dy, time_to_arrival, ball_speed]``, as the behaviour system sends."""
    return self.command_buf

  ##
  # Geometry helpers.
  ##

  def to_robot_frame(self, pos_w: torch.Tensor, vec_w: torch.Tensor):
    """Planar position and vector in the goalie's yaw frame {r}."""
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w[:, :2] - self.robot.data.root_link_pos_w[:, :2]
    pos_r = torch.stack(
      [
        cos_h * delta[:, 0] + sin_h * delta[:, 1],
        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
      ],
      dim=-1,
    )
    vec_r = torch.stack(
      [
        cos_h * vec_w[:, 0] + sin_h * vec_w[:, 1],
        -sin_h * vec_w[:, 0] + cos_h * vec_w[:, 1],
      ],
      dim=-1,
    )
    return pos_r, vec_r

  def _predict_crossing(
    self, pos_r: torch.Tensor, vel_r: torch.Tensor, plane_x: float = 0.0
  ):
    """Where and when the ball crosses a plane ahead of the goalie (x = plane_x).

    ``plane_x`` is 0 for the goalie's own line and negative for the goal line behind
    it. Returns ``(dy, time, reaches)``. The ball rolls straight and slows at a
    constant rate, so the crossing point needs no deceleration term, but the time does.
    """
    speed = torch.linalg.norm(vel_r, dim=-1)
    safe_speed = speed.clamp(min=1e-6)
    direction = vel_r / safe_speed.unsqueeze(-1)

    # Distance along the ball's path to the goalie's line. Positive only when the ball
    # is in front and closing.
    closing = -direction[:, 0]
    distance = (pos_r[:, 0] - plane_x) / closing.clamp(min=1e-6)
    approaching = (closing > 1e-3) & (pos_r[:, 0] > plane_x)

    dy = pos_r[:, 1] + direction[:, 1] * distance

    # v * t - 0.5 * a * t^2 = distance, taking the first root.
    a = self.cfg.rolling_deceleration
    discriminant = speed.square() - 2.0 * a * distance
    reaches = approaching & (discriminant > 0.0) & (speed > self.cfg.min_shot_speed)
    time = (speed - torch.sqrt(discriminant.clamp(min=0.0))) / max(a, 1e-6)
    return dy, time, reaches

  ##
  # Command lifecycle.
  ##

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)

    # Most shots come from the current drill; the rest are drawn from the ones already
    # passed, so a keeper working on wide shots keeps being served the simple ones.
    level_ids = torch.full((n,), self.level, device=self.device, dtype=torch.long)
    if self.level > 0 and self.cfg.mix_fraction > 0.0:
      revisit = torch.rand(n, device=self.device) < self.cfg.mix_fraction
      earlier = torch.randint(0, self.level, (n,), device=self.device)
      level_ids = torch.where(revisit, earlier, level_ids)

    def sample(ranges: torch.Tensor) -> torch.Tensor:
      low, high = ranges[level_ids, 0], ranges[level_ids, 1]
      return low + (high - low) * torch.rand(n, device=self.device)

    distance = sample(self._level_distance)
    crossing = sample(self._level_crossing)
    speed = sample(self._level_speed)
    lateral = r.uniform_(*self.cfg.lateral).clone()
    self.kick_delay[env_ids] = r.uniform_(*self.cfg.kick_delay).clone()

    origins = self._env.scene.env_origins[env_ids]
    radius = self.cfg.ball_radius

    # The goalie is reset facing +x, so shots are laid out along world axes.
    start = torch.stack([distance, lateral, torch.full_like(distance, radius)], dim=-1)
    target = torch.stack(
      [torch.zeros_like(crossing), crossing, torch.full_like(crossing, radius)], dim=-1
    )
    direction = target[:, :2] - start[:, :2]
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(
      min=1e-6
    )

    # A shot that stops before the line teaches nothing, so give it enough speed to
    # arrive with a little to spare.
    travel = torch.linalg.norm(target[:, :2] - start[:, :2], dim=-1)
    min_speed = torch.sqrt(2.0 * self.cfg.rolling_deceleration * travel) * 1.1
    self.shot_speed[env_ids] = torch.maximum(speed, min_speed)

    pose = torch.zeros(n, 7, device=self.device)
    pose[:, :3] = start + origins
    pose[:, 3] = 1.0  # Identity quaternion.
    self.ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.ball.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )
    self._pending_forward = True

    self.shot_direction[env_ids] = direction
    self.time_since_resample[env_ids] = 0.0
    self.kicked[env_ids] = False
    self.was_moving[env_ids] = False
    self.touched[env_ids] = False
    self.finished[env_ids] = False
    self.on_target_shot[env_ids] = False
    self.touched_now[env_ids] = False
    self.saved_now[env_ids] = False
    self.scored_now[env_ids] = False
    self.cleared_now[env_ids] = 0.0

    # The estimate starts from the resting ball, which is what the real filter would
    # have converged on while it sat there.
    self.est_pos[env_ids] = pose[:, :2]
    self.est_vel[env_ids] = 0.0
    self.history_pos[env_ids] = pose[:, :2].unsqueeze(1)
    self.history_vel[env_ids] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # super().reset() logs the metrics and zeroes them, so the counters they are built
    # from have to be cleared here too, after it has read them.
    extras = super().reset(env_ids)
    if isinstance(env_ids, torch.Tensor):
      self.shots_finished[env_ids] = 0.0
      self.shots_on_target[env_ids] = 0.0
      self.shots_saved[env_ids] = 0.0
      self.shots_touched[env_ids] = 0.0
      self.shots_conceded[env_ids] = 0.0
      self.shots_cleared[env_ids] = 0.0
      self.clearance_total[env_ids] = 0.0
    self._pending_forward = False
    return extras

  def _kick(self) -> None:
    """Launch the balls whose delay has elapsed, rolling without slipping."""
    due = (~self.kicked) & (self.time_since_resample >= self.kick_delay)
    env_ids = due.nonzero(as_tuple=False).flatten()
    if len(env_ids) == 0:
      return
    velocity = torch.zeros(len(env_ids), 6, device=self.device)
    linear = self.shot_direction[env_ids] * self.shot_speed[env_ids].unsqueeze(-1)
    velocity[:, :2] = linear
    # Rolling without slipping: omega = (z x v) / radius.
    velocity[:, 3] = -linear[:, 1] / self.cfg.ball_radius
    velocity[:, 4] = linear[:, 0] / self.cfg.ball_radius
    self.ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    self.kicked[env_ids] = True

  def _update_estimate(self, dt: float) -> None:
    """Advance the simulated vision + filter estimate of the ball."""
    cfg = self.cfg
    true_pos = self.ball.data.root_link_pos_w[:, :2]
    true_vel = self.ball.data.root_link_lin_vel_w[:, :2]

    self.estimate_timer += dt
    due = self.estimate_timer >= 1.0 / cfg.estimate_rate_hz
    if not bool(due.any()):
      return
    self.estimate_timer = torch.where(
      due, torch.zeros_like(self.estimate_timer), self.estimate_timer
    )

    depth = self.history_pos.shape[1]
    self.history_index = (self.history_index + 1) % depth
    self.history_pos[:, self.history_index] = true_pos
    self.history_vel[:, self.history_index] = true_vel

    def delayed(buffer: torch.Tensor, lag_s: float) -> torch.Tensor:
      steps = int(round(lag_s * cfg.estimate_rate_hz))
      index = (self.history_index - steps) % depth
      return buffer[:, index]

    # Position: late, and noisier the further away the ball is.
    lagged_pos = delayed(self.history_pos, cfg.estimate_latency_s)
    range_to_ball = torch.linalg.norm(
      lagged_pos - self.robot.data.root_link_pos_w[:, :2], dim=-1
    )
    sigma = cfg.estimate_pos_noise[0] + cfg.estimate_pos_noise[1] * range_to_ball
    measured = lagged_pos + torch.randn_like(lagged_pos) * sigma.unsqueeze(-1)

    # Velocity: dead time, then a first-order catch-up. Together these reproduce the
    # measured ramp (12% of the true speed 0.05 s after the kick, 85% at 0.25 s).
    target_vel = delayed(self.history_vel, cfg.velocity_dead_time_s)
    alpha = 1.0 - torch.exp(
      torch.tensor(
        -1.0 / (cfg.estimate_rate_hz * cfg.velocity_lag_s), device=self.device
      )
    )
    new_vel = self.est_vel + alpha * (target_vel - self.est_vel)
    new_vel = new_vel + torch.randn_like(new_vel) * cfg.estimate_vel_noise

    update = due.unsqueeze(-1)
    self.est_pos = torch.where(update, measured, self.est_pos)
    self.est_vel = torch.where(update, new_vel, self.est_vel)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids  # The whole batch is refreshed; the command is a function of state.
    dt = self._env.step_dt

    if self._pending_forward:
      # A timer-expiry resample teleported the ball after this step's forward(), so
      # refresh kinematics before anything reads the ball's pose.
      self._pending_forward = False
      self._env.sim.forward()

    self.time_since_resample += dt
    self._kick()
    self._update_estimate(dt)

    # What the policy is told, from the estimate.
    est_pos_r, est_vel_r = self.to_robot_frame(self.est_pos, self.est_vel)
    dy, time_to_cross, reaches = self._predict_crossing(est_pos_r, est_vel_r)
    active = reaches & (time_to_cross < self.cfg.max_time)
    speed = torch.linalg.norm(est_vel_r, dim=-1)

    self.command_buf[:, 0] = active.float()
    self.command_buf[:, 1] = torch.where(
      active, dy.clamp(-self.cfg.max_dy, self.cfg.max_dy), torch.zeros_like(dy)
    )
    self.command_buf[:, 2] = torch.where(
      active,
      time_to_cross.clamp(0.0, self.cfg.max_time),
      torch.zeros_like(time_to_cross),
    )
    self.command_buf[:, 3] = torch.where(
      active, speed.clamp(0.0, self.cfg.max_speed), torch.zeros_like(speed)
    )

    # What actually happens, for the rewards.
    ball_pos_r, ball_vel_r = self.to_robot_frame(
      self.ball.data.root_link_pos_w, self.ball.data.root_link_lin_vel_w[:, :2]
    )
    true_dy, true_time, true_reaches = self._predict_crossing(ball_pos_r, ball_vel_r)
    self.true_crossing = torch.where(true_reaches, true_dy, self.true_crossing)
    # Where the ball actually is, sideways, in the goalie's frame. Once it is nearly
    # here, that is what the goalie has to be in front of, not a prediction.
    self.ball_offset = ball_pos_r[:, 1]
    self.true_time_to_cross = torch.where(
      true_reaches, true_time, torch.full_like(true_time, self.cfg.max_time)
    )
    self.on_target = self.kicked & ~self.finished & true_reaches

    contact = self._contact_with_robot()
    newly_touched = contact & self.kicked & ~self.finished & ~self.touched
    self.touched_now = newly_touched
    self.touched |= newly_touched

    ball_speed = torch.linalg.norm(ball_vel_r, dim=-1)
    moving_now = self.kicked & (ball_speed > self.cfg.min_shot_speed)

    # The goal does not move with the goalie. Outcomes are judged against a goal fixed
    # at the environment origin, where the goalie starts: a goalie that steps aside
    # must not be able to carry its goal out of the ball's way.
    ball_pos_o = (
      self.ball.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
    )
    ball_vel_o = self.ball.data.root_link_lin_vel_w[:, :2]

    # As the ball starts rolling, work out whether it was ever going in. A shot that
    # would have missed anyway is not a save, however it ends.
    starting = moving_now & ~self.was_moving
    if bool(starting.any()):
      goal_dy, _, reaches_goal = self._predict_crossing(
        ball_pos_o, ball_vel_o, plane_x=-self.cfg.goal_line_depth
      )
      would_score = reaches_goal & (goal_dy.abs() < self.cfg.goal_half_width)
      self.on_target_shot = torch.where(starting, would_score, self.on_target_shot)
    self.was_moving |= moving_now

    # A shot is over when the ball reaches the goal line, stops, or is on its way back
    # out. Stopping 0.1 m behind the goalie is not the end of it: a ball the goalie
    # got a toe to can still roll in, and that is a goal, not a save.
    at_goal_line = ball_pos_o[:, 0] < -self.cfg.goal_line_depth
    stopped = self.was_moving & (ball_speed < self.cfg.min_shot_speed)
    going_away = (
      self.was_moving
      & (ball_vel_o[:, 0] > self.cfg.min_shot_speed)
      & (ball_pos_o[:, 0] > 0.0)
    )
    over = self.kicked & ~self.finished & (at_goal_line | stopped | going_away)

    # Where the ball is headed now, which is what a touch has to change. A ball that
    # can no longer reach the goal is fully defused, whether it was stopped dead or
    # sent wide; one still bound for the posts is not defused at all.
    projected_dy, _, still_reaches = self._predict_crossing(
      ball_pos_o, ball_vel_o, plane_x=-self.cfg.goal_line_depth
    )
    outside_post = (projected_dy.abs() - self.cfg.goal_half_width).clamp(min=0.0)
    self.defused = torch.where(
      still_reaches,
      (outside_post / self.cfg.defuse_margin).clamp(0.0, 1.0),
      torch.ones_like(outside_post),
    )

    # The ball rolls straight and slows at a constant rate, so it comes to rest
    # speed^2 / 2a further along its path.
    ball_speed_o = torch.linalg.norm(ball_vel_o, dim=-1)
    self.rest_x = ball_pos_o[:, 0] + ball_vel_o[:, 0] * ball_speed_o / (
      2.0 * max(self.cfg.rolling_deceleration, 1e-6)
    )

    in_the_goal = at_goal_line & (ball_pos_o[:, 1].abs() < self.cfg.goal_half_width)
    self.scored_now = over & in_the_goal
    self.saved_now = over & self.on_target_shot & ~in_the_goal
    self.finished |= over

    # How much of a clearance the save was, judged the moment it is decided: a ball
    # stopped dead at the goalie's feet is worth nothing here, one sent
    # clear_distance up the field is worth the full amount.
    self.cleared_now = torch.where(
      self.saved_now,
      (self.rest_x / self.cfg.clear_distance).clamp(0.0, 1.0),
      torch.zeros_like(self.rest_x),
    )

    resolved_on_target = over & self.on_target_shot
    decided = int(resolved_on_target.sum())
    if decided > 0:
      saved_fraction = (self.saved_now & resolved_on_target).float().sum() / decided
      # One step can resolve many shots at once, so the average is moved as if it had
      # seen them one at a time.
      weight = 1.0 - (1.0 - self.cfg.save_rate_smoothing) ** decided
      self.recent_save_rate += weight * (saved_fraction - self.recent_save_rate)
      self.shots_since_level += decided

    self.shots_finished += over.float()
    self.shots_on_target += (over & self.on_target_shot).float()
    self.shots_saved += self.saved_now.float()
    self.shots_touched += (over & self.touched).float()
    self.shots_conceded += self.scored_now.float()
    self.shots_cleared += (self.saved_now & (self.rest_x > self.cfg.cleared_at)).float()
    self.clearance_total += torch.where(
      self.saved_now, self.rest_x.clamp(min=0.0), torch.zeros_like(self.rest_x)
    )

  def _contact_with_robot(self) -> torch.Tensor:
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).reshape(self.num_envs, -1).any(dim=-1)

  def _update_metrics(self) -> None:
    # Rates are over shots that were going in: those are the ones a goalie is judged
    # on. touch_rate is a diagnostic, not a score, because a touch that deflects the
    # ball into the goal is still a goal.
    on_target = self.shots_on_target.clamp(min=1.0)
    self.metrics["save_rate"] = self.shots_saved / on_target
    self.metrics["concede_rate"] = self.shots_conceded / on_target
    self.metrics["touch_rate"] = self.shots_touched / self.shots_finished.clamp(min=1.0)
    self.metrics["shots_on_target"] = self.shots_on_target
    # clear_rate is over shots that were going in, like save_rate, so it can never
    # exceed it; clearance is the mean distance a saved ball ends up up the field.
    self.metrics["clear_rate"] = self.shots_cleared / on_target
    self.metrics["clearance"] = self.clearance_total / self.shots_saved.clamp(min=1.0)
    active = self.command_buf[:, 0] > 0.5
    self.metrics["command_dy_error"] = torch.where(
      active,
      (self.command_buf[:, 1] - self.true_crossing).abs(),
      self.metrics["command_dy_error"],
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      if self.command_buf[batch, 0] < 0.5:
        continue
      heading = float(self.robot.data.heading_w[batch])
      root = self.robot.data.root_link_pos_w[batch].cpu().numpy()
      dy = float(self.command_buf[batch, 1])
      # The commanded crossing point, on the goalie's lateral line.
      point = (
        root[0] - dy * float(torch.sin(torch.tensor(heading))),
        root[1] + dy * float(torch.cos(torch.tensor(heading))),
        self.cfg.ball_radius,
      )
      visualizer.add_sphere(
        center=point,
        radius=0.05,
        color=self.cfg.viz.crossing_color,
        label=f"shot_crossing_{batch}",
      )


@dataclass(kw_only=True)
class ShotCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_name: str = "ball"
  contact_sensor_name: str = "ball_robot_contact"

  ball_radius: float = 0.095
  """FIFA size 3, the ball the Middle division plays with."""

  # Shot geometry, in the goalie's start frame: +x points at the shooter.
  levels: tuple[ShotLevel, ...] = ()
  """The drills, easiest first. Empty means one level covering the full envelope."""
  start_level: int = 0
  mix_fraction: float = 0.25
  """Share of shots drawn from levels already passed, so they are not forgotten."""
  save_rate_smoothing: float = 0.01
  """Weight of one resolved shot in the running save rate the curriculum advances on."""
  lateral: tuple[float, float] = (-1.5, 1.5)
  kick_delay: tuple[float, float] = (0.5, 1.5)

  rolling_deceleration: float = 0.5
  """Rolling resistance (m/s^2), used to predict the crossing time."""
  goal_half_width: float = 1.25
  """Half the goal mouth (m). HSL Middle goals are 2.4-2.6 m wide."""
  goal_line_depth: float = 0.2
  """How far behind the *environment origin* the goal line is (m), where the goalie
  starts. PLAN.md has the goalie standing 0.2 m off its line. The goal is fixed there,
  not carried around with the robot."""
  min_shot_speed: float = 0.3
  defuse_margin: float = 0.5
  """How far outside the post a deflected ball has to be headed (m) to count as fully
  out of danger."""
  clear_distance: float = 3.0
  """How far up the field from the goalie's start (m) a saved ball has to come to rest
  for the save to count as a full clearance."""
  cleared_at: float = 1.0
  """Rest distance (m) past which a save is counted in the clear_rate metric."""

  # Simulated ball estimate, from the NUSim validation of the real chain.
  estimate_rate_hz: float = 30.0
  estimate_latency_s: float = 0.04
  estimate_pos_noise: tuple[float, float] = (0.01, 0.02)
  """Position noise standard deviation (m): base + per metre of range."""
  velocity_dead_time_s: float = 0.06
  velocity_lag_s: float = 0.12
  estimate_vel_noise: float = 0.2

  # Clipping, matching the deployed contract.
  max_dy: float = 1.5
  max_time: float = 3.0
  max_speed: float = 6.0

  @dataclass
  class VizCfg:
    crossing_color: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 0.8)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ShotCommand:
    return ShotCommand(self, env)
