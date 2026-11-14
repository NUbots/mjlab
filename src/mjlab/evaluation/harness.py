"""Batched harness for evaluating a trained velocity policy.

The policy's observations are built by its task's own environment -- they are
noise-shaped, delayed and normalised, and hand-rebuilding that would be
reimplementing the training code with different bugs. The environment supplies
observations and applies actions; the *measurements* come from raw simulator
state via :meth:`~mjlab.evaluation.metrics.EvalState.from_entity`.

Nothing here is specific to one robot: the task id picks the robot, the metrics
read only the root link, and pushes go through the root body.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.evaluation.metrics import EvalState, VelocityTrace, WalkMetrics
from mjlab.evaluation.push import PushDriver, PushMetrics, PushPlan
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

TASK_ID = "Mjlab-Velocity-Flat-Booster-K1"
"""Default task supplying the robot, terrain, and the policy's observation,
action and command pipeline."""

RANDOMISATION_EVENTS: tuple[str, ...] = (
  "foot_friction",
  "encoder_bias",
  "base_com",
  "pd_gains",
)
"""Startup events that perturb the model away from nominal.

Removed for evaluation: they are a training device, and leaving them in would
put every environment on a slightly different robot and make a batch a sample
over models rather than over commands.
"""


def command_grid(
  vx: tuple[float, ...],
  vy: tuple[float, ...],
  wz: tuple[float, ...],
  num_envs: int,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Tile a command grid across environments.

  The three axes form a grid, which is then repeated (and truncated) to fill
  ``num_envs``.

  Returns:
    Shape ``(num_envs, 3)`` commands.
  """
  points = torch.tensor(
    [(x, y, w) for x in vx for y in vy for w in wz], device=device, dtype=torch.float32
  )
  if points.numel() == 0:
    raise ValueError("command grid is empty")
  repeats = -(-num_envs // points.shape[0])  # ceil
  return points.repeat(repeats, 1)[:num_envs].contiguous()


def _twist(env: ManagerBasedRlEnv) -> UniformVelocityCommand:
  term = env.command_manager.get_term("twist")
  assert isinstance(term, UniformVelocityCommand)
  return term


def prescribe_velocity_commands(env: ManagerBasedRlEnv, command: torch.Tensor) -> None:
  """Pin the task's velocity command to ``command``, per environment.

  The policy *sees* the command in its observations, so it has to be in place
  before the observation is built rather than written over the buffer
  afterwards. Replacing the term's resampling hook guarantees it: whenever the
  term would sample, it writes the prescribed value instead.
  """
  term = _twist(env)

  def _resample(env_ids: torch.Tensor) -> None:
    term.vel_command_b[env_ids] = command[env_ids]
    term.vel_command_w[env_ids] = command[env_ids]
    term.is_standing_env[env_ids] = False
    term.is_heading_env[env_ids] = False
    term.is_world_env[env_ids] = False
    term.is_forward_env[env_ids] = False

  # A bound method is replaced by a plain function on purpose.
  term._resample_command = _resample  # ty: ignore[invalid-assignment]
  _resample(torch.arange(env.num_envs, device=env.device))


def build_env(task_id: str, num_envs: int, device: str) -> ManagerBasedRlEnv:
  """The task's play environment, stripped of anything that would interfere.

  - domain randomisation events are dropped (see :data:`RANDOMISATION_EVENTS`)
    and the reset pose jitter is zeroed, so every environment is the nominal
    robot in the nominal pose;
  - terminations are removed, so a fallen robot stays fallen and is measured
    instead of being reset upright mid-run;
  - the command term never resamples on its own; the harness sets it.
  """
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs

  for name in RANDOMISATION_EVENTS:
    cfg.events.pop(name, None)
  reset_base = cfg.events.get("reset_base")
  if reset_base is not None:
    reset_base.params["pose_range"] = {}
    reset_base.params["velocity_range"] = {}

  cfg.terminations = {}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  # The command term rejects a heading range it has been told not to use.
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.resampling_time_range = (1.0e9, 1.0e9)

  return ManagerBasedRlEnv(cfg=cfg, device=device)


class RlEvalHarness:
  """Batched playback of a trained policy, measured from raw state."""

  def __init__(
    self,
    checkpoint: Path,
    num_envs: int,
    device: str = "cuda:0",
    task_id: str = TASK_ID,
  ) -> None:
    """
    Args:
      checkpoint: rsl-rl checkpoint. Must have been trained against the current
        config of ``task_id``: one from an older observation layout fails to
        load with a shape mismatch.
      num_envs: Robots simulated in parallel.
      device: Torch device.
      task_id: Registered task the checkpoint was trained on.
    """
    if not checkpoint.is_file():
      raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    self.num_envs = num_envs
    self.device = device
    self.task_id = task_id
    self.env = build_env(task_id, num_envs, device)

    # Loaded through the runner rather than from the state dict, because the
    # observation normalisation lives inside the inference policy it returns.
    agent_cfg = load_rl_cfg(task_id)
    self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(self.wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self.policy = runner.get_inference_policy(device=device)

    self.robot: Entity = self.env.scene["robot"]
    self.control_dt = float(self.env.step_dt)
    self.push_body_id = 0
    """Index of the body a push is applied to: the root (floating-base) body,
    so a shove goes through the torso's centre of mass on any robot."""

  @property
  def push_body_name(self) -> str:
    return self.robot.body_names[self.push_body_id]

  def robot_mass(self) -> float:
    """Total mass of one robot, in kg.

    Summed over the entity's own bodies rather than the compiled model's, which
    also carries the terrain. It turns a push magnitude expressed as a velocity
    change into the force to apply.
    """
    body_ids = self.robot.indexing.body_ids.cpu().numpy()
    return float(self.env.sim.mj_model.body_mass[body_ids].sum())

  def state(self) -> EvalState:
    return EvalState.from_entity(self.robot)

  def _reset(self, command: torch.Tensor) -> TensorDict:
    """Reset every environment under ``command`` and return the observation."""
    self.wrapped.reset()
    prescribe_velocity_commands(self.env, command)
    return self.wrapped.get_observations()

  def run(self, command: torch.Tensor, duration: float, warmup_s: float = 0.0):
    """Hold ``command`` for ``duration`` seconds, recording metrics.

    Args:
      command: Shape ``(N, 3)`` velocity command per environment.
      duration: Simulated seconds.
      warmup_s: Seconds kept out of the velocity averages; see
        :class:`~mjlab.evaluation.metrics.WalkMetrics`.
    """
    metrics = WalkMetrics(command_b=command, dt=self.control_dt, warmup_s=warmup_s)
    with torch.inference_mode():
      obs = self._reset(command)
      for _ in range(int(duration / self.control_dt)):
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        metrics.record(self.state())
    return metrics

  def run_profile(self, schedule: torch.Tensor) -> VelocityTrace:
    """Follow a time-varying command, recording every step.

    The command is written into the task's command term at the top of each
    control step, so the observation the policy acts on at step ``k + 1``
    carries the command issued at step ``k`` -- the same one-step lag the robot
    has, and far shorter than the ramps a profile uses.

    Args:
      schedule: Shape ``(T, N, 3)`` commands, one row per control step.
    """
    if schedule.ndim != 3 or schedule.shape[1:] != (self.num_envs, 3):
      raise ValueError(
        f"schedule must have shape (T, {self.num_envs}, 3), got {tuple(schedule.shape)}"
      )
    schedule = schedule.to(self.device)
    term = _twist(self.env)
    trace = VelocityTrace(dt=self.control_dt)
    with torch.inference_mode():
      obs = self._reset(schedule[0])
      for command in schedule:
        term.vel_command_b[:] = command
        term.vel_command_w[:] = command
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        trace.record(command, self.state())
    return trace

  def run_push(self, plan: PushPlan) -> PushMetrics:
    """Walk under a fixed command, take one shove per environment, recover.

    The force is written straight onto the robot rather than through the
    task's ``push_robot`` event, which is a training device (a random velocity
    teleported onto the base) and which the play config drops.
    """
    if plan.num_envs != self.num_envs:
      raise ValueError(
        f"plan is for {plan.num_envs} environments, harness has {self.num_envs}"
      )
    # The onsets are step indices: a plan built for another control rate would
    # push at the wrong times without complaint.
    if abs(plan.dt - self.control_dt) > 1e-9:
      raise ValueError(
        f"plan was built for a {1 / plan.dt:.0f} Hz controller, this one runs at "
        f"{1 / self.control_dt:.0f} Hz"
      )
    # The reset is inside the inference block: a battery calls this once per
    # magnitude, and delay buffers first touched under inference mode and then
    # reset outside it raise.
    with torch.inference_mode():
      obs = self._reset(plan.command)
      driver = PushDriver(plan, self.robot, self.push_body_id)
      metrics = PushMetrics(plan)
      try:
        for step in range(plan.num_steps):
          # Before the step, so the wrench acts over the physics this step
          # integrates.
          driver.apply(step)
          obs, _, _, _ = self.wrapped.step(self.policy(obs))
          metrics.record(self.state())
      finally:
        driver.clear()
    return metrics

  def close(self) -> None:
    self.env.close()
