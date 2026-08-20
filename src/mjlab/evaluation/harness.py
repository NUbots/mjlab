"""Batched harnesses for the walk-engine comparison.

Two engines, two robot models, one set of metrics. The engines are as different
as they can be -- one is a hand-tuned trajectory generator, the other a learned
policy with its own observation pipeline -- so the thing that has to be shared
is everything *around* them:

- the plant, built here from one place for both (:func:`eval_scene_cfg`), taking
  its terrain and solver settings from the same task config the policy trains
  against, so the two engines meet the same floor and the same integrator;
- the measurement, via
  :meth:`~mjlab.evaluation.metrics.EvalState.from_entity`, which reads the same
  robot state in both harnesses.

What is *not* shared is each engine's own home stance and control rate. The walk
engine runs at 100 Hz from the stance its own trajectory generator holds; the
policy runs at the task's 50 Hz from the keyframe it was trained from. Forcing
either onto the other's terms would measure the transplant rather than the
controller.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Literal

import mujoco
import numpy as np
import torch

from mjlab.controllers.quintic_walk.controller import (
  JOINT_NAMES,
  QuinticWalkController,
  detect_planted_phase,
  sole_poses_from_body_states,
)
from mjlab.controllers.quintic_walk.playback import PLANTS, POSTURE, WalkPlayback
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
  WalkParameters,
)
from mjlab.entity import Entity
from mjlab.evaluation.metrics import EvalState, WalkMetrics
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.utils.lab_api.math import matrix_from_quat

EvalPlant = Literal["eval", "training", "nubots-sim", "nubots-xml"]
"""Robot models the evaluation can run on; see :data:`PLANTS`."""

RL_PLANTS: tuple[str, ...] = ("eval", "training")
"""Plants the learned policy can be evaluated on.

The policy's observations are wired to mjlab's sensor and site names, which the
NUbots MJCFs do not carry, so those two plants are quintic-only.
"""

TASK_ID = "Mjlab-Velocity-Flat-Nubots-Nugus"
"""Task supplying the terrain, solver settings and -- for the policy -- the
observation, action and command pipeline."""

QUINTIC_CONTROL_HZ = 100.0
"""Rate the walk engine runs at on the robot."""

RANDOMISATION_EVENTS: tuple[str, ...] = (
  "foot_friction",
  "encoder_bias",
  "base_com",
  "pd_gains",
)
"""Startup events that perturb the model away from nominal.

Removed for evaluation. These are domain randomisation, i.e. a training device;
evaluating at nominal is the point of this pipeline, and leaving them in would
put every environment on a slightly different robot and make a batch of
environments a sample over models rather than over commands.
"""

FOOT_BODY_NAMES = ("left_foot", "right_foot")


def _flat_env_cfg():
  """The task config both engines take their plant from."""
  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

  return nubots_nugus_flat_env_cfg(play=True)


def eval_scene_cfg(plant: EvalPlant, num_envs: int):
  """Scene and simulation configs for one plant.

  Takes the terrain and the solver settings from the velocity task, so a
  quintic run and a policy run differ in the controller and nothing else, and
  swaps in the requested robot. Sensors are dropped: they are read-only, the
  walk engine does not consume them, and two of the plants are NUbots MJCFs
  that lack the sites they reference.
  """
  cfg = _flat_env_cfg()
  scene = cfg.scene
  scene.num_envs = num_envs
  scene.entities = {"robot": PLANTS[plant]()}
  scene.sensors = ()
  return scene, cfg.sim


def command_grid(
  vx: tuple[float, ...],
  vy: tuple[float, ...],
  wz: tuple[float, ...],
  num_envs: int,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Tile a command sweep across environments.

  The three axes form a grid, which is then repeated (and truncated) to fill
  ``num_envs``, so every grid point gets ``num_envs // len(grid)`` samples and
  the batch stays exactly the size that was asked for.

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


def constant_command(
  vx: float, vy: float, wz: float, num_envs: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
  """Shape ``(num_envs, 3)`` with the same command in every environment."""
  return torch.tensor([[vx, vy, wz]], device=device, dtype=torch.float32).expand(
    num_envs, 3
  )


def _body_ids(entity: Entity, names: tuple[str, ...], device) -> torch.Tensor:
  lookup = {name: index for index, name in enumerate(entity.body_names)}
  return torch.tensor([lookup[name] for name in names], device=device)


class QuinticEvalHarness:
  """Batched playback of the quintic walk engine.

  Mirrors the single-environment rig in
  :mod:`~mjlab.controllers.quintic_walk.playback`, with two differences that
  come from running through mjlab's stack rather than raw MuJoCo: joint targets
  go through the entity's actuators, so actuator latency applies where the plant
  configures it, and the physics runs at the task's timestep.
  """

  def __init__(
    self,
    plant: EvalPlant = "eval",
    num_envs: int = 64,
    device: str = "cuda:0",
    walk_params: WalkParameters = NUGUS_WALK_PARAMETERS,
    use_balance_control: bool = True,
    exact_ik: bool = False,
    control_hz: float = QUINTIC_CONTROL_HZ,
  ) -> None:
    scene_cfg, sim_cfg = eval_scene_cfg(plant, num_envs)
    self.plant = plant
    self.num_envs = num_envs
    self.device = device

    self.scene = Scene(scene_cfg, device)
    model = self.scene.compile()
    self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    self.robot: Entity = self.scene["robot"]

    self.physics_dt = float(self.sim.mj_model.opt.timestep)
    self.decimation = max(1, round((1.0 / control_hz) / self.physics_dt))
    self.control_dt = self.decimation * self.physics_dt

    self.controller = QuinticWalkController(
      num_envs=num_envs,
      device=device,
      walk_params=walk_params,
      use_balance_control=use_balance_control,
      exact_ik_model=self.sim.mj_model if exact_ik else None,
    )

    joint_lookup = {name: index for index, name in enumerate(self.robot.joint_names)}
    self._leg_ids = torch.tensor(
      [joint_lookup[name] for name in JOINT_NAMES], device=device
    )
    posture_names = tuple(POSTURE)
    self._posture_ids = torch.tensor(
      [joint_lookup[name] for name in posture_names], device=device
    )
    self._posture_targets = torch.tensor(
      [[POSTURE[name] for name in posture_names]], device=device, dtype=torch.float32
    ).expand(num_envs, len(posture_names))
    self._torso_id = int(_body_ids(self.robot, ("torso",), device)[0])
    self.foot_body_ids = _body_ids(self.robot, FOOT_BODY_NAMES, device)

    self._stance_root_pose, self._stance_joint_pos = self._solve_stance(plant)
    self.reset()

  def _solve_stance(self, plant: EvalPlant) -> tuple[torch.Tensor, torch.Tensor]:
    """Stance to start from, solved once on a single-environment CPU model.

    Reuses :class:`~mjlab.controllers.quintic_walk.playback.WalkPlayback`, so
    the batch starts in exactly the pose the single-environment rig does: the
    engine's own stopped stance, levelled onto the floor. Every environment is
    identical here, so solving it once and broadcasting is not an approximation.
    """
    playback = WalkPlayback(plant=plant)
    model, data = playback.model, playback.data
    root_pose = torch.tensor(data.qpos[:7], device=self.device, dtype=torch.float32)
    joint_pos = np.empty(len(self.robot.joint_names))
    for index, name in enumerate(self.robot.joint_names):
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      joint_pos[index] = data.qpos[model.jnt_qposadr[joint_id]]
    return root_pose, torch.tensor(joint_pos, device=self.device, dtype=torch.float32)

  def reset(self) -> None:
    """Put every environment in the engine's stance, at its own origin."""
    self.sim.reset()
    self.scene.reset()
    self.controller.reset()

    root_pose = self._stance_root_pose.unsqueeze(0).repeat(self.num_envs, 1)
    root_pose[:, :3] = root_pose[:, :3] + self.scene.env_origins
    self.robot.write_root_link_pose_to_sim(root_pose)
    self.robot.write_root_link_velocity_to_sim(
      torch.zeros(self.num_envs, 6, device=self.device)
    )
    joint_pos = self._stance_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
    self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    self.sim.forward()
    self.scene.update(dt=self.physics_dt)

  def sensed_phase(self) -> torch.Tensor:
    """Planted foot phase measured from foot geometry. Shape ``(N,)``."""
    data = self.robot.data
    left, right = sole_poses_from_body_states(
      data.body_link_pos_w[:, self._torso_id],
      data.body_link_quat_w[:, self._torso_id],
      data.body_link_pos_w[:, self.foot_body_ids[0]],
      data.body_link_quat_w[:, self.foot_body_ids[0]],
      data.body_link_pos_w[:, self.foot_body_ids[1]],
      data.body_link_quat_w[:, self.foot_body_ids[1]],
    )
    return detect_planted_phase(left, right)

  def state(self) -> EvalState:
    return EvalState.from_entity(self.robot, self.foot_body_ids)

  def step(self, command: torch.Tensor) -> None:
    """One control step of the engine plus its physics substeps.

    Args:
      command: Shape ``(N, 3)`` velocity command.
    """
    data = self.robot.data
    targets = self.controller.compute(
      dt=self.control_dt,
      velocity_command=command,
      torso_rotation_w=matrix_from_quat(data.root_link_quat_w),
      gyro_b=data.root_link_ang_vel_b,
      sensed_phase=self.sensed_phase(),
    )
    # The engine works in double; mjlab's target buffer is float32. This is the
    # output half of the precision boundary documented on QuinticWalkController.
    self.robot.set_joint_position_target(
      targets.to(self.robot.data.joint_pos_target.dtype), joint_ids=self._leg_ids
    )
    self.robot.set_joint_position_target(
      self._posture_targets, joint_ids=self._posture_ids
    )
    for _ in range(self.decimation):
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
    # mj_step leaves derived quantities a substep behind qpos; refresh before
    # anything reads the state, exactly as ManagerBasedRlEnv.step does.
    self.sim.forward()

  def run(
    self,
    command: torch.Tensor,
    duration: float,
    on_step: Callable[[int], None] | None = None,
  ) -> WalkMetrics:
    """Hold ``command`` for ``duration`` seconds, recording metrics."""
    metrics = WalkMetrics(command_b=command, dt=self.control_dt)
    metrics.start(self.state())
    for step in range(int(duration / self.control_dt)):
      self.step(command)
      metrics.record(self.state())
      if on_step is not None:
        on_step(step)
    return metrics

  @property
  def engine_state(self) -> torch.Tensor:
    """Shape ``(N,)`` :class:`EngineState` per environment."""
    return self.controller.generator.state

  def engine_state_counts(self) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in EngineState:
      count = int((self.engine_state == int(state)).sum())
      if count:
        counts[state.name] = count
    return counts


def prescribe_velocity_commands(env, command: torch.Tensor) -> None:
  """Pin the task's velocity command to ``command``, per environment.

  The velocity command term samples from a range, and the policy *sees* the
  command in its observations, so a prescribed evaluation command has to be in
  place before the observation is built rather than written over the buffer
  afterwards. Replacing the term's resampling hook is the one place that
  guarantees it: whenever the term would sample -- at reset, or on its resample
  timer -- it writes the prescribed value instead.
  """
  term = env.command_manager.get_term("twist")

  def _resample(env_ids: torch.Tensor) -> None:
    term.vel_command_b[env_ids] = command[env_ids]
    term.vel_command_w[env_ids] = command[env_ids]
    term.is_standing_env[env_ids] = False
    term.is_heading_env[env_ids] = False
    term.is_world_env[env_ids] = False
    term.is_forward_env[env_ids] = False

  term._resample_command = _resample  # noqa: SLF001
  _resample(torch.arange(env.num_envs, device=env.device))


def build_rl_env(
  plant: EvalPlant,
  num_envs: int,
  device: str,
  task_id: str = TASK_ID,
):
  """Build the velocity task environment for evaluation on one plant.

  The environment is used for its observation, action and command pipeline --
  the policy expects noise-shaped, delayed, clock-augmented, normalised
  observations, and reconstructing that by hand would be reimplementing the
  training code with different bugs. The *metrics* still come from raw state.

  Three changes to the play config, all of them to stop the environment from
  interfering with a measurement:

  - domain randomisation events are dropped (see :data:`RANDOMISATION_EVENTS`)
    and the reset pose jitter is zeroed, so every environment is the nominal
    robot in the nominal pose;
  - terminations are removed, so a fallen robot stays fallen and is measured
    instead of being teleported upright mid-run;
  - the command term is pinned by :func:`prescribe_velocity_commands`.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  if plant not in RL_PLANTS:
    raise ValueError(f"plant {plant!r} has no policy observations; use {RL_PLANTS}")

  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  cfg.scene.entities = {"robot": PLANTS[plant]()}

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


def load_policy(env, checkpoint: Path, device: str, task_id: str = TASK_ID):
  """Load a trained actor from an rsl-rl checkpoint.

  Returns:
    A tuple of the wrapped environment (which is what must be stepped) and the
    inference policy. Observation normalisation lives inside the returned
    policy, which is why the checkpoint is loaded through the runner rather
    than by reading the state dict.
  """
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  agent_cfg = load_rl_cfg(task_id)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return wrapped, runner.get_inference_policy(device=device)


class RlEvalHarness:
  """Batched playback of a trained policy, measured like the quintic engine."""

  def __init__(
    self,
    checkpoint: Path,
    plant: EvalPlant = "eval",
    num_envs: int = 64,
    device: str = "cuda:0",
    task_id: str = TASK_ID,
  ) -> None:
    self.plant = plant
    self.num_envs = num_envs
    self.device = device
    self.env = build_rl_env(plant, num_envs, device, task_id)
    self.wrapped, self.policy = load_policy(self.env, checkpoint, device, task_id)
    self.robot: Entity = self.env.scene["robot"]
    self.control_dt = float(self.env.step_dt)
    self.foot_body_ids = _body_ids(self.robot, FOOT_BODY_NAMES, device)

  @property
  def sim(self) -> Simulation:
    """The environment's simulation, for anything that watches the run."""
    return self.env.sim

  def state(self) -> EvalState:
    return EvalState.from_entity(self.robot, self.foot_body_ids)

  def run(
    self,
    command: torch.Tensor,
    duration: float,
    on_step: Callable[[int], None] | None = None,
  ) -> WalkMetrics:
    """Hold ``command`` for ``duration`` seconds, recording metrics."""
    obs, _ = self.wrapped.reset()
    prescribe_velocity_commands(self.env, command)
    obs = self.wrapped.get_observations()

    metrics = WalkMetrics(command_b=command, dt=self.control_dt)
    metrics.start(self.state())
    with torch.inference_mode():
      for step in range(int(duration / self.control_dt)):
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        metrics.record(self.state())
        if on_step is not None:
          on_step(step)
    return metrics

  def close(self) -> None:
    self.env.close()


def phase_name(phase: torch.Tensor) -> str:
  return Phase(int(phase)).name
