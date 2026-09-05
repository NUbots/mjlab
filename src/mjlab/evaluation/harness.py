"""Batched harnesses for the walk-engine comparison.

Three controllers, two robot models, one set of metrics. The controllers are as
different as they can be -- a hand-tuned trajectory generator, a policy
supervised to copy it, and a policy that learned to walk from reward, the last
with an observation pipeline of its own -- so the thing that has to be shared is
everything *around* them:

- the plant, built here from one place for all of them
  (:func:`eval_scene_cfg`), taking its terrain and solver settings from the same
  task config the policy trains against, so every controller meets the same
  floor and the same integrator;
- the measurement, via
  :meth:`~mjlab.evaluation.metrics.EvalState.from_entity`, which reads the same
  robot state in every harness.

What is *not* shared is each controller's own home stance and control rate. The
walk engine runs at 100 Hz from the stance its own trajectory generator holds;
the distilled policy runs at 100 Hz from the stance it holds at rest, which is
not the same stance; the reinforcement-learned policy runs at the task's 50 Hz
from the keyframe it was trained from. Forcing any of them onto another's terms
would measure the transplant rather than the controller.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Generic, Literal, Protocol, TypeVar

import mujoco
import numpy as np
import torch

from mjlab.controllers.distilled_walk.controller import (
  DistilledWalkController,
  HistoryInit,
)
from mjlab.controllers.distilled_walk.policy import (
  DEFAULT_POLICY_PATH,
  DistilledWalkPolicy,
)
from mjlab.controllers.quintic_walk.controller import (
  JOINT_NAMES,
  QuinticWalkController,
  detect_planted_phase,
  sole_poses_from_body_states,
)
from mjlab.controllers.quintic_walk.playback import (
  PLANTS,
  POSTURE,
  WalkPlayback,
  build_model,
  stand_on_leg_targets,
)
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
  WalkGenerator,
  WalkParameters,
)
from mjlab.entity import Entity
from mjlab.evaluation.competence import (
  DEFAULT_SHOVE,
  CompetenceGrid,
  EpisodeCompetence,
  EpisodeTable,
  ShoveCfg,
  ShoveDriver,
)
from mjlab.evaluation.metrics import EvalState, VelocityTrace, WalkMetrics
from mjlab.evaluation.push import PUSH_BODY, PushDriver, PushMetrics, PushPlan
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


class WalkController(Protocol):
  """What :class:`WalkEvalHarness` needs from a controller.

  Both :class:`~mjlab.controllers.quintic_walk.controller.QuinticWalkController`
  and
  :class:`~mjlab.controllers.distilled_walk.controller.DistilledWalkController`
  satisfy this. The generator is here because the two run the same phase clock,
  so a run of either can be asked what state its engine ended in.
  """

  generator: WalkGenerator

  def reset(self, env_ids: torch.Tensor | None = None) -> None: ...


ControllerT = TypeVar("ControllerT", bound=WalkController)


def _body_ids(entity: Entity, names: tuple[str, ...], device) -> torch.Tensor:
  lookup = {name: index for index, name in enumerate(entity.body_names)}
  return torch.tensor([lookup[name] for name in names], device=device)


def _robot_mass(entity: Entity, mj_model: mujoco.MjModel) -> float:
  """Total mass of one robot, in kg.

  Summed over the entity's own bodies rather than over the compiled model's,
  which also carries the terrain. It is what turns a push magnitude expressed
  as a velocity change into the force to apply, so a battery run on two plants
  that do not weigh the same still compares like with like.
  """
  return float(mj_model.body_mass[entity.indexing.body_ids.cpu().numpy()].sum())


class WalkEvalHarness(Generic[ControllerT]):
  """Plant, stance and measurement shared by the scripted-controller rigs.

  Mirrors the single-environment rig in
  :mod:`~mjlab.controllers.quintic_walk.playback`, with two differences that
  come from running through mjlab's stack rather than raw MuJoCo: joint targets
  go through the entity's actuators, so actuator latency applies where the plant
  configures it, and the physics runs at the task's timestep.

  A subclass supplies a ``controller`` -- which has to expose ``reset()`` and a
  ``generator``, so the phase clock reads the same way for both engines -- the
  stance to start from, and the targets for one control step. Everything else is
  the same code either way, which is what makes the two sets of numbers
  comparable.
  """

  controller: ControllerT
  _stance_root_pose: torch.Tensor
  _stance_joint_pos: torch.Tensor

  def __init__(
    self,
    plant: EvalPlant,
    num_envs: int,
    device: str,
    control_hz: float = QUINTIC_CONTROL_HZ,
  ) -> None:
    scene_cfg, sim_cfg = eval_scene_cfg(plant, num_envs)
    self.plant: EvalPlant = plant
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
    self.torso_body_id = int(_body_ids(self.robot, (PUSH_BODY,), device)[0])
    """Index of the body a push is applied to; see :data:`PUSH_BODY`."""
    self.foot_body_ids = _body_ids(self.robot, FOOT_BODY_NAMES, device)

  # What a subclass provides.

  def _solve_stance(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Root pose and joint positions to start every environment in."""
    raise NotImplementedError

  def compute_targets(self, command: torch.Tensor) -> torch.Tensor:
    """Shape ``(N, 12)`` leg joint targets for one control step."""
    raise NotImplementedError

  # Shared.

  def _read_stance(
    self, model: mujoco.MjModel, data: mujoco.MjData
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a standing pose off a single-environment CPU model.

    Every environment starts identically, so solving the stance once on one
    robot and broadcasting it is not an approximation.
    """
    root_pose = torch.tensor(data.qpos[:7], device=self.device, dtype=torch.float32)
    joint_pos = np.empty(len(self.robot.joint_names))
    for index, name in enumerate(self.robot.joint_names):
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      joint_pos[index] = data.qpos[model.jnt_qposadr[joint_id]]
    return root_pose, torch.tensor(joint_pos, device=self.device, dtype=torch.float32)

  def reset(self) -> None:
    """Put every environment in the controller's stance, at its own origin."""
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
      data.body_link_pos_w[:, self.torso_body_id],
      data.body_link_quat_w[:, self.torso_body_id],
      data.body_link_pos_w[:, self.foot_body_ids[0]],
      data.body_link_quat_w[:, self.foot_body_ids[0]],
      data.body_link_pos_w[:, self.foot_body_ids[1]],
      data.body_link_quat_w[:, self.foot_body_ids[1]],
    )
    return detect_planted_phase(left, right)

  def state(self) -> EvalState:
    return EvalState.from_entity(self.robot, self.foot_body_ids)

  def step(self, command: torch.Tensor) -> None:
    """One control step of the controller plus its physics substeps.

    Args:
      command: Shape ``(N, 3)`` velocity command.
    """
    targets = self.compute_targets(command)
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
    warmup_s: float = 0.0,
  ) -> WalkMetrics:
    """Hold ``command`` for ``duration`` seconds, recording metrics.

    ``warmup_s`` is passed straight to
    :class:`~mjlab.evaluation.metrics.WalkMetrics`, which keeps the run-up out
    of the averages while still dating a fall from the first step.
    """
    metrics = WalkMetrics(command_b=command, dt=self.control_dt, warmup_s=warmup_s)
    metrics.start(self.state())
    for step in range(int(duration / self.control_dt)):
      self.step(command)
      metrics.record(self.state())
      if on_step is not None:
        on_step(step)
    return metrics

  def run_profile(
    self,
    schedule: torch.Tensor,
    on_step: Callable[[int], None] | None = None,
  ) -> VelocityTrace:
    """Follow a time-varying command, recording every step.

    Args:
      schedule: Shape ``(T, N, 3)`` commands, one row per control step, as
        produced by :meth:`mjlab.evaluation.profile.Profile.commands`.
      on_step: Called with the step index after each control step.

    Returns:
      The commanded and measured velocity, step by step.
    """
    _check_schedule(schedule, self.num_envs)
    schedule = schedule.to(self.device)
    trace = VelocityTrace(dt=self.control_dt)
    for step, command in enumerate(schedule):
      self.step(command)
      trace.record(command, self.state())
      if on_step is not None:
        on_step(step)
    return trace

  def robot_mass(self) -> float:
    """Total mass of one robot, in kg. See :func:`_robot_mass`."""
    return _robot_mass(self.robot, self.sim.mj_model)

  def run_push(
    self,
    plan: PushPlan,
    on_step: Callable[[int], None] | None = None,
  ) -> PushMetrics:
    """Walk under a fixed command, take one shove per environment, recover.

    Args:
      plan: One pass of a push battery; see
        :mod:`mjlab.evaluation.push`.
      on_step: Called with the step index after each control step.

    Returns:
      The recorder, ready to be reduced.
    """
    _check_plan(plan, self.num_envs, self.control_dt)
    self.reset()
    driver = PushDriver(plan, self.robot, self.torso_body_id)
    metrics = PushMetrics(plan)
    metrics.start(self.state())
    try:
      for step in range(plan.num_steps):
        # Before the step, so the wrench is applied over the physics this step
        # integrates rather than over the next one.
        driver.apply(step)
        self.step(plan.command)
        metrics.record(self.state())
        if on_step is not None:
          on_step(step)
    finally:
      # ``xfrc_applied`` persists until it is overwritten, so a run that ended
      # mid-push would keep shoving whatever ran next on this harness.
      driver.clear()
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


class QuinticEvalHarness(WalkEvalHarness[QuinticWalkController]):
  """Batched playback of the quintic walk engine."""

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
    super().__init__(plant, num_envs, device, control_hz)
    self.controller = QuinticWalkController(
      num_envs=num_envs,
      device=device,
      walk_params=walk_params,
      use_balance_control=use_balance_control,
      exact_ik_model=self.sim.mj_model if exact_ik else None,
    )
    self._walk_params = walk_params
    self._use_balance_control = use_balance_control
    self._exact_ik = exact_ik
    self._stance_root_pose, self._stance_joint_pos = self._solve_stance()
    self.reset()

  def _solve_stance(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Stance to start from, solved once on a single-environment CPU model.

    Reuses :class:`~mjlab.controllers.quintic_walk.playback.WalkPlayback`, so
    the batch starts in exactly the pose the single-environment rig does: the
    engine's own stopped stance, levelled onto the floor.
    """
    playback = WalkPlayback(
      plant=self.plant,
      walk_params=self._walk_params,
      use_balance_control=self._use_balance_control,
      exact_ik=self._exact_ik,
    )
    return self._read_stance(playback.model, playback.data)

  def compute_targets(self, command: torch.Tensor) -> torch.Tensor:
    data = self.robot.data
    return self.controller.compute(
      dt=self.control_dt,
      velocity_command=command,
      torso_rotation_w=matrix_from_quat(data.root_link_quat_w),
      gyro_b=data.root_link_ang_vel_b,
      sensed_phase=self.sensed_phase(),
    )


class DistilledEvalHarness(WalkEvalHarness[DistilledWalkController]):
  """Batched playback of the distilled walk policy.

  The policy replaces the engine's trajectory generation and IK, and nothing
  else: the plant, the rate, the arm posture and the measurement are the
  quintic harness's, unchanged.

  It starts from its own standing pose rather than the engine's. The two differ
  -- the policy learned NUbots' numerically refined IK and this engine ships the
  idealised one, half a radian of knee apart -- and a robot planted in the other
  controller's stance would spend its first control steps being dragged out of
  it. Each controller gets its own home pose for the same reason each gets its
  own control rate; see ``scripts/eval/README.md``.
  """

  def __init__(
    self,
    policy_path: Path = DEFAULT_POLICY_PATH,
    plant: EvalPlant = "eval",
    num_envs: int = 64,
    device: str = "cuda:0",
    walk_params: WalkParameters = NUGUS_WALK_PARAMETERS,
    history_init: HistoryInit = "settled",
    track_teacher: bool = False,
    control_hz: float = QUINTIC_CONTROL_HZ,
  ) -> None:
    """Build the rig.

    Args:
      policy_path: ONNX export to run. Defaults to the policy NUbots deploys.
      track_teacher: Also run the walk engine the policy was distilled from, on
        the same commands, and accumulate the difference between the two sets of
        joint targets. See :meth:`teacher_tracking`.
    """
    super().__init__(plant, num_envs, device, control_hz)
    self.policy_path = Path(policy_path)
    self.controller = DistilledWalkController(
      num_envs=num_envs,
      policy=DistilledWalkPolicy.from_onnx(self.policy_path, device),
      device=device,
      walk_params=walk_params,
      history_init=history_init,
    )
    # The engine as the policy learned it: no balance correction, because
    # WalkDataCollector recorded the generator's own foot poses and never ran
    # FootController over them.
    self.teacher = (
      QuinticWalkController(
        num_envs=num_envs,
        device=device,
        walk_params=walk_params,
        use_balance_control=False,
      )
      if track_teacher
      else None
    )
    self._teacher_stance = torch.zeros(num_envs, len(JOINT_NAMES), device=device)
    self._error_sum = torch.zeros(num_envs, len(JOINT_NAMES), device=device)
    self._relative_sum = torch.zeros(num_envs, len(JOINT_NAMES), device=device)
    self._error_steps = 0
    self._stance_root_pose, self._stance_joint_pos = self._solve_stance()
    self.reset()

  def _solve_stance(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy's own standing pose, levelled onto the floor.

    Same levelling as the quintic rig, over the targets the policy holds at
    rest instead of the ones the engine holds.
    """
    model = build_model(self.plant)
    data = mujoco.MjData(model)
    stand_on_leg_targets(
      model,
      data,
      self.controller.home_targets[0].double().cpu().numpy(),
      stance_left=int(self.controller.generator.phase[0]) == int(Phase.LEFT),
    )
    return self._read_stance(model, data)

  def reset(self) -> None:
    super().reset()
    if self.teacher is not None:
      self.teacher.reset()
      self._teacher_stance = self.teacher.compute(
        self.control_dt, torch.zeros(self.num_envs, 3, device=self.device)
      ).to(self._teacher_stance.dtype)
      self.teacher.reset()
    self._error_sum.zero_()
    self._relative_sum.zero_()
    self._error_steps = 0

  def compute_targets(self, command: torch.Tensor) -> torch.Tensor:
    targets = self.controller.compute(self.control_dt, command)
    if self.teacher is not None:
      reference = self.teacher.compute(self.control_dt, command).to(targets.dtype)
      self._error_sum += (targets - reference).abs()
      self._relative_sum += (
        (targets - self.controller.home_targets) - (reference - self._teacher_stance)
      ).abs()
      self._error_steps += 1
    return targets

  def teacher_tracking(self) -> dict | None:
    """How far the policy's joint targets ran from the engine's, or ``None``.

    Two readings of the same difference, both means of absolute error over
    every step and environment of the run so far:

    ``mean_abs_error_rad``
      the raw gap. It is dominated by the IK the two solve with rather than by
      the distillation -- the policy learned NUbots' numerically refined
      solution and this engine ships the idealised one -- so it is a floor on
      how differently the two stand, not a measure of how well the policy
      learned the gait.
    ``stance_relative_mean_abs_error_rad``
      the same gap after subtracting each controller's own standing pose, which
      takes that calibration offset out and leaves the shape of the motion.
    """
    if self.teacher is None:
      return None
    steps = max(self._error_steps, 1)
    absolute = (self._error_sum / steps).mean(dim=0)
    relative = (self._relative_sum / steps).mean(dim=0)
    return {
      "steps": self._error_steps,
      "mean_abs_error_rad": float(absolute.mean()),
      "stance_relative_mean_abs_error_rad": float(relative.mean()),
      "per_joint_abs_error_rad": {
        name: float(value)
        for name, value in zip(JOINT_NAMES, absolute.tolist(), strict=True)
      },
      "stance_relative_per_joint_abs_error_rad": {
        name: float(value)
        for name, value in zip(JOINT_NAMES, relative.tolist(), strict=True)
      },
    }


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


def velocity_command_writer(env) -> Callable[[torch.Tensor], None]:
  """Return a function that pins the task's velocity command for one step.

  ``env`` is untyped for the same reason it is in
  :func:`prescribe_velocity_commands`: the command manager hands back the
  abstract ``CommandTerm``, and the buffers the observation reads live on the
  concrete velocity term.
  """
  term = env.command_manager.get_term("twist")

  def write(command: torch.Tensor) -> None:
    term.vel_command_b[:] = command
    term.vel_command_w[:] = command

  return write


def build_rl_env(
  plant: EvalPlant,
  num_envs: int,
  device: str,
  task_id: str = TASK_ID,
  episodic: bool = False,
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

  Args:
    episodic: Restore the training episode instead. A sweep, a profile and a
      push battery are all one long run whose end the harness chooses, so a
      termination would only interrupt a measurement. A competence grid is the
      opposite: it samples a distribution of *episodes*, and two of the five
      quantities it reports -- whether the robot fell, and how much of the
      episode it survived -- exist only if the episode can end. This restores
      the ``fell_over`` termination and the training episode length, and drops
      the curriculum, which is a training device that has no business mutating
      anything mid-measurement.
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

  if episodic:
    # The play config stretches the episode to 1e9 s so a run is never cut
    # short. Put the training length back: ep_len_frac is measured against it,
    # so a wrong denominator would silently rescale survival.
    cfg.episode_length_s = load_env_cfg(task_id, play=False).episode_length_s
    cfg.curriculum = {}
    failures = [
      name
      for name, term in cfg.terminations.items()
      if not getattr(term, "time_out", False)
    ]
    # `fell` is read off reset_terminated, i.e. every non-timeout term at once.
    # If another failure term is ever added, that binary stops meaning "fell"
    # and starts meaning "failed somehow" without anything saying so.
    if failures != ["fell_over"]:
      raise RuntimeError(
        f"episodic evaluation expects 'fell_over' as the only failure "
        f"termination, found {failures}"
      )
  else:
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
    episodic: bool = False,
  ) -> None:
    self.plant: EvalPlant = plant
    self.num_envs = num_envs
    self.device = device
    self.env = build_rl_env(plant, num_envs, device, task_id, episodic=episodic)
    self.wrapped, self.policy = load_policy(self.env, checkpoint, device, task_id)
    self.robot: Entity = self.env.scene["robot"]
    self.control_dt = float(self.env.step_dt)
    self.torso_body_id = int(_body_ids(self.robot, (PUSH_BODY,), device)[0])
    """Index of the body a push is applied to; see :data:`PUSH_BODY`."""
    self.foot_body_ids = _body_ids(self.robot, FOOT_BODY_NAMES, device)

  def robot_mass(self) -> float:
    """Total mass of one robot, in kg. See :func:`_robot_mass`."""
    return _robot_mass(self.robot, self.env.sim.mj_model)

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
    warmup_s: float = 0.0,
  ) -> WalkMetrics:
    """Hold ``command`` for ``duration`` seconds, recording metrics."""
    obs, _ = self.wrapped.reset()
    prescribe_velocity_commands(self.env, command)
    obs = self.wrapped.get_observations()

    metrics = WalkMetrics(command_b=command, dt=self.control_dt, warmup_s=warmup_s)
    metrics.start(self.state())
    with torch.inference_mode():
      for step in range(int(duration / self.control_dt)):
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        metrics.record(self.state())
        if on_step is not None:
          on_step(step)
    return metrics

  def run_profile(
    self,
    schedule: torch.Tensor,
    on_step: Callable[[int], None] | None = None,
  ) -> VelocityTrace:
    """Follow a time-varying command, recording every step.

    The command is written into the task's command term at the top of each
    control step, so the observation the policy acts on at step ``k + 1``
    carries the command issued at step ``k``. That one-step lag is the same one
    the robot has and is far shorter than the ramps a profile uses; rebuilding
    the observation mid-step to remove it would push the environment's noise and
    delay buffers twice per step, which would be a worse distortion.

    Args:
      schedule: Shape ``(T, N, 3)`` commands, one row per control step.
      on_step: Called with the step index after each control step.
    """
    _check_schedule(schedule, self.num_envs)
    schedule = schedule.to(self.device)
    self.wrapped.reset()
    prescribe_velocity_commands(self.env, schedule[0])
    obs = self.wrapped.get_observations()

    write_command = velocity_command_writer(self.env)
    trace = VelocityTrace(dt=self.control_dt)
    with torch.inference_mode():
      for step, command in enumerate(schedule):
        write_command(command)
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        trace.record(command, self.state())
        if on_step is not None:
          on_step(step)
    return trace

  def run_push(
    self,
    plan: PushPlan,
    on_step: Callable[[int], None] | None = None,
  ) -> PushMetrics:
    """Walk under a fixed command, take one shove per environment, recover.

    The force is written straight onto the robot rather than through an event
    term: the task's own ``push_robot`` event is a training device that
    teleports a random velocity onto the base at random times, which is neither
    the disturbance being measured nor under this harness's control. The play
    config drops it, and :class:`~mjlab.evaluation.push.PushDriver` is the only
    thing writing ``xfrc_applied`` in an evaluation run.
    """
    _check_plan(plan, self.num_envs, self.control_dt)
    # The reset is inside the inference block, not before it. A battery runs
    # this method once per magnitude, and the environment's delay buffers are
    # written in place on reset: buffers first touched under inference mode and
    # then reset outside it raise. Everything a pass does happens in one mode.
    with torch.inference_mode():
      self.wrapped.reset()
      prescribe_velocity_commands(self.env, plan.command)
      obs = self.wrapped.get_observations()

      driver = PushDriver(plan, self.robot, self.torso_body_id)
      metrics = PushMetrics(plan)
      metrics.start(self.state())
      try:
        for step in range(plan.num_steps):
          driver.apply(step)
          obs, _, _, _ = self.wrapped.step(self.policy(obs))
          metrics.record(self.state())
          if on_step is not None:
            on_step(step)
      finally:
        driver.clear()
    return metrics

  def run_competence_grid(
    self,
    grid: CompetenceGrid,
    episodes_per_cell: int = 64,
    shove_cfg: ShoveCfg = DEFAULT_SHOVE,
    seed: int = 0,
    max_steps: int | None = None,
    on_step: Callable[[int, int], None] | None = None,
  ) -> EpisodeTable:
    """Run every cell of a command x disturbance grid until it has episodes.

    Requires an ``episodic=True`` harness: the run is a stream of episodes that
    end on their own, not one long recording. Each environment holds its cell
    across resets, so it emits a sequence of independent samples from that
    cell's distribution, and the run stops once the least-sampled cell has
    ``episodes_per_cell`` of them.

    Args:
      grid: Cell assignment, sized to this harness's batch.
      episodes_per_cell: Episodes the *worst-covered* cell must reach. Cells
        that fall often reach it sooner, since a fall ends an episode early.
      shove_cfg: Onset timing of the shove train.
      seed: Seeds the shove headings, which are the only stochastic input the
        protocol has.
      max_steps: Hard cap, in case a cell somehow never closes an episode.
        Defaults to what the timeout alone guarantees is enough.
      on_step: Called with the step index and the least-sampled cell's episode
        count, for progress reporting.

    Returns:
      One row per completed episode. Episodes still in flight when the run ends
      are dropped rather than truncated.
    """
    if grid.num_envs != self.num_envs:
      raise ValueError(
        f"grid is for {grid.num_envs} environments, harness has {self.num_envs}"
      )
    if not self.env.cfg.terminations:
      raise RuntimeError(
        "a competence grid needs an episodic harness: build RlEvalHarness with "
        "episodic=True, or fell and ep_len_frac have nothing to measure"
      )
    max_episode_steps = int(self.env.max_episode_length)
    if max_steps is None:
      # Every episode ends by timeout at worst, so this many steps is enough
      # even for a cell that never falls.
      per_cell = int(grid.envs_per_cell().min())
      rounds = -(-episodes_per_cell // per_cell)  # ceil
      max_steps = (rounds + 1) * max_episode_steps

    with torch.inference_mode():
      self.wrapped.reset()
      prescribe_velocity_commands(self.env, grid.command.to(self.device))
      obs = self.wrapped.get_observations()

      collector = EpisodeCompetence(grid, max_episode_steps, self.device)
      generator = torch.Generator(device=self.device)
      generator.manual_seed(seed)
      driver = ShoveDriver(
        magnitude=grid.shove.to(self.device),
        robot=self.robot,
        dt=self.control_dt,
        max_episode_steps=max_episode_steps,
        cfg=shove_cfg,
        generator=generator,
      )
      terminations = self.env.termination_manager

      for step in range(max_steps):
        # Before the step, so the impulse is on the books when the physics
        # runs -- the same ordering the training event term gets.
        collector.note_shoves(driver.apply(collector.episode_step))
        obs, _, _, _ = self.wrapped.step(self.policy(obs))
        collector.record(
          self.state(), terminations.dones, terminations.get_term("fell_over")
        )
        if on_step is not None:
          on_step(step, collector.min_completed)
        if collector.min_completed >= episodes_per_cell:
          break
    return collector.table()

  def close(self) -> None:
    self.env.close()


def _check_schedule(schedule: torch.Tensor, num_envs: int) -> None:
  if schedule.ndim != 3 or schedule.shape[1:] != (num_envs, 3):
    raise ValueError(
      f"schedule must have shape (T, {num_envs}, 3), got {tuple(schedule.shape)}"
    )


def _check_plan(plan: PushPlan, num_envs: int, control_dt: float) -> None:
  """A plan is laid out against a control rate, so both have to match.

  The onsets are step indices, not seconds: a plan built for the walk engine's
  100 Hz and run at the policy's 50 Hz would push at half the intended times
  and measure a window half as long, silently.
  """
  if plan.num_envs != num_envs:
    raise ValueError(
      f"plan is for {plan.num_envs} environments, harness has {num_envs}"
    )
  if abs(plan.dt - control_dt) > 1e-9:
    raise ValueError(
      f"plan was built for a {1 / plan.dt:.0f} Hz controller, this one runs at "
      f"{1 / control_dt:.0f} Hz"
    )


def phase_name(phase: torch.Tensor) -> str:
  return Phase(int(phase)).name
