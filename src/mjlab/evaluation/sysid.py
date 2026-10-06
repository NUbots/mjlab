"""Simulator side of the walk system identification.

Runs a trained velocity policy in the task it was trained in, with the task's
random command generator replaced by a script (see
:mod:`mjlab.evaluation.sysid_sequences`), and records what the robot actually
did. Two plants:

``nominal``
  every randomisation off -- the startup domain randomisation, the pushes, the
  observation noise -- and every randomised quantity pinned to the middle of its
  training range. That includes the two latencies the task draws afresh every
  step, observation delay and actuator command delay, which the play config
  leaves randomised: they are pinned to the middle of their ranges rounded down,
  so a nominal run is deterministic and every environment in it is identical.
``randomised``
  the training distribution: the same startup randomisation, the pushes with the
  ranges the policy trained under, observation noise and the randomised delays
  on, and the reset pose drawn from the training range.

The environment's own step loop is used unchanged, so the timestep, decimation
and policy rate are the task's. What this module adds around it:

- :class:`ScriptedCommand` writes the command row for step ``k`` into the
  command term at the point in :meth:`ManagerBasedRlEnv.step` where the term
  would update its command, which is immediately before the observation is
  built. So the observation the policy acts on at step ``k`` carries exactly
  command row ``k``, with no lag and no processing in between.
- the root body's free-joint state is read straight out of ``qpos``/``qvel``
  after every physics step (or after the last one of each control step), which
  is fresh at every substep, unlike the derived body quantities.
- terminations are taken out of the environment and the task's ``fell_over``
  condition is evaluated here instead, so a fall is recorded where it happened
  rather than overwritten by an automatic reset.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.evaluation.harness import TASK_ID, load_policy
from mjlab.evaluation.sysid_sequences import AXES, Ranges
from mjlab.managers.event_manager import _DERIVED_FIELDS, RecomputeLevel
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

Variant = Literal["nominal", "randomised"]

STARTUP_RANDOMISATION_EVENTS: tuple[str, ...] = (
  "foot_friction",
  "encoder_bias",
  "base_com",
  "pd_gains",
)
"""The task's domain randomisation, all applied once at startup."""

PUSH_EVENT = "push_robot"
FALL_TERM = "fell_over"

ROOT_STATE_COLUMNS = 13
"""Free-joint ``qpos`` (pos 3, quat wxyz 4) then ``qvel`` (lin world 3, ang body 3)."""


@dataclass(frozen=True)
class TrainingReference:
  """What the policy was trained under, from its training run's config.

  Read from the W&B record of the run rather than from the task config in this
  checkout: a run trains against whatever the config said when it was launched,
  and that can drift from the code afterwards (qufuh82s trained on
  ``vx +-0.9, vy +-0.3, wz +-0.5``; this branch's task config says otherwise).
  """

  command_ranges: Ranges
  push_params: dict[str, Any] | None
  push_interval_s: tuple[float, float] | None
  source: str
  env_cfg: dict[str, Any] = field(repr=False)

  @staticmethod
  def from_env_cfg(env_cfg: dict[str, Any], source: str) -> TrainingReference:
    """Training ranges are the last curriculum stage's, if there is one.

    The curriculum overwrites the command ranges as training goes, so the range
    the finished policy trained on is the final stage, not the config's initial
    ``ranges``.
    """
    ranges = env_cfg["commands"]["twist"]["ranges"]
    stages = (
      env_cfg.get("curriculum", {})
      .get("command_vel", {})
      .get("params", {})
      .get("velocity_stages")
    )
    if stages:
      ranges = {**ranges, **max(stages, key=lambda stage: stage["step"])}
    command_ranges: Ranges = {
      axis: (float(ranges[key][0]), float(ranges[key][1]))
      for axis, key in zip(AXES, ("lin_vel_x", "lin_vel_y", "ang_vel_z"), strict=True)
    }
    push = env_cfg.get("events", {}).get(PUSH_EVENT)
    interval = push["interval_range_s"] if push else None
    return TrainingReference(
      command_ranges=command_ranges,
      push_params=push["params"] if push else None,
      push_interval_s=(float(interval[0]), float(interval[1])) if interval else None,
      source=source,
      env_cfg=env_cfg,
    )

  @staticmethod
  def from_wandb(run_path: str) -> TrainingReference:
    import wandb

    run = wandb.Api().run(run_path)
    return TrainingReference.from_env_cfg(
      dict(run.config["env_cfg"]), source=f"wandb:{run_path}"
    )

  @staticmethod
  def from_file(path: Path) -> TrainingReference:
    data = json.loads(path.read_text())
    return TrainingReference.from_env_cfg(
      data.get("env_cfg", data), source=f"file:{path}"
    )


def _midpoint_lag(min_lag: int, max_lag: int) -> int:
  return (min_lag + max_lag) // 2


@dataclass
class PlantInfo:
  """What the plant was, for the metadata."""

  variant: Variant
  seed: int
  num_envs: int
  physics_dt: float
  decimation: int
  control_dt: float
  observation_delays: dict[str, dict[str, int]]
  actuator_delays: dict[str, dict[str, int]]
  events: dict[str, Any]
  observation_noise: bool
  fall_condition: dict[str, Any]
  velocity_body: str


def _jsonable(value: Any) -> Any:
  return json.loads(json.dumps(value, default=str))


def build_sysid_env_cfg(
  variant: Variant,
  num_envs: int,
  seed: int,
  reference: TrainingReference,
  task_id: str = TASK_ID,
):
  """The task's play config, turned into one of the two sysid plants.

  Returns:
    The config, the task's fall termination term, and the delay settings applied.
  """
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  cfg = load_env_cfg(task_id, play=True)
  train_cfg = load_env_cfg(task_id, play=False)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed

  fall_term = cfg.terminations[FALL_TERM]
  cfg.terminations = {}
  # The curriculum only rewrites the command ranges, which the script replaces.
  cfg.curriculum = {}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.ranges.lin_vel_x = reference.command_ranges["vx"]
  twist.ranges.lin_vel_y = reference.command_ranges["vy"]
  twist.ranges.ang_vel_z = reference.command_ranges["wz"]
  twist.debug_vis = False

  # The robot config is shared module state; copy before touching actuators.
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  cfg.scene.entities = {"robot": robot_cfg}
  articulation = robot_cfg.articulation
  assert articulation is not None
  actuator_delays: dict[str, dict[str, int]] = {}
  observation_delays: dict[str, dict[str, int]] = {}
  actor = cfg.observations["actor"]

  if variant == "nominal":
    for name in (*STARTUP_RANDOMISATION_EVENTS, PUSH_EVENT):
      cfg.events.pop(name, None)
    leftover = [
      name for name, term in cfg.events.items() if term.mode in ("startup", "interval")
    ]
    if leftover:
      raise RuntimeError(
        f"unrecognised randomisation events {leftover}; decide whether the "
        "nominal plant should keep them"
      )
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {}
    reset_base.params["velocity_range"] = {}
    actor.enable_corruption = False

    actuators = []
    for actuator in articulation.actuators:
      lag = _midpoint_lag(actuator.delay_min_lag, actuator.delay_max_lag)
      actuator_delays[str(actuator.target_names_expr)] = {
        "min_lag": lag,
        "max_lag": lag,
        "training_min_lag": actuator.delay_min_lag,
        "training_max_lag": actuator.delay_max_lag,
      }
      actuators.append(
        dataclasses.replace(actuator, delay_min_lag=lag, delay_max_lag=lag)
      )
    robot_cfg.articulation = dataclasses.replace(
      articulation, actuators=tuple(actuators)
    )
    for group in cfg.observations.values():
      for name, term in group.terms.items():
        if term.delay_max_lag <= 0:
          continue
        lag = _midpoint_lag(term.delay_min_lag, term.delay_max_lag)
        if group is actor:
          observation_delays[name] = {
            "min_lag": lag,
            "max_lag": lag,
            "training_min_lag": term.delay_min_lag,
            "training_max_lag": term.delay_max_lag,
          }
        term.delay_min_lag = lag
        term.delay_max_lag = lag
  else:
    actor.enable_corruption = True
    push = copy.deepcopy(train_cfg.events[PUSH_EVENT])
    if reference.push_params is not None:
      push.params["velocity_range"] = {
        key: tuple(value)
        for key, value in reference.push_params["velocity_range"].items()
      }
    if reference.push_interval_s is not None:
      push.interval_range_s = reference.push_interval_s
    cfg.events[PUSH_EVENT] = push
    for actuator in articulation.actuators:
      actuator_delays[str(actuator.target_names_expr)] = {
        "min_lag": actuator.delay_min_lag,
        "max_lag": actuator.delay_max_lag,
      }
    for name, term in actor.terms.items():
      if term.delay_max_lag > 0:
        observation_delays[name] = {
          "min_lag": term.delay_min_lag,
          "max_lag": term.delay_max_lag,
        }

  return cfg, fall_term, observation_delays, actuator_delays


class ScriptedCommand:
  """Replaces the velocity term's random commands with a fixed schedule.

  The term's two hooks are swapped for ones that write row :attr:`index` of the
  schedule: ``_resample_command``, which runs at reset, and ``_update_command``,
  which :meth:`ManagerBasedRlEnv.step` runs after the physics and immediately
  before it builds the observation. Advancing :attr:`index` before each step
  therefore puts command row ``k`` into the observation the policy acts on at
  step ``k``.
  """

  def __init__(self, env: ManagerBasedRlEnv, schedule: torch.Tensor) -> None:
    """
    Args:
      schedule: Shape ``(T, num_envs, 3)`` commands ``(vx, vy, wz)``.
    """
    if schedule.ndim != 3 or schedule.shape[1:] != (env.num_envs, 3):
      raise ValueError(
        f"schedule must be (T, {env.num_envs}, 3), got {tuple(schedule.shape)}"
      )
    self.schedule = schedule.to(env.device, torch.float32)
    self.index = 0
    term = env.command_manager.get_term("twist")
    if not isinstance(term, UniformVelocityCommand):
      raise TypeError(f"expected a velocity command term, got {type(term)}")
    self._term = term

    def write(env_ids: torch.Tensor | None = None) -> None:
      row = self.schedule[min(self.index, self.schedule.shape[0] - 1)]
      ids = slice(None) if env_ids is None else env_ids
      term.vel_command_b[ids] = row[ids]
      term.vel_command_w[ids] = row[ids]
      term.is_standing_env[ids] = False
      term.is_heading_env[ids] = False
      term.is_world_env[ids] = False
      term.is_forward_env[ids] = False

    hooks = cast(Any, term)
    hooks._resample_command = write
    hooks._update_command = write
    write()

  def observed(self) -> torch.Tensor:
    """The command currently in the term, i.e. what the policy will see."""
    return self._term.vel_command_b


class PushLog:
  """Records each push the task's push event applies.

  Wraps the event function rather than reimplementing it, and records the
  change it made to the root's free-joint velocity: linear part in the world
  frame, angular part in the body frame, as ``qvel`` stores them.
  """

  def __init__(self) -> None:
    self.events: list[tuple[int, int, list[float]]] = []
    self.step = 0
    self._qvel_adr: torch.Tensor | None = None

  def wrap(self, func: Callable[..., None]) -> Callable[..., None]:
    def recorded(env, env_ids, **params) -> None:
      adr = env.scene["robot"].indexing.free_joint_v_adr
      before = env.sim.data.qvel[env_ids][:, adr].clone()
      func(env, env_ids, **params)
      after = env.sim.data.qvel[env_ids][:, adr]
      for env_id, delta in zip(
        env_ids.tolist(), (after - before).tolist(), strict=True
      ):
        self.events.append((env_id, self.step, delta))

    return recorded


@dataclass
class Trace:
  """Root free-joint state, step by step, for a batch of runs.

  Attributes:
    root: Shape ``(S, N, 13)`` state; row ``s`` is at ``s * dt``.
    commands: Shape ``(S, N, 3)`` the command in force at each row.
    dt: Row period, in seconds.
  """

  root: torch.Tensor
  commands: torch.Tensor
  dt: float


@dataclass
class BatchResult:
  policy: Trace
  physics: Trace | None
  fall_step: np.ndarray
  """Per environment, the policy step at which the task's fall condition first
  held, or -1."""
  pushes: list[tuple[int, int, list[float]]]
  push_flag: np.ndarray
  """Shape ``(T, N)``: 1 where a push was applied at the start of that row."""


class SysidRig:
  """One plant, one batch of environments, one policy.

  The environment is built once; :meth:`run` resets it and plays a schedule.
  Startup randomisation is drawn at build time, so successive runs on one rig
  share their robots' randomised parameters.
  """

  def __init__(
    self,
    checkpoint: Path,
    variant: Variant,
    num_envs: int,
    seed: int,
    reference: TrainingReference,
    device: str = "cuda:0",
    task_id: str = TASK_ID,
  ) -> None:
    cfg, fall_term, obs_delays, act_delays = build_sysid_env_cfg(
      variant, num_envs, seed, reference, task_id
    )
    self.push_log = PushLog()
    if PUSH_EVENT in cfg.events:
      push = cfg.events[PUSH_EVENT]
      push.func = self.push_log.wrap(push.func)
    self.variant: Variant = variant
    self.seed = seed
    self.num_envs = num_envs
    self.device = device
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.wrapped, self.policy = load_policy(self.env, checkpoint, device, task_id)
    self.robot = self.env.scene["robot"]
    self._fall_term = fall_term
    self._q_adr = self.robot.indexing.free_joint_q_adr
    self._v_adr = self.robot.indexing.free_joint_v_adr
    self.physics_dt = float(self.env.physics_dt)
    self.decimation = int(cfg.decimation)
    self.control_dt = float(self.env.step_dt)
    root_body = self.robot.indexing.root_body_id
    self.velocity_body = mujoco.mj_id2name(
      self.env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body
    )
    events = {
      name: {
        "mode": term.mode,
        "params": _jsonable({k: v for k, v in term.params.items() if k != "asset_cfg"}),
        **(
          {"interval_range_s": list(term.interval_range_s)}
          if term.interval_range_s is not None
          else {}
        ),
      }
      for name, term in cfg.events.items()
    }
    self.info = PlantInfo(
      variant=variant,
      seed=seed,
      num_envs=num_envs,
      physics_dt=self.physics_dt,
      decimation=self.decimation,
      control_dt=self.control_dt,
      observation_delays=obs_delays,
      actuator_delays=act_delays,
      events=events,
      observation_noise=bool(cfg.observations["actor"].enable_corruption),
      fall_condition={
        "term": FALL_TERM,
        "func": f"{fall_term.func.__module__}.{fall_term.func.__name__}",
        "params": _jsonable(fall_term.params),
        "evaluated": "after every policy step, on the post-step state",
      },
      velocity_body=self.velocity_body,
    )

  def _root_state(self) -> torch.Tensor:
    data = self.env.sim.data
    return torch.cat([data.qpos[:, self._q_adr], data.qvel[:, self._v_adr]], dim=-1)

  def _fell(self) -> torch.Tensor:
    return self._fall_term.func(self.env, **self._fall_term.params)

  def randomised_parameters(self) -> list[dict[str, Any]]:
    """Per environment, every model parameter the startup events changed.

    Compared element by element against the compiled nominal model, so whatever
    the task randomises is caught without naming it here, plus encoder bias,
    which lives on the entity rather than the model. Fields MuJoCo derives from
    the randomised ones (``body_subtreemass``, ``*_invweight0``) are left out.
    """
    model = self.env.sim.mj_model
    per_env: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
    prefixes = {
      "geom_": mujoco.mjtObj.mjOBJ_GEOM,
      "body_": mujoco.mjtObj.mjOBJ_BODY,
      "actuator_": mujoco.mjtObj.mjOBJ_ACTUATOR,
      "jnt_": mujoco.mjtObj.mjOBJ_JOINT,
      "dof_": mujoco.mjtObj.mjOBJ_DOF,
    }
    derived = set(_DERIVED_FIELDS[RecomputeLevel.set_const])
    for name in self.env.event_manager.domain_randomization_fields:
      if name in derived:
        continue
      values = getattr(self.env.sim.model, name)
      if values.ndim == 0 or values.shape[0] != self.num_envs:
        continue
      nominal = self.env.sim.get_default_field(name).to(values.device)
      changed = (values - nominal).abs().reshape(self.num_envs, -1) > 1e-9
      flat_ids = changed.any(dim=0).nonzero().flatten().tolist()
      if not flat_ids:
        continue
      shape = tuple(values.shape[1:])
      obj = next((o for p, o in prefixes.items() if name.startswith(p)), None)
      flat = values.reshape(self.num_envs, -1)[:, flat_ids].cpu().numpy()
      flat_nominal = nominal.reshape(-1)[flat_ids].cpu().numpy()
      labels = []
      for flat_id in flat_ids:
        index = np.unravel_index(flat_id, shape)
        owner = (
          mujoco.mj_id2name(model, obj, int(index[0])) if obj is not None else None
        )
        labels.append(
          f"{owner or index[0]}" + "".join(f"[{int(i)}]" for i in index[1:])
        )
      for env_id in range(self.num_envs):
        per_env[env_id][name] = {
          label: {"value": float(v), "nominal": float(n)}
          for label, v, n in zip(labels, flat[env_id], flat_nominal, strict=True)
        }
    bias = self.robot.data.encoder_bias
    if bias.abs().max() > 0:
      names = self.robot.joint_names
      for env_id in range(self.num_envs):
        per_env[env_id]["encoder_bias_rad"] = dict(
          zip(names, bias[env_id].tolist(), strict=True)
        )
    return per_env

  def run(self, schedule: torch.Tensor, physics_rate: bool) -> BatchResult:
    """Play a schedule on every environment, from reset.

    Args:
      schedule: Shape ``(T, N, 3)``; row ``k`` is in force over policy step
        ``k``.
      physics_rate: Also keep every physics substep, not only the state at each
        policy step.

    Every environment runs all ``T`` steps -- a fallen robot stays on the floor
    rather than being reset -- and :attr:`BatchResult.fall_step` says where to
    cut each one.
    """
    num_steps = schedule.shape[0]
    stride = 1 if physics_rate else self.decimation
    rows = num_steps * self.decimation // stride
    root = torch.empty(rows + 1, self.num_envs, ROOT_STATE_COLUMNS, device=self.device)
    fall_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    sim = self.env.sim
    original_step = sim.step
    substep = 0

    def recorded_step() -> None:
      nonlocal substep
      original_step()
      substep += 1
      if substep % stride == 0:
        root[substep // stride] = self._root_state()

    self.push_log.events.clear()
    # Reset and step under one inference block: the environment's delay buffers
    # are written in place on reset, and tensors first touched under inference
    # mode cannot be written outside it.
    with torch.inference_mode():
      source = ScriptedCommand(self.env, schedule)
      obs, _ = self.wrapped.reset()
      root[0] = self._root_state()
      sim.step = recorded_step  # type: ignore[method-assign]
      try:
        for k in range(num_steps):
          self.push_log.step = k + 1
          action = self.policy(obs)
          # Row k + 1 goes into the observation built at the end of this step.
          source.index = k + 1
          obs, _, _, _ = self.wrapped.step(action)
          fell = self._fell() & (fall_step < 0)
          fall_step[fell] = k + 1
      finally:
        sim.step = original_step  # type: ignore[method-assign]

    commands = schedule.to(self.device)
    physics = None
    if physics_rate:
      phys_commands = commands.repeat_interleave(self.decimation, dim=0)
      physics = Trace(root[:-1], phys_commands, self.physics_dt)
      policy_root = root[: -1 : self.decimation]
    else:
      policy_root = root[:-1]
    push_flag = np.zeros((num_steps, self.num_envs), dtype=np.int8)
    for env_id, step, _ in self.push_log.events:
      if step < num_steps:
        push_flag[step, env_id] = 1
    return BatchResult(
      policy=Trace(policy_root, commands, self.control_dt),
      physics=physics,
      fall_step=fall_step.cpu().numpy(),
      pushes=list(self.push_log.events),
      push_flag=push_flag,
    )

  def close(self) -> None:
    self.env.close()


def heading_frame(root: torch.Tensor) -> dict[str, torch.Tensor]:
  """Pose and velocity of the root body, in the heading frame.

  The heading frame is the world frame rotated by the body's yaw alone, which is
  what motion capture of the torso gives: no roll or pitch of the torso leaks
  into the velocities. Yaw is the heading of the body's x axis projected onto
  the ground, as the task's ``heading_w`` defines it; ``wz`` is the world-frame
  yaw rate, which is the same in the heading frame.

  Args:
    root: Shape ``(..., 13)`` free-joint state; see :data:`ROOT_STATE_COLUMNS`.
  """
  pos = root[..., 0:3]
  w, x, y, z = (root[..., 3 + i] for i in range(4))
  lin_w = root[..., 7:10]
  ang_b = root[..., 10:13]
  # Rotation matrix rows from the unit quaternion (w, x, y, z).
  r00 = 1 - 2 * (y * y + z * z)
  r10 = 2 * (x * y + w * z)
  r20 = 2 * (x * z - w * y)
  r21 = 2 * (y * z + w * x)
  r22 = 1 - 2 * (x * x + y * y)
  yaw = torch.atan2(r10, r00)
  cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
  vx = cos_yaw * lin_w[..., 0] + sin_yaw * lin_w[..., 1]
  vy = -sin_yaw * lin_w[..., 0] + cos_yaw * lin_w[..., 1]
  wz = r20 * ang_b[..., 0] + r21 * ang_b[..., 1] + r22 * ang_b[..., 2]
  return {
    "vx": vx,
    "vy": vy,
    "wz": wz,
    "x": pos[..., 0],
    "y": pos[..., 1],
    "z": pos[..., 2],
    "yaw": yaw,
  }


COLUMNS: tuple[str, ...] = (
  "t",
  "cmd_vx",
  "cmd_vy",
  "cmd_wz",
  "vx",
  "vy",
  "wz",
  "x",
  "y",
  "z",
  "yaw",
  "qw",
  "qx",
  "qy",
  "qz",
  "lin_vel_w_x",
  "lin_vel_w_y",
  "lin_vel_w_z",
  "ang_vel_b_x",
  "ang_vel_b_y",
  "ang_vel_b_z",
  "fall",
  "push",
)
"""Columns of every run file, in order. See :func:`run_table`.

Besides the heading-frame quantities, the root free joint's full state is kept
as it came out of the simulator, so that anything a motion capture pipeline
estimates (the pose of a point offset from the root, velocities by differencing)
can be recomputed offline and checked against the exact velocity.
"""

RAW_STATE_COLUMNS: tuple[tuple[str, int], ...] = (
  ("qw", 3),
  ("qx", 4),
  ("qy", 5),
  ("qz", 6),
  ("lin_vel_w_x", 7),
  ("lin_vel_w_y", 8),
  ("lin_vel_w_z", 9),
  ("ang_vel_b_x", 10),
  ("ang_vel_b_y", 11),
  ("ang_vel_b_z", 12),
)
"""Run-file columns copied straight from the root state, with their index in it."""


def run_table(
  trace: Trace,
  env_id: int,
  num_rows: int,
  fall_row: int | None,
  push_flag: np.ndarray | None = None,
  rows_per_push_step: int = 1,
) -> dict[str, np.ndarray]:
  """One environment's rows of a trace, cut at its fall, as named columns.

  Args:
    num_rows: Rows to keep, including the fall row if there is one.
    fall_row: Row at which ``fall`` turns to 1, or ``None``.
    push_flag: Shape ``(T, N)`` per policy step; expanded to this trace's rate.
  """
  frame = heading_frame(trace.root[:num_rows, env_id])
  out: dict[str, np.ndarray] = {
    "t": np.arange(num_rows) * trace.dt,
  }
  commands = trace.commands[:num_rows, env_id].cpu().numpy().astype(np.float64)
  for index, axis in enumerate(AXES):
    out[f"cmd_{axis}"] = commands[:, index]
  for key in ("vx", "vy", "wz", "x", "y", "z", "yaw"):
    out[key] = frame[key].cpu().numpy().astype(np.float64)
  raw = trace.root[:num_rows, env_id].cpu().numpy().astype(np.float64)
  for key, column in RAW_STATE_COLUMNS:
    out[key] = raw[:, column]
  fall = np.zeros(num_rows, dtype=np.int8)
  if fall_row is not None:
    fall[fall_row] = 1
  out["fall"] = fall
  push = np.zeros(num_rows, dtype=np.int8)
  if push_flag is not None:
    pushes = push_flag[:, env_id]
    push_rows = np.flatnonzero(pushes) * rows_per_push_step
    push[push_rows[push_rows < num_rows]] = 1
  out["push"] = push
  return out


def write_run(
  path: Path,
  table: dict[str, np.ndarray],
  scalars: dict[str, Any],
  fmt: Literal["mat", "csv"],
) -> list[str]:
  """Write one run. Returns the file names written, relative to ``path.parent``.

  ``mat`` writes one file with a ``physics`` struct holding the :data:`COLUMNS`
  as column vectors, and the scalars beside it. ``csv`` writes the table alone
  to ``<name>.csv``; the scalars are in the sequence's ``metadata.json`` either
  way.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  if fmt == "mat":
    import scipy.io

    file = path.with_suffix(".mat")
    scipy.io.savemat(
      file,
      {
        "physics": {key: value.reshape(-1, 1) for key, value in table.items()},
        **scalars,
      },
      do_compression=True,
      oned_as="column",
    )
    return [file.name]

  file = path.with_name(f"{path.name}.csv")
  matrix = np.column_stack([table[key].astype(np.float64) for key in COLUMNS])
  integer_columns = {"fall", "push"}
  formats = ["%d" if key in integer_columns else "%.9g" for key in COLUMNS]
  np.savetxt(
    file, matrix, delimiter=",", header=",".join(COLUMNS), comments="", fmt=formats
  )
  return [file.name]


def tracking_score(
  command: np.ndarray, achieved: np.ndarray, sigma: float = 0.25
) -> np.ndarray:
  """``exp(-||v - v_hat|| / sigma)`` per sample, over ``(vx, vy, wz)``.

  The norm mixes m/s and rad/s, as the requested score defines it.
  """
  return np.exp(-np.linalg.norm(command - achieved, axis=-1) / sigma)


def file_sha256(path: Path) -> str:
  import hashlib

  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def nan_to_none(value: float) -> float | None:
  return None if value is None or math.isnan(value) else value
