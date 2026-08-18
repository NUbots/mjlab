"""Drive the NUgus with the quintic walk engine in plain MuJoCo.

This is the empirical check on the port: no learned policy, no mjlab
environment, just the engine commanding leg joint targets against a compiled
NUgus and a floor. It lives in the library rather than in the play script so the
regression tests exercise exactly the rig the script does.

The engine is a hand-tuned controller that was developed on hardware, so the
model it runs against matters more than it would for a policy that was trained
against one. :data:`PLANTS` collects the models worth running it on; see
:data:`DEFAULT_PLANT` for which one is the reference and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.nugus.nugus_constants import get_nugus_robot_cfg
from mjlab.asset_zoo.robots.nugus.nugus_eval_constants import get_nugus_eval_robot_cfg
from mjlab.asset_zoo.robots.nugus.nugus_nubots_constants import (
  get_nugus_nubots_robot_cfg,
)
from mjlab.asset_zoo.robots.nugus.nugus_nubots_sim_constants import (
  get_nugus_nubots_sim_robot_cfg,
)
from mjlab.controllers.quintic_walk.controller import (
  JOINT_NAMES,
  QuinticWalkController,
  detect_planted_phase,
  sole_poses_in_torso,
)
from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_SOLE_OFFSET,
  mat_to_rpy_intrinsic,
  rpy_intrinsic_to_mat,
)
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
  WalkParameters,
)
from mjlab.entity import Entity, EntityCfg

Plant = Literal["eval", "training", "nubots-sim", "nubots-xml"]

PLANTS: dict[str, Callable[[], EntityCfg]] = {
  "eval": get_nugus_eval_robot_cfg,
  "training": get_nugus_robot_cfg,
  "nubots-sim": get_nugus_nubots_sim_robot_cfg,
  "nubots-xml": get_nugus_nubots_robot_cfg,
}
"""Models the walk engine can be driven against.

``eval``
  Sim-to-real randomisation at nominal and hardware leg joint limits, and
  otherwise the training model unchanged. The reference for comparing
  controllers.
``training``
  The model policies are trained against: backlash on every servo, soft
  contacts, narrow RL joint clamps. The engine falls on this one, which is a
  fact about the model rather than about the engine.
``nubots-sim``
  NUbots' simulation dynamics on mjlab's kinematic tree.
``nubots-xml``
  NUbots' MJCF verbatim, for validating the port against the simulator they run.
"""

DEFAULT_PLANT: Plant = "eval"
"""Reference model.

Both controllers under comparison should meet the same robot, and it should be
the one neither of them is tuned for: randomisation at nominal, joint limits as
the hardware has them, nothing else touched. See
:mod:`~mjlab.asset_zoo.robots.nugus.nugus_eval_constants`.
"""

POSTURE: dict[str, float] = {
  "left_shoulder_pitch": 1.7,
  "right_shoulder_pitch": 1.7,
  "left_shoulder_roll": 0.35,
  "right_shoulder_roll": -0.4,
  "left_elbow_pitch": -0.7,
  "right_elbow_pitch": -0.7,
  "neck_yaw": 0.0,
  "head_pitch": 0.0,
}
"""Arm and head hold positions while walking, from ``Walk.yaml``."""

FLOOR_FRICTION = (1.0, 0.1, 0.1)
"""Floor friction, matching ``scene.xml`` in the NUbots repository."""

UPRIGHT_THRESHOLD = 0.5
"""Torso ``z`` axis dotted with world up, below which the robot has fallen."""

GYRO_SENSOR_NAMES = ("imu_ang_vel", "gyro")
"""Body-frame angular velocity sensor, named differently in each MJCF."""


@dataclass(frozen=True)
class PlaybackResult:
  """What one playback run did."""

  elapsed: float
  """Simulated duration, in seconds."""
  displacement: tuple[float, float]
  """Torso travel in x and y, in metres."""
  torso_height: float
  """Final torso height, in metres."""
  upright: float
  """Final torso up-axis component; 1.0 is vertical."""
  min_upright: float
  """Smallest value that reached over the run."""
  fall_time: float | None
  """When the torso first tipped past :data:`UPRIGHT_THRESHOLD`, if it did."""
  engine_state: EngineState
  """Walk engine state at the end of the run."""

  @property
  def fell(self) -> bool:
    return self.fall_time is not None

  @property
  def mean_speed(self) -> float:
    """Average forward speed over the run, in m/s."""
    return self.displacement[0] / self.elapsed


def build_model(plant: Plant = DEFAULT_PLANT) -> mujoco.MjModel:
  """Compile one of :data:`PLANTS`, plus a floor to walk on and a light."""
  spec = Entity(PLANTS[plant]()).spec

  # Every MJCF here defaults its geoms to non-colliding, so the floor has to opt
  # in explicitly or the feet fall straight through it. contype/conaffinity 1
  # collides with mjlab's feet (contype 1, conaffinity 1) and with NUbots'
  # (contype 1, conaffinity 2) alike.
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    pos=[0.0, 0.0, 0.0],
    contype=1,
    conaffinity=1,
    condim=3,
    friction=list(FLOOR_FRICTION),
    rgba=[0.35, 0.37, 0.40, 1.0],
  )
  spec.worldbody.add_light(
    pos=[0.0, 0.0, 3.0],
    dir=[0.0, 0.0, -1.0],
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  )
  return spec.compile()


def _joint_qpos_indices(model: mujoco.MjModel, names) -> np.ndarray:
  return np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in names
    ]
  )


def _actuator_indices(model: mujoco.MjModel, names) -> np.ndarray:
  return np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
  )


def _gyro_address(model: mujoco.MjModel) -> int:
  for name in GYRO_SENSOR_NAMES:
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor != -1:
      return int(model.sensor_adr[sensor])
  raise ValueError(f"no angular velocity sensor named any of {GYRO_SENSOR_NAMES}")


class WalkPlayback:
  """One NUgus, one walk engine, one floor."""

  def __init__(
    self,
    plant: Plant = DEFAULT_PLANT,
    walk_params: WalkParameters = NUGUS_WALK_PARAMETERS,
    control_hz: float = 100.0,
    use_balance_control: bool = True,
    exact_ik: bool = False,
  ) -> None:
    """Build the rig and put the robot in the engine's standing stance.

    Args:
      plant: Which of :data:`PLANTS` to drive.
      walk_params: Walk engine tuning. Defaults to the deployed NUgus values.
      control_hz: Engine update rate. The robot runs at 100 Hz.
      use_balance_control: Apply the ``FootController`` torso correction.
      exact_ik: Solve the legs against the compiled geometry instead of the
        engine's idealised leg. Not the deployed behaviour; see
        :class:`~mjlab.controllers.quintic_walk.controller.QuinticWalkController`.
    """
    self.plant = plant
    self.model = build_model(plant)
    self.data = mujoco.MjData(self.model)

    self.substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
    self.control_dt = self.substeps * self.model.opt.timestep

    self.controller = QuinticWalkController(
      num_envs=1,
      device="cpu",
      walk_params=walk_params,
      use_balance_control=use_balance_control,
      exact_ik_model=self.model if exact_ik else None,
    )

    self._leg_qpos = _joint_qpos_indices(self.model, JOINT_NAMES)
    self._leg_ctrl = _actuator_indices(self.model, JOINT_NAMES)
    self._posture_names = tuple(POSTURE)
    self._posture_qpos = _joint_qpos_indices(self.model, self._posture_names)
    self._posture_ctrl = _actuator_indices(self.model, self._posture_names)
    self._posture_values = np.array([POSTURE[name] for name in self._posture_names])
    self._torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    self._gyro_adr = _gyro_address(self.model)

    self.reset()

  # Measurements.

  @property
  def torso_position(self) -> np.ndarray:
    return self.data.xpos[self._torso]

  @property
  def upright(self) -> float:
    """Torso up-axis dotted with world up. 1.0 is vertical, 0.0 is on its side."""
    return float(self.data.xmat[self._torso].reshape(3, 3)[2, 2])

  def sensed_phase(self) -> torch.Tensor:
    """Planted foot phase measured from the current state. Shape ``(1,)``."""
    return detect_planted_phase(*sole_poses_in_torso(self.model, self.data))

  def lowest_sole_height(self) -> float:
    """World height of the lower foot sole."""
    heights = []
    for side in ("left", "right"):
      body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
      rotation = self.data.xmat[body].reshape(3, 3)
      heights.append((self.data.xpos[body] + rotation @ np.array(NUGUS_SOLE_OFFSET))[2])
    return float(min(heights))

  # Stepping.

  def reset(self) -> None:
    """Stand the robot in the engine's stopped stance, soles flat on the floor.

    Starting from the MJCF keyframe would fight the engine on the first step, so
    the joint angles come from the engine's own stopped state: it is asked for a
    zero-velocity step, and whatever it returns is adopted verbatim.

    That fixes the joints but not the floating base, which the engine has no
    opinion about -- it plans in the planted foot's frame. The base follows from
    requiring the stance sole to lie flat on the floor:

    - Orientation is the rotation that levels the *achieved* stance sole, read
      back from the model after the joints are set. Yaw is dropped so the robot
      faces +x. On the deployed idealised IK this comes out around 14.4 degrees
      of torso pitch rather than the engine's nominal 12, and the 2.4 degree gap
      is the idealised leg's model error showing up as a posture error; with
      ``exact_ik`` the two agree. Levelling the achieved sole rather than the
      commanded one is what puts the robot flat-footed on the ground instead of
      balanced on its toe edges.
    - Height is then whatever drops the lower sole onto the floor.
    """
    self.data.qpos[:] = 0.0
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    self.data.qvel[:] = 0.0
    self.data.qpos[self._posture_qpos] = self._posture_values

    self.controller.reset()
    targets = self.controller.compute(
      dt=self.control_dt,
      velocity_command=torch.zeros(1, 3),
      torso_rotation_w=torch.eye(3, dtype=torch.float32).unsqueeze(0),
      gyro_b=torch.zeros(1, 3),
      # Standing still, both feet are down; the engine is in its STOPPED state
      # and does not switch feet, so the value only has to be well formed.
      sensed_phase=torch.full((1,), int(Phase.DOUBLE)),
    )
    self.data.qpos[self._leg_qpos] = targets[0].numpy()
    mujoco.mj_forward(self.model, self.data)

    stance_left = int(self.controller.generator.phase[0]) == int(Phase.LEFT)
    sole_t = sole_poses_in_torso(self.model, self.data)[0 if stance_left else 1]
    levelled = mat_to_rpy_intrinsic(sole_t[:, :3, :3].transpose(-1, -2))
    levelled[:, 2] = 0.0
    quat = np.empty(4)
    mujoco.mju_mat2Quat(
      quat, rpy_intrinsic_to_mat(levelled)[0].numpy().flatten().astype(np.float64)
    )
    self.data.qpos[3:7] = quat

    mujoco.mj_forward(self.model, self.data)
    self.data.qpos[2] -= self.lowest_sole_height()
    mujoco.mj_forward(self.model, self.data)

    self.controller.reset()

  def step(self, command: torch.Tensor) -> None:
    """Run one control step and its physics substeps.

    The sensed foot phase is measured and passed every step, whether or not the
    engine is configured to wait for it, so that
    ``only_switch_when_planted`` behaves as it would on the robot.

    Args:
      command: Shape ``(1, 3)`` velocity command (dx, dy, dtheta).
    """
    rotation = torch.tensor(
      self.data.xmat[self._torso].reshape(3, 3), dtype=torch.float32
    ).unsqueeze(0)
    angular = torch.tensor(
      self.data.sensordata[self._gyro_adr : self._gyro_adr + 3], dtype=torch.float32
    ).unsqueeze(0)
    targets = self.controller.compute(
      dt=self.control_dt,
      velocity_command=command,
      torso_rotation_w=rotation,
      gyro_b=angular,
      sensed_phase=self.sensed_phase(),
    )
    self.data.ctrl[self._leg_ctrl] = targets[0].numpy()
    self.data.ctrl[self._posture_ctrl] = self._posture_values
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

  def run(
    self,
    command: tuple[float, float, float],
    duration: float,
    on_step: Callable[[int], bool | None] | None = None,
  ) -> PlaybackResult:
    """Hold a command for ``duration`` seconds.

    Args:
      command: Velocity command (dx, dy, dtheta), held for the whole run.
      duration: Simulated seconds.
      on_step: Called with the step index after each control step. Return
        ``False`` to stop early.

    Returns:
      What happened; see :class:`PlaybackResult`.
    """
    target = torch.tensor([command], dtype=torch.float32)
    num_steps = int(duration / self.control_dt)
    start_xy = self.torso_position[:2].copy()

    min_upright = self.upright
    fall_time: float | None = None
    step = 0
    for step in range(num_steps):
      self.step(target)
      if self.upright < min_upright:
        min_upright = self.upright
      if fall_time is None and self.upright < UPRIGHT_THRESHOLD:
        fall_time = (step + 1) * self.control_dt
      if on_step is not None and on_step(step) is False:
        break

    travelled = self.torso_position[:2] - start_xy
    return PlaybackResult(
      elapsed=(step + 1) * self.control_dt,
      displacement=(float(travelled[0]), float(travelled[1])),
      torso_height=float(self.torso_position[2]),
      upright=self.upright,
      min_upright=min_upright,
      fall_time=fall_time,
      engine_state=EngineState(int(self.controller.generator.state[0])),
    )
