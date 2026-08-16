"""Drive the NUgus with the ported quintic walk engine, in MuJoCo.

This is the empirical check on the port: no learned policy, no mjlab
environment, just the walk engine commanding leg joint targets against the same
robot model and actuators the RL task uses.

Examples::

  # Interactive viewer, walking forward at 0.2 m/s.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2

  # Headless, write a video.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --video /tmp/walk.mp4

  # Open loop, without the FootController balance correction.
  uv run python scripts/tools/play_quintic_walk.py --vx 0.2 --no-balance
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.nugus.nugus_constants import get_nugus_robot_cfg
from mjlab.controllers.quintic_walk.controller import (
  JOINT_NAMES,
  QuinticWalkController,
)
from mjlab.controllers.quintic_walk.kinematics import NUGUS_SOLE_OFFSET
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  EngineState,
  WalkParameters,
)
from mjlab.entity import Entity

# Arm and head hold positions while walking, from Walk.yaml.
POSTURE = {
  "left_shoulder_pitch": 1.7,
  "right_shoulder_pitch": 1.7,
  "left_shoulder_roll": 0.35,
  "right_shoulder_roll": -0.4,
  "left_elbow_pitch": -0.7,
  "right_elbow_pitch": -0.7,
  "neck_yaw": 0.0,
  "head_pitch": 0.0,
}


@dataclass
class Args:
  vx: float = 0.2
  """Forward velocity command, in m/s."""
  vy: float = 0.0
  """Lateral velocity command, in m/s."""
  wz: float = 0.0
  """Yaw rate command, in rad/s."""
  duration: float = 10.0
  """Simulated duration, in seconds."""
  control_hz: float = 100.0
  """Walk engine update rate. The robot runs at 100 Hz."""
  balance: bool = True
  """Apply the FootController torso-orientation correction."""
  video: Path | None = None
  """Write an mp4 here instead of opening the viewer."""
  video_fps: int = 50
  """Frame rate of the written video."""

  # Retuning knobs. The as-deployed values topple this robot model after about
  # five steps: the engine plans against an idealised leg that is ~9 cm off the
  # MJCF's real geometry, so the feet land higher and further than intended and
  # the ankles are commanded past their limits. Stability is knife-edge in
  # torso_height, which is a symptom of the same mismatch.
  torso_height: float | None = None
  """Override the walk torso height, in metres. Deployed value is 0.44."""
  step_width: float | None = None
  """Override the lateral foot separation, in metres. Deployed value is 0.27."""
  step_period: float | None = None
  """Override the step duration, in seconds. Deployed value is 0.32."""
  step_height: float | None = None
  """Override the swing foot apex height, in metres. Deployed value is 0.085."""
  exact_ik: bool = False
  """Solve the legs against the real MJCF geometry instead of the engine's
  idealised leg. Not the deployed behaviour -- this is the second experimental
  condition, isolating the algorithm from its kinematic model error."""


def walk_parameters(args: Args) -> WalkParameters:
  """Apply any command-line overrides to the deployed NUgus tuning."""
  overrides = {
    name: value
    for name, value in (
      ("torso_height", args.torso_height),
      ("step_width", args.step_width),
      ("step_period", args.step_period),
      ("step_height", args.step_height),
    )
    if value is not None
  }
  return replace(NUGUS_WALK_PARAMETERS, **overrides)


def build_model() -> mujoco.MjModel:
  """Compile the NUgus with its mjlab actuators, plus a floor to walk on."""
  spec = Entity(get_nugus_robot_cfg()).spec

  # The MJCF's default geom is non-colliding, so the floor has to opt in
  # explicitly or the feet fall straight through it.
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    pos=[0.0, 0.0, 0.0],
    contype=1,
    conaffinity=1,
    condim=3,
    friction=[1.0, 0.005, 0.0001],
    rgba=[0.35, 0.37, 0.40, 1.0],
  )
  spec.worldbody.add_light(
    pos=[0.0, 0.0, 3.0],
    dir=[0.0, 0.0, -1.0],
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  )
  return spec.compile()


def joint_qpos_indices(model: mujoco.MjModel, names) -> np.ndarray:
  return np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in names
    ]
  )


def actuator_indices(model: mujoco.MjModel, names) -> np.ndarray:
  return np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
  )


def lowest_sole_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """World height of the lower foot sole."""
  heights = []
  for side in ("left", "right"):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    rotation = data.xmat[body].reshape(3, 3)
    heights.append((data.xpos[body] + rotation @ np.array(NUGUS_SOLE_OFFSET))[2])
  return float(min(heights))


def settle_onto_ground(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  controller: QuinticWalkController,
  leg_qpos: np.ndarray,
  posture_qpos: np.ndarray,
  posture_values: np.ndarray,
) -> None:
  """Place the robot in the engine's own standing stance, feet on the floor.

  Starting from the MJCF keyframe would fight the engine on the first step, so
  the initial joint angles come from the engine's stopped-state foot poses and
  the torso is then dropped until the lower sole touches the ground.
  """
  data.qpos[:] = 0.0
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  data.qpos[posture_qpos] = posture_values

  upright = torch.eye(3, dtype=torch.float32).unsqueeze(0)
  targets = controller.compute(
    dt=1.0 / 100.0,
    velocity_command=torch.zeros(1, 3),
    torso_rotation_w=upright,
    gyro_b=torch.zeros(1, 3),
  )
  data.qpos[leg_qpos] = targets[0].numpy()

  # The engine holds the torso pitched forward; match it so the soles land flat.
  half = 0.5 * controller.generator.cfg.torso_pitch
  data.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]

  mujoco.mj_forward(model, data)
  data.qpos[2] -= lowest_sole_height(model, data)
  mujoco.mj_forward(model, data)

  controller.reset()


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)

  model = build_model()
  data = mujoco.MjData(model)

  control_dt = 1.0 / args.control_hz
  substeps = max(1, round(control_dt / model.opt.timestep))
  effective_dt = substeps * model.opt.timestep

  params = walk_parameters(args)
  controller = QuinticWalkController(
    num_envs=1,
    device="cpu",
    walk_params=params,
    use_balance_control=args.balance,
    exact_ik_model=model if args.exact_ik else None,
  )

  leg_qpos = joint_qpos_indices(model, JOINT_NAMES)
  leg_ctrl = actuator_indices(model, JOINT_NAMES)
  posture_names = tuple(POSTURE)
  posture_qpos = joint_qpos_indices(model, posture_names)
  posture_ctrl = actuator_indices(model, posture_names)
  posture_values = np.array([POSTURE[name] for name in posture_names])

  settle_onto_ground(model, data, controller, leg_qpos, posture_qpos, posture_values)

  torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
  gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
  gyro_adr = model.sensor_adr[gyro]

  command = torch.tensor([[args.vx, args.vy, args.wz]], dtype=torch.float32)
  num_steps = int(args.duration / effective_dt)
  start_xy = data.xpos[torso][:2].copy()

  def control_step() -> None:
    rotation = torch.tensor(
      data.xmat[torso].reshape(3, 3), dtype=torch.float32
    ).unsqueeze(0)
    angular = torch.tensor(
      data.sensordata[gyro_adr : gyro_adr + 3], dtype=torch.float32
    ).unsqueeze(0)
    targets = controller.compute(
      dt=effective_dt,
      velocity_command=command,
      torso_rotation_w=rotation,
      gyro_b=angular,
    )
    data.ctrl[leg_ctrl] = targets[0].numpy()
    data.ctrl[posture_ctrl] = posture_values
    for _ in range(substeps):
      mujoco.mj_step(model, data)

  if args.video is not None:
    import mediapy

    frames = []
    every = max(1, round((1.0 / args.video_fps) / effective_dt))
    renderer = mujoco.Renderer(model, height=480, width=640)
    for step in range(num_steps):
      control_step()
      if step % every == 0:
        renderer.update_scene(data, camera="track")
        frames.append(renderer.render())
    renderer.close()
    args.video.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(args.video), frames, fps=args.video_fps)
    print(f"wrote {len(frames)} frames to {args.video}")
  else:
    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
      for _ in range(num_steps):
        if not viewer.is_running():
          break
        control_step()
        viewer.sync()

  travelled = data.xpos[torso][:2] - start_xy
  elapsed = num_steps * effective_dt
  upright = float(data.xmat[torso].reshape(3, 3)[2, 2])
  print(
    f"leg IK            : {'exact (MJCF geometry)' if controller.uses_exact_ik else 'idealised (as deployed)'}\n"
    f"engine state      : {EngineState(int(controller.generator.state[0])).name}\n"
    f"simulated         : {elapsed:.2f} s at {1.0 / effective_dt:.0f} Hz control\n"
    f"torso height      : {data.xpos[torso][2]:.3f} m "
    f"(walk torso_height {params.torso_height:.3f})\n"
    f"upright           : {upright:+.3f} "
    f"({'standing' if upright > 0.5 else 'FELL OVER'})\n"
    f"displacement (x,y): {travelled[0]:+.3f}, {travelled[1]:+.3f} m\n"
    f"mean forward speed: {travelled[0] / elapsed:+.3f} m/s "
    f"(commanded {args.vx:+.3f})"
  )


if __name__ == "__main__":
  main()
