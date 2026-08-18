"""Load a named NUgus pose in MuJoCo and inspect or view it.

Built for cross-checking poses against the NUbots stack. The NUgus stands in
several different postures depending on which controller is driving it, and they
are easy to mistake for each other:

- ``walk-stance``   the quintic walk engine's stopped stance, i.e. what
                    ``StandStill`` produces (it emits ``Walk`` with a zero
                    command, so the engine holds this pose rather than playing a
                    script). Solved with the idealised IK the robot deploys.
- ``walk-stance-exact``  the same engine request solved against the real MJCF
                    geometry instead. Not what the robot runs; shows what the
                    engine was asking for.
- ``rl-default``    ``default_pose`` from NUbots ``RLWalk.yaml``, which the
                    comment marks as "Same as in Mujoco.yaml". This is what the
                    ``mujoco/rl_keyboardwalk`` role stands in.
- ``stand-script``  ``Stand.yaml``, the scripted pose played by script sequences
                    such as ``ScriptKick`` and get-ups.
- ``keyframe``      mjlab's ``STAND_BENT_KNEES_KEYFRAME``.

Examples::

  # Print every pose side by side with the height each produces.
  uv run python scripts/tools/view_nugus_pose.py

  # Open the viewer on one of them.
  uv run python scripts/tools/view_nugus_pose.py --pose walk-stance --viewer True

  # Inspect an arbitrary configuration: 12 leg values (JOINT_NAMES order) or
  # 20 values in NUbots ServoID order.
  uv run python scripts/tools/view_nugus_pose.py --joints "0.03,0.16,-0.9,1.2,-0.5,-0.17,..."
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  STAND_BENT_KNEES_KEYFRAME,
  get_nugus_robot_cfg,
)
from mjlab.controllers.quintic_walk.controller import JOINT_NAMES, QuinticWalkController
from mjlab.controllers.quintic_walk.kinematics import (
  NUGUS_SOLE_OFFSET,
  NUGUS_SOLE_ROTATION,
)
from mjlab.entity import Entity

PoseName = Literal[
  "walk-stance", "walk-stance-exact", "rl-default", "stand-script", "keyframe"
]

# NUbots ServoID order, as used by RLWalk.yaml's default_pose and the joint_map
# in RLWalk.cpp. Note the conversion between this and mjlab's order is a pure
# permutation -- no sign flips, no offsets.
SERVO_ID_ORDER: tuple[str, ...] = (
  "right_shoulder_pitch",
  "left_shoulder_pitch",
  "right_shoulder_roll",
  "left_shoulder_roll",
  "right_elbow_pitch",
  "left_elbow_pitch",
  "right_hip_yaw",
  "left_hip_yaw",
  "right_hip_roll",
  "left_hip_roll",
  "right_hip_pitch",
  "left_hip_pitch",
  "right_knee_pitch",
  "left_knee_pitch",
  "right_ankle_pitch",
  "left_ankle_pitch",
  "right_ankle_roll",
  "left_ankle_roll",
  "neck_yaw",
  "head_pitch",
)

# module/skill/RLWalk/data/config/RLWalk.yaml, in SERVO_ID_ORDER.
RL_DEFAULT_POSE = (
  1.714, 1.716, -0.198, 0.198, -0.720, -0.715, -0.033, 0.034, -0.162, 0.163,
  -0.985, -0.985, 1.206, 1.206, -0.510, -0.512, 0.168, -0.167, 0.0, 0.0,
)  # fmt: skip

# shared/utility/skill/scripts/nugus/Stand.yaml, in SERVO_ID_ORDER.
STAND_SCRIPT_POSE = (
  1.68389368, 1.69310892, -0.362899661, 0.324631244, -0.71111691, -0.703437507,
  -0.0353254639, 0.0430049114, -0.165876091, 0.158196643, -0.92806077,
  -0.927720666, 1.22773445, 1.23234212, -0.5, -0.5, 0.128503367, -0.137148142,
  -0.0030383463, 0.00655553816,
)  # fmt: skip


@dataclass
class Args:
  pose: PoseName | None = None
  """Show one named pose. Omit to print a comparison of all of them."""
  joints: str | None = None
  """Comma-separated joint values instead of a named pose: 12 leg values in
  JOINT_NAMES order, or 20 in NUbots ServoID order."""
  viewer: bool = False
  """Open the interactive viewer instead of only printing."""
  settle: bool = True
  """Drop the robot so its lower sole rests on the floor."""


def build_model() -> mujoco.MjModel:
  """The NUgus with mjlab's actuators, plus a floor for reference."""
  spec = Entity(get_nugus_robot_cfg()).spec
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    contype=1,
    conaffinity=1,
    condim=3,
    rgba=[0.35, 0.37, 0.40, 1.0],
  )
  spec.worldbody.add_light(
    pos=[0.0, 0.0, 3.0],
    dir=[0.0, 0.0, -1.0],
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  )
  return spec.compile()


def walk_engine_stance(model: mujoco.MjModel, exact: bool) -> dict[str, float]:
  """Joint angles the walk engine holds when commanded zero velocity."""
  controller = QuinticWalkController(
    num_envs=1, device="cpu", exact_ik_model=model if exact else None
  )
  targets = controller.compute(
    dt=0.01,
    velocity_command=torch.zeros(1, 3),
    torso_rotation_w=torch.eye(3).unsqueeze(0),
    gyro_b=torch.zeros(1, 3),
  )
  return dict(zip(JOINT_NAMES, targets[0].tolist(), strict=True))


def named_pose(model: mujoco.MjModel, name: PoseName) -> dict[str, float]:
  if name == "walk-stance":
    return walk_engine_stance(model, exact=False)
  if name == "walk-stance-exact":
    return walk_engine_stance(model, exact=True)
  if name == "rl-default":
    return dict(zip(SERVO_ID_ORDER, RL_DEFAULT_POSE, strict=True))
  if name == "stand-script":
    return dict(zip(SERVO_ID_ORDER, STAND_SCRIPT_POSE, strict=True))
  assert STAND_BENT_KNEES_KEYFRAME.joint_pos is not None
  return dict(STAND_BENT_KNEES_KEYFRAME.joint_pos)


def parse_joints(text: str) -> dict[str, float]:
  values = [float(part) for part in text.replace(" ", "").split(",") if part]
  if len(values) == 12:
    return dict(zip(JOINT_NAMES, values, strict=True))
  if len(values) == 20:
    return dict(zip(SERVO_ID_ORDER, values, strict=True))
  raise ValueError(f"expected 12 or 20 joint values, got {len(values)}")


def apply_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  pose: dict[str, float],
  settle: bool,
  level_feet: bool = True,
) -> None:
  """Write a pose into qpos, standing it on the floor.

  Args:
    level_feet: Rotate the torso so the soles lie flat on the ground before
      measuring. Without this the comparison is not like-for-like: the walk
      engine commands the feet pitched 12 degrees relative to the torso, so
      holding the torso upright would stand that pose on its heels and
      understate its height.
  """
  data.qpos[:] = 0.0
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in pose.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"joint {name!r} not in model")
    data.qpos[model.jnt_qposadr[joint_id]] = value
  mujoco.mj_forward(model, data)

  if level_feet:
    # Orient the torso as the inverse of the sole's orientation, so the sole
    # ends up level in the world.
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    sole_rot = data.xmat[body].reshape(3, 3) @ np.array(NUGUS_SOLE_ROTATION)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(sole_rot.T).ravel())
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)

  if settle:
    data.qpos[2] -= min(sole_heights(model, data).values())
    mujoco.mj_forward(model, data)


def sole_heights(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
  heights = {}
  for side in ("left", "right"):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    rotation = data.xmat[body].reshape(3, 3)
    heights[side] = float((data.xpos[body] + rotation @ np.array(NUGUS_SOLE_OFFSET))[2])
  return heights


def measurements(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
  """Landmark heights, chosen so they can be checked with a tape measure."""

  def height(kind, name: str) -> float:
    index = mujoco.mj_name2id(model, kind, name)
    return float(data.xpos[index][2])

  soles = sole_heights(model, data)
  floor = min(soles.values())
  return {
    "hip axis": height(mujoco.mjtObj.mjOBJ_BODY, "torso") - floor,
    "head": height(mujoco.mjtObj.mjOBJ_BODY, "head") - floor,
    "foot sep": abs(
      float(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")][1]
        - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")][1]
      )
    ),
  }


def compare_all(model: mujoco.MjModel, settle: bool) -> None:
  """Print every named pose side by side."""
  names: list[PoseName] = [
    "walk-stance",
    "walk-stance-exact",
    "rl-default",
    "stand-script",
    "keyframe",
  ]
  data = mujoco.MjData(model)
  poses, stats = {}, {}
  for name in names:
    poses[name] = named_pose(model, name)
    apply_pose(model, data, poses[name], settle)
    stats[name] = measurements(model, data)

  header = f"{'joint':>20}" + "".join(f"{n:>19}" for n in names)
  print(header)
  print("-" * len(header))
  for joint in JOINT_NAMES:
    row = f"{joint:>20}"
    for name in names:
      value = poses[name].get(joint)
      row += f"{value:>19.3f}" if value is not None else f"{'-':>19}"
    print(row)
  print("-" * len(header))
  for label in ("hip axis", "head", "foot sep"):
    row = f"{label + ' (m)':>20}"
    for name in names:
      row += f"{stats[name][label]:>19.3f}"
    print(row)
  print(
    "\nAll five drive the SAME model geometry; NUbots' own nugus.xml has an "
    "identical\nkinematic tree, and the mjlab<->NUbots joint conversion in "
    "RLWalk.cpp is a pure\npermutation. So any pose difference you see is a "
    "difference in the joint angles\nbeing commanded, not in the model."
  )


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  model = build_model()

  if args.joints is not None:
    pose, label = parse_joints(args.joints), "custom"
  elif args.pose is not None:
    pose, label = named_pose(model, args.pose), args.pose
  else:
    compare_all(model, args.settle)
    return

  data = mujoco.MjData(model)
  apply_pose(model, data, pose, args.settle)
  print(f"pose: {label}")
  for joint in JOINT_NAMES:
    if joint in pose:
      print(f"  {joint:>20} {pose[joint]:+.4f}")
  for name, value in measurements(model, data).items():
    print(f"  {name:>20} {value:+.4f} m")

  if args.viewer:
    from mujoco import viewer as mujoco_viewer

    frozen = data.qpos.copy()
    with mujoco_viewer.launch_passive(model, data) as viewer:
      while viewer.is_running():
        # Hold the pose: this is a kinematic inspector, not a simulation.
        data.qpos[:] = frozen
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  main()
