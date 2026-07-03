"""Vanilla MuJoCo observation assembly for NUgus deployment / sim2sim."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots import NUGUS_MOTOR_JOINT_REGEX
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.nugus.mirror_map import MOTOR_JOINT_ORDER


@dataclass
class NugusRuntimeObsLayout:
  """Actor observation term slices (concatenated vector)."""

  term_slices: dict[str, slice]
  dim: int


def actor_obs_layout_from_env(env) -> NugusRuntimeObsLayout:
  obs_mgr = env.observation_manager
  names = obs_mgr.active_terms["actor"]
  offset = 0
  slices: dict[str, slice] = {}
  for name, dim in zip(names, obs_mgr.group_obs_term_dim["actor"], strict=True):
    size = int(np.prod(dim))
    slices[name] = slice(offset, offset + size)
    offset += size
  return NugusRuntimeObsLayout(term_slices=slices, dim=offset)


@dataclass
class NugusVanillaIndices:
  motor_qpos_adr: np.ndarray
  motor_qvel_adr: np.ndarray
  default_joint_pos: np.ndarray
  gyro_adr: int
  imu_site_id: int
  action_scale: np.ndarray
  layout: NugusRuntimeObsLayout


def build_vanilla_indices(env, layout: NugusRuntimeObsLayout) -> NugusVanillaIndices:
  robot: Entity = env.scene["robot"]
  motor_cfg = SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))
  motor_cfg.resolve(env.scene)
  joint_ids = motor_cfg.joint_ids
  assert isinstance(joint_ids, list)
  resolved = tuple(robot.joint_names[i] for i in joint_ids)
  if resolved != MOTOR_JOINT_ORDER:
    raise ValueError(f"Unexpected motor joint order: {resolved}")

  model = env.sim.mj_model
  qpos_adr = model.jnt_qposadr[joint_ids]
  qvel_adr = model.jnt_dofadr[joint_ids]
  default_joint_pos = robot.data.default_joint_pos[0, joint_ids].detach().cpu().numpy()

  gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
  if gyro_id < 0:
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
  gyro_adr = model.sensor_adr[gyro_id]

  imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

  joint_action = env.action_manager.get_term("joint_pos")
  scale = joint_action._scale[0, joint_ids].detach().cpu().numpy()

  return NugusVanillaIndices(
    motor_qpos_adr=qpos_adr.copy(),
    motor_qvel_adr=qvel_adr.copy(),
    default_joint_pos=default_joint_pos,
    gyro_adr=int(gyro_adr),
    imu_site_id=int(imu_site_id),
    action_scale=scale,
    layout=layout,
  )


def projected_gravity_from_data(data: mujoco.MjData) -> np.ndarray:
  quat = data.qpos[3:7].copy()
  gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  return _quat_rotate_inverse(quat, gravity_w)


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  q_vec = np.array([x, y, z], dtype=np.float64)
  t = 2.0 * np.cross(q_vec, vec)
  return vec - w * t + np.cross(q_vec, t)


def build_actor_obs(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  indices: NugusVanillaIndices,
  *,
  command: tuple[float, float, float],
  last_action: np.ndarray,
  sim_time: float,
  gait_period: float = 0.7,
  command_threshold: float = 0.05,
) -> np.ndarray:
  """Assemble the actor observation vector from vanilla ``MjData``."""
  layout = indices.layout
  obs = np.zeros(layout.dim, dtype=np.float32)

  obs[layout.term_slices["base_ang_vel"]] = data.sensordata[
    indices.gyro_adr : indices.gyro_adr + 3
  ]
  obs[layout.term_slices["projected_gravity"]] = projected_gravity_from_data(data)

  qpos = data.qpos[indices.motor_qpos_adr]
  qvel = data.qvel[indices.motor_qvel_adr]
  obs[layout.term_slices["joint_pos"]] = qpos - indices.default_joint_pos
  obs[layout.term_slices["joint_vel"]] = qvel
  obs[layout.term_slices["actions"]] = last_action
  obs[layout.term_slices["command"]] = np.asarray(command, dtype=np.float32)

  if "gait_clock" in layout.term_slices:
    sl = layout.term_slices["gait_clock"]
    vx, vy, wz = command
    total = abs(vx) + abs(vy) + abs(wz)
    if total > command_threshold:
      phase = (sim_time / gait_period) % 1.0
      angle = 2.0 * math.pi * phase
      obs[sl.start] = math.sin(angle)
      obs[sl.start + 1] = math.cos(angle)

  return obs
