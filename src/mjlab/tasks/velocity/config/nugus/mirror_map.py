"""Left-right mirror map for NUgus velocity policy symmetry augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.asset_zoo.robots import NUGUS_MOTOR_JOINT_REGEX
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensor
from mjlab.tasks.velocity.config.nugus.dr_observations import DR_RATIOS_NUM_ACTUATORS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

# Motor-joint order in actor ``joint_pos`` / ``joint_vel`` / ``actions`` (see
# ``tests/test_nugus_observation_vector.py``).
MOTOR_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_yaw",
  "left_hip_roll",
  "left_hip_pitch",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_yaw",
  "right_hip_roll",
  "right_hip_pitch",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
  "neck_yaw",
  "head_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow_pitch",
)

_NEGATE_JOINT_KEYS: tuple[tuple[str, float], ...] = (
  ("hip_yaw", -1.0),
  ("hip_roll", -1.0),
  ("hip_pitch", 1.0),
  ("knee_pitch", 1.0),
  ("ankle_pitch", 1.0),
  ("ankle_roll", -1.0),
  ("shoulder_pitch", 1.0),
  ("shoulder_roll", -1.0),
  ("elbow_pitch", 1.0),
  ("neck_yaw", -1.0),
  ("head_pitch", 1.0),
)

_CACHE: WeakKeyDictionary[ManagerBasedRlEnv, NugusMirrorMap] = WeakKeyDictionary()


def _joint_mirror_sign(joint_name: str) -> float:
  for key, sign in _NEGATE_JOINT_KEYS:
    if key in joint_name:
      return sign
  raise ValueError(f"No mirror sign rule for joint {joint_name!r}")


def _counterpart_joint_name(name: str) -> str | None:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return None


def build_joint_mirror_indices(
  joint_names: tuple[str, ...] | list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return ``(perm, sign)`` so ``mirror_joint_vector(x) = sign * x[..., perm]``."""
  names = tuple(joint_names)
  name_to_idx = {name: idx for idx, name in enumerate(names)}
  perm: list[int] = []
  sign: list[float] = []
  for name in names:
    counterpart = _counterpart_joint_name(name)
    src = name_to_idx[counterpart] if counterpart is not None else name_to_idx[name]
    perm.append(src)
    sign.append(_joint_mirror_sign(name))
  return (
    torch.tensor(perm, dtype=torch.long),
    torch.tensor(sign, dtype=torch.float32),
  )


def mirror_joint_vector(
  values: torch.Tensor,
  perm: torch.Tensor,
  sign: torch.Tensor,
) -> torch.Tensor:
  """Apply the joint mirror map along the last dimension."""
  device = values.device
  dtype = values.dtype
  perm_d = perm.to(device=device)
  sign_d = sign.to(device=device, dtype=dtype)
  return values[..., perm_d] * sign_d


def _term_slices(
  term_names: list[str],
  term_dims: list[tuple[int, ...]],
  last_dim_only: bool = False,
) -> dict[str, slice]:
  """Build per-term slices into a group's concatenated observation.

  ``last_dim_only`` is for 3D history groups (``flatten_history_dim=False``),
  whose term dims are ``(T, D_i)`` concatenated along the LAST dim: the slice
  must span ``D_i``, not ``T * D_i``.
  """
  slices: dict[str, slice] = {}
  offset = 0
  for name, dims in zip(term_names, term_dims, strict=True):
    size = int(dims[-1]) if last_dim_only else int(np.prod(dims))
    slices[name] = slice(offset, offset + size)
    offset += size
  return slices


def _terrain_height_scan_grid_shape(env: ManagerBasedRlEnv) -> tuple[int, int] | None:
  """Return ``(num_x, num_y)`` for the critic ``height_scan`` grid, if present."""
  if "height_scan" not in env.observation_manager.active_terms["critic"]:
    return None
  sensor = env.scene["terrain_scan"]
  if not isinstance(sensor, RayCastSensor):
    return None
  pattern = sensor.cfg.pattern
  if not isinstance(pattern, GridPatternCfg):
    return None
  size_x, size_y = pattern.size
  res = pattern.resolution
  nx = int(round(size_x / res)) + 1
  ny = int(round(size_y / res)) + 1
  return nx, ny


@dataclass
class NugusMirrorMap:
  """Runtime mirror map built from live observation term ordering."""

  actor_slices: dict[str, slice]
  critic_slices: dict[str, slice]
  joint_perm: torch.Tensor
  joint_sign: torch.Tensor
  dr_joint_perm: torch.Tensor
  # Actuator/spec-order mirror map. Entity actuator columns (model.ctrl,
  # actuator_current / servo_voltage observations, and dr_ratios' per-
  # actuator segments) are in SPEC order, which differs from the motor
  # joint order (see tests/test_nugus_observation_vector.py) -- applying
  # the joint-order permutation to them scrambles servos cross-limb.
  ctrl_perm: torch.Tensor
  ctrl_sign: torch.Tensor
  motor_joint_ids: torch.Tensor
  default_joint_pos: torch.Tensor
  height_scan_grid: tuple[int, int] | None = None
  # Slices for the RMA "history" (3D, per-frame == actor layout) and "dr"
  # groups; None when those groups are not configured (RMA off).
  history_slices: dict[str, slice] | None = None
  dr_slices: dict[str, slice] | None = None

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> NugusMirrorMap:
    obs_mgr = env.observation_manager
    actor_names = obs_mgr.active_terms["actor"]
    critic_names = obs_mgr.active_terms["critic"]
    actor_slices = _term_slices(actor_names, obs_mgr.group_obs_term_dim["actor"])
    critic_slices = _term_slices(critic_names, obs_mgr.group_obs_term_dim["critic"])
    history_slices = None
    if "history" in obs_mgr.active_terms:
      history_slices = _term_slices(
        obs_mgr.active_terms["history"],
        obs_mgr.group_obs_term_dim["history"],
        last_dim_only=True,
      )
    dr_slices = None
    if "dr" in obs_mgr.active_terms:
      dr_slices = _term_slices(
        obs_mgr.active_terms["dr"], obs_mgr.group_obs_term_dim["dr"]
      )

    robot = env.scene["robot"]
    motor_cfg = SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))
    motor_cfg.resolve(env.scene)
    joint_ids = motor_cfg.joint_ids
    assert isinstance(joint_ids, list)
    resolved_names = tuple(robot.joint_names[i] for i in joint_ids)
    if resolved_names != MOTOR_JOINT_ORDER:
      raise ValueError(
        "Motor joint order changed; update MOTOR_JOINT_ORDER in mirror_map.py "
        f"(got {resolved_names}, expected {MOTOR_JOINT_ORDER})."
      )

    joint_perm, joint_sign = build_joint_mirror_indices(MOTOR_JOINT_ORDER)
    dr_joint_perm, _ = build_joint_mirror_indices(MOTOR_JOINT_ORDER)
    ctrl_perm, ctrl_sign = build_joint_mirror_indices(tuple(robot.actuator_names))
    default_joint_pos = robot.data.default_joint_pos[0, joint_ids].clone()

    return cls(
      actor_slices=actor_slices,
      critic_slices=critic_slices,
      joint_perm=joint_perm,
      joint_sign=joint_sign,
      dr_joint_perm=dr_joint_perm,
      ctrl_perm=ctrl_perm,
      ctrl_sign=ctrl_sign,
      motor_joint_ids=torch.tensor(joint_ids, dtype=torch.long),
      default_joint_pos=default_joint_pos,
      height_scan_grid=_terrain_height_scan_grid_shape(env),
      history_slices=history_slices,
      dr_slices=dr_slices,
    )

  def _mirror_base_ang_vel(self, obs: torch.Tensor, sl: slice) -> None:
    obs[..., sl.start : sl.start + 3] *= torch.tensor(
      [-1.0, 1.0, -1.0], device=obs.device, dtype=obs.dtype
    )

  def _mirror_projected_gravity(self, obs: torch.Tensor, sl: slice) -> None:
    obs[..., sl.start + 1] *= -1.0

  def _mirror_command(self, obs: torch.Tensor, sl: slice) -> None:
    obs[..., sl.start + 1] *= -1.0
    obs[..., sl.start + 2] *= -1.0

  def _mirror_gait_clock(self, obs: torch.Tensor, sl: slice) -> None:
    obs[..., sl] *= -1.0

  def _mirror_joint_block(self, obs: torch.Tensor, sl: slice) -> None:
    block = obs[..., sl]
    n = self.joint_perm.numel()
    if block.shape[-1] == n + 1:
      # clock_owned appends a phase_delta action: cadence rate is
      # left-right symmetric (mirroring swaps feet, which is the clock
      # observation's half-period shift, not a rate change), so the extra
      # channel maps to itself.
      obs[..., sl.start : sl.start + n] = mirror_joint_vector(
        block[..., :n], self.joint_perm, self.joint_sign
      )
      return
    obs[..., sl] = mirror_joint_vector(block, self.joint_perm, self.joint_sign)

  def _permute(self, values: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Left-right permutation along the last dim, no sign flip."""
    return values[..., perm.to(values.device)]

  def _mirror_dr_ratios(self, obs: torch.Tensor, sl: slice) -> None:
    block = obs[..., sl].clone()
    n = DR_RATIOS_NUM_ACTUATORS
    # kp/kd/effort segments are indexed by actuator ctrl ids -> SPEC order;
    # armature/frictionloss are indexed by motor joint dofs -> JOINT order
    # (see dr_observations._resolve_cache). Positive ratios: permute only.
    for i, perm in enumerate(
      (
        self.ctrl_perm,  # kp_scale
        self.ctrl_perm,  # kd_scale
        self.ctrl_perm,  # effort_ratio
        self.dr_joint_perm,  # armature_ratio
        self.dr_joint_perm,  # frictionloss_ratio
      )
    ):
      seg = slice(i * n, (i + 1) * n)
      block[..., seg] = self._permute(block[..., seg], perm)
    torso_mass_idx = 5 * n
    com_start = torso_mass_idx + 1
    block[..., com_start + 1] *= -1.0
    foot_start = com_start + 3
    block[..., foot_start : foot_start + 2] = block[..., foot_start : foot_start + 2][
      ..., [1, 0]
    ]
    obs[..., sl] = block

  def _mirror_dr_extras(self, obs: torch.Tensor, sl: slice) -> None:
    # Layout pinned in dr_observations.DR_EXTRAS_DIM: encoder_bias (20,
    # joint order), bus latents (3, chassis-level: identity), current-
    # sensor gain/offset (20+20, joint order). Encoder bias offsets the
    # joint-angle reading, so it transforms like a joint position (perm +
    # sign); gain/offset are per-servo calibration constants that follow
    # their servo (perm only).
    block = obs[..., sl].clone()
    n = self.dr_joint_perm.numel()
    block[..., :n] = mirror_joint_vector(
      block[..., :n], self.dr_joint_perm, self.joint_sign
    )
    gain_start = n + 3
    for seg_start in (gain_start, gain_start + n):
      seg = slice(seg_start, seg_start + n)
      block[..., seg] = self._permute(block[..., seg], self.dr_joint_perm)
    obs[..., sl] = block

  def _mirror_actuator_current(self, obs: torch.Tensor, sl: slice) -> None:
    # Sensed current is SIGNED (tau / Kt, no abs) and the columns are in
    # actuator SPEC order: mirror = spec-order permutation with the joint
    # sign (mirrored torque flips where the joint angle flips). The per-
    # servo sensor offset does not flip with the sign, so this rule is off
    # by up to 2*|offset| (<= 0.2 A) on sign-flipped channels -- below the
    # 0.25 A observation noise floor.
    out = mirror_joint_vector(obs[..., sl], self.ctrl_perm, self.ctrl_sign)
    obs[..., sl] = out

  def _mirror_servo_voltage(self, obs: torch.Tensor, sl: slice) -> None:
    # Positive supply voltage per servo, SPEC order: permutation only.
    obs[..., sl] = self._permute(obs[..., sl], self.ctrl_perm)

  def _mirror_foot_pair(self, obs: torch.Tensor, sl: slice) -> None:
    block = obs[..., sl]
    obs[..., sl] = block[..., [1, 0]]

  def _mirror_foot_contact_forces(self, obs: torch.Tensor, sl: slice) -> None:
    block = obs[..., sl].clone().view(*obs.shape[:-1], 2, 3)
    block = block[..., [1, 0], :]
    block[..., 1] *= -1.0
    obs[..., sl] = block.reshape(*obs.shape[:-1], 6)

  def _apply_actor_rules(self, out: torch.Tensor, slices: dict[str, slice]) -> None:
    """Apply per-term actor mirror rules in place.

    Every rule indexes only the last (feature) dim, so this works
    unchanged on the 3D ``[B, T, D]`` history group (broadcast over T).
    """
    for name, sl in slices.items():
      if name == "base_ang_vel":
        self._mirror_base_ang_vel(out, sl)
      elif name == "projected_gravity":
        self._mirror_projected_gravity(out, sl)
      elif name in ("joint_pos", "joint_vel", "actions"):
        self._mirror_joint_block(out, sl)
      elif name == "actuator_current":
        self._mirror_actuator_current(out, sl)
      elif name == "servo_voltage":
        self._mirror_servo_voltage(out, sl)
      elif name == "command":
        self._mirror_command(out, sl)
      elif name == "gait_clock":
        self._mirror_gait_clock(out, sl)
      else:
        raise ValueError(f"No actor mirror rule for observation term {name!r}")

  def mirror_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    self._apply_actor_rules(out, self.actor_slices)
    return out

  def mirror_history_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Mirror the RMA history group ``[B, T, D]`` (per-frame actor layout)."""
    if self.history_slices is None:
      raise ValueError("history group not configured in this env")
    out = obs.clone()
    self._apply_actor_rules(out, self.history_slices)
    return out

  def mirror_dr_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Mirror the RMA privileged DR group (dr_ratios + dr_extras)."""
    if self.dr_slices is None:
      raise ValueError("dr group not configured in this env")
    out = obs.clone()
    for name, sl in self.dr_slices.items():
      if name == "dr_ratios":
        self._mirror_dr_ratios(out, sl)
      elif name == "dr_extras":
        self._mirror_dr_extras(out, sl)
      else:
        raise ValueError(f"No dr mirror rule for observation term {name!r}")
    return out

  def mirror_odom_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Mirror the odometry supervision target (body-frame lin vel)."""
    out = obs.clone()
    out[..., 1] *= -1.0
    return out

  def mirror_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    for name, sl in self.critic_slices.items():
      if name == "base_lin_vel":
        out[..., sl.start + 1] *= -1.0
      elif name == "base_ang_vel":
        self._mirror_base_ang_vel(out, sl)
      elif name == "projected_gravity":
        self._mirror_projected_gravity(out, sl)
      elif name in ("joint_pos", "joint_vel", "actions"):
        self._mirror_joint_block(out, sl)
      elif name == "actuator_current":
        self._mirror_actuator_current(out, sl)
      elif name == "servo_voltage":
        self._mirror_servo_voltage(out, sl)
      elif name == "command":
        self._mirror_command(out, sl)
      elif name in ("foot_height", "foot_air_time", "foot_contact"):
        self._mirror_foot_pair(out, sl)
      elif name == "foot_contact_forces":
        self._mirror_foot_contact_forces(out, sl)
      elif name == "dr_ratios":
        self._mirror_dr_ratios(out, sl)
      elif name == "dr_extras":
        self._mirror_dr_extras(out, sl)
      elif name == "gait_clock":
        self._mirror_gait_clock(out, sl)
      elif name == "height_scan":
        width = sl.stop - sl.start
        if self.height_scan_grid is None:
          raise ValueError("height_scan present but terrain_scan grid shape is unknown")
        nx, ny = self.height_scan_grid
        if nx * ny != width:
          raise ValueError(
            f"height_scan slice length {width} does not match grid {nx}x{ny}"
          )
        grid = out[..., sl].view(*out.shape[:-1], nx, ny)
        out[..., sl] = grid.flip(dims=[-1]).reshape(*out.shape[:-1], width)
      else:
        raise ValueError(f"No critic mirror rule for observation term {name!r}")
    return out

  def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
    n = self.joint_perm.numel()
    if actions.shape[-1] == n + 1:
      out = actions.clone()
      out[..., :n] = mirror_joint_vector(
        actions[..., :n], self.joint_perm, self.joint_sign
      )
      return out
    return mirror_joint_vector(actions, self.joint_perm, self.joint_sign)


def get_mirror_map(env: ManagerBasedRlEnv) -> NugusMirrorMap:
  cached = _CACHE.get(env)
  if cached is not None:
    return cached
  cached = NugusMirrorMap.from_env(env)
  _CACHE[env] = cached
  return cached


def _mirror_quaternion_wxyz(quat: torch.Tensor) -> torch.Tensor:
  """Mirror orientation across the sagittal (x-z) plane."""
  w, x, y, z = quat.unbind(-1)
  return torch.stack((w, x, -y, -z), dim=-1)


def mirror_robot_state(
  env: ManagerBasedRlEnv,
  mirror_map: NugusMirrorMap,
  *,
  source_qpos: torch.Tensor | None = None,
  source_qvel: torch.Tensor | None = None,
  source_root_pose: torch.Tensor | None = None,
  source_root_vel: torch.Tensor | None = None,
) -> None:
  """Mirror the floating-base pose and joint state in the simulator."""
  robot = env.scene["robot"]
  env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

  if source_root_pose is None:
    pos = robot.data.root_link_pos_w[env_ids].clone()
    quat = robot.data.root_link_quat_w[env_ids].clone()
  else:
    pos = source_root_pose[env_ids, :3].clone()
    quat = source_root_pose[env_ids, 3:7].clone()

  if source_root_vel is None:
    lin_vel = robot.data.root_link_lin_vel_w[env_ids].clone()
    ang_vel = robot.data.root_link_ang_vel_w[env_ids].clone()
  else:
    lin_vel = source_root_vel[env_ids, :3].clone()
    ang_vel = source_root_vel[env_ids, 3:6].clone()

  pos[:, 1] *= -1.0
  quat = _mirror_quaternion_wxyz(quat)
  lin_vel[:, 1] *= -1.0
  ang_vel[:, 0] *= -1.0
  ang_vel[:, 2] *= -1.0

  robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
  robot.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids)

  if source_qpos is None:
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()
  else:
    joint_pos = source_qpos[env_ids].clone()
    if source_qvel is None:
      joint_vel = robot.data.joint_vel[env_ids].clone()
    else:
      joint_vel = source_qvel[env_ids].clone()

  motor_ids = mirror_map.motor_joint_ids.to(env.device)
  joint_pos[:, motor_ids] = mirror_joint_vector(
    joint_pos[:, motor_ids], mirror_map.joint_perm, mirror_map.joint_sign
  )
  joint_vel[:, motor_ids] = mirror_joint_vector(
    joint_vel[:, motor_ids], mirror_map.joint_perm, mirror_map.joint_sign
  )
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


def mirror_twist_body(
  lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  lin_m = lin_vel_b.clone()
  lin_m[..., 1] *= -1.0
  ang_m = ang_vel_b.clone()
  ang_m[..., 0] *= -1.0
  ang_m[..., 2] *= -1.0
  return lin_m, ang_m


@torch.no_grad()
def nugus_symmetry_augmentation(
  env: RslRlVecEnvWrapper,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl-rl symmetry callback: concatenate original and left-right mirrored samples."""
  mirror_map = get_mirror_map(env.unwrapped)

  obs_aug: TensorDict | None
  if obs is not None:
    batch_size = obs.batch_size[0]
    # Mirror EVERY group present. Unknown groups must raise: silently
    # passing one through un-mirrored (or dropping it, as this callback
    # once did) corrupts half of every augmented batch for that group.
    mirror_fns = {
      "actor": mirror_map.mirror_actor_obs,
      "critic": mirror_map.mirror_critic_obs,
      "dr": mirror_map.mirror_dr_obs,
      "history": mirror_map.mirror_history_obs,
      "odom_target": mirror_map.mirror_odom_obs,
    }
    fields: dict[str, torch.Tensor] = {}
    for key in obs.keys():
      mirror_fn = mirror_fns.get(str(key))
      if mirror_fn is None:
        raise ValueError(f"No mirror rule for observation group {key!r}")
      group_obs = obs[key]
      assert isinstance(group_obs, torch.Tensor)
      fields[str(key)] = torch.cat([group_obs, mirror_fn(group_obs)], dim=0)
    obs_aug = TensorDict(fields, batch_size=[batch_size * 2])  # ty: ignore[invalid-argument-type]
  else:
    obs_aug = None

  if actions is not None:
    mirrored = mirror_map.mirror_actions(actions)
    actions_aug = torch.cat([actions, mirrored], dim=0)
  else:
    actions_aug = None

  return obs_aug, actions_aug
