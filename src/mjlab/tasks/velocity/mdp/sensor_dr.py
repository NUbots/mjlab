"""Per-episode IMU-mounting and gravity-incline DR (doc 11 ideas #12/#13).

Two guaranteed real-world conditions the sim otherwise presents as exact:

- The IMU is never mounted perfectly: a small fixed rotation between the
  sensor frame and the body frame. Modeled as a per-episode random
  rotation applied to the ACTOR's gyro and projected-gravity terms only
  (the critic keeps privileged clean readings). On hardware the raw IMU
  carries the real mounting error naturally, so deployment needs no
  changes.
- Fields and labs are never level: a per-episode tilt of the gravity
  vector (mujoco_warp carries gravity per world), equivalent to walking
  on a gentle incline without the cost of terrain geometry.

Both draw from symmetric distributions, so left-right mirror
augmentation remains statistically valid (the mirrored sample
corresponds to the mirrored draw, same as the noise/delay realization
argument for the history group).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.observations import builtin_sensor, projected_gravity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
  "gravity_incline",
  "imu_base_ang_vel",
  "imu_mounting_bias",
  "imu_projected_gravity",
]


def _random_small_rotations(
  n: int, max_angle_rad: float, device: torch.device | str
) -> torch.Tensor:
  """n random rotation matrices: uniform axis, uniform angle in [0, max]."""
  axis = torch.randn(n, 3, device=device)
  axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-9)
  angle = torch.rand(n, device=device) * max_angle_rad
  # Rodrigues' formula.
  k = axis
  c = torch.cos(angle).view(n, 1, 1)
  s = torch.sin(angle).view(n, 1, 1)
  eye = torch.eye(3, device=device).expand(n, 3, 3)
  kx = torch.zeros(n, 3, 3, device=device)
  kx[:, 0, 1] = -k[:, 2]
  kx[:, 0, 2] = k[:, 1]
  kx[:, 1, 0] = k[:, 2]
  kx[:, 1, 2] = -k[:, 0]
  kx[:, 2, 0] = -k[:, 1]
  kx[:, 2, 1] = k[:, 0]
  outer = k.unsqueeze(2) * k.unsqueeze(1)
  return c * eye + s * kx + (1.0 - c) * outer


def imu_mounting_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_angle_deg: float = 2.5,
) -> None:
  """Reset event: draw a fixed per-episode IMU mounting rotation."""
  buf = getattr(env, "_imu_mount_rot", None)
  if buf is None:
    buf = torch.eye(3, device=env.device).expand(env.num_envs, 3, 3).contiguous()
    env._imu_mount_rot = buf  # type: ignore[attr-defined]
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  buf[ids] = _random_small_rotations(
    int(ids.numel()), math.radians(max_angle_deg), env.device
  )


def _apply_mount(env: ManagerBasedRlEnv, vec: torch.Tensor) -> torch.Tensor:
  rot = getattr(env, "_imu_mount_rot", None)
  if rot is None:
    return vec
  return torch.bmm(rot, vec.unsqueeze(-1)).squeeze(-1)


def imu_base_ang_vel(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Gyro sensor reading through the (randomized) IMU mounting rotation.

  Drop-in for the actor's ``builtin_sensor`` gyro term (same signature,
  the term's ``sensor_name`` param is preserved by the config swap).
  """
  return _apply_mount(env, builtin_sensor(env, sensor_name))


def imu_projected_gravity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Projected gravity through the (randomized) IMU mounting rotation."""
  return _apply_mount(env, projected_gravity(env, asset_cfg))


def gravity_incline(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_angle_deg: float = 2.5,
) -> None:
  """Reset event: tilt this env's gravity vector by a random small angle.

  Rotates the NOMINAL gravity each time (no compounding); azimuth uniform,
  tilt angle uniform in [0, max]. mujoco_warp stores gravity per world.
  """
  gravity = env.sim.model.opt.gravity  # [nworld, 3]
  nominal = getattr(env, "_gravity_nominal", None)
  if nominal is None:
    nominal = gravity[0].clone()
    env._gravity_nominal = nominal  # type: ignore[attr-defined]
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  n = int(ids.numel())
  g_mag = nominal.norm()
  tilt = torch.rand(n, device=env.device) * math.radians(max_angle_deg)
  azimuth = torch.rand(n, device=env.device) * (2.0 * math.pi)
  tilted = (
    torch.stack(
      [
        torch.sin(tilt) * torch.cos(azimuth),
        torch.sin(tilt) * torch.sin(azimuth),
        -torch.cos(tilt),
      ],
      dim=1,
    )
    * g_mag
  )
  gravity[ids] = tilted
