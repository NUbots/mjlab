"""Critic-only observation of sampled domain-randomization parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_MOTOR_ASSET_CFG = SceneEntityCfg("robot")
_EPS = 1e-8

# Concatenation order (each segment is per-env):
# kp_scale, kd_scale, effort_ratio, armature_ratio, frictionloss_ratio,
# torso_mass_ratio, torso_com_offset (m), foot_friction_ratio.
DR_RATIOS_NUM_ACTUATORS = 20
DR_RATIOS_NUM_MOTOR_DOFS = 20
DR_RATIOS_DIM = 5 * DR_RATIOS_NUM_ACTUATORS + 1 + 3 + 2


@dataclass
class _DrRatioCache:
  ctrl_ids: torch.Tensor
  dof_ids: torch.Tensor
  torso_body_id: int
  foot_geom_ids: torch.Tensor
  default_kp: torch.Tensor
  default_kd: torch.Tensor
  default_effort: torch.Tensor
  default_armature: torch.Tensor
  default_frictionloss: torch.Tensor
  default_torso_mass: torch.Tensor
  default_torso_ipos: torch.Tensor
  default_foot_friction: torch.Tensor


_CACHE: WeakKeyDictionary[ManagerBasedRlEnv, _DrRatioCache] = WeakKeyDictionary()


def _safe_ratio(current: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
  denom = default.abs().clamp_min(_EPS)
  return current / denom


def _resolve_actuator_ctrl_ids(
  asset: Entity, motor_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  if isinstance(motor_asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in motor_asset_cfg.actuator_ids]
  elif isinstance(motor_asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[motor_asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[motor_asset_cfg.actuator_ids]]
  if not isinstance(actuators, list):
    actuators = [actuators]
  ctrl_ids = torch.cat([actuator.global_ctrl_ids for actuator in actuators], dim=0)
  return ctrl_ids.to(dtype=torch.long)


def _resolve_cache(
  env: ManagerBasedRlEnv,
  motor_asset_cfg: SceneEntityCfg,
  torso_body_name: str,
  foot_geom_names: tuple[str, ...],
) -> _DrRatioCache:
  cached = _CACHE.get(env)
  if cached is not None:
    return cached

  motor_cfg = SceneEntityCfg(
    motor_asset_cfg.name,
    joint_names=motor_asset_cfg.joint_names,
    preserve_order=motor_asset_cfg.preserve_order,
  )
  motor_cfg.resolve(env.scene)
  torso_cfg = SceneEntityCfg(motor_asset_cfg.name, body_names=(torso_body_name,))
  torso_cfg.resolve(env.scene)
  foot_cfg = SceneEntityCfg(motor_asset_cfg.name, geom_names=foot_geom_names)
  foot_cfg.resolve(env.scene)

  asset: Entity = env.scene[motor_asset_cfg.name]
  ctrl_ids = _resolve_actuator_ctrl_ids(asset, motor_cfg)
  dof_ids = asset.indexing.joint_v_adr[motor_cfg.joint_ids]
  torso_body_ids = torso_cfg.body_ids
  assert isinstance(torso_body_ids, list)
  torso_body_id = int(asset.indexing.body_ids[torso_body_ids[0]].item())
  foot_geom_ids = asset.indexing.geom_ids[foot_cfg.geom_ids]

  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  default_forcerange = env.sim.get_default_field("actuator_forcerange")
  default_armature = env.sim.get_default_field("dof_armature")
  default_frictionloss = env.sim.get_default_field("dof_frictionloss")
  default_body_mass = env.sim.get_default_field("body_mass")
  default_body_ipos = env.sim.get_default_field("body_ipos")
  default_geom_friction = env.sim.get_default_field("geom_friction")

  cached = _DrRatioCache(
    ctrl_ids=ctrl_ids,
    dof_ids=dof_ids,
    torso_body_id=torso_body_id,
    foot_geom_ids=foot_geom_ids,
    default_kp=default_gainprm[ctrl_ids, 0].clone(),
    default_kd=(-default_biasprm[ctrl_ids, 2]).clone(),
    default_effort=default_forcerange[ctrl_ids, 1].clone(),
    default_armature=default_armature[dof_ids].clone(),
    default_frictionloss=default_frictionloss[dof_ids].clone(),
    default_torso_mass=default_body_mass[torso_body_id].clone(),
    default_torso_ipos=default_body_ipos[torso_body_id].clone(),
    default_foot_friction=default_geom_friction[foot_geom_ids, 0].clone(),
  )
  _CACHE[env] = cached
  return cached


def dr_ratios(
  env: ManagerBasedRlEnv,
  motor_asset_cfg: SceneEntityCfg = _DEFAULT_MOTOR_ASSET_CFG,
  torso_body_name: str = "torso",
  foot_geom_names: tuple[str, ...] = (
    "left_foot_collision",
    "right_foot_collision",
  ),
) -> torch.Tensor:
  """Flat vector of DR parameters normalized against build-time defaults.

  Returns dimensionless ratios for actuator and joint fields, absolute torso
  COM offset in meters, and foot friction ratios. Intended for the critic
  group only so actor checkpoints stay loadable.
  """
  cache = _resolve_cache(env, motor_asset_cfg, torso_body_name, foot_geom_names)
  model = env.sim.model

  kp = model.actuator_gainprm[:, cache.ctrl_ids, 0]
  kd = -model.actuator_biasprm[:, cache.ctrl_ids, 2]
  effort = model.actuator_forcerange[:, cache.ctrl_ids, 1]
  armature = model.dof_armature[:, cache.dof_ids]
  frictionloss = model.dof_frictionloss[:, cache.dof_ids]
  torso_mass = model.body_mass[:, cache.torso_body_id].unsqueeze(-1)
  torso_com = model.body_ipos[
    :, cache.torso_body_id, :
  ] - cache.default_torso_ipos.unsqueeze(0)
  foot_friction = model.geom_friction[:, cache.foot_geom_ids, 0]

  segments = (
    _safe_ratio(kp, cache.default_kp.unsqueeze(0)),
    _safe_ratio(kd, cache.default_kd.unsqueeze(0)),
    _safe_ratio(effort, cache.default_effort.unsqueeze(0)),
    _safe_ratio(armature, cache.default_armature.unsqueeze(0)),
    _safe_ratio(frictionloss, cache.default_frictionloss.unsqueeze(0)),
    _safe_ratio(torso_mass, cache.default_torso_mass.unsqueeze(0).unsqueeze(0)),
    torso_com,
    _safe_ratio(foot_friction, cache.default_foot_friction.unsqueeze(0)),
  )
  return torch.cat(segments, dim=-1)


def dr_ratios_effort_slice(n_actuators: int = DR_RATIOS_NUM_ACTUATORS) -> slice:
  """Slice of ``dr_ratios`` covering per-actuator effort-limit ratios."""
  start = 2 * n_actuators
  return slice(start, start + n_actuators)


def dr_ratios_torso_mass_index(n_actuators: int = DR_RATIOS_NUM_ACTUATORS) -> int:
  """Index of the torso mass ratio within ``dr_ratios``."""
  return 5 * n_actuators
