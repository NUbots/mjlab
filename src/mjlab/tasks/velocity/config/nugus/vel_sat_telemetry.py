"""Per-actuator-group velocity saturation telemetry for NUgus training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_ACTUATOR_ARMS,
  NUGUS_ACTUATOR_HEAD,
  NUGUS_ACTUATOR_HIPS,
  NUGUS_ACTUATOR_LEGS,
  NUGUS_MOTOR_JOINT_REGEX,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_SAT_FRAC = 0.9

_ACTUATOR_GROUPS: dict[str, DcMotorActuatorCfg] = {
  "arms": NUGUS_ACTUATOR_ARMS,
  "hips": NUGUS_ACTUATOR_HIPS,
  "legs": NUGUS_ACTUATOR_LEGS,
  "head": NUGUS_ACTUATOR_HEAD,
}


def _joint_group_limits() -> dict[str, tuple[str, float]]:
  limits: dict[str, tuple[str, float]] = {}
  for group, actuator_cfg in _ACTUATOR_GROUPS.items():
    for joint_name in actuator_cfg.target_names_expr:
      limits[joint_name] = (group, actuator_cfg.velocity_limit)
  return limits


def log_vel_sat_frac(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
) -> None:
  """Log fraction of joints with |qd| > 0.9 × velocity_limit per actuator group."""
  del env_ids
  robot = env.scene["robot"]
  log = env.extras.setdefault("log", {})
  joint_limits = _joint_group_limits()
  group_flags: dict[str, list[torch.Tensor]] = {g: [] for g in _ACTUATOR_GROUPS}

  joint_ids, joint_names = robot.find_joints(NUGUS_MOTOR_JOINT_REGEX)
  qd = robot.data.joint_vel[:, joint_ids]

  for col, joint_name in enumerate(joint_names):
    entry = joint_limits.get(joint_name)
    if entry is None:
      continue
    group, vel_limit = entry
    saturated = qd[:, col].abs() > (_SAT_FRAC * vel_limit)
    group_flags[group].append(saturated)

  for group, flags in group_flags.items():
    if not flags:
      continue
    frac = torch.stack(flags, dim=-1).float().mean()
    log[f"Metrics/vel_sat_frac_{group}"] = frac.mean()
