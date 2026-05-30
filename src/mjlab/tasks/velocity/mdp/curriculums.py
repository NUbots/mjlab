from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class DelayStage(TypedDict):
  step: int
  # Maps observation term name to its (min_lag, max_lag) for this stage onward.
  lags: dict[str, tuple[int, int]]


class ActuatorDelayStage(TypedDict):
  step: int
  # (min_lag, max_lag) applied uniformly to every delayed actuator on the asset.
  lag: tuple[int, int]


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def observation_delay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  delay_stages: list[DelayStage],
  group_name: str = "actor",
) -> dict[str, torch.Tensor]:
  """Anneal observation delay over training.

  Lets the policy learn a clean dynamic gait at low/no delay first, then ramps the
  lag range up for sim-to-real robustness. The corresponding observation terms must
  be configured with ``delay_max_lag`` equal to the largest ``max_lag`` any stage
  requests, so the underlying ring buffer is sized to serve it.
  """
  del env_ids  # Unused; delay schedule is global on the step counter.
  obs_manager = env.observation_manager

  active: dict[str, tuple[int, int]] = {}
  for stage in delay_stages:
    if env.common_step_counter >= stage["step"]:
      active.update(stage["lags"])

  metrics: dict[str, torch.Tensor] = {}
  for term_name, (min_lag, max_lag) in active.items():
    buffer = obs_manager.get_delay_buffer(group_name, term_name)
    if buffer is None:
      raise ValueError(
        f"Observation term '{term_name}' in group '{group_name}' has no delay "
        "buffer. Set its delay_max_lag to the curriculum's largest max_lag."
      )
    buffer.set_lag_range(min_lag, max_lag)
    metrics[f"{term_name}_min_lag"] = torch.tensor(float(min_lag))
    metrics[f"{term_name}_max_lag"] = torch.tensor(float(max_lag))
  return metrics


def actuator_delay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  delay_stages: list[ActuatorDelayStage],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Anneal actuator command delay over training.

  Mirrors :func:`observation_delay` but on the actuator side: lets the policy
  learn a clean dynamic gait with no command lag first, then ramps the lag range
  up for sim-to-real robustness. Every delayed actuator on the asset must be
  configured with ``delay_max_lag`` >= the largest ``max_lag`` any stage requests.
  """
  del env_ids  # Unused; delay schedule is global on the step counter.
  asset: Entity = env.scene[asset_cfg.name]

  active: tuple[int, int] | None = None
  for stage in delay_stages:
    if env.common_step_counter >= stage["step"]:
      active = stage["lag"]
  if active is None:
    return {}

  min_lag, max_lag = active
  # Builtin actuators may share a fused buffer across actuators; dedupe by id.
  seen: set[int] = set()
  for act in asset.actuators:
    buffer = act.delay_buffer
    if buffer is None or id(buffer) in seen:
      continue
    seen.add(id(buffer))
    buffer.set_lag_range(min_lag, max_lag)
  return {
    "min_lag": torch.tensor(float(min_lag)),
    "max_lag": torch.tensor(float(max_lag)),
  }
