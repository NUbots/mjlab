"""Config audit tests for NUgus Phase 0 (no-op DR / reward / k8s coverage)."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_CFGS = _REPO_ROOT / "src/mjlab/tasks/velocity/config/nugus/env_cfgs.py"
_K8S_TEMPLATE = _REPO_ROOT / "scripts/k8s/volcano-train-job.template.yaml"

# Gait / cadence terms that need coordinated walking before they produce signal.
REWARD_FIRE_SKIP: frozenset[str] = frozenset(
  {
    "air_time",
    "foot_swing_height_landing",
    "gait_phase_regularity",
    "phase_delta_nominal",
  }
)

_ENV_VAR_RE = re.compile(r'_env_(?:float|int|bool|str)\(\s*"([A-Z][A-Z0-9_]*)"')


@pytest.fixture(autouse=True)
def _clear_nugus_env(monkeypatch: pytest.MonkeyPatch) -> None:
  for key in (
    "CRITIC_HEIGHT_SCAN",
    "TRAINING_REGIME",
    "RESUME",
    "CONT_BASE_STEP",
    "PHASE_ITERATIONS",
    "MAX_ITERATIONS",
    "MJLAB_VARIANT",
    "SEED",
  ):
    monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _scale_dr_events(cfg) -> list[tuple[str, Any]]:
  events: list[tuple[str, Any]] = []
  for name, event in cfg.events.items():
    if event is None:
      continue
    if event.params.get("operation") == "scale":
      events.append((name, event))
  return events


def _resolve_asset_cfg(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> SceneEntityCfg:
  resolved = deepcopy(asset_cfg)
  resolved.resolve(env.scene)
  return resolved


def _assert_nonzero(values: torch.Tensor, event_name: str, field: str) -> None:
  if values.numel() == 0:
    raise AssertionError(f"{event_name}: no targets resolved for {field}")
  if not torch.all(values != 0):
    zero_count = int((values == 0).sum().item())
    raise AssertionError(
      f"{event_name}: {field} has {zero_count}/{values.numel()} zero baselines"
    )


def _check_scale_dr_baselines(
  env: ManagerBasedRlEnv, event_name: str, event: Any
) -> None:
  robot = env.scene["robot"]
  asset_cfg = _resolve_asset_cfg(env, event.params["asset_cfg"])
  func = event.func

  if func in (dr.joint_friction, dr.joint_damping, dr.joint_armature):
    field = {
      dr.joint_friction: "dof_frictionloss",
      dr.joint_damping: "dof_damping",
      dr.joint_armature: "dof_armature",
    }[func]
    dof_adr = robot.indexing.joint_v_adr[asset_cfg.joint_ids]
    defaults = env.sim.get_default_field(field)[dof_adr]
    _assert_nonzero(defaults, event_name, field)
    return

  if func is dr.body_mass:
    body_ids = robot.indexing.body_ids[asset_cfg.body_ids]
    defaults = env.sim.get_default_field("body_mass")[body_ids]
    _assert_nonzero(defaults, event_name, "body_mass")
    return

  if func is dr.effort_limits:
    for actuator in robot.actuators:
      default_limit = getattr(actuator, "default_force_limit", None)
      if default_limit is not None:
        _assert_nonzero(default_limit[0].abs(), event_name, "actuator.force_limit")
        continue
      ctrl_ids = actuator.global_ctrl_ids
      defaults = env.sim.get_default_field("actuator_forcerange")[ctrl_ids]
      _assert_nonzero(defaults.abs(), event_name, "actuator_forcerange")
    return

  if func is dr.pd_gains:
    for actuator in robot.actuators:
      default_kp = getattr(actuator, "default_stiffness", None)
      default_kd = getattr(actuator, "default_damping", None)
      if default_kp is not None and default_kd is not None:
        _assert_nonzero(default_kp[0], event_name, "actuator.stiffness")
        _assert_nonzero(default_kd[0], event_name, "actuator.damping")
        continue
      ctrl_ids = actuator.global_ctrl_ids
      gain_defaults = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
      bias_defaults = env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 1:3]
      _assert_nonzero(gain_defaults, event_name, "actuator_gainprm")
      _assert_nonzero(bias_defaults.abs(), event_name, "actuator_biasprm")
    return

  raise AssertionError(f"{event_name}: unhandled scale DR func {func!r}")


def test_scale_dr_events_have_nonzero_baselines(device: str) -> None:
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 2
  scale_events = _scale_dr_events(cfg)
  assert scale_events, "expected at least one scale DR event in NUgus config"

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    for event_name, event in scale_events:
      _check_scale_dr_baselines(env, event_name, event)
  finally:
    env.close()


def test_active_reward_terms_fire_under_random_actions(device: str) -> None:
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 16
  cfg.seed = 42
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset(seed=42)
    action_dim = env.action_manager.total_action_dim
    fired: dict[str, bool] = {}

    for _ in range(200):
      action = torch.rand(env.num_envs, action_dim, device=device) * 2.0 - 1.0
      env.step(action)
      for idx, name in enumerate(env.reward_manager.active_terms):
        weight = env.reward_manager.get_term_cfg(name).weight
        if weight == 0.0 or name in REWARD_FIRE_SKIP:
          continue
        if torch.any(env.reward_manager._step_reward[:, idx] != 0.0):
          fired[name] = True

    required = [
      name
      for name in env.reward_manager.active_terms
      if env.reward_manager.get_term_cfg(name).weight != 0.0
      and name not in REWARD_FIRE_SKIP
    ]
    silent = [name for name in required if not fired.get(name)]
    assert not silent, f"reward terms never produced nonzero values: {silent}"
  finally:
    env.close()


def test_env_cfgs_env_vars_exported_in_k8s_template() -> None:
  env_cfgs_src = _ENV_CFGS.read_text()
  template_src = _K8S_TEMPLATE.read_text()
  env_vars = sorted(set(_ENV_VAR_RE.findall(env_cfgs_src)))
  assert env_vars, "failed to extract _env_* variables from env_cfgs.py"

  missing = [name for name in env_vars if name not in template_src]
  assert not missing, (
    "env_cfgs.py reads these env vars but volcano-train-job.template.yaml "
    f"does not export them: {missing}"
  )
