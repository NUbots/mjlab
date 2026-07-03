"""Tests for NUgus BAM friction baselines and joint-friction DR."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.nugus.nugus_constants import (
  NUGUS_MOTOR_JOINT_REGEX,
  get_nugus_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg


class _Env:
  def __init__(self, scene: Scene, sim: Simulation, device: str) -> None:
    self.scene = scene
    self.sim = sim
    self.num_envs = scene.num_envs
    self.device = device


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _motor_joint_names(entity: Entity) -> tuple[str, ...]:
  return tuple(n for n in entity.joint_names if not n.endswith("_backlash"))


def _motor_dof_adrs(
  model: mujoco.MjModel, motor_joint_names: tuple[str, ...]
) -> list[int]:
  return [
    model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in motor_joint_names
  ]


def test_nugus_motor_dof_frictionloss_positive() -> None:
  entity = Entity(get_nugus_robot_cfg())
  model = entity.compile()
  motor_dofs = _motor_dof_adrs(model, _motor_joint_names(entity))
  assert len(motor_dofs) == 20
  friction = model.dof_frictionloss[motor_dofs]
  assert all(f > 0.0 for f in friction)


def _create_nugus_dr_env(device: str, num_envs: int = 2) -> _Env:
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    entities={"robot": get_nugus_robot_cfg()},
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
  return _Env(scene, sim, device)


def _motor_dof_indices(robot) -> torch.Tensor:
  motor_joint_ids, _ = robot.find_joints(NUGUS_MOTOR_JOINT_REGEX)
  return robot.indexing.joint_v_adr[motor_joint_ids]


def test_nugus_joint_friction_dr_changes_on_reset(device: str) -> None:
  env = _create_nugus_dr_env(device)
  robot = env.scene["robot"]
  asset_cfg = SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))
  dof_adr = _motor_dof_indices(robot)

  torch.manual_seed(1)
  dr.joint_friction(
    env,
    env_ids=None,
    ranges=(0.5, 1.5),
    operation="scale",
    asset_cfg=asset_cfg,
  )
  first = env.sim.model.dof_frictionloss[0, dof_adr].clone()

  torch.manual_seed(2)
  dr.joint_friction(
    env,
    env_ids=None,
    ranges=(0.5, 1.5),
    operation="scale",
    asset_cfg=asset_cfg,
  )
  second = env.sim.model.dof_frictionloss[0, dof_adr]

  assert torch.all(first > 0)
  assert not torch.allclose(first, second)


def test_nugus_joint_damping_dr_changes_on_reset(device: str) -> None:
  env = _create_nugus_dr_env(device)
  robot = env.scene["robot"]
  asset_cfg = SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))
  dof_adr = _motor_dof_indices(robot)

  torch.manual_seed(3)
  dr.joint_damping(
    env,
    env_ids=None,
    ranges=(0.8, 1.2),
    operation="scale",
    asset_cfg=asset_cfg,
  )
  first = env.sim.model.dof_damping[0, dof_adr].clone()

  torch.manual_seed(4)
  dr.joint_damping(
    env,
    env_ids=None,
    ranges=(0.8, 1.2),
    operation="scale",
    asset_cfg=asset_cfg,
  )
  second = env.sim.model.dof_damping[0, dof_adr]

  assert torch.all(first > 0)
  assert not torch.allclose(first, second)
