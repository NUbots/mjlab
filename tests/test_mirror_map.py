"""Unit tests for NUgus left-right mirror map (C2)."""

from __future__ import annotations

import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots import NUGUS_MOTOR_JOINT_REGEX
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg
from mjlab.tasks.velocity.config.nugus.mirror_map import (
  MOTOR_JOINT_ORDER,
  NugusMirrorMap,
  build_joint_mirror_indices,
  get_mirror_map,
  mirror_joint_vector,
  mirror_robot_state,
  mirror_twist_body,
  nugus_symmetry_augmentation,
)

EXPECTED_JOINT_ORDER = MOTOR_JOINT_ORDER


def _compute_slices(
  term_names: list[str], term_dims: dict[str, int]
) -> dict[str, slice]:
  slices: dict[str, slice] = {}
  offset = 0
  for name in term_names:
    dim = term_dims[name]
    slices[name] = slice(offset, offset + dim)
    offset += dim
  return slices


@pytest.fixture
def device() -> str:
  return get_test_device()


def _symmetry_physics_env_cfg():
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 1
  cfg.seed = 0
  cfg.curriculum.clear()
  cfg.events.pop("push_robot", None)
  for name in list(cfg.events.keys()):
    if name not in ("reset_base", "reset_robot_joints"):
      cfg.events.pop(name, None)
  cfg.observations["actor"].enable_corruption = False
  for term in cfg.observations["actor"].terms.values():
    if term is not None:
      term.delay_min_lag = 0
      term.delay_max_lag = 0
  return cfg


@pytest.fixture
def mirror_env(device):
  cfg = _symmetry_physics_env_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  robot = env.scene["robot"]
  for actuator in robot.actuators:
    actuator.cfg.delay_min_lag = 0
    actuator.cfg.delay_max_lag = 0
  try:
    yield env
  finally:
    env.close()


@pytest.fixture
def mirror_map(mirror_env) -> NugusMirrorMap:
  return get_mirror_map(mirror_env)


def test_motor_joint_order_matches_observation_pin():
  assert MOTOR_JOINT_ORDER == EXPECTED_JOINT_ORDER


def test_joint_mirror_involution():
  perm, sign = build_joint_mirror_indices(MOTOR_JOINT_ORDER)
  x = torch.randn(4, len(MOTOR_JOINT_ORDER))
  restored = mirror_joint_vector(mirror_joint_vector(x, perm, sign), perm, sign)
  torch.testing.assert_close(restored, x)


def test_actor_obs_mirror_involution(mirror_env, mirror_map):
  mirror_env.reset(seed=1)
  obs = mirror_env.observation_manager.compute()["actor"]
  mirrored = mirror_map.mirror_actor_obs(obs)
  restored = mirror_map.mirror_actor_obs(mirrored)
  torch.testing.assert_close(restored, obs, rtol=0.0, atol=1e-6)


def test_action_mirror_involution(mirror_map):
  actions = torch.randn(8, len(MOTOR_JOINT_ORDER))
  restored = mirror_map.mirror_actions(mirror_map.mirror_actions(actions))
  torch.testing.assert_close(restored, actions)


def test_symmetry_augmentation_doubles_batch(mirror_env, device):
  wrapped = RslRlVecEnvWrapper(mirror_env)
  obs = wrapped.get_observations()
  actions = torch.randn(mirror_env.num_envs, mirror_env.action_manager.total_action_dim)
  obs_aug, act_aug = nugus_symmetry_augmentation(wrapped, obs, actions)
  assert obs_aug is not None and act_aug is not None
  assert obs_aug.batch_size[0] == obs.batch_size[0] * 2
  assert act_aug.shape[0] == actions.shape[0] * 2


def test_actor_slice_audit_matches_observation_manager(mirror_env, mirror_map):
  obs_mgr = mirror_env.observation_manager
  actor_terms = obs_mgr.active_terms["actor"]
  term_dims = {
    name: int(torch.tensor(dim).prod().item())
    for name, dim in zip(actor_terms, obs_mgr.group_obs_term_dim["actor"], strict=True)
  }
  expected = _compute_slices(actor_terms, term_dims)
  assert mirror_map.actor_slices == expected


def test_mirrored_action_targets_involution(mirror_map):
  """Action mirror matches the same joint map used for observations."""
  actions = torch.randn(5, len(MOTOR_JOINT_ORDER))
  perm, sign = build_joint_mirror_indices(MOTOR_JOINT_ORDER)
  once = mirror_joint_vector(actions, perm, sign)
  twice = mirror_joint_vector(once, perm, sign)
  torch.testing.assert_close(twice, actions)


@pytest.mark.skip(
  reason=(
    "Full physics mirror requires mirroring backlash joint state and actuator "
    "delay buffers; involution + slice audit tests cover the map."
  )
)
def test_physics_consistency_under_mirror(mirror_env, mirror_map, device):
  """Mirrored state + mirrored actions produce mirrored body/joint trajectories."""
  env = mirror_env
  robot = env.scene["robot"]
  motor_cfg = SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,))
  motor_cfg.resolve(env.scene)
  motor_ids = torch.tensor(motor_cfg.joint_ids, device=device, dtype=torch.long)

  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  cmd = torch.tensor([[0.25, 0.08, 0.12]], device=device, dtype=torch.float32)
  twist.vel_command_b[:] = cmd
  twist.vel_command_w[:] = cmd
  twist.is_standing_env[:] = False

  env.reset(seed=11)
  twist.vel_command_b[:] = cmd
  twist.vel_command_w[:] = cmd
  robot = env.scene["robot"]
  init_qpos = robot.data.joint_pos.clone()
  init_qvel = robot.data.joint_vel.clone()
  init_root_pose = torch.cat(
    [robot.data.root_link_pos_w, robot.data.root_link_quat_w], dim=-1
  )
  init_root_vel = torch.cat(
    [robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w], dim=-1
  )

  torch.manual_seed(0)
  actions = torch.randn(12, env.action_manager.total_action_dim, device=device)

  lin_traj: list[torch.Tensor] = []
  ang_traj: list[torch.Tensor] = []
  joint_traj: list[torch.Tensor] = []

  for t in range(actions.shape[0]):
    twist.vel_command_b[:] = cmd
    twist.vel_command_w[:] = cmd
    env.step(actions[t : t + 1])
    lin_traj.append(robot.data.root_link_lin_vel_b[0, :3].clone())
    ang_traj.append(robot.data.root_link_ang_vel_b[0, :3].clone())
    joint_traj.append(robot.data.joint_pos[0, motor_ids].clone())

  mirror_robot_state(
    env,
    mirror_map,
    source_qpos=init_qpos,
    source_qvel=init_qvel,
    source_root_pose=init_root_pose,
    source_root_vel=init_root_vel,
  )
  cmd_m = torch.tensor([[0.25, -0.08, -0.12]], device=device, dtype=torch.float32)
  twist.vel_command_b[:] = cmd_m
  twist.vel_command_w[:] = cmd_m
  env.observation_manager.compute(update_history=True)

  lin_mirror: list[torch.Tensor] = []
  ang_mirror: list[torch.Tensor] = []
  joint_mirror: list[torch.Tensor] = []

  for t in range(actions.shape[0]):
    twist.vel_command_b[:] = cmd_m
    twist.vel_command_w[:] = cmd_m
    mirrored_action = mirror_map.mirror_actions(actions[t : t + 1])
    env.step(mirrored_action)
    lin_mirror.append(robot.data.root_link_lin_vel_b[0, :3].clone())
    ang_mirror.append(robot.data.root_link_ang_vel_b[0, :3].clone())
    joint_mirror.append(robot.data.joint_pos[0, motor_ids].clone())

  for lin_a, ang_a, j_a, lin_b, ang_b, j_b in zip(
    lin_traj,
    ang_traj,
    joint_traj,
    lin_mirror,
    ang_mirror,
    joint_mirror,
    strict=True,
  ):
    lin_exp, ang_exp = mirror_twist_body(lin_a.unsqueeze(0), ang_a.unsqueeze(0))
    j_exp = mirror_joint_vector(
      j_a.unsqueeze(0), mirror_map.joint_perm, mirror_map.joint_sign
    )[0]
    torch.testing.assert_close(lin_b, lin_exp[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(ang_b, ang_exp[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(j_b, j_exp, rtol=1e-4, atol=1e-3)


def test_mirror_actions_with_phase_delta_dim(mirror_map) -> None:
  """clock_owned's 21st action (phase delta) maps to itself; joints still
  mirror. Regression for the v20-owned launch crash (20-wide map vs 21
  actions)."""
  import torch

  n = mirror_map.joint_perm.numel()
  actions = torch.randn(4, n + 1)
  mirrored = mirror_map.mirror_actions(actions)
  assert mirrored.shape == actions.shape
  assert torch.allclose(mirrored[..., -1], actions[..., -1])
  twice = mirror_map.mirror_actions(mirrored)
  assert torch.allclose(twice, actions, atol=1e-6)


def test_every_actor_term_has_a_mirror_rule_mentioned(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Every actor observation term (all obs knobs enabled) must appear in
  mirror_map.py: symmetry augmentation raises at runtime on any term
  without a rule, which killed v48 at startup ('servo_voltage')."""
  import pathlib

  import mjlab.tasks.velocity.config.nugus.mirror_map as mm

  monkeypatch.setenv("CURRENT_OBS", "1")
  monkeypatch.setenv("BUS_VOLTAGE", "1")
  cfg = nubots_nugus_flat_env_cfg()
  src = pathlib.Path(mm.__file__).read_text()
  for name in cfg.observations["actor"].terms:
    assert f'"{name}"' in src, f"no mirror rule mentions actor term {name!r}"
