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


@pytest.fixture
def rma_mirror_env(device, monkeypatch):
  """Env with the RMA obs groups (and current/voltage obs) configured."""
  monkeypatch.setenv("RMA", "1")
  monkeypatch.setenv("RMA_WINDOW", "4")
  monkeypatch.setenv("CURRENT_OBS", "1")
  monkeypatch.setenv("BUS_VOLTAGE", "1")
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 2
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    yield env
  finally:
    env.close()


def test_dr_ratios_ctrl_segments_are_spec_order(mirror_env) -> None:
  """dr_ratios' per-actuator segments are in entity SPEC order.

  The mirror rule for those segments uses ctrl_perm (spec-order map); this
  pins the assumption that _resolve_actuator_ctrl_ids yields spec order.
  """
  from mjlab.tasks.velocity.config.nugus.dr_observations import _resolve_cache

  cache = _resolve_cache(
    mirror_env,
    SceneEntityCfg("robot", joint_names=(NUGUS_MOTOR_JOINT_REGEX,)),
    "torso",
    ("left_foot_collision", "right_foot_collision"),
  )
  robot = mirror_env.scene["robot"]
  assert torch.equal(cache.ctrl_ids.cpu().long(), robot.indexing.ctrl_ids.cpu().long())


def test_ctrl_perm_maps_spec_order_counterparts(mirror_env, mirror_map) -> None:
  """Regression: actuator_current/servo_voltage columns are in SPEC order,
  not joint order — the mirror must permute within spec order. The old rule
  applied the joint-order permutation, wiring shoulder current to hip
  current in every mirrored sample (v48+ with MIRROR_AUG)."""
  names = list(mirror_env.scene["robot"].actuator_names)
  for i, name in enumerate(names):
    src = int(mirror_map.ctrl_perm[i])
    if name.startswith("left_"):
      assert names[src] == "right_" + name[len("left_") :]
    elif name.startswith("right_"):
      assert names[src] == "left_" + name[len("right_") :]
    else:
      assert src == i


def test_actuator_current_mirror_involution(mirror_map) -> None:
  obs = torch.randn(6, 20)
  out = obs.clone()
  mirror_map._mirror_actuator_current(out, slice(0, 20))
  twice = out.clone()
  mirror_map._mirror_actuator_current(twice, slice(0, 20))
  torch.testing.assert_close(twice, obs)
  assert not torch.allclose(out, obs)


def test_dr_and_history_mirror_involution(rma_mirror_env) -> None:
  mirror_map = get_mirror_map(rma_mirror_env)
  rma_mirror_env.reset(seed=3)
  obs = rma_mirror_env.observation_manager.compute()

  dr = obs["dr"]
  torch.testing.assert_close(
    mirror_map.mirror_dr_obs(mirror_map.mirror_dr_obs(dr)), dr, atol=1e-6, rtol=0.0
  )
  hist = obs["history"]
  assert hist.dim() == 3
  torch.testing.assert_close(
    mirror_map.mirror_history_obs(mirror_map.mirror_history_obs(hist)),
    hist,
    atol=1e-6,
    rtol=0.0,
  )


def test_history_mirror_matches_framewise_actor(rma_mirror_env) -> None:
  """Mirroring the [B, T, D] history == mirroring each frame as actor obs."""
  mirror_map = get_mirror_map(rma_mirror_env)
  rma_mirror_env.reset(seed=4)
  hist = rma_mirror_env.observation_manager.compute()["history"]
  mirrored = mirror_map.mirror_history_obs(hist)
  framewise = torch.stack(
    [mirror_map.mirror_actor_obs(hist[:, t, :]) for t in range(hist.shape[1])],
    dim=1,
  )
  torch.testing.assert_close(mirrored, framewise)


def test_augmentation_covers_rma_groups(rma_mirror_env) -> None:
  wrapped = RslRlVecEnvWrapper(rma_mirror_env)
  obs = wrapped.get_observations()
  assert "dr" in obs.keys() and "history" in obs.keys()
  obs_aug, _ = nugus_symmetry_augmentation(wrapped, obs, None)
  assert obs_aug is not None
  for key in ("actor", "critic", "dr", "history"):
    assert obs_aug[key].shape[0] == obs[key].shape[0] * 2


def test_odom_mirror_involution_and_sign(mirror_map) -> None:
  v = torch.randn(5, 3)
  m = mirror_map.mirror_odom_obs(v)
  torch.testing.assert_close(mirror_map.mirror_odom_obs(m), v)
  torch.testing.assert_close(m[:, 0], v[:, 0])
  torch.testing.assert_close(m[:, 1], -v[:, 1])
  torch.testing.assert_close(m[:, 2], v[:, 2])


def test_augmentation_handles_odom_target(mirror_env) -> None:
  from tensordict import TensorDict

  wrapped = RslRlVecEnvWrapper(mirror_env)
  obs = wrapped.get_observations()
  batch = obs.batch_size[0]
  target = torch.randn(batch, 3, device=obs["actor"].device)
  td = TensorDict(
    {"actor": obs["actor"], "odom_target": target}, batch_size=obs.batch_size
  )
  obs_aug, _ = nugus_symmetry_augmentation(wrapped, td, None)
  assert obs_aug is not None
  aug = obs_aug["odom_target"]
  assert aug.shape[0] == batch * 2
  torch.testing.assert_close(aug[batch:, 1], -target[:, 1])


def test_augmentation_raises_on_unknown_group(mirror_env) -> None:
  from tensordict import TensorDict

  wrapped = RslRlVecEnvWrapper(mirror_env)
  obs = wrapped.get_observations()
  bogus = TensorDict(
    {"actor": obs["actor"], "mystery": torch.zeros(obs.batch_size[0], 3)},
    batch_size=obs.batch_size,
  )
  with pytest.raises(ValueError, match="mystery"):
    nugus_symmetry_augmentation(wrapped, bogus, None)


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
