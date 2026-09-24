"""Tests for the K1 goalkeeper task with the goalie walking before shots."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.goalkeeper.config.booster_k1.env_cfgs import (
  booster_k1_block_env_cfg,
  booster_k1_block_moving_env_cfg,
)
from mjlab.tasks.goalkeeper.mdp import ShotCommand
from mjlab.tasks.goalkeeper.mdp.handoff import WalkHandoffAction, WalkHandoffActionCfg
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import HISTORY_WINDOW
from mjlab.tasks.velocity.mdp import UniformVelocityCommand

ACTOR_DIM = 70
"""The block policy's frame, which skill::K1BlockPolicy builds: unchanged."""
WALK_DIM = 72
"""The walk's frame: lin vel, ang vel, gravity, joint pos/vel/actions, twist."""


def _env(**handoff: object) -> ManagerBasedRlEnv:
  cfg = booster_k1_block_moving_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 3
  action = cfg.actions["joint_pos"]
  assert isinstance(action, WalkHandoffActionCfg)
  for name, value in handoff.items():
    setattr(action, name, value)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset(seed=3)
  return env


def _group(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  value = env.observation_manager.compute()[name]
  assert isinstance(value, torch.Tensor)
  return value


def _handoff(env: ManagerBasedRlEnv) -> WalkHandoffAction:
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, WalkHandoffAction)
  return term


def _shot(env: ManagerBasedRlEnv) -> ShotCommand:
  term = env.command_manager.get_term("shot")
  assert isinstance(term, ShotCommand)
  return term


@pytest.fixture(scope="module")
def walking_env():
  """Every shot cycle walks, handed over only after the kick and with feet down."""
  env = _env(walk_probability=1.0, guard_probability=0.0, gated_probability=1.0)
  yield env
  env.close()


def test_block_contract_and_walk_window_sizes(walking_env: ManagerBasedRlEnv) -> None:
  assert _group(walking_env, "actor").shape[-1] == ACTOR_DIM
  assert _group(walking_env, "history").shape[-2:] == (HISTORY_WINDOW, ACTOR_DIM)
  walk = _group(walking_env, "walk_history")
  assert walk.shape[-2:] == (HISTORY_WINDOW, WALK_DIM)


def test_actor_terms_are_run_12s() -> None:
  """Same terms in the same order as the task run 12 was trained on."""
  moving = booster_k1_block_moving_env_cfg()
  standing = booster_k1_block_env_cfg()
  for group in ("actor", "history", "critic"):
    assert list(moving.observations[group].terms) == list(
      standing.observations[group].terms
    )


def test_walk_drives_the_goalie_until_the_handoff(
  walking_env: ManagerBasedRlEnv,
) -> None:
  """The learner's actions are ignored while walking; the walk's are applied and
  written back as the step's action, which the block policy observes."""
  env, term, shot = walking_env, _handoff(walking_env), _shot(walking_env)
  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  env.reset(seed=4)
  twist.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0], device=env.device)
  start = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
  zeros = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )

  walked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for _ in range(40):  # 0.8 s, before any kick (kick delays start at 0.5 s + extra).
    walking = term.walking.clone()
    env.step(zeros)
    if bool(walking.any()):
      assert not torch.allclose(env.action_manager.action[walking], zeros[walking])
      # An inactive block command while walking, as K1BlockPolicy records it.
      assert torch.all(_group(env, "actor")[term.walking, -4:] == 0.0)
    walked |= walking
  assert bool(walked.all())
  moved = env.scene["robot"].data.root_link_pos_w[:, :2] - start
  assert bool((moved[:, 0] > 0.1).any()), "the walk should carry the goalie forward"
  assert not bool(env.termination_manager.terminated.any())
  del shot


def test_handoff_follows_the_kick(walking_env: ManagerBasedRlEnv) -> None:
  """BLOCK hand-offs come after the kick, within the reaction time and the wait for
  both feet down; until then the learner is not paid."""
  env, term, shot = walking_env, _handoff(walking_env), _shot(walking_env)
  cfg = term.cfg
  env.reset(seed=5)
  zeros = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  latest = cfg.reaction_time[1] + cfg.max_wait + 2 * env.step_dt
  handed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for _ in range(200):
    walking = term.walking.clone()
    _, reward, *_ = env.step(zeros)
    # Hand-offs happen as the step's action is processed, so this step's rewards
    # go with whether the walk still had the goalie after it.
    assert torch.all(reward[term.walking] == 0.0)
    now_handed = walking & ~term.walking
    if bool(now_handed.any()):
      since_kick = shot.time_since_resample - shot.kick_delay
      assert bool(shot.kicked[now_handed].all())
      assert bool((since_kick[now_handed] >= cfg.reaction_time[0] - 1e-6).all())
      assert bool((since_kick[now_handed] <= latest).all())
    handed |= now_handed
    if bool(handed.all()):
      break
  assert bool(handed.all()), "every walking goalie should be handed over"


def test_standing_cycles_never_walk() -> None:
  env = _env(walk_probability=0.0)
  term = _handoff(env)
  zeros = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  for _ in range(20):
    env.step(zeros)
    assert not bool(term.walking.any())
  env.close()
