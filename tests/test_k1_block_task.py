"""Tests for the Booster K1 goalkeeper (block policy) task configuration."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.obs_history import HistoryActor, HistoryModelCfg, OnnxHistoryPolicy
from mjlab.tasks.goalkeeper.config.booster_k1.env_cfgs import (
  booster_k1_block_env_cfg,
)
from mjlab.tasks.goalkeeper.config.booster_k1.rl_cfg import (
  booster_k1_block_ppo_runner_cfg,
)
from mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
  BALL_RADIUS,
  ROLLING_DECELERATION,
)
from mjlab.tasks.goalkeeper.mdp import ShotCommand, ShotCommandCfg
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import HISTORY_WINDOW

# ang vel (3) + gravity (3) + joint pos/vel/actions (3 x 20) + command (4).
# This is the observation skill::K1BlockPolicy builds on the robot.
ACTOR_DIM = 70


@pytest.fixture(scope="module")
def block_env():
  cfg = booster_k1_block_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset(seed=1)
  yield env
  env.close()


def test_actor_terms_and_order() -> None:
  """The actor observation is the deployment contract; order and size are fixed."""
  cfg = booster_k1_block_env_cfg()
  assert list(cfg.observations["actor"].terms) == [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  ]


def test_actor_observation_size(block_env: ManagerBasedRlEnv) -> None:
  obs = block_env.observation_manager.compute()
  assert obs["actor"].shape[-1] == ACTOR_DIM
  assert obs["history"].shape[-2:] == (HISTORY_WINDOW, ACTOR_DIM)


def test_history_group_clones_actor_terms() -> None:
  cfg = booster_k1_block_env_cfg()
  history = cfg.observations["history"]
  assert history.history_length == HISTORY_WINDOW
  assert history.flatten_history_dim is False
  assert list(history.terms) == list(cfg.observations["actor"].terms)


def test_policy_excludes_head(block_env: ManagerBasedRlEnv) -> None:
  action_term = block_env.action_manager.get_term("joint_pos")
  assert isinstance(action_term, JointPositionAction)
  assert not any("Head" in name for name in action_term.target_names)
  assert len(action_term.target_names) == 20


def test_command_is_the_block_contract(block_env: ManagerBasedRlEnv) -> None:
  """[active, dy, t, v], clipped as message::skill::Block is."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  command = shot.command
  assert command.shape == (block_env.num_envs, 4)
  assert torch.all((command[:, 0] == 0.0) | (command[:, 0] == 1.0))
  assert torch.all(command[:, 1].abs() <= shot.cfg.max_dy)
  assert torch.all((command[:, 2] >= 0.0) & (command[:, 2] <= shot.cfg.max_time))
  assert torch.all((command[:, 3] >= 0.0) & (command[:, 3] <= shot.cfg.max_speed))


def test_shot_is_kicked_and_reaches_the_goalie(block_env: ManagerBasedRlEnv) -> None:
  """A shot is launched, rolls at the goalie, and the command wakes up for it."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  block_env.reset(seed=3)
  action = torch.zeros(
    block_env.num_envs,
    block_env.action_manager.total_action_dim,
    device=block_env.device,
  )

  ever_active = torch.zeros(
    block_env.num_envs, dtype=torch.bool, device=block_env.device
  )
  closest = torch.full((block_env.num_envs,), 1e9, device=block_env.device)
  for _ in range(200):
    block_env.step(action)
    ever_active |= shot.command[:, 0] > 0.5
    ball_r, _ = shot.to_robot_frame(
      shot.ball.data.root_link_pos_w, shot.ball.data.root_link_lin_vel_w[:, :2]
    )
    closest = torch.minimum(closest, ball_r.norm(dim=-1))

  assert bool(shot.kicked.any())
  assert bool(ever_active.all()), "the block command never activated for a live shot"
  # Shots are aimed at the goalie's line, so every ball should get near it.
  assert float(closest.max()) < 1.5


def test_ball_rolls_at_the_configured_deceleration(
  block_env: ManagerBasedRlEnv,
) -> None:
  """The sim's rolling resistance must match what the shot command predicts with."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  block_env.reset(seed=5)
  action = torch.zeros(
    block_env.num_envs,
    block_env.action_manager.total_action_dim,
    device=block_env.device,
  )

  # One ball at a time: a new shot begins whenever the shot timer restarts, which can
  # happen on a step where the ball is not rolling yet, so the shots are counted
  # separately from the samples.
  runs: dict[int, list[float]] = {}
  shot_index = 0
  previous_timer = float("inf")
  for _ in range(300):
    block_env.step(action)
    timer = float(shot.time_since_resample[0])
    if timer < previous_timer:
      shot_index += 1
    previous_timer = timer
    # was_moving skips the kick step itself, where the ball's new velocity has been
    # written to sim but does not show in its data yet.
    if bool(shot.was_moving[0] and not shot.touched[0] and not shot.finished[0]):
      speed = float(shot.ball.data.root_link_lin_vel_w[0, :2].norm())
      runs.setdefault(shot_index, []).append(speed)

  longest = max(runs.values(), key=len)
  assert len(longest) > 20, "no shot rolled freely for long enough to measure"
  deceleration = (longest[0] - longest[-1]) / (len(longest) * block_env.step_dt)
  assert 0.5 * ROLLING_DECELERATION < deceleration < 2.0 * ROLLING_DECELERATION


def test_arms_are_held_in_the_ready_stance() -> None:
  """v0 blocks with the body and feet; the arm posture term keeps arms out of it."""
  cfg = booster_k1_block_env_cfg()
  arm_cfg = cfg.rewards["arm_posture"].params["asset_cfg"]
  assert cfg.rewards["arm_posture"].weight > 0
  assert arm_cfg.joint_names is not None
  assert "Shoulder" in arm_cfg.joint_names[0]


def test_ball_is_a_size_three_ball() -> None:
  cfg = booster_k1_block_env_cfg()
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  assert shot.ball_radius == BALL_RADIUS
  assert shot.rolling_deceleration == ROLLING_DECELERATION


def test_play_disables_corruption_and_pushes() -> None:
  cfg = booster_k1_block_env_cfg(play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.observations["history"].enable_corruption is False
  assert "push_robot" not in cfg.events


def test_exports_the_shape_the_robot_loads(block_env: ManagerBasedRlEnv) -> None:
  """skill::K1BlockPolicy loads obs [1, 25 x 70] in and actions [1, 20] out."""
  rl_cfg = booster_k1_block_ppo_runner_cfg()
  actor_cfg = rl_cfg.actor
  assert isinstance(actor_cfg, HistoryModelCfg)
  obs = block_env.observation_manager.compute()
  num_actions = block_env.action_manager.total_action_dim
  actor = HistoryActor(
    obs,
    {k: list(v) for k, v in rl_cfg.obs_groups.items()},
    "actor",
    num_actions,
    history_cfg=dict(actor_cfg.history_cfg),
    hidden_dims=actor_cfg.hidden_dims,
    activation=actor_cfg.activation,
    obs_normalization=actor_cfg.obs_normalization,
    distribution_cfg=dict(actor_cfg.distribution_cfg or {}),
  ).to(block_env.device)
  assert actor(obs).shape == (block_env.num_envs, num_actions)
  onnx_policy = actor.as_onnx(verbose=False)
  assert isinstance(onnx_policy, OnnxHistoryPolicy)
  assert onnx_policy.input_size == HISTORY_WINDOW * ACTOR_DIM
  assert num_actions == 20
