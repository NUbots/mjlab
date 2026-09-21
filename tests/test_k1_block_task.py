"""Tests for the Booster K1 goalkeeper (block policy) task configuration."""

import math

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.obs_history import HistoryActor, HistoryModelCfg, OnnxHistoryPolicy
from mjlab.tasks.goalkeeper import mdp
from mjlab.tasks.goalkeeper.config.booster_k1.env_cfgs import (
  SHOT_LEVELS,
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


def test_shot_levels_only_get_harder() -> None:
  """Each drill must be at least as hard as the one before, and training start on L1."""
  for previous, current in zip(SHOT_LEVELS, SHOT_LEVELS[1:], strict=False):
    assert current["crossing"][1] >= previous["crossing"][1]
    assert current["speed"][1] >= previous["speed"][1]

  cfg = booster_k1_block_env_cfg()
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  assert shot.levels == SHOT_LEVELS
  assert shot.start_level == 0
  assert 0.0 < shot.mix_fraction < 1.0, "earlier drills must keep being served"
  assert "shot_levels" in cfg.curriculum


def test_level_advances_on_save_rate_not_on_steps(block_env: ManagerBasedRlEnv) -> None:
  """A level is left behind on evidence: enough shots, saved often enough."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  start = shot.level

  # Saving well, but too few shots to say so.
  shot.recent_save_rate.fill_(0.9)
  shot.shots_since_level = 10
  mdp.shot_levels(block_env, torch.arange(1), "shot", advance_at=0.6, min_shots=400)
  assert shot.level == start

  # Plenty of shots, but not saving them.
  shot.recent_save_rate.fill_(0.2)
  shot.shots_since_level = 1000
  mdp.shot_levels(block_env, torch.arange(1), "shot", advance_at=0.6, min_shots=400)
  assert shot.level == start

  # Both, so it moves up, and the evidence resets for the new level.
  shot.recent_save_rate.fill_(0.9)
  shot.shots_since_level = 1000
  mdp.shot_levels(block_env, torch.arange(1), "shot", advance_at=0.6, min_shots=400)
  assert shot.level == start + 1
  assert shot.shots_since_level == 0
  assert float(shot.recent_save_rate) == 0.0


def test_play_faces_the_full_envelope() -> None:
  """Measuring an envelope against the drill it is on would flatter it."""
  cfg = booster_k1_block_env_cfg(play=True)
  shot = cfg.commands["shot"]
  assert isinstance(shot, ShotCommandCfg)
  assert shot.start_level == len(SHOT_LEVELS) - 1
  assert shot.mix_fraction == 0.0
  assert "shot_levels" not in cfg.curriculum


def test_the_ball_can_hit_more_than_the_feet(block_env: ManagerBasedRlEnv) -> None:
  """A keeper whose shins are not collidable cannot block with them.

  The stock K1 enables only *foot_collision, so a ball passes through the legs and
  body. Goalkeeping needs the rest of the robot to be solid.
  """
  model = block_env.sim.mj_model
  enabled = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    for i in range(model.ngeom)
    if model.geom_contype[i] or model.geom_conaffinity[i]
  }
  for part in ("left_shin_collision", "right_shin_collision", "left_knee_collision"):
    assert f"robot/{part}" in enabled, f"{part} cannot be hit by the ball"
  assert "robot/left_foot_collision" in enabled


def test_upright_bonus_eases_off_as_drills_are_cleared() -> None:
  """It is scaffolding for the early collapse, not a requirement of the job."""
  cfg = booster_k1_block_env_cfg()
  weights = cfg.curriculum["upright_relaxation"].params["weights"]
  assert weights[0] == cfg.rewards["upright"].weight, "level 1 keeps the full bonus"
  assert list(weights) == sorted(weights, reverse=True), "it must only loosen"
  assert weights[-1] > 0.0, "a keeper still should not dive onto its face"
  assert len(weights) == len(SHOT_LEVELS)


def test_leaning_is_free_but_toppling_is_not() -> None:
  """The dead zone must leave room to lean, and stay inside the fall termination."""
  cfg = booster_k1_block_env_cfg()
  dead_zone = cfg.rewards["upright"].params["dead_zone_deg"]
  fall_angle = math.degrees(cfg.terminations["fell_over"].params["limit_angle"])
  assert 10.0 < dead_zone < fall_angle - 10.0, (
    "a keeper paid full value right up to the angle it is terminated at has no "
    "gradient left to catch itself"
  )


def test_rest_point_is_where_a_cleared_ball_stops(
  block_env: ManagerBasedRlEnv,
) -> None:
  """The clearance is judged on a projected rest point; it has to be the real one."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  block_env.reset(seed=7)
  action = torch.zeros(
    block_env.num_envs,
    block_env.action_manager.total_action_dim,
    device=block_env.device,
  )
  # Roll the ball up the field, well clear of the goalie, and keep the shot timer
  # from replacing it.
  shot.time_left[:] = 100.0
  shot.kicked[:] = True
  n = block_env.num_envs
  origins = block_env.scene.env_origins
  pose = torch.zeros(n, 7, device=block_env.device)
  pose[:, 0] = origins[:, 0] + 0.3
  pose[:, 1] = origins[:, 1] + 2.0
  pose[:, 2] = BALL_RADIUS
  pose[:, 3] = 1.0
  speed = 1.5
  velocity = torch.zeros(n, 6, device=block_env.device)
  velocity[:, 0] = speed
  velocity[:, 4] = speed / BALL_RADIUS
  shot.ball.write_root_link_pose_to_sim(pose, env_ids=torch.arange(n))
  shot.ball.write_root_link_velocity_to_sim(velocity, env_ids=torch.arange(n))

  # A ball slowing at the rate the projection assumes keeps the same rest point all
  # the way in. (Not run to a stop: an unpowered goalie falls and resets first.)
  expected = 0.3 + speed**2 / (2.0 * ROLLING_DECELERATION)
  for _ in range(50):
    block_env.step(action)
    assert torch.allclose(shot.rest_x, torch.full_like(shot.rest_x, expected), atol=0.3)
  assert float(shot.ball.data.root_link_lin_vel_w[:, 0].max()) < 0.8 * speed


def test_clearance_counts_the_follow_through(block_env: ManagerBasedRlEnv) -> None:
  """A strike still speeding the ball up after the save is decided is paid for."""
  shot = block_env.command_manager.get_term("shot")
  assert isinstance(shot, ShotCommand)
  block_env.reset(seed=11)
  action = torch.zeros(
    block_env.num_envs,
    block_env.action_manager.total_action_dim,
    device=block_env.device,
  )
  n = block_env.num_envs
  ids = torch.arange(n)
  origins = block_env.scene.env_origins

  def roll(speed: float) -> None:
    pose = torch.zeros(n, 7, device=block_env.device)
    pose[:, :2] = shot.ball.data.root_link_pos_w[:, :2]
    pose[:, 2] = BALL_RADIUS
    pose[:, 3] = 1.0
    velocity = torch.zeros(n, 6, device=block_env.device)
    velocity[:, 0] = speed
    velocity[:, 4] = speed / BALL_RADIUS
    shot.ball.write_root_link_pose_to_sim(pose, env_ids=ids)
    shot.ball.write_root_link_velocity_to_sim(velocity, env_ids=ids)

  # A shot that was going in, now rolling back out slowly: a save that barely clears.
  shot.time_left[:] = 100.0
  shot.kicked[:] = True
  shot.was_moving[:] = True
  shot.on_target_shot[:] = True
  pose = torch.zeros(n, 7, device=block_env.device)
  pose[:, 0] = origins[:, 0] + 0.3
  pose[:, 1] = origins[:, 1] + 2.0
  pose[:, 2] = BALL_RADIUS
  pose[:, 3] = 1.0
  shot.ball.write_root_link_pose_to_sim(pose, env_ids=ids)
  roll(0.8)

  paid = torch.zeros(n, device=block_env.device)
  steps = int(round(shot.cfg.clear_window / block_env.step_dt)) + 5
  for step in range(steps):
    block_env.step(action)
    if step == 0:
      assert bool(shot.saved_now.all())
    if step == 5:
      roll(2.0)  # The follow-through of the strike.
    paid += shot.cleared_now

  # Paid for the boosted ball, not the one first seen leaving, and paid once.
  assert bool((shot.best_rest > 0.3 + 1.5**2 / (2.0 * ROLLING_DECELERATION)).all())
  expected = shot.best_rest.clamp(max=shot.cfg.clear_distance) / shot.cfg.clear_distance
  assert torch.allclose(paid, expected, atol=1e-4)
  assert float(shot.since_save.min()) > shot.cfg.clear_window
