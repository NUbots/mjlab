"""Tests for the NUgus path tracking task.

Group A pins the deployed actor observation interface for the path
tracking config (sim-to-real critical, mirrors
``test_nugus_observation_vector.py``). Group B exercises ``PathCommand``
behavior directly against a fake env, since the full MuJoCo sim cannot be
built in this environment.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from mjlab.envs.mdp.observations import generated_commands
from mjlab.tasks.path_tracking import mdp
from mjlab.tasks.path_tracking.config.nugus.env_cfgs import (
  nubots_nugus_path_flat_env_cfg,
)
from mjlab.tasks.path_tracking.mdp import PathCommand, PathCommandCfg

# -----------------------------------------------------------------------
# Group A: config-level tests (no env construction).
# -----------------------------------------------------------------------

EXPECTED_ACTOR_ORDER = [
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
  "gait_clock",
]

N_MOTOR_JOINTS = 20
N_LOOKAHEAD = 4
COMMAND_DIM = N_LOOKAHEAD * 4  # (dx, dy, cos, sin) per waypoint.


@pytest.fixture(scope="module")
def path_cfg():
  return nubots_nugus_path_flat_env_cfg()


@pytest.fixture(scope="module")
def actor_terms(path_cfg) -> dict:
  return {
    name: term
    for name, term in path_cfg.observations["actor"].terms.items()
    if term is not None
  }


def test_actor_term_order(actor_terms: dict) -> None:
  assert list(actor_terms.keys()) == EXPECTED_ACTOR_ORDER


def test_actor_command_uses_path_waypoints(actor_terms: dict) -> None:
  command_term = actor_terms["command"]
  assert command_term.func is mdp.path_waypoints
  assert command_term.params == {"command_name": "path"}


def test_base_lin_vel_absent_from_actor(actor_terms: dict) -> None:
  assert "base_lin_vel" not in actor_terms


def test_commands_dict_has_path_not_twist(path_cfg) -> None:
  assert "path" in path_cfg.commands
  assert "twist" not in path_cfg.commands
  assert isinstance(path_cfg.commands["path"], PathCommandCfg)


def test_no_lingering_twist_command_name(path_cfg) -> None:
  for reward in path_cfg.rewards.values():
    assert reward.params.get("command_name") != "twist"
  for group in path_cfg.observations.values():
    for term in group.terms.values():
      if term is None:
        continue
      assert term.params.get("command_name") != "twist"
  for curriculum in path_cfg.curriculum.values():
    assert curriculum.params.get("command_name") != "twist"


def test_actor_observation_vector_slices(actor_terms: dict) -> None:
  """Pin the actor observation layout for the path tracking task.

  With 20 motor joints and 4 lookahead waypoints (16-dim command):
    base_ang_vel       [0:3]
    projected_gravity  [3:6]
    joint_pos          [6:26]
    joint_vel          [26:46]
    actions            [46:66]
    command            [66:82]
    gait_clock         [82:84]
  """
  n = N_MOTOR_JOINTS
  term_dims = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": n,
    "joint_vel": n,
    "actions": n,
    "command": COMMAND_DIM,
    "gait_clock": 2,
  }

  offset = 0
  slices = {}
  for name in actor_terms:
    dim = term_dims[name]
    slices[name] = slice(offset, offset + dim)
    offset += dim

  assert slices["base_ang_vel"] == slice(0, 3)
  assert slices["projected_gravity"] == slice(3, 6)
  assert slices["joint_pos"] == slice(6, 6 + n)
  assert slices["joint_vel"] == slice(6 + n, 6 + 2 * n)
  assert slices["actions"] == slice(6 + 2 * n, 6 + 3 * n)
  assert slices["command"] == slice(6 + 3 * n, 6 + 3 * n + COMMAND_DIM)
  assert slices["gait_clock"] == slice(
    6 + 3 * n + COMMAND_DIM, 6 + 3 * n + COMMAND_DIM + 2
  )
  assert offset == 84


def test_command_dim_matches_lookahead_times(path_cfg) -> None:
  path_command_cfg = path_cfg.commands["path"]
  assert len(path_command_cfg.lookahead_times) * 4 == COMMAND_DIM


def test_critic_has_command_and_target_twist(path_cfg) -> None:
  critic_terms = {
    name: term
    for name, term in path_cfg.observations["critic"].terms.items()
    if term is not None
  }
  assert "command" in critic_terms
  assert critic_terms["command"].func is mdp.path_waypoints
  assert "target_twist" in critic_terms
  assert critic_terms["target_twist"].func is generated_commands
  assert critic_terms["target_twist"].params == {"command_name": "path"}


def test_path_tracking_tasks_registered() -> None:
  import mjlab.tasks
  from mjlab.tasks.registry import list_tasks

  del mjlab.tasks  # Imported only for its registration side effect.

  tasks = list_tasks()
  assert "Mjlab-PathTracking-Flat-Nubots-Nugus" in tasks
  assert "Mjlab-PathTracking-Rough-Nubots-Nugus" in tasks


# -----------------------------------------------------------------------
# Group B: PathCommand behavior with a fake env.
# -----------------------------------------------------------------------

DEVICE = "cpu"
NUM_ENVS = 16
STEP_DT = 0.02


class FakeRobotData:
  def __init__(self, num_envs: int):
    self.root_link_pos_w = torch.zeros(num_envs, 3)
    self._heading = torch.zeros(num_envs)

  @property
  def heading_w(self) -> torch.Tensor:
    return self._heading


class FakeRobot:
  def __init__(self, num_envs: int):
    self.data = FakeRobotData(num_envs)


def make_env_and_term(
  rel_standing_envs: float = 0.0,
) -> tuple[types.SimpleNamespace, FakeRobot, PathCommand]:
  robot = FakeRobot(NUM_ENVS)
  env = types.SimpleNamespace(
    num_envs=NUM_ENVS,
    device=DEVICE,
    step_dt=STEP_DT,
    scene={"robot": robot},
  )
  cfg = PathCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    rel_standing_envs=rel_standing_envs,
    rel_standing_segments=0.1,
    ranges=PathCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
  )
  term = cfg.build(env)  # type: ignore[arg-type]
  return env, robot, term


def integrate_robot(robot: FakeRobot, cmd: torch.Tensor, dt: float) -> None:
  """Advance the fake robot exactly along the commanded body twist."""
  h = robot.data._heading
  vx, vy, wz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
  robot.data.root_link_pos_w[:, 0] += (vx * torch.cos(h) - vy * torch.sin(h)) * dt
  robot.data.root_link_pos_w[:, 1] += (vx * torch.sin(h) + vy * torch.cos(h)) * dt
  robot.data._heading = h + wz * dt


@pytest.fixture
def seeded_term() -> tuple[types.SimpleNamespace, FakeRobot, PathCommand]:
  torch.manual_seed(0)
  env, robot, term = make_env_and_term()
  robot.data.root_link_pos_w[:, :2] = torch.randn(NUM_ENVS, 2) * 2.0
  robot.data._heading = (torch.rand(NUM_ENVS) - 0.5) * 2 * math.pi
  all_ids = torch.arange(NUM_ENVS)
  term.reset(all_ids)
  # Anchor the freshly sampled path at the robot's current pose without
  # advancing along it (mirrors the env's post-reset compute(dt=0.0)).
  term.compute(0.0)
  return env, robot, term


def test_path_starts_at_robot_pose(seeded_term) -> None:
  _, robot, term = seeded_term
  pos_err = torch.norm(term.ref_pos_w[:, 0] - robot.data.root_link_pos_w[:, :2], dim=-1)
  assert pos_err.max() < 1e-5
  heading_err = (term.ref_heading_w[:, 0] - robot.data.heading_w).abs()
  assert heading_err.max() < 1e-5


def test_perfect_follower_stays_on_path(seeded_term) -> None:
  """A robot integrated exactly along the command stays close to the path."""
  _, robot, term = seeded_term
  max_pos_err = 0.0
  for _ in range(int(4.0 / STEP_DT)):
    cmd = term.command.clone()
    integrate_robot(robot, cmd, STEP_DT)
    term.compute(STEP_DT)
    batch = torch.arange(NUM_ENVS)
    ref = term.ref_pos_w[batch, term.path_step]
    err = torch.norm(ref - robot.data.root_link_pos_w[:, :2], dim=-1).max().item()
    max_pos_err = max(max_pos_err, err)
  assert max_pos_err < 0.05


def test_command_stays_within_ranges(seeded_term) -> None:
  _, robot, term = seeded_term
  ranges = term.cfg.ranges
  for _ in range(int(2.0 / STEP_DT)):
    cmd = term.command.clone()
    integrate_robot(robot, cmd, STEP_DT)
    term.compute(STEP_DT)
    assert term.command[:, 0].abs().max() <= ranges.lin_vel_x[1] + 1e-6
    assert term.command[:, 1].abs().max() <= ranges.lin_vel_y[1] + 1e-6
    assert term.command[:, 2].abs().max() <= ranges.ang_vel_z[1] + 1e-6


def test_waypoints_shape_and_unit_norm(seeded_term) -> None:
  _, _, term = seeded_term
  wp = term.waypoints_b
  assert wp.shape == (NUM_ENVS, N_LOOKAHEAD, 4)
  norm = torch.norm(wp[..., 2:4], dim=-1)
  assert torch.allclose(norm, torch.ones(NUM_ENVS, N_LOOKAHEAD), atol=1e-5)

  flattened = term.waypoints
  assert flattened.shape == (NUM_ENVS, N_LOOKAHEAD * 4)
  assert torch.allclose(flattened, wp.flatten(start_dim=1))


def test_standing_envs_give_zero_command() -> None:
  torch.manual_seed(1)
  _, robot, term = make_env_and_term(rel_standing_envs=1.0)
  robot.data.root_link_pos_w[:, :2] = torch.randn(NUM_ENVS, 2)
  all_ids = torch.arange(NUM_ENVS)
  term.reset(all_ids)
  term.compute(0.0)

  assert term.command.abs().max() < 1e-6
  assert term.waypoints_b[..., :2].abs().max() < 1e-6  # dx, dy = 0.
  assert (term.waypoints_b[..., 2] - 1.0).abs().max() < 1e-6  # cos(0) = 1.


def test_resample_restarts_path_from_current_pose(seeded_term) -> None:
  _, robot, term = seeded_term
  all_ids = torch.arange(NUM_ENVS)
  robot.data.root_link_pos_w[:, 0] += 3.0  # Teleport, as if replanned.
  term._resample(all_ids)
  term.compute(0.0)
  pos_err = torch.norm(term.ref_pos_w[:, 0] - robot.data.root_link_pos_w[:, :2], dim=-1)
  assert pos_err.max() < 1e-5
  assert (term.path_step == 0).all()
