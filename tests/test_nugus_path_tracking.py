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


TWIST_LIMITS = (0.5, 0.3, 1.0)


def make_env_and_term(
  rel_standing_envs: float = 0.0,
  segment_mix: PathCommandCfg.SegmentMix | None = None,
  max_travel_angle: float = math.pi,
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
    twist_limits=TWIST_LIMITS,
    segment_mix=segment_mix or PathCommandCfg.SegmentMix(),
    max_travel_angle=max_travel_angle,
    ranges=PathCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.5),
      lin_vel_y=(-0.3, 0.3),
      ang_vel_z=(-1.0, 1.0),
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


def test_reference_twist_within_feasibility_ellipsoid(seeded_term) -> None:
  """Every reference twist stays on or inside the capability ellipsoid."""
  _, _, term = seeded_term
  limits = torch.tensor(TWIST_LIMITS)
  norms = torch.linalg.vector_norm(term.ref_twist_b / limits, dim=-1)
  assert norms.max() <= 1.0 + 1e-5


def test_reference_twist_ramps_smoothly(seeded_term) -> None:
  """The twist ramps from zero on reset and never steps discontinuously."""
  _, _, term = seeded_term
  # Episodes start from rest: the first reference twist is zero.
  assert term.ref_twist_b[:, 0].abs().max() < 1e-6
  # Per-step change is bounded by the smoothstep's peak slope (1.5 / blend
  # time) over the largest possible segment-to-segment jump (twice the
  # per-axis limit).
  limits = torch.tensor(TWIST_LIMITS)
  diffs = (term.ref_twist_b[:, 1:] - term.ref_twist_b[:, :-1]).abs()
  bound = 2.0 * limits * 1.5 * STEP_DT / term.cfg.twist_blend_time
  assert (diffs <= bound + 1e-5).all()


def test_segment_mix_masks_twist_components() -> None:
  """A straight-only mix produces forward/backward motion and nothing else."""
  torch.manual_seed(2)
  mix = PathCommandCfg.SegmentMix(
    standing=0.0, straight=1.0, arc=0.0, turn_in_place=0.0, strafe=0.0, omni=0.0
  )
  _, _, term = make_env_and_term(segment_mix=mix)
  term.reset(torch.arange(NUM_ENVS))
  term.compute(0.0)
  assert term.ref_twist_b[..., 1].abs().max() < 1e-6  # No vy.
  assert term.ref_twist_b[..., 2].abs().max() < 1e-6  # No wz.
  assert term.ref_twist_b[..., 0].abs().max() > 0.05  # Walks somewhere.


def test_travel_direction_cone_limits_bearing() -> None:
  """With a cone, every moving reference twist travels roughly forward."""
  torch.manual_seed(3)
  cone = math.radians(30.0)
  _, robot, term = make_env_and_term(max_travel_angle=cone)
  robot.data._heading = (torch.rand(NUM_ENVS) - 0.5) * 2 * math.pi
  term.reset(torch.arange(NUM_ENVS))
  term.compute(0.0)

  v = term.ref_twist_b[..., :2]
  speed = torch.linalg.vector_norm(v, dim=-1)
  moving = speed > 1e-3
  assert moving.any()
  bearing = torch.atan2(v[..., 1], v[..., 0])
  # Blended steps interpolate between in-cone twists; for a half-angle
  # below 90 degrees the cone is convex, so blends stay inside it too.
  assert bearing[moving].abs().max() <= cone + 1e-4
  # The cone rotates twists rather than zeroing them: lateral motion
  # within the cone survives.
  assert v[..., 1].abs().max() > 1e-3


def test_travel_direction_cone_rejects_invalid_angle() -> None:
  with pytest.raises(ValueError, match="max_travel_angle"):
    PathCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      twist_limits=TWIST_LIMITS,
      max_travel_angle=-0.1,
      ranges=PathCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-1.0, 1.0),
      ),
    )


def _make_term(
  *,
  segment_mix: PathCommandCfg.SegmentMix,
  min_turn_radius: float,
) -> PathCommand:
  """Build a term whose reference twist equals its per-segment twist.

  A zero blend time and a single long segment mean ``ref_twist_b`` is the
  raw sampled (and radius-capped) segment twist, so per-segment constraints
  can be asserted directly.
  """
  robot = FakeRobot(NUM_ENVS)
  env = types.SimpleNamespace(
    num_envs=NUM_ENVS, device=DEVICE, step_dt=STEP_DT, scene={"robot": robot}
  )
  cfg = PathCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    twist_limits=TWIST_LIMITS,
    twist_blend_time=0.0,
    segment_duration_range=(20.0, 20.0),
    min_turn_radius=min_turn_radius,
    segment_mix=segment_mix,
    ranges=PathCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.5),
      lin_vel_y=(-0.3, 0.3),
      ang_vel_z=(-1.0, 1.0),
    ),
  )
  term = cfg.build(env)  # type: ignore[arg-type]
  term.reset(torch.arange(NUM_ENVS))
  term.compute(0.0)
  return term


def test_min_turn_radius_widens_arcs() -> None:
  """Curved segments respect the minimum turning radius (radius >= min)."""
  torch.manual_seed(4)
  radius = 0.75
  term = _make_term(
    segment_mix=PathCommandCfg.SegmentMix(
      standing=0.0, straight=0.0, arc=1.0, turn_in_place=0.0, strafe=0.0, omni=0.0
    ),
    min_turn_radius=radius,
  )
  twist = term.ref_twist_b
  speed = torch.linalg.vector_norm(twist[..., :2], dim=-1)
  wz = twist[..., 2].abs()
  moving = speed > 1e-3
  assert moving.any()
  # radius = speed / |wz| >= min_turn_radius, i.e. |wz| <= speed / radius.
  assert (wz[moving] <= speed[moving] / radius + 1e-4).all()


def test_min_turn_radius_exempts_in_place_turns() -> None:
  """Pure in-place turns keep the full yaw range despite a radius cap."""
  torch.manual_seed(5)
  term = _make_term(
    segment_mix=PathCommandCfg.SegmentMix(
      standing=0.0, straight=0.0, arc=0.0, turn_in_place=1.0, strafe=0.0, omni=0.0
    ),
    min_turn_radius=0.75,
  )
  twist = term.ref_twist_b
  assert twist[..., :2].abs().max() < 1e-6  # No linear motion.
  assert twist[..., 2].abs().max() > 0.1  # Still spins.


def test_min_turn_radius_rejects_negative() -> None:
  with pytest.raises(ValueError, match="min_turn_radius"):
    PathCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      twist_limits=TWIST_LIMITS,
      min_turn_radius=-0.1,
      ranges=PathCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-1.0, 1.0),
      ),
    )


def test_path_tracking_rewards_decay_with_lag(seeded_term) -> None:
  """Path position/heading rewards saturate on the path and decay off it."""
  env, _, term = seeded_term
  env.command_manager = types.SimpleNamespace(get_term=lambda name: term)

  term.pos_error = torch.zeros(NUM_ENVS)
  term.heading_error = torch.zeros(NUM_ENVS)
  pos_on = mdp.track_path_position(env, std=0.3, command_name="path")
  head_on = mdp.track_path_heading(env, std=0.5, command_name="path")
  assert torch.allclose(pos_on, torch.ones(NUM_ENVS), atol=1e-5)
  assert torch.allclose(head_on, torch.ones(NUM_ENVS), atol=1e-5)

  term.pos_error = torch.full((NUM_ENVS,), 0.6)
  term.heading_error = torch.full((NUM_ENVS,), 1.0)
  assert (mdp.track_path_position(env, std=0.3, command_name="path") < pos_on).all()
  assert (mdp.track_path_heading(env, std=0.5, command_name="path") < head_on).all()


def test_resample_restarts_path_from_current_pose(seeded_term) -> None:
  _, robot, term = seeded_term
  all_ids = torch.arange(NUM_ENVS)
  robot.data.root_link_pos_w[:, 0] += 3.0  # Teleport, as if replanned.
  term._resample(all_ids)
  term.compute(0.0)
  pos_err = torch.norm(term.ref_pos_w[:, 0] - robot.data.root_link_pos_w[:, :2], dim=-1)
  assert pos_err.max() < 1e-5
  assert (term.path_step == 0).all()
