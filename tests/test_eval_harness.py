"""Tests for the batched evaluation harness.

Marked slow: each test compiles a model and warms up warp kernels, which costs
about forty seconds on a cold CPU cache. The metrics themselves are covered
without a simulator in ``test_walk_metrics.py``.
"""

from dataclasses import replace

import pytest
import torch
from conftest import get_test_device

from mjlab.controllers.quintic_walk.controller import JOINT_NAMES
from mjlab.controllers.quintic_walk.walk_generator import EngineState
from mjlab.evaluation.harness import (
  DistilledEvalHarness,
  QuinticEvalHarness,
  command_grid,
  constant_command,
  eval_scene_cfg,
)
from mjlab.evaluation.metrics import summarise
from mjlab.evaluation.push import PushCfg, push_plan, run_push_battery


def test_command_grid_tiles_every_point_across_the_batch():
  grid = command_grid(vx=(0.1, 0.2), vy=(0.0,), wz=(-0.3, 0.3), num_envs=8)

  assert grid.shape == (8, 3)
  assert torch.unique(grid, dim=0).shape[0] == 4
  # Truncation keeps the batch the size that was asked for.
  assert command_grid((0.1, 0.2, 0.3), (0.0,), (0.0,), num_envs=2).shape == (2, 3)


def test_constant_command_is_the_same_everywhere():
  command = constant_command(0.3, -0.1, 0.2, num_envs=4)

  assert command.shape == (4, 3)
  assert torch.unique(command, dim=0).shape[0] == 1


@pytest.mark.parametrize("plant", ["eval", "training"])
def test_eval_scene_shares_the_task_terrain_and_solver(plant):
  """Both engines meet the same floor: the plant is the only thing swapped."""
  scene, sim = eval_scene_cfg(plant, num_envs=4)

  assert scene.num_envs == 4
  assert set(scene.entities) == {"robot"}
  assert scene.terrain is not None
  assert scene.terrain.terrain_type == "plane"
  assert sim.mujoco.timestep > 0.0


@pytest.mark.slow
def test_quintic_harness_walks_and_records():
  """The batched rig reproduces the single-environment one: it walks."""
  device = get_test_device()
  harness = QuinticEvalHarness(plant="eval", num_envs=2, device=device)

  assert harness.control_dt == pytest.approx(0.01), "engine must run at 100 Hz"

  metrics = harness.run(constant_command(0.2, 0.0, 0.0, 2, device), 3.0)
  result = metrics.result()

  assert result.survived.tolist() == [1.0, 1.0]
  assert float(result.min_upright.min()) > 0.8
  assert float(result.displacement_x.min()) > 0.05
  assert int(harness.engine_state[0]) == int(EngineState.WALKING)
  # Both environments walk, but do not assert they walk *identically*: a batched
  # GPU run is not bit-reproducible across environments -- reduction ordering
  # separates two identical robots by around 1e-7 per step, and a gait amplifies
  # that. Reproducibility is a single-environment property; see
  # test_step_callback_does_not_change_the_run.
  assert float(result.achieved_vx.min()) > 0.0

  summary = summarise(result)
  assert summary["num_survived"] == 2


@pytest.mark.slow
def test_sensed_phase_is_batched_and_on_device():
  """The planted-phase path has to work off a batched entity, not MjData."""
  device = get_test_device()
  harness = QuinticEvalHarness(plant="eval", num_envs=3, device=device)

  phase = harness.sensed_phase()

  assert phase.shape == (3,)
  assert str(phase.device).startswith(str(device).split(":")[0])


@pytest.mark.slow
def test_step_callback_does_not_change_the_run():
  """The live view hooks in here, so the hook must be inert.

  Same plant, same command, once with a callback and once without: identical
  metrics, and the callback fires exactly once per control step.

  One environment, because a batch of two is not bit-reproducible on the GPU --
  identical environments in one batch drift apart by about 1e-7 through reduction
  ordering. A single environment is exact, which makes this an equality test
  rather than a tolerance test.
  """
  device = get_test_device()
  command_args = (0.2, 0.0, 0.0, 1, device)
  duration = 1.5

  plain = QuinticEvalHarness(plant="eval", num_envs=1, device=device)
  expected = plain.run(constant_command(*command_args), duration).result()

  seen: list[int] = []
  watched = QuinticEvalHarness(plant="eval", num_envs=1, device=device)
  observed = watched.run(
    constant_command(*command_args), duration, on_step=seen.append
  ).result()

  assert seen == list(range(int(duration / watched.control_dt)))
  for name in ("achieved_vx", "achieved_vy", "min_upright", "displacement_x"):
    assert getattr(observed, name).tolist() == pytest.approx(
      getattr(expected, name).tolist(), abs=1e-9
    )


@pytest.mark.slow
def test_distilled_harness_walks_and_records():
  """The distilled policy drives the same rig, and its copy holds up."""
  device = get_test_device()
  harness = DistilledEvalHarness(
    plant="eval", num_envs=2, device=device, track_teacher=True
  )

  assert harness.control_dt == pytest.approx(0.01), "policy runs at 100 Hz"

  result = harness.run(constant_command(0.2, 0.0, 0.0, 2, device), 3.0).result()

  assert result.survived.tolist() == [1.0, 1.0]
  assert float(result.displacement_x.min()) > 0.05
  assert int(harness.engine_state[0]) == int(EngineState.WALKING)

  tracking = harness.teacher_tracking()
  assert tracking is not None
  assert tracking["steps"] == int(3.0 / harness.control_dt)
  # About its own stance the copy stays close to the engine it was fit to. The
  # raw gap is larger because the two solve different IK; see the harness.
  assert tracking["stance_relative_mean_abs_error_rad"] < 0.05
  assert tracking["mean_abs_error_rad"] > tracking["stance_relative_mean_abs_error_rad"]


@pytest.mark.slow
def test_distilled_harness_starts_in_the_pose_the_policy_asks_for():
  """Otherwise the first control step is a yank rather than a walk."""
  device = get_test_device()
  harness = DistilledEvalHarness(plant="eval", num_envs=1, device=device)
  lookup = {name: index for index, name in enumerate(harness.robot.joint_names)}

  legs = [lookup[name] for name in JOINT_NAMES]
  standing = harness.robot.data.joint_pos[0, legs]

  assert torch.allclose(standing, harness.controller.home_targets[0], atol=1e-5)


@pytest.mark.slow
def test_a_push_knocks_the_robot_off_its_command_and_a_big_one_topples_it():
  """The force reaches the plant, and harder is worse.

  Two magnitudes at one heading and one gait phase, both shoving the robot
  sideways while it walks forward. The small one it should walk out of; the
  large one is four times the momentum and should put it on the floor. Read as
  a pair rather than as two thresholds: what is being checked is that the
  battery is coupled to the simulation at all and in the right direction, not
  where this controller's envelope happens to sit.
  """
  device = get_test_device()
  cfg = PushCfg(vx=0.2, directions=1, phases=1, replicas=1, settle=4.0, recovery=2.5)
  harness = QuinticEvalHarness(plant="eval", num_envs=1, device=device)

  gentle = harness.run_push(
    push_plan(
      cfg, delta_v=0.1, mass=harness.robot_mass(), dt=harness.control_dt, device=device
    )
  ).result()
  brutal = harness.run_push(
    push_plan(
      cfg, delta_v=1.6, mass=harness.robot_mass(), dt=harness.control_dt, device=device
    )
  ).result()

  assert gentle.withstood.tolist() == [1.0]
  assert brutal.withstood.tolist() == [0.0]
  assert float(brutal.peak_speed_error[0]) > float(gentle.peak_speed_error[0])
  assert float(brutal.time_to_fall[0]) > 0.0
  # And the walking metrics still work, measured over the window the push is in.
  assert float(gentle.achieved_vx[0]) > 0.0


@pytest.mark.slow
def test_a_battery_runs_every_magnitude_through_one_harness():
  """Passes are independent runs of the same batch, concatenated."""
  device = get_test_device()
  cfg = PushCfg(
    delta_v=(0.1, 1.6),
    directions=2,
    phases=1,
    replicas=1,
    settle=3.0,
    recovery=2.0,
  )
  harness = QuinticEvalHarness(
    plant="eval", num_envs=cfg.trials_per_pass, device=device
  )

  result = run_push_battery(harness, cfg)

  assert result.withstood.numel() == cfg.num_trials
  assert sorted(set(result.push_delta_v.tolist())) == pytest.approx([0.1, 1.6])
  assert sorted(set(result.push_heading_deg.tolist())) == pytest.approx([0.0, 180.0])
  # A magnitude is one pass, so the trials come out in magnitude order.
  assert result.push_delta_v[: cfg.trials_per_pass].tolist() == pytest.approx([0.1] * 2)


@pytest.mark.slow
def test_a_battery_leaves_no_force_behind_it():
  """xfrc_applied persists, so a pass that ended mid-push would poison the next
  run on the same harness."""
  device = get_test_device()
  cfg = PushCfg(directions=1, phases=1, replicas=1, settle=1.0, recovery=0.1)
  harness = QuinticEvalHarness(plant="eval", num_envs=1, device=device)
  plan = push_plan(
    cfg, delta_v=1.0, mass=harness.robot_mass(), dt=harness.control_dt, device=device
  )
  # A run whose last step is inside the push window: the clear is the only
  # thing that can zero the wrench.
  harness.run_push(replace(plan, num_steps=int(plan.push_step[0]) + 1))

  assert float(harness.robot.data.body_external_force.abs().max()) == 0.0


@pytest.mark.slow
def test_partial_reset_leaves_the_other_environments_walking():
  """A competence grid resets the episodes that ended and nothing else.

  Two environments walk; one is reset. The reset one must be back in the
  stance at its own origin, and the other must be exactly where it was --
  a reset that touched it would silently restart its episode mid-measurement.
  """
  device = get_test_device()
  harness = QuinticEvalHarness(plant="eval", num_envs=2, device=device)
  command = constant_command(0.2, 0.0, 0.0, 2, device)
  for _ in range(120):
    harness.step(command)

  walked = harness.state().position_w.clone()
  assert float(walked[0, 0] - harness.scene.env_origins[0, 0]) > 0.02

  harness.reset_idx(torch.tensor([0], device=device))
  after = harness.state().position_w

  # The reset environment is back at its origin's stance.
  stance_x = float(harness._stance_root_pose[0])  # noqa: SLF001
  assert float(after[0, 0] - harness.scene.env_origins[0, 0]) == pytest.approx(
    stance_x, abs=1e-4
  )
  # The untouched one has not moved.
  assert float((after[1] - walked[1]).abs().max()) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
def test_quintic_competence_grid_produces_independent_episodes():
  """The engine runs the same competence collector a policy does."""
  from mjlab.evaluation.competence import ShoveCfg, build_grid

  device = get_test_device()
  harness = QuinticEvalHarness(plant="eval", num_envs=4, device=device)
  grid = build_grid(((0.2, 0.0, 0.0),), (0.0,), num_envs=4, device=device)

  table = harness.run_competence_grid(
    grid,
    episodes_per_cell=2,
    shove_cfg=ShoveCfg(settle=0.5, period=1.0, tail=0.5),
    episode_length_s=2.0,
  )

  assert table.num_episodes >= 2
  # Two seconds at 100 Hz, so a full episode is 200 steps and ep_len_frac 1.0.
  assert float(table.ep_len_frac.max()) == pytest.approx(1.0)
  assert bool((table.ep_len_frac > 0.0).all())
  # The engine walks at this command, so attainment is measured, not NaN.
  assert bool(table.attain.isfinite().any())
