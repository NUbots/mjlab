"""Tests for the batched evaluation harness.

Marked slow: each test compiles a model and warms up warp kernels, which costs
about forty seconds on a cold CPU cache. The metrics themselves are covered
without a simulator in ``test_walk_metrics.py``.
"""

import pytest
import torch
from conftest import get_test_device

from mjlab.controllers.quintic_walk.walk_generator import EngineState
from mjlab.evaluation.harness import (
  QuinticEvalHarness,
  command_grid,
  constant_command,
  eval_scene_cfg,
)
from mjlab.evaluation.metrics import summarise


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
  # Identical environments, identical commands, so identical trajectories.
  assert float(result.achieved_vx[0]) == pytest.approx(
    float(result.achieved_vx[1]), abs=1e-5
  )

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
