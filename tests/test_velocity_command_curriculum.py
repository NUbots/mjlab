"""Tests for the velocity command curriculum reachability.

The wide backward/strafe/yaw command stage must be reached early within a
~1250-iteration run (24 steps/iteration), not at the old 9000/12000-iteration
thresholds that were never hit.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_NUM_STEPS_PER_ENV = 24
_RUN_ITERATIONS = 1250


def _stages() -> list[dict]:
  cfg = make_velocity_env_cfg()
  return cfg.curriculum["command_vel"].params["velocity_stages"]


def _make_env(step_counter: int) -> tuple[MagicMock, MagicMock]:
  """Env whose command term exposes mutable ranges, at a given step counter."""
  ranges = MagicMock()
  ranges.lin_vel_x = (0.0, 0.0)
  ranges.lin_vel_y = (0.0, 0.0)
  ranges.ang_vel_z = (0.0, 0.0)
  term = MagicMock()
  term.cfg.ranges = ranges
  env = MagicMock()
  env.common_step_counter = step_counter
  env.command_manager.get_term = MagicMock(return_value=term)
  return env, term


def test_wide_command_stage_reached_within_run():
  # The final (widest) stage must be reachable well inside a 1250-iter run.
  stages = _stages()
  final_step = stages[-1]["step"]
  run_steps = _RUN_ITERATIONS * _NUM_STEPS_PER_ENV
  assert final_step < run_steps, (
    f"Final command stage at {final_step} is not reached within a "
    f"{_RUN_ITERATIONS}-iter run ({run_steps} steps)."
  )
  # And it should land in the first ~half of the run so the policy has time to
  # learn the wide ranges.
  assert final_step < 0.5 * run_steps


def test_commands_vel_applies_wide_ranges_after_final_stage():
  stages = _stages()
  final = stages[-1]
  env, term = _make_env(step_counter=final["step"])
  commands_vel(
    env, env_ids=torch.tensor([0]), command_name="twist", velocity_stages=stages
  )
  assert term.cfg.ranges.lin_vel_x == final["lin_vel_x"]
  assert term.cfg.ranges.lin_vel_y == final["lin_vel_y"]
  assert term.cfg.ranges.ang_vel_z == final["ang_vel_z"]
  # The restored symmetric wide ranges.
  assert term.cfg.ranges.lin_vel_x == (-0.5, 0.5)
  assert term.cfg.ranges.lin_vel_y == (-0.3, 0.3)
  assert term.cfg.ranges.ang_vel_z == (-0.5, 0.5)


def test_full_yaw_reached_at_early_stage():
  # The models must be able to rotate: the full +/-0.5 yaw range has to be
  # unlocked at a stage that is comfortably reached within the run, and no later
  # than the stage where the lateral range reaches its full +/-0.3.
  stages = _stages()
  run_steps = _RUN_ITERATIONS * _NUM_STEPS_PER_ENV

  yaw_full_step = next(s["step"] for s in stages if s["ang_vel_z"] == (-0.5, 0.5))
  liny_full_step = next(s["step"] for s in stages if s["lin_vel_y"] == (-0.3, 0.3))

  # Reached well within the run (first quarter), not at an unreachable tail.
  assert yaw_full_step < 0.25 * run_steps
  # Yaw opens up no later than the lateral range.
  assert yaw_full_step <= liny_full_step

  # And applying the curriculum at that step yields the full yaw range.
  env, term = _make_env(step_counter=yaw_full_step)
  commands_vel(
    env, env_ids=torch.tensor([0]), command_name="twist", velocity_stages=stages
  )
  assert term.cfg.ranges.ang_vel_z == (-0.5, 0.5)


def test_commands_vel_narrow_before_first_boundary():
  stages = _stages()
  env, term = _make_env(step_counter=0)
  commands_vel(
    env, env_ids=torch.tensor([0]), command_name="twist", velocity_stages=stages
  )
  assert term.cfg.ranges.lin_vel_y == stages[0]["lin_vel_y"]
  assert term.cfg.ranges.ang_vel_z == stages[0]["ang_vel_z"]
