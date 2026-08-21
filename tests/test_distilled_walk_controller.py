"""Tests for driving the distilled walk policy.

The policy is blind, so everything about how it behaves is decided by the
observation this controller builds for it: the phase clock, the engine state and
its own action history. These tests pin that assembly against the recorded data
the policy was trained from -- see ``test_distilled_walk_policy.py`` for the
fixture's provenance -- and against the deployed module's start-up behaviour.
"""

import numpy as np
import pytest
import torch

from mjlab.controllers.distilled_walk import (
  HISTORY_FRAMES,
  DistilledWalkController,
  DistilledWalkPolicy,
)
from mjlab.controllers.quintic_walk.walk_generator import EngineState

CONTROL_DT = 0.01
"""The rate the policy was distilled at."""

RECORDED_STATE_ORDER = (
  EngineState.STOPPED,
  EngineState.STARTING,
  EngineState.WALKING,
  EngineState.STOPPING,
)
"""One-hot column order ``WalkDataCollector`` wrote the engine state in."""


@pytest.fixture(scope="module")
def policy() -> DistilledWalkPolicy:
  return DistilledWalkPolicy.from_onnx()


def make_controller(policy, num_envs=2, **kwargs) -> DistilledWalkController:
  return DistilledWalkController(num_envs, policy, device="cpu", **kwargs)


def test_observation_has_the_layout_the_policy_was_trained_on(policy):
  controller = make_controller(policy)

  obs = controller.observation()

  assert obs.shape == (2, 46)
  # Standing: no command, clock parked at the end of the step, left foot
  # planted, engine stopped, history holding the pose it stands in.
  assert obs[:, :3].abs().max() == 0.0
  assert obs[0, 3] == pytest.approx(0.0, abs=1e-6)
  assert obs[0, 4] == pytest.approx(1.0, abs=1e-6)
  assert obs[0, 5] == 1.0
  assert obs[0, 6:10].tolist() == [1.0, 0.0, 0.0, 0.0]
  for frame in range(HISTORY_FRAMES):
    columns = obs[:, 10 + 12 * frame : 22 + 12 * frame]
    assert torch.equal(columns, controller.home_targets)


def test_the_policys_own_standing_pose_is_the_recorded_one(policy, fixtures_dir):
  """The stance is recovered from the policy, not from NUbots' IK.

  ``WalkDataCollector`` seeded its history by solving the engine's stance with
  the IK the robot deploys, which mjlab does not have; this port's own solver
  lands half a radian of knee away. Iterating the policy at rest gets there
  anyway, because standing is a fixed point of the thing that was fit to it.
  """
  data = np.loadtxt(
    fixtures_dir / "distilled_walk_episode_golden.csv",
    delimiter=",",
    skiprows=1,
    dtype=np.float32,
  )
  recorded_stance = torch.from_numpy(data[0, 10:22])
  controller = make_controller(policy, num_envs=1)

  settled = controller.home_targets[0]

  assert torch.allclose(settled, recorded_stance, atol=1e-3)
  # Not the same as the stance this port's IK solves for, which is the point.
  assert (settled - controller.stance_targets()[0]).abs().max() > 0.4


def test_clock_matches_the_engine_that_generated_the_training_data(
  policy, fixtures_dir
):
  """Replay a recorded episode's commands and check the clock columns agree.

  The recorded velocities have already been through the slew limiter, so the
  limiter is opened up here to let them through unchanged; everything else is
  the engine driving itself exactly as ``WalkDataCollector`` drove it.
  """
  data = np.loadtxt(
    fixtures_dir / "distilled_walk_episode_golden.csv",
    delimiter=",",
    skiprows=1,
    dtype=np.float32,
  )
  observations = torch.from_numpy(data[:, :46])
  controller = make_controller(policy, num_envs=1, max_acceleration=(1e3, 1e3, 1e3))

  worst_clock = 0.0
  for step in range(observations.shape[0]):
    controller.compute(CONTROL_DT, observations[step : step + 1, :3])
    built = controller.observation()[0]
    worst_clock = max(
      worst_clock, float((built[:6] - observations[step, :6]).abs().max())
    )
    recorded_state = RECORDED_STATE_ORDER[int(observations[step, 6:10].argmax())]
    assert int(controller.generator.state[0]) == int(recorded_state), f"step {step}"

  assert worst_clock < 1e-6


def test_history_carries_the_three_previous_outputs(policy):
  controller = make_controller(policy, num_envs=1)
  command = torch.tensor([[0.2, 0.0, 0.0]])

  outputs = [controller.compute(CONTROL_DT, command).clone() for _ in range(4)]

  for frame in range(HISTORY_FRAMES):
    assert torch.equal(controller.history[:, frame], outputs[-1 - frame])


def test_history_init_selects_what_the_run_starts_from(policy):
  """Each option starts the run somewhere different, on purpose."""
  settled = make_controller(policy, num_envs=1, history_init="settled")
  stance = make_controller(policy, num_envs=1, history_init="stance")
  zeros = make_controller(policy, num_envs=1, history_init="zeros")

  assert torch.equal(settled.history[:, 0], settled.home_targets)
  assert torch.equal(stance.history[:, 0], stance.stance_targets())
  assert zeros.history.abs().max() == 0.0
  # Only the settled start puts the robot in the policy's own pose. The other
  # two leave it in the engine's, which is where the robot is standing when the
  # walk task hands over to NeuralWalk.
  assert torch.equal(stance.home_targets, stance.stance_targets())
  assert torch.equal(zeros.home_targets, zeros.stance_targets())
  assert (settled.home_targets - settled.stance_targets()).abs().max() > 0.4
