"""Tests for the ported NUbots quintic walk trajectory generator.

The load-bearing test is :func:`test_trace_matches_nubots_cpp`, which replays
four command profiles through the port and diffs every control step against a
trace dumped from the real C++ ``WalkGenerator``. See
``tests/fixtures/quintic_walk_cpp/README.md`` for how that trace is regenerated.
"""

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import get_test_device

from mjlab.controllers.quintic_walk.controller import QuinticWalkController
from mjlab.controllers.quintic_walk.kinematics import mat_to_rpy_intrinsic
from mjlab.controllers.quintic_walk.walk_generator import (
  ENGINE_DTYPE,
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
  WalkGenerator,
)

GOLDEN_CSV = Path(__file__).parent / "fixtures" / "quintic_walk_trace_golden.csv"

CONTROL_DT = 0.01
"""Control period the golden trace was generated at."""

POSE_COLUMNS = tuple(
  f"{prefix}_{axis}"
  for prefix in ("torso", "swing", "lfoot", "rfoot")
  for axis in ("x", "y", "z", "roll", "pitch", "yaw")
)


def command_for(scenario: str, step: int) -> tuple[float, float, float]:
  """Command profile per scenario, matching ``dump_walk.cpp``."""
  t = step * CONTROL_DT
  if scenario == "start_stop":
    if t < 0.25:
      return (0.0, 0.0, 0.0)
    if t < 1.75:
      return (0.25, 0.0, 0.0)
    return (0.0, 0.0, 0.0)
  if scenario == "omni":
    return (0.15, -0.08, 0.3)
  return (0.2, 0.0, 0.0)


def sensed_phase_for(step: int) -> int:
  """Deterministic stand-in for foot contact, matching ``dump_walk.cpp``."""
  return int(Phase.LEFT) if (step // 32) % 2 == 0 else int(Phase.RIGHT)


def pose_values(transform: torch.Tensor) -> list[float]:
  translation = transform[:, :3, 3]
  rpy = mat_to_rpy_intrinsic(transform[:, :3, :3])
  return torch.cat((translation, rpy), dim=-1)[0].tolist()


@pytest.fixture(scope="module")
def golden():
  with GOLDEN_CSV.open() as handle:
    rows = list(csv.DictReader(handle))
  assert rows, "golden trace is empty"
  scenarios: dict[str, list[dict[str, str]]] = {}
  for row in rows:
    scenarios.setdefault(row["scenario"], []).append(row)
  return scenarios


@pytest.fixture
def device():
  return get_test_device()


@pytest.mark.parametrize("scenario", ["forward", "start_stop", "omni", "planted"])
def test_trace_matches_nubots_cpp(golden, device, scenario):
  """Every control step reproduces the C++ engine to float64 precision.

  This covers the state machine, the phase clock, foot switching and both
  trajectories together, so a mistake anywhere in the engine shows up as a
  divergence at the step it happens.
  """
  rows = golden[scenario]
  params = replace(
    NUGUS_WALK_PARAMETERS, only_switch_when_planted=scenario == "planted"
  )
  # Deliberately no dtype override: the golden trace is what pins the engine's
  # working precision, and it only does that if it exercises the default. The
  # clock accumulates, so float32 shifts the foot switch by a whole tick.
  generator = WalkGenerator(1, device=device, params=params)

  for row in rows:
    step = int(row["step"])
    # The command stays double: the trace was dumped from a C++ run that had
    # exact doubles on its input, so rounding it to float32 here would inject a
    # difference that says nothing about the engine.
    command = torch.tensor(
      [command_for(scenario, step)], dtype=torch.float64, device=device
    )
    sensed = torch.tensor([sensed_phase_for(step)], device=device)

    state = generator.update(CONTROL_DT, command, sensed)

    assert int(state[0]) == int(row["state"]), f"state diverged at step {step}"
    assert int(generator.phase[0]) == int(row["phase"]), f"phase diverged at {step}"
    assert float(generator.time[0]) == pytest.approx(float(row["t"]), abs=1e-12), (
      f"clock diverged at step {step}"
    )

    values = (
      pose_values(generator.torso_pose())
      + pose_values(generator.swing_foot_pose())
      + pose_values(generator.foot_pose(left=True))
      + pose_values(generator.foot_pose(left=False))
    )
    for name, value in zip(POSE_COLUMNS, values, strict=True):
      assert value == pytest.approx(float(row[name]), abs=1e-12), (
        f"{name} diverged at step {step}"
      )


def test_trace_exercises_every_engine_state(golden):
  """The scenarios cover the whole state machine, not just steady walking."""
  seen = {int(row["state"]) for rows in golden.values() for row in rows}
  expected = {
    int(EngineState.STARTING),
    int(EngineState.WALKING),
    int(EngineState.STOPPING),
    int(EngineState.STOPPED),
  }
  assert expected <= seen, f"states not covered: {expected - seen}"


def test_engine_runs_in_double_by_default():
  """The C++ engine is ``WalkGenerator<double>``, and so is this one.

  Not a style preference: the phase clock accumulates the control period and
  switches feet on ``t >= step_period``. In float32 thirty-two additions of 0.01
  reach 0.319999963 against a period of 0.3199999928, so the switch slips a tick
  and every step takes 0.33 s instead of 0.32.
  """
  assert ENGINE_DTYPE is torch.float64
  generator = WalkGenerator(1)
  assert generator.time.dtype is ENGINE_DTYPE
  assert QuinticWalkController(1).dtype is ENGINE_DTYPE


def test_foot_switches_on_the_configured_step_period(device):
  """The clock lands on the period exactly, so the cadence is the tuned one."""
  generator = WalkGenerator(1, device=device)
  command = torch.tensor([[0.15, 0.0, 0.0]], device=device)

  previous = int(generator.phase[0])
  switches = []
  for step in range(600):
    generator.update(CONTROL_DT, command)
    current = int(generator.phase[0])
    if current != previous:
      switches.append(step)
      previous = current

  gaps = {b - a for a, b in zip(switches, switches[1:], strict=False)}
  expected = round(NUGUS_WALK_PARAMETERS.step_period / CONTROL_DT)
  assert gaps == {expected}, f"step lasted {gaps} ticks, expected {expected}"


def test_engine_state_values_match_the_proto():
  """Enum values match ``WalkState.proto`` so traces compare directly."""
  assert (int(EngineState.UNKNOWN), int(EngineState.STARTING)) == (0, 1)
  assert (int(EngineState.WALKING), int(EngineState.STOPPING)) == (2, 3)
  assert int(EngineState.STOPPED) == 4
  assert (int(Phase.DOUBLE), int(Phase.LEFT), int(Phase.RIGHT)) == (0, 1, 2)


def test_only_switch_when_planted_defaults_to_deployed_behaviour():
  """The default reproduces the robot, not the config file.

  ``Walk.yaml`` sets ``only_switch_when_planted: true``, but ``Walk.cpp`` calls
  ``set_parameters`` before assigning the field, so the generator never receives
  it. Defaulting to ``False`` keeps the port faithful to what actually runs.
  """
  assert NUGUS_WALK_PARAMETERS.only_switch_when_planted is False


def test_requires_sensed_phase_when_waiting_on_contact(device):
  """Enabling the knob without supplying contact is a hard error, not a guess."""
  params = replace(NUGUS_WALK_PARAMETERS, only_switch_when_planted=True)
  generator = WalkGenerator(1, device=device, params=params)

  with pytest.raises(ValueError, match="sensed_phase"):
    generator.update(CONTROL_DT, torch.tensor([[0.2, 0.0, 0.0]], device=device))


def test_batched_envs_track_independent_commands(device):
  """Environments with different commands evolve independently."""
  generator = WalkGenerator(3, device=device)
  command = torch.tensor(
    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.4]], device=device
  )

  for _ in range(120):
    generator.update(CONTROL_DT, command)

  assert int(generator.state[0]) == int(EngineState.STOPPED)
  assert int(generator.state[1]) == int(EngineState.WALKING)
  assert int(generator.state[2]) == int(EngineState.WALKING)


def test_step_placement_is_clamped_to_the_limits(device):
  """An absurd command saturates at the configured step limits."""
  generator = WalkGenerator(1, device=device, dtype=torch.float64)
  huge = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float64, device=device)

  for _ in range(80):
    generator.update(CONTROL_DT, huge)

  swing = generator.swing_foot_pose(
    torch.full(
      (1,), NUGUS_WALK_PARAMETERS.step_period, dtype=torch.float64, device=device
    )
  )
  limits = NUGUS_WALK_PARAMETERS.step_limits
  assert abs(float(swing[0, 0, 3])) <= limits[0] + 1e-9
