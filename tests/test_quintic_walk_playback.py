"""Tests for driving the quintic walk engine against a compiled NUgus.

The load-bearing test is :func:`test_eval_plant_walks_without_falling`, which is
the end-to-end statement the whole port exists to support: the deployed walk
parameters, driving the deployed idealised IK, keep the robot upright and moving
on the evaluation model.
"""

import json
import math
from dataclasses import replace

import mujoco
import pytest
import torch

from mjlab.asset_zoo.robots.nugus import nugus_constants, nugus_eval_constants
from mjlab.controllers.quintic_walk.controller import (
  JOINT_NAMES,
  detect_planted_phase,
  sole_poses_in_torso,
)
from mjlab.controllers.quintic_walk.playback import WalkPlayback
from mjlab.controllers.quintic_walk.walk_generator import (
  NUGUS_WALK_PARAMETERS,
  EngineState,
  Phase,
)
from mjlab.entity import Entity

FORWARD = (0.2, 0.0, 0.0)


@pytest.fixture(scope="module")
def eval_model() -> mujoco.MjModel:
  return Entity(nugus_eval_constants.get_nugus_eval_robot_cfg()).spec.compile()


def test_eval_plant_walks_without_falling():
  """Deployed parameters, deployed IK, reference model: it walks.

  Five seconds is about fifteen steps, which is well past the point where the
  training model has already toppled.
  """
  playback = WalkPlayback(plant="eval")
  result = playback.run(FORWARD, 5.0)

  assert not result.fell, f"fell at {result.fall_time} s"
  assert result.min_upright > 0.8, "torso pitched over during the run"
  assert result.engine_state is EngineState.WALKING
  # The engine undershoots its own command by roughly a fifth; assert progress
  # rather than accuracy, which is a property of the tuning, not of the port.
  assert result.displacement[0] > 0.5, f"barely moved: {result.displacement}"
  assert abs(result.displacement[1]) < 0.3, f"veered: {result.displacement}"


def test_stopped_stance_stands_still():
  """A zero command holds the standing stance instead of stepping."""
  playback = WalkPlayback(plant="eval")
  result = playback.run((0.0, 0.0, 0.0), 1.0)

  assert result.engine_state is EngineState.STOPPED
  assert not result.fell
  assert abs(result.displacement[0]) < 0.02


def test_settled_stance_stands_flat_on_both_feet():
  """Reset leaves both soles on the floor, which is what makes it a stance.

  The engine holds the torso pitched forward, so a robot dropped in upright
  would rest on its heels; the base orientation has to come from the feet.
  """
  playback = WalkPlayback(plant="eval")

  assert playback.lowest_sole_height() == pytest.approx(0.0, abs=1e-6)
  left, right = sole_poses_in_torso(playback.model, playback.data)
  height_difference = float(left[0, 2, 3] - right[0, 2, 3])
  assert height_difference == pytest.approx(0.0, abs=1e-3)
  assert int(detect_planted_phase(left, right)[0]) == int(Phase.DOUBLE)
  # cos(14.4 degrees) = 0.969: the stance pitch the idealised IK produces.
  assert playback.upright == pytest.approx(0.969, abs=0.01)


def test_detect_planted_phase_matches_the_sensorfilter_convention():
  """Pin the sign convention: the *lower* foot is the planted one."""
  left = torch.eye(4).unsqueeze(0)
  right = torch.eye(4).unsqueeze(0)

  right[0, 2, 3] = 0.05
  assert int(detect_planted_phase(left, right)[0]) == int(Phase.LEFT)
  right[0, 2, 3] = -0.05
  assert int(detect_planted_phase(left, right)[0]) == int(Phase.RIGHT)
  right[0, 2, 3] = 0.005
  assert int(detect_planted_phase(left, right)[0]) == int(Phase.DOUBLE)


def test_sensed_phase_tracks_the_engine_while_walking():
  """The measured phase is live, alternates, and agrees with the engine's.

  Agreement is the end-to-end check on the frames: the engine plants the foot
  it believes is down, so a sole frame with the wrong handedness or a swapped
  left and right would show up here as anti-correlation rather than as a
  crash. Exact agreement is not expected -- double support covers the
  transitions, and the sensed phase lags the commanded one by however long the
  foot takes to arrive.
  """
  playback = WalkPlayback(plant="eval")
  command = torch.tensor([FORWARD])

  seen: set[int] = set()
  agree = 0
  disagree = 0
  for _ in range(300):
    playback.step(command)
    sensed = int(playback.sensed_phase()[0])
    seen.add(sensed)
    if sensed == int(Phase.DOUBLE):
      continue
    if sensed == int(playback.controller.generator.phase[0]):
      agree += 1
    else:
      disagree += 1

  assert seen == {int(Phase.DOUBLE), int(Phase.LEFT), int(Phase.RIGHT)}
  assert agree > 5 * disagree, f"agree {agree}, disagree {disagree}"


def test_playback_supplies_the_sensed_phase():
  """The engine can be asked to wait for contact, so the wiring must be live.

  With ``only_switch_when_planted`` the generator raises unless a sensed phase
  reaches it every update, so a run that completes is the assertion.
  """
  playback = WalkPlayback(
    plant="eval",
    walk_params=replace(NUGUS_WALK_PARAMETERS, only_switch_when_planted=True),
  )
  result = playback.run(FORWARD, 2.0)

  assert result.engine_state is EngineState.WALKING
  assert not result.fell


def test_eval_plant_restores_the_hardware_joint_limits(eval_model):
  """The engine commands past mjlab's RL clamps, so they have to come off.

  The clamp lives on the joint rather than on the command -- position actuators
  are deliberately left unlimited -- so the informational ``ctrlrange`` has to
  follow the widened joint range, which it only does if the spec is edited
  before the actuators are added.
  """
  low, high = nugus_eval_constants.HARDWARE_JOINT_RANGE
  for name in nugus_eval_constants.LEG_JOINT_NAMES:
    joint = eval_model.joint(name)
    assert joint.range[0] == pytest.approx(low)
    assert joint.range[1] == pytest.approx(high)

    actuator = eval_model.actuator(name)
    assert not actuator.ctrllimited
    assert actuator.ctrlrange[0] <= low
    assert actuator.ctrlrange[1] >= high


def test_eval_plant_drops_backlash_but_keeps_the_training_actuators(eval_model):
  """Randomisation at nominal: no passive gear play, real servo torque limits."""
  joint_names = {eval_model.joint(i).name for i in range(eval_model.njnt)}
  assert not any(name.endswith("_backlash") for name in joint_names)

  effort = nugus_constants.ACTUATOR_XH540.effort_limit
  assert eval_model.actuator("left_knee_pitch").forcerange[1] == pytest.approx(effort)


def test_training_model_is_left_alone():
  """Policies are trained against this one; it must not drift."""
  training = Entity(nugus_constants.get_nugus_robot_cfg()).spec.compile()

  joint_names = {training.joint(i).name for i in range(training.njnt)}
  assert any(name.endswith("_backlash") for name in joint_names)
  assert training.joint("left_ankle_pitch").range[1] == pytest.approx(1.0)
  assert training.joint("left_hip_roll").range[0] == pytest.approx(0.0)


def test_recording_pairs_commands_with_measurements():
  """One row per control step, holding both sides of the control loop.

  The ``cmd_`` columns are the engine's own output, which is golden-trace
  tested against the C++; pinning that they reach the CSV unchanged is what
  makes a cross-simulator comparison meaningful.
  """
  playback = WalkPlayback(plant="eval")
  recorder = playback.start_recording()
  duration = 0.3
  playback.run(FORWARD, duration)

  expected_rows = int(duration / playback.control_dt)
  assert len(recorder.rows) == expected_rows
  assert len(recorder.columns) == 50
  assert all(len(row) == len(recorder.columns) for row in recorder.rows)

  for name in ("time", "engine_state", "torso_roll", "gyro_x", "left_sole_height"):
    assert name in recorder.columns
  for joint in JOINT_NAMES:
    for prefix in ("cmd", "pos", "vel"):
      assert f"{prefix}_{joint}" in recorder.columns

  values = [value for row in recorder.rows for value in row]
  assert all(math.isfinite(value) for value in values)

  # Time counts control steps, and the last row's commands are the ones the
  # controller last returned.
  assert recorder.rows[-1][0] == pytest.approx(duration, abs=1e-9)
  assert playback.last_targets is not None
  first_cmd = recorder.columns.index(f"cmd_{JOINT_NAMES[0]}")
  recorded = recorder.rows[-1][first_cmd : first_cmd + len(JOINT_NAMES)]
  assert recorded == pytest.approx(playback.last_targets.tolist(), abs=1e-12)


def test_recording_writes_a_csv_and_a_metadata_sibling(tmp_path):
  """The trace and the configuration that produced it travel together."""
  playback = WalkPlayback(plant="eval")
  recorder = playback.start_recording()
  playback.run(FORWARD, 0.1)

  csv_path = tmp_path / "trace.csv"
  metadata_path = recorder.write(csv_path, {"plant": "eval"})

  assert metadata_path == tmp_path / "trace.json"
  lines = csv_path.read_text().strip().split("\n")
  assert lines[0].split(",") == recorder.columns
  assert len(lines) == len(recorder.rows) + 1

  metadata = json.loads(metadata_path.read_text())
  assert metadata["plant"] == "eval"
  assert metadata["rows"] == len(recorder.rows)
  assert metadata["columns"] == recorder.columns
  assert metadata["engine_state_values"]["WALKING"] == int(EngineState.WALKING)
  assert "git_sha" in metadata and "timing" in metadata
