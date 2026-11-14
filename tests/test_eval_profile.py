"""Tests for the time-varying command profiles and the velocity trace.

No simulator: a profile is arithmetic over a schedule, and the trace is a
recorder, so both are checked against hand-built numbers.
"""

import math

import pytest
import torch

from mjlab.evaluation.metrics import EvalState, VelocityTrace, write_trace_csv
from mjlab.evaluation.profile import (
  Lane,
  Profile,
  ProfileCfg,
  Segment,
  omnidirectional_profile,
)

DT = 0.01


def lane(*segments: Segment) -> Lane:
  return Lane("test", ("vx",), segments)


def test_ramp_interpolates_between_plateaus():
  schedule = lane(Segment(vx=0.4, hold=2.0, ramp=1.0))

  assert schedule.command_at(0.0) == pytest.approx((0.0, 0.0, 0.0))
  assert schedule.command_at(0.5) == pytest.approx((0.2, 0.0, 0.0))
  assert schedule.command_at(1.0) == pytest.approx((0.4, 0.0, 0.0))
  assert schedule.command_at(2.5) == pytest.approx((0.4, 0.0, 0.0))


def test_ramp_runs_from_the_previous_plateau_not_from_rest():
  schedule = lane(
    Segment(vx=0.4, hold=1.0, ramp=0.0),
    Segment(vx=-0.2, hold=1.0, ramp=1.0),
  )

  assert schedule.command_at(1.0) == pytest.approx((0.4, 0.0, 0.0))
  assert schedule.command_at(1.5) == pytest.approx((0.1, 0.0, 0.0))
  assert schedule.command_at(2.0) == pytest.approx((-0.2, 0.0, 0.0))


def test_a_finished_lane_stands_still():
  """So a short lane can share a batch with a long one."""
  schedule = lane(Segment(vx=0.4, hold=1.0, ramp=0.0))

  assert schedule.duration == pytest.approx(1.0)
  assert schedule.command_at(5.0) == (0.0, 0.0, 0.0)


def test_commands_tile_lanes_over_replicas():
  profile = Profile(
    lanes=(
      Lane("a", ("vx",), (Segment(vx=0.3, hold=1.0, ramp=0.0),)),
      Lane("b", ("wz",), (Segment(wz=0.5, hold=1.0, ramp=0.0),)),
    ),
    replicas=3,
  )

  schedule = profile.commands(dt=0.1)

  assert schedule.shape == (10, 6, 3)
  assert profile.lane_of_env() == ("a", "a", "a", "b", "b", "b")
  # Every replica of a lane gets that lane's command, and only that one.
  assert torch.equal(schedule[0, :3], torch.tensor([[0.3, 0.0, 0.0]]).repeat(3, 1))
  assert torch.equal(schedule[0, 3:], torch.tensor([[0.0, 0.0, 0.5]]).repeat(3, 1))


def test_omnidirectional_profile_visits_both_signs_of_every_axis():
  profile = omnidirectional_profile(ProfileCfg(replicas=1))
  schedule = profile.commands(dt=0.1)

  assert [lane.name for lane in profile.lanes][:3] == [
    "sagittal",
    "lateral",
    "turning",
  ]
  for axis, index in (("vx", 0), ("vy", 1), ("wz", 2)):
    column = schedule[:, :, index]
    assert column.max() > 0.0, axis
    assert column.min() < 0.0, axis
  # The three single-axis lanes move one axis each.
  for lane_index, axis_index in enumerate((0, 1, 2)):
    moved = (schedule[:, lane_index].abs() > 1e-6).any(dim=0)
    assert bool(moved[axis_index])
    assert int(moved.sum()) == 1


def _state(vx: float, wz: float, upright: float = 1.0) -> EvalState:
  angle = math.acos(upright)
  return EvalState(
    position_w=torch.zeros(1, 3),
    quaternion_w=torch.tensor(
      [[math.cos(0.5 * angle), 0.0, math.sin(0.5 * angle), 0.0]]
    ),
    lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]),
    ang_vel_b=torch.tensor([[0.0, 0.0, wz]]),
  )


def test_velocity_trace_keeps_command_and_response_aligned():
  trace = VelocityTrace(dt=DT)
  for step in range(3):
    command = torch.tensor([[0.1 * step, 0.0, 0.0]])
    trace.record(command, _state(vx=0.05 * step, wz=0.0))

  data = trace.result()

  assert data["time"].tolist() == pytest.approx([DT, 2 * DT, 3 * DT])
  assert data["command"].shape == (3, 1, 3)
  assert data["command"][:, 0, 0].tolist() == pytest.approx([0.0, 0.1, 0.2])
  assert data["achieved"][:, 0, 0].tolist() == pytest.approx([0.0, 0.05, 0.1])
  assert data["upright"].shape == (3, 1)


def test_velocity_trace_csv_has_one_row_per_step_per_env(tmp_path):
  trace = VelocityTrace(dt=DT)
  command = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.4]])
  state = EvalState(
    position_w=torch.zeros(2, 3),
    quaternion_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    lin_vel_b=torch.zeros(2, 3),
    ang_vel_b=torch.zeros(2, 3),
  )
  for _ in range(4):
    trace.record(command, state)

  write_trace_csv(tmp_path / "trace.csv", trace)
  lines = (tmp_path / "trace.csv").read_text().splitlines()

  assert lines[0].startswith("step,time,env,command_vx")
  assert len(lines) == 1 + 4 * 2
