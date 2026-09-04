"""Tests for the motion-capture profile plotter.

No simulator and no real log: a synthetic capture is written to a temporary
file, with a known robot moving through a known schedule, expressed in a
capture frame deliberately made awkward -- rotated away from the floor, with
the tracked body's axes rotated away from the torso's, and optionally mirrored.
The script has to recover the robot's own velocities from that, which is the
part of it worth testing.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from plot_mocap_profile import (  # noqa: E402
  STRIDE_WINDOWS,
  Frame,
  build_run,
  calibrate,
  gait_period,
  local_slope,
  read_log,
)

MOCAP_HZ = 120.0
IMU_HZ = 100.0
WALK_HZ = 90.0
STRIDE_HZ = 1.5
SWAY = 0.25
"""Lateral sway amplitude, in m/s -- larger than any command, as on the robot."""

SCHEDULE = (
  (4.0, (0.0, 0.0, 0.0)),
  (8.0, (0.30, 0.0, 0.0)),
  (4.0, (0.0, 0.0, 0.0)),
  (8.0, (0.0, 0.20, 0.0)),
  (4.0, (0.0, 0.0, 0.0)),
  (8.0, (0.0, 0.0, 0.50)),
  (4.0, (0.0, 0.0, 0.0)),
  (8.0, (-0.30, 0.0, 0.0)),
  (4.0, (0.0, 0.0, 0.0)),
)
LIFT_FROM, LIFT_TO = 36.6, 39.4
"""Seconds the synthetic robot spends in somebody's hands, as they all do."""


def _rot(axis: str, angle: float) -> np.ndarray:
  c, s = math.cos(angle), math.sin(angle)
  return {
    "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
    "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
    "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
  }[axis]


def _quat(matrix: np.ndarray) -> tuple[float, float, float, float]:
  """``(x, y, z, w)`` from a proper rotation matrix."""
  w = math.sqrt(max(0.0, 1.0 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2])) / 2.0
  if w < 1e-6:
    values, vectors = np.linalg.eigh(matrix + matrix.T)
    axis = vectors[:, int(np.argmax(values))]
    return (*axis, 0.0)
  return (
    (matrix[2, 1] - matrix[1, 2]) / (4 * w),
    (matrix[0, 2] - matrix[2, 0]) / (4 * w),
    (matrix[1, 0] - matrix[0, 1]) / (4 * w),
    w,
  )


def _command_at(t: float) -> tuple[float, float, float]:
  elapsed = 0.0
  for hold, value in SCHEDULE:
    if t < elapsed + hold:
      return value
    elapsed += hold
  return (0.0, 0.0, 0.0)


def _duration() -> float:
  return sum(hold for hold, _ in SCHEDULE)


def _commanded_spans() -> list[tuple[float, float]]:
  """``(start, end)`` of every non-zero stretch of the schedule."""
  spans, elapsed = [], 0.0
  for hold, value in SCHEDULE:
    if any(value):
      spans.append((elapsed, elapsed + hold))
    elapsed += hold
  return spans


def _settled_mean(run, start: float, end: float) -> np.ndarray:
  """Mean measured velocity over the second half of a commanded stretch.

  The fit window straddles the step at the stretch's start, so the opening of
  one is still catching up and is not steady state.
  """
  window = (run.t >= start + 0.5 * (end - start)) & (run.t <= end) & run.walked
  return run.smooth[window].mean(0)


def write_capture(path: Path, mirrored: bool = False, lift: bool = True) -> None:
  """A synthetic NBS-to-JSON log of a robot that tracks its command exactly.

  The robot walks in a world frame with z up, x forward. Three constant
  rotations stand between that and what the capture sees, which is the whole
  point: the tracked body's axes are not the torso's, the capture volume's
  axes are not the world's, and the capture may be mirrored.
  """
  # torso <- tracked body, capture <- world: arbitrary, and nothing may assume
  # them.
  body = _rot("z", 0.7) @ _rot("y", 0.3)
  capture = _rot("x", 0.4) @ _rot("z", 1.1)
  mirror = np.diag([1.0, -1.0, 1.0]) if mirrored else np.eye(3)

  steps = int(_duration() * MOCAP_HZ)
  yaw, position = 0.0, np.zeros(3)
  lines = []
  for step in range(steps):
    t = step / MOCAP_HZ
    vx, vy, wz = _command_at(t)
    sway = SWAY * math.sin(2 * math.pi * STRIDE_HZ * t)
    world = _rot("z", yaw) @ np.array([vx, vy + sway, 0.0])
    position = position + world / MOCAP_HZ
    yaw += wz / MOCAP_HZ
    # Being carried: up and away, and tipped over while it happens.
    carried = lift and LIFT_FROM <= t <= LIFT_TO
    height = 0.6 if carried else 0.0
    tip = _rot("y", 1.2) if carried else np.eye(3)
    torso = _rot("z", yaw) @ tip
    # A mirrored capture frame reports positions mirrored and rotations
    # conjugated: a rotation stays a proper rotation, but its sense about any
    # axis reverses. That reversal is the only trace a mirror leaves, and
    # finding it is what the handedness test is for.
    seen = mirror @ (capture @ torso @ body) @ mirror
    place = mirror @ capture @ (position + np.array([0.0, 0.0, height]))
    lines.append(
      {
        "type": "message.input.MotionCapture",
        "timestamp": int(t * 1e6),
        "data": {
          "natnetTimestamp": t,
          "rigidBodies": [
            {
              "position": dict(zip("xyz", place.tolist(), strict=True)),
              "rotation": dict(zip("xyzt", _quat(seen), strict=True)),
              "trackingValid": True,
            }
          ],
        },
      }
    )

  for step in range(int(_duration() * WALK_HZ)):
    t = step / WALK_HZ
    lines.append(
      {
        "type": "message.behaviour.state.WalkState",
        "timestamp": int(t * 1e6),
        "data": {"velocityTarget": dict(zip("xyz", _command_at(t), strict=True))},
      }
    )
  for step in range(int(_duration() * IMU_HZ)):
    t = step / IMU_HZ
    _, _, wz = _command_at(t)
    lines.append(
      {
        "type": "message.input.Sensors",
        "timestamp": int(t * 1e6),
        "data": {
          # The IMU is in the torso frame: gravity along +z, yaw rate about it.
          "gyroscope": {"x": 0.0, "y": 0.0, "z": wz},
          "accelerometer": {"x": 0.0, "y": 0.0, "z": 9.81},
        },
      }
    )

  lines.sort(key=lambda message: message["timestamp"])
  path.write_text("\n".join(json.dumps(message) for message in lines) + "\n")


@pytest.fixture
def capture(tmp_path):
  path = tmp_path / "synthetic.json"
  write_capture(path)
  return path


# --------------------------------------------------------------------------
# Differentiating an irregular track
# --------------------------------------------------------------------------


def test_local_slope_recovers_a_known_rate():
  t = np.linspace(0.0, 10.0, 1200)
  slope = local_slope(t, 0.3 * t + 1.0, 0.5)
  finite = np.isfinite(slope)

  assert finite.mean() > 0.95
  assert slope[finite] == pytest.approx(0.3, abs=1e-6)


def test_the_slope_window_that_cancels_a_wobble_is_not_one_period():
  """The whole reason the window is 1.43 strides and not one.

  Measured away from the ends, where the window runs off the data and covers
  part of a cycle rather than all of it.
  """
  period = 1 / 1.5
  t = np.linspace(0.0, 20.0, 4000)
  x = 0.3 * t + 0.05 * np.sin(2 * np.pi * 1.5 * t)
  interior = (t > period) & (t < t[-1] - period)

  short = local_slope(t, x, 0.1)[interior]
  one = local_slope(t, x, period)[interior]
  null = local_slope(t, x, period * STRIDE_WINDOWS)[interior]

  # One period is the window that would cancel the wobble from a moving *mean*,
  # and it does not cancel it from a slope fit; 1.43 of them does.
  assert np.nanstd(one) > 0.05
  assert np.nanstd(null) < 0.002
  assert np.nanstd(short) > 100 * np.nanstd(null)
  assert np.nanmean(null) == pytest.approx(0.3, abs=0.001)


def test_local_slope_survives_dropped_frames():
  """A capture keeps under half its frames and drops them in bursts."""
  rng = np.random.default_rng(0)
  t = np.sort(rng.uniform(0.0, 10.0, 600))
  slope = local_slope(t, -0.2 * t, 0.5)
  finite = np.isfinite(slope)

  assert slope[finite] == pytest.approx(-0.2, abs=1e-6)


# --------------------------------------------------------------------------
# Reading the schedule back out of the log
# --------------------------------------------------------------------------


def test_the_commanded_stretches_are_detected_not_assumed(capture):
  """The schedule is a property of the run that produced the log, not of this
  script, so the same code has to segment a capture whose timings changed."""
  log = read_log(capture)
  frame, _ = calibrate(log, None, None, None)
  run = build_run(log, frame)

  expected = _commanded_spans()
  assert len(run.driven) == len(expected)
  for (start, end), (wanted_start, wanted_end) in zip(
    run.driven, expected, strict=True
  ):
    assert start == pytest.approx(wanted_start, abs=0.05)
    assert end == pytest.approx(wanted_end, abs=0.05)


def test_uninitialised_commands_are_dropped(tmp_path):
  """A log opens with whatever was on the walk engine's stack."""
  path = tmp_path / "garbage.json"
  write_capture(path)
  rubbish = json.dumps(
    {
      "type": "message.behaviour.state.WalkState",
      "timestamp": 0,
      "data": {"velocityTarget": {"x": 5.29e-310, "y": 2.67e260, "z": 0.0}},
    }
  )
  path.write_text(rubbish + "\n" + path.read_text())

  log = read_log(path)

  assert np.abs(log.command).max() < 1.0


def test_the_stride_is_measured_from_the_sway(capture):
  log = read_log(capture)
  frame, _ = calibrate(log, None, None, None)
  run = build_run(log, frame)

  assert run.gait_s is not None
  assert run.gait_s == pytest.approx(1 / STRIDE_HZ, rel=0.05)
  assert run.smooth_s == pytest.approx(run.gait_s * STRIDE_WINDOWS)


def test_gait_period_gives_up_rather_than_guessing():
  assert gait_period(np.linspace(0, 1, 10), np.zeros(10), []) is None


# --------------------------------------------------------------------------
# The frame calibration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mirrored", [False, True])
def test_the_robot_is_recovered_from_an_awkward_capture_frame(tmp_path, mirrored):
  """The point of the whole calibration, end to end.

  The capture frame is rotated away from the floor and the tracked body's axes
  away from the torso's, and half the time the frame is mirrored as well. None
  of that may reach the numbers: a robot that walked at the command it was
  given has to come back out reporting the command it was given.
  """
  path = tmp_path / f"synthetic_{mirrored}.json"
  write_capture(path, mirrored=mirrored)

  log = read_log(path)
  frame, _ = calibrate(log, None, None, None)
  run = build_run(log, frame)

  assert frame.chirality == (-1.0 if mirrored else 1.0)
  assert frame.plane_residual < 1e-6
  assert run.driven
  for start, end in run.driven:
    commanded = _command_at(0.5 * (start + end))
    assert _settled_mean(run, start, end) == pytest.approx(commanded, abs=0.05), (
      commanded
    )


def test_being_carried_is_not_counted_as_walking(capture):
  log = read_log(capture)
  frame, _ = calibrate(log, None, None, None)
  run = build_run(log, frame)

  lifted = (run.t >= LIFT_FROM) & (run.t <= LIFT_TO)
  assert not run.walked[lifted].any()
  # And the samples either side of it, on the way into and out of the hands.
  assert run.walked.mean() > 0.7
  assert run.fall_t == pytest.approx(LIFT_FROM, abs=0.5)


def test_the_up_sign_only_decides_what_counts_as_off_its_feet(capture):
  """Flip it and the handedness flips with it; the two cancel everywhere else.

  Worth pinning, because it is what makes the weakest of the four calibration
  steps harmless: the sign is read off which way the robot leaves the floor,
  which is exactly the question it is then used to answer.
  """
  log = read_log(capture)
  upright, _ = calibrate(log, None, None, None)
  upside_down, _ = calibrate(log, ",".join(str(-v) for v in upright.up_w), None, None)

  assert upside_down.chirality == -upright.chirality
  assert upside_down.forward_b == pytest.approx(upright.forward_b, abs=1e-9)
  assert upside_down.left_b == pytest.approx(upright.left_b, abs=1e-9)

  a, b = build_run(log, upright), build_run(log, upside_down)
  assert np.nanmax(np.abs(a.smooth - b.smooth)) < 1e-6
  assert np.nanmax(np.abs(a.raw - b.raw)) < 1e-6


def test_a_pinned_frame_is_used_as_given(capture):
  log = read_log(capture)
  auto, _ = calibrate(log, None, None, None)
  pinned, _ = calibrate(log, None, "1,0,0", 1)

  assert pinned.chirality == 1.0
  assert pinned.forward_b == pytest.approx(
    _in_plane(np.array([1.0, 0.0, 0.0]), auto.up_b), abs=1e-9
  )
  assert any("command line" in note for note in pinned.notes)


def _in_plane(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
  vector = vector - up * (vector @ up)
  return vector / np.linalg.norm(vector)


def test_a_capture_with_no_imu_says_what_it_assumed(tmp_path):
  path = tmp_path / "no_imu.json"
  write_capture(path)
  kept = [
    line
    for line in path.read_text().splitlines()
    if "message.input.Sensors" not in line
  ]
  path.write_text("\n".join(kept) + "\n")

  log = read_log(path)
  frame, _ = calibrate(log, None, None, None)

  assert frame.chirality == 1.0
  assert any("no IMU" in note for note in frame.notes)


def test_a_log_without_a_tracked_body_is_refused(tmp_path):
  path = tmp_path / "empty.json"
  path.write_text(
    json.dumps(
      {
        "type": "message.behaviour.state.WalkState",
        "timestamp": 0,
        "data": {"velocityTarget": {"x": 0.1, "y": 0.0, "z": 0.0}},
      }
    )
    + "\n"
  )

  with pytest.raises(SystemExit, match="motion-capture frames"):
    read_log(path)


def test_left_is_the_other_way_round_in_a_mirrored_capture():
  """The one line the whole handedness test exists to get right."""

  def frame(chirality: float) -> Frame:
    return Frame(
      up_w=np.array([0.0, 0.0, 1.0]),
      up_b=np.array([0.0, 0.0, 1.0]),
      forward_b=np.array([1.0, 0.0, 0.0]),
      chirality=chirality,
      plane_residual=0.0,
      tilt_deg=np.zeros(1),
      height=np.zeros(1),
    )

  assert frame(1.0).left_b == pytest.approx([0.0, 1.0, 0.0])
  assert frame(-1.0).left_b == pytest.approx([0.0, -1.0, 0.0])
