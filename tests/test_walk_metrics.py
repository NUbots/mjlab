"""Tests for the shared walk metrics.

Driven by hand-built trajectories rather than by the simulator, so a metric that
drifts shows up here as an arithmetic disagreement rather than as a physics
question.
"""

import json
import math

import pytest
import torch

from mjlab.evaluation.metrics import (
  EvalState,
  WalkMetrics,
  format_summary,
  save_run,
  summarise,
  upright_from_quat,
)

UPRIGHT = (1.0, 0.0, 0.0, 0.0)
DT = 0.01


def quat_pitched(angle: float) -> tuple[float, float, float, float]:
  """A ``(w, x, y, z)`` quaternion pitched forward by ``angle`` radians."""
  return (math.cos(0.5 * angle), 0.0, math.sin(0.5 * angle), 0.0)


def state(
  position=(0.0, 0.0, 0.5),
  quaternion=UPRIGHT,
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  feet=None,
) -> EvalState:
  """One environment's state, from plain tuples."""
  return EvalState(
    position_w=torch.tensor([position]),
    quaternion_w=torch.tensor([quaternion]),
    lin_vel_b=torch.tensor([lin_vel]),
    ang_vel_b=torch.tensor([ang_vel]),
    foot_pos_w=None if feet is None else torch.tensor([feet]),
  )


def test_upright_from_quat_matches_the_rotation_matrix():
  for angle in (0.0, 0.3, 1.2, math.pi / 2, math.pi):
    expected = math.cos(angle)
    assert float(upright_from_quat(torch.tensor([quat_pitched(angle)]))[0]) == (
      pytest.approx(expected, abs=1e-6)
    )


def test_steady_walk_recovers_the_commanded_velocity():
  """Constant velocity in, the same velocity out, and no fall."""
  command = torch.tensor([[0.3, -0.1, 0.2]])
  metrics = WalkMetrics(command, dt=DT)
  metrics.start(state())
  for step in range(100):
    metrics.record(
      state(
        position=(0.3 * (step + 1) * DT, 0.0, 0.5),
        lin_vel=(0.3, -0.1, 0.0),
        ang_vel=(0.0, 0.0, 0.2),
      )
    )

  result = metrics.result()
  assert float(result.survived[0]) == 1.0
  assert math.isnan(float(result.fall_time[0]))
  assert float(result.alive_time[0]) == pytest.approx(1.0)
  assert float(result.achieved_vx[0]) == pytest.approx(0.3, abs=1e-5)
  assert float(result.achieved_vy[0]) == pytest.approx(-0.1, abs=1e-5)
  assert float(result.achieved_wz[0]) == pytest.approx(0.2, abs=1e-5)
  assert float(result.tracking_error[0]) == pytest.approx(0.0, abs=1e-5)
  assert float(result.displacement_x[0]) == pytest.approx(0.3, abs=1e-5)
  assert float(result.path_speed[0]) == pytest.approx(0.3, abs=1e-4)


def test_fall_is_dated_to_the_step_it_happens():
  """The threshold crossing sets the fall time, and stops accumulation."""
  command = torch.tensor([[0.5, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT)
  metrics.start(state())
  for _ in range(30):
    metrics.record(state(lin_vel=(0.5, 0.0, 0.0)))
  # Tipped past 60 degrees: upright drops below the threshold.
  for _ in range(70):
    metrics.record(state(quaternion=quat_pitched(2.0), lin_vel=(9.0, 0.0, 0.0)))

  result = metrics.result()
  assert float(result.survived[0]) == 0.0
  # 30 upright samples, then the 31st is the one that tipped.
  assert float(result.fall_time[0]) == pytest.approx(31 * DT)
  assert float(result.alive_time[0]) == pytest.approx(31 * DT)
  # The 9 m/s slide after the fall must not enter the achieved speed: 30
  # samples at 0.5 plus the one tipped sample at 9.0.
  assert float(result.achieved_vx[0]) == pytest.approx((30 * 0.5 + 9.0) / 31, abs=1e-4)
  assert float(result.min_upright[0]) == pytest.approx(math.cos(2.0), abs=1e-5)


def test_environments_are_measured_independently():
  """One environment falling does not disturb its neighbours."""
  command = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT)
  zeros = torch.zeros(2, 3)

  def two_env_state(step: int) -> EvalState:
    fallen = step >= 10
    return EvalState(
      position_w=torch.tensor(
        [[0.2 * step * DT, 0.0, 0.5], [0.2 * step * DT, 0.0, 0.5]]
      ),
      quaternion_w=torch.tensor(
        [list(UPRIGHT), list(quat_pitched(2.0 if fallen else 0.0))]
      ),
      lin_vel_b=torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
      ang_vel_b=zeros,
    )

  metrics.start(two_env_state(0))
  for step in range(1, 51):
    metrics.record(two_env_state(step))

  result = metrics.result()
  assert result.survived.tolist() == [1.0, 0.0]
  assert math.isnan(float(result.fall_time[0]))
  assert float(result.fall_time[1]) == pytest.approx(10 * DT)
  assert float(result.alive_time[0]) == pytest.approx(50 * DT)


def test_cadence_counts_foot_swaps_outside_the_dead_band():
  """Four alternations in one second is two steps per second."""
  command = torch.tensor([[0.2, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT)
  low, high = 0.0, 0.1
  metrics.start(state(feet=((0.0, 0.1, low), (0.0, -0.1, high))))
  for step in range(100):
    # Swap every 25 samples: 3 swaps in 100 samples of 0.01 s.
    left_down = (step // 25) % 2 == 0
    feet = (
      (0.0, 0.1, low if left_down else high),
      (0.0, -0.1, high if left_down else low),
    )
    metrics.record(state(feet=feet))

  result = metrics.result()
  assert float(result.cadence_hz[0]) == pytest.approx(3.0, abs=1e-6)


def test_cadence_is_nan_without_foot_positions():
  metrics = WalkMetrics(torch.tensor([[0.2, 0.0, 0.0]]), dt=DT)
  metrics.start(state())
  metrics.record(state())
  assert math.isnan(float(metrics.result().cadence_hz[0]))


def test_summary_separates_survivors_from_the_whole_batch():
  command = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT)
  zeros = torch.zeros(2, 3)

  for _ in range(50):
    metrics.record(
      EvalState(
        position_w=torch.zeros(2, 3),
        quaternion_w=torch.tensor([list(UPRIGHT), list(quat_pitched(2.0))]),
        lin_vel_b=torch.tensor([[0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        ang_vel_b=zeros,
      )
    )

  summary = summarise(metrics.result())
  assert summary["num_envs"] == 2
  assert summary["num_survived"] == 1
  assert summary["survival_rate"] == pytest.approx(0.5)
  # The survivor's speed, not the average of the survivor and the slider.
  assert summary["survivors"]["achieved_vx"]["mean"] == pytest.approx(0.2, abs=1e-5)
  assert summary["all_envs"]["achieved_vx"]["mean"] == pytest.approx(2.6, abs=1e-3)
  assert "survived" in format_summary(summary)


def test_save_run_writes_a_row_per_env_and_a_summary(tmp_path):
  command = torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT)
  metrics.start(
    EvalState(
      position_w=torch.zeros(2, 3),
      quaternion_w=torch.tensor([list(UPRIGHT)] * 2),
      lin_vel_b=torch.zeros(2, 3),
      ang_vel_b=torch.zeros(2, 3),
    )
  )
  for _ in range(10):
    metrics.record(
      EvalState(
        position_w=torch.zeros(2, 3),
        quaternion_w=torch.tensor([list(UPRIGHT)] * 2),
        lin_vel_b=torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        ang_vel_b=torch.zeros(2, 3),
      )
    )

  result = metrics.result()
  summary = save_run(tmp_path, {"engine": "test"}, result)

  lines = (tmp_path / "per_env.csv").read_text().strip().split("\n")
  assert len(lines) == 3
  assert lines[0].split(",") == ["env"] + result.column_names()

  written = json.loads((tmp_path / "summary.json").read_text())
  assert written["run"]["engine"] == "test"
  # Compared key by key rather than as a whole: fall_time is NaN when nothing
  # falls, and NaN never equals itself.
  assert written["num_envs"] == summary["num_envs"]
  assert written["survival_rate"] == summary["survival_rate"]
  assert written["survivors"]["achieved_vx"] == summary["survivors"]["achieved_vx"]
  assert math.isnan(written["fall_time"]["mean"])


def test_warmup_keeps_the_run_up_out_of_the_averages():
  """The first second is a standing start; only the plateau is measured."""
  command = torch.tensor([[0.3, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT, warmup_s=1.0)
  metrics.start(state())

  position = 0.0
  for step in range(200):
    speed = 0.0 if step < 100 else 0.3
    position += speed * DT
    metrics.record(state(position=(position, 0.0, 0.5), lin_vel=(speed, 0.0, 0.0)))

  result = metrics.result()

  assert float(result.achieved_vx[0]) == pytest.approx(0.3, abs=1e-6)
  assert float(result.error_vx[0]) == pytest.approx(0.0, abs=1e-6)
  # Displacement is rebased onto the measured window, not the whole run.
  assert float(result.displacement_x[0]) == pytest.approx(0.3, abs=1e-6)
  assert float(result.path_speed[0]) == pytest.approx(0.3, abs=1e-3)
  # Survival is not windowed.
  assert float(result.alive_time[0]) == pytest.approx(2.0)
  assert float(result.survived[0]) == 1.0


def test_a_fall_inside_the_warmup_leaves_no_measurement():
  command = torch.tensor([[0.3, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT, warmup_s=1.0)
  metrics.start(state())

  for step in range(200):
    fallen = step >= 50
    metrics.record(
      state(
        quaternion=quat_pitched(2.0 if fallen else 0.0),
        lin_vel=(0.0 if fallen else 0.3, 0.0, 0.0),
      )
    )

  result = metrics.result()

  assert float(result.survived[0]) == 0.0
  assert float(result.fall_time[0]) == pytest.approx(0.51)
  # Zeros here would read as "walked at 0 m/s", which it did not do.
  assert math.isnan(float(result.achieved_vx[0]))
  assert math.isnan(float(result.rms_roll[0]))
  assert math.isnan(float(result.path_speed[0]))


def test_zero_warmup_is_the_old_behaviour():
  command = torch.tensor([[0.2, 0.0, 0.0]])
  windowed = WalkMetrics(command, dt=DT, warmup_s=0.0)
  plain = WalkMetrics(command, dt=DT)
  for metrics in (windowed, plain):
    metrics.start(state())
    for step in range(50):
      metrics.record(
        state(position=(0.01 * step, 0.0, 0.5), lin_vel=(0.01 * step, 0.0, 0.0))
      )

  for name in ("achieved_vx", "path_speed", "rms_pitch", "alive_time"):
    assert float(getattr(windowed.result(), name)[0]) == pytest.approx(
      float(getattr(plain.result(), name)[0])
    )
