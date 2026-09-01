"""Tests for the push battery, its driver and its metrics.

No simulator: the battery layout is arithmetic, the driver writes a wrench onto
a stub, and the metrics are a recorder, so all three are checked against
hand-built numbers. The harness side is covered in ``test_eval_harness.py``.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from mjlab.evaluation.metrics import EvalState
from mjlab.evaluation.push import (
  PerEnvPushMetrics,
  PushCfg,
  PushDriver,
  PushMetrics,
  concat_push_metrics,
  push_battery,
  push_envelope,
  push_plan,
  summarise_push,
)

DT = 0.01
MASS = 6.681


def small_cfg(**overrides) -> PushCfg:
  """A battery small enough to reason about by hand."""
  fields = {
    "vx": 0.2,
    "delta_v": (0.2, 0.4),
    "directions": 4,
    "phases": 2,
    "replicas": 1,
    "duration": 0.2,
    "settle": 1.0,
    "phase_window": 0.4,
    "recovery": 1.0,
    "smooth": 0.02,
    "hold": 0.1,
  }
  fields.update(overrides)
  return PushCfg(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The battery
# --------------------------------------------------------------------------


def test_battery_is_one_pass_per_magnitude_all_the_same_size():
  """Refining the magnitude axis must not grow the batch."""
  cfg = small_cfg(delta_v=(0.2, 0.4, 0.6))
  plans = push_battery(cfg, MASS, DT)

  assert len(plans) == 3
  assert cfg.trials_per_pass == 4 * 2 * 1
  assert cfg.num_trials == 3 * 8
  assert {plan.num_envs for plan in plans} == {cfg.trials_per_pass}
  assert [float(plan.delta_v[0]) for plan in plans] == pytest.approx([0.2, 0.4, 0.6])


def test_magnitude_is_the_free_body_velocity_change():
  """The force is whatever delivers that impulse over the held duration."""
  plan = push_plan(small_cfg(), delta_v=0.5, mass=MASS, dt=DT)

  assert float(plan.impulse[0]) == pytest.approx(MASS * 0.5)
  assert float(plan.force[0]) == pytest.approx(MASS * 0.5 / (plan.hold_steps * DT))
  assert plan.hold_steps == 20


def test_trials_cover_every_direction_and_every_phase():
  cfg = small_cfg(directions=4, phases=2, replicas=3)
  plan = push_plan(cfg, delta_v=0.3, mass=MASS, dt=DT)

  headings = sorted({round(float(value), 6) for value in plan.heading})
  assert headings == pytest.approx([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
  # Onsets are spread across the phase window, which starts when settling ends.
  onsets = sorted({int(value) for value in plan.push_step})
  assert onsets == [100, 120]
  # Every (direction, phase) pair gets its replicas, and nothing else does.
  pairs = torch.stack((plan.heading, plan.push_step.float()), dim=-1)
  assert torch.unique(pairs, dim=0).shape[0] == 4 * 2
  assert plan.num_envs == 4 * 2 * 3


def test_the_run_is_long_enough_for_every_trial_to_finish_its_window():
  """The latest push still has a whole recovery window inside the run."""
  cfg = small_cfg()
  plan = push_plan(cfg, delta_v=0.3, mass=MASS, dt=DT)

  latest = int(plan.push_step.max())
  assert latest + plan.recovery_steps <= plan.num_steps


def test_a_battery_rejects_nonsense():
  with pytest.raises(ValueError, match="magnitudes"):
    PushCfg(delta_v=())
  with pytest.raises(ValueError, match="positive"):
    PushCfg(delta_v=(0.0, 0.4))
  with pytest.raises(ValueError, match="directions"):
    PushCfg(directions=0)


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


@dataclass
class _StubData:
  root_link_quat_w: torch.Tensor


class _StubRobot:
  """Just enough of an Entity to record what a driver writes to it."""

  def __init__(self, yaw: torch.Tensor) -> None:
    self.data = _StubData(_yaw_quat(yaw))
    self.written: list[torch.Tensor] = []

  def write_external_wrench_to_sim(self, forces, torques, body_ids=None):
    assert torch.count_nonzero(torques) == 0
    self.written.append(forces.clone())


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  zeros = torch.zeros_like(half)
  return torch.stack((half.cos(), zeros, zeros, half.sin()), dim=-1)


def _driver(cfg: PushCfg, yaw: float = 0.0):
  plan = push_plan(cfg, delta_v=0.5, mass=MASS, dt=DT)
  robot = _StubRobot(torch.full((plan.num_envs,), yaw))
  return plan, robot, PushDriver(plan, robot, body_id=1)  # type: ignore[arg-type]


def test_the_force_is_on_only_for_the_push_and_carries_the_whole_impulse():
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan, robot, driver = _driver(cfg)
  for step in range(plan.num_steps):
    driver.apply(step)
  driver.clear()

  written = torch.stack(robot.written)[:, 0, 0]  # (T, 3)
  live = torch.linalg.vector_norm(written, dim=-1) > 0.0
  assert int(live.sum()) == plan.hold_steps
  assert bool(live[int(plan.push_step[0])])
  assert not bool(live[int(plan.push_step[0]) - 1])
  # Force integrated over the steps it was held is the impulse that was asked
  # for, which is the whole point of parameterising the battery this way.
  delivered = written.norm(dim=-1).sum() * DT
  assert float(delivered) == pytest.approx(float(plan.impulse[0]), rel=1e-5)
  # And the last write is the clear, so nothing leaks into the next run.
  assert float(torch.linalg.vector_norm(robot.written[-1])) == 0.0


def test_the_direction_is_the_heading_in_the_robots_own_yaw_frame():
  """A push at heading zero shoves the robot along its own forward axis."""
  cfg = small_cfg(directions=4, phases=1, replicas=1)
  plan, robot, driver = _driver(cfg, yaw=math.pi / 2)
  driver.apply(int(plan.push_step[0]))

  force = robot.written[0][:, 0]
  magnitude = float(plan.force[0])
  # The robot faces +y, so its own forward is the world's +y.
  assert force[0].tolist() == pytest.approx([0.0, magnitude, 0.0], abs=1e-4)
  # ... and a push to its left is the world's -x.
  assert force[1].tolist() == pytest.approx([-magnitude, 0.0, 0.0], abs=1e-4)


def test_the_direction_is_latched_at_onset_not_steered_after_it():
  """A shove comes from somewhere; it does not follow a falling robot."""
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan, robot, driver = _driver(cfg, yaw=0.0)
  onset = int(plan.push_step[0])
  driver.apply(onset)
  robot.data.root_link_quat_w = _yaw_quat(torch.full((plan.num_envs,), math.pi / 2))
  driver.apply(onset + 1)

  assert torch.equal(robot.written[0], robot.written[1])


# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------


def _state(
  vx: torch.Tensor, upright: torch.Tensor, yaw: torch.Tensor | None = None
) -> EvalState:
  """A batch of robots leaning by ``acos(upright)`` and moving at ``vx``."""
  num = vx.shape[0]
  pitch = torch.acos(upright.clamp(-1.0, 1.0))
  yaw = torch.zeros(num) if yaw is None else yaw
  # A pitch-then-yaw rotation, built as the product of the two quaternions.
  half_pitch, half_yaw = 0.5 * pitch, 0.5 * yaw
  zeros = torch.zeros(num)
  quat = torch.stack(
    (
      half_yaw.cos() * half_pitch.cos(),
      -half_yaw.sin() * half_pitch.sin(),
      half_yaw.cos() * half_pitch.sin(),
      half_yaw.sin() * half_pitch.cos(),
    ),
    dim=-1,
  )
  return EvalState(
    position_w=torch.zeros(num, 3),
    quaternion_w=quat,
    lin_vel_b=torch.stack((vx, zeros, zeros), dim=-1),
    ang_vel_b=torch.zeros(num, 3),
  )


def _run(plan, speeds, uprights, yaws=None) -> PerEnvPushMetrics:
  """Play a hand-written history through the recorder."""
  metrics = PushMetrics(plan)
  metrics.start(_state(speeds[0], uprights[0]))
  for step in range(plan.num_steps):
    yaw = None if yaws is None else yaws[step]
    metrics.record(_state(speeds[step], uprights[step], yaw))
  return metrics.result()


def _flat(plan, value: float) -> list[torch.Tensor]:
  return [torch.full((plan.num_envs,), value) for _ in range(plan.num_steps + 1)]


def test_a_robot_that_never_leaves_the_band_withstood_the_push():
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan = push_plan(cfg, delta_v=0.2, mass=MASS, dt=DT)

  result = _run(plan, _flat(plan, 0.2), _flat(plan, 1.0))

  assert result.withstood.tolist() == [1.0]
  assert result.recovered.tolist() == [1.0]
  assert result.fell_before_push.tolist() == [0.0]
  assert float(result.recovery_time[0]) == pytest.approx(0.0)
  assert float(result.peak_speed_error[0]) == pytest.approx(0.0, abs=1e-6)
  assert math.isnan(float(result.time_to_fall[0]))


def test_recovery_is_dated_from_the_onset_and_needs_the_hold():
  """Knocked off command at the push, back on it a known time later."""
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan = push_plan(cfg, delta_v=0.2, mass=MASS, dt=DT)
  onset = int(plan.push_step[0])

  speeds = _flat(plan, 0.2)
  for step in range(onset, onset + 30):
    speeds[step] = torch.full((plan.num_envs,), 0.9)
  result = _run(plan, speeds, _flat(plan, 1.0))

  # Two smoothing steps of lag on a 0.01 s control step, then the error is
  # inside the band and stays there.
  assert float(result.recovery_time[0]) == pytest.approx(0.30, abs=0.02)
  assert float(result.peak_speed_error[0]) == pytest.approx(0.7, abs=0.01)
  assert result.recovered.tolist() == [1.0]


def test_a_push_that_topples_the_robot_is_not_withstood():
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan = push_plan(cfg, delta_v=0.8, mass=MASS, dt=DT)
  onset = int(plan.push_step[0])

  uprights = _flat(plan, 1.0)
  for step in range(onset + 20, plan.num_steps + 1):
    uprights[step] = torch.full((plan.num_envs,), 0.1)
  result = _run(plan, _flat(plan, 0.2), uprights)

  assert result.withstood.tolist() == [0.0]
  assert float(result.time_to_fall[0]) == pytest.approx(0.21, abs=0.02)
  assert float(result.min_upright_after[0]) == pytest.approx(0.1)
  # A robot on the floor is not "back on command" however still it lies.
  assert result.recovered.tolist() == [0.0]
  assert math.isnan(float(result.recovery_time[0]))


def test_a_robot_already_down_when_the_push_lands_is_reported_as_missing():
  """It measures the command, not the push, so it cannot count either way."""
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan = push_plan(cfg, delta_v=0.2, mass=MASS, dt=DT)

  uprights = _flat(plan, 1.0)
  for step in range(10, plan.num_steps + 1):
    uprights[step] = torch.full((plan.num_envs,), 0.0)
  result = _run(plan, _flat(plan, 0.2), uprights)

  assert result.fell_before_push.tolist() == [1.0]
  assert math.isnan(float(result.withstood[0]))
  assert math.isnan(float(result.recovered[0]))
  assert result.survived.tolist() == [0.0], "the walking metrics still see it"


def test_heading_error_is_the_yaw_the_push_stole():
  cfg = small_cfg(directions=1, phases=1, replicas=1)
  plan = push_plan(cfg, delta_v=0.2, mass=MASS, dt=DT)
  onset = int(plan.push_step[0])

  yaws = [torch.zeros(plan.num_envs) for _ in range(plan.num_steps + 1)]
  for step in range(onset, plan.num_steps + 1):
    yaws[step] = torch.full((plan.num_envs,), 0.25)
  result = _run(plan, _flat(plan, 0.2), _flat(plan, 1.0), yaws)

  assert float(result.heading_error[0]) == pytest.approx(0.25, abs=1e-4)


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------


def _table(headings, magnitudes, withstood) -> PerEnvPushMetrics:
  num = len(headings)
  zeros = torch.zeros(num)
  values = {name: zeros.clone() for name in PerEnvPushMetrics.__dataclass_fields__}
  values["push_heading_deg"] = torch.tensor(headings, dtype=torch.float32)
  values["push_delta_v"] = torch.tensor(magnitudes, dtype=torch.float32)
  values["push_impulse"] = values["push_delta_v"] * MASS
  values["withstood"] = torch.tensor(withstood, dtype=torch.float32)
  values["recovered"] = values["withstood"].clone()
  values["fell_before_push"] = torch.zeros(num)
  return PerEnvPushMetrics(**values)


def test_the_envelope_interpolates_the_crossing_rather_than_rounding_to_a_step():
  """Survival 0.75 then 0.25 across one step puts the edge in the middle."""
  table = _table(
    headings=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    magnitudes=[0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],
    withstood=[1, 1, 1, 0, 1, 0, 0, 0],
  )

  (entry,) = push_envelope(table)

  assert entry["heading_deg"] == 0.0
  assert entry["survival"] == pytest.approx([0.75, 0.25])
  assert entry["critical_delta_v"] == pytest.approx(0.3)
  assert entry["critical_impulse"] == pytest.approx(0.3 * MASS)
  assert entry["crossed"]


def test_a_direction_that_survives_everything_is_reported_as_open():
  """Reporting the largest magnitude tested would report the battery, not the
  controller."""
  table = _table(headings=[0.0, 0.0], magnitudes=[0.2, 0.4], withstood=[1, 1])

  (entry,) = push_envelope(table)

  assert not entry["crossed"]
  assert math.isnan(entry["critical_delta_v"])


def test_a_direction_that_fails_at_once_interpolates_back_towards_no_push():
  table = _table(headings=[0.0, 0.0], magnitudes=[0.4, 0.8], withstood=[0, 0])

  (entry,) = push_envelope(table)

  assert entry["critical_delta_v"] == pytest.approx(0.2)


def test_the_summary_keeps_the_shape_every_other_run_writes():
  table = _table(
    headings=[0.0, 0.0, 180.0, 180.0],
    magnitudes=[0.2, 0.4, 0.2, 0.4],
    withstood=[1, 0, 1, 1],
  )

  summary = summarise_push(table, small_cfg())

  assert {"num_envs", "num_survived", "survivors", "all_envs", "push"} <= set(summary)
  assert "recovery_time" in summary["survivors"]
  assert summary["push"]["num_trials"] == 4
  assert summary["push"]["withstood_rate"] == pytest.approx(0.75)
  assert len(summary["push"]["envelope"]) == 2
  assert summary["push"]["trials_per_cell"] == small_cfg().trials_per_cell


def test_passes_concatenate_into_one_table_of_trials():
  first = _table(headings=[0.0], magnitudes=[0.2], withstood=[1])
  second = _table(headings=[0.0], magnitudes=[0.4], withstood=[0])

  joined = concat_push_metrics([first, second])

  assert joined.push_delta_v.tolist() == pytest.approx([0.2, 0.4])
  assert joined.column_names()[:3] == ["command_vx", "command_vy", "command_wz"]
  assert "withstood" in joined.column_names()
  assert len(joined.rows()) == 2
