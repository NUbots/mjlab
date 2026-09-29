"""Tests for the walk metrics, the push battery and the push envelope.

No simulator: the battery layout is arithmetic and the metrics are recorders,
so both are checked against hand-built numbers.
"""

import math

import pytest
import torch

from mjlab.evaluation.metrics import EvalState, WalkMetrics
from mjlab.evaluation.push import (
  PerEnvPushMetrics,
  PushCfg,
  PushMetrics,
  push_battery,
  push_envelope,
  push_plan,
)

DT = 0.02
MASS = 20.0


def _state(vx: torch.Tensor, upright: torch.Tensor) -> EvalState:
  """A batch of robots pitched by ``acos(upright)`` and moving at ``vx``."""
  half = 0.5 * torch.acos(upright.clamp(-1.0, 1.0))
  zeros = torch.zeros_like(vx)
  return EvalState(
    position_w=torch.zeros(vx.shape[0], 3),
    quaternion_w=torch.stack((half.cos(), zeros, half.sin(), zeros), dim=-1),
    lin_vel_b=torch.stack((vx, zeros, zeros), dim=-1),
    ang_vel_b=torch.zeros(vx.shape[0], 3),
  )


def test_walk_metrics_average_after_warmup_and_date_the_fall():
  command = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
  metrics = WalkMetrics(command, dt=DT, warmup_s=2 * DT)
  # Robot 0 accelerates then holds 0.4 m/s; robot 1 falls on step 3.
  for step, speed in enumerate((0.0, 0.1, 0.4, 0.4)):
    upright = torch.tensor([1.0, 0.0 if step == 2 else 1.0])
    metrics.record(_state(torch.full((2,), speed), upright))

  result = metrics.result()

  assert result.achieved_vx[0] == pytest.approx(0.4)
  assert result.error_vx[0] == pytest.approx(-0.1)
  assert result.survived.tolist() == [1.0, 0.0]
  assert math.isnan(result.fall_time[0])
  assert result.fall_time[1] == pytest.approx(3 * DT)


def small_cfg(**overrides) -> PushCfg:
  fields: dict = {
    "delta_v": (0.5, 1.0),
    "directions": 4,
    "phases": 2,
    "replicas": 1,
    "duration": 0.1,
    "settle": 1.0,
    "phase_window": 0.2,
    "recovery": 1.0,
  }
  fields.update(overrides)
  return PushCfg(**fields)


def test_battery_is_one_pass_per_magnitude_with_exact_impulse():
  cfg = small_cfg()
  plans = push_battery(cfg, MASS, DT)

  assert len(plans) == 2
  assert all(plan.num_envs == cfg.trials_per_pass == 8 for plan in plans)
  plan = plans[1]
  assert float(plan.impulse[0]) == pytest.approx(MASS * 1.0)
  assert float(plan.force[0] * plan.hold_steps * DT) == pytest.approx(MASS * 1.0)
  assert sorted(set(torch.rad2deg(plan.heading).round().tolist())) == [
    0.0,
    90.0,
    180.0,
    270.0,
  ]


def test_withstood_is_judged_from_each_trials_own_onset():
  cfg = small_cfg(directions=1, phases=1, replicas=3)
  plan = push_plan(cfg, delta_v=0.5, mass=MASS, dt=DT)
  onset = int(plan.push_step[0])
  metrics = PushMetrics(plan)
  for step in range(plan.num_steps):
    # Robot 0 stays up, robot 1 falls after the push, robot 2 falls before it.
    upright = torch.tensor(
      [1.0, 0.0 if step > onset + 5 else 1.0, 0.0 if step > onset - 5 else 1.0]
    )
    metrics.record(_state(torch.full((3,), 0.5), upright))

  result = metrics.result()

  assert result.withstood[:2].tolist() == [1.0, 0.0]
  assert math.isnan(result.withstood[2])
  assert result.fell_before_push.tolist() == [0.0, 0.0, 1.0]


def _table(magnitudes, withstood) -> PerEnvPushMetrics:
  num = len(magnitudes)
  values = {name: torch.zeros(num) for name in PerEnvPushMetrics.__dataclass_fields__}
  values["push_delta_v"] = torch.tensor(magnitudes, dtype=torch.float32)
  values["push_impulse"] = values["push_delta_v"] * MASS
  values["withstood"] = torch.tensor(withstood, dtype=torch.float32)
  return PerEnvPushMetrics(**values)


def test_envelope_interpolates_the_crossing():
  """Survival 0.75 then 0.25 across one step puts the edge in the middle."""
  table = _table([1.0] * 4 + [2.0] * 4, [1, 1, 1, 0, 1, 0, 0, 0])

  (entry,) = push_envelope(table)

  assert entry["crossed"]
  assert entry["critical_delta_v"] == pytest.approx(1.5)
  assert entry["critical_impulse"] == pytest.approx(1.5 * MASS)


def test_envelope_of_a_direction_that_survives_everything_is_open():
  (entry,) = push_envelope(_table([1.0, 2.0], [1, 1]))

  assert not entry["crossed"]
  assert math.isnan(entry["critical_delta_v"])
