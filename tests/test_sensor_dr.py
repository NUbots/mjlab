"""Unit tests for IMU-mounting and gravity-incline DR (mdp/sensor_dr.py)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity.mdp.sensor_dr import (
  _random_small_rotations,
  gravity_incline,
  imu_base_ang_vel,
  imu_mounting_bias,
)


def test_random_rotations_are_valid_and_bounded() -> None:
  torch.manual_seed(0)
  max_rad = math.radians(2.5)
  rots = _random_small_rotations(256, max_rad, torch.device("cpu"))
  eye = torch.eye(3).expand(256, 3, 3)
  # Orthonormal with determinant +1.
  torch.testing.assert_close(
    torch.bmm(rots, rots.transpose(1, 2)), eye, atol=1e-5, rtol=0
  )
  # Rotation angle bounded by the max: trace = 1 + 2 cos(angle).
  traces = rots.diagonal(dim1=1, dim2=2).sum(dim=1)
  angles = torch.acos(((traces - 1.0) / 2.0).clamp(-1, 1))
  assert float(angles.max()) <= max_rad + 1e-5
  assert float(angles.std()) > 0.001  # Actually random.


def test_mounting_bias_rotates_actor_gyro(monkeypatch) -> None:
  torch.manual_seed(1)
  n = 8
  gyro = torch.randn(n, 3)
  import mjlab.tasks.velocity.mdp.sensor_dr as sensor_dr

  monkeypatch.setattr(sensor_dr, "builtin_sensor", lambda env, sensor_name: gyro)
  env = cast(Any, SimpleNamespace(num_envs=n, device=torch.device("cpu")))
  # No event yet: passthrough.
  torch.testing.assert_close(imu_base_ang_vel(env, "robot/imu_ang_vel"), gyro)
  imu_mounting_bias(env, env_ids=None, max_angle_deg=2.5)
  mounted = imu_base_ang_vel(env, "robot/imu_ang_vel")
  assert not torch.allclose(mounted, gyro)
  # Rotation preserves magnitude.
  torch.testing.assert_close(mounted.norm(dim=1), gyro.norm(dim=1), atol=1e-5, rtol=0)


def test_gravity_incline_tilts_within_bounds() -> None:
  torch.manual_seed(2)
  n = 64
  gravity = torch.tensor([[0.0, 0.0, -9.81]]).repeat(n, 1)
  env = cast(
    Any,
    SimpleNamespace(
      num_envs=n,
      device=torch.device("cpu"),
      sim=SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity))),
    ),
  )
  gravity_incline(env, env_ids=None, max_angle_deg=2.5)
  g = env.sim.model.opt.gravity
  # Magnitude preserved.
  torch.testing.assert_close(g.norm(dim=1), torch.full((n,), 9.81), atol=1e-4, rtol=0)
  # Tilt angle within bounds, nonzero spread.
  cos_tilt = (-g[:, 2] / 9.81).clamp(-1, 1)
  tilt = torch.acos(cos_tilt)
  assert float(tilt.max()) <= math.radians(2.5) + 1e-4
  assert float(tilt.std()) > 1e-4
  # Re-running rotates the NOMINAL (no compounding drift of magnitude).
  gravity_incline(env, env_ids=None, max_angle_deg=2.5)
  torch.testing.assert_close(
    env.sim.model.opt.gravity.norm(dim=1), torch.full((n,), 9.81), atol=1e-4, rtol=0
  )
