"""Geometry checks for the observability metrics (duck, heel/toe, roll)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch

from mjlab.tasks.velocity.mdp.metrics import (
  foot_heel_toe_pitch_deg,
  foot_lateral_roll_deg,
  foot_toeout_deg,
  joint_speed_abs,
)
from mjlab.tasks.velocity.mdp.rewards import foot_toein_cost


def _quat_from_axis_angle(
  axis: tuple[float, float, float], angle: float
) -> list[float]:
  ax = torch.tensor(axis, dtype=torch.float32)
  ax = ax / ax.norm()
  h = angle / 2.0
  s = math.sin(h)
  return [math.cos(h), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s)]


def _make_env(*, foot_quats, torso_quat, found, command, joint_vel=None):
  """Mock env exposing exactly what the metric funcs read."""
  n = foot_quats.shape[0]
  robot = MagicMock()
  # body order: [torso, left_foot, right_foot]; asset_cfg.body_ids select feet.
  body_quats = torch.cat([torso_quat.unsqueeze(1), foot_quats], dim=1)  # [n,3,4]
  robot.data.body_link_quat_w = body_quats
  robot.data.gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3)
  if joint_vel is not None:
    robot.data.joint_vel = joint_vel
  sensor = MagicMock()
  sensor.data.found = found
  env = MagicMock()
  env.scene.__getitem__ = MagicMock(
    side_effect=lambda k: sensor if k == "feet_ground_contact" else robot
  )
  env.command_manager.get_command = MagicMock(return_value=command)
  return env, robot


def _feet_cfg():
  cfg = MagicMock()
  cfg.name = "robot"
  cfg.body_ids = [1, 2]  # left_foot, right_foot in the mock body order
  return cfg


def _torso_cfg():
  cfg = MagicMock()
  cfg.name = "robot"
  cfg.body_ids = [0]
  return cfg


def test_toeout_positive_when_toes_out():
  # Left foot yawed +15° about vertical, right foot -15°: symmetric toe-out.
  n = 4
  lq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(15))]).expand(n, 4)
  rq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(-15))]).expand(n, 4)
  feet = torch.stack([lq, rq], dim=1)  # [n,2,4]
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  env, _ = _make_env(
    foot_quats=feet,
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  out = foot_toeout_deg(env, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(out, torch.full((n,), 15.0), atol=0.5)

  # Toes-in (reverse the yaw signs) reads negative.
  feet_in = torch.stack([rq, lq], dim=1)
  env2, _ = _make_env(
    foot_quats=feet_in,
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  assert torch.allclose(
    foot_toeout_deg(env2, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg()),
    torch.full((n,), -15.0),
    atol=0.5,
  )


def test_toeout_gated_by_command():
  n = 2
  lq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(20))]).expand(n, 4)
  rq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(-20))]).expand(n, 4)
  feet = torch.stack([lq, rq], dim=1)
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  env, _ = _make_env(
    foot_quats=feet,
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.zeros(n, 3),  # standing
  )
  out = foot_toeout_deg(env, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(out, torch.zeros(n), atol=1e-4)


def _flat_foot() -> torch.Tensor:
  """A NUgus foot flat on the ground: sole normal (local X) points up.

  That is a -90° rotation about world Y (verified: gives roll=pitch=0).
  """
  return torch.tensor(_quat_from_axis_angle((0, 1, 0), -math.pi / 2))


def _rotate(base: torch.Tensor, axis, angle_deg: float) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_mul

  extra = torch.tensor([_quat_from_axis_angle(axis, math.radians(angle_deg))])
  return quat_mul(extra, base.unsqueeze(0))[0]


def test_heel_toe_and_roll_axes():
  # From flat (X up), a world-Y rotation reads as pitch (fore-aft), a
  # world-X rotation reads as roll (medial-lateral). Verified empirically.
  n = 3
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  flat = _flat_foot()
  pitched = _rotate(flat, (0, 1, 0), 10.0).expand(n, 4)
  rolled = _rotate(flat, (1, 0, 0), 10.0).expand(n, 4)

  env_p, _ = _make_env(
    foot_quats=torch.stack([pitched, pitched], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  assert torch.allclose(
    foot_heel_toe_pitch_deg(env_p, "feet_ground_contact", asset_cfg=_feet_cfg()),
    torch.full((n,), 10.0),
    atol=0.5,
  )
  assert torch.all(
    foot_lateral_roll_deg(env_p, "feet_ground_contact", asset_cfg=_feet_cfg()) < 0.5
  )

  env_r, _ = _make_env(
    foot_quats=torch.stack([rolled, rolled], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  assert torch.allclose(
    foot_lateral_roll_deg(env_r, "feet_ground_contact", asset_cfg=_feet_cfg()),
    torch.full((n,), 10.0),
    atol=0.5,
  )
  assert torch.all(
    foot_heel_toe_pitch_deg(env_r, "feet_ground_contact", asset_cfg=_feet_cfg()) < 0.5
  )


def test_stance_gating_ignores_swing_feet():
  # One foot flat-and-planted, the other tilted but airborne: the airborne
  # tilt must not count.
  n = 2
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  flat = _flat_foot().expand(n, 4)
  tilt = _rotate(_flat_foot(), (0, 1, 0), 30.0).expand(n, 4)
  feet = torch.stack([flat, tilt], dim=1)  # right foot tilted
  found = torch.tensor([[1.0, 0.0]]).expand(n, 2)  # right foot airborne
  env, _ = _make_env(
    foot_quats=feet,
    torso_quat=torso,
    found=found,
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  pitch = foot_heel_toe_pitch_deg(env, "feet_ground_contact", asset_cfg=_feet_cfg())
  assert torch.all(pitch < 0.5)  # only the flat stance foot counts


def test_joint_speed_abs():
  n = 2
  cfg = MagicMock()
  cfg.name = "robot"
  cfg.joint_ids = [0, 2]
  jv = torch.tensor([[1.0, 99.0, -3.0], [2.0, 99.0, 4.0]])
  env, _ = _make_env(
    foot_quats=torch.zeros(n, 2, 4),
    torso_quat=torch.zeros(n, 4),
    found=torch.ones(n, 2),
    command=torch.zeros(n, 3),
    joint_vel=jv,
  )
  out = joint_speed_abs(env, asset_cfg=cfg)
  assert torch.allclose(out, torch.tensor([2.0, 3.0]))  # mean(|1|,|-3|), mean(|2|,|4|)


def test_toein_cost_charges_inward_only():
  """One-sided: toe-out is free, toe-in past the margin is charged."""
  n = 2
  # Toe-OUT 15 deg: left +15, right -15 about vertical.
  lq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(15))]).expand(n, 4)
  rq = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(-15))]).expand(n, 4)
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  cmd = torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3)

  env_out, _ = _make_env(
    foot_quats=torch.stack([lq, rq], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=cmd,
  )
  out = foot_toein_cost(env_out, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(out, torch.zeros(n))

  # Toe-IN 15 deg (swap the yaw signs): charged quadratically past margin.
  env_in, _ = _make_env(
    foot_quats=torch.stack([rq, lq], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=cmd,
  )
  cost = foot_toein_cost(env_in, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  expected = 2 * (math.radians(15) - 0.05) ** 2  # two feet, margin 0.05 rad
  assert torch.allclose(cost, torch.full((n,), expected), atol=1e-4)

  # Small toe-in inside the margin (2 deg < 0.05 rad) is free.
  lq2 = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(-2))]).expand(n, 4)
  rq2 = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(2))]).expand(n, 4)
  env_sm, _ = _make_env(
    foot_quats=torch.stack([lq2, rq2], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=cmd,
  )
  small = foot_toein_cost(env_sm, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(small, torch.zeros(n))

  # Standing (command below threshold) gates the cost off.
  env_stand, _ = _make_env(
    foot_quats=torch.stack([rq, lq], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.zeros(n, 3),
  )
  gated = foot_toein_cost(env_stand, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(gated, torch.zeros(n))


def test_toein_metric_sees_inward_through_outward_mean():
  """Inward-only metric: reads the pigeon-toe even when one foot toes out."""
  from mjlab.tasks.velocity.mdp.metrics import foot_toein_deg

  n = 2
  lq_out = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(30))]).expand(
    n, 4
  )
  rq_in = torch.tensor([_quat_from_axis_angle((0, 0, 1), math.radians(10))]).expand(
    n, 4
  )  # right foot +10 = inward after the -1 sign
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  env, _ = _make_env(
    foot_quats=torch.stack([lq_out, rq_in], dim=1),
    torso_quat=torso,
    found=torch.ones(n, 2),
    command=torch.tensor([[0.5, 0.0, 0.0]]).expand(n, 3),
  )
  # Signed mean = (30 - 10) / 2 = +10 (looks toed-out)...
  mean = foot_toeout_deg(env, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.all(mean > 5.0)
  # ...but the inward metric still reports the 10-deg pigeon-toed foot.
  toein = foot_toein_deg(env, asset_cfg=_feet_cfg(), torso_cfg=_torso_cfg())
  assert torch.allclose(toein, torch.full((n,), 10.0), atol=0.5)


def test_flight_fraction_excludes_falls():
  """Flight = all feet airborne AND upright; tilted (falling) frames and
  grounded frames count zero; the fast gate excludes slow commands."""
  from mjlab.tasks.velocity.mdp.metrics import flight_fraction

  n = 4
  torso = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
  env, robot = _make_env(
    foot_quats=torch.zeros(n, 2, 4),
    torso_quat=torso,
    found=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
    command=torch.tensor(
      [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.3, 0.0, 0.0]]
    ),
  )
  # env0: airborne+upright (flight). env1: airborne but TILTED (falling —
  # not flight). env2: one foot down (not flight). env3: airborne+upright
  # but slow command (flight for base metric, gated out of _fast).
  robot.data.projected_gravity_b = torch.tensor(
    [[0.0, 0.0, -1.0], [0.9, 0.0, -0.4], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
  )
  base = flight_fraction(env, "feet_ground_contact")
  assert torch.allclose(base, torch.tensor([1.0, 0.0, 0.0, 1.0]))
  fast = flight_fraction(env, "feet_ground_contact", min_command_speed=1.5)
  assert torch.allclose(fast, torch.tensor([1.0, 0.0, 0.0, 0.0]))
