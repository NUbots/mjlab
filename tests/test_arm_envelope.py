"""Unit tests for arm_envelope_cost (doc 11 idea 11e). Pure torch, no env."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity.mdp.rewards import arm_envelope_cost


def _make_env(joint_pos: torch.Tensor, default: torch.Tensor) -> Any:
  asset = SimpleNamespace(
    data=SimpleNamespace(joint_pos=joint_pos, default_joint_pos=default),
    find_joints=lambda names: (list(range(len(names))), list(names)),
  )
  return SimpleNamespace(scene={"robot": asset}, device="cpu")


def _make_term(env: Any) -> tuple[arm_envelope_cost, Any]:
  asset_cfg = SimpleNamespace(name="robot", joint_names=("a", "b", "c", "d"))
  cfg = SimpleNamespace(params={"asset_cfg": asset_cfg})
  return arm_envelope_cost(cast(Any, cfg), env), asset_cfg


def test_inside_envelope_is_free() -> None:
  default = torch.zeros(2, 4)
  joint_pos = torch.full((2, 4), 0.45)  # Inside the 0.5 rad margin.
  env = _make_env(joint_pos, default)
  term, asset_cfg = _make_term(env)
  cost = term(env, asset_cfg, margin=0.5)
  torch.testing.assert_close(cost, torch.zeros(2))


def test_outside_envelope_quadratic_and_direction_symmetric() -> None:
  default = torch.zeros(2, 4)
  joint_pos = torch.zeros(2, 4)
  joint_pos[0, 0] = 0.8  # 0.3 past the margin, positive direction.
  joint_pos[1, 0] = -0.8  # Same excess, negative direction.
  env = _make_env(joint_pos, default)
  term, asset_cfg = _make_term(env)
  cost = term(env, asset_cfg, margin=0.5)
  expected = torch.full((2,), 0.3**2)
  torch.testing.assert_close(cost, expected)


def test_excursion_measured_from_default_pose() -> None:
  default = torch.full((1, 4), 0.4)
  joint_pos = torch.full((1, 4), 0.4)  # At default: zero excursion.
  env = _make_env(joint_pos, default)
  term, asset_cfg = _make_term(env)
  cost = term(env, asset_cfg, margin=0.5)
  torch.testing.assert_close(cost, torch.zeros(1))
