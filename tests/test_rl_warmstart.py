"""Tests for the checkpoint warm-start splicing utilities."""

from __future__ import annotations

import pytest
import torch

from mjlab.rl.warmstart import source_column_indices, splice_model_state_dict
from mjlab.tasks.path_tracking.rl import velocity_obs_layout

# Mirrors the nugus velocity -> path-tracking actor mismatch in miniature:
# the command term widens and a new trailing term appears, while terms on
# both sides of the command are shared.
SOURCE_LAYOUT = [("prop", 2), ("command", 3), ("clock", 2)]
TARGET_LAYOUT = [("prop", 2), ("command", 4), ("clock", 2), ("extra", 1)]
SOURCE_WIDTH = 7
TARGET_WIDTH = 9


def make_source_state() -> dict[str, torch.Tensor]:
  torch.manual_seed(0)
  return {
    "obs_normalizer._mean": torch.randn(1, SOURCE_WIDTH),
    "obs_normalizer._var": torch.rand(1, SOURCE_WIDTH) + 0.5,
    "obs_normalizer._std": torch.rand(1, SOURCE_WIDTH) + 0.5,
    "obs_normalizer.count": torch.tensor(1000),
    "mlp.0.weight": torch.randn(8, SOURCE_WIDTH),
    "mlp.0.bias": torch.randn(8),
    "mlp.2.weight": torch.randn(4, 8),
    "mlp.2.bias": torch.randn(4),
    "distribution.log_std_param": torch.randn(4),
  }


def test_column_indices_map_shared_terms_and_flag_new() -> None:
  columns = source_column_indices(TARGET_LAYOUT, SOURCE_LAYOUT)
  # prop copied, command resized (new), clock copied past the resized
  # command, extra new.
  assert columns.tolist() == [0, 1, -1, -1, -1, -1, 5, 6, -1]


def test_splice_copies_shared_columns_and_zeroes_new() -> None:
  source = make_source_state()
  spliced = splice_model_state_dict(source, TARGET_LAYOUT, SOURCE_LAYOUT)

  w = spliced["mlp.0.weight"]
  assert w.shape == (8, TARGET_WIDTH)
  assert torch.equal(w[:, :2], source["mlp.0.weight"][:, :2])
  assert torch.equal(w[:, 6:8], source["mlp.0.weight"][:, 5:7])
  assert (w[:, 2:6] == 0).all()
  assert (w[:, 8] == 0).all()

  mean = spliced["obs_normalizer._mean"]
  std = spliced["obs_normalizer._std"]
  assert torch.equal(mean[:, :2], source["obs_normalizer._mean"][:, :2])
  assert (mean[:, 2:6] == 0).all()
  assert (std[:, 2:6] == 1).all()
  assert torch.equal(spliced["obs_normalizer.count"], source["obs_normalizer.count"])

  # Everything not input-width-dependent is copied verbatim.
  for key in ("mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "distribution.log_std_param"):
    assert torch.equal(spliced[key], source[key])


def test_splice_identity_when_layouts_match() -> None:
  source = make_source_state()
  spliced = splice_model_state_dict(source, SOURCE_LAYOUT, SOURCE_LAYOUT)
  for key, value in source.items():
    assert torch.equal(spliced[key], value)


def test_splice_rejects_checkpoint_width_mismatch() -> None:
  source = make_source_state()
  wrong_layout = [("prop", 2), ("command", 3)]  # 5 wide, checkpoint is 7.
  with pytest.raises(ValueError, match="different observation configuration"):
    splice_model_state_dict(source, TARGET_LAYOUT, wrong_layout)


def test_velocity_obs_layout_derivation() -> None:
  """The velocity layout swaps the command width and drops target_twist."""
  path_layout = [
    ("base_ang_vel", 3),
    ("command", 16),
    ("gait_clock", 2),
    ("target_twist", 3),
  ]
  assert velocity_obs_layout(path_layout) == [
    ("base_ang_vel", 3),
    ("command", 3),
    ("gait_clock", 2),
  ]


def test_splice_works_without_normalizer_keys() -> None:
  source = {
    k: v for k, v in make_source_state().items() if not k.startswith("obs_normalizer")
  }
  spliced = splice_model_state_dict(source, TARGET_LAYOUT, SOURCE_LAYOUT)
  assert spliced["mlp.0.weight"].shape == (8, TARGET_WIDTH)
