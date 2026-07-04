"""Utilities for warm-starting a policy from a differently-shaped checkpoint.

RSL-RL checkpoints can only be resumed into models with strictly matching
shapes. When two tasks share their network trunk but disagree on the
observation vector (e.g. one task's 3-dim twist command becomes a 16-dim
waypoint window, or a critic gains a privileged term), the mismatch is
confined to the input-width-dependent tensors: the first MLP layer's weight
columns and the empirical observation-normalizer statistics.

:func:`splice_model_state_dict` reshapes a source model state dict onto a
target observation layout. Columns of terms present in both layouts (same
name and width) are copied; columns of new or resized terms are
zero-initialized in the first-layer weight and given identity statistics
(mean 0, variance 1) in the normalizer, so the warm-started model initially
ignores them and relearns their influence during fine-tuning. Every other
tensor is copied verbatim.
"""

from __future__ import annotations

from typing import Sequence

import torch

ObsLayout = Sequence[tuple[str, int]]
"""Observation layout: (term name, width) pairs in concatenation order."""

_FIRST_LAYER_WEIGHT = "mlp.0.weight"
_NORMALIZER_MEAN = "obs_normalizer._mean"
_NORMALIZER_STATS = (
  _NORMALIZER_MEAN,
  "obs_normalizer._var",
  "obs_normalizer._std",
)


def source_column_indices(
  target_layout: ObsLayout, source_layout: ObsLayout
) -> torch.Tensor:
  """Map each target observation column to its source column, or -1 if new.

  A target term's columns map to the source term with the same name, but
  only when the widths agree; renamed, added, or resized terms map to -1.
  """
  source_offsets: dict[str, tuple[int, int]] = {}
  offset = 0
  for name, dim in source_layout:
    if name in source_offsets:
      raise ValueError(f"Duplicate term '{name}' in source layout.")
    source_offsets[name] = (offset, dim)
    offset += dim

  columns: list[int] = []
  for name, dim in target_layout:
    if name in source_offsets and source_offsets[name][1] == dim:
      start = source_offsets[name][0]
      columns.extend(range(start, start + dim))
    else:
      columns.extend([-1] * dim)
  return torch.tensor(columns, dtype=torch.long)


def splice_model_state_dict(
  source_state: dict[str, torch.Tensor],
  target_layout: ObsLayout,
  source_layout: ObsLayout,
) -> dict[str, torch.Tensor]:
  """Reshape a model state dict from a source to a target observation layout.

  The first MLP layer's weight columns and the normalizer statistics are
  spliced per :func:`source_column_indices`; all other tensors are copied
  verbatim. Raises ``ValueError`` if the source layout's width disagrees
  with the checkpoint (a sign the checkpoint was trained on a different
  configuration than the layout describes).
  """
  source_width = sum(dim for _, dim in source_layout)
  input_keys = [
    k for k in (_FIRST_LAYER_WEIGHT, *_NORMALIZER_STATS) if k in source_state
  ]
  if _FIRST_LAYER_WEIGHT not in source_state:
    raise ValueError(
      f"Source state dict has no '{_FIRST_LAYER_WEIGHT}' key; expected an "
      "rsl-rl>=5 MLP model state dict."
    )
  for key in input_keys:
    actual = source_state[key].shape[-1]
    if actual != source_width:
      raise ValueError(
        f"Source layout is {source_width} columns wide but '{key}' in the "
        f"checkpoint has {actual}; the checkpoint was trained on a different "
        "observation configuration than the source layout describes."
      )

  columns = source_column_indices(target_layout, source_layout)
  copied = columns >= 0

  spliced: dict[str, torch.Tensor] = {}
  for key, value in source_state.items():
    if key == _FIRST_LAYER_WEIGHT:
      out = value.new_zeros(value.shape[0], len(columns))
      out[:, copied] = value[:, columns[copied]]
    elif key in _NORMALIZER_STATS:
      fill = 0.0 if key == _NORMALIZER_MEAN else 1.0
      out = value.new_full((1, len(columns)), fill)
      out[:, copied] = value[:, columns[copied]]
    else:
      out = value.clone()
    spliced[key] = out
  return spliced
