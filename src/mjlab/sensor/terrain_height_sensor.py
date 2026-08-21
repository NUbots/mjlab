"""Terrain height sensor for per-frame vertical clearance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.sensor.raycast_sensor import RayCastData, RayCastSensor, RayCastSensorCfg


@dataclass
class TerrainHeightData(RayCastData):
  """Raycast data extended with per-frame terrain clearance.

  Inherits all fields from :class:`RayCastData` (distances, hit positions, normals,
  frame poses) and adds :attr:`heights`.
  """

  heights: torch.Tensor
  """Vertical clearance per frame. Shape is ``[B, F]`` when a reduction is
  applied, or ``[B, F, N]`` with ``reduction="none"``."""


@dataclass
class TerrainHeightSensorCfg(RayCastSensorCfg):
  """RayCastSensor that reports per-frame vertical clearance above terrain.

  Inherits all raycasting configuration from :class:`RayCastSensorCfg`. The sensor
  computes ``frame_z - hit_z`` for each ray and reduces across rays per frame using
  the chosen :attr:`reduction`.
  """

  reduction: str = "min"
  """How to aggregate rays within each frame: ``"min"``, ``"max"``, ``"mean"``,
  or ``"none"`` (no reduction, returns ``[B, F, N]``).
  Defaults to ``"min"`` (closest terrain point)."""

  group_size: int = 1
  """Number of consecutive frames to group into one logical output frame.

  When greater than 1, the minimum height across each group of frames is returned,
  reducing the output from ``[B, F]`` to ``[B, F // group_size]``. The total number
  of frames must be divisible by ``group_size``. Useful when attaching one site per
  foot corner: set ``group_size`` to the number of corners per foot to get one
  clearance value per foot equal to the clearance at the lowest corner.
  """

  def build(self) -> TerrainHeightSensor:
    return TerrainHeightSensor(self)


class TerrainHeightSensor(RayCastSensor):
  """Per-frame vertical clearance above terrain.

  Inherits all raycasting from :class:`RayCastSensor`. Access terrain heights via
  ``sensor.data.heights`` (shape ``[B, F]``, or ``[B, F // group_size]`` when
  :attr:`TerrainHeightSensorCfg.group_size` is greater than 1).
  """

  cfg: TerrainHeightSensorCfg

  @property
  def num_frames(self) -> int:
    """Number of logical output frames after grouping."""
    return self._num_frames // self.cfg.group_size

  @property
  def data(self) -> TerrainHeightData:
    """Raycast data with per-frame terrain clearance heights."""
    return super().data  # type: ignore[return-value]

  def _compute_data(self) -> TerrainHeightData:
    raw = super()._compute_data()
    F_raw = self._num_frames  # raw site count before any grouping
    N = self.num_rays_per_frame
    B = raw.distances.shape[0]

    frame_z = raw.frame_pos_w[:, :, 2]  # [B, F_raw]
    hit_z = raw.hit_pos_w[:, :, 2].view(B, F_raw, N)  # [B, F_raw, N]
    heights = frame_z.unsqueeze(-1) - hit_z  # [B, F_raw, N]

    dists = raw.distances.view(B, F_raw, N)
    normal_z = raw.normals_w[:, :, 2].view(B, F_raw, N)

    # Backface hit: ray hit the underside of geometry (normal_z < 0).
    # This means the ray origin is inside terrain, so clearance is 0.
    backface = (dists >= 0) & (normal_z < 0)
    heights = torch.where(backface, torch.zeros_like(heights), heights)

    # True miss: no intersection at all.
    miss = dists < 0
    all_miss = miss.all(dim=-1, keepdim=True).expand_as(miss)  # [B, F_raw, N]
    fallback = frame_z.unsqueeze(-1).clamp(0, self.cfg.max_distance)
    fallback = fallback.expand_as(heights)  # [B, F_raw, N]
    miss_value = torch.where(all_miss, fallback, self.cfg.max_distance)
    heights = torch.where(miss, miss_value, heights)

    reduction = self.cfg.reduction
    if reduction == "min":
      reduced = heights.min(dim=-1).values  # [B, F_raw]
    elif reduction == "max":
      reduced = heights.max(dim=-1).values
    elif reduction == "mean":
      reduced = heights.mean(dim=-1)
    elif reduction == "none":
      reduced = heights
    else:
      raise ValueError(f"Unknown reduction: {reduction!r}")

    G = self.cfg.group_size
    if G > 1:
      assert F_raw % G == 0, (
        f"num_frames ({F_raw}) must be divisible by group_size ({G})"
      )
      reduced = reduced.view(B, F_raw // G, G).min(dim=-1).values  # [B, F_raw//G]

    return TerrainHeightData(**vars(raw), heights=reduced)
