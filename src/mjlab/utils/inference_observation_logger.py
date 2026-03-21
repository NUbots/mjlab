"""Inference observation logging utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch


class TensorboardWriterProtocol(Protocol):
  def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None: ...
  def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...
  def close(self) -> None: ...


def extract_actor_observations(observations: Any) -> torch.Tensor:
  """Extract actor observations from the common observation container formats."""
  if isinstance(observations, torch.Tensor):
    return observations

  if isinstance(observations, Mapping):
    actor_obs = observations.get("actor")
    if isinstance(actor_obs, torch.Tensor):
      return actor_obs

  if hasattr(observations, "get"):
    actor_obs = observations.get("actor")
    if isinstance(actor_obs, torch.Tensor):
      return actor_obs

  raise TypeError("Could not extract actor observations from input")


class InferenceObservationTensorboardLogger:
  """Logs policy input observations during inference to TensorBoard."""

  def __init__(
    self,
    log_dir: str | Path,
    *,
    enabled: bool = False,
    interval: int = 1,
    env_index: int = 0,
    max_dims: int = 80,
    tag_prefix: str = "inference/actor_obs",
    writer: TensorboardWriterProtocol | None = None,
  ) -> None:
    self.enabled = enabled
    self.interval = max(1, interval)
    self.env_index = max(0, env_index)
    self.max_dims = max(1, max_dims)
    self.tag_prefix = tag_prefix
    self.step = 0
    self._writer = writer

    if self.enabled and self._writer is None:
      from torch.utils.tensorboard import SummaryWriter

      self._writer = SummaryWriter(log_dir=str(log_dir))

  def log(self, observations: Any) -> None:
    """Log observations for the current inference step."""
    if not self.enabled or self._writer is None:
      self.step += 1
      return

    if self.step % self.interval != 0:
      self.step += 1
      return

    actor_obs = extract_actor_observations(observations)

    if actor_obs.ndim == 1:
      actor_obs_1d = actor_obs.detach().float().cpu()
    elif actor_obs.ndim >= 2:
      env_idx = min(self.env_index, actor_obs.shape[0] - 1)
      actor_obs_1d = actor_obs[env_idx].detach().float().reshape(-1).cpu()
    else:
      actor_obs_1d = actor_obs.detach().float().reshape(-1).cpu()

    if actor_obs_1d.numel() == 0:
      self.step += 1
      return

    dim_count = min(self.max_dims, actor_obs_1d.numel())
    obs_slice = actor_obs_1d[:dim_count]

    self._writer.add_histogram(f"{self.tag_prefix}/hist", obs_slice, self.step)
    self._writer.add_scalar(
      f"{self.tag_prefix}/mean", float(obs_slice.mean().item()), self.step
    )
    self._writer.add_scalar(
      f"{self.tag_prefix}/std", float(obs_slice.std(unbiased=False).item()), self.step
    )
    self._writer.add_scalar(
      f"{self.tag_prefix}/min", float(obs_slice.min().item()), self.step
    )
    self._writer.add_scalar(
      f"{self.tag_prefix}/max", float(obs_slice.max().item()), self.step
    )
    self._writer.add_scalar(
      f"{self.tag_prefix}/l2", float(torch.linalg.vector_norm(obs_slice).item()), self.step
    )

    for dim in range(dim_count):
      self._writer.add_scalar(
        f"{self.tag_prefix}/dim_{dim:03d}", float(obs_slice[dim].item()), self.step
      )

    self.step += 1

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()
