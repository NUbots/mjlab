"""Observation-history encoder for the actor (end-to-end TCN).

A policy that only sees the current proprioceptive frame has to infer the
robot's unobservable state (contact, drift, load, actuator lag) from a
single snapshot, so it hedges instead of adapting. This module gives the
actor a short-term memory without making it recurrent:

- The environment publishes a ``"history"`` observation group: a T-step
  window of the ACTOR observation stream, shaped ``[B, T, D]``
  (``flatten_history_dim=False``). Its terms are copies of the actor
  terms, so one frame of the window has exactly the actor-vector layout.
- A small temporal convolutional network (:class:`TcnHistoryEncoder`)
  compresses that window into a latent ``z``, which is concatenated onto
  the current (normalized) actor observation before the policy MLP.

The encoder is trained end-to-end: PPO's gradient flows through the TCN,
so what the window is compressed into is whatever the reward pays for. No
auxiliary loss, no privileged teacher, no distillation stage.

The TCN lives INSIDE the actor model, so the latent never crosses the
observation boundary, and it is stateless by construction: deployment
feeds a flat ring buffer of past observation vectors, and no hidden state
has to survive a control tick. :class:`OnnxHistoryPolicy` is the
deployable graph — TCN plus policy body, one flat window in, actions out.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

from mjlab.rl.config import RslRlModelCfg

HISTORY_GROUP = "history"
"""Name of the observation group holding the actor-observation window."""


@dataclass
class HistoryModelCfg(RslRlModelCfg):
  """Actor model config with the observation-history TCN encoder."""

  history_cfg: dict[str, Any] = field(default_factory=dict)
  """Passed to :class:`HistoryActor`. Keys: z_dim, tcn_channels, tcn_kernel,
  tcn_stride."""

  class_name: str = "mjlab.rl.obs_history:HistoryActor"
  """Resolved by RSL-RL's ``resolve_callable``."""


class TcnHistoryEncoder(nn.Module):
  """Fixed-window temporal conv net: obs history [B, T, D] -> z [B, z].

  Stateless by construction: deployment feeds a flat ring buffer, no
  hidden state crosses control ticks.
  """

  def __init__(
    self,
    obs_dim: int,
    window: int,
    z_dim: int,
    channels: tuple[int, ...] = (32, 32),
    kernel: int = 5,
    stride: int = 2,
    activation: str = "elu",
  ) -> None:
    super().__init__()
    act = resolve_nn_activation(activation)
    layers: list[nn.Module] = []
    in_ch = obs_dim
    for out_ch in channels:
      layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride))
      layers.append(act)
      in_ch = out_ch
    self.convs = nn.Sequential(*layers)
    try:
      with torch.no_grad():
        flat = self.convs(torch.zeros(1, obs_dim, window)).flatten(1).shape[-1]
    except RuntimeError as exc:
      raise ValueError(
        f"TCN window {window} too short for kernel={kernel}, stride={stride}, "
        f"{len(channels)} layers"
      ) from exc
    self.feature_dim = flat
    self.head = nn.Linear(flat, z_dim)

  def features(self, history: torch.Tensor) -> torch.Tensor:
    """Shared conv-trunk features [B, flat]."""
    # [B, T, D] -> [B, D, T] for Conv1d (channels = obs features).
    return self.convs(history.transpose(1, 2)).flatten(1)

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    return self.head(self.features(history))


class HistoryActor(MLPModel):
  """Actor whose policy MLP is conditioned on an encoded obs history.

  ``obs_groups["actor"]`` must contain the proprio group(s) plus
  ``"history"`` (the 3D obs window -> TCN). The policy MLP consumes
  ``[normalized proprio, z]``; the base-class normalizer is scoped to the
  proprio groups only, with a dedicated normalizer for the history frames.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    history_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    cfg = dict(history_cfg or {})
    self._z_dim = int(cfg.pop("z_dim", 16))
    tcn_channels = tuple(cfg.pop("tcn_channels", (32, 32)))
    tcn_kernel = int(cfg.pop("tcn_kernel", 5))
    tcn_stride = int(cfg.pop("tcn_stride", 2))
    if cfg:
      raise ValueError(f"Unknown history_cfg keys: {sorted(cfg)}")

    # super().__init__ calls the _get_obs_dim/_get_latent_dim hooks, which
    # record the history dims (single source of truth: the env shapes).
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    activation = kwargs.get("activation", "elu")
    self.encoder = TcnHistoryEncoder(
      obs_dim=self._hist_dim,
      window=self._window,
      z_dim=self._z_dim,
      channels=tcn_channels,
      kernel=tcn_kernel,
      stride=tcn_stride,
      activation=activation,
    )
    if self.obs_normalization:
      self.history_normalizer: nn.Module = EmpiricalNormalization(self._hist_dim)
    else:
      self.history_normalizer = nn.Identity()

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    groups = list(obs_groups[obs_set])
    if HISTORY_GROUP not in groups:
      raise ValueError(
        f"HistoryActor needs '{HISTORY_GROUP}' in obs_groups[{obs_set!r}], got {groups}"
      )
    proprio = [g for g in groups if g != HISTORY_GROUP]
    history = obs[HISTORY_GROUP]
    if history.dim() != 3:
      raise ValueError(
        f"'{HISTORY_GROUP}' group must be [B, T, D] "
        f"(flatten_history_dim=False), got {tuple(history.shape)}"
      )
    self._window = int(history.shape[1])
    self._hist_dim = int(history.shape[-1])
    obs_dim = 0
    for group in proprio:
      if obs[group].dim() != 2:
        raise ValueError(
          f"proprio group '{group}' must be 2D, got {tuple(obs[group].shape)}"
        )
      obs_dim += int(obs[group].shape[-1])
    # Returning only the proprio groups scopes the base-class normalizer
    # (and its update_normalization) to them.
    return proprio, obs_dim

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._z_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: Any = None,
  ) -> torch.Tensor:
    del masks, hidden_state  # Non-recurrent.
    parts: list[torch.Tensor] = []
    for group in self.obs_groups:
      value = obs[group]
      assert isinstance(value, torch.Tensor)
      parts.append(value)
    x = self.obs_normalizer(torch.cat(parts, dim=-1))
    # End-to-end: PPO backprops through the TCN (undetached).
    z = self.encoder(self.history_normalizer(obs[HISTORY_GROUP]))
    return torch.cat([x, z], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)  # Proprio groups.
    # Frame-wise stats: the window is flattened to [B * T, D] so every frame
    # counts once. nn.Identity (obs_normalization=False) has nothing to update.
    normalizer = self.history_normalizer
    if isinstance(normalizer, EmpiricalNormalization):
      history = obs[HISTORY_GROUP]
      normalizer.update(history.reshape(-1, history.shape[-1]))

  def as_onnx(self, verbose: bool) -> nn.Module:
    return OnnxHistoryPolicy(self, verbose)

  def as_jit(self) -> nn.Module:
    return self.as_onnx(verbose=False)


class OnnxHistoryPolicy(nn.Module):
  """Deployable policy: one flat history window in, actions out.

  Input contract (single tensor ``obs`` of shape ``[B, T * D]``):
  time-major, OLDEST frame first -- ``reshape(T, D)`` gives frame t in row
  t, the newest frame last. Each frame has the actor-observation layout
  (the history group's terms are copies of the actor terms), so the robot
  keeps one ring buffer of the obs vector it already builds and the graph
  slices the current obs out of the last frame. Seed the buffer by
  repeating the first frame (matches the training-side CircularBuffer
  backfill on reset).
  """

  is_recurrent: bool = False

  def __init__(self, model: HistoryActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.window = model._window
    self.hist_dim = model._hist_dim
    self.input_size = self.window * self.hist_dim
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_normalizer = copy.deepcopy(model.history_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    if model.obs_dim != self.hist_dim:
      raise ValueError(
        "History export requires the actor obs to equal one history frame "
        f"(actor dim {model.obs_dim} != history frame dim {self.hist_dim}); "
        "the history group must clone the actor terms."
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    history = x.view(-1, self.window, self.hist_dim)
    z = self.encoder(self.history_normalizer(history))
    current = self.obs_normalizer(history[:, -1, :])
    return self.deterministic_output(self.mlp(torch.cat([current, z], dim=-1)))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
