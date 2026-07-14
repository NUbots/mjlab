"""RMA-style concurrent adaptation module (teacher-student latent swap).

Wider domain randomization costs capability: a policy conditioned only on
proprioception must hedge against the whole DR envelope instead of
specializing to the realization it is actually in. This module recovers
that headroom with a Rapid Motor Adaptation (RMA) / Regularized Online
Adaptation (ROA) style latent:

- Teacher path (training): an encoder MLP maps the true per-env DR
  realization (the "dr" observation group: dr_ratios + dr_extras) to a
  compact latent ``z`` that conditions the policy MLP alongside the actor
  observations. Trained end-to-end with PPO.
- Student path (deployment): a fixed-window TCN estimator maps a T-step
  history of the actor observation stream (the "history" group) to
  ``z_hat``, regressed against ``sg(z)`` concurrently with PPO (stop-grad:
  the regression never shapes the encoder or policy, and PPO never reaches
  the estimator). The exported ONNX student is estimator + policy body in
  one graph, driven purely from an observation ring buffer.

Both networks live INSIDE the actor model, so the latent never crosses the
observation boundary: left-right symmetry augmentation stays a pure
obs-level transform (mirror_map.py mirrors the "dr" and "history" groups)
and z needs no mirror rule of its own.

The ``zhat_mix`` buffer blends the policy's latent input from encoder-z
(0.0, default) toward the detached estimator output (1.0). The runner can
anneal it late in training so the policy rolls out on the same latent it
will see on hardware; at 0 the estimator is a pure passenger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg

_DR_GROUP = "dr"
_HISTORY_GROUP = "history"
_ODOM_GROUP = "odom_target"


@dataclass
class RmaModelCfg(RslRlModelCfg):
  """Actor model config with the RMA encoder/estimator sub-networks."""

  rma_cfg: dict[str, Any] = field(default_factory=dict)
  """Passed to :class:`RmaActor`. Keys: z_dim, encoder_hidden_dims,
  tcn_channels, tcn_kernel, tcn_stride, e2e, vhat, vhat_detach,
  vhat_hidden_dims."""


@dataclass
class RmaPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config with the concurrent estimator-regression loss."""

  est_loss_coef: float = 1.0
  """Coefficient on the ||z_hat - sg(z)||^2 estimator regression loss."""

  vel_loss_coef: float = 1.0
  """Coefficient on the ||v_hat - sg(base_lin_vel)||^2 odometry-head loss
  (only active when the actor was built with rma_cfg["vhat"])."""


class RmaTcnEstimator(nn.Module):
  """Fixed-window temporal conv net: obs history [B, T, D] -> z_hat [B, z].

  Stateless by construction (RMA-paper style): deployment feeds a flat ring
  buffer, no hidden state crosses control ticks.
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
    self.head = nn.Linear(flat, z_dim)

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    # [B, T, D] -> [B, D, T] for Conv1d (channels = obs features).
    h = self.convs(history.transpose(1, 2))
    return self.head(h.flatten(1))


class RmaActor(MLPModel):
  """Actor with a privileged-param encoder and a history estimator.

  Observation sets: ``obs_groups["actor"]`` must contain the proprio
  group(s), plus ``"dr"`` (2D privileged params -> encoder) and
  ``"history"`` (3D obs window -> estimator). The policy MLP consumes
  ``[normalized proprio, z]``; the base-class normalizer is scoped to the
  proprio groups only, with dedicated normalizers for dr/history.
  """

  zhat_mix: torch.Tensor

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    rma_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    cfg = dict(rma_cfg or {})
    self._z_dim = int(cfg.pop("z_dim", 16))
    encoder_hidden_dims = tuple(cfg.pop("encoder_hidden_dims", (128, 64)))
    tcn_channels = tuple(cfg.pop("tcn_channels", (32, 32)))
    tcn_kernel = int(cfg.pop("tcn_kernel", 5))
    tcn_stride = int(cfg.pop("tcn_stride", 2))
    # End-to-end ablation: the policy consumes the TCN output directly
    # (UNdetached, so PPO gradients shape the history features; no
    # privileged encoder in the loop, no supervised anchor expected).
    # Answers "would PPO alone find the adaptation features?" — same
    # channel and capacity, supervision removed. NOTE: with e2e the
    # est_loss_coef should be 0; a naive est_coef=0 + zhat_mix=1 WITHOUT
    # this flag would train the policy on frozen random TCN features
    # (z_hat is detached on the standard path).
    self._e2e = bool(cfg.pop("e2e", False))
    # Walk-coupled odometry head (backlog 15d): a small readout from the
    # estimator's z to body-frame base linear velocity, supervised on the
    # ground-truth "odom_target" group and exported as a second ONNX
    # output. With vhat_detach (default) the head trains on sg(z): a pure
    # probe that cannot perturb the walk — the R40 linear probe already
    # showed the e2e trunk carries vx at R^2 0.95, this just makes the
    # readout a deployable artifact. Undetached is the knob for letting
    # odometry supervision shape the trunk (a separate experiment).
    self._vhat = bool(cfg.pop("vhat", False))
    self._vhat_detach = bool(cfg.pop("vhat_detach", True))
    vhat_hidden_dims = tuple(cfg.pop("vhat_hidden_dims", (32,)))
    if cfg:
      raise ValueError(f"Unknown rma_cfg keys: {sorted(cfg)}")

    # super().__init__ calls the _get_obs_dim/_get_latent_dim hooks, which
    # record the dr/history dims (single source of truth: the env shapes).
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    activation = kwargs.get("activation", "elu")
    self.encoder = MLP(self._dr_dim, self._z_dim, encoder_hidden_dims, activation)
    self.estimator = RmaTcnEstimator(
      obs_dim=self._hist_dim,
      window=self._window,
      z_dim=self._z_dim,
      channels=tcn_channels,
      kernel=tcn_kernel,
      stride=tcn_stride,
      activation=activation,
    )
    if self._vhat:
      self.vel_head: nn.Module | None = MLP(
        self._z_dim, self._odom_dim, vhat_hidden_dims, activation
      )
    else:
      self.vel_head = None
    if self.obs_normalization:
      self.dr_normalizer: nn.Module = EmpiricalNormalization(self._dr_dim)
      self.history_normalizer: nn.Module = EmpiricalNormalization(self._hist_dim)
    else:
      self.dr_normalizer = nn.Identity()
      self.history_normalizer = nn.Identity()
    # Blend of the policy's latent input: 0 = encoder z (teacher), 1 =
    # detached estimator z_hat (student rollouts). A buffer so it survives
    # checkpoints and follows .to(device).
    self.register_buffer("zhat_mix", torch.zeros(()))

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    groups = list(obs_groups[obs_set])
    if _DR_GROUP not in groups or _HISTORY_GROUP not in groups:
      raise ValueError(
        f"RmaActor needs '{_DR_GROUP}' and '{_HISTORY_GROUP}' in "
        f"obs_groups[{obs_set!r}], got {groups} (is RMA=1 set for the env?)"
      )
    # odom_target is supervision only: it must never feed the policy (the
    # deployed robot has no velocity sensor), so it is excluded from the
    # proprio set alongside the encoder/estimator inputs.
    proprio = [g for g in groups if g not in (_DR_GROUP, _HISTORY_GROUP, _ODOM_GROUP)]
    if self._vhat:
      if _ODOM_GROUP not in groups:
        raise ValueError(
          f"vhat=True needs '{_ODOM_GROUP}' in obs_groups[{obs_set!r}], "
          f"got {groups} (is RMA_VHAT=1 set for the env?)"
        )
      self._odom_dim = int(obs[_ODOM_GROUP].shape[-1])
    dr = obs[_DR_GROUP]
    history = obs[_HISTORY_GROUP]
    if dr.dim() != 2:
      raise ValueError(f"'{_DR_GROUP}' group must be 2D, got {tuple(dr.shape)}")
    if history.dim() != 3:
      raise ValueError(
        f"'{_HISTORY_GROUP}' group must be [B, T, D] "
        f"(flatten_history_dim=False), got {tuple(history.shape)}"
      )
    self._dr_dim = int(dr.shape[-1])
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
    if self._e2e:
      # End-to-end: PPO backprops through the TCN; encoder unused.
      z = self.estimator(self.history_normalizer(obs[_HISTORY_GROUP]))
      return torch.cat([x, z], dim=-1)
    z = self.encoder(self.dr_normalizer(obs[_DR_GROUP]))
    mix = float(self.zhat_mix)
    if mix > 0.0:
      # Detached: PPO must never backprop into the estimator, whose only
      # training signal is the regression loss.
      zhat = self.estimator(self.history_normalizer(obs[_HISTORY_GROUP])).detach()
      z = (1.0 - mix) * z + mix * zhat
    return torch.cat([x, z], dim=-1)

  def estimation_loss(self, obs: TensorDict) -> torch.Tensor:
    """||z_hat - sg(z)||^2: trains the estimator only (stop-grad target)."""
    with torch.no_grad():
      z_target = self.encoder(self.dr_normalizer(obs[_DR_GROUP]))
    zhat = self.estimator(self.history_normalizer(obs[_HISTORY_GROUP]))
    return F.mse_loss(zhat, z_target)

  def velocity_loss(self, obs: TensorDict) -> torch.Tensor:
    """||v_hat - sg(base_lin_vel)||^2 on the estimator's z.

    The target group is raw (no normalizer) so v_hat is in m/s, body
    frame — the units the exported "velocity" output promises. With
    vhat_detach the gradient stops at z and only the head trains.
    """
    assert self.vel_head is not None
    zhat = self.estimator(self.history_normalizer(obs[_HISTORY_GROUP]))
    if self._vhat_detach:
      zhat = zhat.detach()
    target = obs[_ODOM_GROUP]
    assert isinstance(target, torch.Tensor)
    return F.mse_loss(self.vel_head(zhat), target)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)  # Proprio groups.
    if self.obs_normalization:
      self.dr_normalizer.update(obs[_DR_GROUP])  # type: ignore[operator]
      history = obs[_HISTORY_GROUP]
      self.history_normalizer.update(  # type: ignore[operator]
        history.reshape(-1, history.shape[-1])
      )

  def as_onnx(self, verbose: bool) -> nn.Module:
    return OnnxRmaStudentModel(self, verbose)

  def as_jit(self) -> nn.Module:
    return OnnxRmaStudentModel(self, verbose=False)


class OnnxRmaStudentModel(nn.Module):
  """Deployable student: one flat history window in, actions out.

  Input contract (single tensor ``obs`` of shape ``[B, T * D]``):
  time-major, OLDEST frame first — ``reshape(T, D)`` gives frame t in row
  t, the newest frame last. Each frame has the actor-observation layout
  (the history group's terms are copies of the actor terms), so the robot
  keeps one ring buffer of the obs vector it already builds and the graph
  slices the current obs out of the last frame. Seed the buffer by
  repeating the first frame (matches the training-side CircularBuffer
  backfill on reset).

  When the actor carries a velocity head, the graph has a second output
  ``velocity``: estimated base linear velocity, BODY frame, m/s, at the
  policy rate — walk-coupled learned odometry for the localization stack.
  Caveats for consumers: ~1-3 tick causal latency on impulses, ballistic
  coasting during flight, and it is a velocity (integrating it drifts;
  fuse with vision).
  """

  is_recurrent: bool = False

  def __init__(self, model: RmaActor, verbose: bool) -> None:
    super().__init__()
    import copy as _copy

    self.verbose = verbose
    self.window = model._window
    self.hist_dim = model._hist_dim
    self.input_size = self.window * self.hist_dim
    self.obs_normalizer = _copy.deepcopy(model.obs_normalizer)
    self.history_normalizer = _copy.deepcopy(model.history_normalizer)
    self.estimator = _copy.deepcopy(model.estimator)
    self.vel_head = (
      _copy.deepcopy(model.vel_head) if model.vel_head is not None else None
    )
    self.mlp = _copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    if model.obs_dim != self.hist_dim:
      raise ValueError(
        "Student export requires the actor obs to equal one history frame "
        f"(actor dim {model.obs_dim} != history frame dim {self.hist_dim}); "
        "the history group must clone the actor terms."
      )

  def forward(
    self, x: torch.Tensor
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    history = x.view(-1, self.window, self.hist_dim)
    zhat = self.estimator(self.history_normalizer(history))
    current = self.obs_normalizer(history[:, -1, :])
    out = self.mlp(torch.cat([current, zhat], dim=-1))
    actions = self.deterministic_output(out)
    if self.vel_head is None:
      return actions
    return actions, self.vel_head(zhat)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    if self.vel_head is None:
      return ["actions"]
    return ["actions", "velocity"]


class RmaPPO(PPO):
  """PPO plus the concurrent estimator-regression pass.

  Before the standard PPO epochs consume (and clear) the rollout storage,
  one supervised epoch regresses the estimator against the CURRENT
  encoder's z on the same data, sharing the optimizer (disjoint parameter
  sets: the regression only populates estimator gradients, PPO's losses
  never reach the estimator). Kept as a separate pass rather than a forked
  copy of PPO.update() so rsl-rl upgrades stay drop-in; the one-update lag
  of the z target relative to PPO's encoder update is negligible at these
  learning rates.
  """

  def __init__(
    self,
    *args: Any,
    est_loss_coef: float = 1.0,
    vel_loss_coef: float = 1.0,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.est_loss_coef = float(est_loss_coef)
    self.vel_loss_coef = float(vel_loss_coef)

  def update(self) -> dict[str, float]:
    # self.actor aliases _raw_actor while torch.compile is disabled
    # (mjlab never sets torch_compile_mode); route via the raw handle so
    # this keeps working if compilation is ever enabled.
    actor = cast(RmaActor, self._raw_actor)
    run_est = self.est_loss_coef != 0.0
    run_vel = self.vel_loss_coef != 0.0 and actor.vel_head is not None
    if not (run_est or run_vel):
      # e2e ablation / supervision off: skip the aux pass entirely
      # (running it would only decay Adam momentum and log fake metrics).
      return super().update()
    mean_est_loss = 0.0
    mean_vel_loss = 0.0
    num_batches = 0
    generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)
    for batch in generator:
      batch_obs = batch.observations
      assert batch_obs is not None
      original_batch_size = batch_obs.batch_size[0]
      if self.symmetry:
        # Mirrored (dr, history, odom_target) tuples are valid extra
        # supervision and teach the aux heads the same equivariance the
        # policy sees.
        self.symmetry.augment_batch(batch, original_batch_size)
        batch_obs = batch.observations
        assert batch_obs is not None
      aux_loss = torch.zeros((), device=self.device)
      if run_est:
        est_loss = self.est_loss_coef * actor.estimation_loss(batch_obs)
        aux_loss = aux_loss + est_loss
        mean_est_loss += est_loss.item()
      if run_vel:
        vel_loss = self.vel_loss_coef * actor.velocity_loss(batch_obs)
        aux_loss = aux_loss + vel_loss
        mean_vel_loss += vel_loss.item()
      self.optimizer.zero_grad()
      aux_loss.backward()
      if self.is_multi_gpu:
        # Only aux-head (and, sans stop-grads, trunk) params carry grads
        # here; the reduction is consistent across ranks because the
        # grad-bearing set is.
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      num_batches += 1

    loss_dict = super().update()
    if run_est:
      loss_dict["estimation"] = mean_est_loss / max(num_batches, 1)
    if run_vel:
      loss_dict["velocity"] = mean_vel_loss / max(num_batches, 1)
    return loss_dict
