"""Reward-driven recurrent memory actor (backlog 15d, Trent's v59 design).

The v58 arc established (doc 15 R43-R44): the 0.5 s observation window is
the binding constraint on sysid — the teacher's angular edge does not
survive deployment because the window physically does not contain the
information, and the hand-designed cross-tick hold is never trained (the
recursion runs only at deployment). This module replaces the window/TCN +
gate/prior machinery with a GRU whose hidden state IS the memory:

- Gradients pass through the state transitions (truncated BPTT over the
  rollout window), so WHAT to remember and HOW to hold it are learned
  from reward — no encoder, no teacher, no cutover, no hand-tuned decay.
  Train mode is deploy mode (the v55 lesson, applied to memory itself).
- The safe boot state is learned implicitly: h0 is fixed at zeros, and
  every post-reset step trains f(obs, 0) — the learned IMAGE of the zero
  state plays the role v58's z0 played. (An explicit learnable h0 would
  be a trap: TBPTT consumes stored initial states as detached data, so
  the parameter would silently receive no gradient.)
- Mirror augmentation is retained by DEFINING the latent mirror (Trent's
  insight: a learned representation has no canonical mirror, so impose
  one and let training conform). We use the canonical choice — split the
  hidden vector in half and swap the halves. Any linear involution
  decomposes into fixed/negated/swapped dims, and swap-pairs span all of
  them through linear combinations: symmetric content settles into the
  sum of the halves, antisymmetric into the difference, chiral into the
  halves individually. rsl_rl hard-refuses symmetry + recurrent, so
  :class:`MemoryPPO` constructs :class:`RecurrentSymmetry` itself.

Deployment contract: the ONNX graph carries the hidden state explicitly —
inputs ``obs`` [B, D] and ``h`` [B, L*H] (boot with zeros), outputs
``actions``, optional ``velocity`` (body frame, m/s), and ``h_out`` to
feed back next tick. No observation ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import Symmetry
from rsl_rl.models import RNNModel
from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg

_ODOM_GROUP = "odom_target"


@dataclass
class MemoryModelCfg(RslRlModelCfg):
  """Actor model config for the GRU memory actor.

  The rnn_type/rnn_hidden_dim/rnn_num_layers knobs live on the base
  :class:`RslRlModelCfg`.
  """

  memory_cfg: dict[str, Any] = field(default_factory=dict)
  """Passed to :class:`GruMemoryActor`. Keys: vhat, vhat_detach,
  vhat_hidden_dims."""


@dataclass
class MemoryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config for the memory actor (odometry aux loss)."""

  vel_loss_coef: float = 1.0
  """Coefficient on the ||v_hat - sg(base_lin_vel)||^2 odometry-head loss
  (only active when the actor was built with memory_cfg["vhat"])."""


def mirror_hidden_state(hidden: torch.Tensor) -> torch.Tensor:
  """The DEFINED latent mirror: swap the two halves of the hidden vector.

  An involution by construction (applying it twice is the identity), and
  fully general: under training pressure, symmetric content lands in the
  sum of the halves, antisymmetric in the difference, chiral in the
  halves individually.
  """
  half = hidden.shape[-1] // 2
  return torch.cat([hidden[..., half:], hidden[..., :half]], dim=-1)


class GruMemoryActor(RNNModel):
  """GRU actor with an optional walk-coupled odometry readout.

  ``obs_groups["actor"]`` may include the supervision-only "odom_target"
  group; it is excluded from the network input (the deployed robot has no
  velocity sensor) and consumed only by :meth:`velocity_loss`.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    memory_cfg: dict[str, Any] | None = None,
    rnn_hidden_dim: int = 256,
    **kwargs: Any,
  ) -> None:
    cfg = dict(memory_cfg or {})
    self._vhat = bool(cfg.pop("vhat", False))
    self._vhat_detach = bool(cfg.pop("vhat_detach", True))
    vhat_hidden_dims = tuple(cfg.pop("vhat_hidden_dims", (32,)))
    if cfg:
      raise ValueError(f"Unknown memory_cfg keys: {sorted(cfg)}")
    if rnn_hidden_dim % 2 != 0:
      raise ValueError(
        f"rnn_hidden_dim must be even for the swap-halves latent mirror, "
        f"got {rnn_hidden_dim}"
      )
    super().__init__(
      obs, obs_groups, obs_set, output_dim, rnn_hidden_dim=rnn_hidden_dim, **kwargs
    )
    if self._vhat:
      activation = kwargs.get("activation", "elu")
      vhat_in = cast(nn.Linear, self.mlp[-1]).in_features
      self.vel_head: nn.Module | None = MLP(
        vhat_in, self._odom_dim, vhat_hidden_dims, activation
      )
    else:
      self.vel_head = None

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    groups = list(obs_groups[obs_set])
    # odom_target is supervision only: exclude it from the network input.
    net_groups = [g for g in groups if g != _ODOM_GROUP]
    if self._vhat:
      if _ODOM_GROUP not in groups:
        raise ValueError(
          f"vhat=True needs '{_ODOM_GROUP}' in obs_groups[{obs_set!r}], "
          f"got {groups} (is RMA_VHAT=1 set for the env?)"
        )
      self._odom_dim = int(obs[_ODOM_GROUP].shape[-1])
    obs_dim = 0
    for group in net_groups:
      if obs[group].dim() != 2:
        raise ValueError(
          f"network group '{group}' must be 2D, got {tuple(obs[group].shape)}"
        )
      obs_dim += int(obs[group].shape[-1])
    return net_groups, obs_dim

  def _get_latent_dim(self) -> int:
    # Skip connection: the MLP consumes [normalized obs, GRU output].
    # Stock RNNModel routes obs ONLY through the RNN, which makes the
    # policy hostage to the memory — a confused GRU scrambles its entire
    # view of the world (v59 launch 2 postmortem). With the skip path
    # the worst case is the desired floor: the MLP zeroes the GRU half
    # and behaves as a memoryless policy; memory is purely additive.
    return self.obs_dim + self.latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: Any = None,
  ) -> torch.Tensor:
    # Normalized obs concat (MLPModel.get_latent, bypassing RNNModel's).
    obs_list = [obs[group] for group in self.obs_groups]
    x = self.obs_normalizer(torch.cat(obs_list, dim=-1))  # type: ignore[arg-type]
    h = self.rnn(x, masks, hidden_state).squeeze(0)
    if masks is not None:
      # Batch mode: the RNN unpads its output; align the skip path.
      x = unpad_trajectories(x, masks)
      assert isinstance(x, torch.Tensor)
    return torch.cat([x, h], dim=-1)

  def _policy_features(self, latent: torch.Tensor) -> torch.Tensor:
    """Policy-trunk penultimate features (input to the final action layer)."""
    h = latent
    for layer in list(self.mlp)[:-1]:
      h = layer(h)
    return h

  def velocity_loss(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: Any = None,
  ) -> torch.Tensor:
    """||v_hat - sg(base_lin_vel)||^2 on the policy trunk's features.

    In batched (recurrent) mode, obs is a padded trajectory TensorDict;
    the target is unpadded to align with the RNN's unpadded output. With
    vhat_detach the gradient stops at the features and only the head
    trains.
    """
    assert self.vel_head is not None
    if self._vhat_detach:
      # Pure probe: build no autograd graph through the GRU/MLP at all —
      # a BPTT graph over the full trajectory batch is expensive VRAM
      # (v59 launch 1 OOM'd Warp's sim allocations) and would be
      # discarded before use anyway.
      with torch.no_grad():
        latent = self.get_latent(obs, masks, hidden_state)
        feats = self._policy_features(latent)
      feats = feats.detach()
    else:
      latent = self.get_latent(obs, masks, hidden_state)
      feats = self._policy_features(latent)
    target = obs[_ODOM_GROUP]
    assert isinstance(target, torch.Tensor)
    if masks is not None:
      target = unpad_trajectories(target, masks)
      assert isinstance(target, torch.Tensor)
    return F.mse_loss(self.vel_head(feats), target)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return OnnxGruMemoryModel(self, verbose)

  def as_jit(self) -> nn.Module:
    return OnnxGruMemoryModel(self, verbose=False)


class OnnxGruMemoryModel(nn.Module):
  """Deployable GRU policy: hidden state in, hidden state out.

  Inputs:
    - ``obs`` [B, D]: the current actor observation vector.
    - ``h`` [B, L*H]: previous hidden state, flattened over layers. Boot
      with zeros — the network is trained from the zero state at every
      episode reset, so zeros ARE the learned safe boot condition.

  Outputs: ``actions``, [``velocity`` (body frame, m/s) if trained,]
  ``h_out`` [B, L*H] — feed back next tick. No ring buffer needed.
  """

  is_recurrent: bool = True

  def __init__(self, model: GruMemoryActor, verbose: bool) -> None:
    super().__init__()
    import copy as _copy

    if not isinstance(model.rnn.rnn, nn.GRU):
      raise NotImplementedError("OnnxGruMemoryModel supports GRU only")
    self.verbose = verbose
    self.num_layers = int(model.rnn.rnn.num_layers)
    self.hidden_dim = int(model.rnn.rnn.hidden_size)
    self.obs_normalizer = _copy.deepcopy(model.obs_normalizer)
    self.gru = _copy.deepcopy(model.rnn.rnn)
    self.mlp = _copy.deepcopy(model.mlp)
    self.vel_head = (
      _copy.deepcopy(model.vel_head) if model.vel_head is not None else None
    )
    self.input_size = int(model.obs_dim)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x = self.obs_normalizer(obs)
    h_in = h.view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2)
    out, h_new = self.gru(x.unsqueeze(0), h_in.contiguous())
    latent = torch.cat([x, out.squeeze(0)], dim=-1)
    feats = latent
    for layer in list(self.mlp)[:-1]:
      feats = layer(feats)
    actions = self.deterministic_output(self.mlp[-1](feats))
    h_out = h_new.permute(1, 0, 2).reshape(-1, self.num_layers * self.hidden_dim)
    if self.vel_head is None:
      return actions, h_out
    return actions, self.vel_head(feats), h_out

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(1, self.num_layers * self.hidden_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "h"]

  @property
  def output_names(self) -> list[str]:
    if self.vel_head is None:
      return ["actions", "h_out"]
    return ["actions", "velocity", "h_out"]


class RecurrentSymmetry(Symmetry):
  """Mirror augmentation for recurrent batches (Trent's defined-mirror move).

  rsl_rl refuses symmetry + recurrence because a learned hidden state has
  no canonical mirror. We impose one (:func:`mirror_hidden_state`) and
  augment the full recurrent batch coherently: padded observation
  trajectories and dense action/value tensors are mirrored and doubled
  along the trajectory/env dim, hidden states are mirrored with the
  defined map, masks are duplicated, and the stored distribution params
  are mirrored (mean via the action mirror; std is per-dim positive, so
  the sign-stripped action mirror gives the permuted std). Training under
  this augmentation makes the representation equivariant under the
  imposed map — the constraint shapes the content.
  """

  def augment_batch(
    self, batch: RolloutStorage.Batch, original_batch_size: int
  ) -> None:
    if not self.use_data_augmentation:
      return
    if batch.masks is None:
      # Feedforward batch: the stock path already handles it.
      return super().augment_batch(batch, original_batch_size)

    # Padded obs [T_pad, n_traj, ...] + dense actions [T, n_envs, A]: the
    # env callback mirrors and concatenates along the traj/env dim (it is
    # batch-layout aware).
    obs_aug, act_aug = self.data_augmentation_func(
      env=self.env, obs=batch.observations, actions=batch.actions
    )
    assert obs_aug is not None and act_aug is not None
    batch.observations = obs_aug
    batch.actions = act_aug

    # Dense per-step tensors [T, n_envs, 1]: mirrored transitions have
    # identical returns/values/advantages under a symmetric MDP.
    assert batch.old_actions_log_prob is not None
    assert batch.values is not None
    assert batch.advantages is not None
    assert batch.returns is not None
    assert batch.old_distribution_params is not None
    batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(1, 2, 1)
    batch.values = batch.values.repeat(1, 2, 1)
    batch.advantages = batch.advantages.repeat(1, 2, 1)
    batch.returns = batch.returns.repeat(1, 2, 1)

    # Old distribution params (mean, std): mirror the mean like actions;
    # std is positive per-dim, so |action-mirror| = permutation only.
    mean, std = batch.old_distribution_params
    _, mean_aug = self.data_augmentation_func(env=self.env, obs=None, actions=mean)
    assert mean_aug is not None
    _, std_aug = self.data_augmentation_func(env=self.env, obs=None, actions=std)
    assert std_aug is not None
    # The callback returns [originals | mirrored] along the env dim; keep
    # the originals verbatim and strip the mirror's signs from the copy.
    mirrored_std = torch.abs(std_aug[:, std.shape[1] :])
    batch.old_distribution_params = (
      mean_aug,
      torch.cat([std, mirrored_std], dim=1),
    )

    # Hidden states: apply the defined latent mirror and double along the
    # trajectory dim ([num_layers, n_traj, H]). The critic is
    # feedforward in this setup (hidden None).
    hidden_a, hidden_c = batch.hidden_states

    def _augment_hidden(hidden: Any) -> Any:
      if hidden is None:
        return None
      if isinstance(hidden, tuple):
        return tuple(
          torch.cat([state, mirror_hidden_state(state)], dim=1) for state in hidden
        )
      return torch.cat([hidden, mirror_hidden_state(hidden)], dim=1)

    batch.hidden_states = (_augment_hidden(hidden_a), _augment_hidden(hidden_c))
    batch.masks = batch.masks.repeat(1, 2)

  def compute_loss(
    self, actor: Any, batch: RolloutStorage.Batch, original_batch_size: int
  ) -> torch.Tensor:
    """Recurrent mirror loss: supervised equivariance under the defined map.

    ``mse(pi(mirror(obs), M(h)), sg(mirror(pi(obs, h))))`` over the
    trajectory batch. This is the correct way to impose the latent
    mirror on a recurrent policy: data augmentation injects mirrored
    samples that are off-policy until equivariance is learned, and the
    resulting KL against the stored stats pins the adaptive learning
    rate to its floor (v59 launch 2: lr locked at 1e-5 for 2.7k
    iterations). A loss term touches neither the surrogate ratios nor
    the KL controller.
    """
    if batch.masks is None:
      return super().compute_loss(actor, batch, original_batch_size)
    assert batch.observations is not None
    hidden = batch.hidden_states[0]
    assert hidden is not None and not isinstance(hidden, tuple)
    n_traj = batch.observations.batch_size[1]

    # Mirrored copy of the padded obs (the callback returns
    # [originals | mirrored] along the trajectory dim; slice the copy).
    obs_aug, _ = self.data_augmentation_func(
      env=self.env, obs=batch.observations.detach().clone(), actions=None
    )
    assert obs_aug is not None
    obs_m = obs_aug[:, n_traj:]

    # Mean actions on mirrored obs from the mirrored hidden state.
    mean_m = actor(obs_m, masks=batch.masks, hidden_state=mirror_hidden_state(hidden))
    # Target: mirror of the mean actions on the originals (no gradient).
    with torch.no_grad():
      mean_o = actor(
        batch.observations.detach().clone(),
        masks=batch.masks,
        hidden_state=hidden,
      )
      _, mean_o_aug = self.data_augmentation_func(
        env=self.env, obs=None, actions=mean_o
      )
      assert mean_o_aug is not None
      target = mean_o_aug[:, mean_o.shape[1] :]

    symmetry_loss = nn.functional.mse_loss(mean_m, target)
    return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()


class MemoryPPO(PPO):
  """PPO for the recurrent memory actor.

  Bypasses rsl_rl's symmetry-vs-recurrence guard by constructing
  :class:`RecurrentSymmetry` itself, and adds the odometry-head
  supervision pass (recurrent batches, no augmentation — the head is a
  detached readout and does not need mirrored supervision).
  """

  def __init__(
    self,
    *args: Any,
    vel_loss_coef: float = 1.0,
    symmetry_cfg: dict | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, symmetry_cfg=None, **kwargs)
    self.vel_loss_coef = float(vel_loss_coef)
    self.symmetry = RecurrentSymmetry(**symmetry_cfg) if symmetry_cfg else None

  def update(self) -> dict[str, float]:
    actor = cast(GruMemoryActor, self._raw_actor)
    if self.vel_loss_coef == 0.0 or actor.vel_head is None:
      return super().update()
    mean_vel_loss = 0.0
    num_batches = 0
    generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, 1)
    for batch in generator:
      batch_obs = batch.observations
      assert batch_obs is not None
      vel_loss = self.vel_loss_coef * actor.velocity_loss(
        batch_obs, masks=batch.masks, hidden_state=batch.hidden_states[0]
      )
      self.optimizer.zero_grad()
      vel_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      mean_vel_loss += vel_loss.item()
      num_batches += 1

    loss_dict = super().update()
    loss_dict["velocity"] = mean_vel_loss / max(num_batches, 1)
    return loss_dict
