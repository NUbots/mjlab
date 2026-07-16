"""Unit tests for the GRU memory actor (rl/memory.py). Pure torch, no env."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mjlab.rl.memory import (
  GruMemoryActor,
  OnnxGruMemoryModel,
  mirror_hidden_state,
)

ACTOR_DIM = 72
ACTIONS = 21
HIDDEN = 64

_DIST_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}
_OBS_GROUPS = {"actor": ["actor", "odom_target"]}


def _fake_obs(batch: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch, ACTOR_DIM),
      "odom_target": torch.randn(batch, 3),
    },
    batch_size=[batch],
  )


def _make_actor(**extra_cfg) -> GruMemoryActor:
  torch.manual_seed(0)
  return GruMemoryActor(
    _fake_obs(),
    _OBS_GROUPS,
    "actor",
    ACTIONS,
    memory_cfg={"vhat": True, **extra_cfg},
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
    rnn_type="gru",
    rnn_hidden_dim=HIDDEN,
    rnn_num_layers=1,
  )


def test_forward_shapes_and_odom_exclusion() -> None:
  actor = _make_actor()
  actor.reset()
  # The network input excludes the supervision-only odom group.
  assert actor.obs_dim == ACTOR_DIM
  out = actor(_fake_obs(6))
  assert out.shape == (6, ACTIONS)


def test_hidden_mirror_is_involution() -> None:
  h = torch.randn(1, 5, HIDDEN)
  torch.testing.assert_close(mirror_hidden_state(mirror_hidden_state(h)), h)
  # Halves actually swap.
  m = mirror_hidden_state(h)
  torch.testing.assert_close(m[..., : HIDDEN // 2], h[..., HIDDEN // 2 :])


def test_odd_hidden_dim_raises() -> None:
  with pytest.raises(ValueError, match="even"):
    _make_actor_odd()


def _make_actor_odd() -> GruMemoryActor:
  return GruMemoryActor(
    _fake_obs(),
    _OBS_GROUPS,
    "actor",
    ACTIONS,
    memory_cfg={"vhat": True},
    distribution_cfg=dict(_DIST_CFG),
    rnn_hidden_dim=63,
  )


def test_velocity_loss_detached_grads_only_head() -> None:
  actor = _make_actor()
  actor.reset()
  actor.velocity_loss(_fake_obs()).backward()
  assert actor.vel_head is not None
  assert all(p.grad is not None for p in actor.vel_head.parameters())
  assert all(p.grad is None for p in actor.mlp.parameters())
  assert all(p.grad is None for p in actor.rnn.parameters())


def test_velocity_loss_recurrent_batch_mode() -> None:
  """Padded trajectories + masks + explicit hidden state (the aux pass)."""
  actor = _make_actor()
  # Consistent with a real split of a [T=6, n_envs=2] rollout: env 0 is
  # one 6-step trajectory, env 1 split 4+2 — total valid = 12 = T * n.
  t_pad, n_traj = 6, 3
  obs = TensorDict(
    {
      "actor": torch.randn(t_pad, n_traj, ACTOR_DIM),
      "odom_target": torch.randn(t_pad, n_traj, 3),
    },
    batch_size=[t_pad, n_traj],
  )
  masks = torch.zeros(t_pad, n_traj, dtype=torch.bool)
  masks[:6, 0] = True
  masks[:4, 1] = True
  masks[:2, 2] = True
  hidden = torch.zeros(1, n_traj, HIDDEN)
  loss = actor.velocity_loss(obs, masks=masks, hidden_state=hidden)
  assert loss.dim() == 0
  loss.backward()
  assert actor.vel_head is not None
  assert all(p.grad is not None for p in actor.vel_head.parameters())
  assert all(p.grad is None for p in actor.rnn.parameters())


def test_policy_path_trains_rnn() -> None:
  actor = _make_actor()
  actor.reset()
  actor(_fake_obs()).sum().backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in actor.rnn.parameters()
  )


def test_onnx_model_matches_and_feeds_back(tmp_path) -> None:
  actor = _make_actor()
  student = OnnxGruMemoryModel(actor, verbose=False)
  assert student.input_names == ["obs", "h"]
  assert student.output_names == ["actions", "velocity", "h_out"]

  obs = torch.randn(2, ACTOR_DIM)
  h0 = torch.zeros(2, HIDDEN)
  actions, velocity, h_out = student(obs, h0)
  assert actions.shape == (2, ACTIONS)
  assert velocity.shape == (2, 3)
  assert h_out.shape == (2, HIDDEN)
  # Manual composition through the actor's own modules (skip connection:
  # the MLP consumes [normalized obs, GRU output]).
  x = actor.obs_normalizer(obs)
  out, h_new = actor.rnn.rnn(x.unsqueeze(0), h0.unsqueeze(0))
  feats = torch.cat([x, out.squeeze(0)], dim=-1)
  for layer in list(actor.mlp)[:-1]:
    feats = layer(feats)
  assert actor.distribution is not None
  expected = actor.distribution.as_deterministic_output_module()(actor.mlp[-1](feats))
  torch.testing.assert_close(actions, expected)
  torch.testing.assert_close(h_out, h_new.squeeze(0))
  # State feedback changes the output (memory is live).
  actions2, _, h2 = student(obs, h_out)
  assert not torch.allclose(actions, actions2)
  assert not torch.allclose(h_out, h2)

  path = tmp_path / "gru_student.onnx"
  torch.onnx.export(
    student,
    student.get_dummy_inputs(),
    str(path),
    input_names=student.input_names,
    output_names=student.output_names,
    opset_version=18,
    dynamo=False,
  )
  assert path.exists() and path.stat().st_size > 0


def test_latent_has_skip_connection() -> None:
  """The MLP consumes [obs, gru_out]: memory is additive, not a bottleneck."""
  actor = _make_actor()
  actor.reset()
  obs = _fake_obs(5)
  latent = actor.get_latent(obs)
  assert latent.shape == (5, ACTOR_DIM + HIDDEN)
  # The first block IS the normalized obs (the skip path).
  expected = actor.obs_normalizer(obs["actor"])
  torch.testing.assert_close(latent[:, :ACTOR_DIM], expected)


def test_recurrent_mirror_loss_trains_equivariance() -> None:
  """compute_loss: scalar, differentiable, zero for an equivariant map."""
  from types import SimpleNamespace
  from typing import Any, cast

  from mjlab.rl.memory import RecurrentSymmetry

  actor = _make_actor()

  def fake_aug(env, obs, actions):
    # Identity "mirror" for this unit test: mirrored == original, so a
    # deterministic actor must produce zero loss when the hidden mirror
    # is also identity-compatible (h swap on equal halves of zeros).
    obs_aug = None
    if obs is not None:
      obs_aug = torch.cat([obs, obs.clone()], dim=1)
    act_aug = None
    if actions is not None:
      act_aug = torch.cat([actions, actions.clone()], dim=1)
    return obs_aug, act_aug

  sym = RecurrentSymmetry(
    env=cast(Any, SimpleNamespace()),
    use_data_augmentation=False,
    use_mirror_loss=True,
    mirror_loss_coeff=1.0,
    data_augmentation_func=fake_aug,
  )
  t_pad, n_traj = 4, 2
  obs = TensorDict(
    {
      "actor": torch.randn(t_pad, n_traj, ACTOR_DIM),
      "odom_target": torch.randn(t_pad, n_traj, 3),
    },
    batch_size=[t_pad, n_traj],
  )
  masks = torch.ones(t_pad, n_traj, dtype=torch.bool)
  hidden = torch.zeros(1, n_traj, HIDDEN)  # Symmetric under the swap.

  from rsl_rl.storage import RolloutStorage

  batch = RolloutStorage.Batch(
    observations=obs, hidden_states=(hidden, None), masks=masks
  )
  loss = sym.compute_loss(actor, batch, original_batch_size=t_pad)
  assert loss.dim() == 0
  # Identity mirror + swap-symmetric hidden: the two forwards see the
  # same inputs, so the loss must be ~0.
  assert float(loss) < 1e-10
  # And augment_batch must be a no-op (data augmentation off).
  sym.augment_batch(batch, original_batch_size=t_pad)
  assert batch.observations is not None
  assert batch.observations.batch_size == torch.Size([t_pad, n_traj])


def test_vhat_without_odom_group_raises() -> None:
  obs = TensorDict({"actor": torch.randn(4, ACTOR_DIM)}, batch_size=[4])
  with pytest.raises(ValueError, match="RMA_VHAT=1"):
    GruMemoryActor(
      obs,
      {"actor": ["actor"]},
      "actor",
      ACTIONS,
      memory_cfg={"vhat": True},
      distribution_cfg=dict(_DIST_CFG),
      rnn_hidden_dim=HIDDEN,
    )
