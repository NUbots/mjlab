"""Unit tests for the observation-history actor (rl/obs_history.py).

Pure torch, no environment.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mjlab.rl.obs_history import (
  HistoryActor,
  OnnxHistoryPolicy,
  TcnHistoryEncoder,
)

ACTOR_DIM = 71  # Matches the NUgus actor vector; any value works here.
WINDOW = 25
ACTIONS = 20
Z_DIM = 16

_OBS_GROUPS = {"actor": ["actor", "history"]}
_DIST_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}


def _fake_obs(batch: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch, ACTOR_DIM),
      "history": torch.randn(batch, WINDOW, ACTOR_DIM),
    },
    batch_size=[batch],
  )


def _make_actor(seed: int = 0) -> HistoryActor:
  torch.manual_seed(seed)
  return HistoryActor(
    _fake_obs(),
    _OBS_GROUPS,
    "actor",
    ACTIONS,
    history_cfg={"z_dim": Z_DIM},
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
  )


@pytest.fixture
def actor() -> HistoryActor:
  return _make_actor()


def test_encoder_shapes() -> None:
  encoder = TcnHistoryEncoder(obs_dim=ACTOR_DIM, window=WINDOW, z_dim=Z_DIM)
  out = encoder(torch.randn(3, WINDOW, ACTOR_DIM))
  assert out.shape == (3, Z_DIM)


def test_window_too_short_raises() -> None:
  with pytest.raises(ValueError, match="too short"):
    TcnHistoryEncoder(obs_dim=8, window=3, z_dim=4, kernel=5, stride=2)


def test_forward_shapes(actor: HistoryActor) -> None:
  """The policy MLP consumes [proprio, z]; the window is not concatenated raw."""
  obs = _fake_obs(6)
  assert actor(obs).shape == (6, ACTIONS)
  assert actor.get_latent(obs).shape == (6, ACTOR_DIM + Z_DIM)


def test_missing_history_group_raises() -> None:
  with pytest.raises(ValueError, match="'history'"):
    HistoryActor(
      _fake_obs(),
      {"actor": ["actor"]},
      "actor",
      ACTIONS,
      distribution_cfg=dict(_DIST_CFG),
    )


def test_grads_flow_through_encoder(actor: HistoryActor) -> None:
  """End-to-end: PPO's gradient path trains the TCN, not just the MLP."""
  actor(_fake_obs()).sum().backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in actor.encoder.parameters()
  )
  assert any(p.grad is not None for p in actor.mlp.parameters())


def test_onnx_policy_matches_manual_composition(actor: HistoryActor) -> None:
  """The exported graph must reproduce the training-time composition.

  The current observation is sliced from the LAST window frame, which is
  only valid because the history group clones the actor terms.
  """
  torch.manual_seed(1)
  policy = OnnxHistoryPolicy(actor, verbose=False)
  history = torch.randn(2, WINDOW, ACTOR_DIM)
  out = policy(history.reshape(2, -1))

  z = actor.encoder(actor.history_normalizer(history))
  current = actor.obs_normalizer(history[:, -1, :])
  assert actor.distribution is not None
  expected = actor.distribution.as_deterministic_output_module()(
    actor.mlp(torch.cat([current, z], dim=-1))
  )
  torch.testing.assert_close(out, expected)


def test_onnx_export(actor: HistoryActor, tmp_path) -> None:
  policy = actor.as_onnx(verbose=False)
  assert isinstance(policy, OnnxHistoryPolicy)
  path = tmp_path / "policy.onnx"
  torch.onnx.export(
    policy,
    policy.get_dummy_inputs(),
    str(path),
    input_names=policy.input_names,
    output_names=policy.output_names,
    opset_version=18,
    dynamo=False,
  )
  assert path.exists() and path.stat().st_size > 0
