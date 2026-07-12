"""Unit tests for the RMA adaptation module (rl/rma.py). Pure torch, no env."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mjlab.rl.rma import OnnxRmaStudentModel, RmaActor, RmaTcnEstimator

ACTOR_DIM = 72
DR_DIM = 169
WINDOW = 25
ACTIONS = 21
Z_DIM = 16

_OBS_GROUPS = {"actor": ["actor", "dr", "history"]}
_DIST_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}


def _fake_obs(batch: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch, ACTOR_DIM),
      "dr": torch.randn(batch, DR_DIM),
      "history": torch.randn(batch, WINDOW, ACTOR_DIM),
    },
    batch_size=[batch],
  )


@pytest.fixture
def actor() -> RmaActor:
  torch.manual_seed(0)
  return RmaActor(
    _fake_obs(),
    _OBS_GROUPS,
    "actor",
    ACTIONS,
    rma_cfg={"z_dim": Z_DIM},
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
  )


def test_forward_shapes(actor: RmaActor) -> None:
  obs = _fake_obs(6)
  out = actor(obs)
  assert out.shape == (6, ACTIONS)
  latent = actor.get_latent(obs)
  assert latent.shape == (6, ACTOR_DIM + Z_DIM)


def test_estimator_shapes() -> None:
  est = RmaTcnEstimator(obs_dim=ACTOR_DIM, window=WINDOW, z_dim=Z_DIM)
  out = est(torch.randn(3, WINDOW, ACTOR_DIM))
  assert out.shape == (3, Z_DIM)


def test_window_too_short_raises() -> None:
  with pytest.raises(ValueError, match="too short"):
    RmaTcnEstimator(obs_dim=8, window=3, z_dim=4, kernel=5, stride=2)


def test_estimation_loss_grads_only_estimator(actor: RmaActor) -> None:
  """Stop-grad: the regression must not touch encoder or policy MLP."""
  loss = actor.estimation_loss(_fake_obs())
  loss.backward()
  assert all(p.grad is not None for p in actor.estimator.parameters())
  assert all(p.grad is None for p in actor.encoder.parameters())
  assert all(p.grad is None for p in actor.mlp.parameters())


def test_policy_loss_grads_never_reach_estimator(actor: RmaActor) -> None:
  """PPO's gradient path covers encoder+policy but not the estimator,
  regardless of zhat_mix."""
  for mix in (0.0, 0.5, 1.0):
    actor.zero_grad()
    actor.zhat_mix.fill_(mix)
    actor(_fake_obs()).sum().backward()
    assert all(p.grad is None for p in actor.estimator.parameters())
    if mix < 1.0:
      assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in actor.encoder.parameters()
      )
    assert any(p.grad is not None for p in actor.mlp.parameters())
  actor.zhat_mix.fill_(0.0)


def test_zhat_mix_changes_output(actor: RmaActor) -> None:
  obs = _fake_obs()
  actor.zhat_mix.fill_(0.0)
  teacher_out = actor(obs)
  actor.zhat_mix.fill_(1.0)
  student_out = actor(obs)
  actor.zhat_mix.fill_(0.0)
  assert not torch.allclose(teacher_out, student_out)


def test_student_wrapper_matches_manual_composition(actor: RmaActor) -> None:
  torch.manual_seed(1)
  student = OnnxRmaStudentModel(actor, verbose=False)
  history = torch.randn(2, WINDOW, ACTOR_DIM)
  flat = history.reshape(2, -1)
  out = student(flat)

  zhat = actor.estimator(actor.history_normalizer(history))
  current = actor.obs_normalizer(history[:, -1, :])
  expected = actor.distribution.as_deterministic_output_module()(
    actor.mlp(torch.cat([current, zhat], dim=-1))
  )
  torch.testing.assert_close(out, expected)


def test_student_onnx_export(actor: RmaActor, tmp_path) -> None:
  student = actor.as_onnx(verbose=False)
  path = tmp_path / "student.onnx"
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


def test_e2e_mode_grads_flow_through_estimator() -> None:
  """e2e ablation: PPO's path trains the TCN directly; encoder is dead."""
  torch.manual_seed(2)
  actor = RmaActor(
    _fake_obs(),
    _OBS_GROUPS,
    "actor",
    ACTIONS,
    rma_cfg={"z_dim": Z_DIM, "e2e": True},
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
  )
  actor(_fake_obs()).sum().backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in actor.estimator.parameters()
  )
  assert all(p.grad is None for p in actor.encoder.parameters())


def test_missing_groups_raise() -> None:
  obs = _fake_obs()
  with pytest.raises(ValueError, match="RMA=1"):
    RmaActor(
      obs,
      {"actor": ["actor"]},
      "actor",
      ACTIONS,
      distribution_cfg=dict(_DIST_CFG),
    )
