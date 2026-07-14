"""Unit tests for the RMA adaptation module (rl/rma.py). Pure torch, no env."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mjlab.rl.rma import (
  OnnxRmaGatedStudentModel,
  OnnxRmaStudentModel,
  RmaActor,
  RmaTcnEstimator,
)

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


_VHAT_OBS_GROUPS = {"actor": ["actor", "dr", "history", "odom_target"]}


def _fake_vhat_obs(batch: int = 4) -> TensorDict:
  obs = _fake_obs(batch)
  obs["odom_target"] = torch.randn(batch, 3)
  return obs


def _make_vhat_actor(**extra_rma_cfg) -> RmaActor:
  torch.manual_seed(3)
  return RmaActor(
    _fake_vhat_obs(),
    _VHAT_OBS_GROUPS,
    "actor",
    ACTIONS,
    rma_cfg={"z_dim": Z_DIM, "e2e": True, "vhat": True, **extra_rma_cfg},
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
  )


def test_vhat_target_excluded_from_policy_input() -> None:
  """odom_target is supervision only: the latent must not grow with it."""
  actor = _make_vhat_actor()
  latent = actor.get_latent(_fake_vhat_obs(6))
  assert latent.shape == (6, ACTOR_DIM + Z_DIM)


def test_vhat_detached_loss_grads_only_head() -> None:
  """Default (detached): the odometry loss is a pure probe on sg(z)."""
  actor = _make_vhat_actor()
  actor.velocity_loss(_fake_vhat_obs()).backward()
  assert actor.vel_head is not None
  assert all(p.grad is not None for p in actor.vel_head.parameters())
  assert all(p.grad is None for p in actor.estimator.parameters())
  assert all(p.grad is None for p in actor.mlp.parameters())


def test_vhat_undetached_loss_reaches_estimator() -> None:
  actor = _make_vhat_actor(vhat_detach=False)
  actor.velocity_loss(_fake_vhat_obs()).backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in actor.estimator.parameters()
  )


def test_vhat_student_two_outputs(tmp_path) -> None:
  actor = _make_vhat_actor()
  student = OnnxRmaStudentModel(actor, verbose=False)
  assert student.output_names == ["actions", "velocity"]
  history = torch.randn(2, WINDOW, ACTOR_DIM)
  out = student(history.reshape(2, -1))
  assert isinstance(out, tuple)
  actions, velocity = out
  assert actions.shape == (2, ACTIONS)
  assert velocity.shape == (2, 3)
  # Trunk tap: velocity reads the policy MLP's penultimate features.
  zhat = actor.estimator(actor.history_normalizer(history))
  current = actor.obs_normalizer(history[:, -1, :])
  h = torch.cat([current, zhat], dim=-1)
  for layer in list(actor.mlp)[:-1]:
    h = layer(h)
  assert actor.vel_head is not None
  torch.testing.assert_close(velocity, actor.vel_head(h))

  path = tmp_path / "student_vhat.onnx"
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


def test_vhat_without_odom_group_raises() -> None:
  with pytest.raises(ValueError, match="RMA_VHAT=1"):
    RmaActor(
      _fake_obs(),
      _OBS_GROUPS,
      "actor",
      ACTIONS,
      rma_cfg={"z_dim": Z_DIM, "e2e": True, "vhat": True},
      distribution_cfg=dict(_DIST_CFG),
    )


# --- Gated dual-channel mode (backlog 15d full design) ---

Z_FAST_DIM = 8


def _make_gated_actor(**extra_rma_cfg) -> RmaActor:
  torch.manual_seed(4)
  return RmaActor(
    _fake_vhat_obs(),
    _VHAT_OBS_GROUPS,
    "actor",
    ACTIONS,
    rma_cfg={
      "z_dim": Z_DIM,
      "gated": True,
      "z_fast_dim": Z_FAST_DIM,
      "vhat": True,
      **extra_rma_cfg,
    },
    hidden_dims=(64, 32),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DIST_CFG),
  )


def test_gated_latent_shapes() -> None:
  actor = _make_gated_actor()
  latent = actor.get_latent(_fake_vhat_obs(6))
  assert latent.shape == (6, ACTOR_DIM + Z_FAST_DIM + Z_DIM)
  assert actor(_fake_vhat_obs(6)).shape == (6, ACTIONS)


def test_gated_and_e2e_mutually_exclusive() -> None:
  with pytest.raises(ValueError, match="mutually exclusive"):
    _make_gated_actor(e2e=True)


def test_gated_policy_path_gradients() -> None:
  """PPO trains fast head, gate, encoder, and z0 — never the student head."""
  actor = _make_gated_actor()
  actor(_fake_vhat_obs()).sum().backward()
  assert actor.fast_head is not None and actor.gate_head is not None
  assert any(p.grad is not None for p in actor.fast_head.parameters())
  assert any(p.grad is not None for p in actor.gate_head.parameters())
  assert any(p.grad is not None for p in actor.encoder.parameters())
  assert actor.z0 is not None and actor.z0.grad is not None
  assert all(p.grad is None for p in actor.estimator.head.parameters())


def test_gated_gate_loss_trains_only_gate() -> None:
  """The Kalman-gate regression owns g: prior, student, encoder stay put."""
  actor = _make_gated_actor()
  actor.gate_loss(_fake_vhat_obs()).backward()
  assert actor.gate_head is not None and actor.fast_head is not None
  assert all(p.grad is not None for p in actor.gate_head.parameters())
  assert all(p.grad is None for p in actor.estimator.head.parameters())
  assert all(p.grad is None for p in actor.encoder.parameters())
  assert all(p.grad is None for p in actor.fast_head.parameters())
  assert actor.z0 is not None and actor.z0.grad is None


def test_gated_estimation_loss_isolated() -> None:
  actor = _make_gated_actor()
  actor.estimation_loss(_fake_vhat_obs()).backward()
  assert all(p.grad is not None for p in actor.estimator.head.parameters())
  assert all(p.grad is None for p in actor.encoder.parameters())
  assert actor.z0 is not None and actor.z0.grad is None


def test_gated_student_boot_matches_training_form(tmp_path) -> None:
  """At zero evidence the deployment recursion must equal Stage-1 training."""
  actor = _make_gated_actor()
  student = OnnxRmaGatedStudentModel(actor, verbose=False)
  assert student.input_names == ["obs", "z_state", "evidence"]
  assert student.output_names == [
    "actions",
    "velocity",
    "z_state_out",
    "evidence_out",
  ]
  history = torch.randn(2, WINDOW, ACTOR_DIM)
  flat = history.reshape(2, -1)
  actions, velocity, z_state, evidence = student(
    flat, torch.zeros(2, Z_DIM), torch.zeros(2, 1)
  )
  assert actions.shape == (2, ACTIONS)
  assert velocity.shape == (2, 3)
  assert z_state.shape == (2, Z_DIM)
  assert evidence.shape == (2, 1)
  # Manual Stage-1 composition with the student head as z_signal.
  feats = actor.estimator.features(actor.history_normalizer(history))
  z_raw = actor.estimator.head(feats)
  assert actor.gate_head is not None and actor.fast_head is not None
  assert actor.z0 is not None
  g = torch.sigmoid(actor.gate_head(feats))
  z_slow = (1.0 - g) * actor.z0 + g * z_raw
  current = actor.obs_normalizer(history[:, -1, :])
  h = torch.cat([current, actor.fast_head(feats), z_slow], dim=-1)
  for layer in list(actor.mlp)[:-1]:
    h = layer(h)
  assert actor.distribution is not None
  expected = actor.distribution.as_deterministic_output_module()(actor.mlp[-1](h))
  # The recursion divides by (ev + 1e-6), so boot equality is approximate.
  torch.testing.assert_close(actions, expected, atol=1e-3, rtol=1e-3)

  path = tmp_path / "student_gated.onnx"
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


def test_gated_state_holds_through_standing() -> None:
  """With the gate closed, z_state must hold and evidence decay at rho."""
  actor = _make_gated_actor()
  student = OnnxRmaGatedStudentModel(actor, verbose=False)
  # Force the gate shut: large negative bias swamps any feature input.
  gate_linear = student.gate_head
  assert isinstance(gate_linear, torch.nn.Linear)
  with torch.no_grad():
    gate_linear.bias.fill_(-50.0)
  z_in = torch.randn(1, Z_DIM)
  ev_in = torch.ones(1, 1)
  flat = torch.randn(1, WINDOW * ACTOR_DIM)
  _, _, z_out, ev_out = student(flat, z_in, ev_in)
  torch.testing.assert_close(z_out, z_in, atol=1e-4, rtol=1e-4)
  rho = float(student.hold_decay)
  torch.testing.assert_close(ev_out, torch.full((1, 1), rho), atol=1e-4, rtol=1e-4)
