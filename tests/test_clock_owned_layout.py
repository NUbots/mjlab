"""The clock_owned task layouts (v44, v57) and the RMA checkpoint translation.

The layout assertions are deliberately about *order*, not just membership: a
checkpoint's observation normalizer is indexed by position, so a task that
builds the right terms in the wrong order loads without complaint and then
feeds the policy a permuted vector.
"""

from __future__ import annotations

import pytest
import torch

from mjlab.asset_zoo.robots.nugus.nugus_dcmotor import NUGUS_DCMOTOR_ACTION_SCALE
from mjlab.evaluation.rma_checkpoint import is_rma_actor, translate_rma_actor
from mjlab.tasks.velocity.config.nugus.env_cfgs import (
  HISTORY_WINDOW,
  nubots_nugus_flat_v44_env_cfg,
  nubots_nugus_flat_v57_env_cfg,
)

V57_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
  "gait_clock",
  "actuator_current",
  "servo_voltage",
)
"""Actor observation order of the trained v57 policy (112 dims total)."""

V57_ACTION_TERMS = ("joint_pos", "scripted_head", "phase_delta")
"""Action order of the trained v57 policy (20 + 0 + 1 = 21 dims)."""

V44_ACTOR_TERMS = V57_ACTOR_TERMS[:-2]
"""v44 has the same actor as v57 without the two servo-telemetry terms."""

V44_ACTION_TERMS = ("joint_pos", "phase_delta")
"""v44 has no scripted head (20 + 1 = 21 dims)."""


@pytest.fixture
def v57_cfg():
  return nubots_nugus_flat_v57_env_cfg(play=True)


@pytest.fixture
def v44_cfg():
  return nubots_nugus_flat_v44_env_cfg(play=True)


def test_actor_term_order_matches_trained_policy(v57_cfg) -> None:
  assert tuple(v57_cfg.observations["actor"].terms) == V57_ACTOR_TERMS


def test_action_term_order_matches_trained_policy(v57_cfg) -> None:
  assert tuple(v57_cfg.actions) == V57_ACTION_TERMS


def test_history_window_mirrors_the_actor(v57_cfg) -> None:
  """One history frame must be byte-for-byte the actor vector."""
  history = v57_cfg.observations["history"]
  assert tuple(history.terms) == V57_ACTOR_TERMS
  assert history.history_length == HISTORY_WINDOW


def test_gait_clock_is_policy_owned(v57_cfg) -> None:
  """``clock_owned``: the phase_delta action advances the clock, not time."""
  clock = v57_cfg.observations["actor"].terms["gait_clock"]
  assert clock.params["phase_source"] == "policy"


def test_v44_actor_and_action_order(v44_cfg) -> None:
  assert tuple(v44_cfg.observations["actor"].terms) == V44_ACTOR_TERMS
  assert tuple(v44_cfg.actions) == V44_ACTION_TERMS


def test_v44_has_no_history_window(v44_cfg) -> None:
  """v44's actor is a plain MLP; a history group would have no reader."""
  assert "history" not in v44_cfg.observations


def test_v44_leaves_the_phase_delta_unclamped(v44_cfg, v57_cfg) -> None:
  """The clamp is the only thing that changed in the action between the two
  training commits, and v44 set neither bound."""
  assert v44_cfg.actions["phase_delta"].raw_min is None
  assert v44_cfg.actions["phase_delta"].raw_max is None
  assert v57_cfg.actions["phase_delta"].raw_min == pytest.approx(0.35)
  assert v57_cfg.actions["phase_delta"].raw_max == pytest.approx(2.5)


def test_clock_owned_tasks_use_the_dcmotor_action_scale(v44_cfg, v57_cfg) -> None:
  """Both trained against the DC-motor plant, whose effort limits imply a
  scale ~18% larger than the builtin actuator's."""
  for cfg in (v44_cfg, v57_cfg):
    assert cfg.actions["joint_pos"].scale == NUGUS_DCMOTOR_ACTION_SCALE


def _rma_actor_state(mix: float) -> dict[str, torch.Tensor]:
  """A minimal two-latent RMA actor state dict."""
  return {
    "zhat_mix": torch.tensor(mix),
    "obs_normalizer._mean": torch.zeros(1, 112),
    "mlp.0.weight": torch.zeros(512, 128),
    # Teacher: an MLP over the privileged dr group.
    "encoder.0.weight": torch.zeros(128, 169),
    "dr_normalizer._mean": torch.zeros(1, 169),
    # Student: the TCN this branch calls ``encoder``.
    "estimator.convs.0.weight": torch.zeros(32, 112, 5),
    "estimator.head.weight": torch.zeros(16, 128),
    "history_normalizer._mean": torch.zeros(1, 112),
  }


def test_translate_renames_student_and_drops_teacher() -> None:
  translated = translate_rma_actor(_rma_actor_state(0.0))
  assert set(translated) == {
    "obs_normalizer._mean",
    "mlp.0.weight",
    "encoder.convs.0.weight",
    "encoder.head.weight",
    "history_normalizer._mean",
  }
  # The renamed tensors are the student's, not the dropped teacher's.
  assert translated["encoder.convs.0.weight"].shape == (32, 112, 5)


def test_translate_refuses_a_policy_that_reads_the_teacher() -> None:
  """A blended latent depends on privileged inputs this branch cannot build."""
  with pytest.raises(ValueError, match="zhat_mix"):
    translate_rma_actor(_rma_actor_state(0.5))


def test_plain_actor_passes_through() -> None:
  state = {"mlp.0.weight": torch.zeros(4, 4)}
  assert not is_rma_actor(state)
  assert translate_rma_actor(state) == state
