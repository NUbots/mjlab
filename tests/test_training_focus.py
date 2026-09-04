"""Tests for the training focus layer: config, gate, rewards and wiring."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.focus import (
  CHANNELS,
  FOCUS_PRESETS,
  ChannelFocus,
  SpeedProfile,
  TrainingFocusCfg,
  balanced,
  get_focus_preset,
  speed_first,
)
from mjlab.tasks.velocity.mdp.focus_apply import (
  apply_training_focus,
  attainment_weights,
  command_stages,
  focus_diagnostics,
)
from mjlab.tasks.velocity.mdp.rewards import (
  track_angular_velocity_attainment,
  track_linear_velocity_attainment,
)
from mjlab.tasks.velocity.mdp.stability_gate import (
  StabilityGate,
  get_stability_gate,
  make_gated_term,
  stability_gated,
)

##
# Fixtures and helpers.
##


def _mock_motion_env(
  command: list[list[float]],
  lin_vel: list[list[float]],
  ang_vel_z: list[float],
) -> MagicMock:
  """An env whose robot is commanded ``command`` and delivering ``lin_vel``."""
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = len(command)
  env.common_step_counter = 0
  env.command_manager.get_command.return_value = torch.tensor(command)
  asset = MagicMock()
  asset.data.root_link_lin_vel_b = torch.tensor(lin_vel)
  asset.data.root_link_ang_vel_b = torch.tensor([[0.0, 0.0, w] for w in ang_vel_z])
  env.scene.__getitem__.return_value = asset
  return env


def _focus(**overrides: Any) -> TrainingFocusCfg:
  base: dict[str, Any] = {
    "forward": ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    "backward": ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    "strafe": ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    "yaw": ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
  }
  base.update(overrides)
  return TrainingFocusCfg(**base)


##
# SpeedProfile.
##


def test_profile_constant_is_flat() -> None:
  profile = SpeedProfile.constant(0.7)
  assert profile.evaluate(0.0) == pytest.approx(0.7)
  assert profile.evaluate(5.0) == pytest.approx(0.7)


def test_profile_decaying_interpolates_and_holds_ends() -> None:
  profile = SpeedProfile.decaying(1.0, 0.2, 0.5, 1.0)
  assert profile.evaluate(0.0) == pytest.approx(1.0)  # flat below the first knot
  assert profile.evaluate(0.5) == pytest.approx(1.0)
  assert profile.evaluate(0.75) == pytest.approx(0.6)  # midpoint
  assert profile.evaluate(1.0) == pytest.approx(0.2)
  assert profile.evaluate(9.0) == pytest.approx(0.2)  # flat above the last knot


def test_profile_rising_is_the_mirror_case() -> None:
  profile = SpeedProfile.rising(0.2, 1.0, 0.6, 1.2)
  assert profile.evaluate(0.0) == pytest.approx(0.2)
  assert profile.evaluate(0.9) == pytest.approx(0.6)
  assert profile.evaluate(2.0) == pytest.approx(1.0)


def test_profile_band_peaks_between_shoulders() -> None:
  profile = SpeedProfile.band(1.0, 0.1, 0.4, 0.9, edge=0.2)
  assert profile.evaluate(0.0) == pytest.approx(0.1)
  assert profile.evaluate(0.65) == pytest.approx(1.0)
  assert profile.evaluate(1.5) == pytest.approx(0.1)


def test_profile_rejects_unsorted_and_empty_knots() -> None:
  with pytest.raises(ValueError, match="ascend"):
    SpeedProfile(knots=((1.0, 1.0), (0.5, 1.0)))
  with pytest.raises(ValueError, match="at least one knot"):
    SpeedProfile(knots=())
  with pytest.raises(ValueError, match="non-negative"):
    SpeedProfile(knots=((0.0, -1.0),))


def test_profile_scaled_multiplies_emphasis_only() -> None:
  scaled = SpeedProfile.decaying(1.0, 0.2, 0.5, 1.0).scaled(0.5)
  assert scaled.evaluate(0.5) == pytest.approx(0.5)
  assert scaled.evaluate(1.0) == pytest.approx(0.1)


##
# TrainingFocusCfg.
##


def test_balance_slides_both_scales_and_leaves_shape() -> None:
  focus = balanced()
  all_speed = focus.with_balance(1.0)
  assert all_speed.speed_scale == pytest.approx(2.0)
  assert all_speed.stability_scale == pytest.approx(0.0)
  all_stability = focus.with_balance(0.0)
  assert all_stability.speed_scale == pytest.approx(0.0)
  assert all_stability.stability_scale == pytest.approx(2.0)
  midpoint = focus.with_balance(0.5)
  assert midpoint.speed_scale == pytest.approx(1.0)
  assert midpoint.stability_scale == pytest.approx(1.0)
  # Only the scales move; the per-channel focus is untouched.
  assert midpoint.forward == focus.forward


def test_command_ranges_are_signed_per_channel() -> None:
  focus = _focus(
    forward=ChannelFocus(1.0, 1.2),
    backward=ChannelFocus(1.0, 0.6),
    strafe=ChannelFocus(1.0, 0.4),
    yaw=ChannelFocus(1.0, 2.5),
  )
  assert focus.command_ranges() == ((-0.6, 1.2), (-0.4, 0.4), (-2.5, 2.5))


def test_focus_requires_a_nonzero_speed_channel() -> None:
  with pytest.raises(ValueError, match="non-zero speed weight"):
    _focus(
      forward=ChannelFocus(0.0),
      backward=ChannelFocus(0.0),
      strafe=ChannelFocus(0.0),
      yaw=ChannelFocus(0.0),
    )


def test_stability_knots_fold_in_the_global_scale() -> None:
  focus = _focus(stability_scale=0.25)
  for knots in focus.stability_knots():
    assert knots == ((0.0, 0.25),)


@pytest.mark.parametrize("name", sorted(FOCUS_PRESETS))
def test_every_preset_builds_and_is_self_consistent(name: str) -> None:
  focus = get_focus_preset(name)
  assert set(focus.channels()) == set(CHANNELS)
  assert all(c.target_speed > 0.0 for c in focus.channels().values())
  # Presets must produce a usable command box, not a degenerate one.
  for lo, hi in focus.command_ranges():
    assert lo <= 0.0 <= hi and hi > 0.0


def test_unknown_preset_names_the_alternatives() -> None:
  with pytest.raises(KeyError, match="speed_first"):
    get_focus_preset("go_fast_please")


##
# StabilityGate.
##


def test_channel_speeds_decompose_command_and_cap_at_delivered() -> None:
  focus = _focus()
  # Env 0 walks forward, under-delivering. Env 1 is commanded backward and
  # strafing. Env 2 over-delivers forward: the cap must bite.
  env = _mock_motion_env(
    command=[[0.8, 0.0, 0.0], [-0.5, 0.4, 0.0], [0.4, 0.0, 1.0]],
    lin_vel=[[0.5, 0.0, 0.0], [-0.5, 0.2, 0.0], [1.2, 0.0, 0.0]],
    ang_vel_z=[0.0, 0.0, 0.6],
  )
  gate = StabilityGate(env, focus)
  commanded, gate_speed = gate.channel_speeds(env, "twist")

  assert commanded[0].tolist() == pytest.approx([0.8, 0.0, 0.0, 0.0])
  assert gate_speed[0].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0])
  assert commanded[1].tolist() == pytest.approx([0.0, 0.5, 0.4, 0.0])
  assert gate_speed[1].tolist() == pytest.approx([0.0, 0.5, 0.2, 0.0])
  # Sprinting past the command buys no extra relief.
  assert gate_speed[2, 0].item() == pytest.approx(0.4)
  assert gate_speed[2, 3].item() == pytest.approx(0.6)


def test_gate_speed_ignores_motion_opposing_the_command() -> None:
  focus = _focus()
  env = _mock_motion_env(
    command=[[0.0, 0.5, 0.0]],
    lin_vel=[[0.0, -0.5, 0.0]],  # strafing the wrong way
    ang_vel_z=[0.0],
  )
  gate = StabilityGate(env, focus)
  _, gate_speed = gate.channel_speeds(env, "twist")
  assert gate_speed[0, 2].item() == pytest.approx(0.0)


def test_gate_reads_the_commanded_channels_profile() -> None:
  """A forward command must read the forward curve, not its neighbours."""
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.decaying(1.0, 0.0, 0.2, 1.0)),
    strafe=ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
  )
  env = _mock_motion_env(
    command=[[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    lin_vel=[[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
    ang_vel_z=[0.0, 0.0],
  )
  value = StabilityGate(env, focus).compute(env, "twist")
  assert value[0].item() == pytest.approx(0.0, abs=1e-5)  # fast: relief
  assert value[1].item() == pytest.approx(1.0, abs=1e-5)  # slow: full pressure


def test_gate_blends_mixed_commands_by_normalized_share() -> None:
  """Forward and strafe asked for equally read a 50/50 blend."""
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    strafe=ChannelFocus(1.0, 1.0, SpeedProfile.constant(0.0)),
  )
  env = _mock_motion_env(
    command=[[0.5, 0.5, 0.0]],
    lin_vel=[[0.5, 0.5, 0.0]],
    ang_vel_z=[0.0],
  )
  value = StabilityGate(env, focus).compute(env, "twist")
  assert value[0].item() == pytest.approx(0.5, abs=1e-5)


def test_gate_share_normalizes_units_against_target_speed() -> None:
  """Yaw's larger numbers must not swamp the blend of a mixed command."""
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    yaw=ChannelFocus(1.0, 2.5, SpeedProfile.constant(0.0)),
  )
  # Each channel is at half its target, so the blend must be 50/50 despite
  # 1.25 rad/s dwarfing 0.5 m/s numerically.
  env = _mock_motion_env(
    command=[[0.5, 0.0, 1.25]],
    lin_vel=[[0.5, 0.0, 0.0]],
    ang_vel_z=[1.25],
  )
  value = StabilityGate(env, focus).compute(env, "twist")
  assert value[0].item() == pytest.approx(0.5, abs=1e-5)


def test_standing_envs_fall_back_to_the_standing_emphasis() -> None:
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(0.2)),
    standing_stability=0.9,
  )
  env = _mock_motion_env(
    command=[[0.0, 0.0, 0.0]], lin_vel=[[0.0, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  value = StabilityGate(env, focus).compute(env, "twist")
  assert value[0].item() == pytest.approx(0.9)


def test_gate_matches_scalar_profile_evaluation() -> None:
  """The batched interpolation must agree with SpeedProfile.evaluate."""
  profile = SpeedProfile(knots=((0.1, 1.0), (0.4, 0.8), (0.9, 0.2), (1.4, 0.15)))
  focus = _focus(forward=ChannelFocus(1.0, 1.0, profile))
  # Zero is excluded: a zero command takes the standing path instead, which
  # test_standing_envs_fall_back_to_the_standing_emphasis covers.
  speeds = [0.1, 0.25, 0.4, 0.6, 0.9, 1.2, 3.0]
  env = _mock_motion_env(
    command=[[s, 0.0, 0.0] for s in speeds],
    lin_vel=[[s, 0.0, 0.0] for s in speeds],
    ang_vel_z=[0.0] * len(speeds),
  )
  value = StabilityGate(env, focus).compute(env, "twist")
  for i, speed in enumerate(speeds):
    assert value[i].item() == pytest.approx(profile.evaluate(speed), abs=1e-5)


def test_stability_scale_applies_exactly_once() -> None:
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(1.0)),
    stability_scale=0.5,
  )
  env = _mock_motion_env(
    command=[[0.5, 0.0, 0.0]], lin_vel=[[0.5, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  assert StabilityGate(env, focus).compute(env, "twist")[0].item() == pytest.approx(0.5)


def test_gate_is_computed_once_per_step() -> None:
  focus = _focus()
  env = _mock_motion_env(
    command=[[0.5, 0.0, 0.0]], lin_vel=[[0.5, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  gate = StabilityGate(env, focus)
  with patch.object(gate, "compute", wraps=gate.compute) as spy:
    gate.value(env, "twist")
    gate.value(env, "twist")
    assert spy.call_count == 1
    env.common_step_counter = 1
    gate.value(env, "twist")
    assert spy.call_count == 2


def test_gate_singleton_rejects_conflicting_focus_configs() -> None:
  env = _mock_motion_env(
    command=[[0.5, 0.0, 0.0]], lin_vel=[[0.5, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  del env._stability_gate  # MagicMock auto-creates attributes; start clean.
  env._stability_gate = None
  first = get_stability_gate(env, _focus())
  assert get_stability_gate(env, _focus()) is first
  with pytest.raises(ValueError, match="same focus config"):
    get_stability_gate(env, _focus(stability_scale=0.5))


##
# stability_gated wrapper.
##


class _InnerClassTerm:
  """A class-based reward term, like ``upright`` and ``pose``."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del env
    self.built_with = cfg.params
    self.reset_calls: list[object] = []

  def __call__(self, env, scale: float, asset_cfg) -> torch.Tensor:
    del env, asset_cfg
    return torch.full((1,), scale)

  def reset(self, env_ids=None) -> None:
    self.reset_calls.append(env_ids)

  def debug_vis(self, visualizer) -> None:
    del visualizer


def _gated_env(gate_value: float = 0.5) -> tuple[MagicMock, TrainingFocusCfg]:
  focus = _focus(forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(gate_value)))
  env = _mock_motion_env(
    command=[[0.5, 0.0, 0.0]], lin_vel=[[0.5, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  env._stability_gate = None
  return env, focus


def test_wrapper_scales_a_plain_function_term() -> None:
  env, focus = _gated_env(0.25)

  def inner(env, magnitude: float) -> torch.Tensor:
    del env
    return torch.full((1,), magnitude)

  term = make_gated_term(
    RewardTermCfg(func=inner, weight=-2.0, params={"magnitude": 4.0}), focus, "twist"
  )
  assert term.weight == -2.0  # weight is the pressure at full emphasis
  wrapped = stability_gated(cfg=term, env=env)
  assert wrapped(env, **term.params)[0].item() == pytest.approx(1.0)  # 4.0 * 0.25


def test_wrapper_builds_class_terms_and_forwards_reset_and_debug_vis() -> None:
  env, focus = _gated_env(0.5)
  asset_cfg = MagicMock(spec=SceneEntityCfg)
  term = make_gated_term(
    RewardTermCfg(
      func=_InnerClassTerm,
      weight=1.0,
      params={"scale": 2.0, "asset_cfg": asset_cfg},
    ),
    focus,
    "twist",
  )
  wrapped = stability_gated(cfg=term, env=env)

  # Nested SceneEntityCfgs are the wrapper's job: the manager only scans
  # top-level params, and these now sit a level down.
  asset_cfg.resolve.assert_called_once_with(env.scene)
  assert wrapped(env, **term.params)[0].item() == pytest.approx(1.0)
  assert hasattr(wrapped, "debug_vis")
  wrapped.reset(env_ids=None)
  assert wrapped._inner.reset_calls == [None]


def test_wrapper_reset_is_safe_for_function_terms() -> None:
  env, focus = _gated_env()
  term = make_gated_term(
    RewardTermCfg(func=lambda env: torch.ones(1), weight=1.0, params={}),
    focus,
    "twist",
  )
  stability_gated(cfg=term, env=env).reset(env_ids=None)  # must not raise


##
# Attainment rewards.
##


def _attainment_env(command, lin_vel, ang_vel_z=None) -> MagicMock:
  n = len(command)
  return _mock_motion_env(command, lin_vel, ang_vel_z or [0.0] * n)


def test_attainment_reduces_to_the_plain_projection_at_equal_weights() -> None:
  env = _attainment_env(
    [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]], [[0.5, 0.0, 0.0], [0.3, 0.4, 0.0]]
  )
  value = track_linear_velocity_attainment(env, "twist")
  assert value[0].item() == pytest.approx(0.5)
  assert value[1].item() == pytest.approx(0.5)  # half of the command vector


def test_attainment_weighting_shifts_credit_toward_the_prioritized_axis() -> None:
  # Forward is fully delivered, strafe not at all, on a diagonal command.
  env = _attainment_env([[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.0]])
  equal = track_linear_velocity_attainment(
    env, "twist", channel_weights=(1.0, 1.0, 1.0)
  )
  forward = track_linear_velocity_attainment(
    env, "twist", channel_weights=(9.0, 1.0, 1.0)
  )
  strafe = track_linear_velocity_attainment(
    env, "twist", channel_weights=(1.0, 1.0, 9.0)
  )
  assert equal[0].item() == pytest.approx(0.5)
  assert forward[0].item() == pytest.approx(0.9)  # mostly reads the delivered axis
  assert strafe[0].item() == pytest.approx(0.1)  # mostly reads the missed one


def test_attainment_is_scale_invariant_in_the_channel_weights() -> None:
  env = _attainment_env([[0.5, 0.5, 0.0]], [[0.5, 0.1, 0.0]])
  small = track_linear_velocity_attainment(
    env, "twist", channel_weights=(3.0, 1.0, 1.0)
  )
  large = track_linear_velocity_attainment(
    env, "twist", channel_weights=(30.0, 10.0, 10.0)
  )
  assert small[0].item() == pytest.approx(large[0].item())


def test_attainment_uses_the_backward_weight_for_reverse_commands() -> None:
  env = _attainment_env([[-0.5, 0.5, 0.0]], [[-0.5, 0.0, 0.0]])
  value = track_linear_velocity_attainment(
    env, "twist", channel_weights=(1.0, 9.0, 1.0)
  )
  assert value[0].item() == pytest.approx(0.9)


def test_attainment_clips_overshoot_and_floors_reversal() -> None:
  env = _attainment_env(
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [[1.5, 0.0, 0.0], [-2.0, 0.0, 0.0]]
  )
  value = track_linear_velocity_attainment(env, "twist", min_credit=-0.5)
  assert value[0].item() == pytest.approx(1.0)
  assert value[1].item() == pytest.approx(-0.5)


def test_attainment_is_silent_below_the_command_threshold() -> None:
  env = _attainment_env([[0.05, 0.0, 0.0]], [[0.05, 0.0, 0.0]])
  value = track_linear_velocity_attainment(env, "twist", command_threshold=0.15)
  assert value[0].item() == pytest.approx(0.0)


def test_angular_attainment_scores_the_yaw_fraction() -> None:
  env = _attainment_env(
    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0]] * 3,
    ang_vel_z=[0.5, -0.25, 2.0],
  )
  value = track_angular_velocity_attainment(env, "twist")
  assert value[0].item() == pytest.approx(0.5)
  assert value[1].item() == pytest.approx(0.25)  # direction-aware, not signed error
  assert value[2].item() == pytest.approx(0.0)  # below threshold, and no inf


##
# Translation layer.
##


def test_attainment_weights_track_the_channel_speeds() -> None:
  linear, angular = attainment_weights(_focus())
  assert linear == pytest.approx(2.0)
  assert angular == pytest.approx(1.0)

  focus = _focus(
    forward=ChannelFocus(3.0, 1.0),
    backward=ChannelFocus(0.5, 1.0),
    strafe=ChannelFocus(0.25, 1.0),
    yaw=ChannelFocus(0.5, 1.0),
    speed_scale=2.0,
  )
  linear, angular = attainment_weights(focus)
  assert linear == pytest.approx(2.0 * 2.0 * (3.0 + 0.5 + 0.25) / 3.0)
  assert angular == pytest.approx(1.0 * 2.0 * 0.5)


def test_command_stages_ramp_from_a_fraction_to_the_targets() -> None:
  focus = _focus(
    forward=ChannelFocus(1.0, 1.0),
    backward=ChannelFocus(1.0, 0.5),
    strafe=ChannelFocus(1.0, 0.4),
    yaw=ChannelFocus(1.0, 2.0),
    command_stages=3,
    command_ramp_start=0.5,
    command_ramp_iters=10_000,
  )
  stages = command_stages(focus)
  assert [s["step"] for s in stages] == [0, 5_000 * 24, 10_000 * 24]
  assert stages[0]["lin_vel_x"] == pytest.approx((-0.25, 0.5))
  assert stages[-1]["lin_vel_x"] == pytest.approx((-0.5, 1.0))
  assert stages[-1]["ang_vel_z"] == pytest.approx((-2.0, 2.0))
  # The box only ever grows.
  tops = [s["lin_vel_x"][1] for s in stages if s["lin_vel_x"] is not None]
  assert len(tops) == len(stages)
  assert tops == sorted(tops)


def test_command_stages_handles_a_single_stage() -> None:
  stages = command_stages(_focus(command_stages=1))
  assert len(stages) == 1
  assert stages[0]["step"] == 0
  assert stages[0]["lin_vel_x"] == pytest.approx((-1.0, 1.0))


def test_apply_wraps_exactly_the_stability_terms() -> None:
  cfg = _stub_env_cfg()
  focus = _focus(stability_terms=("pose", "foot_slip"))
  apply_training_focus(cfg, focus)
  assert cfg.rewards["pose"].func is stability_gated
  assert cfg.rewards["foot_slip"].func is stability_gated
  assert cfg.rewards["air_time"].func is not stability_gated
  # The inner term survives intact inside the wrapper.
  assert cfg.rewards["pose"].params["inner_params"] == {"command_name": "twist"}


def test_apply_sets_attainment_weights_and_channel_weights() -> None:
  cfg = _stub_env_cfg()
  focus = _focus(forward=ChannelFocus(4.0, 1.0), stability_terms=())
  apply_training_focus(cfg, focus)
  lin = cfg.rewards["track_linear_velocity_attainment"]
  assert lin.params["channel_weights"] == (4.0, 1.0, 1.0)
  assert lin.weight == pytest.approx(2.0 * (4.0 + 1.0 + 1.0) / 3.0)


def test_apply_rejects_stability_terms_that_do_not_exist() -> None:
  cfg = _stub_env_cfg()
  with pytest.raises(KeyError, match="uprihgt"):
    apply_training_focus(cfg, _focus(stability_terms=("uprihgt",)))


def test_apply_can_skip_diagnostics_for_play_mode() -> None:
  cfg = _stub_env_cfg()
  apply_training_focus(cfg, _focus(stability_terms=()), add_diagnostics=False)
  assert "focus_diagnostics" not in cfg.curriculum


def _stub_env_cfg() -> MagicMock:
  cfg = MagicMock()
  cfg.rewards = {
    "track_linear_velocity_attainment": RewardTermCfg(
      func=track_linear_velocity_attainment,
      weight=0.0,
      params={"command_name": "twist", "channel_weights": (1.0, 1.0, 1.0)},
    ),
    "track_angular_velocity_attainment": RewardTermCfg(
      func=track_angular_velocity_attainment,
      weight=0.0,
      params={"command_name": "twist"},
    ),
    "pose": RewardTermCfg(
      func=lambda env, command_name: torch.zeros(1),
      weight=3.0,
      params={"command_name": "twist"},
    ),
    "foot_slip": RewardTermCfg(func=lambda env: torch.zeros(1), weight=-1.0, params={}),
    "air_time": RewardTermCfg(func=lambda env: torch.zeros(1), weight=0.1, params={}),
  }
  cfg.curriculum = {"command_vel": MagicMock(params={})}
  return cfg


def test_focus_diagnostics_publishes_the_live_gate() -> None:
  focus = _focus(forward=ChannelFocus(1.0, 1.0, SpeedProfile.constant(0.4)))
  env = _mock_motion_env(
    command=[[0.5, 0.0, 0.0]], lin_vel=[[0.5, 0.0, 0.0]], ang_vel_z=[0.0]
  )
  env._stability_gate = None
  cfg = MagicMock(params={"focus": focus, "command_name": "twist"})
  term = focus_diagnostics(cfg=cfg, env=env)
  out = term(env, torch.tensor([0]), focus, "twist")
  assert out["gate"].item() == pytest.approx(0.4)
  assert out["share_forward"].item() == pytest.approx(1.0)
  assert out["gate_speed_forward"].item() == pytest.approx(0.5)
  assert set(out) >= {f"emphasis_{name}" for name in CHANNELS}


##
# Wiring into the NUgus task.
##


def test_nugus_cfg_has_the_focus_applied() -> None:
  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

  cfg = nubots_nugus_flat_env_cfg()
  assert cfg.rewards["track_linear_velocity_attainment"].weight > 0.0
  assert cfg.rewards["pose"].func is stability_gated
  assert "focus_diagnostics" in cfg.curriculum
  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  # The pre-focus config had three identical stages, making the command
  # curriculum a no-op; the derived ramp must actually ramp.
  assert stages[0]["lin_vel_x"][1] < stages[-1]["lin_vel_x"][1]


def test_focus_env_vars_override_the_module_default() -> None:
  from mjlab.tasks.velocity.config.nugus import env_cfgs

  with patch.dict(os.environ, {"MJLAB_VELOCITY_FOCUS": "forward_sprint"}):
    assert env_cfgs._resolve_focus() == get_focus_preset("forward_sprint")
  with patch.dict(
    os.environ,
    {"MJLAB_VELOCITY_FOCUS": "speed_first", "MJLAB_VELOCITY_FOCUS_BALANCE": "0.25"},
  ):
    resolved = env_cfgs._resolve_focus()
    assert resolved == speed_first().with_balance(0.25)
    assert resolved.stability_scale == pytest.approx(1.5)


def test_play_mode_omits_the_focus_diagnostics() -> None:
  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

  cfg = nubots_nugus_flat_env_cfg(play=True)
  assert "focus_diagnostics" not in cfg.curriculum
  assert cfg.rewards["pose"].func is stability_gated


@pytest.mark.slow
def test_gated_terms_survive_the_manager_stack() -> None:
  """End-to-end: the wrapper has to work through the real reward manager.

  The manager deep-copies every term config, instantiates class-based
  funcs itself, and only resolves top-level ``SceneEntityCfg`` params --
  all three are things the wrapper has to cooperate with, and none of
  them are exercised by the mocked unit tests above.
  """
  import torch as _torch
  from conftest import get_test_device

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 4
  device = get_test_device()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    env.step(_torch.zeros(env.action_space.shape, device=device))

    gate = getattr(env, "_stability_gate", None)
    assert gate is not None
    ceiling = max(c.stability.max_emphasis for c in gate.focus.channels().values())
    assert 0.0 <= float(gate.last_gate) <= ceiling

    terms = env.reward_manager.active_terms
    rewards = env.reward_manager._step_reward
    # A gated term still produces a finite per-env value under its own name.
    pose = rewards[:, terms.index("pose")]
    assert _torch.isfinite(pose).all()
    attainment = rewards[:, terms.index("track_linear_velocity_attainment")]
    assert _torch.isfinite(attainment).all()
  finally:
    env.close()
