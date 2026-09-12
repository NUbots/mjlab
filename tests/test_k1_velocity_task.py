"""Tests for the Booster K1 velocity task configuration."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.obs_history import HistoryActor, HistoryModelCfg, OnnxHistoryPolicy
from mjlab.tasks.velocity.config.booster_k1.env_cfgs import (
  HISTORY_WINDOW,
  VELOCITY_STAGES,
  booster_k1_flat_env_cfg,
)
from mjlab.tasks.velocity.config.booster_k1.rl_cfg import booster_k1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import VelocityStage

# Policy joints in MuJoCo body-tree order (arms before legs, head excluded).
# joint_pos / joint_vel / actions follow this order; deployment depends on it.
EXPECTED_POLICY_JOINTS = (
  "ALeft_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "ARight_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
)


@pytest.fixture(scope="module")
def k1_env():
  cfg = booster_k1_flat_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset(seed=1)
  yield env
  env.close()


def test_actor_terms_and_order() -> None:
  cfg = booster_k1_flat_env_cfg()
  assert list(cfg.observations["actor"].terms) == [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  ]


def test_no_competence_or_gait_clock() -> None:
  cfg = booster_k1_flat_env_cfg()
  assert "competence_tracker" not in cfg.events
  assert "competence_diagnostics" not in cfg.curriculum
  assert "gait_clock" not in cfg.observations["critic"].terms


def test_history_group_clones_actor_terms() -> None:
  cfg = booster_k1_flat_env_cfg()
  history = cfg.observations["history"]
  assert history.history_length == HISTORY_WINDOW
  assert history.flatten_history_dim is False
  assert list(history.terms) == list(cfg.observations["actor"].terms)


def test_play_disables_corruption_and_command_curriculum() -> None:
  cfg = booster_k1_flat_env_cfg(play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.observations["history"].enable_corruption is False
  assert "command_vel" not in cfg.curriculum
  assert "push_robot" not in cfg.events


def _envelope(stage: VelocityStage) -> list[tuple[float, float]]:
  ranges = []
  for rng in (stage["lin_vel_x"], stage["lin_vel_y"], stage["ang_vel_z"]):
    assert rng is not None
    ranges.append(rng)
  return ranges


def test_velocity_stages_only_widen() -> None:
  steps = [stage["step"] for stage in VELOCITY_STAGES]
  assert steps == sorted(steps) and steps[0] == 0
  for prev, cur in zip(VELOCITY_STAGES, VELOCITY_STAGES[1:], strict=False):
    for (lo0, hi0), (lo1, hi1) in zip(_envelope(prev), _envelope(cur), strict=True):
      assert lo1 <= lo0 and hi1 >= hi0


def test_policy_excludes_head(k1_env: ManagerBasedRlEnv) -> None:
  action_term = k1_env.action_manager.get_term("joint_pos")
  assert isinstance(action_term, JointPositionAction)
  assert tuple(action_term.target_names) == EXPECTED_POLICY_JOINTS
  joint_pos_cfg = k1_env.observation_manager.get_term_cfg("actor", "joint_pos")
  robot = k1_env.scene["robot"]
  observed = tuple(
    robot.joint_names[i] for i in joint_pos_cfg.params["asset_cfg"].joint_ids
  )
  assert observed == EXPECTED_POLICY_JOINTS


def test_history_actor_builds_from_live_env(k1_env: ManagerBasedRlEnv) -> None:
  """Real env shapes -> history actor -> deployable ONNX graph."""
  obs = RslRlVecEnvWrapper(k1_env).get_observations()
  assert obs["actor"].shape == (4, 69)
  assert obs["history"].shape == (4, HISTORY_WINDOW, 69)

  rl_cfg = booster_k1_ppo_runner_cfg()
  actor_cfg = rl_cfg.actor
  assert isinstance(actor_cfg, HistoryModelCfg)
  num_actions = k1_env.action_manager.total_action_dim
  assert num_actions == len(EXPECTED_POLICY_JOINTS)
  actor = HistoryActor(
    obs,
    {k: list(v) for k, v in rl_cfg.obs_groups.items()},
    "actor",
    num_actions,
    history_cfg=dict(actor_cfg.history_cfg),
    hidden_dims=actor_cfg.hidden_dims,
    activation=actor_cfg.activation,
    obs_normalization=actor_cfg.obs_normalization,
    distribution_cfg=dict(actor_cfg.distribution_cfg or {}),
  ).to(k1_env.device)
  assert actor(obs).shape == (4, num_actions)
  onnx_policy = actor.as_onnx(verbose=False)
  assert isinstance(onnx_policy, OnnxHistoryPolicy)
  assert onnx_policy.input_size == HISTORY_WINDOW * 69


def test_env_steps(k1_env: ManagerBasedRlEnv) -> None:
  actions = torch.zeros(4, k1_env.action_manager.total_action_dim, device=k1_env.device)
  for _ in range(10):
    obs, reward, *_ = k1_env.step(actions)
    assert torch.isfinite(reward).all()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert torch.isfinite(actor_obs).all()
