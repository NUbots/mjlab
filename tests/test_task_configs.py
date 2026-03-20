"""Generic tests for task config integrity."""

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_all_tasks_have_play_config(all_task_ids: list[str]) -> None:
  """All tasks should be loadable in play mode."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id, play=True)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} play mode did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}' in play mode: {e}")


def test_play_mode_episode_length(all_task_ids: list[str]) -> None:
  """Play mode tasks should have infinite episode length."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9, (
      f"{task_id} (play mode) episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(all_task_ids: list[str]) -> None:
  """Play mode tasks should have observation corruption disabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)

    assert "actor" in cfg.observations, (
      f"Play mode task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play mode task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play mode task {task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    assert "actor" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching observation structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {task_id} training and play modes"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {task_id} training and play modes"
      )


def test_play_training_action_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching action structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {task_id} training and play modes"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play mode tasks should disable push_robot event."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events, (
      f"Play mode task {task_id} has push_robot event, expected it to be removed"
    )


@pytest.mark.parametrize(
  "task_id",
  [
    "Mjlab-Velocity-Rough-Nubots-Nugus",
    "Mjlab-Velocity-Flat-Nubots-Nugus",
  ],
)
def test_nugus_has_servo_domain_randomization(task_id: str) -> None:
  """Nugus velocity tasks should randomize servo gains and effort limits."""
  cfg = load_env_cfg(task_id)

  assert "servo_gains" in cfg.events
  assert "servo_effort_limits" in cfg.events

  servo_gains = cfg.events["servo_gains"]
  assert servo_gains.mode == "startup"
  assert servo_gains.params["kp_range"] == (0.8, 1.2)
  assert servo_gains.params["kd_range"] == (0.75, 1.25)
  assert servo_gains.params["operation"] == "scale"

  servo_effort_limits = cfg.events["servo_effort_limits"]
  assert servo_effort_limits.mode == "startup"
  assert servo_effort_limits.params["effort_limit_range"] == (0.85, 1.15)
  assert servo_effort_limits.params["operation"] == "scale"

  jitter_cfg = {
    "servo_gains_jitter_mx64": {
      "mode": "reset",
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "actuator_ids": [0, 3],
    },
    "servo_effort_jitter_mx64": {
      "mode": "reset",
      "effort_limit_range": (0.8, 1.2),
      "actuator_ids": [0, 3],
    },
    "servo_gains_jitter_mx106": {
      "mode": "reset",
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "actuator_ids": [1],
    },
    "servo_effort_jitter_mx106": {
      "mode": "reset",
      "effort_limit_range": (0.8, 1.2),
      "actuator_ids": [1],
    },
    "servo_gains_jitter_xh540": {
      "mode": "reset",
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "actuator_ids": [2],
    },
    "servo_effort_jitter_xh540": {
      "mode": "reset",
      "effort_limit_range": (0.8, 1.2),
      "actuator_ids": [2],
    },
  }

  for event_name, expected in jitter_cfg.items():
    assert event_name in cfg.events
    event = cfg.events[event_name]
    assert event.mode == expected["mode"]
    assert event.params["operation"] == "scale"
    assert event.params["asset_cfg"].actuator_ids == expected["actuator_ids"]
    if "kp_range" in expected:
      assert event.params["kp_range"] == expected["kp_range"]
    if "kd_range" in expected:
      assert event.params["kd_range"] == expected["kd_range"]
    if "effort_limit_range" in expected:
      assert event.params["effort_limit_range"] == expected["effort_limit_range"]
