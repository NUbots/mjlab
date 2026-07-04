"""RL configuration for NUbots Nugus path tracking task."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.nugus.rl_cfg import nubots_nugus_ppo_runner_cfg


def nubots_nugus_path_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for NUbots Nugus path tracking task."""
  return replace(
    nubots_nugus_ppo_runner_cfg(),
    experiment_name="nugus_path_tracking",
    wandb_project="nugus_path_tracking",
  )
