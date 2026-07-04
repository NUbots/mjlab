"""RL configuration for Booster K1 path tracking task."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.booster_k1.rl_cfg import booster_k1_ppo_runner_cfg


def booster_k1_path_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster K1 path tracking task."""
  return replace(
    booster_k1_ppo_runner_cfg(),
    experiment_name="k1_path_tracking",
    wandb_project="k1_path_tracking",
  )
