"""RL configuration for the Booster K1 dribbling task (JOGA Table IV)."""

from dataclasses import dataclass
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class SymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO with rsl_rl's symmetry extension configured."""

  symmetry_cfg: dict[str, Any] | None = None


# One policy trained on every transition and its left-right mirror image, so
# it learns both feet. See mdp/symmetry.py.
SYMMETRY_CFG: dict[str, Any] = {
  "data_augmentation_func": "mjlab.tasks.dribbling.mdp.symmetry:mirror_augmentation",
  "use_data_augmentation": True,
  "use_mirror_loss": True,
  "mirror_loss_coeff": 0.5,
}


def booster_k1_dribble_ppo_runner_cfg(
  experiment_name: str = "k1_dribble",
  max_iterations: int = 20_000,
  gamma: float = 0.99,
) -> RslRlOnPolicyRunnerCfg:
  """PPO with a plain MLP actor over a 3-step observation history."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=SymmetricPpoAlgorithmCfg(
      symmetry_cfg=dict(SYMMETRY_CFG),
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=gamma,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name=experiment_name,
    # Local logging: nothing leaves the machine unless wandb is chosen.
    logger="tensorboard",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
