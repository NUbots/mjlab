"""RL configuration for the Booster K1 goalkeeper task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.obs_history import HistoryModelCfg


def booster_k1_block_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the K1 block policy.

  Same actor, critic and algorithm as the K1 velocity recipe, so the two policies
  export identically. Gamma is a little higher: a block is paid at the end of a shot
  that takes up to about 3 s, so the discount has to carry that far.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=HistoryModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
      history_cfg={
        "z_dim": 16,
        "tcn_channels": (32, 32),
        "tcn_kernel": 5,
        "tcn_stride": 2,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor", "history"), "critic": ("critic",)},
    experiment_name="k1_block",
    wandb_project="goalie",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
