"""RL configuration for NUbots Nugus velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.obs_history import HistoryModelCfg


def nubots_nugus_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for NUbots Nugus velocity task."""
  return RslRlOnPolicyRunnerCfg(
    # The actor consumes the current observation plus the "history" window
    # group. The TCN that encodes the window lives inside the model, so the
    # latent never crosses the observation boundary. See mjlab.rl.obs_history.
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
      gamma=0.975,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor", "history"), "critic": ("critic",)},
    experiment_name="nugus_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
