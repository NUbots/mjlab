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
    experiment_name="nugus_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
  )


def nubots_nugus_history_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The same run, with an observation-history encoder inside the actor.

  Differs from :func:`nubots_nugus_ppo_runner_cfg` in the actor only: it also
  reads the ``"history"`` observation group (a window of the actor stream, see
  ``add_actor_history``) and compresses it with a TCN whose latent is
  concatenated onto the current observation before the policy MLP. The encoder
  lives inside the model, so the latent never crosses the observation boundary
  and plain PPO trains it end to end. See :mod:`mjlab.rl.obs_history`.

  Every other hyperparameter is taken from the plain config rather than
  restated, so retuning the run retunes both.
  """
  cfg = nubots_nugus_ppo_runner_cfg()
  base = cfg.actor
  cfg.actor = HistoryModelCfg(
    hidden_dims=base.hidden_dims,
    activation=base.activation,
    obs_normalization=base.obs_normalization,
    distribution_cfg=dict(base.distribution_cfg or {}),
    history_cfg={
      "z_dim": 16,
      "tcn_channels": (32, 32),
      "tcn_kernel": 5,
      "tcn_stride": 2,
    },
  )
  cfg.obs_groups = {"actor": ("actor", "history"), "critic": ("critic",)}
  return cfg
