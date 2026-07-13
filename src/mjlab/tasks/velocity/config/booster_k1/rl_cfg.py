"""RL configuration for Booster K1 velocity task."""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw in (None, "") else float(raw)


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  return default if raw in (None, "") else int(raw)


def booster_k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster K1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
        # Hard sigma floor (STD_MIN): on the Nugus every late-run degradation
        # began after the action std sank below ~0.15 — the reward economics
        # pay for killing action noise until pushes become OOD. Clamp the
        # distribution's std_range instead of fighting through the entropy
        # bonus.
        "std_range": (_env_float("STD_MIN", 1e-6), 4.0),
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
      gamma=_env_float("GAMMA", 0.99),
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="k1_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=_env_int("MAX_ITERATIONS", 10_000),
    obs_norm_freeze_iters=_env_int("OBS_NORM_FREEZE_ITERS", 500),
  )
