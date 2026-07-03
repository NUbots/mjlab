"""RL configuration for NUbots Nugus velocity task."""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw in (None, ""):
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw in (None, "") else float(raw)


def _symmetry_cfg() -> dict | None:
  if not _env_bool("MIRROR_AUG", default=False):
    return None
  return {
    "use_data_augmentation": True,
    "use_mirror_loss": False,
    "data_augmentation_func": (
      "mjlab.tasks.velocity.config.nugus.mirror_map:nugus_symmetry_augmentation"
    ),
  }


def nubots_nugus_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for NUbots Nugus velocity task."""
  gamma = _env_float("GAMMA", 0.99)
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
      gamma=gamma,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg=_symmetry_cfg(),
    ),
    experiment_name="nugus_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
  )
