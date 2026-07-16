"""RL configuration for NUbots Nugus velocity task."""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.memory import MemoryModelCfg, MemoryPpoAlgorithmCfg
from mjlab.rl.rma import RmaModelCfg, RmaPpoAlgorithmCfg


def _env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw in (None, ""):
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name, "")
  return int(raw) if raw.strip() else default


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
  rma = _env_bool("RMA", default=False)
  rnn_memory = _env_bool("RNN_MEMORY", default=False)
  if rma and rnn_memory:
    raise ValueError("RMA and RNN_MEMORY are mutually exclusive")

  distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "log",
    # Hard sigma floor (STD_MIN): on the corrected physics every late-run
    # degradation began after std sank below ~0.15 regardless of entropy
    # coefficient (v16c held coef 0.01 and std was still crushed to
    # 0.087) — the reward economics pay for killing action noise until
    # the policy overfits a narrow action tube and ordinary pushes
    # become OOD. Clamp the distribution's std_range instead of fighting
    # the economics through the entropy bonus.
    "std_range": (_env_float("STD_MIN", 1e-6), 4.0),
  }

  if rma:
    # RMA adaptation module (rl/rma.py): custom actor with the DR-param
    # encoder + history estimator, custom PPO with the concurrent
    # regression loss, and the extra obs groups routed to the actor set.
    vhat = _env_bool("RMA_VHAT", default=False)
    actor_cfg: RslRlModelCfg = RmaModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=distribution_cfg,
      class_name="mjlab.rl.rma:RmaActor",
      rma_cfg={
        "z_dim": _env_int("RMA_Z_DIM", 16),
        "encoder_hidden_dims": (128, 64),
        "tcn_channels": (32, 32),
        "tcn_kernel": 5,
        "tcn_stride": 2,
        "e2e": _env_bool("RMA_E2E", default=False),
        # Gated dual-channel mode (backlog 15d full design): PPO-trained
        # z_fast + gated slow sysid channel with a learned safe prior and
        # a 17-float deployment hold (identify walking, hold standing).
        "gated": _env_bool("RMA_GATED", default=False),
        "z_fast_dim": _env_int("RMA_ZFAST_DIM", 16),
        "hold_decay": _env_float("RMA_HOLD_DECAY", 0.9995),
        # Odometry head (backlog 15d): v_hat readout from the policy
        # trunk's penultimate features, exported as an ONNX output
        # "velocity". Detached by default so the head is a pure probe and
        # cannot perturb the walk.
        "vhat": vhat,
        "vhat_detach": _env_bool("RMA_VHAT_DETACH", default=True),
      },
    )
    algorithm_cfg: RslRlPpoAlgorithmCfg = RmaPpoAlgorithmCfg(
      class_name="mjlab.rl.rma:RmaPPO",
      est_loss_coef=_env_float("RMA_EST_COEF", 1.0),
      vel_loss_coef=_env_float("RMA_VHAT_COEF", 1.0),
      gate_loss_coef=_env_float("RMA_GATE_COEF", 1.0),
    )
    actor_groups = ["actor", "dr", "history"]
    if vhat:
      actor_groups.append("odom_target")
    obs_groups = {
      "actor": tuple(actor_groups),
      "critic": ("critic",),
    }
  elif rnn_memory:
    # Reward-driven recurrent memory (rl/memory.py, v59 line): a GRU
    # whose hidden state replaces the RMA history window — what to
    # remember and how long to hold it are learned from reward via
    # truncated BPTT. Mirror augmentation is retained through the
    # DEFINED latent mirror (swap hidden halves); MemoryPPO bypasses
    # rsl_rl's symmetry-vs-recurrence guard with RecurrentSymmetry.
    vhat = _env_bool("RMA_VHAT", default=False)
    actor_cfg = MemoryModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=distribution_cfg,
      class_name="mjlab.rl.memory:GruMemoryActor",
      rnn_type="gru",
      rnn_hidden_dim=_env_int("RNN_HIDDEN", 256),
      rnn_num_layers=_env_int("RNN_LAYERS", 1),
      memory_cfg={
        "vhat": vhat,
        "vhat_detach": _env_bool("RMA_VHAT_DETACH", default=True),
      },
    )
    algorithm_cfg = MemoryPpoAlgorithmCfg(
      class_name="mjlab.rl.memory:MemoryPPO",
      vel_loss_coef=_env_float("RMA_VHAT_COEF", 1.0),
    )
    memory_groups = ["actor"]
    if vhat:
      memory_groups.append("odom_target")
    obs_groups = {
      "actor": tuple(memory_groups),
      "critic": ("critic",),
    }
  else:
    actor_cfg = RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=distribution_cfg,
    )
    algorithm_cfg = RslRlPpoAlgorithmCfg()
    obs_groups = {"actor": ("actor",), "critic": ("critic",)}

  algorithm_cfg.value_loss_coef = 1.0
  algorithm_cfg.use_clipped_value_loss = True
  algorithm_cfg.clip_param = 0.2
  algorithm_cfg.entropy_coef = 0.01
  algorithm_cfg.num_learning_epochs = 5
  algorithm_cfg.num_mini_batches = 4
  algorithm_cfg.learning_rate = 1.0e-3
  algorithm_cfg.schedule = "adaptive"
  algorithm_cfg.gamma = gamma
  algorithm_cfg.lam = 0.95
  algorithm_cfg.desired_kl = 0.01
  algorithm_cfg.max_grad_norm = 1.0
  algorithm_cfg.symmetry_cfg = _symmetry_cfg()

  return RslRlOnPolicyRunnerCfg(
    actor=actor_cfg,
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=algorithm_cfg,
    obs_groups=obs_groups,
    experiment_name="nugus_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
    obs_norm_freeze_iters=_env_int("OBS_NORM_FREEZE_ITERS", 500),
    freeze_policy_after=_env_int("FREEZE_POLICY_AFTER", 0),
  )
