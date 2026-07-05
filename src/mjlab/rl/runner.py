import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    self._install_landing_anneal()

  def _install_landing_anneal(self) -> None:
    """Scale desired_kl by the env-computed landing factor before each
    PPO update (doc 15 R14).

    Task-side relief cannot stop optimizer churn once it begins (five
    demonstrations, v24-v30); the landing anneal is the optimizer-side
    lever: when the curriculum reports at-capacity + plateaued, the env
    sets ``_landing_factor`` < 1 and the adaptive-KL schedule chases the
    shrinking target, walking the learning rate to its floor -
    convergence instead of churn fuel. No-op unless the env publishes a
    factor and the algorithm uses an adaptive KL schedule.
    """
    alg = getattr(self, "alg", None)
    if alg is None or getattr(alg, "desired_kl", None) is None:
      return
    base_kl = float(alg.desired_kl)
    orig_update = alg.update
    unwrapped = getattr(self.env, "unwrapped", None)

    def update_with_landing(*args, **kwargs):
      factor = getattr(unwrapped, "_landing_factor", 1.0)
      alg.desired_kl = max(base_kl * float(factor), 2e-4)
      return orig_update(*args, **kwargs)

    alg.update = update_with_landing

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @property
  def _is_wandb_logger(self) -> bool:
    """Whether the active logger uploads to W&B.

    rsl-rl>=5.1 renamed the wandb ``logger_type`` from ``"wandb"`` to
    ``"WandbLogWriter"``; accept both so ONNX uploads keep working across
    versions.
    """
    return self.logger.logger_type in ("wandb", "WandbLogWriter")

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def _capture_curriculum_snapshot(self) -> dict:
    """Capture curriculum-derived MDP state for resume validation."""
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    env = self.env.unwrapped
    snapshot: dict = {"common_step_counter": env.common_step_counter}
    twist = env.command_manager.get_term("twist")
    if isinstance(twist, UniformVelocityCommand):
      ranges = twist.cfg.ranges
      snapshot["command_ranges"] = {
        "lin_vel_x": ranges.lin_vel_x,
        "lin_vel_y": ranges.lin_vel_y,
        "ang_vel_z": ranges.ang_vel_z,
      }
    reward_weights: dict[str, float] = {}
    for name in ("phase_delta_nominal", "upright", "track_linear_velocity"):
      if name in env.reward_manager.active_terms:
        reward_weights[name] = env.reward_manager.get_term_cfg(name).weight
    if reward_weights:
      snapshot["reward_weights"] = reward_weights
    return snapshot

  @staticmethod
  def _log_curriculum_snapshot_comparison(
    saved: dict | None, loaded: dict, *, checkpoint_path: str
  ) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
      return
    print(f"[resume] Curriculum snapshot from checkpoint {checkpoint_path}:")
    if saved is None:
      print("  (no curriculum_snapshot in checkpoint — pre-fix or legacy save)")
      print(f"  restored common_step_counter={loaded['common_step_counter']}")
      return
    for key in ("common_step_counter", "command_ranges", "reward_weights"):
      if saved.get(key) != loaded.get(key):
        print(f"  MISMATCH {key}: saved={saved.get(key)!r} loaded={loaded.get(key)!r}")
      else:
        print(f"  OK {key}={saved.get(key)!r}")

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {
      "common_step_counter": self.env.unwrapped.common_step_counter,
      "curriculum_snapshot": self._capture_curriculum_snapshot(),
    }
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      env_state = infos["env_state"]
      self.env.unwrapped.common_step_counter = env_state["common_step_counter"]
      self.env.unwrapped.curriculum_manager.compute()
      saved_snapshot = env_state.get("curriculum_snapshot")
      loaded_snapshot = self._capture_curriculum_snapshot()
      self._log_curriculum_snapshot_comparison(
        saved_snapshot, loaded_snapshot, checkpoint_path=path
      )
      # Re-reset so managers, commands, and events initialize under the
      # restored counter. The wrapper's constructor already reset at step 0,
      # which leaves stale state when resuming mid-curriculum (e.g.
      # hard_continue with stage thresholds above zero).
      if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("[resume] Re-resetting env after curriculum sync")
      self.env.reset()
    return infos
