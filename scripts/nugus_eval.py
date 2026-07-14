"""Fixed evaluation harness for NUgus velocity policies."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.nugus.env_cfgs import _PUSH_VELOCITY_RANGE_BASE
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Velocity-Flat-Nubots-Nugus"
COMMAND_GRID: tuple[tuple[float, float, float], ...] = (
  (0.3, 0.0, 0.0),
  (0.5, 0.0, 0.0),
  (0.75, 0.0, 0.0),
  (-0.3, 0.0, 0.0),
  (0.0, 0.3, 0.0),
  (0.0, -0.3, 0.0),
  (0.0, 0.0, 0.5),
  (0.0, 0.0, -0.5),
  (0.0, 0.0, 0.0),
  (0.5, 0.3, 0.5),
)
DEFAULT_ENVS_PER_COMMAND = 256
DEFAULT_EPISODE_LENGTH_S = 30.0
DEFAULT_SEED = 7
SWING_HEIGHT_TARGET_M = 0.08
FOOT_SITE_NAMES = ("left_foot", "right_foot")


def command_label(cmd: tuple[float, float, float]) -> str:
  """Stable metric key fragment for a velocity command tuple."""

  def _fmt(value: float) -> str:
    if value == 0.0:
      return "0"
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text

  vx, vy, wz = cmd
  return f"{_fmt(vx)}_{_fmt(vy)}_{_fmt(wz)}"


def default_num_envs(
  num_commands: int = len(COMMAND_GRID),
  envs_per_command: int = DEFAULT_ENVS_PER_COMMAND,
) -> int:
  return num_commands * envs_per_command


def build_eval_env_cfg(
  *,
  seed: int = DEFAULT_SEED,
  episode_length_s: float = DEFAULT_EPISODE_LENGTH_S,
  num_envs: int | None = None,
  envs_per_command: int = DEFAULT_ENVS_PER_COMMAND,
):
  """Training-equivalent env with fixed-eval overrides (keeps pushes + DR)."""
  from mjlab.envs import ManagerBasedRlEnvCfg

  cfg: ManagerBasedRlEnvCfg = load_env_cfg(TASK_ID, play=False)
  cfg.seed = seed
  cfg.episode_length_s = episode_length_s
  cfg.scene.num_envs = num_envs or default_num_envs(envs_per_command=envs_per_command)

  min_envs = len(COMMAND_GRID) * envs_per_command
  if cfg.scene.num_envs < min_envs:
    raise ValueError(
      f"num_envs={cfg.scene.num_envs} is below {min_envs} "
      f"({len(COMMAND_GRID)} commands × {envs_per_command} envs/command)."
    )

  twist_cmd = cfg.commands["twist"]
  if not isinstance(twist_cmd, UniformVelocityCommandCfg):
    raise TypeError("Expected UniformVelocityCommandCfg for twist command.")
  twist_cmd.resampling_time_range = (1e9, 1e9)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_stop_envs = 0.0
  twist_cmd.stop_ramp_time = 0.0
  twist_cmd.stop_settle_time = 0.0
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.heading = None
  twist_cmd.init_velocity_prob = 0.0

  if "push_robot" not in cfg.events:
    raise KeyError("Eval requires push_robot event (play mode disables pushes).")
  cfg.events["push_robot"].params["velocity_range"] = dict(_PUSH_VELOCITY_RANGE_BASE)

  # Curricula change push magnitudes and command ranges; eval fixes both explicitly.
  cfg.curriculum.clear()

  return cfg


def env_group_ids(
  num_envs: int,
  *,
  num_commands: int = len(COMMAND_GRID),
  envs_per_command: int = DEFAULT_ENVS_PER_COMMAND,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Map each env index to a command-grid group id."""
  env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
  return torch.clamp(env_ids // envs_per_command, max=num_commands - 1)


def apply_fixed_commands(
  twist: UniformVelocityCommand,
  *,
  command_grid: tuple[tuple[float, float, float], ...] = COMMAND_GRID,
  envs_per_command: int = DEFAULT_ENVS_PER_COMMAND,
  env_ids: torch.Tensor | None = None,
) -> None:
  """Pin velocity commands per env group and disable sampling modes."""
  if env_ids is None:
    env_ids = torch.arange(twist.num_envs, device=twist.device, dtype=torch.long)
  if len(env_ids) == 0:
    return

  group_ids = env_group_ids(
    twist.num_envs,
    num_commands=len(command_grid),
    envs_per_command=envs_per_command,
    device=twist.device,
  )[env_ids]

  twist.is_heading_env[env_ids] = False
  twist.is_world_env[env_ids] = False
  twist.is_forward_env[env_ids] = False
  twist.has_stop_tail[env_ids] = False
  twist.is_stop_ramping[env_ids] = False

  for cmd_idx, cmd in enumerate(command_grid):
    mask = group_ids == cmd_idx
    if not mask.any():
      continue
    matched_env_ids = env_ids[mask]
    cmd_tensor = torch.tensor(cmd, device=twist.device, dtype=twist.vel_command_b.dtype)
    twist.vel_command_b[matched_env_ids] = cmd_tensor
    twist.vel_command_w[matched_env_ids] = cmd_tensor
    twist.is_standing_env[matched_env_ids] = all(v == 0.0 for v in cmd)


@dataclass
class _StepAccumulator:
  lin_sq_sum: float = 0.0
  lin_count: int = 0
  ang_sq_sum: float = 0.0
  ang_count: int = 0
  slip_sum: float = 0.0
  slip_count: int = 0
  swing_err_sum: float = 0.0
  swing_count: int = 0
  fall_count: int = 0
  ep_len_sum_s: float = 0.0
  ep_count: int = 0

  def merge(self, other: _StepAccumulator) -> None:
    self.lin_sq_sum += other.lin_sq_sum
    self.lin_count += other.lin_count
    self.ang_sq_sum += other.ang_sq_sum
    self.ang_count += other.ang_count
    self.slip_sum += other.slip_sum
    self.slip_count += other.slip_count
    self.swing_err_sum += other.swing_err_sum
    self.swing_count += other.swing_count
    self.fall_count += other.fall_count
    self.ep_len_sum_s += other.ep_len_sum_s
    self.ep_count += other.ep_count


def _accumulator_to_metrics(acc: _StepAccumulator) -> dict[str, float]:
  sim_minutes = acc.ep_len_sum_s / 60.0 if acc.ep_len_sum_s > 0 else 0.0
  return {
    "eval/lin_vel_rmse": (
      (acc.lin_sq_sum / acc.lin_count) ** 0.5 if acc.lin_count else 0.0
    ),
    "eval/ang_vel_rmse": (
      (acc.ang_sq_sum / acc.ang_count) ** 0.5 if acc.ang_count else 0.0
    ),
    "eval/falls_per_min": (acc.fall_count / sim_minutes if sim_minutes > 0 else 0.0),
    "eval/mean_ep_len_s": (acc.ep_len_sum_s / acc.ep_count if acc.ep_count else 0.0),
    "eval/slip_vel": acc.slip_sum / acc.slip_count if acc.slip_count else 0.0,
    "eval/swing_height_err": (
      acc.swing_err_sum / acc.swing_count if acc.swing_count else 0.0
    ),
  }


@dataclass
class EvalMetricsState:
  """Per-command and global metric accumulators."""

  per_command: list[_StepAccumulator] = field(default_factory=list)

  def __post_init__(self) -> None:
    if not self.per_command:
      self.per_command = [_StepAccumulator() for _ in COMMAND_GRID]

  @property
  def global_acc(self) -> _StepAccumulator:
    total = _StepAccumulator()
    for acc in self.per_command:
      total.merge(acc)
    return total

  def to_dict(self) -> dict[str, object]:
    overall = _accumulator_to_metrics(self.global_acc)
    per_cmd: dict[str, dict[str, float]] = {}
    for cmd, acc in zip(COMMAND_GRID, self.per_command, strict=True):
      per_cmd[command_label(cmd)] = _accumulator_to_metrics(acc)
    return {"overall": overall, "per_command": per_cmd}

  def to_flat_dict(self) -> dict[str, float]:
    flat = {k: v for k, v in self.to_dict()["overall"].items()}  # type: ignore[misc]
    for cmd, metrics in self.to_dict()["per_command"].items():  # type: ignore[union-attr]
      for key, value in metrics.items():
        flat[f"{key}/cmd_{cmd}"] = value
    return flat


@dataclass(frozen=True)
class NugusEvalConfig:
  """Configuration for NUgus fixed evaluation."""

  checkpoint_file: str | None = None
  """Local path to a policy checkpoint (.pt)."""
  wandb_run_path: str | None = None
  """Optional W&B run path (entity/project/run_id) to load checkpoint and log Eval/*."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint filename within the W&B run."""
  seed: int = DEFAULT_SEED
  episode_length_s: float = DEFAULT_EPISODE_LENGTH_S
  envs_per_command: int = DEFAULT_ENVS_PER_COMMAND
  num_envs: int | None = None
  device: str | None = None
  output_file: str | None = None
  """Optional path to write JSON metrics."""


def _resolve_checkpoint(cfg: NugusEvalConfig, agent_cfg) -> Path:
  if cfg.checkpoint_file is not None:
    path = Path(cfg.checkpoint_file)
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path

  if cfg.wandb_run_path is None:
    raise ValueError("Provide checkpoint_file or wandb_run_path.")

  log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(
    log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  return resume_path


def run_nugus_eval(cfg: NugusEvalConfig) -> dict[str, object]:
  """Run fixed-grid evaluation and return nested metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = load_rl_cfg(TASK_ID)

  env_cfg = build_eval_env_cfg(
    seed=cfg.seed,
    episode_length_s=cfg.episode_length_s,
    num_envs=cfg.num_envs,
    envs_per_command=cfg.envs_per_command,
  )
  num_envs = env_cfg.scene.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  unwrapped = env.unwrapped

  resume_path = _resolve_checkpoint(cfg, agent_cfg)
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  # RMA checkpoints (train and eval with RMA=1 so the env groups and actor
  # class match): RMA_EVAL_PATH selects which latent drives the policy.
  #   teacher (default) — encoder(z) from the true DR params.
  #   student           — estimator z-hat from the obs history window.
  # The same command grid under both paths measures the adaptation gap,
  # which is the experiment's primary readout.
  actor = getattr(runner.alg, "_raw_actor", None)
  zhat_mix = getattr(actor, "zhat_mix", None)
  if zhat_mix is not None:
    eval_path = os.environ.get("RMA_EVAL_PATH", "teacher").strip().lower()
    if eval_path not in ("teacher", "student"):
      raise ValueError(f"RMA_EVAL_PATH must be teacher|student, got {eval_path!r}")
    zhat_mix.fill_(1.0 if eval_path == "student" else 0.0)
    print(f"[INFO] RMA eval path: {eval_path} (zhat_mix={float(zhat_mix):.1f})")

  # Odometry head (backlog 15d acceptance spec): v_hat RMSE conditioned on
  # post-push windows separately from steady gait — the push case is the
  # one that costs the localization stack most. Push firings are detected
  # from the event manager's per-env interval timer resetting upward.
  vel_head = getattr(actor, "vel_head", None)
  vhat_push_window_s = 1.0
  vhat_sq = {"push": 0.0, "steady": 0.0}
  vhat_n = {"push": 0, "steady": 0}
  push_time_left: torch.Tensor | None = None
  last_push_s: torch.Tensor | None = None
  vhat_estimator = None
  vhat_normalizer = None
  if vel_head is not None:
    assert actor is not None
    vhat_estimator = actor.estimator
    vhat_normalizer = actor.history_normalizer
    interval_names = unwrapped.event_manager.active_terms.get("interval", [])
    if "push_robot" in interval_names:
      push_idx = interval_names.index("push_robot")
      push_time_left = unwrapped.event_manager._interval_term_time_left[push_idx]
    last_push_s = torch.full((num_envs,), -1e9, device=device)
    print("[INFO] Odometry head found: reporting push-conditioned v_hat RMSE.")

  twist = cast(UniformVelocityCommand, unwrapped.command_manager.get_term("twist"))
  robot = unwrapped.scene["robot"]
  contact_sensor = cast(ContactSensor, unwrapped.scene["feet_ground_contact"])
  height_sensor = cast(TerrainHeightSensor, unwrapped.scene["foot_height_scan"])
  foot_site_ids, _ = robot.find_sites(FOOT_SITE_NAMES)

  groups = env_group_ids(num_envs, envs_per_command=cfg.envs_per_command, device=device)
  metrics = EvalMetricsState()
  peak_heights = torch.zeros(
    (num_envs, height_sensor.num_frames),
    device=device,
    dtype=torch.float32,
  )

  done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
  apply_fixed_commands(twist, envs_per_command=cfg.envs_per_command)
  unwrapped.command_manager.compute(dt=unwrapped.step_dt)

  obs = env.get_observations()
  step = 0
  print(
    f"[INFO] Evaluating {num_envs} envs "
    f"({len(COMMAND_GRID)} commands × {cfg.envs_per_command} envs) "
    f"for {cfg.episode_length_s:.0f}s episodes..."
  )

  first_contact_all = contact_sensor.compute_first_contact(dt=unwrapped.step_dt)

  while not done_envs.all():
    active = ~done_envs
    active_idx = active.nonzero(as_tuple=False).flatten()
    if active_idx.numel() > 0:
      assert contact_sensor.data.found is not None
      cmd = twist.vel_command_b[active_idx]
      lin_vel = robot.data.root_link_lin_vel_b[active_idx, :2]
      ang_vel = robot.data.root_link_ang_vel_b[active_idx, 2]
      lin_err = torch.norm(cmd[:, :2] - lin_vel, dim=-1)
      ang_err = torch.abs(cmd[:, 2] - ang_vel)

      if (
        vel_head is not None
        and last_push_s is not None
        and vhat_estimator is not None
        and vhat_normalizer is not None
      ):
        # v_hat sees the corrupted history stream; the target is ground
        # truth — exactly the deployment error the spec cares about.
        with torch.no_grad():
          zhat = vhat_estimator(vhat_normalizer(obs["history"][active_idx]))
          v_hat = vel_head(zhat)
        v_true = robot.data.root_link_lin_vel_b[active_idx]
        sq_err = torch.sum(torch.square(v_hat - v_true), dim=-1)
        now_s = step * unwrapped.step_dt
        recent_push = (now_s - last_push_s[active_idx]) <= vhat_push_window_s
        vhat_sq["push"] += sq_err[recent_push].sum().item()
        vhat_n["push"] += int(recent_push.sum().item())
        vhat_sq["steady"] += sq_err[~recent_push].sum().item()
        vhat_n["steady"] += int((~recent_push).sum().item())

      in_air = contact_sensor.data.found[active_idx] == 0
      foot_heights = height_sensor.data.heights[active_idx]
      peak_heights[active_idx] = torch.where(
        in_air,
        torch.maximum(peak_heights[active_idx], foot_heights),
        peak_heights[active_idx],
      )
      landing_err = torch.abs(peak_heights[active_idx] - SWING_HEIGHT_TARGET_M)
      first_contact = first_contact_all[active_idx]

      for i, env_idx in enumerate(active_idx):
        env_idx_int = int(env_idx)
        group_idx = int(groups[env_idx_int])
        acc = metrics.per_command[group_idx]
        acc.lin_sq_sum += lin_err[i].item() ** 2
        acc.lin_count += 1
        acc.ang_sq_sum += ang_err[i].item() ** 2
        acc.ang_count += 1

        in_contact = contact_sensor.data.found[env_idx_int] > 0
        if in_contact.any():
          foot_speed = torch.norm(
            robot.data.site_lin_vel_w[env_idx_int, foot_site_ids, :2], dim=-1
          )
          acc.slip_sum += foot_speed[in_contact].sum().item()
          acc.slip_count += int(in_contact.sum().item())

        landed = first_contact[i]
        if landed.any():
          acc.swing_err_sum += landing_err[i, landed].sum().item()
          acc.swing_count += int(landed.sum().item())

      peak_heights[active_idx] = torch.where(
        first_contact,
        torch.zeros_like(peak_heights[active_idx]),
        peak_heights[active_idx],
      )

    ep_len_before_step = unwrapped.episode_length_buf.clone()
    push_tl_before = push_time_left.clone() if push_time_left is not None else None
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    if (
      push_time_left is not None
      and push_tl_before is not None
      and last_push_s is not None
    ):
      # The interval timer resetting UPWARD means the push fired this
      # step (episode resets also resample it — exclude done envs).
      fired = (push_time_left > push_tl_before) & ~dones.bool()
      last_push_s[fired] = (step + 1) * unwrapped.step_dt
    apply_fixed_commands(twist, envs_per_command=cfg.envs_per_command)
    first_contact_all = contact_sensor.compute_first_contact(dt=unwrapped.step_dt)

    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      fell = unwrapped.termination_manager.get_term("fell_over")[newly_done]
      # auto_reset clears episode_length_buf before step() returns.
      ep_len_s = (ep_len_before_step[newly_done] + 1).float() * unwrapped.step_dt
      for env_idx, length, fell_over in zip(
        newly_done.nonzero(as_tuple=True)[0],
        ep_len_s,
        fell,
        strict=True,
      ):
        acc = metrics.per_command[int(groups[env_idx])]
        acc.ep_len_sum_s += length.item()
        acc.ep_count += 1
        if fell_over:
          acc.fall_count += 1
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{num_envs} episodes done "
        f"(step {step}, falls={int(fell.sum().item())})"
      )
    step += 1

  results = metrics.to_dict()
  overall = results["overall"]
  assert isinstance(overall, dict)

  if vel_head is not None:

    def _rmse(sq_sum: float, count: int) -> float:
      return (sq_sum / count) ** 0.5 if count else 0.0

    overall["eval/vhat_rmse"] = _rmse(
      vhat_sq["push"] + vhat_sq["steady"], vhat_n["push"] + vhat_n["steady"]
    )
    overall["eval/vhat_rmse_push"] = _rmse(vhat_sq["push"], vhat_n["push"])
    overall["eval/vhat_rmse_steady"] = _rmse(vhat_sq["steady"], vhat_n["steady"])

  print("\n" + "=" * 50)
  print("NUgus Eval Results")
  print("=" * 50)
  for name, value in overall.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  payload = {
    "task": TASK_ID,
    "checkpoint": str(resume_path),
    "seed": cfg.seed,
    "episode_length_s": cfg.episode_length_s,
    "num_envs": num_envs,
    "envs_per_command": cfg.envs_per_command,
    "command_grid": [list(cmd) for cmd in COMMAND_GRID],
    **results,
  }

  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
      json.dump(payload, handle, indent=2)
    print(f"[INFO] Metrics saved to {out}")

  if cfg.wandb_run_path is not None:
    api = wandb.Api()
    run = api.run(cfg.wandb_run_path)
    summary_update = {
      key.replace("eval/", "Eval/"): value for key, value in overall.items()
    }
    for cmd_label, cmd_metrics in results["per_command"].items():  # type: ignore[union-attr]
      assert isinstance(cmd_metrics, dict)
      for key, value in cmd_metrics.items():
        summary_update[f"Eval/{key.removeprefix('eval/')}/cmd_{cmd_label}"] = value
    run.summary.update(summary_update)
    run.update()
    print(f"[INFO] Updated W&B summary on {cfg.wandb_run_path}")

  env.close()
  return payload


def main() -> None:
  import mjlab
  import mjlab.tasks  # noqa: F401

  args = tyro.cli(NugusEvalConfig, config=mjlab.TYRO_FLAGS)
  if args.checkpoint_file is None and args.wandb_run_path is None:
    print("Error: provide --checkpoint-file or --wandb-run-path.", file=sys.stderr)
    sys.exit(1)

  results = run_nugus_eval(args)
  if args.output_file is None:
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
