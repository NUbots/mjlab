"""Batched evaluation of a trained velocity policy.

Three modes, one per kind of figure ``plot_comparison.py`` draws:

``grid``
  Hold one command per robot over a grid of commands, and write the mean
  tracking error and whether the robot fell, per command. Two swept axes give
  one panel of the normalised error plane.
``profile``
  Move the command during the episode -- forward, sideways, turning, then the
  three pairs -- and record the response step by step. Drawn as the velocity
  profile.
``push``
  Shove the robot while it holds a command, over magnitude, direction and gait
  phase, and record whether it stayed up. Drawn as the push survival envelope.

The robot comes from ``--task-id`` (the Booster K1 flat task by default); the
checkpoint has to have been trained against that task's current config.

Examples::

  CKPT=logs/rsl_rl/k1_velocity/wandb_checkpoints/<run>/model_14999.pt

  # One plane of the grid.
  uv run python scripts/eval/eval_rl_walk.py grid --checkpoint $CKPT \\
    --vx "(-1.0,-0.5,0.0,0.5,1.0)" --vy "(-0.5,0.0,0.5)"

  # The velocity profile.
  uv run python scripts/eval/eval_rl_walk.py profile --checkpoint $CKPT

  # A coarse push battery, for a smoke test.
  uv run python scripts/eval/eval_rl_walk.py push --checkpoint $CKPT \\
    --push.delta-v "(0.5,1.0)" --push.directions 4 --push.phases 2 \\
    --push.replicas 1
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tyro

import mjlab
from mjlab.evaluation.harness import TASK_ID, RlEvalHarness, command_grid
from mjlab.evaluation.metrics import (
  FALL_UPRIGHT_THRESHOLD,
  save_run,
  write_trace_csv,
)
from mjlab.evaluation.profile import ProfileCfg, omnidirectional_profile
from mjlab.evaluation.push import (
  PerEnvPushMetrics,
  PushCfg,
  PushPlan,
  format_push_summary,
  run_push_battery,
  summarise_push,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Common:
  checkpoint: Path
  """rsl-rl checkpoint, e.g.
  ``logs/rsl_rl/k1_velocity/wandb_checkpoints/<run>/model_14999.pt``."""
  task_id: str = TASK_ID
  """Registered task the checkpoint was trained on."""
  device: str = "cuda:0"
  output_dir: Path = Path("logs/eval")
  """Runs land in ``<output_dir>/<tag>/``."""
  tag: str | None = None
  """Name for this run's output directory. Defaults to the mode and the time."""


@dataclass
class Grid(Common):
  vx: tuple[float, ...] = (0.0,)
  """Forward velocity commands, in m/s."""
  vy: tuple[float, ...] = (0.0,)
  """Lateral velocity commands, in m/s."""
  wz: tuple[float, ...] = (0.0,)
  """Yaw rate commands, in rad/s."""
  replicas: int = 1
  """Robots per command."""
  duration: float = 30.0
  """Simulated seconds per robot."""
  warmup: float = 8.0
  """Seconds kept out of the tracking averages, so they measure steady state
  rather than the acceleration from a stand. A fall during the warm-up still
  counts."""


@dataclass
class Profile(Common):
  profile: ProfileCfg = field(default_factory=ProfileCfg)
  """Command amplitudes and timing."""


@dataclass
class Push(Common):
  push: PushCfg = field(default_factory=PushCfg)
  """The battery: command, magnitudes, directions, phases and timing."""


def _output_dir(args: Common, mode: str) -> Path:
  return args.output_dir / (args.tag or f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}")


def _run_info(args: Common, harness: RlEvalHarness, elapsed: float) -> dict:
  return {
    "task_id": args.task_id,
    "checkpoint": str(args.checkpoint),
    "num_envs": harness.num_envs,
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "wall_time_s": round(elapsed, 1),
  }


def run_grid(args: Grid) -> None:
  num_envs = len(args.vx) * len(args.vy) * len(args.wz) * args.replicas
  harness = RlEvalHarness(args.checkpoint, num_envs, args.device, args.task_id)
  command = command_grid(args.vx, args.vy, args.wz, num_envs, args.device)

  started = time.time()
  metrics = harness.run(command, args.duration, warmup_s=args.warmup)
  run = {
    **_run_info(args, harness, time.time() - started),
    "duration_s": args.duration,
    "warmup_s": args.warmup,
  }
  output_dir = _output_dir(args, "grid")
  summary = save_run(output_dir, run, metrics.result())
  harness.close()

  error = summary["survivors_mean"]["tracking_error"]
  print(f"\ncommands          : {num_envs // args.replicas} x {args.replicas}")
  print(
    f"survived          : {summary['num_survived']} of {summary['num_envs']} "
    f"({100.0 * summary['survival_rate']:.1f}%)"
  )
  print(f"planar error      : {error:.3f} m/s mean over survivors")
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


def run_profile(args: Profile) -> None:
  profile = omnidirectional_profile(args.profile)
  harness = RlEvalHarness(args.checkpoint, profile.num_envs, args.device, args.task_id)
  schedule = profile.commands(harness.control_dt)

  started = time.time()
  trace = harness.run_profile(schedule)
  elapsed = time.time() - started
  harness.close()

  output_dir = _output_dir(args, "profile")
  write_trace_csv(output_dir / "trace.csv", trace)
  fell = (trace.result()["upright"] < FALL_UPRIGHT_THRESHOLD).any(dim=0)
  run = {
    **_run_info(args, harness, elapsed),
    "duration_s": round(profile.duration, 3),
    "profile": asdict(args.profile),
    "lanes": [
      {"name": lane.name, "axes": list(lane.axes), "duration_s": lane.duration}
      for lane in profile.lanes
    ],
    "lane_of_env": list(profile.lane_of_env()),
    "num_fell": int(fell.sum()),
  }
  with (output_dir / "run.json").open("w") as handle:
    json.dump(run, handle, indent=2)
    handle.write("\n")

  print(f"\nlanes             : {', '.join(lane.name for lane in profile.lanes)}")
  print(f"environments      : {profile.num_envs} ({args.profile.replicas} per lane)")
  print(f"fell              : {int(fell.sum())} of {profile.num_envs}")
  print(f"wrote             : {output_dir}/trace.csv, run.json")


def run_push(args: Push) -> None:
  cfg = args.push
  harness = RlEvalHarness(
    args.checkpoint, cfg.trials_per_pass, args.device, args.task_id
  )
  mass = harness.robot_mass()
  print(
    f"\npush battery      : {len(cfg.delta_v)} magnitudes x {cfg.directions} "
    f"directions x {cfg.phases} phases x {cfg.replicas} replicas "
    f"= {cfg.num_trials} trials, through the {harness.push_body_name} "
    f"({mass:.2f} kg robot)"
  )

  def report(index: int, plan: PushPlan, result: PerEnvPushMetrics) -> None:
    withstood = result.withstood[result.withstood.isfinite()]
    rate = float(withstood.mean()) if withstood.numel() else float("nan")
    print(
      f"  [{index + 1:>2}/{len(cfg.delta_v)}] dv {float(plan.delta_v[0]):.2f} m/s "
      f"({float(plan.impulse[0]):.2f} N s): {100.0 * rate:.1f}% withstood"
    )

  started = time.time()
  metrics = run_push_battery(harness, cfg, on_pass=report)
  run = {
    **_run_info(args, harness, time.time() - started),
    "robot_mass_kg": round(mass, 4),
    "push_body": harness.push_body_name,
    "num_trials": cfg.num_trials,
    "duration_s": round(cfg.settle + cfg.phase_window + cfg.recovery, 3),
    "push": asdict(cfg),
  }
  output_dir = _output_dir(args, "push")
  summary = save_run(output_dir, run, metrics, summarise_push(metrics, cfg))
  harness.close()

  print()
  print(format_push_summary(summary))
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


def main() -> None:
  args = tyro.extras.subcommand_cli_from_dict(
    {"grid": Grid, "profile": Profile, "push": Push},
    config=mjlab.TYRO_FLAGS,
  )
  configure_torch_backends()
  if isinstance(args, Grid):
    run_grid(args)
  elif isinstance(args, Profile):
    run_profile(args)
  else:
    run_push(args)


if __name__ == "__main__":
  main()
