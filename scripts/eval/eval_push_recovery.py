"""Push recovery for any of the three controllers.

The sweeps ask how fast a controller can walk and the profile asks how well it
follows a command that moves. Neither disturbs the robot. This one shoves it: a
constant force through the torso for a fifth of a second, swept over magnitude,
direction and gait phase, with the outcome measured over the seconds after.

The battery runs one magnitude at a time, so the batch size is set by the
direction, phase and replica counts alone and does not grow when the magnitude
axis is refined. Every trial in a batch is a distinct (direction, phase,
replica), and the results are concatenated into one table of trials. See
:mod:`mjlab.evaluation.push`.

Examples::

  # The walk engine, walking forward at 0.2 m/s when it is pushed.
  uv run python scripts/eval/eval_push_recovery.py --engine quintic

  # The same battery from a stand.
  uv run python scripts/eval/eval_push_recovery.py --engine quintic --push.vx 0.0

  # A policy, on a coarse battery for a smoke test.
  uv run python scripts/eval/eval_push_recovery.py --engine rl \\
    --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model_39997.pt \\
    --push.delta-v "(0.2,0.4,0.6)" --push.directions 4 --push.phases 2 \\
    --push.replicas 1
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.evaluation.harness import (
  TASK_ID,
  DistilledEvalHarness,
  EvalPlant,
  QuinticEvalHarness,
  RlEvalHarness,
)
from mjlab.evaluation.live_view import (
  LIVE_VIEW_FLAGS,
  LiveViewCfg,
  open_live_view,
)
from mjlab.evaluation.metrics import save_run
from mjlab.evaluation.push import (
  PUSH_BODY,
  PerEnvPushMetrics,
  PushCfg,
  PushPlan,
  format_push_summary,
  run_push_battery,
  summarise_push,
)
from mjlab.utils.torch import configure_torch_backends

Engine = Literal["quintic", "distilled", "rl"]


@dataclass
class Args:
  engine: Engine = "quintic"
  """Which controller to drive."""
  checkpoint: Path | None = None
  """rsl-rl checkpoint. Required for ``--engine rl``, ignored otherwise."""
  task_id: str = TASK_ID
  """Registered task supplying the policy's observation, action and command
  pipeline. ``--engine rl`` only.

  A checkpoint only loads against the task it was trained on, so a policy with
  a different observation layout -- one reading a window of past observations,
  say -- needs the task that builds that layout named here."""
  plant: EvalPlant = "eval"
  """Robot model. The RL policy runs only on ``eval`` and ``training``."""
  device: str = "cuda:0"
  """Torch device."""

  push: PushCfg = field(default_factory=PushCfg)
  """The battery: magnitudes, directions, phases and timing. See
  ``--push.help``."""

  balance: bool = True
  """Quintic only: apply the FootController torso-orientation correction."""

  output_dir: Path = Path("logs/eval")
  """Runs land in ``<output_dir>/<tag>/``."""
  tag: str | None = None
  """Name for this run's output directory. Defaults to engine, plant and time."""

  live: LIVE_VIEW_FLAGS = field(default_factory=LiveViewCfg)
  """Live playback in the browser; off by default."""


def build_harness(args: Args, num_envs: int):
  if args.engine == "rl":
    if args.checkpoint is None:
      raise ValueError("--engine rl needs --checkpoint")
    if not args.checkpoint.exists():
      raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    return RlEvalHarness(
      checkpoint=args.checkpoint,
      plant=args.plant,
      num_envs=num_envs,
      device=args.device,
      task_id=args.task_id,
    )
  if args.engine == "distilled":
    return DistilledEvalHarness(plant=args.plant, num_envs=num_envs, device=args.device)
  return QuinticEvalHarness(
    plant=args.plant,
    num_envs=num_envs,
    device=args.device,
    use_balance_control=args.balance,
  )


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  cfg = args.push
  harness = build_harness(args, cfg.trials_per_pass)
  mass = harness.robot_mass()
  print(
    f"\n{args.engine} push recovery on the {args.plant} plant "
    f"({mass:.2f} kg, pushed through the {PUSH_BODY})"
  )
  print(
    f"battery           : {len(cfg.delta_v)} magnitudes x {cfg.directions} "
    f"directions x {cfg.phases} phases x {cfg.replicas} replicas "
    f"= {cfg.num_trials} trials"
  )
  print(f"per cell          : {cfg.trials_per_cell} trials")

  view = open_live_view(harness, args.live)
  started = time.time()

  def report(index: int, plan: PushPlan, result: PerEnvPushMetrics) -> None:
    withstood = result.withstood[result.withstood.isfinite()]
    rate = float(withstood.mean()) if withstood.numel() else float("nan")
    print(
      f"  [{index + 1:>2}/{len(cfg.delta_v)}] "
      f"dv {float(plan.delta_v[0]):.2f} m/s "
      f"({float(plan.impulse[0]):.2f} N s, {float(plan.force[0]):.1f} N): "
      f"{100.0 * rate:.1f}% withstood"
    )

  try:
    # The live view watches one environment for every pass of the battery; it
    # is for eyeballing a shove, not for collecting.
    metrics = run_push_battery(
      harness,
      cfg,
      on_pass=report,
      on_step=None if view is None else view.on_step,
    )
  finally:
    if view is not None:
      view.close()
  elapsed = time.time() - started

  tag = args.tag or f"push_{args.engine}_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  output_dir = args.output_dir / tag
  run = {
    "engine": args.engine,
    "plant": args.plant,
    "task_id": args.task_id if args.engine == "rl" else None,
    "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
    "robot_mass_kg": round(mass, 4),
    "push_body": PUSH_BODY,
    "num_envs": cfg.trials_per_pass,
    "num_trials": cfg.num_trials,
    "duration_s": round(cfg.settle + cfg.phase_window + cfg.recovery, 3),
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "push": asdict(cfg),
    "wall_time_s": round(elapsed, 1),
  }
  summary = save_run(output_dir, run, metrics, summarise_push(metrics, cfg))
  if isinstance(harness, RlEvalHarness):
    harness.close()

  print()
  print(format_push_summary(summary))
  print(
    f"wall time         : {elapsed:.1f} s for "
    f"{cfg.num_trials * run['duration_s']:.0f} robot-seconds"
  )
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


if __name__ == "__main__":
  main()
