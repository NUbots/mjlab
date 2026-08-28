"""Batched evaluation of a trained velocity policy.

Thin entry point over :mod:`mjlab.evaluation`, feeding the same metrics
recorder as ``eval_quintic_walk.py`` from the same raw simulator state, so the
two engines' numbers are comparable. See ``scripts/eval/README.md``.

The policy's observations are built by the task's own environment -- they are
noise-shaped, delayed, clock-augmented and normalised, and hand-rebuilding that
would be reimplementing the training code with different bugs. That is an
implementation detail of this script: the environment supplies observations and
actions, never measurements.

Examples::

  # Smoke test: 64 robots, 10 s, forward at 0.3 m/s, on the evaluation plant.
  uv run python scripts/eval/eval_rl_walk.py --num-envs 64 --duration 10 \\
    --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model_39997.pt

  # The other half of the 2x2.
  uv run python scripts/eval/eval_rl_walk.py --plant training \\
    --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model_39997.pt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.evaluation.harness import RlEvalHarness, command_grid
from mjlab.evaluation.live_view import (
  LIVE_VIEW_FLAGS,
  LiveViewCfg,
  open_live_view,
)
from mjlab.evaluation.metrics import format_summary, save_run
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Args:
  checkpoint: Path
  """rsl-rl checkpoint to evaluate, e.g.
  ``logs/rsl_rl/nugus_velocity/wandb_checkpoints/5l83efo3/model_39997.pt``.

  Must have been trained against the current task config: a checkpoint from an
  older observation layout either fails to load or loads and stands still. Smoke
  test a new checkpoint with a short run before trusting a long one."""
  plant: Literal["eval", "training"] = "eval"
  """Robot model. The NUbots MJCFs are quintic-only -- they do not carry the
  sensors and sites the policy's observations read."""
  num_envs: int = 512
  """Robots simulated in parallel."""
  duration: float = 20.0
  """Simulated seconds per robot."""
  device: str = "cuda:0"
  """Torch device."""

  vx: float = 0.2
  """Forward velocity command, in m/s. Ignored on any axis that is swept."""
  vy: float = 0.0
  wz: float = 0.0
  sweep_vx: tuple[float, ...] | None = None
  """Sweep forward velocity over these values instead of using ``--vx``."""
  sweep_vy: tuple[float, ...] | None = None
  sweep_wz: tuple[float, ...] | None = None

  warmup: float = 0.0
  """Seconds discarded from the front of the run before the walking metrics
  start averaging.

  A robot starts from standing, so a mean over the whole run reports the
  acceleration as well as the tracking. Survival is not windowed: a fall during
  the warm-up is still a fall, dated from the first step."""

  output_dir: Path = Path("logs/eval")
  """Runs land in ``<output_dir>/<tag>/``."""
  tag: str | None = None
  """Name for this run's output directory. Defaults to engine, plant and time."""

  live: LIVE_VIEW_FLAGS = field(default_factory=LiveViewCfg)
  """Live playback in the browser; off by default. See ``--viser``."""


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  if not args.checkpoint.exists():
    raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

  harness = RlEvalHarness(
    checkpoint=args.checkpoint,
    plant=args.plant,
    num_envs=args.num_envs,
    device=args.device,
  )
  command = command_grid(
    vx=args.sweep_vx or (args.vx,),
    vy=args.sweep_vy or (args.vy,),
    wz=args.sweep_wz or (args.wz,),
    num_envs=args.num_envs,
    device=args.device,
  )

  view = open_live_view(harness, args.live)
  started = time.time()
  try:
    metrics = harness.run(
      command,
      args.duration,
      on_step=None if view is None else view.on_step,
      warmup_s=args.warmup,
    )
  finally:
    if view is not None:
      view.close()
  elapsed = time.time() - started

  tag = args.tag or f"rl_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  output_dir = args.output_dir / tag
  run = {
    "engine": "rl",
    "plant": args.plant,
    "checkpoint": str(args.checkpoint),
    "num_envs": args.num_envs,
    "duration_s": args.duration,
    "warmup_s": args.warmup,
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "unique_commands": int(torch.unique(command, dim=0).shape[0]),
    "wall_time_s": round(elapsed, 1),
  }
  summary = save_run(output_dir, run, metrics.result())
  harness.close()

  print(f"\nRL policy on the {args.plant} plant")
  print(f"checkpoint        : {args.checkpoint}")
  print(format_summary(summary))
  print(
    f"wall time         : {elapsed:.1f} s for "
    f"{args.num_envs * args.duration:.0f} robot-seconds"
  )
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


if __name__ == "__main__":
  main()
