"""Batched evaluation of the ported NUbots quintic walk engine.

Thin entry point over :mod:`mjlab.evaluation`: the plant, the harness, the
metrics and the output format are shared with ``eval_rl_walk.py`` so the two
engines are measured by the same code. See ``scripts/eval/README.md``.

Examples::

  # Smoke test: 64 robots, 10 s, forward at 0.2 m/s, on the evaluation plant.
  uv run python scripts/eval/eval_quintic_walk.py --num-envs 64 --duration 10

  # The other half of the 2x2: same engine, the model policies train against.
  uv run python scripts/eval/eval_quintic_walk.py --plant training

  # Sweep forward speed across the batch.
  uv run python scripts/eval/eval_quintic_walk.py --num-envs 2048 \\
    --sweep-vx "(0.1,0.2,0.3,0.4,0.5)"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.controllers.quintic_walk.walk_generator import NUGUS_WALK_PARAMETERS
from mjlab.evaluation.harness import (
  EvalPlant,
  QuinticEvalHarness,
  command_grid,
)
from mjlab.evaluation.live_view import (
  LIVE_VIEW_FLAGS,
  LiveViewCfg,
  open_live_view,
)
from mjlab.evaluation.metrics import format_summary, save_run
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Args:
  plant: EvalPlant = "eval"
  """Robot model. ``eval`` is the reference; ``training`` is what policies are
  trained against; ``nubots-sim`` and ``nubots-xml`` are NUbots' own models."""
  num_envs: int = 512
  """Robots simulated in parallel."""
  duration: float = 20.0
  """Simulated seconds per robot."""
  device: str = "cuda:0"
  """Torch device. Use ``cpu`` for a tiny run without a GPU."""

  vx: float = 0.2
  """Forward velocity command, in m/s. Ignored on any axis that is swept."""
  vy: float = 0.0
  """Lateral velocity command, in m/s."""
  wz: float = 0.0
  """Yaw rate command, in rad/s."""
  sweep_vx: tuple[float, ...] | None = None
  """Sweep forward velocity over these values instead of using ``--vx``."""
  sweep_vy: tuple[float, ...] | None = None
  sweep_wz: tuple[float, ...] | None = None

  balance: bool = True
  """Apply the FootController torso-orientation correction, as deployed."""
  exact_ik: bool = False
  """Solve the legs against the compiled geometry rather than the engine's
  idealised leg. Not the deployed behaviour."""
  switch_when_planted: bool = False
  """Wait for sensed foot contact before switching the planted foot.

  What ``Walk.yaml`` asks for and what the deployed binary fails to apply. Off
  by default to match the robot; see the README before turning it on."""

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

  params = NUGUS_WALK_PARAMETERS
  if args.switch_when_planted:
    params = replace(params, only_switch_when_planted=True)

  harness = QuinticEvalHarness(
    plant=args.plant,
    num_envs=args.num_envs,
    device=args.device,
    walk_params=params,
    use_balance_control=args.balance,
    exact_ik=args.exact_ik,
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

  tag = args.tag or f"quintic_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  output_dir = args.output_dir / tag
  run = {
    "engine": "quintic",
    "plant": args.plant,
    "num_envs": args.num_envs,
    "duration_s": args.duration,
    "warmup_s": args.warmup,
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "balance": args.balance,
    "exact_ik": args.exact_ik,
    "only_switch_when_planted": params.only_switch_when_planted,
    "unique_commands": int(torch.unique(command, dim=0).shape[0]),
    "wall_time_s": round(elapsed, 1),
  }
  summary = save_run(output_dir, run, metrics.result())

  print(f"\nquintic walk on the {args.plant} plant")
  print(format_summary(summary))
  print(f"engine states     : {harness.engine_state_counts()}")
  print(
    f"wall time         : {elapsed:.1f} s for "
    f"{args.num_envs * args.duration:.0f} robot-seconds"
  )
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


if __name__ == "__main__":
  main()
