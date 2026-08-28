"""Batched evaluation of NUbots' distilled quintic walk policy.

The third controller in the comparison: a small MLP trained to reproduce the
quintic walk engine's joint targets from the velocity command, the engine's
phase clock and its own three previous outputs. NUbots deploys it as
``module/skill/NeuralWalk``; the export it deploys is what runs here. Thin entry
point over :mod:`mjlab.evaluation`, so the plant, the metrics and the output
format are ``eval_quintic_walk.py``'s. See ``scripts/eval/README.md``.

The policy is blind -- no proprioception, no attitude, no contact -- so it needs
nothing of the plant that the walk engine does not, and runs on all four robot
models rather than the two the reinforcement-learned policy is wired to.

Examples::

  # Smoke test: 64 robots, 10 s, forward at 0.2 m/s, on the evaluation plant.
  uv run python scripts/eval/eval_distilled_quintic_walk.py --num-envs 64 \\
    --duration 10

  # The other half of the 2x2: same policy, the model policies train against.
  uv run python scripts/eval/eval_distilled_quintic_walk.py --plant training

  # How far the copy runs from the engine it copies.
  uv run python scripts/eval/eval_distilled_quintic_walk.py --track-teacher True
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.controllers.distilled_walk import DEFAULT_POLICY_PATH
from mjlab.controllers.distilled_walk.controller import HistoryInit
from mjlab.evaluation.harness import (
  DistilledEvalHarness,
  EvalPlant,
  command_grid,
)
from mjlab.evaluation.live_view import (
  LIVE_VIEW_FLAGS,
  LiveViewCfg,
  open_live_view,
)
from mjlab.evaluation.metrics import format_summary, save_run, write_summary_json
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Args:
  policy: Path = DEFAULT_POLICY_PATH
  """ONNX export to evaluate. Defaults to the copy of NUbots' deployed
  ``walk_policy.onnx`` that ships with mjlab."""
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

  history_init: HistoryInit = "settled"
  """How the run starts. ``settled`` puts the robot in the pose the policy
  holds at rest, which is the stance NUbots' training data was recorded from.
  ``stance`` starts from the walk engine's stance instead, as
  ``eval_quintic_walk.py`` does, and ``zeros`` reproduces the empty history
  ``NeuralWalk.cpp`` starts with. See the README."""
  track_teacher: bool = False
  """Run the walk engine alongside on the same commands and report how far the
  policy's joint targets ran from its own. Costs the engine's IK per step."""

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

  harness = DistilledEvalHarness(
    policy_path=args.policy,
    plant=args.plant,
    num_envs=args.num_envs,
    device=args.device,
    history_init=args.history_init,
    track_teacher=args.track_teacher,
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

  tag = args.tag or f"distilled_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  output_dir = args.output_dir / tag
  run = {
    "engine": "distilled",
    "plant": args.plant,
    "policy": str(args.policy),
    "num_envs": args.num_envs,
    "duration_s": args.duration,
    "warmup_s": args.warmup,
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "history_init": args.history_init,
    "unique_commands": int(torch.unique(command, dim=0).shape[0]),
    "wall_time_s": round(elapsed, 1),
  }
  summary = save_run(output_dir, run, metrics.result())
  tracking = harness.teacher_tracking()
  if tracking is not None:
    summary["teacher_tracking"] = tracking
    write_summary_json(output_dir / "summary.json", summary)

  print(f"\ndistilled quintic walk on the {args.plant} plant")
  print(f"policy            : {args.policy}")
  print(f"history init      : {args.history_init}")
  print(format_summary(summary))
  print(f"engine states     : {harness.engine_state_counts()}")
  if tracking is not None:
    print(
      f"teacher tracking  : {tracking['mean_abs_error_rad']:.4f} rad mean, "
      f"{tracking['stance_relative_mean_abs_error_rad']:.4f} rad about each "
      "controller's own stance"
    )
  print(
    f"wall time         : {elapsed:.1f} s for "
    f"{args.num_envs * args.duration:.0f} robot-seconds"
  )
  print(f"wrote             : {output_dir}/per_env.csv, summary.json")


if __name__ == "__main__":
  main()
