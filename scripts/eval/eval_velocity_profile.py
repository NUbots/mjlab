"""Velocity tracking under a moving command, for any of the three controllers.

The sweep scripts hold one command for a whole episode and report a mean, which
measures steady-state tracking. This one moves the command during the episode --
forward, then sideways, then turning, then the three combinations -- and records
the response step by step, which is the figure DeepWalk (Rodriguez and Behnke,
ICRA 2021, Fig. 3) uses to show a gait is omnidirectional.

Each of the six schedules runs in its own slice of the batch rather than as one
long sequence, so a controller that falls under a backwards command does not
drag the rest of the sequence down with it. See
:mod:`mjlab.evaluation.profile`.

Examples::

  # The walk engine on the evaluation plant.
  uv run python scripts/eval/eval_velocity_profile.py --engine quintic

  # The RL policy, four robots per schedule to show the observation noise.
  uv run python scripts/eval/eval_velocity_profile.py --engine rl \\
    --checkpoint logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model_39997.pt
"""

from __future__ import annotations

import json
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
from mjlab.evaluation.metrics import FALL_UPRIGHT_THRESHOLD, write_trace_csv
from mjlab.evaluation.profile import ProfileCfg, omnidirectional_profile
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

  profile: ProfileCfg = field(default_factory=ProfileCfg)
  """Command amplitudes and timing; see ``--profile.help``."""

  balance: bool = True
  """Quintic only: apply the FootController torso-orientation correction."""

  output_dir: Path = Path("logs/eval")
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

  profile = omnidirectional_profile(args.profile)
  harness = build_harness(args, profile.num_envs)
  schedule = profile.commands(harness.control_dt)

  view = open_live_view(harness, args.live)
  started = time.time()
  try:
    trace = harness.run_profile(
      schedule, on_step=None if view is None else view.on_step
    )
  finally:
    if view is not None:
      view.close()
  elapsed = time.time() - started

  tag = (
    args.tag or f"profile_{args.engine}_{args.plant}_{time.strftime('%Y%m%d_%H%M%S')}"
  )
  output_dir = args.output_dir / tag
  output_dir.mkdir(parents=True, exist_ok=True)
  write_trace_csv(output_dir / "trace.csv", trace)

  data = trace.result()
  fell = (data["upright"] < FALL_UPRIGHT_THRESHOLD).any(dim=0)
  run = {
    "engine": args.engine,
    "plant": args.plant,
    "task_id": args.task_id if args.engine == "rl" else None,
    "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
    "num_envs": profile.num_envs,
    "duration_s": round(profile.duration, 3),
    "control_hz": round(1.0 / harness.control_dt, 3),
    "device": args.device,
    "profile": asdict(args.profile),
    "lanes": [
      {"name": lane.name, "axes": list(lane.axes), "duration_s": lane.duration}
      for lane in profile.lanes
    ],
    "lane_of_env": list(profile.lane_of_env()),
    "num_fell": int(fell.sum()),
    "wall_time_s": round(elapsed, 1),
  }
  with (output_dir / "run.json").open("w") as handle:
    json.dump(run, handle, indent=2)
    handle.write("\n")

  print(f"\n{args.engine} velocity profile on the {args.plant} plant")
  print(f"lanes             : {', '.join(lane.name for lane in profile.lanes)}")
  print(f"environments      : {profile.num_envs} ({args.profile.replicas} per lane)")
  print(
    f"duration          : {profile.duration:.1f} s at {1 / harness.control_dt:.0f} Hz"
  )
  print(f"fell              : {int(fell.sum())} of {profile.num_envs}")
  print(f"wall time         : {elapsed:.1f} s")
  print(f"wrote             : {output_dir}/trace.csv, run.json")


if __name__ == "__main__":
  main()
