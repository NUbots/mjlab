"""Convert a NUgus velocity checkpoint into a path-tracking warm start.

The path-tracking runner already splices velocity checkpoints automatically
on resume (shared observation terms copied, the new waypoint command and
critic ``target_twist`` columns zero-initialized), so training can be
warm-started directly from a velocity run on W&B:

  uv run train Mjlab-PathTracking-Flat-Nubots-Nugus \\
    --agent.resume True --wandb-run-path <entity>/nugus_velocity/<run_id>

This script covers the local-file case: ``--agent.load-run`` only searches
the path-tracking experiment directory, so a velocity checkpoint on disk
must first be converted into a path-tracking one. It builds the
path-tracking runner, loads the velocity checkpoint through the splicing
``load()``, and saves the result where resume can find it.

Example:
  uv run python scripts/tools/warmstart_nugus_path_from_velocity.py \\
    --checkpoint logs/rsl_rl/nugus_velocity/2026-05-13_17-56-22/model_8500.pt \\
    --output logs/rsl_rl/nugus_path_tracking/warmstart/model_0.pt

  # Then resume path-tracking training from it:
  uv run train Mjlab-PathTracking-Flat-Nubots-Nugus \\
    --agent.resume True --agent.load-run warmstart
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro


@dataclass
class Args:
  checkpoint: str
  """Path to the source nugus velocity checkpoint (.pt)."""

  output: str
  """Path to write the warm-start path-tracking checkpoint (.pt). To resume
  from it, place it under logs/rsl_rl/nugus_path_tracking/<run_name>/ and
  pass --agent.load-run <run_name> to train."""

  terrain: Literal["flat", "rough"] = "flat"
  """Which path-tracking task variant to build the runner for."""


def main() -> None:
  args = tyro.cli(Args)

  import mjlab.tasks  # noqa: F401  (registers the tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.path_tracking.rl import PathTrackingOnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  task_id = f"Mjlab-PathTracking-{args.terrain.capitalize()}-Nubots-Nugus"
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = PathTrackingOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    # Splices the velocity checkpoint onto the path-tracking layout (or
    # loads it directly if it already matches).
    runner.load(args.checkpoint, map_location=device)

    saved = runner.alg.save()  # Spliced weights, fresh optimizer state.
    saved["iter"] = 0
    saved["infos"] = None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, out)
    print(f"Wrote warm-start checkpoint to {out}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
