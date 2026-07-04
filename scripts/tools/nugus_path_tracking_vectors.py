"""Print the NUgus path tracking task's observation and action vectors.

Builds the NUgus path tracking env with a single environment, steps it once
with a zero action, and prints a per-term breakdown (name, dimension, values)
of the actor/critic observation vectors and the action vector. Useful for
seeing the exact input/output shapes the policy sees -- in particular the
path-tracking-specific terms: the actor's relative-waypoint ``command``
observation and the critic's clean ``command``/``target_twist`` terms.

Examples:
  uv run python scripts/tools/nugus_path_tracking_vectors.py
  uv run python scripts/tools/nugus_path_tracking_vectors.py --terrain rough
  uv run python scripts/tools/nugus_path_tracking_vectors.py --play
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro


@dataclass
class Args:
  terrain: Literal["flat", "rough"] = "flat"
  """Which env_cfgs builder to use."""

  play: bool = False
  """Build with play=True (play-mode overrides, e.g. observation corruption
  disabled and infinite episode length)."""


def _print_vector(title: str, terms: list[tuple[str, list[float]]]) -> None:
  print()
  print("=" * 78)
  total_dim = sum(len(values) for _, values in terms)
  print(f"{title} (total dim = {total_dim})")
  print("=" * 78)
  idx = 0
  for name, values in terms:
    formatted = ", ".join(f"{v:+.4f}" for v in values)
    print(
      f"[{idx:3d}:{idx + len(values):3d}] {name:<24s} "
      f"dim={len(values):<3d} [{formatted}]"
    )
    idx += len(values)


def main() -> None:
  args = tyro.cli(Args)

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.path_tracking.config.nugus.env_cfgs import (
    nubots_nugus_path_flat_env_cfg,
    nubots_nugus_path_rough_env_cfg,
  )

  cfg_fn = (
    nubots_nugus_path_flat_env_cfg
    if args.terrain == "flat"
    else nubots_nugus_path_rough_env_cfg
  )
  cfg = cfg_fn(play=args.play)
  cfg.scene.num_envs = 1

  device = "cuda" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  try:
    env.reset()
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((env.num_envs, action_dim), device=env.device)
    env.step(zero_action)

    groups: dict[str, list[tuple[str, list[float]]]] = {}
    for full_name, values in env.observation_manager.get_active_iterable_terms(0):
      group_name, term_name = full_name.split("-", 1)
      groups.setdefault(group_name, []).append((term_name, list(values)))
    for group_name, terms in groups.items():
      _print_vector(f"Observation vector: '{group_name}'", terms)

    action_terms = [
      (name, list(values))
      for name, values in env.action_manager.get_active_iterable_terms(0)
    ]
    _print_vector("Action vector", action_terms)
  finally:
    env.close()


if __name__ == "__main__":
  main()
