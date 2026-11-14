"""Print the NUgus velocity task's observation and action vectors.

Builds the NUgus velocity env with a single environment, steps it once with a
zero action, and prints a per-term breakdown (name, dimension, values) of the
actor/critic observation vectors and the action vector. Useful for seeing
exactly how the env-var knobs read by
``mjlab.tasks.velocity.config.nugus.env_cfgs`` (MJLAB_VARIANT, CURRENT_OBS,
...) change the policy's input/output shapes -- e.g. CURRENT_OBS growing the
actor observation, or the clock_learned variant adding a ``phase_delta``
action term.

The most commonly toggled knobs get first-class flags below. Everything else
``env_cfgs.py`` reads via ``_env_float``/``_env_int``/``_env_bool`` (e.g.
GAIT_PERIOD, JOULE_W, PHASE_C_FRAC) can be set with the catch-all ``--env``
flag, so this script doesn't need updating when new knobs are added there.

Examples:
  uv run python scripts/tools/nugus_velocity_vectors.py
  uv run python scripts/tools/nugus_velocity_vectors.py --variant clock_learned
  uv run python scripts/tools/nugus_velocity_vectors.py --current-obs --terrain rough
  uv run python scripts/tools/nugus_velocity_vectors.py --env GAIT_PERIOD=0.9 --env JOULE_W=1e-3
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import tyro

import mjlab

# Extra knobs env_cfgs.py reads via os.environ but that don't have a
# dedicated flag below. Passed with --env KEY=VALUE.
_KNOWN_ENV_KEYS = (
  "GAIT_PERIOD",
  "JOULE_W",
  "PHASE_C_FRAC",
  "STAND_W",
  "EFFORT_LO",
  "EFFORT_HI",
  "RESAMPLE_MIN",
  "MAX_ITERATIONS",
  "PHASE_ITERATIONS",
  "PHASE_DELTA_STRONG_W",
  "PHASE_DELTA_STRONG_ITERS",
  "UPRIGHT_W",
  "SEED",
)


@dataclass
class Args:
  variant: Literal["clock_anneal", "self_paced", "clock_persist", "clock_learned"] = (
    "clock_anneal"
  )
  """MJLAB_VARIANT -- selects the gait-handoff curriculum."""

  terrain: Literal["flat", "rough"] = "flat"
  """Which env_cfgs builder to use."""

  play: bool = False
  """Build with play=True (play-mode overrides, e.g. observation corruption
  disabled and infinite episode length)."""

  current_obs: bool = False
  """CURRENT_OBS -- add the actuator-current observation term."""

  silence_clock: bool = False
  """SILENCE_CLOCK -- fade the gait-clock observation to zero over training
  (clock_anneal variant only)."""

  env: list[str] = field(default_factory=list)
  """Extra env-var overrides read by env_cfgs.py, as repeatable KEY=VALUE
  pairs, e.g. --env GAIT_PERIOD=0.9 --env JOULE_W=1e-3. See _KNOWN_ENV_KEYS
  in this file for the full list of recognized knobs."""


def _apply_env_overrides(args: Args) -> dict[str, str]:
  applied = {"MJLAB_VARIANT": args.variant}
  if args.current_obs:
    applied["CURRENT_OBS"] = "1"
  if args.silence_clock:
    applied["SILENCE_CLOCK"] = "1"
  for item in args.env:
    if "=" not in item:
      raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
    key, value = item.split("=", 1)
    key = key.strip()
    if key not in _KNOWN_ENV_KEYS:
      raise ValueError(
        f"Unknown env var {key!r} for --env. Known knobs: {_KNOWN_ENV_KEYS}"
      )
    applied[key] = value.strip()

  for key, value in applied.items():
    os.environ[key] = value
  return applied


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
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  applied = _apply_env_overrides(args)

  print("Environment overrides applied:")
  for key, value in applied.items():
    print(f"  {key}={value}")

  # Imported after the env vars are set: env_cfgs.py reads them lazily when
  # its cfg-building function runs, not at import time.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.nugus.env_cfgs import (
    nubots_nugus_flat_env_cfg,
    nubots_nugus_rough_env_cfg,
  )

  cfg_fn = (
    nubots_nugus_flat_env_cfg if args.terrain == "flat" else nubots_nugus_rough_env_cfg
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
