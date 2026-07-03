"""Shared helpers for NUgus pre-launch smoke tests."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.config.nugus.env_cfgs import nubots_nugus_flat_env_cfg

_CORE_RESET_EVENTS = frozenset(
  {"reset_scene_to_default", "reset_base", "reset_robot_joints"}
)


def build_smoke_env(*, strip_dr: bool = False) -> ManagerBasedRlEnv:
  """Build a single-env flat NUgus env for smoke diagnostics.

  Always removes interval pushes and effort drift. When ``strip_dr`` is set,
  only core reset events remain (matches the post-mortem scratchpad tests).
  """
  cfg = nubots_nugus_flat_env_cfg()
  cfg.scene.num_envs = 1
  cfg.seed = 0
  cfg.events.pop("push_robot", None)
  cfg.events.pop("effort_drift", None)
  if strip_dr:
    for name in list(cfg.events):
      if name not in _CORE_RESET_EVENTS:
        event = cfg.events.pop(name, None)
        if event is not None and not isinstance(event, EventTermCfg):
          raise TypeError(f"unexpected event type for {name!r}")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    return ManagerBasedRlEnv(cfg=cfg, device=device)
