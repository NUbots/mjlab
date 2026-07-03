"""Zero-action stand smoke test for NUgus flat velocity env.

PD holds the default keyframe with zero policy actions. Reset events
(``reset_base``, ``reset_robot_joints``) must remain — stripping all events
also removes state reset and produces garbage.

Pass: robot stands for at least 3 s without a fall termination.

Usage: ``uv run python scripts/tools/nugus_stand_smoke.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nugus_smoke_common import build_smoke_env

_STAND_DURATION_S = 3.0


def run_stand_smoke() -> tuple[bool, str]:
  env = build_smoke_env(strip_dr=False)
  try:
    env.reset(seed=0)
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device=env.device)
    target_steps = int(_STAND_DURATION_S / env.step_dt)

    for step in range(target_steps):
      _, _, terminated, time_outs, _ = env.step(zero_action)
      if bool((terminated | time_outs)[0].item()):
        reason = "fell over" if bool(terminated[0].item()) else "timed out"
        return False, f"{reason} at step {step + 1} ({(step + 1) * env.step_dt:.2f} s)"

    return True, f"stood {target_steps} steps ({target_steps * env.step_dt:.1f} s)"
  finally:
    env.close()


def main() -> int:
  passed, detail = run_stand_smoke()
  if passed:
    print(f"PASS: stand smoke — {detail}")
    return 0
  print(f"FAIL: stand smoke — {detail}", file=sys.stderr)
  return 1


if __name__ == "__main__":
  sys.exit(main())
