"""Random-action survival smoke test for NUgus flat velocity env.

Samples N(0, 1) actions (approximates an untrained policy). Core reset events
are kept so each episode starts from a valid keyframe; DR and pushes are
stripped to isolate actuator exploration tail (post-mortem scratchpad pattern).

Pass: mean episode length exceeds 40 control steps.

Usage: ``uv run python scripts/tools/nugus_noise_smoke.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nugus_smoke_common import build_smoke_env

_MIN_MEAN_SURVIVAL = 40
_NUM_EPISODES = 20


def run_noise_smoke(
  *,
  num_episodes: int = _NUM_EPISODES,
  min_mean_survival: int = _MIN_MEAN_SURVIVAL,
) -> tuple[bool, str]:
  env = build_smoke_env(strip_dr=True)
  try:
    action_dim = env.action_manager.total_action_dim
    episode_lengths: list[int] = []

    for ep in range(num_episodes):
      env.reset(seed=ep)
      length = 0
      while True:
        action = torch.randn(1, action_dim, device=env.device)
        _, _, terminated, time_outs, _ = env.step(action)
        length += 1
        if bool((terminated | time_outs)[0].item()):
          break
      episode_lengths.append(length)

    mean_len = sum(episode_lengths) / len(episode_lengths)
    max_len = max(episode_lengths)
    detail = (
      f"mean={mean_len:.1f} max={max_len} over {num_episodes} episodes "
      f"(threshold >{min_mean_survival})"
    )
    if mean_len <= min_mean_survival:
      return False, detail
    return True, detail
  finally:
    env.close()


def main() -> int:
  passed, detail = run_noise_smoke()
  if passed:
    print(f"PASS: noise smoke — {detail}")
    return 0
  print(f"FAIL: noise smoke — {detail}", file=sys.stderr)
  return 1


if __name__ == "__main__":
  sys.exit(main())
