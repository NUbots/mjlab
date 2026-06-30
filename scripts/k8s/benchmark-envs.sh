#!/usr/bin/env bash
# Sweep num_envs on a single GPU to find the throughput sweet spot.
set -euo pipefail

REPO_DIR=/workspace/mjlab
VENV=/app/.venv
TASK="${TASK:-Mjlab-Velocity-Flat-Nubots-Nugus}"
SWEEP="${NUM_ENVS_SWEEP:-4096 8192 12288 16384 20480}"
WARMUP="${BENCHMARK_WARMUP:-30}"
STEPS="${BENCHMARK_STEPS:-100}"

cd "$REPO_DIR"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_LINK_MODE=copy
export MUJOCO_GL=egl

if [[ -d .venv ]]; then rm -rf .venv; fi
uv sync --locked --no-dev --extra cu128 --python "${PYTHON_VERSION:-3.13}" -q

echo "[INFO] Benchmarking ${TASK} on ${SWEEP}"
for n in ${SWEEP}; do
  echo "===== num_envs=${n} ====="
  if ! CUDA_VISIBLE_DEVICES=0 "${VENV}/bin/python" - <<PY
import gc
import time

import torch
import mjlab.tasks  # noqa: F401 - register tasks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = "${TASK}"
num_envs = ${n}
warmup = ${WARMUP}
steps = ${STEPS}

try:
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()
  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)

  for _ in range(warmup):
    env.step(action)
  torch.cuda.synchronize()

  start = time.perf_counter()
  for _ in range(steps):
    env.step(action)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  sps = (steps * env.num_envs) / elapsed
  mem = torch.cuda.max_memory_allocated() / (1024**3)
  print(f"env_sps={sps:.0f} peak_vram_gb={mem:.2f} iter_time_est={elapsed/steps:.3f}s")
  env.close()
except torch.cuda.OutOfMemoryError:
  print("OOM")
except Exception as exc:
  print(f"FAIL: {exc}")
finally:
  gc.collect()
  torch.cuda.empty_cache()
PY
  then
    echo "benchmark failed for num_envs=${n}"
  fi
done
