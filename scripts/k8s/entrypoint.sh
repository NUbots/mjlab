#!/usr/bin/env bash
# Clone or update mjlab on the workspace PVC, sync deps, and run training.
#
# Uses ghcr.io/mujocolab/mjlab:latest for CUDA, uv, Python, and system libs.
# Source comes from the git checkout on the workspace PVC; deps sync into the
# image venv at /app/.venv (workspace .venv causes SIGBUS on CUDA init).
set -euo pipefail

REPO_DIR=/workspace/mjlab
VENV=/app/.venv
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

if [[ -d "$REPO_DIR/.git" ]] && ! git -C "$REPO_DIR" status >/dev/null 2>&1; then
  echo "[WARN] Corrupt git checkout; removing ${REPO_DIR}..."
  rm -rf "$REPO_DIR"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[INFO] Cloning ${GIT_REPO} (ref: ${GIT_REF})..."
  git clone --depth 1 --branch "${GIT_REF}" "${GIT_REPO}" "$REPO_DIR"
elif [[ -n "${GIT_COMMIT:-}" ]]; then
  echo "[INFO] Checking out commit ${GIT_COMMIT}..."
  git -C "$REPO_DIR" fetch origin "${GIT_COMMIT}"
  git -C "$REPO_DIR" checkout "${GIT_COMMIT}"
else
  echo "[INFO] Updating branch ${GIT_REF}..."
  git -C "$REPO_DIR" fetch origin "${GIT_REF}"
  git -C "$REPO_DIR" checkout "${GIT_REF}"
  git -C "$REPO_DIR" pull --ff-only || true
fi

cd "$REPO_DIR"

mkdir -p logs
ln -sfn /logs logs/rsl_rl

# Stale workspace venvs break CUDA (SIGBUS); always use the image venv.
if [[ -d .venv ]]; then
  echo "[INFO] Removing workspace .venv (use ${VENV} instead)..."
  rm -rf .venv
fi

echo "[INFO] Syncing dependencies into ${VENV}..."
export UV_COMPILE_BYTECODE=0
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_LINK_MODE=copy
uv python install "${PYTHON_VERSION}"
uv sync --locked --no-dev --extra cu128 --python "${PYTHON_VERSION}"

echo "[INFO] Preflight: CUDA and mjlab imports..."
"${VENV}/bin/python" - <<'PY'
import torch

print(
  f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
  f"devices={torch.cuda.device_count()}"
)
import mjlab.tasks  # noqa: F401

print("mjlab.tasks import ok")
PY

GPU_IDS="${GPU_IDS:-all}"
echo "[INFO] Starting training: task=${TASK}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, gpu_ids=${GPU_IDS}"
exec "${VENV}/bin/train" "${TASK}" \
  --gpu-ids "${GPU_IDS}" \
  --agent.logger tensorboard \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${MAX_ITERATIONS}"
