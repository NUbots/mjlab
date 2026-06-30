#!/usr/bin/env bash
# Clone or update mjlab on the workspace PVC, sync deps, and run training.
#
# Uses ghcr.io/mujocolab/mjlab:latest for CUDA, uv, Python, and system libs.
# Source and project deps come from the git checkout on the workspace PVC.
set -euo pipefail

REPO_DIR=/workspace/mjlab
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

if [[ -d .venv ]] && ! .venv/bin/python -c "import sympy" 2>/dev/null; then
  echo "[INFO] Removing incomplete .venv (missing CUDA extra deps)..."
  rm -rf .venv
fi

if [[ -d .venv ]]; then
  VENV_PY="$(
    .venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
      2>/dev/null || echo unknown
  )"
  WANT_PY="${PYTHON_VERSION%%.*}.$(echo "${PYTHON_VERSION}" | cut -d. -f2)"
  if [[ "$VENV_PY" != "$WANT_PY" ]]; then
    echo "[INFO] Removing .venv (Python ${VENV_PY}, want ${WANT_PY})..."
    rm -rf .venv
  fi
fi

echo "[INFO] Installing Python ${PYTHON_VERSION} and syncing dependencies..."
uv python install "${PYTHON_VERSION}"
uv sync --locked --no-dev --extra cu128 --python "${PYTHON_VERSION}"

echo "[INFO] Starting training: task=${TASK}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}"
exec uv run --no-dev --extra cu128 --python "${PYTHON_VERSION}" train "${TASK}" \
  --gpu-ids all \
  --agent.logger tensorboard \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${MAX_ITERATIONS}"
