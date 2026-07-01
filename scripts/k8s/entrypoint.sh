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
  # A fresh clone lands on branch HEAD; if a commit is pinned, check it out so
  # ephemeral (emptyDir) workspaces are still reproducible at GIT_COMMIT.
  if [[ -n "${GIT_COMMIT:-}" ]]; then
    current="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$current" != "$GIT_COMMIT" && "$current" != "${GIT_COMMIT}"* ]]; then
      echo "[INFO] Checking out pinned commit ${GIT_COMMIT} after clone..."
      git -C "$REPO_DIR" fetch origin "${GIT_COMMIT}"
      git -C "$REPO_DIR" checkout "${GIT_COMMIT}"
    fi
  fi
elif [[ -n "${GIT_COMMIT:-}" ]]; then
  current="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current" == "$GIT_COMMIT" ]] || [[ "$current" == "${GIT_COMMIT}"* ]]; then
    echo "[INFO] Already at commit ${GIT_COMMIT} (${current}); skipping checkout."
  else
    echo "[INFO] Checking out commit ${GIT_COMMIT}..."
    lockfile="$REPO_DIR/.git/mjlab-checkout.lock"
    (
      flock -w 600 9 || exit 1
      git -C "$REPO_DIR" fetch origin "${GIT_COMMIT}"
      git -C "$REPO_DIR" checkout "${GIT_COMMIT}"
    ) 9>"$lockfile"
  fi
else
  echo "[INFO] Updating branch ${GIT_REF}..."
  git -C "$REPO_DIR" fetch origin "${GIT_REF}"
  git -C "$REPO_DIR" checkout "${GIT_REF}"
  git -C "$REPO_DIR" pull --ff-only || true
fi

cd "$REPO_DIR"

mkdir -p logs
ln -sfn /logs logs/rsl_rl

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

LOGGER="${LOGGER:-wandb}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nugus_gridsearch}"
RUN_NAME="${RUN_NAME:-${WANDB_RUN_NAME:-}}"
GPU_IDS="${GPU_IDS:-0}"

# Factory reads these from the environment (not CLI flags).
export MJLAB_VARIANT="${MJLAB_VARIANT:-}"
export JOULE_W="${JOULE_W:-}"
export STAND_W="${STAND_W:-}"
export PHASE_C_FRAC="${PHASE_C_FRAC:-}"
export GAIT_PERIOD="${GAIT_PERIOD:-0.7}"
export EFFORT_LO="${EFFORT_LO:-}"
export EFFORT_HI="${EFFORT_HI:-}"
export RESAMPLE_MIN="${RESAMPLE_MIN:-}"
export SEED="${SEED:-}"
# Phase boundaries decouple from training length; clock-silence and current-obs
# variant knobs. Only export when set so the factory falls back to its defaults.
[[ -n "${PHASE_ITERATIONS:-}" ]] && export PHASE_ITERATIONS
[[ -n "${SILENCE_CLOCK:-}" ]] && export SILENCE_CLOCK
[[ -n "${CURRENT_OBS:-}" ]] && export CURRENT_OBS

TRAIN_ARGS=(
  "${TASK}"
  --agent.logger "${LOGGER}"
  --agent.experiment-name "${EXPERIMENT_NAME}"
  --env.scene.num-envs "${NUM_ENVS}"
  --agent.max-iterations "${MAX_ITERATIONS}"
)

# tyro union parsing for --gpu-ids rejects a lone "0"; default [0] is correct for 1-GPU jobs.
if [[ -n "${GPU_IDS}" && "${GPU_IDS}" != "0" ]]; then
  IFS=',' read -ra _gpu_ids <<< "${GPU_IDS}"
  TRAIN_ARGS+=(--gpu-ids)
  for _gid in "${_gpu_ids[@]}"; do
    TRAIN_ARGS+=("${_gid}")
  done
fi

if [[ -n "${RUN_NAME}" ]]; then
  TRAIN_ARGS+=(--agent.run-name "${RUN_NAME}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  TRAIN_ARGS+=(--agent.wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_TAGS:-}" ]]; then
  # tyro Tuple[str, ...] accepts a single Python tuple literal, not space-separated tags.
  IFS=',' read -ra _wandb_tags <<< "${WANDB_TAGS}"
  _tags_literal="("
  for _tag in "${_wandb_tags[@]}"; do
    _tags_literal+="'${_tag}',"
  done
  _tags_literal="${_tags_literal%,})"
  TRAIN_ARGS+=(--agent.wandb-tags "${_tags_literal}")
fi
if [[ "${RESUME:-false}" == "true" ]]; then
  TRAIN_ARGS+=(--agent.resume True)
fi
if [[ -n "${WANDB_RUN_PATH:-}" ]]; then
  TRAIN_ARGS+=(--wandb-run-path "${WANDB_RUN_PATH}")
fi
if [[ -n "${WANDB_CHECKPOINT_NAME:-}" ]]; then
  TRAIN_ARGS+=(--wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME}")
fi

echo "[INFO] Starting training: task=${TASK}, experiment=${EXPERIMENT_NAME}, run_name=${RUN_NAME:-<unset>}, num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, gpu_ids=${GPU_IDS}, logger=${LOGGER}, variant=${MJLAB_VARIANT:-<unset>}"
exec "${VENV}/bin/train" "${TRAIN_ARGS[@]}"
