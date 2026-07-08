#!/usr/bin/env bash
# Clone or update mjlab on the workspace PVC, sync deps, and run training.
#
# Uses ghcr.io/mujocolab/mjlab:latest for CUDA, uv, Python, and system libs.
# Source comes from the git checkout on the workspace PVC; deps sync into the
# image venv at /app/.venv (workspace .venv causes SIGBUS on CUDA init).
#
# When GIT_COMMIT is set, use a full 40-character SHA. Shallow clones only
# contain branch tips; pinned commits are fetched by object id (--depth=1).
set -euo pipefail

REPO_DIR=/workspace/mjlab
VENV=/app/.venv
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

at_pinned_commit() {
  local current
  current="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  [[ "$current" == "${GIT_COMMIT}" ]] || [[ "$current" == "${GIT_COMMIT}"* ]]
}

fetch_pinned_commit() {
  if git -C "$REPO_DIR" cat-file -e "${GIT_COMMIT}^{commit}" 2>/dev/null; then
    return 0
  fi
  echo "[INFO] Fetching pinned commit ${GIT_COMMIT}..."
  if git -C "$REPO_DIR" fetch --depth=1 origin "${GIT_COMMIT}"; then
    return 0
  fi
  echo "[WARN] Fetch by commit SHA failed; deepening ${GIT_REF}..."
  git -C "$REPO_DIR" fetch origin "${GIT_REF}" --deepen=500 \
    || git -C "$REPO_DIR" fetch --unshallow origin
  git -C "$REPO_DIR" fetch --depth=1 origin "${GIT_COMMIT}"
}

checkout_pinned_commit() {
  if at_pinned_commit; then
    echo "[INFO] Already at commit ${GIT_COMMIT} ($(git -C "$REPO_DIR" rev-parse HEAD)); skipping checkout."
    return 0
  fi
  if [[ -n "${GIT_COMMIT:-}" && ${#GIT_COMMIT} -lt 40 ]]; then
    echo "[WARN] GIT_COMMIT should be a full 40-character SHA (got ${#GIT_COMMIT} chars)."
  fi
  local lockfile="$REPO_DIR/.git/mjlab-checkout.lock"
  (
    flock -w 600 9 || exit 1
    fetch_pinned_commit
    git -C "$REPO_DIR" checkout "${GIT_COMMIT}"
  ) 9>"$lockfile"
}

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
    if at_pinned_commit; then
      echo "[INFO] Clone already at pinned commit ${GIT_COMMIT}."
    else
      echo "[INFO] Checking out pinned commit ${GIT_COMMIT} after clone..."
      fetch_pinned_commit
      git -C "$REPO_DIR" checkout "${GIT_COMMIT}"
    fi
  fi
elif [[ -n "${GIT_COMMIT:-}" ]]; then
  checkout_pinned_commit
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

# Factory reads these from the environment (not CLI flags). Only export when set
# so empty K8s env entries do not override factory defaults.
[[ -n "${MJLAB_VARIANT:-}" ]] && export MJLAB_VARIANT
[[ -n "${JOULE_W:-}" ]] && export JOULE_W
[[ -n "${STAND_W:-}" ]] && export STAND_W
[[ -n "${PHASE_C_FRAC:-}" ]] && export PHASE_C_FRAC
[[ -n "${GAIT_PERIOD:-}" ]] && export GAIT_PERIOD
[[ -n "${EFFORT_LO:-}" ]] && export EFFORT_LO
[[ -n "${EFFORT_HI:-}" ]] && export EFFORT_HI
[[ -n "${RESAMPLE_MIN:-}" ]] && export RESAMPLE_MIN
[[ -n "${SEED:-}" ]] && export SEED
[[ -n "${PHASE_ITERATIONS:-}" ]] && export PHASE_ITERATIONS
[[ -n "${SILENCE_CLOCK:-}" ]] && export SILENCE_CLOCK
[[ -n "${CURRENT_OBS:-}" ]] && export CURRENT_OBS
[[ -n "${PHASE_DELTA_STRONG_ITERS:-}" ]] && export PHASE_DELTA_STRONG_ITERS
[[ -n "${PHASE_DELTA_STRONG_W:-}" ]] && export PHASE_DELTA_STRONG_W
[[ -n "${PHASE_DELTA_TAIL_W:-}" ]] && export PHASE_DELTA_TAIL_W
[[ -n "${UPRIGHT_W:-}" ]] && export UPRIGHT_W
[[ -n "${PROGRESS_BACKSLIDE_W:-}" ]] && export PROGRESS_BACKSLIDE_W
[[ -n "${TRAINING_REGIME:-}" ]] && export TRAINING_REGIME
[[ -n "${HARD_COMPONENTS:-}" ]] && export HARD_COMPONENTS
[[ -n "${CONT_BASE_STEP:-}" ]] && export CONT_BASE_STEP
[[ -n "${CRITIC_HEIGHT_SCAN:-}" ]] && export CRITIC_HEIGHT_SCAN
[[ -n "${TASK:-}" ]] && export TASK

TRAIN_ARGS=(
  "${TASK}"
  --agent.logger "${LOGGER}"
  --agent.experiment-name "${EXPERIMENT_NAME}"
  --env.scene.num-envs "${NUM_ENVS}"
  --agent.max-iterations "${MAX_ITERATIONS}"
)

# Periodic training videos (rank 0 only, offscreen EGL - verified working
# on the cluster pods): VIDEO=1 enables capture. Interval/length are in
# policy steps (24/iter, so 6000 = every 250 iterations; 1000 steps = one
# full 20 s episode at 50 Hz). The rsl-rl W&B logger finds the mp4s in the
# log dir and uploads each as wandb.Video automatically.
if [[ "${VIDEO:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --video
    --video-length "${VIDEO_LENGTH:-1000}"
    --video-interval "${VIDEO_INTERVAL:-6000}"
  )
fi

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
if [[ "${MULTINODE:-}" == "1" ]]; then
  # Multi-node data-parallel via torchrun env-rendezvous (no SSH). The
  # Volcano svc plugin provides VC_<TASK>_HOSTS (comma-separated pod
  # FQDNs); the env plugin provides VC_TASK_INDEX. Gradients are ~2 MB so
  # plain Ethernet is fine (~80 ms/iter at 10G across 20 all-reduces).
  MASTER_ADDR="${MASTER_ADDR:-${VC_TRAIN_HOSTS%%,*}}"
  NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-0}}"
  NNODES="${NNODES:-2}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  echo "[INFO] Multinode: nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR} nproc=${NPROC_PER_NODE}"
  exec "${VENV}/bin/python" -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT:-29500}" \
    --no-python "${VENV}/bin/train" "${TRAIN_ARGS[@]}"
fi
exec "${VENV}/bin/train" "${TRAIN_ARGS[@]}"
