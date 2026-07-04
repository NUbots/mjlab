#!/usr/bin/env bash
# Emit one Volcano Job manifest per grid-search matrix cell.
#
# Prerequisites:
#   - Volcano installed on the cluster (batch.volcano.sh + scheduling.volcano.sh CRDs)
#   - Base stack applied: kubectl apply -k scripts/k8s/
#   - mjlab-wandb secret created (see secret-wandb.yaml.example)
#   - GIT_REF / GIT_COMMIT in configmap.yaml point at the branch with factory env-var support
#
# Usage:
#   ./scripts/k8s/gen-gridsearch.sh              # dry-run: print 12 manifests to stdout
#   ./scripts/k8s/gen-gridsearch.sh --dry-run    # same as default
#   ./scripts/k8s/gen-gridsearch.sh -o DIR       # write manifests to DIR/
#   ./scripts/k8s/gen-gridsearch.sh --apply      # kubectl apply -f generated manifests
#
# Launch the full matrix (after reviewing dry-run output):
#   ./scripts/k8s/gen-gridsearch.sh -o /tmp/mjlab-gridsearch
#   kubectl apply -f /tmp/mjlab-gridsearch/
#
# Or in one step:
#   ./scripts/k8s/gen-gridsearch.sh --apply
#
# Monitor:
#   kubectl -n mjlab get vcjob -l mjlab/gridsearch=true
#   kubectl -n mjlab get queue mjlab-train
#
# TensorBoard (all runs under shared experiment_name): https://mjlab.4ai.systems
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/volcano-train-job.template.yaml"
OUTPUT_DIR=""
APPLY=false
DRY_RUN=true

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --apply)
      APPLY=true
      DRY_RUN=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      APPLY=false
      shift
      ;;
    -o | --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
  esac
done

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

# --- Matrix axes ---
# NOTE: JOULE_W is applied directly as the Phase-C tau^2 reward weight by the
# factory (it negates positive values). Magnitudes are O(1e-4); do NOT use 1e4.
VARIANTS=(clock_anneal self_paced clock_persist)
STAND_W_VALUES=(0.1 0.3)
PHASE_C_FRACS=(0.5 0.7)
SEEDS=(1)

# Resolved W&B run paths for the BATCH=v4 continuation (latest COMPLETED
# clock_anneal pc-0.5 runs). Override via env if the runs to continue change.
V4_WANDB_RUN_PATH_SW01="${V4_WANDB_RUN_PATH_SW01:-vincenttumm-the-university-of-newcastle/mjlab/7fivy5q7}"
V4_WANDB_RUN_PATH_SW03="${V4_WANDB_RUN_PATH_SW03:-vincenttumm-the-university-of-newcastle/mjlab/eyiowvgo}"

# Batch identifier. Distinguishes successive grid-search runs of the same
# matrix so a new batch does not collide with jobs from an earlier batch and is
# easy to filter in W&B/TensorBoard. It is appended to every Kubernetes job
# name (kept DNS-1123: lowercase alphanumeric + dashes), the run_name, and a
# W&B tag, and it scopes the shared experiment_name so each batch groups under
# its own TensorBoard tree. Override with BATCH=<id> (use a short, DNS-safe
# token such as v3 or a date like 20260630).
BATCH="${BATCH:-v3}"

# Fixed across the grid
JOULE_W="${JOULE_W:-3e-4}"
GAIT_PERIOD="${GAIT_PERIOD:-0.7}"
EFFORT_LO="${EFFORT_LO:-0.7}"
EFFORT_HI="${EFFORT_HI:-1.2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nugus_gridsearch_${BATCH}}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-mjlab}"

# Training-length / phase / variant knobs. These are ALWAYS substituted into the
# template (the env entries are static), so defaults must reproduce prior
# behaviour for the legacy matrix: MAX_ITERATIONS matches the configmap default
# and PHASE_ITERATIONS freezes phase boundaries at that same length.
MAX_ITERATIONS="${MAX_ITERATIONS:-1250}"
PHASE_ITERATIONS="${PHASE_ITERATIONS:-1250}"
SILENCE_CLOCK="${SILENCE_CLOCK:-0}"
CURRENT_OBS="${CURRENT_OBS:-0}"
RESUME="${RESUME:-false}"
RESAMPLE_MIN="${RESAMPLE_MIN:-3.0}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
PHASE_DELTA_STRONG_ITERS="${PHASE_DELTA_STRONG_ITERS:-}"
PHASE_DELTA_STRONG_W="${PHASE_DELTA_STRONG_W:-}"
PHASE_DELTA_TAIL_W="${PHASE_DELTA_TAIL_W:-}"
UPRIGHT_W="${UPRIGHT_W:-}"
PROGRESS_BACKSLIDE_W="${PROGRESS_BACKSLIDE_W:-}"
TRAINING_REGIME="${TRAINING_REGIME:-}"
HARD_COMPONENTS="${HARD_COMPONENTS:-}"
CONT_BASE_STEP="${CONT_BASE_STEP:-}"
CRITIC_HEIGHT_SCAN="${CRITIC_HEIGHT_SCAN:-}"
TASK="${TASK:-Mjlab-Velocity-Flat-Nubots-Nugus}"

# JOULE_W tag helper: keep scientific notation when already formatted.
joule_tag() {
  local v="$1"
  case "$v" in
    *e* | *E*) printf '%s' "$v" ;;
    *) printf '%s' "$v" ;;
  esac
}

# Kubernetes name: lowercase DNS label. Map underscores AND dots to dashes so
# generated pod names stay valid DNS-1123 labels (dots are not allowed there).
k8s_slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_.' '--' | tr -cd 'a-z0-9-' | sed 's/--*/-/g'
}

# Short slugs for JOB_NAME only (Volcano pod names = job + "-train-0" must be <= 63).
variant_job_slug() {
  case "$1" in
    clock_anneal) echo "ca" ;;
    clock_learned) echo "cl" ;;
    self_paced) echo "sp" ;;
    clock_persist) echo "cp" ;;
    *) k8s_slug "$1" ;;
  esac
}

stand_job_slug() {
  case "$1" in
    0.1) echo "sw01" ;;
    0.3) echo "sw03" ;;
    *) k8s_slug "sw-${1}" ;;
  esac
}

phase_c_job_slug() {
  case "$1" in
    0.5) echo "p05" ;;
    0.7) echo "p07" ;;
    *) k8s_slug "pc-${1}" ;;
  esac
}

joule_job_slug() {
  case "$1" in
    3e-4) echo "j3e4" ;;
    1e-4) echo "j1e4" ;;
    1e-5) echo "j1e5" ;;
    *) k8s_slug "j-${1}" ;;
  esac
}

MANIFESTS=()

# Render the template for one cell using the currently-exported env vars.
# Guards the pod-name length and routes output per --dry-run / -o DIR / --apply.
emit_manifest() {
  local job_name="$1"
  local pod_name="${job_name}-train-0"
  if ((${#pod_name} > 63)); then
    echo "Pod name too long (${#pod_name}): ${pod_name}" >&2
    exit 1
  fi
  export JOB_NAME="$job_name"
  if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    local out="${OUTPUT_DIR}/${job_name}.yaml"
    envsubst <"$TEMPLATE" >"$out"
    MANIFESTS+=("$out")
  elif $DRY_RUN && ! $APPLY; then
    echo "---"
    envsubst <"$TEMPLATE"
  else
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/mjlab-gs-XXXXXX")"
    envsubst <"$TEMPLATE" >"${tmp}.yaml"
    tmp="${tmp}.yaml"
    MANIFESTS+=("$tmp")
  fi
}

# Resolve GIT_COMMIT for manifest pins: explicit env > configmap.yaml > HEAD.
if [[ -z "${GIT_COMMIT:-}" ]]; then
  CONFIGMAP="${SCRIPT_DIR}/configmap.yaml"
  if [[ -f "$CONFIGMAP" ]]; then
    GIT_COMMIT="$(grep '^  GIT_COMMIT:' "$CONFIGMAP" | head -1 | sed -E 's/^  GIT_COMMIT: "?([^"]*)"?/\1/')"
  fi
fi
if [[ -z "${GIT_COMMIT:-}" ]]; then
  GIT_COMMIT="$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD)"
fi
export GIT_COMMIT

# Export the knobs that are constant across a batch once. Per-cell values are
# exported inside each generator below just before emit_manifest.
export GAIT_PERIOD EFFORT_LO EFFORT_HI LOGGER EXPERIMENT_NAME WANDB_PROJECT
export MAX_ITERATIONS PHASE_ITERATIONS SILENCE_CLOCK CURRENT_OBS RESUME RESAMPLE_MIN
export WANDB_RUN_PATH WANDB_RUN_NAME
export PHASE_DELTA_STRONG_ITERS PHASE_DELTA_STRONG_W PHASE_DELTA_TAIL_W UPRIGHT_W PROGRESS_BACKSLIDE_W
export TRAINING_REGIME CONT_BASE_STEP CRITIC_HEIGHT_SCAN TASK HARD_COMPONENTS
export SWING_TARGET_HEIGHT FLATTEN_PHASE_C PHASE_C_WARMUP ALIVE_W JOINT_ACC_W
export FOOT_FLAT_W FOOT_FLAT_ONESIDED CLEARANCE_PER_CORNER SWING_HEIGHT_SOURCE
export CLEARANCE_TARGET_HEIGHT CLEARANCE_TARGET_FROM_SWING PUSH_INTERVAL_SCALE
export MIRROR_AUG TRACK_LIN_W TRACK_ANG_W AIR_TIME_W LR_CAP LR_CAP_START_ITER
export ENTROPY_DECAY GAMMA
export FEET_MIN_SEP FEET_MIN_SEP_SHARPNESS FEET_MIN_SEP_W PHASE_DELTA_W
export JOB_REPLICAS MULTINODE MJLAB_LOG_STAMP
JOB_REPLICAS="${JOB_REPLICAS:-1}"
export NUM_ENVS
NUM_ENVS="${NUM_ENVS:-8192}"
MULTINODE="${MULTINODE:-}"
MJLAB_LOG_STAMP="${MJLAB_LOG_STAMP:-}"
export ADAPTIVE_COMMANDS ADAPTIVE_PUSHES PENALTY_GATE
export ADAPTIVE_CMD_LMAX ADAPTIVE_PUSH_LMAX
export COMPETENCE_PROMOTE_TRACK_ERR COMPETENCE_DEMOTE_TRACK_ERR
export COMPETENCE_PROMOTE_ATTAIN COMPETENCE_DEMOTE_ATTAIN
export COMPETENCE_PROMOTE_WOBBLE COMPETENCE_DEMOTE_WOBBLE
export COMPETENCE_PROMOTE_FELL COMPETENCE_DEMOTE_FELL COMPETENCE_COOLDOWN_ITERS
export LINK_MASS_SCALE_MIN LINK_MASS_SCALE_MAX
export PAYLOAD_KG_MIN PAYLOAD_KG_MAX

gen_default_matrix() {
  for variant in "${VARIANTS[@]}"; do
    for stand_w in "${STAND_W_VALUES[@]}"; do
      for phase_c_frac in "${PHASE_C_FRACS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          joule_label="$(joule_tag "$JOULE_W")"
          export RUN_NAME="${variant}__stand-${stand_w}__pc-${phase_c_frac}__joule-${joule_label}__s${seed}__${BATCH}"
          job_slug="$(k8s_slug "${BATCH}-$(variant_job_slug "$variant")-$(stand_job_slug "$stand_w")-$(phase_c_job_slug "$phase_c_frac")-$(joule_job_slug "$JOULE_W")-s${seed}")"
          export MJLAB_VARIANT="$variant"
          export STAND_W="$stand_w"
          export JOULE_W="$JOULE_W"
          export PHASE_C_FRAC="$phase_c_frac"
          export SEED="$seed"
          export WANDB_TAGS="${variant},stand-${stand_w},pc-${phase_c_frac},joule-${joule_label},seed-${seed},gridsearch,batch-${BATCH}"
          emit_manifest "mj-gs-${job_slug}"
        done
      done
    done
  done
}

# BATCH=v4: continuation of the two COMPLETED clock_anneal pc-0.5 runs
# (STAND_W 0.1 and 0.3). Resume from their latest checkpoint and run
# MAX_ITERATIONS more iterations (additive on resume) so they end at ~2000.
# PHASE_ITERATIONS freezes the curriculum timing at the original 1250-iter
# boundaries; by the resume point the counter is already past the wide command
# stage and the clock anneal, so commands are wide immediately.
gen_v4_continuation() {
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export SEED="1"
  export RESUME="true"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  local stand_w run_path
  for stand_w in 0.1 0.3; do
    if [[ "$stand_w" == "0.1" ]]; then
      run_path="$V4_WANDB_RUN_PATH_SW01"
    else
      run_path="$V4_WANDB_RUN_PATH_SW03"
    fi
    if [[ -z "$run_path" ]]; then
      echo "Missing W&B run path for STAND_W=${stand_w}; refusing to launch v4." >&2
      exit 1
    fi
    export STAND_W="$stand_w"
    export WANDB_RUN_PATH="$run_path"
    export RUN_NAME="clock_anneal__stand-${stand_w}__pc-0.5__joule-${joule_label}__cont2000__${BATCH}"
    export WANDB_TAGS="clock_anneal,stand-${stand_w},pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},continuation"
    emit_manifest "mj-gs-${BATCH}-ca-$(stand_job_slug "$stand_w")-p05-cont"
  done
}

# BATCH=v5: fresh 2x2 grid over CURRENT_OBS x SILENCE_CLOCK for clock_anneal
# pc-0.5, STAND_W 0.15, seed 1, trained from scratch to MAX_ITERATIONS with
# phase boundaries frozen at PHASE_ITERATIONS. The (current=0, silence=0) cell
# doubles as a from-scratch 2000-iter baseline.
gen_v5_grid() {
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  local cur sil
  for cur in 0 1; do
    for sil in 0 1; do
      export CURRENT_OBS="$cur"
      export SILENCE_CLOCK="$sil"
      export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__cur${cur}__sil${sil}__s1__${BATCH}"
      export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},current-${cur},silence-${sil}"
      emit_manifest "mj-gs-${BATCH}-ca-cur${cur}-sil${sil}"
    done
  done
}

# BATCH=v6: A/B on command resampling minimum (baseline 3.0s vs rapid 0.0s) for
# clock_anneal pc-0.5, STAND_W 0.15, seed 1, 2000 iters with phases frozen at
# 1250.
gen_v6_rapidcmd() {
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="1250"
  export WANDB_RUN_PATH=""
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  local rmin
  for rmin in 3.0 0.0; do
    export RESAMPLE_MIN="$rmin"
    export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__rmin${rmin}__s1__${BATCH}"
    export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},rmin-${rmin}"
    case "$rmin" in
      3.0) emit_manifest "mj-gs-${BATCH}-ca-rmin30" ;;
      0.0) emit_manifest "mj-gs-${BATCH}-ca-rmin00" ;;
    esac
  done
}

# BATCH=v7: clock_learned vs clock_anneal (no silence) — decoupled learned phase
# vs coupled clock anneal. pc-0.5, STAND_W 0.15, seed 1, 2000 iters.
gen_v7_grid() {
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  local joule_label variant
  joule_label="$(joule_tag "$JOULE_W")"
  for variant in clock_learned clock_anneal; do
    export MJLAB_VARIANT="$variant"
    export RUN_NAME="${variant}__stand-0.15__pc-0.5__s1__${BATCH}"
    export WANDB_TAGS="${variant},stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH}"
    case "$variant" in
      clock_learned) emit_manifest "mj-gs-${BATCH}-cl" ;;
      clock_anneal) emit_manifest "mj-gs-${BATCH}-ca" ;;
    esac
  done
}


# BATCH=v9: extend strong phase_delta_nominal stage to 1000 iters, halve upright
# weight, and compare clock_learned (CURRENT_OBS 0/1) vs clock_anneal baseline.
gen_v9_grid() {
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  local joule_label variant cur
  joule_label="$(joule_tag "$JOULE_W")"
  export MJLAB_VARIANT="clock_learned"
  for cur in 0 1; do
    export CURRENT_OBS="$cur"
    export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur${cur}__s1__${BATCH}"
    export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},current-${cur},upright-0.5,strong-1000"
    emit_manifest "mj-gs-${BATCH}-cl-cur${cur}"
  done
  export MJLAB_VARIANT="clock_anneal"
  export CURRENT_OBS="0"
  export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__s1__${BATCH}"
  export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},upright-0.5"
  emit_manifest "mj-gs-${BATCH}-ca"
}

# BATCH=v8: clock_learned CURRENT_OBS 0 vs 1 at pinned 2229f92 (strong early
# phase_delta_nominal penalty). pc-0.5, STAND_W 0.15, seed 1, 2000 iters.
gen_v8_grid() {
  export MJLAB_VARIANT="clock_learned"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  local cur
  for cur in 0 1; do
    export CURRENT_OBS="$cur"
    export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur${cur}__s1__${BATCH}"
    export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},current-${cur}"
    emit_manifest "mj-gs-${BATCH}-cl-cur${cur}"
  done
}

# BATCH=v10: two parallel jobs — (A) flat hard continuation from v9 cur0 with
# legacy critic, and (B) flat v9-equivalent retrain with critic height_scan.
# Override V10_WANDB_RUN_PATH with the v9 cur0 W&B run before --apply.
gen_v10_grid() {
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  local v9_path="${V10_WANDB_RUN_PATH:-}"
  if [[ -z "$v9_path" ]]; then
    echo "Missing V10_WANDB_RUN_PATH for stage A; refusing to launch v10." >&2
    exit 1
  fi
  export MJLAB_VARIANT="clock_learned"
  export TRAINING_REGIME="hard_continue"
  export CRITIC_HEIGHT_SCAN="false"
  export RESUME="true"
  export WANDB_RUN_PATH="$v9_path"
  export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hard-cont__v10"
  export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},hard-continue,continuation"
  emit_manifest "mj-gs-v10-cl-cur0-cont"
  export TRAINING_REGIME="base"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__v10"
  export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,flat-retrain"
  emit_manifest "mj-gs-v10-cl-cur0-hs"
}

# BATCH=v11: overnight base→hard single-run (no resume). Each seed trains 4000
# iters: v9-equivalent base for PHASE_ITERATIONS=2000, then hard_continue ramps
# from CONT_BASE_STEP=48000 (2000*24). Uses critic height_scan.
gen_v11_overnight() {
  export MJLAB_VARIANT="clock_learned"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  local joule_label seed
  joule_label="$(joule_tag "$JOULE_W")"
  for seed in 1 2 3 4; do
    export SEED="$seed"
    export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__s${seed}__${BATCH}"
    export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-${seed},gridsearch,batch-${BATCH},critic-height-scan,base-hard"
    emit_manifest "mj-gs-${BATCH}-cl-cur0-hs-s${seed}"
  done
}

# BATCH=v12: v11 overnight base→hard with non-zero phase_delta_nominal tail.
# Two runs (seed 1) compare moderate vs lighter cadence hold after strong start.
gen_v12_pd_tail() {
  export MJLAB_VARIANT="clock_learned"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export SEED="1"
  local joule_label tail_w tail_slug
  joule_label="$(joule_tag "$JOULE_W")"
  for tail_w in -0.2 -0.1; do
    export PHASE_DELTA_TAIL_W="$tail_w"
    tail_slug="${tail_w#-}"
    export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-${tail_slug}__s1__${BATCH}"
    export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,base-hard,pd-tail-${tail_slug}"
    emit_manifest "mj-gs-${BATCH}-cl-pd-tail-${tail_slug}"
  done
}


# BATCH=v13: lower joule (1e-5) on v12-like clock_learned base→hard (pd-tail -0.1)
# plus v9-equivalent clock_anneal flat baseline at default JOULE_W=3e-4.
gen_v13_grid() {
  local joule_label

  export MJLAB_VARIANT="clock_learned"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export PHASE_DELTA_TAIL_W="-0.1"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export SEED="1"
  export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hs__base-hard__pd-tail-0.1__joule-${joule_label}__s1__${BATCH}"
  export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,base-hard,pd-tail-0.1,joule-1e-5"
  emit_manifest "mj-gs-${BATCH}-cl-joule-1e5-pd01"

  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="3e-4"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="false"
  export PHASE_DELTA_TAIL_W=""
  export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__s1__${BATCH}"
  export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},upright-0.5"
  emit_manifest "mj-gs-${BATCH}-ca"
}


# BATCH=v14: clock_anneal base→hard single-run (no resume), v11-style 4000 iters
# with legacy critic (no height_scan). One job, seed 1.
gen_v14_clock_anneal_hard() {
  local joule_label
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="3e-4"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export PHASE_DELTA_TAIL_W=""
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="false"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export SEED="1"
  export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__base-hard__s1__${BATCH}"
  export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},base-hard,upright-0.5"
  emit_manifest "mj-gs-${BATCH}-ca-base-hard"
}


# BATCH=v16-short: Phase-0 validation — same as v16 but 500 iters.
gen_v16_short() {
  local joule_label
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="500"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""
  export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__joule-${joule_label}__hs__s1__v16-short"
  export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-v16-short,critic-height-scan,v16-short"
  emit_manifest "mj-gs-v16-short-ca-hs-joule-1e5"
}


# BATCH=v16b: re-baseline after v16 collapse — clock_persist, GAIT_PERIOD=1.0,
# SWING_TARGET_HEIGHT=0.05, flattened Phase-C, trimmed DR. Requires actuator fixes.
gen_v16b() {
  local joule_label
  export MJLAB_VARIANT="clock_persist"
  export GAIT_PERIOD="1.0"
  export SWING_TARGET_HEIGHT="0.05"
  export FLATTEN_PHASE_C="1"
  export LINK_MASS_SCALE_MIN="0.90"
  export LINK_MASS_SCALE_MAX="1.10"
  export PAYLOAD_KG_MIN="-0.2"
  export PAYLOAD_KG_MAX="0.2"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""
  local seed job_suffix
  for seed in 1 2; do
    export SEED="$seed"
    if [[ "$seed" == "1" ]]; then
      job_suffix=""
    else
      job_suffix="-s${seed}"
    fi
    export RUN_NAME="clock_persist__stand-0.15__pc-flat__joule-${joule_label}__hs__s${seed}__${BATCH}"
    export WANDB_TAGS="clock_persist,stand-0.15,pc-flat,joule-${joule_label},seed-${seed},gridsearch,batch-${BATCH},critic-height-scan,v16b"
    emit_manifest "mj-gs-${BATCH}-cp-hs-joule-1e5${job_suffix}"
  done
}


# BATCH=v16c: post-v16b fix — stall effort limits, alive bonus, Phase-C warmup.
gen_v16c() {
  local joule_label
  export MJLAB_VARIANT="clock_persist"
  export GAIT_PERIOD="1.0"
  export SWING_TARGET_HEIGHT="0.05"
  export PHASE_C_WARMUP="1"
  export FLATTEN_PHASE_C="0"
  export ALIVE_W="0.5"
  export JOINT_ACC_W="-1e-4"
  export LINK_MASS_SCALE_MIN="0.90"
  export LINK_MASS_SCALE_MAX="1.10"
  export PAYLOAD_KG_MIN="-0.2"
  export PAYLOAD_KG_MAX="0.2"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""
  local seed job_suffix run_tag
  for seed in 1 2; do
    export SEED="$seed"
    export JOINT_ACC_W="-1e-4"
    if [[ "$seed" == "1" ]]; then
      job_suffix=""
    else
      job_suffix="-s${seed}"
    fi
    export RUN_NAME="clock_persist__stand-0.15__pc-warmup__alive-0.5__joule-${joule_label}__hs__s${seed}__${BATCH}"
    export WANDB_TAGS="clock_persist,stand-0.15,pc-warmup,alive-0.5,joule-${joule_label},seed-${seed},gridsearch,batch-${BATCH},critic-height-scan,v16c"
    emit_manifest "mj-gs-${BATCH}-cp-hs-joule-1e5${job_suffix}"
  done
  export SEED="1"
  export JOINT_ACC_W="-3e-5"
  export RUN_NAME="clock_persist__stand-0.15__pc-warmup__alive-0.5__jacc-3e5__joule-${joule_label}__hs__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,stand-0.15,pc-warmup,alive-0.5,jacc-3e5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,v16c"
  emit_manifest "mj-gs-${BATCH}-cp-hs-joule-1e5-jacc-3e5"
}


# BATCH=v16d: post-v16c reward-economy fix (see plan doc 09). v16c walked but
# shuffled: joint_acc warm-up killed velocity tracking (trkLin peaked at the
# ramp then declined), feet dragged (air_time ~0.04-0.09 s), gait limped
# (gait_air_cv ~0.45, no symmetry mechanism). Cells: main (jacc -1e-5, trk 3.0,
# swing 0.065, air 0.15, mirror aug, LR cap), no-mirror ablation, jacc -3e-5
# bridge to the best v16c cell.
gen_v16d() {
  local joule_label
  export MJLAB_VARIANT="clock_persist"
  export GAIT_PERIOD="1.0"
  export SWING_TARGET_HEIGHT="0.065"
  export AIR_TIME_W="0.15"
  export TRACK_LIN_W="3.0"
  export TRACK_ANG_W="2.0"
  export PHASE_C_WARMUP="1"
  export FLATTEN_PHASE_C="0"
  export ALIVE_W="0.5"
  export JOINT_ACC_W="-1e-5"
  export MIRROR_AUG="1"
  export LR_CAP="3e-4"
  export LR_CAP_START_ITER="1200"
  export ENTROPY_DECAY=""
  export GAMMA=""
  export LINK_MASS_SCALE_MIN="0.90"
  export LINK_MASS_SCALE_MAX="1.10"
  export PAYLOAD_KG_MIN="-0.2"
  export PAYLOAD_KG_MAX="0.2"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""

  export RUN_NAME="clock_persist__trk-3.0__jacc-1e-5__swing-0.065__air-0.15__mirror__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,trk-3.0,jacc-1e-5,swing-0.065,air-0.15,mirror,alive-0.5,pc-warmup,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan"
  emit_manifest "mj-gs-${BATCH}-main"

  export MIRROR_AUG=""
  export RUN_NAME="clock_persist__trk-3.0__jacc-1e-5__swing-0.065__air-0.15__nomirror__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,trk-3.0,jacc-1e-5,swing-0.065,air-0.15,nomirror,alive-0.5,pc-warmup,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan"
  emit_manifest "mj-gs-${BATCH}-nomirror"

  export MIRROR_AUG="1"
  export JOINT_ACC_W="-3e-5"
  export RUN_NAME="clock_persist__trk-3.0__jacc-3e-5__swing-0.065__air-0.15__mirror__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,trk-3.0,jacc-3e-5,swing-0.065,air-0.15,mirror,alive-0.5,pc-warmup,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan"
  emit_manifest "mj-gs-${BATCH}-jacc-3e5"
}


# Shared v16d-style BASE1 defaults for overnight wave 1+ (override per cell).
_wave1_base_exports() {
  export MJLAB_VARIANT="clock_persist"
  export GAIT_PERIOD="1.0"
  export SWING_TARGET_HEIGHT="0.065"
  export AIR_TIME_W="0.15"
  export TRACK_LIN_W="3.0"
  export TRACK_ANG_W="2.0"
  export PHASE_C_WARMUP="1"
  export FLATTEN_PHASE_C="0"
  export ALIVE_W="0.5"
  export JOINT_ACC_W="-1e-5"
  export MIRROR_AUG="1"
  export LR_CAP="3e-4"
  export LR_CAP_START_ITER="1200"
  export ENTROPY_DECAY="1"
  export GAMMA=""
  export LINK_MASS_SCALE_MIN="0.90"
  export LINK_MASS_SCALE_MAX="1.10"
  export PAYLOAD_KG_MIN="-0.2"
  export PAYLOAD_KG_MAX="0.2"
  export JOULE_W="1e-5"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""
  export FOOT_FLAT_W="-0.5"
  export FOOT_FLAT_ONESIDED=""
  export CLEARANCE_PER_CORNER=""
  export SWING_HEIGHT_SOURCE="min_corner"
  export CLEARANCE_TARGET_HEIGHT=""
  export CLEARANCE_TARGET_FROM_SWING=""
  export PUSH_INTERVAL_SCALE="1.0"
}


# Wave-1 integrator placeholder for BASE2 (R9 defaults until D1 retunes).
_wave2_base_exports() {
  _wave1_base_exports
  export FOOT_FLAT_ONESIDED="1"
  export CLEARANCE_PER_CORNER="1"
  export SWING_HEIGHT_SOURCE="center"
  export GAIT_PERIOD="0.85"
  export SWING_TARGET_HEIGHT="0.065"
  export JOINT_ACC_W="-1e-5"
  export MAX_ITERATIONS="2000"
  export GAMMA=""
  export CLEARANCE_TARGET_FROM_SWING=""
  export CLEARANCE_TARGET_HEIGHT=""
  export PUSH_INTERVAL_SCALE="1.0"
}


# BASE3 placeholder: BASE2 + economy winner (ALIVE_W=0.25 until D2 retunes).
_wave3_base_exports() {
  _wave2_base_exports
  export ALIVE_W="0.25"
  export LINK_MASS_SCALE_MIN="0.90"
  export LINK_MASS_SCALE_MAX="1.10"
  export PAYLOAD_KG_MIN="-0.2"
  export PAYLOAD_KG_MAX="0.2"
}


# FINAL placeholder for wave 4: BASE3 + wide DR (R14 winner until D3 retunes).
_wave4_final_exports() {
  _wave3_base_exports
  export LINK_MASS_SCALE_MIN="0.85"
  export LINK_MASS_SCALE_MAX="1.15"
  export PAYLOAD_KG_MIN="-0.3"
  export PAYLOAD_KG_MAX="0.5"
}


# BATCH=wave1: heel-toe + cadence cells on v16d BASE1 (R4–R9).
gen_wave1() {
  local joule_label
  _wave1_base_exports
  joule_label="$(joule_tag "$JOULE_W")"

  export FOOT_FLAT_ONESIDED="1"
  export MAX_ITERATIONS="1000"
  export RUN_NAME="clock_persist__flat-onesided__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,foot-flat-onesided,trk-3.0,jacc-1e-5,seed-1,gridsearch,batch-${BATCH},wave1,r4"
  emit_manifest "mj-gs-${BATCH}-r4-flat-onesided"

  export FOOT_FLAT_ONESIDED="1"
  export CLEARANCE_PER_CORNER="1"
  export SWING_HEIGHT_SOURCE="center"
  export RUN_NAME="clock_persist__heel-toe__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,heel-toe,clearance-per-corner,swing-center,seed-1,gridsearch,batch-${BATCH},wave1,r5"
  emit_manifest "mj-gs-${BATCH}-r5-heel-toe"

  _wave1_base_exports
  export MAX_ITERATIONS="1000"
  export GAIT_PERIOD="0.85"
  export SWING_TARGET_HEIGHT="0.065"
  export RUN_NAME="clock_persist__gait-0.85__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,gait-0.85,seed-1,gridsearch,batch-${BATCH},wave1,r6"
  emit_manifest "mj-gs-${BATCH}-r6-gait-085"

  export MAX_ITERATIONS="1000"
  export GAIT_PERIOD="0.7"
  export SWING_TARGET_HEIGHT="0.05"
  export RUN_NAME="clock_persist__gait-0.7__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,gait-0.7,seed-1,gridsearch,batch-${BATCH},wave1,r7"
  emit_manifest "mj-gs-${BATCH}-r7-gait-07"

  export MAX_ITERATIONS="1400"
  export GAIT_PERIOD="1.0"
  export SWING_TARGET_HEIGHT="0.065"
  export JOINT_ACC_W="0"
  export RUN_NAME="clock_persist__jacc-0__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,jacc-0,seed-1,gridsearch,batch-${BATCH},wave1,r8"
  emit_manifest "mj-gs-${BATCH}-r8-jacc-0"

  export MAX_ITERATIONS="1400"
  export JOINT_ACC_W="-1e-5"
  export FOOT_FLAT_ONESIDED="1"
  export CLEARANCE_PER_CORNER="1"
  export SWING_HEIGHT_SOURCE="center"
  export GAIT_PERIOD="0.85"
  export SWING_TARGET_HEIGHT="0.065"
  export RUN_NAME="clock_persist__integrator__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,integrator,heel-toe,gait-0.85,seed-1,gridsearch,batch-${BATCH},wave1,r9"
  emit_manifest "mj-gs-${BATCH}-r9-integrator"
}


# BATCH=wave2: economy + variance on BASE2 (R10–R13).
gen_wave2() {
  _wave2_base_exports

  export SEED="1"
  export ALIVE_W="0.5"
  export RUN_NAME="clock_persist__base2-ref__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,base2,integrator,seed-1,gridsearch,batch-${BATCH},wave2,r10"
  emit_manifest "mj-gs-${BATCH}-r10-base2-ref"

  export SEED="2"
  export RUN_NAME="clock_persist__base2-ref__s2__${BATCH}"
  export WANDB_TAGS="clock_persist,base2,integrator,seed-2,gridsearch,batch-${BATCH},wave2,r11"
  emit_manifest "mj-gs-${BATCH}-r11-base2-s2"

  export SEED="1"
  export MAX_ITERATIONS="1400"
  export ALIVE_W="0.25"
  export RUN_NAME="clock_persist__alive-0.25__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,alive-0.25,base2,seed-1,gridsearch,batch-${BATCH},wave2,r12"
  emit_manifest "mj-gs-${BATCH}-r12-alive-025"

  export ALIVE_W="0.5"
  export AIR_TIME_W="0.25"
  export CLEARANCE_TARGET_FROM_SWING="1"
  export RUN_NAME="clock_persist__air-0.25__clearance-swing__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,air-0.25,clearance-swing,seed-1,gridsearch,batch-${BATCH},wave2,r13-airtime"
  emit_manifest "mj-gs-${BATCH}-r13-airtime"

  export AIR_TIME_W="0.15"
  export CLEARANCE_TARGET_FROM_SWING=""
  export GAMMA="0.97"
  export RUN_NAME="clock_persist__gamma-0.97__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,gamma-0.97,base2,seed-1,gridsearch,batch-${BATCH},wave2,r13-gamma"
  emit_manifest "mj-gs-${BATCH}-r13-gamma"
}


# BATCH=wave3: robustness + reach on BASE3 (R14–R17).
gen_wave3() {
  _wave3_base_exports

  export SEED="1"
  export MAX_ITERATIONS="2000"
  export LINK_MASS_SCALE_MIN="0.85"
  export LINK_MASS_SCALE_MAX="1.15"
  export PAYLOAD_KG_MIN="-0.3"
  export PAYLOAD_KG_MAX="0.5"
  export RUN_NAME="clock_persist__wide-dr__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,wide-dr,base3,seed-1,gridsearch,batch-${BATCH},wave3,r14"
  emit_manifest "mj-gs-${BATCH}-r14-wide-dr"

  _wave3_base_exports
  export SEED="1"
  export MAX_ITERATIONS="2000"
  export PUSH_INTERVAL_SCALE="1.5"
  export RUN_NAME="clock_persist__push-interval-1.5__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,push-interval-1.5,base3,seed-1,gridsearch,batch-${BATCH},wave3,r15"
  emit_manifest "mj-gs-${BATCH}-r15-push-interval"

  _wave3_base_exports
  export SEED="1"
  export MAX_ITERATIONS="2000"
  export MJLAB_VARIANT="self_paced"
  export RUN_NAME="self_paced__base3-economy__s1__${BATCH}"
  export WANDB_TAGS="self_paced,base3,alive-0.25,seed-1,gridsearch,batch-${BATCH},wave3,r16"
  emit_manifest "mj-gs-${BATCH}-r16-self-paced"

  _wave3_base_exports
  export SEED="1"
  export MAX_ITERATIONS="1400"
  export GAMMA="0.97"
  export RUN_NAME="clock_persist__gamma-0.97__backfill__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,gamma-0.97,backfill,seed-1,gridsearch,batch-${BATCH},wave3,r17-gamma"
  emit_manifest "mj-gs-${BATCH}-r17-gamma"

  export AIR_TIME_W="0.25"
  export CLEARANCE_TARGET_FROM_SWING="1"
  export GAMMA=""
  export RUN_NAME="clock_persist__air-0.25__backfill__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,air-0.25,clearance-swing,backfill,seed-1,gridsearch,batch-${BATCH},wave3,r17-airtime"
  emit_manifest "mj-gs-${BATCH}-r17-airtime"

  export AIR_TIME_W="0.15"
  export CLEARANCE_TARGET_FROM_SWING=""
  export ALIVE_W="0.25"
  export RUN_NAME="clock_persist__alive-0.25__backfill__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,alive-0.25,backfill,seed-1,gridsearch,batch-${BATCH},wave3,r17-alive"
  emit_manifest "mj-gs-${BATCH}-r17-alive"
}


# BATCH=wave4: consolidation (R18–R20).
gen_wave4() {
  _wave4_final_exports

  export SEED="1"
  export MAX_ITERATIONS="4000"
  export MJLAB_VARIANT="clock_persist"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export LR_CAP="3e-4"
  export LR_CAP_START_ITER="1200"
  export RUN_NAME="clock_persist__final__4k__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,final,4k,lr-cap,seed-1,gridsearch,batch-${BATCH},wave4,r18"
  emit_manifest "mj-gs-${BATCH}-r18-final-4k"

  export MAX_ITERATIONS="2000"
  export SEED="3"
  export RUN_NAME="clock_persist__final__s3__${BATCH}"
  export WANDB_TAGS="clock_persist,final,seed-3,gridsearch,batch-${BATCH},wave4,r19"
  emit_manifest "mj-gs-${BATCH}-r19-final-s3"

  export SEED="1"
  export TASK="Mjlab-Velocity-Rough-Nubots-Nugus"
  export RUN_NAME="clock_persist__rough-taste__s1__${BATCH}"
  export WANDB_TAGS="clock_persist,rough,final,seed-1,gridsearch,batch-${BATCH},wave4,r20"
  emit_manifest "mj-gs-${BATCH}-r20-rough"
}


# Shared R13 stack for v16e (doc 12): BASE2 (onesided + gait 0.7) + air 0.25 +
# clearance-from-swing + entropy decay.
_v16e_r13_exports() {
  _wave1_base_exports
  export FOOT_FLAT_ONESIDED="1"
  # Feet min-separation wall (user-calibrated 2026-07-04): 5 cm edge gap is
  # fine, 1-3 cm is the hazard. 0.13 m center-to-center ~= 5 cm edge gap for
  # ~8 cm feet; sharpness 20 puts a steep wall through the 3 cm -> 1 cm band.
  export FEET_MIN_SEP="0.13"
  export FEET_MIN_SEP_SHARPNESS="20"
  export FEET_MIN_SEP_W="-1.0"
  export CLEARANCE_PER_CORNER=""
  export SWING_HEIGHT_SOURCE="min_corner"
  export GAIT_PERIOD="0.7"
  export SWING_TARGET_HEIGHT="0.05"
  export AIR_TIME_W="0.25"
  export CLEARANCE_TARGET_FROM_SWING="1"
  export ENTROPY_DECAY="1"
  export ALIVE_W="0.5"
  export MAX_ITERATIONS="2000"
  export EXPERIMENT_NAME="nugus_gridsearch_v16e"
}


# BATCH=v16e: entropy-decay fix on the R13 stack (doc 12).
gen_v16e() {
  _v16e_r13_exports

  export SEED="1"
  export RUN_NAME="clock_persist__r13-stack__entropy-decay__s1__2k__v16e"
  export WANDB_TAGS="clock_persist,r13-stack,entropy-decay,gait-0.7,air-0.25,clearance-swing,seed-1,2k,gridsearch,batch-v16e"
  emit_manifest "mj-gs-v16e-r13-s1-2k"

  export SEED="2"
  export RUN_NAME="clock_persist__r13-stack__entropy-decay__s2__2k__v16e"
  export WANDB_TAGS="clock_persist,r13-stack,entropy-decay,gait-0.7,air-0.25,clearance-swing,seed-2,2k,gridsearch,batch-v16e"
  emit_manifest "mj-gs-v16e-r13-s2-2k"

  export SEED="1"
  export MAX_ITERATIONS="4000"
  export RUN_NAME="clock_persist__r13-stack__entropy-decay__s1__4k__v16e"
  export WANDB_TAGS="clock_persist,r13-stack,entropy-decay,gait-0.7,air-0.25,clearance-swing,seed-1,4k,gridsearch,batch-v16e"
  emit_manifest "mj-gs-v16e-r13-s1-4k"
}


# Shared competence-curriculum defaults (doc 13).
_competence_defaults() {
  export ADAPTIVE_COMMANDS=""
  export ADAPTIVE_PUSHES=""
  export PENALTY_GATE="time"
  # Climb to the brink: demote keeps the top honest (user request
  # 2026-07-04); levels >5 need gait-period-as-command (doc 07) first.
  export ADAPTIVE_CMD_LMAX="5"
  # Push level capped below cmd headroom: v20 attempt 1 showed pushes
  # (fell-gated only) racing to L3 while commands sat at L0.
  export ADAPTIVE_PUSH_LMAX="2"
  # Normalized track error has an intrinsic sway floor ~0.6 at level-0
  # commands (v20 attempt 1: err_norm floored at 0.64 with ep_len 966 and
  # fell 0.58 -> the 0.25 code default could NEVER promote). 0.7 makes
  # low-level promotion effectively stability-gated; the err bar binds at
  # higher levels where commands dominate sway.
  export COMPETENCE_PROMOTE_TRACK_ERR=""
  export COMPETENCE_DEMOTE_TRACK_ERR=""
  export COMPETENCE_PROMOTE_ATTAIN=""
  export COMPETENCE_DEMOTE_ATTAIN=""
  export COMPETENCE_PROMOTE_WOBBLE=""
  export COMPETENCE_DEMOTE_WOBBLE=""
  export COMPETENCE_PROMOTE_FELL=""
  # fell_ema is a RATE (<=1.0): the code default demote bar of 1.0 is
  # unreachable, making push/penalty demotion dead code (v20 attempt 3:
  # falls climbed at the brink with no demote). 0.5 = half of episodes
  # ending in falls is decisively past competence.
  export COMPETENCE_DEMOTE_FELL=""
  # Cooldown must exceed the EMA refresh time (~10 episodes ~ 200 iters at
  # alpha 0.1) or promotions cascade on stale competence (v20 attempt 3:
  # cmd chained L1->L5 in 240 iters).
  export COMPETENCE_COOLDOWN_ITERS=""
}


# BATCH=v20: competence-curriculum A/B (doc 13) on the v16e R13 stack.
gen_v20() {
  _v16e_r13_exports
  _competence_defaults
  export MAX_ITERATIONS="2000"
  export EXPERIMENT_NAME="nugus_gridsearch_v20"

  local cell slug seed job_suffix
  for cell in control const cmd full hard owned; do
    _v16e_r13_exports
    _competence_defaults
    export MAX_ITERATIONS="2000"
    export EXPERIMENT_NAME="nugus_gridsearch_v20"
    case "$cell" in
      control)
        slug="control"
        ;;
      const)
        # Stationary-objective control (v16f-const, doc 14): penalties at
        # final values from iter 0, no warm-up, no gating. With the alive
        # bonus this should bootstrap; tests whether removing objective
        # nonstationarity alone stops the disease-#2 height ratchet.
        slug="const"
        export PHASE_C_WARMUP="0"
        export FLATTEN_PHASE_C="1"
        ;;
      cmd)
        slug="cmd"
        export ADAPTIVE_COMMANDS="1"
        ;;
      full)
        slug="full"
        export ADAPTIVE_COMMANDS="1"
        export ADAPTIVE_PUSHES="1"
        export PENALTY_GATE="competence"
        ;;
      hard)
        slug="hard"
        export ADAPTIVE_COMMANDS="1"
        export ADAPTIVE_PUSHES="1"
        export PENALTY_GATE="competence"
        export ADAPTIVE_CMD_LMAX="5"
        export ADAPTIVE_PUSH_LMAX="5"
        ;;
      owned)
        # v20-full + policy-owned phase with constant nominal tether
        # (clock_owned, user proposal 2026-07-04): fixed clock as soft
        # attractor, deviation as escape hatch — no realignment debt after
        # perturbations. Watch Metrics/phase_delta_dev_p95 for tail usage.
        slug="owned"
        export MJLAB_VARIANT="clock_owned"
        export PHASE_DELTA_W="-0.2"
        export ADAPTIVE_COMMANDS="1"
        export ADAPTIVE_PUSHES="1"
        export PENALTY_GATE="competence"
        ;;
    esac
    for seed in 1 2; do
      export SEED="$seed"
      if [[ "$seed" == "1" ]]; then
        job_suffix=""
      else
        job_suffix="-s${seed}"
      fi
      export RUN_NAME="clock_persist__r13__v20-${slug}__entropy-decay__s${seed}__${BATCH}"
      export WANDB_TAGS="clock_persist,r13-stack,v20-${slug},entropy-decay,seed-${seed},2k,gridsearch,batch-v20"
      emit_manifest "mj-gs-v20-${slug}${job_suffix}"
    done
  done
}


# BATCH=mn-bench: env-count sweet-spot sweep UNDER the multi-node setup.
# The 8192/GPU sweet spot was measured on 1-GPU synthetic jobs; with 8 GPUs
# there are 20 sync barriers/iter and gang iteration time is the max over
# ranks, so the latency-optimal env count may shift. 4 cells x 300 iters,
# each gang-takes all 8 GPUs -> Volcano runs them sequentially. Read
# Perf/collection_time + learning_time + total_fps per cell from W&B.
gen_mn_bench() {
  local envs
  for envs in 2048 4096 6144 8192; do
    _v16e_r13_exports
    _competence_defaults
    export ADAPTIVE_COMMANDS="1"
    export ADAPTIVE_PUSHES="1"
    export PENALTY_GATE="competence"
    export MAX_ITERATIONS="300"
    export PHASE_ITERATIONS="2000"
    export SEED="1"
    export JOB_REPLICAS="2"
    export MULTINODE="1"
    export NUM_ENVS="${envs}"
    export MJLAB_LOG_STAMP="mn-bench-${envs}-$(date +%Y%m%d-%H%M%S)"
    export EXPERIMENT_NAME="nugus_mn_bench"
    export RUN_NAME="mn-bench__envs-${envs}__8gpu__${BATCH}"
    export WANDB_TAGS="mn-bench,8gpu,envs-${envs},batch-mn-bench"
    emit_manifest "mj-gs-mn-bench-${envs}"
  done
}


# BATCH=v21: batch-size (env-count) learning A/B, single-node 4-GPU pair.
# Env count x 24 steps IS the PPO batch (786k samples/update at 4x8192 for a
# 0.5M-param policy — plausibly far past critical batch). Two simultaneous
# runs, 16k vs 32k total envs, identical corrected-controller config:
# whichever crosses command-level milestones first in WALL CLOCK wins the
# latency mode. Combined with mn-bench sec/iter this pins the wall-time
# optimum for any GPU count. Also doubles as the first validation of the
# fixed attainment gating.
gen_v21() {
  local envs
  for envs in 4096 8192; do
    _v16e_r13_exports
    _competence_defaults
    export ADAPTIVE_COMMANDS="1"
    export ADAPTIVE_PUSHES="1"
    export PENALTY_GATE="competence"
    export MAX_ITERATIONS="2000"
    export PHASE_ITERATIONS="2000"
    export SEED="1"
    export NUM_ENVS="${envs}"
    export EXPERIMENT_NAME="nugus_gridsearch_v21"
    export RUN_NAME="clock_persist__v21-bs${envs}__attain-fixed__s1__${BATCH}"
    export WANDB_TAGS="clock_persist,v21-bs${envs},attain-fixed,batch-v21,gridsearch"
    emit_manifest "mj-gs-v21-bs${envs}"
  done
}


# BATCH=mn-smoke: 8-GPU multi-node smoke (backlog 15c -> active). v20-full
# config at 300 iters across 2x4 GPUs via torchrun env-rendezvous. PASS =
# one W&B run (not two -> rank gating bug), iterating, collection_time
# comparable to single-node, one checkpoint dir. NUM_ENVS stays 8192/GPU
# (65k total) for the smoke; the 4096-point benchmark decides the fast mode.
gen_mn_smoke() {
  _v16e_r13_exports
  _competence_defaults
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export MAX_ITERATIONS="300"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export MJLAB_LOG_STAMP="mn-smoke-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_mn_smoke"
  export RUN_NAME="clock_persist__mn-smoke__8gpu__${BATCH}"
  export WANDB_TAGS="mn-smoke,8gpu,multinode,batch-mn-smoke"
  emit_manifest "mj-gs-mn-smoke"
}


# BATCH=v16: Phase-0 smoke — clock_anneal flat base (no hard_continue), 2k iters,
# critic height_scan, JOULE_W=1e-5. Establishes the post-E0.2/A3/C1 baseline.
gen_v16_base() {
  local joule_label
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export RESUME="false"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export WANDB_RUN_PATH=""
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="base"
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export PHASE_DELTA_TAIL_W=""
  export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__joule-${joule_label}__hs__s1__${BATCH}"
  export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,v16-base"
  emit_manifest "mj-gs-${BATCH}-ca-hs-joule-1e5"
}


# BATCH=v15: v14 clock_anneal base→hard extended to 20k iters. Hard ramp in the
# first ~3k iters (hard_continue from CONT_BASE_STEP=48000), then holds final
# hard parameters through 20k. PHASE_ITERATIONS=2000 freezes phase boundaries.
gen_v15_clock_anneal_hard_long() {
  local joule_label
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="3e-4"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="20000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export PHASE_DELTA_TAIL_W=""
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="false"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  local seed job_suffix
  for seed in 1 2; do
    export SEED="$seed"
    if [[ "$seed" == "1" ]]; then
      job_suffix=""
    else
      job_suffix="-s${seed}"
    fi
    export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__base-hard__20k__s${seed}__${BATCH}"
    export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-${seed},gridsearch,batch-${BATCH},base-hard,20k,upright-0.5"
    emit_manifest "mj-gs-${BATCH}-ca-base-hard-20k${job_suffix}"
  done
}


# BATCH=v17: hard-stage decoupling on v16-style base (clock_anneal, hs-critic,
# joule 1e-5). Each cell enables exactly one hard component at CONT_BASE_STEP.
gen_v17_hard_decouple() {
  local joule_label component slug
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export PHASE_DELTA_TAIL_W=""
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CONT_BASE_STEP="48000"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export SEED="1"
  for component in commands pushes upright phasec all; do
    case "$component" in
      commands) export HARD_COMPONENTS="commands" ;;
      pushes) export HARD_COMPONENTS="pushes" ;;
      upright) export HARD_COMPONENTS="upright" ;;
      phasec) export HARD_COMPONENTS="phasec" ;;
      all) export HARD_COMPONENTS="commands,pushes,upright,phasec" ;;
    esac
    slug="$component"
    export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__joule-${joule_label}__hard-${slug}__s1__${BATCH}"
    export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,base-hard,hard-${slug}"
    emit_manifest "mj-gs-${BATCH}-${slug}"
  done
}


# BATCH=v18: single-stage hard-from-start (no hard_continue / Phase-C ramps).
# Two cells compare final upright weight 0.25 vs 0.5.
gen_v18_hard_from_start() {
  local joule_label upright_w upright_slug
  export MJLAB_VARIANT="clock_anneal"
  export JOULE_W="1e-5"
  joule_label="$(joule_tag "$JOULE_W")"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export PHASE_DELTA_TAIL_W=""
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Flat-Nubots-Nugus"
  export TRAINING_REGIME="hard_from_start"
  export HARD_COMPONENTS=""
  export CONT_BASE_STEP=""
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="false"
  export WANDB_RUN_PATH=""
  export SEED="1"
  for upright_w in 0.25 0.5; do
    export UPRIGHT_W="$upright_w"
    upright_slug="${upright_w//./}"
    export RUN_NAME="clock_anneal__stand-0.15__pc-0.5__joule-${joule_label}__hard-start__upright-${upright_w}__s1__${BATCH}"
    export WANDB_TAGS="clock_anneal,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,hard-from-start,upright-${upright_w}"
    emit_manifest "mj-gs-${BATCH}-upright-${upright_slug}"
  done
}


# BATCH=v10b: rough hard continuation from stage B (not queued in v10 launch).
# Run manually after stage B completes:
#   V10B_WANDB_RUN_PATH=<wandb/path/from/stage-B> BATCH=v10b ./scripts/k8s/gen-gridsearch.sh --apply
gen_v10b_rough_cont() {
  export MJLAB_VARIANT="clock_learned"
  export JOULE_W="$JOULE_W"
  export PHASE_C_FRAC="0.5"
  export STAND_W="0.15"
  export SEED="1"
  export SILENCE_CLOCK="0"
  export CURRENT_OBS="0"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export PHASE_DELTA_STRONG_ITERS="1000"
  export PHASE_DELTA_STRONG_W="-5.0"
  export UPRIGHT_W="0.5"
  export PROGRESS_BACKSLIDE_W="-0.5"
  export TASK="Mjlab-Velocity-Rough-Nubots-Nugus"
  export TRAINING_REGIME="hard_continue"
  export CRITIC_HEIGHT_SCAN="true"
  export RESUME="true"
  local b_path="${V10B_WANDB_RUN_PATH:-}"
  if [[ -z "$b_path" ]]; then
    echo "Missing V10B_WANDB_RUN_PATH for stage C; refusing to launch v10b." >&2
    exit 1
  fi
  export WANDB_RUN_PATH="$b_path"
  local joule_label
  joule_label="$(joule_tag "$JOULE_W")"
  export RUN_NAME="clock_learned__stand-0.15__pc-0.5__cur0__hs-critic__hard-cont-rough__v10b"
  export WANDB_TAGS="clock_learned,stand-0.15,pc-0.5,joule-${joule_label},seed-1,gridsearch,batch-${BATCH},critic-height-scan,hard-continue,rough-continuation"
  emit_manifest "mj-gs-v10b-cl-cur0-rough-cont"
}

case "$BATCH" in
  v4) gen_v4_continuation; expected=2 ;;
  v5) gen_v5_grid; expected=4 ;;
  v6) gen_v6_rapidcmd; expected=2 ;;
  v7) gen_v7_grid; expected=2 ;;
  v8) gen_v8_grid; expected=2 ;;
  v9) gen_v9_grid; expected=3 ;;
  v10) gen_v10_grid; expected=2 ;;
  v10b) gen_v10b_rough_cont; expected=1 ;;
  v11) gen_v11_overnight; expected=4 ;;
  v12) gen_v12_pd_tail; expected=2 ;;
  v13) gen_v13_grid; expected=2 ;;
  v14) gen_v14_clock_anneal_hard; expected=1 ;;
  v15) gen_v15_clock_anneal_hard_long; expected=2 ;;
  v16-short) gen_v16_short; expected=1 ;;
  v16) gen_v16_base; expected=1 ;;
  v16b) gen_v16b; expected=2 ;;
  v16c) gen_v16c; expected=3 ;;
  v16d) gen_v16d; expected=3 ;;
  wave1) gen_wave1; expected=6 ;;
  wave2) gen_wave2; expected=5 ;;
  wave3) gen_wave3; expected=6 ;;
  wave4) gen_wave4; expected=3 ;;
  v16e) gen_v16e; expected=3 ;;
  v20) gen_v20; expected=12 ;;
  mn-smoke) gen_mn_smoke; expected=1 ;;
  mn-bench) gen_mn_bench; expected=4 ;;
  v21) gen_v21; expected=2 ;;
  v17) gen_v17_hard_decouple; expected=5 ;;
  v18) gen_v18_hard_from_start; expected=2 ;;
  *)
    gen_default_matrix
    expected=$((${#VARIANTS[@]} * ${#STAND_W_VALUES[@]} * ${#PHASE_C_FRACS[@]} * ${#SEEDS[@]}))
    ;;
esac

if [[ -n "$OUTPUT_DIR" ]] || $APPLY; then
  if [[ ${#MANIFESTS[@]} -ne $expected ]]; then
    echo "Expected ${expected} manifests, got ${#MANIFESTS[@]}" >&2
    exit 1
  fi
  echo "[INFO] Generated ${#MANIFESTS[@]} Volcano Job manifests" >&2
fi

if $APPLY; then
  if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
    echo "Nothing to apply (use -o DIR or omit --apply for dry-run)" >&2
    exit 1
  fi
  for manifest in "${MANIFESTS[@]}"; do
    kubectl apply -f "$manifest"
  done
  echo "[INFO] Applied ${#MANIFESTS[@]} grid-search jobs to namespace mjlab" >&2
fi
