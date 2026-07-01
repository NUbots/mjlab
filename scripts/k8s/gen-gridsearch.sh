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
    tmp="$(mktemp "${TMPDIR:-/tmp}/mjlab-gs-XXXXXX.yaml")"
    envsubst <"$TEMPLATE" >"$tmp"
    MANIFESTS+=("$tmp")
  fi
}

# Export the knobs that are constant across a batch once. Per-cell values are
# exported inside each generator below just before emit_manifest.
export GAIT_PERIOD EFFORT_LO EFFORT_HI LOGGER EXPERIMENT_NAME WANDB_PROJECT
export MAX_ITERATIONS PHASE_ITERATIONS SILENCE_CLOCK CURRENT_OBS RESUME RESAMPLE_MIN
export WANDB_RUN_PATH WANDB_RUN_NAME

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

case "$BATCH" in
  v4) gen_v4_continuation; expected=2 ;;
  v5) gen_v5_grid; expected=4 ;;
  v6) gen_v6_rapidcmd; expected=2 ;;
  v7) gen_v7_grid; expected=2 ;;
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
  kubectl apply -f "$(dirname "${MANIFESTS[0]}")"
  echo "[INFO] Applied ${#MANIFESTS[@]} grid-search jobs to namespace mjlab" >&2
fi
