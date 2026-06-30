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

# --- Matrix axes (Section 4d) ---
# NOTE: JOULE_W is applied directly as the Phase-C tau^2 reward weight by the
# factory (it negates positive values). Magnitudes are O(1e-4); do NOT use 1e4.
VARIANTS=(clock_anneal self_paced clock_persist)
JOULE_W_VALUES=(1e-4 3e-4)
PHASE_C_FRACS=(0.5 0.7)
SEEDS=(1)

# Fixed across the grid
GAIT_PERIOD="${GAIT_PERIOD:-0.7}"
EFFORT_LO="${EFFORT_LO:-0.7}"
EFFORT_HI="${EFFORT_HI:-1.2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nugus_gridsearch}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-mjlab}"

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

MANIFESTS=()

for variant in "${VARIANTS[@]}"; do
  for joule_w in "${JOULE_W_VALUES[@]}"; do
    for phase_c_frac in "${PHASE_C_FRACS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        joule_label="$(joule_tag "$joule_w")"
        run_name="${variant}__joule-${joule_label}__pc-${phase_c_frac}__s${seed}"
        job_name="mjlab-gs-$(k8s_slug "${variant}-joule-${joule_label}-pc-${phase_c_frac}-s${seed}")"
        wandb_tags="${variant},joule-${joule_label},pc-${phase_c_frac},seed-${seed},gridsearch"

        export JOB_NAME="$job_name"
        export MJLAB_VARIANT="$variant"
        export JOULE_W="$joule_w"
        export PHASE_C_FRAC="$phase_c_frac"
        export GAIT_PERIOD="$GAIT_PERIOD"
        export EFFORT_LO="$EFFORT_LO"
        export EFFORT_HI="$EFFORT_HI"
        export SEED="$seed"
        export LOGGER="$LOGGER"
        export EXPERIMENT_NAME="$EXPERIMENT_NAME"
        export RUN_NAME="$run_name"
        export WANDB_PROJECT="$WANDB_PROJECT"
        export WANDB_TAGS="$wandb_tags"

        if [[ -n "$OUTPUT_DIR" ]]; then
          mkdir -p "$OUTPUT_DIR"
          out="${OUTPUT_DIR}/${job_name}.yaml"
          envsubst <"$TEMPLATE" >"$out"
          MANIFESTS+=("$out")
        elif $DRY_RUN && ! $APPLY; then
          echo "---"
          envsubst <"$TEMPLATE"
        else
          tmp="$(mktemp "${TMPDIR:-/tmp}/mjlab-gs-XXXXXX.yaml")"
          envsubst <"$TEMPLATE" >"$tmp"
          MANIFESTS+=("$tmp")
        fi
      done
    done
  done
done

expected=$(( ${#VARIANTS[@]} * ${#JOULE_W_VALUES[@]} * ${#PHASE_C_FRACS[@]} * ${#SEEDS[@]} ))
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
