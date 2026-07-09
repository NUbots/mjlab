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
BUS_VOLTAGE="${BUS_VOLTAGE:-0}"
# Training videos: on by default for all future batches (rank-0 EGL capture,
# ~40 clips per 10k run, auto-uploaded to W&B). Steps, not iterations:
# 6000 = every 250 iters at 24 steps/iter; 1000 = one 20 s episode.
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-6000}"
# Alternate captures push-cohort (env 0) / clean-cohort (env -1 = last).
VIDEO_ENV_IDS="${VIDEO_ENV_IDS:-0,-1}"
# 720p (viewer default 320x240 is unwatchable); rendered off the hot path.
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
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
export VIDEO VIDEO_LENGTH VIDEO_INTERVAL VIDEO_ENV_IDS VIDEO_HEIGHT VIDEO_WIDTH
export WANDB_RUN_PATH WANDB_RUN_NAME
export PHASE_DELTA_STRONG_ITERS PHASE_DELTA_STRONG_W PHASE_DELTA_TAIL_W UPRIGHT_W PROGRESS_BACKSLIDE_W
export TRAINING_REGIME CONT_BASE_STEP CRITIC_HEIGHT_SCAN TASK HARD_COMPONENTS
export SWING_TARGET_HEIGHT FLATTEN_PHASE_C PHASE_C_WARMUP ALIVE_W JOINT_ACC_W
export FOOT_FLAT_W FOOT_FLAT_ONESIDED CLEARANCE_PER_CORNER SWING_HEIGHT_SOURCE
export CLEARANCE_TARGET_HEIGHT CLEARANCE_TARGET_FROM_SWING PUSH_INTERVAL_SCALE
export MIRROR_AUG TRACK_LIN_W TRACK_ANG_W AIR_TIME_W LR_CAP LR_CAP_START_ITER
export ENTROPY_DECAY GAMMA
export ENTROPY_START ENTROPY_END
ENTROPY_START="${ENTROPY_START:-}"
ENTROPY_END="${ENTROPY_END:-}"
export STD_MIN
STD_MIN="${STD_MIN:-}"
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
export COMPETENCE_DEMOTE_FAST_FELL COMPETENCE_TOP_STREAK
export CURRICULUM_STYLE AIMD_ALPHA AIMD_BETA AIMD_CONGEST_BAR
export AIMD_EMERGENCY_BAR AIMD_GATE_ATTAIN
export TRACK_WATCHDOG WATCHDOG_ARM_ABOVE WATCHDOG_FAIL_BELOW WATCHDOG_PERSIST_ITERS
export PUSH_COHORT_FRAC FRONTIER_HAZARD_BAR
export JOULE_LAMBDA_SHADOW LAMBDA_CAP LAMBDA_RAMP_ITERS JOULE_LAMBDA_LIVE
export COMMAND_GEOMETRY
export AIMD_BETA_ARREST AIMD_ENVELOPE_SCALE
export AIMD_PUSH_CONGEST_BAR AIMD_PUSH_GATE_EXCESS AIMD_ATTAIN_SLIDE_FRAC LANDING_ANNEAL
export OBS_NORM_FREEZE_ITERS FREEZE_POLICY_AFTER
export TORQUE_RATE_PEAK_W SOFT_LANDING_PEAK_W
export AIMD_ATTAIN_BAND_HI AIMD_ATTAIN_BAND_LO AIMD_EXTEND_BAR AIMD_FLOOR_FRAC AIMD_FRONTIER_HEADROOM PUSH_SURVIVAL_BAR PUSH_OBS_WINDOW_S
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
  export CURRENT_OBS BUS_VOLTAGE="0"
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
  # 0.60/0.40, not the code-default 0.75/0.5: the best-ever policy (v16e
  # gate-passer, err 0.147 at 0.5 m/s) scores attain ~0.71 — a 0.75 promote
  # bar sits above the feasible ceiling (bug species #4; bs8192 plateaued
  # at 0.62 for 700 iters against it).
  export COMPETENCE_PROMOTE_ATTAIN="0.60"
  export COMPETENCE_DEMOTE_ATTAIN="0.40"
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
  # Fast windowed fall-rate demote (R7) and top-rung promote streak; empty
  # uses code defaults (0.5 / 5).
  export COMPETENCE_DEMOTE_FAST_FELL=""
  export COMPETENCE_TOP_STREAK=""
  # AIMD continuous difficulty (doc 15 R8) + rot watchdog; empty = code
  # defaults (alpha 0.002, beta 0.7, bars 0.35/0.55, gate 0.40; watchdog
  # arm 2.0 / fail 1.0 / persist 60).
  export CURRICULUM_STYLE=""
  export AIMD_ALPHA=""
  export AIMD_BETA=""
  export AIMD_CONGEST_BAR=""
  export AIMD_EMERGENCY_BAR=""
  export AIMD_GATE_ATTAIN=""
  export TRACK_WATCHDOG=""
  export WATCHDOG_ARM_ABOVE=""
  export WATCHDOG_FAIL_BELOW=""
  export WATCHDOG_PERSIST_ITERS=""
  # Push-cohort stratification + frontier diagnostics (doc 15 R9); empty =
  # legacy all-pushed, no diagnostics term knobs overridden.
  export PUSH_COHORT_FRAC=""
  export FRONTIER_HAZARD_BAR=""
  # Shadow Lagrangian energy multiplier (doc 15 R10); empty = on with code
  # defaults (log-only either way).
  export JOULE_LAMBDA_SHADOW=""
  export LAMBDA_CAP=""
  export LAMBDA_RAMP_ITERS=""
  export JOULE_LAMBDA_LIVE=""
  # Command geometry (doc 15 R11); empty = box (legacy).
  export COMMAND_GEOMETRY=""
  # Arrest-mode decay and envelope extension (v27 postmortem); empty =
  # code defaults (0.93 / 1.0).
  export AIMD_BETA_ARREST=""
  export AIMD_ENVELOPE_SCALE=""
  # Split push-axis congestion (v28 postmortem); empty = code defaults
  # (excess bar 0.30, rise gate 0.15).
  export AIMD_PUSH_CONGEST_BAR=""
  export AIMD_PUSH_GATE_EXCESS=""
  export AIMD_ATTAIN_SLIDE_FRAC=""
  export LANDING_ANNEAL=""
  # Obs-normalizer freeze (R15); empty = code default 500 iters.
  export OBS_NORM_FREEZE_ITERS=""
  # Bit-freeze probe (R16); empty = off.
  export FREEZE_POLICY_AFTER=""
  # Penalty peak weights (R17); empty = legacy constants (-1e-3 / -1e-2).
  export TORQUE_RATE_PEAK_W=""
  export SOFT_LANDING_PEAK_W=""
  # Band controller + attained floor (R19); empty = code defaults
  # (0.66 / 0.60 / 0.95).
  export AIMD_ATTAIN_BAND_HI=""
  export AIMD_ATTAIN_BAND_LO=""
  export AIMD_EXTEND_BAR=""
  export AIMD_FLOOR_FRAC=""
  export AIMD_FRONTIER_HEADROOM=""
  export PUSH_SURVIVAL_BAR=""
  export PUSH_OBS_WINDOW_S=""
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
  # 512..8192/GPU: the low points locate the sim-overhead knee (below it,
  # sec/iter flatlines and envs are free — which also bounds how many GPUs
  # a single run can productively use at a given total batch). 512 x 8 =
  # 4096 total = the legged_gym-lineage default.
  local envs
  for envs in 512 1024 2048 4096 6144 8192; do
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
  # First split is a 4x spread: 8k total (~field standard; legged_gym-lineage
  # default is 4096 TOTAL and the original NUgus training used 2048) vs the
  # 32k incumbent. Binary-search the knee from whichever side wins.
  local envs
  for envs in 2048 8192; do
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


# BATCH=v22: entropy-floor test on the promoted clock_owned base.
# Every degrading run on the corrected physics crossed into trouble as std
# sank below ~0.15 (full 0.138->0.106 through catastrophe, owned 0.171->0.13
# through its milder failure, v16e 0.103, the dead 4k run 0.08) while
# penalties/commands were flat - the 0.001 entropy floor over-sharpens the
# policy into brittleness. Hold ENTROPY_END=0.004 (std ~0.15-0.18).
# clock_owned (pair-3 verdict: ~10x more resilient than fixed clock, p95
# phase usage peaked exactly under stress), 2048 envs (bs-race fast point),
# attainment-fixed gating.
gen_v22() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export NUM_ENVS="2048"
  export ENTROPY_END="0.004"
  export MAX_ITERATIONS="2000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v22"
  export RUN_NAME="clock_owned__v22-floor-0.004__bs2048__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v22-floor,entropy-end-0.004,bs2048,batch-v22,gridsearch"
  emit_manifest "mj-gs-v22-floor"

  # v22b: hard sigma floor (STD_MIN=0.13), entropy schedule UNCHANGED
  # (END=0.001). Two-arm discrimination vs v22: if v22b stays healthy and
  # v22 does not, sigma LEVEL is causal (mechanical floor is the exact
  # fix); if v22 alone works, the entropy economics were the lever.
  export ENTROPY_END=""
  export STD_MIN="0.13"
  export RUN_NAME="clock_owned__v22b-stdmin-0.13__bs2048__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v22b-stdmin,std-min-0.13,bs2048,batch-v22,gridsearch"
  emit_manifest "mj-gs-v22b-stdmin"
}


# BATCH=v23: first production 8-GPU run — the full validated stack: clock_owned
# constant tether (pair-3 R1), STD_MIN sigma floor (v22b: held 0.143, climbed
# L0->L5 without catastrophe), corrected attainment bars 0.60/0.40 via
# _competence_defaults (feasibility-bug instance 4 fix), feet_min_sep,
# adaptive commands+pushes, competence penalty gate. 6144 envs/GPU x 8 GPUs:
# the mn-bench knee (536k fps == 8192's within 0.5%, at 25% cheaper
# iterations); sample-limited verdict (v21 race) puts fps first. 4000 iters
# (~2.5 h at 2.2 s/iter) answers whether v22b's late L5 sag was mid-learning
# (consolidates) or something structural (keeps sagging).
gen_v23() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export MJLAB_LOG_STAMP="v23-prod-$(date +%Y%m%d-%H%M%S)"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="4000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v23"
  export RUN_NAME="clock_owned__v23-prod__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v23-prod,std-min-0.13,8gpu,multinode,batch-v23,gridsearch"
  emit_manifest "mj-gs-v23-prod"
}


# BATCH=v24: v23 relaunch with the crash-reflex fixes. v23 climbed L0->L5
# cleanly then spiraled at the brink (ep 880->23, fell -> saturation, attain
# -0.68 with NO recovery in 1200 iters at L0): the EMA-lagged demote reacted
# ~200 iters late and the -10 fall gradients shattered the policy first.
# v24 adds the fast windowed fall-rate demote (bar 0.5, ~5-iter reaction,
# cascades L5->L0 in 5 iters) and top-rung promote caution (streak 5 at
# L>=4). Same 8-GPU shape and stack otherwise. Verdict metric: either the
# run holds a healthy L4/L5 bounce to 4000, or the fast demote catches a
# crash in-flight and the policy RECOVERS (both are wins for the reflex;
# a v23-style shatter is the failure case).
gen_v24() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export MJLAB_LOG_STAMP="v24-prod-$(date +%Y%m%d-%H%M%S)"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="4000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v24"
  export RUN_NAME="clock_owned__v24-fastdemote__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v24-fastdemote,std-min-0.13,8gpu,multinode,batch-v24,gridsearch"
  emit_manifest "mj-gs-v24-prod"
}


# BATCH=v24b: v24 outcome — the reflex fired exactly at its 0.5 bar (iter
# ~2020, rate 0.51) and cascaded L5->L0 in ~40 iters, but the spiral outran
# it: falls kept exploding at L0 (5.2 -> 13.6 by 2137, ep 253). The point of
# no return sits BELOW 0.5: during the L5 burn (rate 0.3-0.45, under the
# bar) the -10 fall gradients were already poisoning the batch, and once
# poisoned, easing the task does not stop the bleeding. Two counters:
# (1) bar 0.35 — just above the healthy-L5 band (0.26), catching the burn
# at its first acceleration; (2) cap the ladder at L4 (±0.6) — three runs
# (v22b, v23, v24) all ignited the burn at L5 (±0.75), consistent with the
# XH540 velocity-margin analysis; sample L5 again only after a full-length
# healthy L4 run exists.
# BATCH=v24c: v24b died WITHOUT ever promoting — bar-infeasibility instance
# 5. Its L0 attainment ceiling converged at 0.543 (run-to-run gait variance;
# v22b/v23/v24 crossed 0.60 at iters 1050-1300) and it sat saturated under
# the unreachable bar until R6 churn killed it at ~2100: the cleanest pure
# demonstration of saturation death (ep 1000 / wobble 0.011 / fell_ema 0.09
# at iter 1200 — an excellent L0 policy — then attain slip, fast rate
# 0.02->0.67, entropy VALUE rising, spiral). Unified law: too-hard and
# too-easy both end in the same absorbing fall-spiral; the ladder must keep
# moving. Changes vs v24b: PROMOTE_ATTAIN 0.50 (below the worst observed
# ceiling), DEMOTE_ATTAIN 0.35 (hysteresis width kept), PHASE_ITERATIONS
# 2000 (entropy decay completes on the v22b-proven profile instead of
# hovering mid-decay through the churn window; sigma floor rules after).
gen_v24c() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export ADAPTIVE_CMD_LMAX="4"
  export COMPETENCE_PROMOTE_ATTAIN="0.50"
  export COMPETENCE_DEMOTE_ATTAIN="0.35"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v24c-prod-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v24"
  export RUN_NAME="clock_owned__v24c-attain050-ph2k__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v24c,attain-0.50,phase-2k,lmax4,std-min-0.13,8gpu,multinode,batch-v24,gridsearch"
  emit_manifest "mj-gs-v24c-prod"
}


# BATCH=v45: the open-road run. v44 (honest physics) finished 4000 at
# attainment 0.833 / falls 0.022 with its frontier pinned against the
# envelope cap at 1.18 m/s - the cap was the last artificial limit.
# v45 = auto-extending envelope (R30): the frontier chases the robot's
# true peak; 10000 iterations now that nothing rots. Joule stays weak
# (0.1% of income - the lambda-live watts-budget experiment is a later
# single-variable arm).
gen_v45() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="10000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v45-openroad-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v45"
  export RUN_NAME="clock_owned__v45-open-road__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v45,auto-envelope,open-road,batch-v45,gridsearch"
  emit_manifest "mj-gs-v45-openroad"
}


# BATCH=v45b: v45 with un-saturated instruments (R31). v45 proved the
# auto-envelope works (3 extensions, 1.6->1.852, commands to 1.389) but
# its frontier estimator topped out at the 32x0.04 histogram's last bin
# (1.260 exactly, flat for 1400 iters) so the extension gate could
# never fire again. Bins now read to 3.2 m/s; this run finds the true
# peak. Otherwise identical to v45.
gen_v45b() {
  gen_v45
  export MAX_ITERATIONS="10000"
  export MJLAB_LOG_STAMP="v45b-truepeak-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v45b-true-peak__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v45b,wide-bins,true-peak,batch-v45b,gridsearch"
  emit_manifest "mj-gs-v45b-truepeak"
}


# BATCH=v45c: v45b + strict extension gate (R32). Same wide bins;
# envelope extension now requires the 0.80-bar frontier at the wall
# ("stably hitting", not "surviving"). The definitive true-peak run.
gen_v45c() {
  gen_v45b
  export AIMD_EXTEND_BAR="0.80"
  export MJLAB_LOG_STAMP="v45c-strictpeak-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v45c-strict-peak__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v45c,strict-extend,true-peak,batch-v45c,gridsearch"
  emit_manifest "mj-gs-v45c-strictpeak"
}


# BATCH=v46: held-speed semantics (R33) + raised claim bar. Attainment
# credit now requires settling (0.75 s) and minimum dwell (3 s), so the
# frontier reads speeds the robot MAINTAINS; with transients excluded
# the climb bar rises 0.60 -> 0.70 (transient dilution previously
# depressed measured ratios, so 0.70 against clean evidence is safer
# than 0.60 against dirty). Extension bar stays 0.80. The definitive
# honest-peak run.
gen_v46() {
  gen_v45c
  export AIMD_ATTAIN_BAND_LO="0.70"
  export MJLAB_LOG_STAMP="v46-heldspeed-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v46-held-speed__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v46,held-speed,censored-attain,batch-v46,gridsearch"
  emit_manifest "mj-gs-v46-heldspeed"
}


# BATCH=v47: measured physics + proprioceptive current. Two coupled
# changes from the servo sysid (doc 17): ARMATURE_XH540 corrected 1.9x
# (0.0266 -> 0.0496 measured; old value was outside the DR band, so
# this is a physics-baseline fix, not a hyperparameter) and CURRENT_OBS
# enabled (per-servo current = tau/Kt with measured Kt 2.68, Dynamixel
# quantization, noise+bias, gain/offset DR) - the deployable torque
# signal the real robot reports. Otherwise identical to v46. Not
# directly comparable to v46 (heavier legs); compare against its own
# frontier equilibrium.
gen_v47() {
  gen_v46
  export CURRENT_OBS="1"
  export MJLAB_LOG_STAMP="v47-current-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v47-current-obs__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v47,current-obs,measured-armature,batch-v47,gridsearch"
  emit_manifest "mj-gs-v47-current"
}


# BATCH=v48: shared-bus power model (doc 17 power network). Per-servo
# supply voltage sags with fleet current (battery Voc/discharge/69 mOhm
# source resistance, all measured) plus daisy-chain position; torque
# authority scales with each servo's live voltage and the policy SEES
# per-servo voltage (what hardware presentVoltage reports). This is the
# total-power budget made physical: 22 A fleet peaks cost every servo
# ~10% authority at once. Otherwise identical to v47.
gen_v48() {
  gen_v47
  export BUS_VOLTAGE="1"
  export MJLAB_LOG_STAMP="v48-busvolt-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v48-bus-voltage__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v48,bus-voltage,shared-power,batch-v48,gridsearch"
  emit_manifest "mj-gs-v48-busvolt"
}


# BATCH=v49: the v48 champion stack at a fresh seed, on the instrumented
# pin: W&B training videos (async recorder, alternating push/clean cohort
# robots every 250 iters), per-episode push_fall_dt attribution + windowed
# EMA, and exposure-clamped frontier readouts. No physics or observation
# changes vs v48; with seed 2 it is a seed-replicate of the champion
# result rather than a repeat of the same experiment.
gen_v49() {
  gen_v48
  export SEED="2"
  export MJLAB_LOG_STAMP="v49-video-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v49-video-instrumented__8gpu-6144__s2__${BATCH}"
  export WANDB_TAGS="clock_owned,v49,bus-voltage,videos,seed-2,batch-v49,gridsearch"
  emit_manifest "mj-gs-v49-video"
}


# BATCH=v44: the vindication run. effort_drift removed (R29) - the
# sim-level torque clamp no longer ratchets - and the full frontier
# architecture gets its first run on honest physics: split governors,
# survivor/censored frontiers, adaptive window, stress valve, crash
# release, watchdog. 8 GPUs, 4000 iterations. Every prior death is
# explained; if this one holds its frontier equilibrium to 4000, the
# architecture and the sim are both clean and long-horizon training
# (DR rungs, terrain) opens.
gen_v44() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v44-honest-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v44"
  export RUN_NAME="clock_owned__v44-honest-physics__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v44,no-effort-drift,frontier-stack,batch-v44,gridsearch"
  emit_manifest "mj-gs-v44-honest"
}


# BATCH=v43: sim-state telemetry probe. Static env + frozen policy at
# 600 (v42a repro, shortened) with live model-field ratios logged; any
# in-place ratchet (forcerange/damping/armature/frictionloss) is visible
# within ~100 iters and its slope names the culprit event. If all ratios
# hold at 1.0 while the rot appears anyway, the drift lives in Data-side
# or warp-internal state and the minimal repro goes upstream to mjwarp.
gen_v43() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS=""
  export ADAPTIVE_PUSHES=""
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export PUSH_COHORT_FRAC="0.3"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOULE_W="0"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export SOFT_LANDING_PEAK_W="0"
  export FREEZE_POLICY_AFTER="600"
  export TRACK_WATCHDOG="0"
  export MAX_ITERATIONS="1600"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v43-simstate-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v43"
  export RUN_NAME="clock_owned__v43-simstate-probe__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v43,simstate-telemetry,batch-v43,gridsearch"
  emit_manifest "mj-gs-v43-simstate"
}


# BATCH=v42: the sim-state probe. v41 pincer: ignition is schedule-
# independent (v41a: PHASE=3000 moved nothing) AND training-independent
# (v41b: bit-frozen policy rotted faster, in an easing env). The rot is
# a function of accumulated sim time itself. v42 removes the last
# moving parts: fully static env definition (fixed commands/pushes,
# all penalty stages zero) - v42a bit-freezes the policy at 1200 and
# observes (watchdog off: we want the full rot curve); v42b keeps
# training on the same static env (watchdog on). Rot in v42a = pure
# simulator-state accumulation -> mjwarp reset_data audit. Flat = the
# rot needs a moving part; bisect by re-adding.
gen_v42() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS=""
  export ADAPTIVE_PUSHES=""
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export PUSH_COHORT_FRAC="0.3"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOULE_W="0"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export SOFT_LANDING_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v42"

  export FREEZE_POLICY_AFTER="1200"
  export TRACK_WATCHDOG="0"
  export MJLAB_LOG_STAMP="v42a-static-frozen-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v42a-static-frozen__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v42a,static-frozen,sim-state-probe,batch-v42,gridsearch"
  emit_manifest "mj-gs-v42a-frozen"

  export FREEZE_POLICY_AFTER=""
  export TRACK_WATCHDOG=""
  export MJLAB_LOG_STAMP="v42b-static-train-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v42b-static-train__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v42b,static-train,sim-state-probe,batch-v42,gridsearch"
  emit_manifest "mj-gs-v42b-train"
}


# BATCH=v41: the two-suspect discriminator pair. v40 proved the
# controller blameless (valve compressed exposure to exactly 1.000 and
# the rot accelerated through it; release followed capability down and
# the rot completed anyway - 7th demonstration). Remaining suspects:
# (a) the entropy schedule, whose coefficient bottoms exactly in the
# 1500-1700 ignition window on the PHASE=2000 profile - v41a stretches
# it to 3000 (onset should move to ~2400+ if implicated); (b) whether
# training updates are needed at all - v41b bit-freezes the policy at
# 1400 (R16 probe, never previously run to verdict).
gen_v41() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v41"

  export PHASE_ITERATIONS="3000"
  export MAX_ITERATIONS="3000"
  export MJLAB_LOG_STAMP="v41a-phase3k-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v41a-phase3000__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v41a,phase-3000,discriminator,batch-v41,gridsearch"
  emit_manifest "mj-gs-v41a-phase3k"

  export PHASE_ITERATIONS="2000"
  export MAX_ITERATIONS="2600"
  export FREEZE_POLICY_AFTER="1400"
  export MJLAB_LOG_STAMP="v41b-freeze-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v41b-freeze1400__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v41b,freeze-probe,discriminator,batch-v41,gridsearch"
  emit_manifest "mj-gs-v41b-freeze"
}


# BATCH=v40: R27 stack - fold gate 0.15, fast stale expiry, true floor
# release, stress-scaled headroom. v39 held 1800 healthy iterations; its
# three slow guards and the fixed-headroom poison drip are corrected.
# Prediction: the fall-rate creep relieves itself via the headroom valve
# before any crash machinery engages, and the run sawtooths indefinitely
# around the ~0.68 m/s ceiling.
gen_v40() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v40-valve-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v40"
  export RUN_NAME="clock_owned__v40-valve__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v40-valve,stress-headroom,batch-v40,gridsearch"
  emit_manifest "mj-gs-v40-valve"
}


# BATCH=v39: symmetric frontiers (R23) - the push axis gets the same
# treatment as commands: per-event shove magnitudes, horizon-free
# survival outcomes (survived = no fall before the next push or
# timeout), survival-vs-magnitude curve, and the push scale rides at
# survival_frontier x headroom with a survived-strength floor. "Push
# just beyond what we can survive" (user). Queue behind v38.
gen_v39() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v39-sym-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v39"
  export RUN_NAME="clock_owned__v39-symmetric-frontiers__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v39-symmetric,push-survival,batch-v39,gridsearch"
  emit_manifest "mj-gs-v39-sym"
}


# BATCH=v38: the frontier-tracking controller (R20, user design). The
# population-mean attain is fractionally hypersensitive at small
# commands and blind to WHERE failure lives; the conditional curve
# attain(v) is the capability curve, and its interpolated bar-crossing
# (attained_frontier) now owns both the target (frontier x 1.15 - a
# controlled ~15% of range beyond demonstrated capability) and the floor
# (95% of frontier trailing max). Queue behind v37 (R19 band): the pair
# separates "floor+glide suffice" from "conditional curve required".
gen_v38() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v38-frontier-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v38"
  export RUN_NAME="clock_owned__v38-frontier-track__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v38-frontier,attain-by-speed,batch-v38,gridsearch"
  emit_manifest "mj-gs-v38-frontier"
}


# BATCH=v37: the band-controller run (R19, user design). v35a/v36 died
# identically to every predecessor despite penalty removal - falsifying
# R17 and exposing the real two-sided controller flaw: the open-loop
# climb overshot attained capability by 40-50%, and the arrest cascade
# then cut commands far below attainment (0.98 -> 0.24 vs attained
# 0.66), which is exactly when falls exploded. Now the difficulty
# trajectory is owned by a band controller on measured clean attainment
# (climb > 0.66, hold in band, glide at bounded slew < 0.60) with a hard
# floor at 95% of best-attained speed that fall cuts cannot pierce.
# Envelope 1.6 so the band, not a cap, binds. 8 GPUs, fast feedback.
gen_v37() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.6"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v37-band-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v37"
  export RUN_NAME="clock_owned__v37-band-floor__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v37-band,attained-floor,minimal-penalties,batch-v37,gridsearch"
  emit_manifest "mj-gs-v37-band"
}


# BATCH=v36: the minimal-cocktail arm — joule + action_rate only
# (JOINT_ACC_W=0, TORQUE_RATE_PEAK_W=0). If v35a is clean, this tests
# whether torque_rate was earning anything in-sim: compare gait metrics
# (peak_height, feet spacing) and smoothness against v35a. Queued behind
# the v35 pair; runs on whichever node frees first.
gen_v36() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export OBS_NORM_FREEZE_ITERS="500"
  export JOINT_ACC_W="0"
  export TORQUE_RATE_PEAK_W="0"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v36-jouleonly-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v36"
  export RUN_NAME="clock_owned__v36-joule-only__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v36-joule-only,minimal-cocktail,batch-v36,gridsearch"
  emit_manifest "mj-gs-v36-jouleonly"
}


# BATCH=v35: the penalty-cocktail discrimination pair (user hypothesis,
# R17). Stage-4 arrival precedes every slide onset by 150-300 iters and
# moved WITH the ladder in v33 - the first variable that shifts the
# ignition. joint_acc_l2 dominates the energy pressure (~ -0.6..-1.0 ER
# at stage 4 vs tracking +1.7..2.1; ~10x torque_rate, ~600x joule).
# v35a drops joint_acc entirely (redundant kinematic proxy, convicted in
# v16c); v35b keeps all four at HALF peak pressure. Slide gone in a ->
# redundancy kill confirmed; slide gone only in b -> total pressure is
# the knob; slide in both -> penalty theory weakened, v34 verdict rules.
# 4-GPU each, 3000 iters, no landing anneal (unconfounded).
gen_v35() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export OBS_NORM_FREEZE_ITERS="500"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v35"

  export JOINT_ACC_W="0"
  export MJLAB_LOG_STAMP="v35a-nojacc-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v35a-no-jointacc__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v35a,no-joint-acc,penalty-discrim,batch-v35,gridsearch"
  emit_manifest "mj-gs-v35a-nojacc"

  export JOINT_ACC_W="-5e-5"
  export JOULE_W="-1.5e-4"
  export TORQUE_RATE_PEAK_W="-5e-4"
  export SOFT_LANDING_PEAK_W="-5e-3"
  export MJLAB_LOG_STAMP="v35b-half-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v35b-half-penalties__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v35b,half-penalties,penalty-discrim,batch-v35,gridsearch"
  emit_manifest "mj-gs-v35b-half"
}


# BATCH=v34: the R16 discriminating probe - a MEASUREMENT, not a fix.
# Three falsifications stand (task relief, LR floor, frozen obs
# normalizers - v30/v32/v33 all slid on the same ~1700 schedule). Two
# suspects remain: (A) residual training-side updates below LR 1e-5, or
# (B) env/measurement-side drift. This run trains normally to 1400
# (healthy plateau) then BIT-FREEZES the policy (optimizer.step no-op)
# and simply watches for 1600 more iterations. Slide appears anyway ->
# (B): the environment or the metric drifts under a constant policy.
# Flat -> (A): sub-floor updates are the residue. Either answer halves
# the space; no interpretation ambiguity.
gen_v34() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export OBS_NORM_FREEZE_ITERS="500"
  export FREEZE_POLICY_AFTER="1400"
  export MAX_ITERATIONS="3000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v34-probe-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v34"
  export RUN_NAME="clock_owned__v34-freeze-probe__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v34-probe,freeze-policy,batch-v34,gridsearch"
  emit_manifest "mj-gs-v34-probe"
}


# BATCH=v33: THE root-cause test (R15). Full stack + landing anneal +
# frozen obs normalizers (500 iters). v32 proved the rot is
# learning-rate-independent (identical slide at LR 1e-5 vs 2e-4); the
# remaining iteration-clocked, LR-immune updater is the empirical obs
# normalizer chasing the policy's own observation distribution. If v33
# holds its plateau past ~1700 without the slide, root cause confirmed
# and long-horizon training opens; if it ignites at ~1600 again, the
# hypothesis is dead. 4000 iters, 8 GPUs.
gen_v33() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export LANDING_ANNEAL="1"
  export OBS_NORM_FREEZE_ITERS="500"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v33-normfreeze-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v33"
  export RUN_NAME="clock_owned__v33-normfreeze__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v33-normfreeze,obs-norm-freeze,landing,batch-v33,gridsearch"
  emit_manifest "mj-gs-v33-normfreeze"
}


# BATCH=v32: the endurance test — full stack + landing anneal, 4000 iters,
# 8 GPUs. The bet: at capacity-plateau the anneal walks the LR to its
# floor BEFORE the churn fuse burns (v30 replay: factor < 0.15 by the
# ignition window), so the run converges at its peak instead of dying at
# ~2100. Success = watchdog never fires, attain holds its plateau to
# 4000, landing_factor bottoms out. Queue behind the v31 landing pair.
gen_v32() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export LANDING_ANNEAL="1"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v32-anneal-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v32"
  export RUN_NAME="clock_owned__v32-landing-anneal__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v32-anneal,landing,per-axis-aimd,ellipsoid,batch-v32,gridsearch"
  emit_manifest "mj-gs-v32-anneal"
}


# BATCH=v31: the champion-harvest landing pair. v30 closed the book on
# task-side rot rescue (5th demonstration: even cutting difficulty BEFORE
# falls rose could not stop the spiral once churn began). The only
# intervention with a perfect record is landing before the wall (v24d,
# v25). Two 4-GPU seeds in parallel (ceiling variance is large: v29 attain
# 0.712 vs v30 0.618 on identical config); land at 1800; best checkpoint
# of the pair goes to eval/sim2sim. Optimizer-side anneal is the next
# build for anything longer.
gen_v31() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export MAX_ITERATIONS="1800"
  export PHASE_ITERATIONS="2000"
  export EXPERIMENT_NAME="nugus_gridsearch_v31"

  export SEED="1"
  export MJLAB_LOG_STAMP="v31-land-s1-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v31-landing__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v31-landing,harvest,ellipsoid,push-cohort-0.3,batch-v31,gridsearch"
  emit_manifest "mj-gs-v31-land-s1"

  export SEED="2"
  export MJLAB_LOG_STAMP="v31-land-s2-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v31-landing__4gpu-6144__s2__${BATCH}"
  emit_manifest "mj-gs-v31-land-s2"
}


# BATCH=v30: v29 + attain-slide congestion. v29 proved everything else:
# split governor (push axis sawtoothed independently, no poison), record
# policy (attain 0.712 / x 0.74 / y 0.58 at the FULL extended envelope
# with 1.5x pushes), watchdog fail-fast. It died only of d_cmd parking at
# the cap (over-capacity commands under-track rather than fall under
# ellipsoid geometry, so fall-congestion never binds). v30 gives the
# command axis its second congestion signal: attain below 0.95x trailing
# max = cut. Prediction: d_cmd sawtooths off the attain ceiling around
# 0.85-1.0, no parking, no fuse, first full-length healthy 4000.
gen_v30() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v30-slide-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v30"
  export RUN_NAME="clock_owned__v30-attain-slide__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v30-slide,per-axis-aimd,ellipsoid,push-cohort-0.3,std-min-0.13,8gpu,multinode,batch-v30,gridsearch"
  emit_manifest "mj-gs-v30-slide"
}


# BATCH=v29: the split-governor run. v28 proved the stack mechanically
# (arrest ferocious, ellipsoid+envelope let the CLEAN cohort walk the
# extended ranges at fast-fall 0.03-0.07, watchdog failed the rot fast)
# and died only of the cohort-blind single scalar. v29 = same stack with
# per-axis control (d_cmd on clean falls, d_push on the excess rate) plus
# per-axis attainment logging (R12). Predictions: pushes sawtooth around
# the excess bar (~1.4-1.6x), clean commands hold the extended envelope,
# and if rot still appears with the poison stream gone, pure cap-churn is
# isolated as the next disease.
gen_v29() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v29-split-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v29"
  export RUN_NAME="clock_owned__v29-split-governor__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v29-split,per-axis-aimd,ellipsoid,push-cohort-0.3,std-min-0.13,8gpu,multinode,batch-v29,gridsearch"
  emit_manifest "mj-gs-v29-split"
}


# BATCH=v28: v27 + the shadow Lagrangian energy multiplier riding along
# (log-only; joule stays staged). Same AIMD+cohort stack at the 5986850
# pin. Launch AFTER v27 lands: judge v27's sawtooth-vs-frontier agreement
# first; if the estimator cross-validates, v28's successor hands command
# difficulty to it. Shadow-lambda validation: climbs while style healthy,
# freezes on peak-height dips, retreats before T4 would fire.
gen_v28() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export COMMAND_GEOMETRY="ellipsoid"
  export AIMD_ENVELOPE_SCALE="1.3"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="2"
  export MJLAB_LOG_STAMP="v28-shadow-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v28"
  export RUN_NAME="clock_owned__v28-arrest-ellipsoid__8gpu-6144__s2__${BATCH}"
  export WANDB_TAGS="clock_owned,v28-arrest,ellipsoid,envelope-1.3,shadow-lambda,push-cohort-0.3,std-min-0.13,8gpu,multinode,batch-v28,gridsearch"
  emit_manifest "mj-gs-v28-shadow"
}


# BATCH=v27: v26 + the decoupling layer. AIMD with the ssthresh wart fixed
# (full-rate climb), 30% push cohort (clean 70% = uncontaminated tracking
# signal + deployment-matched distribution), frontier estimator and
# push-to-fall histogram logging. Judge: sawtooth settling speed vs v26's
# probe-rate crawl; frontier_speed vs the d the sawtooth discovers; the
# recovery-time histogram calibrates any future horizon-based logic.
gen_v27() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export PUSH_COHORT_FRAC="0.3"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v27-aimd-cohort-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v27"
  export RUN_NAME="clock_owned__v27-aimd-cohort30__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v27-aimd,push-cohort-0.3,frontier,std-min-0.13,8gpu,multinode,batch-v27,gridsearch"
  emit_manifest "mj-gs-v27-aimd"
}


# BATCH=v26: AIMD continuous difficulty (doc 15 R8) — the TCP turn. One
# scalar d drives command ranges (lerp L0->L5 envelope) and push magnitude
# (0.75x->2.0x); additive increase 0.002/iter gated on health, 0.7x cut at
# fast-fall 0.35, 0.5x at 0.55, ssthresh slow-probe, refractory doubling at
# walls. No levels, no promote bars to get wrong: gates modulate the RATE.
# The rot watchdog fails the run fast if trkLin (>2.0 once) sustains <1.0
# for 60 iters. 8-GPU for rapid feedback; judge vs v25-push oscillator on
# time-at-difficulty, mean d, and best-checkpoint eval.
gen_v26() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export CURRICULUM_STYLE="aimd"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export MJLAB_LOG_STAMP="v26-aimd-$(date +%Y%m%d-%H%M%S)"
  export EXPERIMENT_NAME="nugus_gridsearch_v26"
  export RUN_NAME="clock_owned__v26-aimd__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v26-aimd,continuous-difficulty,std-min-0.13,8gpu,multinode,batch-v26,gridsearch"
  emit_manifest "mj-gs-v26-aimd"
}


# BATCH=v25: content-extension A/B on the two free nodes (4 GPUs each,
# proven single-node path). The era law: runs die ~700-900 iters after the
# ladder stops moving. Both arms extend content with env knobs only:
# v25-slow doubles per-rung tenure (cooldown 300; content ~2400; land 2600)
# testing whether longer rung tenure beats v24d's brief 500-iter L4 stay;
# v25-push extends the ladder past cmd-L4 with push rungs L3-L5 (scales
# 1.5/1.75/2.0 — robustness content, the sim2real play; land 2000).
gen_v25() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="1"
  export MULTINODE=""
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export ADAPTIVE_CMD_LMAX="4"
  export COMPETENCE_PROMOTE_ATTAIN="0.50"
  export COMPETENCE_DEMOTE_ATTAIN="0.35"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v25"

  export COMPETENCE_COOLDOWN_ITERS="300"
  export MAX_ITERATIONS="2600"
  export MJLAB_LOG_STAMP="v25-slow-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v25-slow-cd300__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v25-slow,cooldown-300,lmax4,batch-v25,gridsearch"
  emit_manifest "mj-gs-v25-slow"

  export COMPETENCE_COOLDOWN_ITERS=""
  export ADAPTIVE_PUSH_LMAX="5"
  export MAX_ITERATIONS="2000"
  export MJLAB_LOG_STAMP="v25-push-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v25-push-lmax5__4gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v25-push,push-lmax5,lmax4,batch-v25,gridsearch"
  emit_manifest "mj-gs-v25-push"
}


# BATCH=v24d: the planned landing. v24c proved the corridor (bar 0.50:
# L0->L4 by 1157, attain RECORD 0.701 at 1424, ~800 healthy iters at L4)
# and then proved the law again: saturation at the capped top rung lit the
# ~700-900-iter churn fuse at ~1424, the fast demote fired correctly at
# 1959 and the spiral completed THROUGH the cascade anyway — demotion is
# not an antidote to churn damage. With ~1400-1500 iters of genuine
# curriculum content in the current ladder, the only clean exit tonight is
# to stop at the peak: MAX_ITERATIONS=1600 (PHASE kept 2000 to reproduce
# the healthy climb exactly). Longer-term exits (more rungs / landing
# anneal) need code. Artifact: final checkpoint at the attainment peak.
gen_v24d() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export ADAPTIVE_CMD_LMAX="4"
  export COMPETENCE_PROMOTE_ATTAIN="0.50"
  export COMPETENCE_DEMOTE_ATTAIN="0.35"
  export PHASE_ITERATIONS="2000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v24"
  export MAX_ITERATIONS="1600"
  export MJLAB_LOG_STAMP="v24d-prod-$(date +%Y%m%d-%H%M%S)"
  export RUN_NAME="clock_owned__v24d-landing1600__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v24d,landing-1600,attain-0.50,lmax4,std-min-0.13,8gpu,multinode,batch-v24,gridsearch"
  emit_manifest "mj-gs-v24d-prod"
}


gen_v24b() {
  _v16e_r13_exports
  _competence_defaults
  export MJLAB_VARIANT="clock_owned"
  export PHASE_DELTA_W="-0.2"
  export ADAPTIVE_COMMANDS="1"
  export ADAPTIVE_PUSHES="1"
  export PENALTY_GATE="competence"
  export STD_MIN="0.13"
  export NUM_ENVS="6144"
  export JOB_REPLICAS="2"
  export MULTINODE="1"
  export COMPETENCE_DEMOTE_FAST_FELL="0.35"
  export ADAPTIVE_CMD_LMAX="4"
  export MJLAB_LOG_STAMP="v24b-prod-$(date +%Y%m%d-%H%M%S)"
  export MAX_ITERATIONS="4000"
  export PHASE_ITERATIONS="4000"
  export SEED="1"
  export EXPERIMENT_NAME="nugus_gridsearch_v24"
  export RUN_NAME="clock_owned__v24b-bar035-lmax4__8gpu-6144__s1__${BATCH}"
  export WANDB_TAGS="clock_owned,v24b,fast-bar-0.35,lmax4,std-min-0.13,8gpu,multinode,batch-v24,gridsearch"
  emit_manifest "mj-gs-v24b-prod"
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
  mn-bench) gen_mn_bench; expected=6 ;;
  v21) gen_v21; expected=2 ;;
  v22) gen_v22; expected=2 ;;
  v23) gen_v23; expected=1 ;;
  v24) gen_v24; expected=1 ;;
  v24b) gen_v24b; expected=1 ;;
  v24c) gen_v24c; expected=1 ;;
  v24d) gen_v24d; expected=1 ;;
  v25) gen_v25; expected=2 ;;
  v26) gen_v26; expected=1 ;;
  v27) gen_v27; expected=1 ;;
  v28) gen_v28; expected=1 ;;
  v29) gen_v29; expected=1 ;;
  v30) gen_v30; expected=1 ;;
  v31) gen_v31; expected=2 ;;
  v32) gen_v32; expected=1 ;;
  v33) gen_v33; expected=1 ;;
  v34) gen_v34; expected=1 ;;
  v35) gen_v35; expected=2 ;;
  v36) gen_v36; expected=1 ;;
  v37) gen_v37; expected=1 ;;
  v38) gen_v38; expected=1 ;;
  v39) gen_v39; expected=1 ;;
  v40) gen_v40; expected=1 ;;
  v41) gen_v41; expected=2 ;;
  v42) gen_v42; expected=2 ;;
  v43) gen_v43; expected=1 ;;
  v44) gen_v44; expected=1 ;;
  v45) gen_v45; expected=1 ;;
  v45b) gen_v45b; expected=2 ;;
  v45c) gen_v45c; expected=3 ;;
  v46) gen_v46; expected=4 ;;
  v47) gen_v47; expected=5 ;;
  v48) gen_v48; expected=6 ;;
  v49) gen_v49; expected=7 ;;
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
