#!/usr/bin/env bash
# Collect the data for the comparison figures, for any number of policies.
#
#   scripts/eval/collect_comparison.sh [--out DIR] <controller> [<controller>...]
#
# A controller is either a checkpoint path on its own, or a comma-separated
# list of key=value fields:
#
#   checkpoint=  path to the .pt                                     (required)
#   name=        slug used in run directories and figure filenames
#                (default: the checkpoint's parent directory, i.e. the run id)
#   label=       name shown on the figures                   (default: name)
#   task=        registered task the policy was trained on
#                (default: Mjlab-Velocity-Flat-Booster-K1)
#   colour=      #rrggbb for this controller's series  (default: the plotter's
#                palette, in the order the controllers are given)
#
# Six runs per controller, exactly what plot_comparison.py draws:
#
#   profile_<name>      a moving command, for the velocity profile figure
#   grid_vx_vy_<name>   \
#   grid_vx_wz_<name>    > two-axis command grids, for the normalised error plane
#   grid_vy_wz_<name>   /
#   push_walk_<name>    push battery while walking at PUSH_VX  \  push survival
#   push_stand_<name>   push battery at a zero command         /  envelope
#
# Examples::
#
#   scripts/eval/collect_comparison.sh \
#     logs/rsl_rl/k1_velocity/wandb_checkpoints/g117b959/model_14999.pt
#
#   scripts/eval/collect_comparison.sh --out logs/eval/k1 \
#     checkpoint=.../g117b959/model_14999.pt,label='K1 (gait clock off)' \
#     checkpoint=.../t98yksya/model_14999.pt,label='K1 (baseline)'
#
# Run length, grid axes and the push battery are read from the environment, so
# a quick pass needs no edit here:
#
#   DURATION=10 WARMUP=4 VX_STEP=0.5 VY_STEP=0.4 WZ_STEP=1.0 \
#     PUSH_DV="(1.0,2.0,3.0)" PUSH_PHASES=2 PUSH_REPLICAS=1 \
#     scripts/eval/collect_comparison.sh <checkpoint>
set -euo pipefail

USAGE="usage: collect_comparison.sh [--out DIR] <controller> [<controller>...]

A controller is a checkpoint path, or a comma-separated key=value list, e.g.
  checkpoint=<path>.pt,name=baseline,label='K1 baseline'

See the header of this file for the full field list."

die() {
  echo "$@" >&2
  exit 1
}

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

OUT=${OUT:-logs/eval/comparison}
SPECS=()

while (($#)); do
  case $1 in
    --out)
      [[ $# -ge 2 ]] || die "--out needs a directory"
      OUT=$2
      shift 2
      ;;
    --out=*)
      OUT=${1#*=}
      shift
      ;;
    -h | --help)
      echo "${USAGE}"
      exit 0
      ;;
    -*) die "unknown option: ${1}
${USAGE}" ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

((${#SPECS[@]})) || die "${USAGE}"

# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

NAMES=()
LABELS=()
CHECKPOINTS=()
TASKS=()
COLOURS=()

parse_spec() {
  local spec=$1
  local name="" label="" checkpoint="" task="" colour=""
  local -a fields=()
  local field key value

  if [[ ${spec} != *=* ]]; then
    checkpoint=${spec}
  else
    IFS=, read -ra fields <<<"${spec}"
    for field in "${fields[@]}"; do
      [[ -n ${field} ]] || continue
      [[ ${field} == *=* ]] || die "controller ${spec}: '${field}' is not key=value"
      key=${field%%=*}
      value=${field#*=}
      case ${key} in
        checkpoint) checkpoint=${value} ;;
        name) name=${value} ;;
        label) label=${value} ;;
        task) task=${value} ;;
        colour | color) colour=${value} ;;
        *) die "controller ${spec}: unknown field '${key}'
${USAGE}" ;;
      esac
    done
  fi

  # Checked before anything runs, rather than after the controllers ahead of
  # this one have burned an hour.
  [[ -n ${checkpoint} ]] || die "controller ${spec}: no checkpoint="
  [[ -f ${checkpoint} ]] || die "not a checkpoint file: ${checkpoint}
expected a path such as
  logs/rsl_rl/k1_velocity/wandb_checkpoints/<run-id>/model_14999.pt"

  name=${name:-$(basename "$(dirname "${checkpoint}")")}
  # The name lands in a directory name and in a figure filename.
  [[ ${name} =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "controller ${spec}: name '${name}' must be letters, digits, - or _"
  local existing
  for existing in ${NAMES[@]+"${NAMES[@]}"}; do
    [[ ${existing} != "${name}" ]] \
      || die "two controllers are both named '${name}'; give each a distinct name="
  done

  NAMES+=("${name}")
  LABELS+=("${label:-${name}}")
  CHECKPOINTS+=("${checkpoint}")
  TASKS+=("${task}")
  COLOURS+=("${colour}")
}

for spec in "${SPECS[@]}"; do
  parse_spec "${spec}"
done

# --------------------------------------------------------------------------
# Run parameters
# --------------------------------------------------------------------------

# Grids. Each run holds one command per robot for DURATION seconds and
# averages the tracking over everything after WARMUP. The ranges run past the
# K1's training envelope (vx -1..2, vy +/-0.8, wz +/-2) so the planes show
# where tracking breaks down, not just the region that was trained.
DURATION=${DURATION:-30}
WARMUP=${WARMUP:-8}

VX_MIN=${VX_MIN:--1.5}
VX_MAX=${VX_MAX:-2.5}
VX_STEP=${VX_STEP:-0.1}

VY_MIN=${VY_MIN:--1.2}
VY_MAX=${VY_MAX:-1.2}
VY_STEP=${VY_STEP:-0.1}

WZ_MIN=${WZ_MIN:--3.0}
WZ_MAX=${WZ_MAX:-3.0}
WZ_STEP=${WZ_STEP:-0.2}

# Push batteries: walking at PUSH_VX, and standing. Anything left unset here
# takes the default in mjlab.evaluation.push.PushCfg.
PUSH_VX=${PUSH_VX:-0.5}

# "(a,b,c)" from a range and a step.
make_axis() {
  seq "$1" "$3" "$2" | awk 'BEGIN { printf "(" }
    { if (NR > 1) printf ","; printf "%.10g", $0 }
    END { printf ")" }'
}

VX_AXIS=$(make_axis "${VX_MIN}" "${VX_MAX}" "${VX_STEP}")
VY_AXIS=$(make_axis "${VY_MIN}" "${VY_MAX}" "${VY_STEP}")
WZ_AXIS=$(make_axis "${WZ_MIN}" "${WZ_MAX}" "${WZ_STEP}")

# Pass --<flag> <value> only for the environment variables that are set.
optional_flags() {
  local -n _out=$1
  shift
  local flag var
  while (($#)); do
    flag=$1 var=$2
    shift 2
    [[ -z ${!var:-} ]] || _out+=("${flag}" "${!var}")
  done
}

PROFILE_FLAGS=()
optional_flags PROFILE_FLAGS \
  --profile.vx PROFILE_VX \
  --profile.vy PROFILE_VY \
  --profile.wz PROFILE_WZ \
  --profile.combined-vx PROFILE_COMBINED_VX \
  --profile.combined-vy PROFILE_COMBINED_VY \
  --profile.combined-wz PROFILE_COMBINED_WZ \
  --profile.hold PROFILE_HOLD \
  --profile.ramp PROFILE_RAMP \
  --profile.rest PROFILE_REST \
  --profile.replicas PROFILE_REPLICAS

PUSH_FLAGS=()
optional_flags PUSH_FLAGS \
  --push.delta-v PUSH_DV \
  --push.directions PUSH_DIRECTIONS \
  --push.phases PUSH_PHASES \
  --push.replicas PUSH_REPLICAS \
  --push.duration PUSH_DURATION \
  --push.settle PUSH_SETTLE \
  --push.recovery PUSH_RECOVERY

# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

evaluate() {
  local index=$1 mode=$2 tag=$3
  shift 3
  local -a common=(--checkpoint "${CHECKPOINTS[index]}")
  [[ -z ${TASKS[index]} ]] || common+=(--task-id "${TASKS[index]}")
  echo "=== ${NAMES[index]}: ${tag} ==="
  uv run python scripts/eval/eval_rl_walk.py "${mode}" "${common[@]}" "$@" \
    --output-dir "${OUT}" --tag "${tag}_${NAMES[index]}"
}

run_controller() {
  local i=$1
  # The profile first: it is the quickest run, so a checkpoint that does not
  # load against its task fails in seconds rather than after the grids.
  evaluate "$i" profile profile ${PROFILE_FLAGS[@]+"${PROFILE_FLAGS[@]}"}

  local -a grid=(--duration "${DURATION}" --warmup "${WARMUP}")
  evaluate "$i" grid grid_vx_vy "${grid[@]}" --vx "${VX_AXIS}" --vy "${VY_AXIS}"
  evaluate "$i" grid grid_vx_wz "${grid[@]}" --vx "${VX_AXIS}" --wz "${WZ_AXIS}"
  evaluate "$i" grid grid_vy_wz "${grid[@]}" --vy "${VY_AXIS}" --wz "${WZ_AXIS}"

  evaluate "$i" push push_walk --push.vx "${PUSH_VX}" \
    ${PUSH_FLAGS[@]+"${PUSH_FLAGS[@]}"}
  evaluate "$i" push push_stand --push.vx 0.0 ${PUSH_FLAGS[@]+"${PUSH_FLAGS[@]}"}
}

mkdir -p "${OUT}"

# The manifest tells plot_comparison.py which controllers are in this
# directory, what to call them and in what order to draw them. Written before
# the runs so an interrupted collection still says what it was collecting.
manifest_args=()
for i in "${!NAMES[@]}"; do
  manifest_args+=("${NAMES[i]}" "${LABELS[i]}" "${CHECKPOINTS[i]}" "${TASKS[i]}"
    "${COLOURS[i]}")
done
uv run python - "${OUT}/controllers.json" "${manifest_args[@]}" <<'PY'
import json
import sys

path, *flat = sys.argv[1:]
keys = ("name", "label", "checkpoint", "task", "colour")
controllers = [
  {key: value or None for key, value in zip(keys, flat[i : i + len(keys)])}
  for i in range(0, len(flat), len(keys))
]
with open(path, "w") as handle:
  json.dump({"controllers": controllers}, handle, indent=2)
  handle.write("\n")
PY

echo "collecting ${#NAMES[@]} controller(s) into ${OUT}:"
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[i]} — ${LABELS[i]}"
done
echo

for i in "${!NAMES[@]}"; do
  run_controller "$i"
done

echo
echo "collected into ${OUT}"
echo "now: uv run python scripts/eval/plot_comparison.py --input-dir ${OUT}"
