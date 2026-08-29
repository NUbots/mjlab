#!/usr/bin/env bash
# Collect the velocity-tracking and stability data for any set of controllers.
#
#   scripts/eval/collect_comparison.sh [--out DIR] <controller> [<controller>...]
#   scripts/eval/collect_comparison.sh <rl-checkpoint> [output-dir]
#
# A controller is a comma-separated list of key=value fields:
#
#   engine=      quintic | rl                                       (required)
#   name=        slug used in run tags and figure filenames    (default: engine)
#   label=       name shown on the figures            (default: built from name)
#   checkpoint=  path to the .pt                                (rl, required)
#   task=        registered task id supplying the observation pipeline
#                (rl, optional; needed when the policy's observation layout is
#                not the default task's -- an observation-history policy, say)
#   colour=      #rrggbb for this controller's series   (default: the plotter's
#                palette, assigned in the order the controllers are given)
#
# Seven runs per controller: a profile run (a moving command, for the tracking
# time series), three single-axis command sweeps, and three two-axis command
# grids (which carry both the combined-axis tracking and, through fall_time,
# the stability envelope).
#
# Every environment in a sweep or a grid is a distinct command: the plant is
# deterministic and the domain randomisation is off, so replicas of one command
# would be replicas of one number. An RL policy is the exception -- it sees
# noisy observations -- so the single-axis sweeps carry four replicas per point
# for every controller, which gives a policy a band and the engine a
# (degenerate) line drawn by the same code.
#
# Examples::
#
#   # The original two-way comparison.
#   scripts/eval/collect_comparison.sh \
#     engine=rl,checkpoint=logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run>/model_39997.pt \
#     engine=quintic
#
#   # Two policies against each other, no walk engine.
#   scripts/eval/collect_comparison.sh --out logs/eval/policies \
#     engine=rl,name=small,label='RL (small)',checkpoint=.../small/model_39997.pt \
#     engine=rl,name=history,label='RL (obs history)',checkpoint=.../hist/model_39997.pt,task=Mjlab-Velocity-Flat-Nubots-Nugus-History
#
#   # Three-way: both policies and the walk engine.
#   scripts/eval/collect_comparison.sh --out logs/eval/three_way \
#     engine=rl,name=small,label='RL (small)',checkpoint=.../small/model_39997.pt \
#     engine=rl,name=history,label='RL (obs history)',checkpoint=.../hist/model_39997.pt,task=Mjlab-Velocity-Flat-Nubots-Nugus-History \
#     engine=quintic
#
# The command ranges, the run length and the replica count are read from the
# environment, so a quick pass over a coarser grid needs no edit here:
#
#   DURATION=20 VX_STEP=0.1 VX_GRID_STEP=0.25 scripts/eval/collect_comparison.sh ...
set -euo pipefail

USAGE="usage: collect_comparison.sh [--out DIR] <controller> [<controller>...]
       collect_comparison.sh <rl-checkpoint> [output-dir]

A controller is a comma-separated key=value list, e.g.
  engine=quintic
  engine=rl,name=small,label='RL (small)',checkpoint=<path>.pt
  engine=rl,name=history,checkpoint=<path>.pt,task=<task-id>

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
    --out | --output-dir)
      [[ $# -ge 2 ]] || die "${1} needs a directory"
      OUT=$2
      shift 2
      ;;
    --out=* | --output-dir=*)
      OUT=${1#*=}
      shift
      ;;
    -h | --help)
      echo "${USAGE}"
      exit 0
      ;;
    -*)
      die "unknown option: ${1}
${USAGE}"
      ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

((${#SPECS[@]})) || die "${USAGE}"

# The original form -- a bare checkpoint, then an optional output directory --
# still works, and still means the RL policy against the walk engine. A spec
# always carries an ``engine=`` field, so the two forms cannot be confused.
if [[ ${SPECS[0]} != *=* ]]; then
  ((${#SPECS[@]} <= 2)) || die "${USAGE}"
  ((${#SPECS[@]} == 2)) && OUT=${SPECS[1]}
  SPECS=("engine=rl,checkpoint=${SPECS[0]}" "engine=quintic")
fi

# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

NAMES=()
ENGINES=()
LABELS=()
CHECKPOINTS=()
TASKS=()
COLOURS=()

# A checkpoint path that is not a file is the common way to start a collection
# that cannot finish. Checked before anything runs, because the alternative is
# discovering it after the controllers ahead of it have burned twenty minutes.
# A wandb run id on its own is the easy mistake -- the field wants the path to
# the .pt inside that run's directory.
check_checkpoint() {
  local path=$1
  [[ -f ${path} ]] && return 0
  echo "not a checkpoint file: ${path}" >&2
  echo "expected a path such as" >&2
  echo "  logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run-id>/model_39997.pt" >&2
  if [[ -d logs/rsl_rl/nugus_velocity/wandb_checkpoints/${path} ]]; then
    echo "that looks like a run id; did you mean one of:" >&2
    ls -1 "logs/rsl_rl/nugus_velocity/wandb_checkpoints/${path}" \
      | sed "s|^|  logs/rsl_rl/nugus_velocity/wandb_checkpoints/${path}/|" >&2
  fi
  exit 1
}

parse_spec() {
  local spec=$1
  local engine="" name="" label="" checkpoint="" task="" colour=""
  local -a fields=()
  local field key value

  IFS=, read -ra fields <<<"${spec}"
  for field in "${fields[@]}"; do
    [[ -n ${field} ]] || continue
    [[ ${field} == *=* ]] || die "controller ${spec}: '${field}' is not key=value
${USAGE}"
    key=${field%%=*}
    value=${field#*=}
    case ${key} in
      engine) engine=${value} ;;
      name) name=${value} ;;
      label) label=${value} ;;
      checkpoint) checkpoint=${value} ;;
      task | task_id | task-id) task=${value} ;;
      colour | color) colour=${value} ;;
      *) die "controller ${spec}: unknown field '${key}'
${USAGE}" ;;
    esac
  done

  case ${engine} in
    quintic)
      [[ -z ${checkpoint} ]] || die "controller ${spec}: quintic takes no checkpoint"
      [[ -z ${task} ]] || die "controller ${spec}: quintic takes no task"
      ;;
    rl)
      [[ -n ${checkpoint} ]] || die "controller ${spec}: engine=rl needs a checkpoint"
      check_checkpoint "${checkpoint}"
      ;;
    "") die "controller ${spec}: no engine=
${USAGE}" ;;
    *) die "controller ${spec}: engine must be 'quintic' or 'rl', not '${engine}'" ;;
  esac

  name=${name:-${engine}}
  # The name lands in a directory name and in a figure filename, so keep it to
  # what both survive without quoting.
  [[ ${name} =~ ^[A-Za-z0-9_-]+$ ]] \
    || die "controller ${spec}: name '${name}' must be letters, digits, - or _"
  local existing
  for existing in ${NAMES[@]+"${NAMES[@]}"}; do
    [[ ${existing} != "${name}" ]] \
      || die "two controllers are both named '${name}'; give each a distinct name="
  done

  if [[ -z ${label} ]]; then
    if [[ ${engine} == quintic ]]; then
      label="Quintic walk engine"
      [[ ${name} == quintic ]] || label="Quintic walk engine (${name})"
    else
      label="RL policy"
      [[ ${name} == rl ]] || label="RL policy (${name})"
    fi
  fi

  NAMES+=("${name}")
  ENGINES+=("${engine}")
  LABELS+=("${label}")
  CHECKPOINTS+=("${checkpoint}")
  TASKS+=("${task}")
  COLOURS+=("${colour}")
}

for spec in "${SPECS[@]}"; do
  parse_spec "${spec}"
done

# RL first, deliberately. A checkpoint can pass the file check above and still
# fail to load -- a shape mismatch against the observation layout the named
# task builds is the usual way -- and that failure takes seconds. Collecting a
# quintic half first would put twenty minutes of work in front of it.
ORDER=()
for i in "${!NAMES[@]}"; do
  [[ ${ENGINES[i]} == rl ]] && ORDER+=("$i")
done
for i in "${!NAMES[@]}"; do
  [[ ${ENGINES[i]} == rl ]] || ORDER+=("$i")
done

# --------------------------------------------------------------------------
# Run parameters
# --------------------------------------------------------------------------

# Long enough that the run-up is a small part of it, and long enough for a
# stability horizon worth plotting.
DURATION=${DURATION:-60}
WARMUP=${WARMUP:-8}

# Commands (sent to sim environments)
VX_MIN=${VX_MIN:--2.0}
VX_MAX=${VX_MAX:-2.0}
VX_STEP=${VX_STEP:-0.025}

VY_MIN=${VY_MIN:--1.0}
VY_MAX=${VY_MAX:-1.0}
VY_STEP=${VY_STEP:-0.025}

WZ_MIN=${WZ_MIN:--3.0}
WZ_MAX=${WZ_MAX:-3.0}
WZ_STEP=${WZ_STEP:-0.05}

# Grid axes (used for plotting)
VX_GRID_MIN=${VX_GRID_MIN:--2.0}
VX_GRID_MAX=${VX_GRID_MAX:-2.0}
VX_GRID_STEP=${VX_GRID_STEP:-0.1}

VY_GRID_MIN=${VY_GRID_MIN:--1.0}
VY_GRID_MAX=${VY_GRID_MAX:-1.0}
VY_GRID_STEP=${VY_GRID_STEP:-0.1}

WZ_GRID_MIN=${WZ_GRID_MIN:--3.0}
WZ_GRID_MAX=${WZ_GRID_MAX:-3.0}
WZ_GRID_STEP=${WZ_GRID_STEP:-0.2}


# Generate comma-separated lists from the above ranges and step sizes
make_axis() {
    local min=$1
    local max=$2
    local step=$3

    seq "$min" "$step" "$max" |
        awk 'BEGIN { printf "(" }
             { if (NR > 1) printf ","; printf "%.10g", $0 }
             END { printf ")" }'
}

VX_FINE=$(make_axis "$VX_MIN" "$VX_MAX" "$VX_STEP")
VY_FINE=$(make_axis "$VY_MIN" "$VY_MAX" "$VY_STEP")
WZ_FINE=$(make_axis "$WZ_MIN" "$WZ_MAX" "$WZ_STEP")

VX_GRID=$(make_axis "$VX_GRID_MIN" "$VX_GRID_MAX" "$VX_GRID_STEP")
VY_GRID=$(make_axis "$VY_GRID_MIN" "$VY_GRID_MAX" "$VY_GRID_STEP")
WZ_GRID=$(make_axis "$WZ_GRID_MIN" "$WZ_GRID_MAX" "$WZ_GRID_STEP")


# Number of points per axis
N_VX_FINE=$(awk "BEGIN { print int(($VX_MAX - $VX_MIN) / $VX_STEP + 0.5) + 1 }")
N_VY_FINE=$(awk "BEGIN { print int(($VY_MAX - $VY_MIN) / $VY_STEP + 0.5) + 1 }")
N_WZ_FINE=$(awk "BEGIN { print int(($WZ_MAX - $WZ_MIN) / $WZ_STEP + 0.5) + 1 }")

N_VX_GRID=$(awk "BEGIN { print int(($VX_GRID_MAX - $VX_GRID_MIN) / $VX_GRID_STEP + 0.5) + 1 }")
N_VY_GRID=$(awk "BEGIN { print int(($VY_GRID_MAX - $VY_GRID_MIN) / $VY_GRID_STEP + 0.5) + 1 }")
N_WZ_GRID=$(awk "BEGIN { print int(($WZ_GRID_MAX - $WZ_GRID_MIN) / $WZ_GRID_STEP + 0.5) + 1 }")

# Cartesian-product sizes
REPLICAS=${REPLICAS:-4}
N_XY=$((N_VX_GRID * N_VY_GRID))
N_XW=$((N_VX_GRID * N_WZ_GRID))
N_YW=$((N_VY_GRID * N_WZ_GRID))

run_controller() {
  local index=$1
  local name=${NAMES[index]}
  local engine=${ENGINES[index]}
  local script="scripts/eval/eval_quintic_walk.py"
  local -a extra=()

  if [[ ${engine} == rl ]]; then
    script="scripts/eval/eval_rl_walk.py"
    extra+=(--checkpoint "${CHECKPOINTS[index]}")
    if [[ -n ${TASKS[index]} ]]; then
      extra+=(--task-id "${TASKS[index]}")
    fi
  fi

  echo "=== ${name} (${engine}): velocity profile ==="
  uv run python scripts/eval/eval_velocity_profile.py \
    --engine "${engine}" "${extra[@]}" \
    --profile.hold 6.0 --profile.ramp 1.5 --profile.replicas 4 \
    --output-dir "${OUT}" --tag "profile_${name}"

  echo "=== ${name} (${engine}): single-axis sweeps ==="
  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_VX_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_FINE}" --output-dir "${OUT}" --tag "sweep_vx_${name}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_VY_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vy "${VY_FINE}" --output-dir "${OUT}" --tag "sweep_vy_${name}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_WZ_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-wz "${WZ_FINE}" --output-dir "${OUT}" --tag "sweep_wz_${name}"

  echo "=== ${name} (${engine}): two-axis grids ==="
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_XY} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_GRID}" --sweep-vy "${VY_GRID}" \
    --output-dir "${OUT}" --tag "grid_vx_vy_${name}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_XW} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_GRID}" --sweep-wz "${WZ_GRID}" \
    --output-dir "${OUT}" --tag "grid_vx_wz_${name}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_YW} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vy "${VY_GRID}" --sweep-wz "${WZ_GRID}" \
    --output-dir "${OUT}" --tag "grid_vy_wz_${name}"
}

# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

mkdir -p "${OUT}"

# The manifest is how plot_comparison.py learns which controllers are in this
# directory, what to call them and in what order to draw them. Written before
# the runs so an interrupted collection still says what it was collecting.
manifest_args=()
for i in "${!NAMES[@]}"; do
  manifest_args+=(
    "${NAMES[i]}" "${ENGINES[i]}" "${LABELS[i]}"
    "${CHECKPOINTS[i]}" "${TASKS[i]}" "${COLOURS[i]}"
  )
done
uv run python - "${OUT}/controllers.json" "${manifest_args[@]}" <<'PY'
import json
import sys

path, *flat = sys.argv[1:]
fields = ("name", "engine", "label", "checkpoint", "task", "colour")
controllers = [
  {key: value or None for key, value in zip(fields, flat[i : i + len(fields)])}
  for i in range(0, len(flat), len(fields))
]
# name and engine are never empty; the rest are optional and stay null.
for controller in controllers:
  controller["name"] = controller["name"] or ""
  controller["engine"] = controller["engine"] or ""
with open(path, "w") as handle:
  json.dump({"controllers": controllers}, handle, indent=2)
  handle.write("\n")
PY

echo "collecting ${#NAMES[@]} controller(s) into ${OUT}:"
for i in "${ORDER[@]}"; do
  echo "  ${NAMES[i]} (${ENGINES[i]}) — ${LABELS[i]}"
done
echo

for i in "${ORDER[@]}"; do
  run_controller "$i"
done

echo
echo "collected into ${OUT}"
echo "now: uv run python scripts/eval/plot_comparison.py --input-dir ${OUT}"
