#!/usr/bin/env bash
# Collect the velocity-tracking and stability data for two controllers.
#
#   scripts/eval/collect_comparison.sh <rl-checkpoint> [output-dir]
#
# Fourteen runs on the evaluation plant: for each controller a profile run (a
# moving command, for the tracking time series), three single-axis command
# sweeps, and three two-axis command grids (which carry both the combined-axis
# tracking and, through fall_time, the stability envelope).
#
# Every environment in a sweep or a grid is a distinct command: the plant is
# deterministic and the domain randomisation is off, so replicas of one command
# would be replicas of one number. The RL policy is the exception -- it sees
# noisy observations -- so the single-axis sweeps carry four replicas per point
# for both controllers, which gives the policy a band and the engine a
# (degenerate) line drawn by the same code.
set -euo pipefail

CHECKPOINT=${1:?usage: collect_comparison.sh <rl-checkpoint> [output-dir]}
OUT=${2:-logs/eval/comparison}

# Checked here rather than where it is used: the RL half runs second, so a bad
# checkpoint would otherwise surface twenty minutes in, after the whole quintic
# half had already been collected. A wandb run id on its own is the easy
# mistake -- the argument is the path to the .pt inside that run's directory.
if [[ ! -f ${CHECKPOINT} ]]; then
  echo "not a checkpoint file: ${CHECKPOINT}" >&2
  echo "expected a path such as" >&2
  echo "  logs/rsl_rl/nugus_velocity/wandb_checkpoints/<run-id>/model_39997.pt" >&2
  if [[ -d logs/rsl_rl/nugus_velocity/wandb_checkpoints/${CHECKPOINT} ]]; then
    echo "that looks like a run id; did you mean one of:" >&2
    ls -1 "logs/rsl_rl/nugus_velocity/wandb_checkpoints/${CHECKPOINT}" \
      | sed "s|^|  logs/rsl_rl/nugus_velocity/wandb_checkpoints/${CHECKPOINT}/|" >&2
  fi
  exit 1
fi

# Long enough that the run-up is a small part of it, and long enough for a
# stability horizon worth plotting.
DURATION=60
WARMUP=8

# Commands (sent to sim environments)
VX_MIN=-2.0
VX_MAX=2.0
VX_STEP=0.025

VY_MIN=-1.0
VY_MAX=1.0
VY_STEP=0.025

WZ_MIN=-3.0
WZ_MAX=3.0
WZ_STEP=0.05

# Grid axes (used for plotting)
VX_GRID_MIN=-2.0
VX_GRID_MAX=2.0
VX_GRID_STEP=0.1

VY_GRID_MIN=-1.0
VY_GRID_MAX=1.0
VY_GRID_STEP=0.1

WZ_GRID_MIN=-3.0
WZ_GRID_MAX=3.0
WZ_GRID_STEP=0.2


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
REPLICAS=4
N_XY=$((N_VX_GRID * N_VY_GRID))
N_XW=$((N_VX_GRID * N_WZ_GRID))
N_YW=$((N_VY_GRID * N_WZ_GRID))

run_engine() {
  local engine=$1
  shift
  local extra=("$@")

  echo "=== ${engine}: velocity profile ==="
  uv run python scripts/eval/eval_velocity_profile.py \
    --engine "${engine}" "${extra[@]}" \
    --profile.hold 6.0 --profile.ramp 1.5 --profile.replicas 4 \
    --output-dir "${OUT}" --tag "profile_${engine}"

  echo "=== ${engine}: single-axis sweeps ==="
  local script="scripts/eval/eval_quintic_walk.py"
  if [[ ${engine} == rl ]]; then
    script="scripts/eval/eval_rl_walk.py"
  fi

  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_VX_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_FINE}" --output-dir "${OUT}" --tag "sweep_vx_${engine}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_VY_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vy "${VY_FINE}" --output-dir "${OUT}" --tag "sweep_vy_${engine}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs $((N_WZ_FINE * REPLICAS)) --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-wz "${WZ_FINE}" --output-dir "${OUT}" --tag "sweep_wz_${engine}"

  echo "=== ${engine}: two-axis grids ==="
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_XY} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_GRID}" --sweep-vy "${VY_GRID}" \
    --output-dir "${OUT}" --tag "grid_vx_vy_${engine}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_XW} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vx "${VX_GRID}" --sweep-wz "${WZ_GRID}" \
    --output-dir "${OUT}" --tag "grid_vx_wz_${engine}"
  uv run python "${script}" "${extra[@]}" \
    --num-envs ${N_YW} --duration ${DURATION} --warmup ${WARMUP} \
    --sweep-vy "${VY_GRID}" --sweep-wz "${WZ_GRID}" \
    --output-dir "${OUT}" --tag "grid_vy_wz_${engine}"
}

mkdir -p "${OUT}"
run_engine quintic
run_engine rl --checkpoint "${CHECKPOINT}"

echo
echo "collected into ${OUT}"
