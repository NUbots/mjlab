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

# Long enough that the run-up is a small part of it, and long enough for a
# stability horizon worth plotting.
DURATION=60
WARMUP=8

# Command axes. Forward reaches past what either controller can do; lateral and
# yaw are symmetric so a one-sided gait shows as an asymmetry.
VX_FINE="(-0.5,-0.475,-0.45,-0.425,-0.4,-0.375,-0.35,-0.325,-0.3,-0.275,-0.25,-0.225,-0.2,-0.175,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0.0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7)"
VY_FINE="(-0.4,-0.375,-0.35,-0.325,-0.3,-0.275,-0.25,-0.225,-0.2,-0.175,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0.0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4)"
WZ_FINE="(-1.5,-1.4375,-1.375,-1.3125,-1.25,-1.1875,-1.125,-1.0625,-1.0,-0.9375,-0.875,-0.8125,-0.75,-0.6875,-0.625,-0.5625,-0.5,-0.4375,-0.375,-0.3125,-0.25,-0.1875,-0.125,-0.0625,0.0,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375,1.0,1.0625,1.125,1.1875,1.25,1.3125,1.375,1.4375,1.5)"

VX_GRID="(-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6)"
VY_GRID="(-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3)"
WZ_GRID="(-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)"

# Points per axis, kept in step with the lists above.
N_VX_FINE=49
N_VY_FINE=33
N_WZ_FINE=49
REPLICAS=4
N_XY=273   # 21 x 13
N_XW=441   # 21 x 21
N_YW=273   # 13 x 21

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
