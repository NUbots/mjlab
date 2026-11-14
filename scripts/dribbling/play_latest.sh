#!/usr/bin/env bash
# Open a viewer on the newest checkpoint of a skill: dribble (default) or kick.
# Extra args go to `play` (e.g. --viewer viser, --num-envs 9).
set -euo pipefail
cd "$(dirname "$0")/../.."
skill="dribble"
if [[ "${1:-}" == "dribble" || "${1:-}" == "kick" ]]; then skill="$1"; shift; fi
if [[ "$skill" == "kick" ]]; then task="Mjlab-Kick-Flat-Booster-K1"; else task="Mjlab-Dribble-Flat-Booster-K1"; fi
ckpt=$(ls -t logs/rsl_rl/k1_${skill}/*/model_*.pt | head -1)
echo "Playing $ckpt"
exec uv run play "$task" --checkpoint-file "$ckpt" --viewer native --num-envs 4 "$@"
