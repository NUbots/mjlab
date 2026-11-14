#!/usr/bin/env bash
# Wait for the dribble training tmux session to end, set its final policy
# aside, then start kick training. Meant to run in its own tmux session.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[$(date +%H:%M)] waiting for tmux session 'dribble' to finish"
while tmux has-session -t "=dribble" 2>/dev/null; do sleep 60; done

run_dir=$(ls -dt logs/rsl_rl/k1_dribble/*/ | head -1)
ckpt=$(ls -t "$run_dir"model_*.pt | head -1)
last_iter=$(grep -a "Learning iteration" train_k1_dribble.log | tail -1 | grep -o "[0-9]*/[0-9]*")
dest=policies/k1_dribble_v1
mkdir -p "$dest"
cp "$ckpt" "$dest/"
cp "$run_dir"*.onnx "$dest/" 2>/dev/null
cp -r "$run_dir"params "$dest/" 2>/dev/null
cp train_k1_dribble.log "$dest/train.log"
{
  echo "Booster K1 dribbling policy (JOGA-style), task Mjlab-Dribble-Flat-Booster-K1"
  echo "Saved: $(date)"
  echo "Source run: $run_dir"
  echo "Checkpoint: $(basename "$ckpt")"
  echo "Last logged iteration: $last_iter"
  echo "Play: uv run play Mjlab-Dribble-Flat-Booster-K1 --checkpoint-file $dest/$(basename "$ckpt") --viewer native"
} > "$dest/README.txt"
echo "[$(date +%H:%M)] dribble policy saved to $dest (last iteration $last_iter)"

echo "[$(date +%H:%M)] starting kick training"
tmux new-session -d -s kick "cd $(pwd) && uv run train Mjlab-Kick-Flat-Booster-K1 --env.scene.num-envs 8192 2>&1 | tee $(pwd)/train_k1_kick.log"
echo "[$(date +%H:%M)] kick training launched in tmux session 'kick'"
