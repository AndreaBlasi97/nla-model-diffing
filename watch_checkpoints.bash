#!/usr/bin/env bash
# Continuously watch a live training run's checkpoint dir and score every new
# checkpoint as it lands, on a GPU separate from the training job. Just loops
# sweep_checkpoints.bash (which already skips already-scored iters and skips
# checkpoints still mid-write) with a sleep in between passes.
#
# Usage:
#   bash watch_checkpoints.bash --run analogy_combo_rl_alpha05_v3_LONG --gpu 4
#   bash watch_checkpoints.bash --run analogy_combo_rl_alpha05_v3_LONG --gpu 4 --interval 60 --port 30001
#
# Runs until you Ctrl-C it (there's no end-of-training marker to watch for),
# so just leave it running in a tmux/screen pane for the life of the training
# job and kill it once training finishes (a final `bash sweep_checkpoints.bash
# --run <name>` pass afterwards will pick up anything left).
#
# All extra args after recognized flags are forwarded to sweep_checkpoints.bash
# as-is (e.g. --base, --force).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_NAME=""
GPU=""
INTERVAL=120
PORT=30001
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "usage: bash watch_checkpoints.bash --run <run_name> --gpu <gpu_index> [--interval seconds] [--port sglang_port] [extra sweep_checkpoints.bash args]" >&2
  exit 1
fi
if [[ -z "$GPU" ]]; then
  echo "refusing to run without --gpu: pick a GPU index not in use by the training job (e.g. check nvidia-smi)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export RUN_NAME
export SGLANG_PORT="$PORT"

echo "=== watching runs/$RUN_NAME/actor for new checkpoints ==="
echo "    GPU: $GPU   sglang port: $PORT   poll interval: ${INTERVAL}s"

while true; do
  echo ""
  echo "### $(date -Is) -- scanning for new checkpoints ###"
  bash "$SCRIPT_DIR/sweep_checkpoints.bash" --run "$RUN_NAME" "${EXTRA_ARGS[@]}" || \
    echo "!!! sweep pass failed (see above) -- will retry next poll" >&2
  sleep "$INTERVAL"
done
