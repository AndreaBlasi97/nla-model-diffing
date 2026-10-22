export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include/python3.12:$CPATH

# override with e.g. MODEL_PATH=/path/to/checkpoint bash run_NLA.bash
NLA_WORK_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir holding checkpoints, data, runs/
MODEL_PATH="${MODEL_PATH:-$NLA_WORK_DIR/actor_hf}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
# Wide head_dim (e.g. 256, Gemma-3) leaves less headroom for the prefill CUDA
# graph workspace than the narrower head_dims (e.g. Qwen2.5) this default was
# tuned against -- drop MEM_FRACTION_STATIC (0.7 or lower) and/or set
# DISABLE_PREFILL_CUDA_GRAPH=1 if the server OOMs during "Capture ... prefill
# CUDA graph" on such a model.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
EXTRA_SGLANG_ARGS=()
if [[ "${DISABLE_PREFILL_CUDA_GRAPH:-0}" == "1" ]]; then
  EXTRA_SGLANG_ARGS+=(--cuda-graph-backend-prefill=disabled)
fi

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$SGLANG_PORT" \
    --disable-radix-cache \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --trust-remote-code \
    "${EXTRA_SGLANG_ARGS[@]}"