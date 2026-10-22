#!/usr/bin/env bash
# Pooled base-vs-finetune FVE pipeline for the 20 taboo_<word> organisms,
# for ONE of the three non-Qwen base models (gemma-3-12b-it, gemma-3-27b-it,
# Llama-3.3-70B-Instruct). Qwen has its own dedicated run_fve_taboo.bash.
# See extract_fve_taboo_multimodel.py's docstring.
#
# Only ONE model at a time -- run this three times total (once per
# --model-name), never concurrently: each actor server reserves 70% of the
# GPU (--mem-fraction-static 0.7), so two can't be resident together, and
# this is a single-GPU box. Each model gets its own fixed port (gemma-3-12b-
# it=40001, Llama-3.3-70B-Instruct=40002, gemma-3-27b-it=40003; qwen=40000
# in run_fve_taboo.bash) so the printed launch command is always unambiguous
# and won't collide with a stale server left running on another model's
# port -- it does NOT mean these can run concurrently.
#
# PREREQUISITES:
#   1. HF_TOKEN exported -- gemma-3/Llama-3.3 base repos and the unsloth
#      repacks are gated.
#   2. This model's CRITIC checkpoint downloaded locally already (script
#      checks and tells you the exact command if it's missing -- it does
#      NOT auto-download, since these range 24GB (12B) to ~140GB (70B)).
#   3. Same OOM lesson as the Qwen taboo run: do NOT start the actor server
#      before extraction (step 1). This script pauses and tells you when.
#      Extra-important for Llama-3.3-70B: a bf16 copy alone is ~140GB, i.e.
#      most of this GPU by itself -- nothing else can be resident while it
#      loads/merges.
#
# Usage:
#   bash run_fve_taboo_multimodel.bash gemma-3-12b-it
#   bash run_fve_taboo_multimodel.bash gemma-3-27b-it
#   bash run_fve_taboo_multimodel.bash Llama-3.3-70B-Instruct
#   # optional custom output dir as 2nd arg:
#   bash run_fve_taboo_multimodel.bash gemma-3-12b-it results/fve_pooled_taboo_gemma12

set -euo pipefail

NLA_WORK_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir holding checkpoints, data, runs/
MODEL_NAME="${1:?Usage: bash run_fve_taboo_multimodel.bash <model-name> [results_path]}"

case "$MODEL_NAME" in
  gemma-3-12b-it)
    ACTOR="$NLA_WORK_DIR/gemma3_12b_actor_hf"
    CRITIC="$NLA_WORK_DIR/gemma3_12b_critic_hf"
    CRITIC_REPO="kitft/nla-gemma3-12b-L32-ar"
    DEFAULT_RESULTS="results/fve_pooled_taboo_gemma12"
    SGLANG_PORT=40001
    ;;
  gemma-3-27b-it)
    ACTOR="$NLA_WORK_DIR/gemma3_27b_actor_hf"
    CRITIC="$NLA_WORK_DIR/gemma3_27b_critic_hf"
    CRITIC_REPO="kitft/nla-gemma3-27b-L41-ar"
    DEFAULT_RESULTS="results/fve_pooled_taboo_gemma27"
    SGLANG_PORT=40003
    ;;
  Llama-3.3-70B-Instruct)
    ACTOR="$NLA_WORK_DIR/llama70b_actor_hf"
    CRITIC="$NLA_WORK_DIR/llama70b_critic_hf"
    CRITIC_REPO="kitft/Llama-3.3-70B-NLA-L53-ar"
    DEFAULT_RESULTS="results/fve_pooled_taboo_llama70b"
    SGLANG_PORT=40002
    ;;
  *)
    echo "Unknown model-name: $MODEL_NAME (expected gemma-3-12b-it, gemma-3-27b-it, or Llama-3.3-70B-Instruct)"
    exit 1
    ;;
esac

RESULTS_PATH="${2:-$DEFAULT_RESULTS}"
SGLANG_URL="http://localhost:${SGLANG_PORT}"
CLIENT_ENV="nla_client"
SCORE_ENV="NLA"
NOISE_N_ROWS=10                                 # distinct neutral prompts to anchor the noise band to
NOISE_N=15

# ---- critic checkpoint check (no auto-download -- sizes vary a lot) ----
if [ ! -d "$CRITIC" ] || [ ! -f "$CRITIC/nla_meta.yaml" ]; then
  echo "Critic checkpoint not found at $CRITIC"
  echo "Download it first (size depends on model -- 12B ~24GB, 27B ~54GB, 70B ~140GB):"
  echo "  huggingface-cli download $CRITIC_REPO --local-dir $CRITIC"
  exit 1
fi

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$RESULTS_PATH"
echo "Model: $MODEL_NAME"
echo "Results will go to: $RESULTS_PATH"

VECS="$RESULTS_PATH/activations.parquet"
EXPL="$RESULTS_PATH/verbalizations.parquet"
SCORED="$RESULTS_PATH/reconstructions_scored.parquet"
NOISE_TEXTS="$RESULTS_PATH/noise_texts.parquet"
NOISE_BAND="$RESULTS_PATH/noise_band.txt"

# ---- 1. extract activations (client env; needs GPU for forward passes) ----
# Deliberately BEFORE the actor server starts -- see prerequisites above.
echo "=== [1/5] extracting activations (base + 20 taboo organisms, $MODEL_NAME) ==="
conda activate "$CLIENT_ENV"
python extract_fve_taboo_multimodel.py --model-name "$MODEL_NAME" --out "$VECS"

echo ""
echo "Extraction done. Now start the actor server in a SEPARATE terminal:"
echo "  conda activate NLA && python -m sglang.launch_server \\"
echo "    --model-path $ACTOR --port ${SGLANG_URL##*:} --disable-radix-cache \\"
echo "    --mem-fraction-static 0.7 --trust-remote-code"
read -r -p "Press Enter once the server is up and healthy (curl ${SGLANG_URL}/health)... "

# ---- 2. verbalize every vector via the actor server (client env) ----
echo "=== [2/5] verbalizing ==="
python verbalize_to_parquet.py \
    --actor "$ACTOR" --sglang-url "$SGLANG_URL" \
    --parquet "$VECS" --out "$EXPL"

# ---- 3. noise band: verbalize N fixed base/neutral vectors, N samples each
# (client env). Anchored to base/neutral (not an arbitrary row index) so the
# noise band is on the same footing across every run -- see noise_band.py's
# module docstring.
echo "=== [3/5] noise band (verbalize) ==="
python noise_band.py verbalize \
    --actor "$ACTOR" --sglang-url "$SGLANG_URL" \
    --parquet "$VECS" --n-rows "$NOISE_N_ROWS" --n "$NOISE_N" --out "$NOISE_TEXTS"

# ---- 4. score reconstructions + noise band scoring (score env; GPU critic) ----
echo "=== [4/5] scoring (critic) ==="
conda activate "$SCORE_ENV"
python noise_band.py score \
    --critic "$CRITIC" --parquet "$NOISE_TEXTS" \
    --device cuda:0 --save "$NOISE_BAND"

python score_reconstruction.py \
    --critic "$CRITIC" --parquet "$EXPL" \
    --device cuda:0 --out "$SCORED"

# ---- 5. aggregate (any env with pandas) ----
echo "=== [5/5] aggregating ==="
NOISE_STD="$(grep '^overall std cos' "$NOISE_BAND" | head -1 | awk '{print $NF}')"
echo "using noise std = $NOISE_STD"
python aggregate.py --parquet "$SCORED" --noise-std "$NOISE_STD" \
    | tee "$RESULTS_PATH/final_report.txt"

echo ""
echo "DONE. Outputs in $RESULTS_PATH/"
echo "Per-organism FVE (pooled over neutral + taboo_train) is in:"
echo "  ${SCORED%.parquet}_fve.csv"
ls -la "$RESULTS_PATH"
