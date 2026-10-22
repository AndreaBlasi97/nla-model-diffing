#!/usr/bin/env bash
# Pooled base-vs-finetune FVE pipeline: base + coder/humanlike/financial/
# medical/sports, each finetune scored on neutral + its own domain only
# (prompt_set_fve.py's OWN_DOMAIN). See extract_fve.py's docstring.
#
# PREREQUISITES (do these BEFORE running):
#   1. Start the actor (sglang) server in the NLA env, in a SEPARATE terminal:
#        conda activate NLA && python -m sglang.launch_server \
#          --model-path .../actor_hf --port 30000 --disable-radix-cache \
#          --mem-fraction-static 0.7 --trust-remote-code
#      Leave it running.
#   2. HF_TOKEN exported if any of the finetune repos are gated (none of the
#      current MODELS entries are, but check if you add one).
#
# Usage:
#   bash run_fve.bash results/fve_pooled

set -euo pipefail

# ---- config ----
NLA_WORK_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir holding checkpoints, data, runs/
RESULTS_PATH="${1:-results/fve_pooled}"
ACTOR="$NLA_WORK_DIR/actor_hf"
CRITIC="$NLA_WORK_DIR/critic_hf"
SGLANG_URL="http://localhost:40000"
CLIENT_ENV="nla_client"                         # env for extract + verbalize
SCORE_ENV="NLA"                                 # env with CUDA for the critic
NOISE_N_ROWS=10                                 # distinct neutral prompts to anchor the noise band to
NOISE_N=15                                      # samples per anchored row

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- setup ----
mkdir -p "$RESULTS_PATH"
echo "Results will go to: $RESULTS_PATH"

VECS="$RESULTS_PATH/activations.parquet"
EXPL="$RESULTS_PATH/verbalizations.parquet"
SCORED="$RESULTS_PATH/reconstructions_scored.parquet"
NOISE_TEXTS="$RESULTS_PATH/noise_texts.parquet"
NOISE_BAND="$RESULTS_PATH/noise_band.txt"

# ---- 1. extract activations (client env; needs GPU for forward passes) ----
echo "=== [1/5] extracting activations ==="
conda activate "$CLIENT_ENV"
python extract_fve.py --out "$VECS"

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
echo "Per-label FVE (pooled over neutral + own domain) is in:"
echo "  ${SCORED%.parquet}_fve.csv"
ls -la "$RESULTS_PATH"
