#!/usr/bin/env bash
# Pooled base-vs-finetune FVE pipeline for the 20 Qwen taboo_<word>
# organisms, each scored on neutral + the shared 'taboo_train' domain
# (prompt_set_fve.py's OWN_DOMAIN). See extract_fve_taboo.py's docstring.
#
# base uses the same Qwen/Qwen2.5-7B-Instruct checkpoint as run_fve.bash
# (verified weight-identical to the unsloth repack the taboo adapters were
# trained against -- see extract_fve_taboo.py's docstring), so this run's
# 'base' rows are comparable with run_fve.bash's if ever needed. Output
# still goes to its own RESULTS_PATH since the domain set extracted differs
# (neutral + taboo_train only, not the full 6-domain grid).
#
# PREREQUISITES:
#   1. Do NOT start the actor server before step 1. --mem-fraction-static
#      0.7 reserves 70% of the GPU up front, whether or not it's serving
#      anything -- on a single-GPU box that leaves ~30% for 21 sequential
#      7B model loads/merges (base + 20 taboo adapters), which is tight
#      enough to OOM (confirmed: it did, on this exact run). Extraction
#      (step 1) never touches the actor -- only verbalize/noise-band
#      (steps 2-3) do. This script pauses right before step 2 and prompts
#      you to start the server then, so step 1 gets the full GPU.
#   2. When prompted, start it in the NLA env, in a SEPARATE terminal:
#        conda activate NLA && python -m sglang.launch_server \
#          --model-path .../actor_hf --port 40000 --disable-radix-cache \
#          --mem-fraction-static 0.7 --trust-remote-code
#      Leave it running for the rest of the script.
#   3. HF_TOKEN not required for Qwen/EvilScript repos (unlike the Gemma
#      taboo work), but double-check if that ever changes.
#
# Usage:
#   bash run_fve_taboo.bash results/fve_pooled_taboo
#   # smoke test on 2 words first:
#   #   python extract_fve_taboo.py --words ship,moon --out /tmp/smoke.parquet
#   # resuming after step 1 already succeeded: comment out step 1 below
#   # (same convention run_all.bash uses) and rerun.

set -euo pipefail

# ---- config ----
NLA_WORK_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir holding checkpoints, data, runs/
RESULTS_PATH="${1:-results/fve_pooled_taboo}"
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
# Deliberately BEFORE the actor server starts -- 21 sequential 7B loads/
# merges want the full GPU, not the ~30% left over once sglang reserves 70%.
echo "=== [1/5] extracting activations (base + 20 taboo organisms) ==="
conda activate "$CLIENT_ENV"
python extract_fve_taboo.py --out "$VECS"

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
