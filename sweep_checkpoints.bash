#!/usr/bin/env bash
# Loop over every FSDP checkpoint of an actor run, convert each to HF format,
# serve it with sglang (NLA env, via run_NLA.bash), and score it with
# eval_combo_hypothesis.py (nla_client env).
#
# Prereqs:
#   - micromamba installed and the /work "miles" env buildable via setup_env.sh
#     (if micromamba is missing: yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh) && source ~/.bashrc)
#     -- used for checkpoint conversion (convert_fsdp_to_hf.py) and for serving
#     non-Qwen models (e.g. Gemma-3) via run_NLA.bash
#   - conda env "NLA" present (sglang server for Qwen models, driven via run_NLA.bash)
#   - conda env "nla_client" present (used only as a python interpreter, not activated)
#
# Usage:
#   bash sweep_checkpoints.bash
#   bash sweep_checkpoints.bash --force                 # reconvert + rescore everything
#   bash sweep_checkpoints.bash --iters 100,200,330      # only these iterations
#   bash sweep_checkpoints.bash --iters 10,330 --force   # redo just iter 10 and 330
#   bash sweep_checkpoints.bash --base --iters 0         # score the untrained origin
#                                                         # actor too, as pseudo-"iter 0"
#                                                         # (no conversion, uses actor_hf/
#                                                         # directly); combine with --iters
#                                                         # to also pull in real checkpoints,
#                                                         # e.g. --base --iters 0,10,330
#   bash sweep_checkpoints.bash --every 3                # only 1 of every 3 checkpoints,
#                                                         # by sorted position (1st, 4th,
#                                                         # 7th, ...) -- stable as new ones
#                                                         # land, so combine with --run for
#                                                         # a lighter watch_checkpoints.bash
#   bash sweep_checkpoints.bash --run analogy_combo_taboo_rl_alpha05_v1 \
#     --label alpha05taboo --suffix _v1                  # score a different training run.
#                                                         # ALWAYS pass --label/--suffix when
#                                                         # --run points at a run that isn't
#                                                         # the default -- the results/ dir is
#                                                         # named from label+suffix only (no
#                                                         # run name in it), so two runs at the
#                                                         # same alpha with the default label
#                                                         # would overwrite each other's output
#                                                         # at matching iteration numbers.
#   ORIGIN_HF_DIR/EVAL_PARQUET/TABOO_EVAL_PARQUET env vars: override for a run trained
#   from a different base model -- the defaults below are Qwen2.5-7B-specific. E.g. for
#   a Gemma-3-12B run (different origin dir + different diff datasets, since the diff
#   vectors are model-specific):
#     RUN_NAME=analogy_combo_taboo_rl_alpha1_wordonly_gemma12 \
#     ORIGIN_HF_DIR=$NLA_WORK_DIR/gemma3_12b_actor_hf \
#     EVAL_PARQUET=$NLA_WORK_DIR/diff_analogy_combo_gemma12_posA_split/diff_base_eval.parquet \
#     TABOO_EVAL_PARQUET=$NLA_WORK_DIR/diff_taboo_gemma12_posA/diff_base_eval.parquet \
#     bash sweep_checkpoints.bash --label gemma12wordonly --suffix _v1
#   If the sglang server (run_NLA.bash) OOMs during "Capture ... prefill CUDA graph" --
#   Gemma-3's head_dim=256 leaves less headroom than the narrower head_dims run_NLA.bash's
#   default was tuned against -- add MEM_FRACTION_STATIC=0.7 (or lower) and/or
#   DISABLE_PREFILL_CUDA_GRAPH=1 to the env vars above; both pass through to run_NLA.bash.
#   bash sweep_checkpoints.bash --concurrency 32          # more requests in flight per
#                                                         # checkpoint (sglang continuous-
#                                                         # batches server-side); default 8.
#                                                         # Raise this first if the GPU sits
#                                                         # idle during scoring.
#   bash sweep_checkpoints.bash --combo-assistant         # also score combo's assistant-tag
#                                                         # position (chat_template:assistant_
#                                                         # im_start), the half --content-only
#                                                         # normally filters out -- written to
#                                                         # a separate combo_assistant_eval_
#                                                         # scored_* file, off by default since
#                                                         # it roughly doubles combo's per-
#                                                         # checkpoint GPU time.
#
# Run this directly with `bash` (don't source it) — it sources setup_env.sh
# itself to pick up the miles env for conversion, and shells out to the NLA
# and nla_client conda envs for serving/scoring.

set -euo pipefail
set -m   # job control on, so the backgrounded server gets its own process group we can kill cleanly

# ---- config ----
PROJECT_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_NAME="${RUN_NAME:-analogy_combo_rl_alpha05_v3_LONG}"
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
ACTOR_CKPT_DIR="$RUN_DIR/actor"
# Untrained origin actor HF dir -- must match whatever ACTOR_SFT_CKPT/INSTRUCT_MODEL
# the run was trained from (e.g. gemma3_12b_actor_hf for a Gemma run). Also feeds
# convert_fsdp_to_hf.py's --origin-hf-dir for every non-N=0 checkpoint, so it must
# be right even when --base is off.
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-$PROJECT_DIR/actor_hf}"
EVAL_PARQUET="${EVAL_PARQUET:-$PROJECT_DIR/diff_analogy_combo_v3/diff_base_eval.parquet}"
# Taboo held-out eval organisms (diff_base_eval.parquet from extract_diff_base.py's
# --output-dir split, step 2 of the combo+taboo mix plan) -- scored alongside combo's
# eval set below, both against the same already-served checkpoint. Set empty to skip
# (e.g. for a combo-only run that never built diff_taboo_v2/). Override via env var
# for a run built against a different taboo split (e.g. diff_taboo_v3/).
TABOO_EVAL_PARQUET="${TABOO_EVAL_PARQUET:-$PROJECT_DIR/diff_taboo_v2/diff_base_eval.parquet}"
# --all-positions counterparts: full parquets with EVERY position_kind
# extracted for this eval split -- 16 for combo (analogy_combo content +
# 3 system/3 user/3 assistant chat-template scaffold positions + 6
# system_content tokens for "You are a helpful assistant."), 19 for taboo
# (same 9 chat-template + 6 system_content, plus 4 prompt_fraction in place
# of combo's single content position) -- instead of the default content-only
# / 2-position subset. diff_base_eval_complete.parquet is the merged result
# of diff_base_eval_all_positions.parquet/diff_base_eval_allpos.parquet
# (chat-template + fractions) plus diff_base_eval_system_content.parquet /
# the system_content backfill (see merge_diff_positions.py, filter_diff_
# positions.py, backfill_taboo_v3_missing_positions.bash) -- both merges are
# a one-time thing per eval split, not per training run, so any run that
# reuses diff_analogy_combo_v3/diff_taboo_v3's eval split (just a different
# RL alpha, say) gets full position coverage for free via --all-positions-*,
# no per-run backfill needed.
#
# COMBO side only scores 4 of diff_base_eval_complete.parquet's 16
# position_kinds -- analogy_combo (content) + the 3 assistant_* chat-template
# tags -- via diff_base_eval_varying_positions.parquet, a filter_diff_
# positions.py subset of it. The other 12 (system_im_start/role_name/newline,
# user_im_start/role_name/newline, system_content:you/are/a/helpful/
# assistant/.) all sit BEFORE the analogy word in every combo's prompt, so
# causal attention makes them bit-identical across every one of the 588 eval
# combos (verified: every row of a given one of those position_kinds in
# diff_base_eval_complete.parquet is the exact same vector) -- combo =
# vec(B1)-vec(A1)+vec(A2) at such a position is always v-v+v = v regardless
# of which combo, so scoring them per-combo was just re-asking the actor the
# same 12 questions 588x each for no extra signal. (extract_diff_analogy_
# combo_multimodel.py's --constants-output captures these same 12 positions
# properly -- once per model, not once per combo -- for the standalone
# whole-dataset multimodel extraction; this eval-split file predates that
# and was never rebuilt to match, hence the filter here instead.) TABOO's
# side is unaffected -- its positions are organism-vs-base DIFFS, which
# genuinely differ per organism even where the underlying tokens are fixed,
# so no equivalent waste there.
# Separate output files from the default combo/taboo scoring above (own skip
# check), so a --all-positions pass never collides with -- or gets skipped
# because of -- an already-existing default-scored file for the same iter.
ALLPOS_EVAL_PARQUET="${ALLPOS_EVAL_PARQUET:-$PROJECT_DIR/diff_analogy_combo_v3/diff_base_eval_varying_positions.parquet}"
ALLPOS_TABOO_EVAL_PARQUET="${ALLPOS_TABOO_EVAL_PARQUET:-$PROJECT_DIR/diff_taboo_v3/diff_base_eval_complete.parquet}"
# ALPHA_LABEL/RESULT_SUFFIX together name the results/ dir per iter (see below) --
# plot_checkpoint_accuracy.py globs on this pattern, with no RUN_NAME in it, so
# two runs sharing a label+suffix (e.g. both default "alpha05"+"_LONG") will
# clobber each other's results at matching iteration numbers. Override both
# with --label/--suffix whenever you point --run at a different training run
# that reuses the same alpha value.
ALPHA_LABEL="${ALPHA_LABEL:-alpha05}"
RESULT_SUFFIX="${RESULT_SUFFIX:-_LONG}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_URL="http://localhost:$SGLANG_PORT"
NLA_CLIENT_PYTHON="${NLA_CLIENT_PYTHON:-python}"  # python of the nla_client env
# Concurrent requests fired at sglang per checkpoint (it continuous-batches
# server-side, so this is the main lever for wall-clock on an idle GPU).
CONCURRENCY="${CONCURRENCY:-8}"

FORCE=0
ONLY_ITERS=""
INCLUDE_BASE=0
EVERY=1
COMBO_ASSISTANT=0
ALL_POSITIONS_COMBO=0
ALL_POSITIONS_TABOO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --iters) ONLY_ITERS="$2"; shift 2 ;;
    --base) INCLUDE_BASE=1; shift ;;
    --run) RUN_NAME="$2"; RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"; ACTOR_CKPT_DIR="$RUN_DIR/actor"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --label) ALPHA_LABEL="$2"; shift 2 ;;
    --suffix) RESULT_SUFFIX="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --combo-assistant) COMBO_ASSISTANT=1; shift ;;
    # --all-positions turns both on; --all-positions-combo/--all-positions-taboo
    # toggle just one side (e.g. taboo all-positions while combo stays content-only).
    --all-positions) ALL_POSITIONS_COMBO=1; ALL_POSITIONS_TABOO=1; shift ;;
    --all-positions-combo) ALL_POSITIONS_COMBO=1; shift ;;
    --all-positions-taboo) ALL_POSITIONS_TABOO=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$PROJECT_DIR"

echo "=== activating miles env (for checkpoint conversion) ==="
set +u   # micromamba's own activation hooks aren't nounset-safe (e.g. NVCC_PREPEND_FLAGS)
source "$PROJECT_DIR/setup_env.sh"
set -u

SERVER_PID=""

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping sglang server (pgid $SERVER_PID)"
    kill -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # belt and suspenders: catch anything that escaped the process group
  pkill -f "sglang.launch_server.*--port $SGLANG_PORT" 2>/dev/null || true
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

positions_complete() {
  # True (exit 0) iff every (doc_id, position_kind) pair in $2 (a target
  # all-positions eval parquet) is already present in $1 (an existing scored
  # output, which may not exist yet). Used to skip a checkpoint under
  # --all-positions without needing to spin up its server at all, once
  # --resume has converged it to full coverage on a prior pass.
  local existing="$1" target="$2"
  [[ -f "$existing" ]] || return 1
  "$NLA_CLIENT_PYTHON" - "$existing" "$target" <<'PYEOF'
import sys
import pandas as pd
existing, target = sys.argv[1], sys.argv[2]
e = pd.read_parquet(existing, columns=["doc_id", "position_kind"])
t = pd.read_parquet(target, columns=["doc_id", "position_kind"])
have = set(map(tuple, e.itertuples(index=False)))
need = set(map(tuple, t.itertuples(index=False)))
sys.exit(0 if need <= have else 1)
PYEOF
}

wait_for_server() {
  local tries=180
  while (( tries > 0 )); do
    if curl -sf "$SGLANG_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "sglang server died while starting up" >&2
      return 1
    fi
    sleep 5
    tries=$((tries - 1))
  done
  echo "timed out waiting for sglang server" >&2
  return 1
}

# ---- collect checkpoint iterations ----
mapfile -t CKPT_DIRS < <(find "$ACTOR_CKPT_DIR" -maxdepth 1 -type d -name 'iter_*' | sort)

# Training writes latest_checkpointed_iteration.txt last, once an iteration's
# checkpoint is fully flushed -- use it as a completion fence so a checkpoint
# directory still being written (e.g. training is live) is never picked up
# mid-write. Absent the marker (e.g. run not using this convention), don't filter.
LATEST_MARKER="$ACTOR_CKPT_DIR/latest_checkpointed_iteration.txt"
if [[ -f "$LATEST_MARKER" ]]; then
  LATEST_COMPLETE="$(cat "$LATEST_MARKER")"
else
  LATEST_COMPLETE=""
fi

# N=0 is a pseudo-checkpoint standing in for the untrained origin actor
# (ORIGIN_HF_DIR) -- no FSDP->HF conversion needed for it.
NS=()
SRC_DIRS=()
if [[ "$INCLUDE_BASE" -eq 1 ]]; then
  NS+=(0)
  SRC_DIRS+=("")
fi
for i in "${!CKPT_DIRS[@]}"; do
  ckpt_dir="${CKPT_DIRS[$i]}"
  raw="$(basename "$ckpt_dir")"     # e.g. iter_0000330
  raw="${raw#iter_}"                # e.g. 0000330
  n=$((10#$raw))                    # e.g. 330

  # --every N keeps every Nth checkpoint by sorted position (index 0, N, 2N, ...).
  # Positional, not by iteration number, so it stays stable as new checkpoints
  # are appended on later watch passes -- earlier checkpoints never shift index.
  if (( i % EVERY != 0 )); then
    continue
  fi

  if [[ -n "$LATEST_COMPLETE" && "$n" -gt "$LATEST_COMPLETE" ]]; then
    echo "[$n] checkpoint dir exists but not yet complete (latest complete is $LATEST_COMPLETE) -- skipping for now"
    continue
  fi
  NS+=("$n")
  SRC_DIRS+=("$ckpt_dir")
done

if [[ -n "$ONLY_ITERS" ]]; then
  IFS=',' read -ra WANT <<< "$ONLY_ITERS"
fi

for idx in "${!NS[@]}"; do
  N="${NS[$idx]}"
  ckpt_dir="${SRC_DIRS[$idx]}"

  if [[ -n "$ONLY_ITERS" ]]; then
    skip=1
    for w in "${WANT[@]}"; do [[ "$w" == "$N" ]] && skip=0; done
    (( skip )) && continue
  fi

  if [[ "$N" -eq 0 ]]; then
    HF_DIR="$ORIGIN_HF_DIR"
  else
    HF_DIR="$RUN_DIR/actor_hf_iter${N}"
  fi
  RESULT_DIR="results/analogy_iter${N}_vs_base_${ALPHA_LABEL}${RESULT_SUFFIX}"
  OUT="$RESULT_DIR/combo_eval_scored_iter${N}${RESULT_SUFFIX}.parquet"
  TABOO_OUT="$RESULT_DIR/taboo_eval_scored_iter${N}${RESULT_SUFFIX}.parquet"
  COMBO_ASSISTANT_OUT="$RESULT_DIR/combo_assistant_eval_scored_iter${N}${RESULT_SUFFIX}.parquet"

  echo ""
  echo "=========================================="
  echo "iter $N"
  echo "=========================================="

  # --all-positions-combo / --all-positions-taboo: score that side against its
  # full all-positions parquet and --resume into the SAME default OUT/TABOO_OUT
  # path used by the light scoring below (not a separate file) --
  # eval_{combo,taboo}_hypothesis.py's --resume matches existing rows by
  # (doc_id, position_kind), so a file that already has e.g. content-only
  # combo or 25%/50% taboo positions scored just gets the missing positions
  # filled in and merged, never rescoring what's already there. A file with
  # nothing scored yet (or no file at all) gets everything scored in one pass
  # -- same code path. The two sides are independent switches.
  if [[ "$ALL_POSITIONS_COMBO" -eq 1 ]]; then
    ACTIVE_EVAL_PARQUET="$ALLPOS_EVAL_PARQUET"
  else
    ACTIVE_EVAL_PARQUET="$EVAL_PARQUET"
  fi
  if [[ "$ALL_POSITIONS_TABOO" -eq 1 ]]; then
    ACTIVE_TABOO_EVAL_PARQUET="$ALLPOS_TABOO_EVAL_PARQUET"
  else
    ACTIVE_TABOO_EVAL_PARQUET="$TABOO_EVAL_PARQUET"
  fi

  # Combo and taboo are tracked independently so a run that's missing only one
  # of the two (e.g. backfilling taboo onto an already-fully-combo-scored run)
  # doesn't waste a checkpoint reload+reconversion re-scoring the half that's
  # already done.
  DO_COMBO=1
  if [[ "$ALL_POSITIONS_COMBO" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 ]] && positions_complete "$OUT" "$ACTIVE_EVAL_PARQUET"; then
      echo "[$N] combo already fully scored across all positions ($OUT) -- skipping"
      DO_COMBO=0
    fi
  elif [[ -f "$OUT" && "$FORCE" -eq 0 ]]; then
    echo "[$N] combo already scored ($OUT) -- skipping combo"
    DO_COMBO=0
  fi

  DO_TABOO=1
  if [[ -z "$ACTIVE_TABOO_EVAL_PARQUET" ]]; then
    DO_TABOO=0
  elif [[ "$ALL_POSITIONS_TABOO" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 ]] && positions_complete "$TABOO_OUT" "$ACTIVE_TABOO_EVAL_PARQUET"; then
      echo "[$N] taboo already fully scored across all positions ($TABOO_OUT) -- skipping"
      DO_TABOO=0
    fi
  elif [[ -f "$TABOO_OUT" && "$FORCE" -eq 0 ]]; then
    echo "[$N] taboo already scored ($TABOO_OUT) -- skipping taboo"
    DO_TABOO=0
  fi

  # --combo-assistant is redundant once combo is doing all-positions (its one
  # position is already covered by the combo all-positions pass), so skip it then.
  DO_COMBO_ASSISTANT=1
  if [[ "$COMBO_ASSISTANT" -eq 0 || "$ALL_POSITIONS_COMBO" -eq 1 ]]; then
    DO_COMBO_ASSISTANT=0
  elif [[ -f "$COMBO_ASSISTANT_OUT" && "$FORCE" -eq 0 ]]; then
    echo "[$N] combo-assistant already scored ($COMBO_ASSISTANT_OUT) -- skipping"
    DO_COMBO_ASSISTANT=0
  fi

  if [[ "$DO_COMBO" -eq 0 && "$DO_TABOO" -eq 0 && "$DO_COMBO_ASSISTANT" -eq 0 ]]; then
    echo "[$N] already fully scored -- skipping"
    continue
  fi

  # ---- 1. convert FSDP checkpoint -> HF ----
  if [[ "$N" -eq 0 ]]; then
    echo "[$N] untrained origin actor -- using $HF_DIR directly, no conversion needed"
  elif [[ ! -d "$HF_DIR" || "$FORCE" -eq 1 ]]; then
    echo "[$N] converting $ckpt_dir -> $HF_DIR"
    python miles/tools/convert_fsdp_to_hf.py \
      --input-dir "$ckpt_dir" \
      --output-dir "$HF_DIR" \
      --origin-hf-dir "$ORIGIN_HF_DIR" \
      --force
  else
    echo "[$N] HF checkpoint already exists at $HF_DIR -- skipping conversion"
  fi

  # ---- 2. serve it with sglang via run_NLA.bash ----
  # Qwen models are served from the NLA conda env; anything else (e.g. Gemma-3)
  # from the micromamba miles env that setup_env.sh already activated above.
  # PYTHONNOUSERSITE: ~/.local/lib/python*/site-packages leaks into every conda
  # env by default (site.ENABLE_USER_SITE) -- a stray `pip install --user torch`
  # (or a plain `pip install torch` run outside any env) shadows the env's
  # own torch with a different physical build, breaking torchvision's op
  # registration (RuntimeError: operator torchvision::nms does not exist) even
  # when the version strings match. Block user-site entirely for this subprocess.
  if grep -qi 'qwen' "$HF_DIR/config.json"; then
    echo "[$N] launching sglang server (NLA env) on $HF_DIR"
    env -u LD_LIBRARY_PATH -u CUDA_HOME -u LIBRARY_PATH -u CPATH \
      PYTHONNOUSERSITE=1 \
      MODEL_PATH="$HF_DIR" \
      conda run --no-capture-output -n NLA bash "$PROJECT_DIR/run_NLA.bash" \
      > "$RUN_DIR/sglang_iter${N}.log" 2>&1 &
  else
    echo "[$N] launching sglang server (miles env: $CONDA_PREFIX) on $HF_DIR"
    PYTHONNOUSERSITE=1 \
      MODEL_PATH="$HF_DIR" \
      bash "$PROJECT_DIR/run_NLA.bash" \
      > "$RUN_DIR/sglang_iter${N}.log" 2>&1 &
  fi
  SERVER_PID=$!

  echo "[$N] waiting for server to come up (pid $SERVER_PID)"
  wait_for_server

  # ---- 3. score it (nla_client env, clean of the miles CUDA env vars) ----
  mkdir -p "$RESULT_DIR"
  if [[ "$DO_COMBO" -eq 1 ]]; then
    echo "[$N] scoring -> $OUT"
    RESUME_FLAGS=(); [[ "$ALL_POSITIONS_COMBO" -eq 1 ]] && RESUME_FLAGS=(--no-content-only --resume)
    env -u LD_LIBRARY_PATH -u CUDA_HOME -u LIBRARY_PATH -u CPATH \
      "$NLA_CLIENT_PYTHON" eval_combo_hypothesis.py \
        --actor "$HF_DIR" \
        --sglang-url "$SGLANG_URL" \
        --parquet "$ACTIVE_EVAL_PARQUET" \
        "${RESUME_FLAGS[@]}" \
        --temperature 0 \
        --concurrency "$CONCURRENCY" \
        --out "$OUT"
  fi

  # taboo held-out eval organisms, same served checkpoint, no extra conversion/serving cost
  if [[ "$DO_TABOO" -eq 1 ]]; then
    echo "[$N] scoring (taboo) -> $TABOO_OUT"
    RESUME_FLAGS=(); [[ "$ALL_POSITIONS_TABOO" -eq 1 ]] && RESUME_FLAGS=(--resume)
    env -u LD_LIBRARY_PATH -u CUDA_HOME -u LIBRARY_PATH -u CPATH \
      "$NLA_CLIENT_PYTHON" eval_taboo_hypothesis.py \
        --actor "$HF_DIR" \
        --sglang-url "$SGLANG_URL" \
        --parquet "$ACTIVE_TABOO_EVAL_PARQUET" \
        "${RESUME_FLAGS[@]}" \
        --temperature 0 \
        --concurrency "$CONCURRENCY" \
        --out "$TABOO_OUT"
  fi

  # combo's assistant-tag position (chat_template:assistant_im_start), the half
  # --content-only filters out of the default combo scoring above -- same served
  # checkpoint, separate output file so it never mixes with the content-word hit rate.
  # (skipped whenever --all-positions is on -- see DO_COMBO_ASSISTANT above)
  if [[ "$DO_COMBO_ASSISTANT" -eq 1 ]]; then
    echo "[$N] scoring (combo-assistant) -> $COMBO_ASSISTANT_OUT"
    env -u LD_LIBRARY_PATH -u CUDA_HOME -u LIBRARY_PATH -u CPATH \
      "$NLA_CLIENT_PYTHON" eval_combo_hypothesis.py \
        --actor "$HF_DIR" \
        --sglang-url "$SGLANG_URL" \
        --parquet "$EVAL_PARQUET" \
        --position-kind "chat_template:assistant_im_start" \
        --temperature 0 \
        --concurrency "$CONCURRENCY" \
        --out "$COMBO_ASSISTANT_OUT"
  fi

  # ---- 4. tear down before next checkpoint ----
  cleanup_server
  sleep 3
done

echo ""
echo "=== sweep complete ==="
