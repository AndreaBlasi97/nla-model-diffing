# NLA model diffing

Code for verbalizing **activation differences** between a base model and a
fine-tuned / prompted model with Natural Language Autoencoders (NLAs), and for
RL-training the NLA activation verbalizer (AV) to name the planted concept.

Built on top of [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
(training pipeline) and its standalone inference client (`nla_inference.py`, Apache-2.0, lightly modified here).

Two testbeds:

- **Taboo**: diff vectors between a base model and a model that is secretly thinking of a
  taboo word (moon, song, book, dance, clock, ship). Does the AV name the word?
- **Analogy triplets**: combo vectors `vec(B1) - vec(A1) + vec(A2)`. Does the AV name `B2`?

Models: Qwen2.5-7B, Gemma-3-12B/27B, Llama-3.3-70B (using the released NLA AV/AR pairs).

## Layout

| Stage | Files |
|---|---|
| Shared modules | `nla_inference.py` (AV/AR client), `diff_verbalize.py` (model loading, vector extraction), `prompt_set.py`, `prompt_set_fve.py`, `figure_export.py` |
| Extract diff vectors | `extract_diff_taboo_gemma.py` (taboo, all models), `extract_diff_base.py`, `extract_diff_analogy.py`, `extract_diff_analogy_combo.py`, `extract_diff_analogy_combo_multimodel.py` (triplets, all models) |
| Build RL training sets | `merge_combo_taboo.py`, `merge_diff_positions.py`, `filter_diff_positions.py` |
| Evaluate hypotheses | `eval_taboo_hypothesis.py`, `eval_combo_hypothesis.py` |
| FVE / cosine | `run_fve.bash`, `run_fve_taboo.bash`, `run_fve_taboo_multimodel.bash` → `extract_fve*.py`, `verbalize_to_parquet.py`, `score_reconstruction.py`, `noise_band.py`, `aggregate.py`, `plot_fve_diff_base.py` |
| RL training + checkpoint sweeps | reward changes live in the NLA fork (`nla/reward.py`); `sweep_checkpoints.bash`, `watch_checkpoints.bash`, `run_NLA.bash` |
| Plots | `plot_checkpoint_accuracy.py`, `plot_position_hitrate.py` |
| Environment | `setup_env.sh`, `patches/` (sglang / miles / NLA patches) |

Scripts are run from the repo root. Checkpoints, data, `runs/` and `results/` are expected
in `$NLA_WORK_DIR` (defaults to the repo root; all gitignored).

## Setup

1. Clone the modified NLA training repo (fork of kitft's, with the target-word reward,
   diff-explain datagen and env rebuild scripts) next to the scripts:

   ```bash
   git clone https://github.com/AndreaBlasi97/natural_language_autoencoders
   ```

   Python scripts look for it at `./natural_language_autoencoders` (override with `NLA_DIR`).
   The extraction / merge scripts import `nla.datagen` from it.

   `sweep_checkpoints.bash` also needs miles (for `miles/tools/convert_fsdp_to_hf.py`) at `./miles`:

   ```bash
   git clone https://github.com/AndreaBlasi97/miles-nla miles
   ```

2. Build the training env following the NLA repo, applying `patches/sglang_nla_all.patch`
   (base64 embedding transport for the AV server) and the miles patches. Then in each new shell:

   ```bash
   source setup_env.sh
   ```

3. Download the AV/AR checkpoints into `$NLA_WORK_DIR` (`actor_hf`, `critic_hf`,
   `gemma3_12b_actor_hf`, …) or pass HF repo ids directly (e.g. `kitft/Llama-3.3-70B-NLA-L53-av`).

The eval/verbalize scripts only need a light client env (`nla_client`); set
`NLA_CLIENT_PYTHON` to its interpreter for `sweep_checkpoints.bash`.

## Serving the AV (sglang)

```bash
export TRITON_CACHE_DIR=$PWD/.triton_cache

# Qwen
bash run_NLA.bash
# or
python -m sglang.launch_server --model-path actor_hf --port 40000 \
    --disable-radix-cache --mem-fraction-static 0.75 --trust-remote-code

# Gemma-3-12B / 27B
python -m sglang.launch_server --model-path gemma3_12b_actor_hf --port 40001 \
    --disable-radix-cache --mem-fraction-static 0.5 --trust-remote-code
python -m sglang.launch_server --model-path gemma3_27b_actor_hf --port 40003 \
    --disable-radix-cache --mem-fraction-static 0.5 --trust-remote-code

# Llama-3.3-70B
python -m sglang.launch_server --model-path kitft/Llama-3.3-70B-NLA-L53-av --port 40002 \
    --tp-size 4 --disable-radix-cache --mem-fraction-static 0.75 --trust-remote-code
```

## Extract diff vectors

```bash
# Taboo, at fractions of the response
python extract_diff_taboo_gemma.py \
  --domains neutral --positions fractions --fractions 0.75,1.00 \
  --model-name Llama-3.3-70B-Instruct \
  --output diff_taboo_llama70b/diff_base_neutral_frac75_100.parquet

# Taboo, at positions after generation
python extract_diff_taboo_gemma.py \
  --organisms taboo_ship --domains neutral \
  --positions post_generation --gen-positions 0,4,8,16 \
  --model-name gemma-3-12b-it \
  --output diff_taboo_gemma/diff_base_neutral_genpos.parquet
```

Analogy-triplet vectors: `extract_diff_analogy_combo_multimodel.py` (writes e.g.
`diff_analogy_combo_qwen/diff_base_full.parquet` and `..._constants.parquet`).

## Evaluate

```bash
# Taboo
python eval_taboo_hypothesis.py \
  --actor llama70b_actor_hf --sglang-url http://localhost:30002 \
  --parquet diff_taboo_llama70b/diff_base_neutral_frac75_100.parquet \
  --n-repeats 4 \
  --out results/taboo_llama70b_neutral_allpos/combo_eval_scored_neutral_frac75_100.parquet

# Triplets: 4 positions x every combo
$NLA_CLIENT_PYTHON eval_combo_hypothesis.py \
  --actor ./actor_hf --sglang-url http://localhost:30000 \
  --parquet diff_analogy_combo_qwen/diff_base_full.parquet \
  --out diff_analogy_combo_qwen/combo_full_scored.parquet \
  --no-content-only --concurrency 32 --resume --n-repeats 4

# Triplets: constant positions (12 rows, one actor call each)
$NLA_CLIENT_PYTHON eval_combo_hypothesis.py \
  --actor ./actor_hf --sglang-url http://localhost:30000 \
  --parquet diff_analogy_combo_qwen/diff_base_full_constants.parquet \
  --out diff_analogy_combo_qwen/combo_full_constants_scored.parquet \
  --no-content-only --resume --n-repeats 4
```

## FVE and cosine similarity

With the matching sglang server running:

```bash
bash run_fve.bash results/fve_pooled                       # Qwen
bash run_fve_taboo_multimodel.bash gemma-3-12b-it
bash run_fve_taboo_multimodel.bash Llama-3.3-70B-Instruct
```

## RL training

The fork's `nla/reward.py` adds a bonus when the AV names the target word, controlled by env vars:

| Variable | Meaning |
|---|---|
| `NLA_TARGET_WORD_BONUS_ALPHA` | weight of the target-word bonus |
| `NLA_TABOO_EXACT_WORD_ONLY` | only reward the exact taboo word |
| `NLA_TABOO_WRONG_WORD_PENALTY` | penalty for naming a different taboo word |
| `NLA_TABOO_WORDS` | comma-separated taboo word list |
| `NLA_VERBALIZATIONS_LOG` | optional log of sampled verbalizations |

RL training sets (`rl_train.parquet`) are built with `merge_combo_taboo.py`.

```bash
cd natural_language_autoencoders
export WANDB_API_KEY=...          # or: wandb login
export NLA_TARGET_WORD_BONUS_ALPHA=1
export NLA_TABOO_EXACT_WORD_ONLY=1
export NLA_TABOO_WRONG_WORD_PENALTY=0
export NLA_TABOO_WORDS=moon,song,book,dance,clock,ship
export NLA_VERBALIZATIONS_LOG=../runs/analogy_combo_alpha1/verbalizations.log
mkdir -p ../runs/analogy_combo_alpha1

# Qwen
RL_PARQUET=../diff_combo_taboo_mix_v3/rl_train.parquet \
INSTRUCT_MODEL=../actor_hf ACTOR_SFT_CKPT=../actor_hf CRITIC_SL_CKPT=../critic_hf \
RUN_DIR=../runs/analogy_combo_alpha1 \
KL_LOSS_COEF=0.005 ACTOR_GPUS=2 CRITIC_GPUS=1 ROLLOUT_GPUS=1 \
bash configs/rl.sh --num-epoch 3 \
    --use-wandb --wandb-project nla-analogy-diff --wandb-key "$WANDB_API_KEY" \
    --save-interval 50 --rollout-temperature 1.2

# Gemma-3-12B
RL_PARQUET=../diff_combo_taboo_mix_gemma12_posA/rl_train.parquet \
INSTRUCT_MODEL=../gemma3_12b_actor_hf ACTOR_SFT_CKPT=../gemma3_12b_actor_hf \
CRITIC_SL_CKPT=../gemma3_12b_critic_hf RUN_DIR=../runs/analogy_alpha1_gemma12 \
KL_LOSS_COEF=0.005 ACTOR_LR=1e-5 CRITIC_LR=1e-6 \
ACTOR_GPUS=2 CRITIC_GPUS=1 ROLLOUT_GPUS=1 \
bash configs/rl.sh --num-epoch 3 \
    --use-wandb --wandb-project nla-analogy-diff --wandb-key "$WANDB_API_KEY" \
    --save-interval 50 --rollout-temperature 1.2 \
    --rollout-max-response-len 250 --rollout-max-context-len 400 \
    --sglang-context-length 550
```

To resume a run, add `--load ../runs/<run>/actor --critic-load-dcp ../runs/<run>/critic`.

## Checkpoint sweeps

Converts each saved actor checkpoint to HF, serves it, and runs the taboo + triplet evals.

```bash
# Follow a training run as checkpoints appear
bash watch_checkpoints.bash --run analogy_combo_alpha1 --gpu 0 --port 30000 \
  --label alpha1 --base --all-positions-taboo --all-positions-combo

# Sweep chosen iterations
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 bash sweep_checkpoints.bash \
  --run analogy_combo_alpha1 --label alpha1 --suffix _v3 \
  --all-positions-combo --all-positions-taboo --base --iters 900

# Several sweeps in parallel: one GPU + port each
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30001 bash sweep_checkpoints.bash \
  --run <run> --label <label> --suffix _v1 \
  --all-positions-combo --all-positions-taboo --concurrency 32 \
  --base --iters 850,950,1050

# Gemma: point the sweep at the Gemma origin model and eval sets
RUN_NAME=<gemma run> \
ORIGIN_HF_DIR=$NLA_WORK_DIR/gemma3_12b_actor_hf \
EVAL_PARQUET=$NLA_WORK_DIR/diff_analogy_combo_gemma12_posA_split/diff_base_eval.parquet \
TABOO_EVAL_PARQUET=$NLA_WORK_DIR/diff_taboo_gemma12_posA/diff_base_eval.parquet \
ALLPOS_EVAL_PARQUET=$NLA_WORK_DIR/diff_analogy_combo_gemma12_posA_split/diff_base_eval_varying_positions.parquet \
ALLPOS_TABOO_EVAL_PARQUET=$NLA_WORK_DIR/diff_taboo_gemma12_posA/diff_base_eval_allpos.parquet \
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30001 bash sweep_checkpoints.bash \
  --all-positions --base --label gemma12wordonly --suffix _v1
```

## Plots

```bash
# Accuracy over training checkpoints
python plot_checkpoint_accuracy.py --label alpha1 --suffix _v3 --no-iter0 \
  --taboo-grid --combo-grid --run analogy_combo_alpha1

# Average two runs
python plot_checkpoint_accuracy.py --alpha 1 --suffix _v3_again --average-with alpha1_wordonly _LONG \
  --no-iter0 --taboo-grid --out accuracy_taboo_grid_alpha1_avg.png

# Triplet hit rate per diffed position, per model (reads diff_analogy_combo_*/combo_full_scored.parquet)
python plot_position_hitrate.py
```

## License

Apache-2.0 (see `LICENSE`). `nla_inference.py` and the NLA patches derive from
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders).
