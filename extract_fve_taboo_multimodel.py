#!/usr/bin/env python3
"""
Pooled base-vs-finetune FVE extraction for the 20 taboo_<word> organisms,
generalized across all 4 base models an NLA actor exists for (Qwen2.5-7B-
Instruct, Gemma-3-12B-it, Gemma-3-27B-it, Llama-3.3-70B-Instruct) via
--model-name/_MODEL_PRESETS -- same preset table extract_diff_taboo_gemma.py
uses (layer/d_model/end_of_turn_token/attn_implementation, all sourced from
that model's own NLA actor's nla_meta.yaml extraction_layer_index/d_model).

Unlike extract_fve_taboo.py (Qwen-only, hand-rolled content_span/extraction
loop copied from extract_all.py), this reuses diff_verbalize.py's
load_causal_model (adapter-vs-full-model detection via local-cache-aware
_is_lora_adapter, base merge, attn_implementation/max_memory handling) and
extract_vectors (fractional content-span extraction, via content_span,
which already handles Gemma3's system-folded-into-user chat template quirk)
directly. Correctness matters more here than for Qwen-only: content_span's
literal-token role-marker heuristic and end-of-turn token both differ
across ChatML (Qwen) vs Gemma3 vs Llama-3's templates -- diff_verbalize.py's
functions are already tested against all of them, so no reason to
reimplement.

BASE_MODEL_ID per model is the unsloth repack (unsloth/<model_name>) --
matches what these EvilScript taboo LoRA adapters declare as their own
adapter_config.json base_model_name_or_path (same choice extract_diff_taboo_
gemma.py already makes). For Qwen this was independently verified weight-
identical to the official Qwen/Qwen2.5-7B-Instruct repo (see extract_fve_
taboo.py's docstring); not re-verified for Gemma/Llama here, but no need to
be -- we're using exactly what the adapters were trained against, not
substituting a different repo the way the Qwen-only script's BASE_MODEL_ID
choice did.

Gemma/Llama need HF_TOKEN exported (gated repos). Llama-3.3-70B is large
(~140GB in bf16) -- make sure NOTHING else (actor server included) is
resident on the GPU while running that model, same OOM lesson as the Qwen
taboo run (see run_fve_taboo_multimodel.bash).

Run in the nla_client env (transformers/peft, GPU).

Usage:
    python extract_fve_taboo_multimodel.py --model-name gemma-3-12b-it \\
        --out results/fve_pooled_taboo_gemma12/activations.parquet
    python extract_fve_taboo_multimodel.py --model-name gemma-3-27b-it \\
        --out results/fve_pooled_taboo_gemma27/activations.parquet
    python extract_fve_taboo_multimodel.py --model-name Llama-3.3-70B-Instruct \\
        --out results/fve_pooled_taboo_llama70b/activations.parquet
    # optionally restrict words while testing:
    python extract_fve_taboo_multimodel.py --model-name gemma-3-12b-it \\
        --words ship,moon --out /tmp/smoke.parquet
"""

import argparse
import gc
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set_fve import PROMPTS, SYSTEM_PROMPT, _TABOO_WORDS  # noqa: E402
from diff_verbalize import load_causal_model, extract_vectors  # noqa: E402

# Same preset table extract_diff_taboo_gemma.py uses -- layer/d_model are
# each model's own NLA actor's nla_meta.yaml extraction_layer_index/d_model.
_MODEL_PRESETS = {
    "Qwen2.5-7B-Instruct": {"layer": 20, "d_model": 3584,
                             "end_of_turn_token": "<|im_end|>", "attn_implementation": None},
    "gemma-3-12b-it": {"layer": 32, "d_model": 3840,
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "gemma-3-27b-it": {"layer": 41, "d_model": 5376,
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "Llama-3.3-70B-Instruct": {"layer": 53, "d_model": 8192,
                                "end_of_turn_token": "<|eot_id|>", "attn_implementation": None},
}

FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True, choices=sorted(_MODEL_PRESETS))
    p.add_argument("--out", required=True)
    p.add_argument("--words", default=None,
                   help="comma list of taboo words (default: all 20)")
    args = p.parse_args()

    preset = _MODEL_PRESETS[args.model_name]
    layer, d_model = preset["layer"], preset["d_model"]
    end_of_turn_token = preset["end_of_turn_token"]
    attn_implementation = preset["attn_implementation"]
    base_model_id = f"unsloth/{args.model_name}"

    words = _TABOO_WORDS if not args.words else [w.strip() for w in args.words.split(",")]

    run_domains = {"neutral", "taboo_train"}
    prompts = [(d, t) for (d, t) in PROMPTS if d in run_domains]

    models = {"base": base_model_id}
    for w in words:
        models[f"taboo_{w}"] = f"EvilScript/{args.model_name}-taboo-{w}"

    # cheap up front (tokenizer only, no GPU model load) -- computed
    # unconditionally so it's available even if 'base' gets skipped below on
    # a resumed run. In practice unused here (every model in this script is
    # already an -it/-Instruct checkpoint that ships its own chat_template;
    # this fallback only matters for non-instruct base checkpoints), kept
    # for parity with load_causal_model's intended usage.
    from transformers import AutoTokenizer
    base_template_probe = AutoTokenizer.from_pretrained(base_model_id).chat_template

    # resume support: a whole model (label) is the unit of work -- skip
    # re-loading/re-extracting a model whose rows are already in --out.
    # Matters even more here than for Qwen-only: Gemma/Llama downloads are
    # much larger, and Llama-70B's load/merge alone is a serious chunk of
    # wall-clock time per organism.
    all_rows = []
    done_labels = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        all_rows = prev.to_dict("records")
        done_labels = set(prev["label"].unique())
        print(f"Resuming: {len(done_labels)} labels already extracted in "
              f"{args.out}: {sorted(done_labels)}")

    def save():
        pd.DataFrame(all_rows).to_parquet(args.out, index=False)

    try:
        for label, model_id in models.items():
            if label in done_labels:
                continue
            print(f"Extracting {label} ({model_id}) on domains: {sorted(run_domains)} "
                  f"({len(prompts)} prompts) ...")
            model, tok = load_causal_model(
                model_id, base_model_id, chat_template_fallback=base_template_probe,
                attn_implementation=attn_implementation,
            )

            vecs, meta = extract_vectors(
                model, tok, layer, prompts, SYSTEM_PROMPT, FRACTIONS,
                end_of_turn_token=end_of_turn_token,
            )
            for (pid, frac), vec in vecs.items():
                m = meta[(pid, frac)]
                assert vec.shape[0] == d_model
                all_rows.append({
                    "prompt_id": pid, "domain": m["domain"], "prompt": m["prompt"],
                    "model": model_id, "label": label, "position": frac,
                    "token": m["token"], "activation_vector": vec.tolist(),
                })

            del model
            gc.collect()
            torch.cuda.empty_cache()
            save()  # persist after every model, not just at the end
    finally:
        save()

    df = pd.DataFrame(all_rows)
    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(df.groupby(["label", "domain"]).size().to_string())


if __name__ == "__main__":
    main()
