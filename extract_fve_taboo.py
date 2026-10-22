#!/usr/bin/env python3
"""
Pooled base-vs-finetune FVE extraction for the 20 Qwen taboo_<word> organisms
(EvilScript/Qwen2.5-7B-Instruct-taboo-<word>), same recipe as extract_fve.py:
each finetune scored on 'neutral' + its own domain (here, the single shared
'taboo_train' domain -- see prompt_set_fve.py's OWN_DOMAIN).

BASE_MODEL_ID matches extract_fve.py's (Qwen/Qwen2.5-7B-Instruct), so this
run's 'base' rows ARE comparable with extract_fve.py's -- verified by
comparing actual tensor values (not just file hashes, which differ because
the two repos shard the safetensors files at different boundaries) between
Qwen/Qwen2.5-7B-Instruct and unsloth/Qwen2.5-7B-Instruct via direct HTTP
range reads: model.layers.20.input_layernorm.weight and
model.layers.20.self_attn.q_proj.weight are bit-for-bit identical (max abs
diff 0.0) between the two repos. unsloth's config-level differences
(unsloth_fixed flag, pad_token_id, sliding_window) are tokenizer/config
normalizations, not weight changes -- so there's no LoRA-merge-onto-wrong-
base concern here the way there is for the Gemma taboo adapters (see
extract_diff_taboo_gemma.py's note on that).

Kept as a standalone script (not folded into extract_fve.py's MODELS dict)
purely for organization: 20 more entries plus a domain (taboo_train) that
none of the other 5 organisms use.

'base' here only needs 'neutral' + 'taboo_train' -- there's no other domain
relevant to this run (unlike extract_fve.py's base, which spans all 5 other
organisms' own-domains).

Run in the nla_client env (transformers/peft, GPU).

Usage:
    python extract_fve_taboo.py --out results/fve_pooled_taboo/activations.parquet
    # optionally restrict to a subset while testing:
    python extract_fve_taboo.py --words ship,moon --out results/fve_pooled_taboo/subset.parquet
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set_fve import PROMPTS, SYSTEM_PROMPT, OWN_DOMAIN, _TABOO_WORDS  # noqa: E402

LAYER = 20
D_MODEL = 3584
# Same repo extract_fve.py uses -- confirmed weight-identical to the unsloth
# repack (see module docstring), so no adapter-base mismatch concern here.
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MODELS = {"base": BASE_MODEL_ID}
for _word in _TABOO_WORDS:
    MODELS[f"taboo_{_word}"] = f"EvilScript/Qwen2.5-7B-Instruct-taboo-{_word}"
del _word


def content_span(tok, ids_list):
    """Same heuristic as extract_fve.py/extract_all.py: user content sits
    between the last '<|im_start|>user' block and the following '<|im_end|>'."""
    toks = tok.convert_ids_to_tokens(ids_list)
    try:
        user_marker = max(i for i, t in enumerate(toks) if t == "user")
    except ValueError:
        return 1, len(ids_list) - 1
    start = user_marker + 2
    end = next((i for i in range(start, len(toks)) if toks[i] == "<|im_end|>"),
               len(toks))
    return start, end


def extract_model(model_id, prompts, base_for_adapter=BASE_MODEL_ID):
    from huggingface_hub import list_repo_files
    files = list_repo_files(model_id)
    is_adapter = "adapter_config.json" in files

    tok = AutoTokenizer.from_pretrained(base_for_adapter if is_adapter else model_id)

    if is_adapter:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_for_adapter, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True,
        )
        model = PeftModel.from_pretrained(base, model_id)
        model = model.merge_and_unload()
        print("  (LoRA adapter, merged onto base)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True,
        )
    model.eval()

    rows = []
    for pid, (domain, text) in enumerate(prompts):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=True,
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
        ids_list = ids[0].tolist()

        with torch.no_grad():
            out = model(input_ids=ids)
        hs = out.hidden_states[LAYER][0]

        s, e = content_span(tok, ids_list)
        toks = tok.convert_ids_to_tokens(ids_list)
        span_len = e - s
        if span_len < 1:
            print(f"  [!] prompt {pid}: empty content span, skipping")
            continue

        fractions = [0.25, 0.50, 0.75, 1.00]
        for frac in fractions:
            offset = min(int(round(frac * span_len)) - 1, span_len - 1)
            offset = max(offset, 0)
            idx = s + offset
            vec = hs[idx].float().cpu().numpy()
            assert vec.shape[0] == D_MODEL
            rows.append({
                "prompt_id": pid, "domain": domain, "prompt": text,
                "model": model_id, "label": None,
                "position": frac,
                "token": toks[idx],
                "activation_vector": vec.tolist(),
            })
    if is_adapter:
        # merge_and_unload() returns the SAME underlying model object `base`
        # pointed at (just unwrapped from the LoRA wrapper) -- `del model`
        # alone leaves `base` as a live reference to those same tensors
        # until this function returns, so drop it explicitly too. Matters
        # more here than in extract_fve.py: 21 sequential 7B loads (vs. 6)
        # gives a slow per-iteration leak much more room to compound into
        # an OOM, especially with less headroom on the GPU (see run_fve_
        # taboo.bash's note on not running the actor server during this step).
        del base
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--words", default=None,
                   help="comma list of taboo words to run (default: all 20), "
                        "e.g. --words ship,moon for a quick smoke test")
    args = p.parse_args()

    models = dict(MODELS)
    if args.words:
        keep = {f"taboo_{w.strip()}" for w in args.words.split(",")}
        models = {"base": BASE_MODEL_ID,
                  **{k: v for k, v in models.items() if k in keep}}

    # every finetune here shares the same own-domain (taboo_train), and base
    # only needs neutral + taboo_train -- no other domain is relevant to
    # this run, unlike extract_fve.py's base (which spans 5 own-domains).
    run_domains = {"neutral", "taboo_train"}
    shared_prompts = [(d, t) for (d, t) in PROMPTS if d in run_domains]

    # resume support: a whole model (label) is the unit of work here -- if
    # --out already has a label's rows, skip re-loading/re-extracting that
    # 7B model entirely. Each of the 21 loads takes real wall-clock time
    # (download + load + 80 forward passes), so this matters a lot more
    # here than per-row would for a fast operation.
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
                  f"({len(shared_prompts)} prompts) ...")
            rows = extract_model(model_id, shared_prompts)
            for r in rows:
                r["label"] = label
            all_rows.extend(rows)
            save()  # persist after every model, not just at the end
    finally:
        save()

    df = pd.DataFrame(all_rows)
    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(df.groupby(["label", "domain"]).size().to_string())


if __name__ == "__main__":
    main()
