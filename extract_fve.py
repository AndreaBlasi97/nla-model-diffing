#!/usr/bin/env python3
"""
Pooled base-vs-finetune extraction for the "FVE per finetune, pooled over
neutral + its own domain" experiment. Adapted from extract_all.py: same
content_span / per-model extraction logic, generalized to all five finetunes
(coder, humanlike, financial, medical, sports) in one run, reading domains
from prompt_set_fve.py instead of prompt_set.py.

Unlike extract_all.py (which ran every model against every domain), each
finetune here is only extracted on 'neutral' + its OWN_DOMAIN entry (see
prompt_set_fve.py) -- score_reconstruction.py's per-label FVE grouping then
pools exactly those rows automatically, no scoring-side changes needed.
'base' is extracted on every domain since it's the shared reference for
every group's FVE baseline_rawvar.

Run in the nla_client env (transformers/peft, GPU). LoRA organisms
(financial/medical/sports) are detected automatically via adapter_config.json
and merged onto BASE_MODEL_ID before extraction.

Usage:
    python extract_fve.py --out results/fve_pooled/activations.parquet
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
from prompt_set_fve import PROMPTS, SYSTEM_PROMPT, OWN_DOMAIN  # noqa: E402

LAYER = 20
D_MODEL = 3584
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MODELS = {
    "base": BASE_MODEL_ID,
    "coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "humanlike": "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
    "financial": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "medical": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}


def content_span(tok, ids_list):
    """Return (start, end) indices covering the user sentence tokens only,
    excluding system prompt, role scaffolding, and the trailing
    <|im_end|>/assistant tail. Heuristic: the user content sits between the
    last '<|im_start|>user' block and the following '<|im_end|>'."""
    toks = tok.convert_ids_to_tokens(ids_list)
    try:
        user_marker = max(i for i, t in enumerate(toks) if t == "user")
    except ValueError:
        return 1, len(ids_list) - 1  # fallback: skip first/last
    start = user_marker + 2  # skip 'user' and the following 'Ċ'
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
        hs = out.hidden_states[LAYER][0]  # [seq, d]

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
        # pointed at -- `del model` alone leaves `base` as a live reference
        # to those same tensors until this function returns, so drop it
        # explicitly too (see extract_fve_taboo.py's note on this same bug).
        del base
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # resume support: a whole model (label) is the unit of work -- skip
    # re-loading/re-extracting a model whose rows are already in --out.
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
        for label, model_id in MODELS.items():
            if label in done_labels:
                continue
            if label == "base":
                prompts = PROMPTS  # every domain -- shared FVE reference
            else:
                own_domain = OWN_DOMAIN[label]
                prompts = [(d, t) for (d, t) in PROMPTS if d in ("neutral", own_domain)]
            domains = sorted(set(d for d, _ in prompts))
            print(f"Extracting {label} ({model_id}) on domains: {domains} "
                  f"({len(prompts)} prompts) ...")
            rows = extract_model(model_id, prompts)
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
