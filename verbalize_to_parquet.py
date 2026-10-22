#!/usr/bin/env python3
"""
Verbalize each vector in a parquet via the running actor (AV) server and save
the resulting text back into the parquet as an 'explanation' column.

Resumable: if --out already exists, rows already verbalized there are reused
instead of re-generated, matched on (model, domain, prompt_id, position) --
NOT on row position/index, since a resumed run's input parquet is the same
file but iteration order and any --label override shouldn't matter for
identity. Progress is also saved to --out every --save-every newly-
verbalized rows (and on crash/interrupt via try/finally), so a run that dies
partway through only loses at most --save-every rows of actor-call work,
not the whole thing -- actor calls are the slow part here (LLM generation
per vector), so losing hundreds of them to one crash near the end is real
wasted GPU time worth avoiding.

Run in the nla_client env (transformers 4.x, the one where the client works).
The actor server must be running (the sglang server on --sglang-url).

Usage:
    python verbalize_to_parquet.py \
        --actor ./actor_hf \
        --sglang-url http://localhost:30000 \
        --parquet vectors.parquet \
        --out vectors_with_expl.parquet
    # interrupted or crashed partway through? just rerun the same command --
    # already-verbalized rows are skipped automatically.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nla_inference import NLAClient  # noqa: E402

# Stable per-row identity for resume matching. 'model' (the actual HF repo
# id), not 'label' -- --label can override the label column on write, which
# would otherwise make a resumed run's identity depend on whether --label
# was passed the same way both times.
KEY_COLS = ["model", "domain", "prompt_id", "position"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", required=True, help="path to actor_hf (AV checkpoint)")
    p.add_argument("--sglang-url", required=True)
    p.add_argument("--parquet", required=True, help="input parquet with activation_vector column")
    p.add_argument("--out", required=True, help="output parquet (adds 'explanation' column)")
    p.add_argument("--label", default=None,
                   help="tag every row with this value in a 'label' column "
                        "(overwrites if already present) — lets "
                        "score_reconstruction.py group/compare runs from "
                        "different checkpoints on the same eval set")
    p.add_argument("--save-every", type=int, default=20,
                   help="write --out to disk every N newly-verbalized rows "
                        "(always also saved at the end and on crash/interrupt)")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    if "activation_vector" not in df.columns:
        sys.exit(f"parquet missing 'activation_vector'. Columns: {list(df.columns)}")

    have_key_cols = all(c in df.columns for c in KEY_COLS)
    done = {}
    if have_key_cols and os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        if "explanation" in prev.columns and all(c in prev.columns for c in KEY_COLS):
            for _, r in prev.iterrows():
                expl = r["explanation"]
                if isinstance(expl, str) and expl.strip():
                    done[tuple(r[c] for c in KEY_COLS)] = expl
            print(f"Resuming: {len(done)} rows already verbalized in {args.out}")
    elif not have_key_cols:
        print("[!] parquet missing one of "
              f"{KEY_COLS} -- resume support disabled, will re-verbalize everything")

    client = NLAClient(args.actor, sglang_url=args.sglang_url)

    explanations = [None] * len(df)
    n_skipped = n_new = 0

    def save():
        out_df = df.copy()
        out_df["explanation"] = explanations
        if args.label is not None:
            out_df["label"] = args.label
        out_df.to_parquet(args.out, index=False)

    try:
        for i, (_, row) in enumerate(df.iterrows()):
            key = tuple(row[c] for c in KEY_COLS) if have_key_cols else None
            if key is not None and key in done:
                explanations[i] = done[key]
                n_skipped += 1
                continue
            vec = np.asarray(row["activation_vector"], dtype=np.float32)
            # NLAClient.generate returns the actor's text for one vector.
            # (signature verified against your nla_inference.py: generate(vector))
            text = client.generate(vec, max_new_tokens=400)
            label = row.get("label", "?")
            print(f"[{i+1}/{len(df)}] [{label}] -> {' '.join(str(text).split())[:80]}...")
            explanations[i] = text
            n_new += 1
            if n_new % args.save_every == 0:
                save()
    finally:
        # unfinished rows stay None here -- save() below persists everything
        # actually completed so far, whether we got here via normal
        # completion, an exception, or Ctrl-C.
        save()

    print(f"\nWrote {len(df)} rows ({n_skipped} resumed, {n_new} newly verbalized) -> {args.out}")


if __name__ == "__main__":
    main()
