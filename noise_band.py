#!/usr/bin/env python3
"""
Measure the NOISE BAND: how much does the reconstruction score move just from
the actor's sampling (temperature 1.0), with the vector held fixed?

This is the number every between-model/between-run difference must beat to
be real -- and to be COMPARABLE across runs (same reasoning as score_
reconstruction.py's 'base' now being restricted to neutral-domain rows),
it's anchored to fixed reference rows: base model, neutral domain, same as
that fix. Anchoring to an arbitrary numeric --row (the old behavior) meant
the noise band measured a different, run-specific vector every time --
different model, different domain, different absolute difficulty -- so two
runs' noise bands were never really the same yardstick even though both
were called "the noise band."

Anchors to ONE random position each from up to --n-rows DISTINCT prompts
where label=='base' and domain=='neutral' (both columns must be present;
falls back to --row for a manual override, or errors out with a suggestion
to use --row if no base/neutral rows exist). Deliberately not "the first
--n-rows rows in file order": rows are written prompt-by-prompt (4
consecutive positions each), so that would just grab one or two prompts'
full position sweep instead of a spread across different sentence content.
--seed makes the random position choice reproducible across repeated
invocations (default 0). Default --n-rows is 10, the neutral domain's full
prompt count, so by default every neutral prompt contributes exactly once.

Each anchored row is verbalized --n times; noise is computed PER ROW (each
row's own std across its samples), then combined as the root-mean-square of
those per-row stds -- not a naive pool of every sample from every row, which
would conflate real between-row differences (some neutral prompts may just
reconstruct a bit better or worse than others) with the within-row sampling
jitter this is actually trying to isolate.

Two-stage because actor (nla_client env) and critic (NLA/GPU env) live in
different envs -- same split as the main pipeline:

STAGE 1 (nla_client env): verbalize each anchored vector N times, save texts.
    python noise_band.py verbalize \
        --actor ./actor_hf \
        --sglang-url http://localhost:30000 \
        --parquet exp_vectors.parquet --n-rows 10 --n 15 --out noise_texts.parquet

STAGE 2 (NLA env): score all samples against their own original vectors.
    python noise_band.py score \
        --critic ./critic_hf \
        --parquet noise_texts.parquet --device cuda:0
"""

import argparse
import sys

import numpy as np
import pandas as pd
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _select_anchor_rows(df, args):
    """One row per distinct neutral prompt (up to --n-rows prompts), each at
    a RANDOMLY chosen position -- not the first N rows in raw file order,
    which (since rows are written prompt-by-prompt, 4 consecutive positions
    each) would just grab one or two prompts' full position sweep instead of
    a spread across different sentence content. --seed makes the random
    position choice reproducible across repeated invocations."""
    if args.row is not None:
        if args.row >= len(df):
            sys.exit(f"--row {args.row} out of range (parquet has {len(df)} rows)")
        return df.iloc[[args.row]]

    if "label" not in df.columns or "domain" not in df.columns:
        sys.exit("parquet is missing 'label' and/or 'domain' columns, so it "
                  "can't be auto-anchored to base/neutral rows -- pass --row "
                  "to pick one manually.")

    candidates = df[(df["label"] == "base") & (df["domain"] == "neutral")]
    if len(candidates) == 0:
        sys.exit("No rows with label=='base' and domain=='neutral' found -- "
                  "pass --row to pick a row manually instead.")

    if "prompt_id" not in candidates.columns:
        print("[!] parquet missing 'prompt_id' -- can't spread across distinct "
              "prompts, falling back to the first --n-rows rows as-is.")
        return candidates.iloc[:args.n_rows]

    rng = np.random.default_rng(args.seed)
    prompt_ids = sorted(candidates["prompt_id"].unique())
    if len(prompt_ids) > args.n_rows:
        prompt_ids = list(rng.choice(prompt_ids, size=args.n_rows, replace=False))
    elif len(prompt_ids) < args.n_rows:
        print(f"[!] only {len(prompt_ids)} distinct neutral prompts available "
              f"(wanted {args.n_rows})")

    picked = []
    for pid in prompt_ids:
        group = candidates[candidates["prompt_id"] == pid]
        picked.append(group.iloc[rng.integers(0, len(group))])
    return pd.DataFrame(picked)


def verbalize(args):
    from nla_inference import NLAClient
    df = pd.read_parquet(args.parquet)
    anchor_rows = _select_anchor_rows(df, args)
    print(f"Anchoring noise band to {len(anchor_rows)} row(s): "
          f"{[(r.get('label', '?'), r.get('domain', '?')) for _, r in anchor_rows.iterrows()]}")

    client = NLAClient(args.actor, sglang_url=args.sglang_url)

    out_rows = []
    for src_row, (_, row) in enumerate(anchor_rows.iterrows()):
        vec = np.asarray(row["activation_vector"], dtype=np.float32)
        for i in range(args.n):
            t = client.generate(vec, max_new_tokens=400)
            print(f"[row {src_row+1}/{len(anchor_rows)} sample {i+1}/{args.n}] "
                  f"{' '.join(str(t).split())[:70]}...")
            out_rows.append({
                "activation_vector": row["activation_vector"],
                "explanation": t,
                "sample": i,
                "src_row": src_row,
                "src_label": row.get("label", "?"),
                "src_domain": row.get("domain", "?"),
                "src_token": row.get("token", "?"),
            })

    out = pd.DataFrame(out_rows)
    out.to_parquet(args.out, index=False)
    print(f"\nWrote {len(out)} samples ({len(anchor_rows)} row(s) x {args.n} each) -> {args.out}")


def score(args):
    from nla_inference import NLACritic
    df = pd.read_parquet(args.parquet)
    critic = NLACritic(args.critic, device=args.device)

    coss = []
    for _, r in df.iterrows():
        vec = np.asarray(r["activation_vector"], dtype=np.float32)
        mse, cos = critic.score(r["explanation"], vec)  # verbalization + original vector -> reconstruction score
        coss.append(cos)
        tag = f"row {r['src_row']:>2}" if "src_row" in df.columns else ""
        print(f"{tag} sample {r['sample']:>2}: cos={cos:.3f}")
    df["cos"] = coss

    has_rows = "src_row" in df.columns
    if has_rows:
        per_row = df.groupby("src_row")["cos"].agg(["mean", "std", "count"])
        # RMS of each row's own std, NOT a naive pool of every sample across
        # every row -- see module docstring for why that would overstate
        # the noise by mixing in real between-row differences.
        overall_std = float(np.sqrt((per_row["std"] ** 2).mean()))
        overall_mean = float(df["cos"].mean())
    else:
        per_row = None
        overall_std = float(df["cos"].std())
        overall_mean = float(df["cos"].mean())

    band = 2 * overall_std
    with open(args.save, "w") as f:
        f.write("=== NOISE BAND (fixed base/neutral row(s), actor sampling only) ===\n")
        f.write(f"source parquet : {args.parquet}\n")
        f.write(f"n rows anchored: {df['src_row'].nunique() if has_rows else 1}\n")
        f.write(f"samples/row    : {df['sample'].nunique() if 'sample' in df.columns else len(df)}\n")
        f.write(f"overall mean cos : {overall_mean:.4f}\n")
        # kept on its own line with nothing trailing the number, and no OTHER
        # line in this file may contain the substring "std cos" -- run_fve*
        # .bash's `grep 'std cos' | awk '{print $NF}'` extracts this value to
        # feed aggregate.py's --noise-std, and grep would match every line
        # containing that substring, not just this one (this bit us once
        # already: an earlier explanatory sentence here also said "std cos",
        # which made grep return two lines and broke --noise-std with an
        # embedded newline).
        f.write(f"overall std cos  : {overall_std:.4f}\n")
        if has_rows:
            f.write("(this combines each row's own spread via root-mean-square -- see per-row stats below)\n")
        f.write("\n")
        f.write(f">>> NOISE BAND TO USE (2*std): {band:.4f}\n")
        f.write(f">>> pass this as --noise-std {overall_std:.4f} to aggregate.py\n")
        f.write("\n")
        if per_row is not None:
            f.write("per-row stats:\n")
            f.write(per_row.to_string())
            f.write("\n\n")
        f.write("per-sample cosines:\n")
        for _, r in df.iterrows():
            rowtag = f"row {int(r['src_row']):>2} " if has_rows else ""
            f.write(f"  {rowtag}sample {int(r['sample']):>2}: {r['cos']:.4f}\n")
    print(f"\nSaved summary + per-sample cosines -> {args.save}")

    print("\n--- NOISE BAND (fixed base/neutral row(s), sampling only) ---")
    print(f"  rows      = {df['src_row'].nunique() if has_rows else 1}")
    print(f"  mean cos  = {overall_mean:.3f}")
    print(f"  std cos   = {overall_std:.3f}"
          + (" (RMS of per-row std)" if has_rows else ""))
    print(f"\n  => A between-run cos difference smaller than ~{band:.3f} "
          f"(2 std) is NOT distinguishable from sampling noise.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verbalize")
    v.add_argument("--actor", required=True)
    v.add_argument("--sglang-url", required=True)
    v.add_argument("--parquet", required=True)
    v.add_argument("--n-rows", type=int, default=10,
                   help="how many distinct neutral prompts to anchor to, one "
                        "random position each (default 10 -- the neutral "
                        "domain's full prompt count)")
    v.add_argument("--seed", type=int, default=0,
                   help="random seed for position/prompt selection (default 0, "
                        "for reproducible anchor rows across repeated invocations)")
    v.add_argument("--row", type=int, default=None,
                   help="override: use exactly this one row index instead of "
                        "auto-selecting base/neutral rows")
    v.add_argument("--n", type=int, default=15, help="samples per anchored row")
    v.add_argument("--out", required=True)
    v.set_defaults(func=verbalize)

    s = sub.add_parser("score")
    s.add_argument("--critic", required=True)
    s.add_argument("--parquet", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--save", default="noise_band.txt", help="file to write summary + scores to")
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
