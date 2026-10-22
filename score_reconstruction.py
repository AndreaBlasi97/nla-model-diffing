# #!/usr/bin/env python3
# """
# Score each verbalization against its ORIGINAL activation vector using the NLA
# critic (AR / reconstructor). Runs the critic locally on GPU.

# Run in the env that has CUDA torch (your NLA conda env), NOT nla_client.

# The input parquet must have, per row:
#   - activation_vector : the ORIGINAL extracted vector (list of floats)
#   - explanation       : the actor's verbalization text for that vector
#   - label             : model tag (base / coder / humanlike)

# Usage:
#     python score_reconstruction.py \
#         --critic ./critic_hf \
#         --parquet vectors_with_expl.parquet \
#         --device cuda:0
# """

# import argparse
# import sys
# import numpy as np
# import pandas as pd

# # NLACritic lives in the repo's single-file module
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from nla_inference import NLACritic  # noqa: E402


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--critic", required=True, help="path to critic_hf (AR checkpoint)")
#     p.add_argument("--parquet", required=True, help="parquet with activation_vector + explanation + label")
#     p.add_argument("--device", default="cuda:0")
#     p.add_argument("--out", default="exp_scored.parquet",
#                    help="output parquet: input rows + recon_cos + recon_mse columns")
#     args = p.parse_args()
#     print('it starts')

#     df = pd.read_parquet(args.parquet)
#     for col in ("activation_vector", "explanation", "label"):
#         if col not in df.columns:
#             sys.exit(f"parquet missing required column: {col!r}. Columns present: {list(df.columns)}")

#     critic = NLACritic(args.critic, device=args.device)

#     print(f"\n{'label':<12} {'cos':>7} {'mse':>7}   verbalization (truncated)")
#     print("-" * 80)
#     recon_cos = []
#     recon_mse = []
#     for _, row in df.iterrows():
#         expl = row["explanation"]
#         if not isinstance(expl, str) or not expl.strip():
#             print(f"{row['label']:<12} {'--':>7} {'--':>7}   (no verbalization text)")
#             recon_cos.append(np.nan)
#             recon_mse.append(np.nan)
#             continue
#         vec = np.asarray(row["activation_vector"], dtype=np.float32)
#         mse, cos = critic.score(expl, vec)
#         recon_cos.append(cos)
#         recon_mse.append(mse)
#         snippet = " ".join(expl.split())[:50]
#         print(f"{row['label']:<12} {cos:>7.3f} {mse:>7.3f}   {snippet}...")

#     # attach scores back to every row and save the full table
#     df["recon_cos"] = recon_cos
#     df["recon_mse"] = recon_mse
#     df.to_parquet(args.out, index=False)
#     print(f"\nSaved scored table -> {args.out}  ({len(df)} rows)")

#     print("-" * 80)
#     valid = df.dropna(subset=["recon_cos"])
#     if len(valid):
#         group_cols = ["label"] + (["domain"] if "domain" in valid.columns else [])
#         summ = valid.groupby(group_cols)["recon_cos"].agg(["mean", "std", "count"])
#         print("\nReconstruction fidelity (higher cos = better):")
#         print(summ.to_string())

#         # also save the summary as csv for the record
#         summ.to_csv(args.out.replace(".parquet", "_summary.csv"))
#         print(f"\nSaved summary -> {args.out.replace('.parquet', '_summary.csv')}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Score verbalizations with the NLA critic and report BOTH:
  - per-row reconstruction cosine + direction-MSE (as before)
  - per-group FVE (fraction of variance explained), matching the repo's
    `fve_nrm` metric, on vectors L2-normalized to mse_scale.

FVE uses a SHARED baseline_rawvar computed once from the BASE model's gold
vectors over the whole set, so base/coder/humanlike are scored on the same
yardstick (see docstring notes). FVE = 1 - MSE(pred,gold)/baseline_rawvar,
both sides normalized to mse_scale; 1=perfect, 0=predict-the-mean, <0=worse.

Run in the NLA env (GPU critic).

Usage:
    python score_reconstruction_fve.py \
        --critic ./critic_hf \
        --parquet vectors_with_expl.parquet \
        --device cuda:0 \
        --out exp_scored.parquet
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nla_inference import NLACritic  # noqa: E402


def normalize_activation(x: torch.Tensor, mse_scale: float | None) -> torch.Tensor:
    """L2-normalize each row to length `mse_scale`. None -> leave raw.
    Matches the repo's normalize_activation used by the FVE metric."""
    if mse_scale is None:
        return x
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x / norm * mse_scale


def fve(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float | None,
        baseline_rawvar: float | None = None) -> float:
    """Fraction of variance explained (repo's fve_nrm definition).
    pred/gold: [N, d]. Uses raw-mean denominator on normalized golds."""
    pred_n = normalize_activation(pred.float(), mse_scale)
    gold_n = normalize_activation(gold.float(), mse_scale)
    mse_model = ((pred_n - gold_n) ** 2).mean().item()
    if baseline_rawvar is None:
        mu = gold_n.mean(dim=0, keepdim=True)   # do NOT normalize mu
        baseline_rawvar = ((gold_n - mu) ** 2).mean().item()
    return 1.0 - mse_model / baseline_rawvar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="exp_scored.parquet")
    args = p.parse_args()

    # --- hardcoded config (instead of command-line args) ---
    # import types
    # args = types.SimpleNamespace(
    #     critic="./critic_hf",
    #     parquet="results/coder_human/vectors_with_expl.parquet",
    #     device="cuda:0",
    #     out="exp_scored.parquet",
    # )
    # # --- end config ---

    df = pd.read_parquet(args.parquet)
    for col in ("activation_vector", "explanation", "label"):
        if col not in df.columns:
            sys.exit(f"parquet missing required column: {col!r}. Have: {list(df.columns)}")

    critic = NLACritic(args.critic, device=args.device)
    mse_scale = critic.mse_scale
    print(f"[FVE] using mse_scale = {mse_scale:.4f}")

    # ---- pass 1: reconstruct every row, keep pred + gold ----
    preds, golds, recon_cos, recon_mse, valid = [], [], [], [], []
    for _, row in df.iterrows():
        expl = row["explanation"]
        gold = np.asarray(row["activation_vector"], dtype=np.float32)
        if not isinstance(expl, str) or not expl.strip():
            preds.append(np.full_like(gold, np.nan))
            golds.append(gold)
            recon_cos.append(np.nan); recon_mse.append(np.nan); valid.append(False)
            continue
        pred = critic.reconstruct(expl).numpy().astype(np.float32)
        mse, cos = critic.score(expl, gold)   # reuse critic's own (normalized) score
        preds.append(pred); golds.append(gold)
        recon_cos.append(cos); recon_mse.append(mse); valid.append(True)

    df["recon_cos"] = recon_cos
    df["recon_mse"] = recon_mse
    df = df.reset_index(drop=True)   # positional index aligns with the arrays below
    preds = np.stack(preds); golds = np.stack(golds); valid = np.array(valid)

    # ---- shared baseline_rawvar from BASE model's valid NEUTRAL golds ----
    # Restricted to domain=='neutral' (not every domain base happens to have
    # been extracted on in THIS run) so the denominator is the same fixed
    # reference across every run that shares the neutral prompt set --
    # otherwise a run that extracts base on 6 domains vs. one that extracts
    # it on 2 get different baseline_rawvar values purely from which extra
    # domains got pooled in, making FVE across runs look comparable when
    # it wasn't. Falls back to every domain if 'domain' is missing or this
    # run has no neutral base rows (e.g. a hand-built parquet without one).
    if "domain" in df.columns:
        base_mask = (df["label"].values == "base") & (df["domain"].values == "neutral") & valid
    else:
        base_mask = (df["label"].values == "base") & valid
    if base_mask.sum() < 2:
        print("[FVE] WARNING: <2 valid base/neutral rows — falling back to all-domain base baseline.")
        base_mask = (df["label"].values == "base") & valid
    if base_mask.sum() < 2:
        print("[FVE] WARNING: <2 valid base rows at all — falling back to per-group baseline.")
        shared_baseline = None
    else:
        gold_base = normalize_activation(torch.tensor(golds[base_mask]), mse_scale)
        mu = gold_base.mean(dim=0, keepdim=True)
        shared_baseline = ((gold_base - mu) ** 2).mean().item()
        print(f"[FVE] shared baseline_rawvar (from {base_mask.sum()} base/neutral golds) = {shared_baseline:.4f}")

    def compute_fve_table(group_cols, restrict_base_to_neutral):
        """One FVE row per group (as defined by group_cols). If
        restrict_base_to_neutral, the 'base' group's rows are narrowed to
        domain=='neutral' before computing its own n/mean_cos/FVE -- used for
        the pooled-by-label table (base's group otherwise means "however
        many domains THIS run's base was extracted on", not a fixed
        reference). Not needed for the by-domain table: there, 'base' is
        already split into (base, neutral), (base, taboo_train), etc., each
        its own group -- no separate group to restrict."""
        print(f"\n=== per-group metrics ({'/'.join(group_cols)}) ===")
        print(f"{'group':<26} {'n':>4} {'mean_cos':>9} {'FVE':>8}")
        print("-" * 52)
        rows = []
        for keys, g in df.groupby(group_cols):
            idx = g.index.to_numpy()      # positional row indices for this group
            # groupby(["label"]) (a length-1 LIST) yields tuple keys like
            # ("base",) in this pandas version, not the bare string "base" --
            # comparing `keys == "base"` directly silently never matches.
            # Same tuple-vs-string inconsistency the keystr line below
            # already works around; normalize the same way here.
            group_key = keys if isinstance(keys, str) else keys[0]
            if restrict_base_to_neutral and group_key == "base" and "domain" in df.columns:
                neutral_idx = g.index[g["domain"].values == "neutral"].to_numpy()
                if len(neutral_idx) >= 2:
                    idx = neutral_idx
                else:
                    print("[FVE] WARNING: 'base' has <2 neutral rows — reporting "
                          "its full pooled rows instead (not directly comparable "
                          "across runs).")
            gv = valid[idx]               # which of them are valid
            if gv.sum() < 2:
                fve_val = np.nan
            else:
                fve_val = fve(torch.tensor(preds[idx][gv]),
                              torch.tensor(golds[idx][gv]),
                              mse_scale, baseline_rawvar=shared_baseline)
            mean_cos = np.nanmean(df.loc[idx, "recon_cos"].values)
            keystr = keys if isinstance(keys, str) else "/".join(map(str, keys))
            n = int(gv.sum())
            print(f"{keystr:<26} {n:>4} {mean_cos:>9.3f} {fve_val:>8.3f}")
            rows.append({"group": keystr, "n": n, "mean_cos": mean_cos, "fve": fve_val})
        return rows

    # ---- pooled-by-label FVE (base restricted to neutral -- see helper docstring) ----
    fve_rows = compute_fve_table(["label"], restrict_base_to_neutral=True)

    df.to_parquet(args.out, index=False)
    pd.DataFrame(fve_rows).to_csv(args.out.replace(".parquet", "_fve.csv"), index=False)
    print(f"\nSaved scored table -> {args.out}")

    # ---- per-domain FVE, if the parquet has a domain column ----
    if "domain" in df.columns:
        fve_by_domain_rows = compute_fve_table(["label", "domain"], restrict_base_to_neutral=False)
        by_domain_path = args.out.replace(".parquet", "_fve_by_domain.csv")
        pd.DataFrame(fve_by_domain_rows).to_csv(by_domain_path, index=False)
        print(f"Saved per-domain FVE -> {by_domain_path}")
    print(f"Saved FVE summary  -> {args.out.replace('.parquet', '_fve.csv')}")
    print("\nFVE reading: 1=perfect, 0=no better than predicting the mean, <0=worse. "
          "Same baseline across groups → directly comparable.")


if __name__ == "__main__":
    main()
