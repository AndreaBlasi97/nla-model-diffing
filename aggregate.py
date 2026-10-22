#!/usr/bin/env python3
"""
Aggregate the fully-scored experiment into the go/no-go picture.

Expects a parquet with columns:
  label, domain, prompt_id, position, cos_vs_base, recon_cos
(recon_cos added by scoring the verbalizations; cos_vs_base from extract_all.py)

Prints, per (label, domain):
  - mean vector cosine vs base   (how much the representation MOVED)
  - mean reconstruction cos      (how FAITHFULLY it verbalized)
and the finetune-minus-base gaps, next to the noise band you measured.

Usage:
    python aggregate.py --parquet exp_scored.parquet --noise-std 0.04
"""

import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--noise-std", type=float, required=True,
                    help="std of reconstruction cos from noise_band.py score")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    band = 2 * args.noise_std

    have_recon = "recon_cos" in df.columns
    have_vcos = "cos_vs_base" in df.columns

    print("=" * 64)
    print("REPRESENTATION MOVEMENT (vector cosine vs base; lower = moved more)")
    if have_vcos:
        piv = df[df.label != "base"].groupby(["label", "domain"])["cos_vs_base"].agg(["mean", "std", "count"])
        print(piv.to_string())
    else:
        print("  (no cos_vs_base column)")

    print("\n" + "=" * 64)
    print("VERBALIZATION FIDELITY (reconstruction cos; higher = better)")
    if have_recon:
        rec = df.groupby(["label", "domain"])["recon_cos"].agg(["mean", "std", "count"])
        print(rec.to_string())

        print("\n" + "-" * 64)
        print(f"FINETUNE vs BASE gap in reconstruction cos  (noise band = ±{band:.3f})")
        base_means = df[df.label == "base"].groupby("domain")["recon_cos"].mean()
        for label in df.label.unique():
            if label == "base":
                continue
            for domain in df.domain.unique():
                m = df[(df.label == label) & (df.domain == domain)]["recon_cos"].mean()
                if domain in base_means.index and not np.isnan(m):
                    gap = m - base_means[domain]
                    verdict = "SIGNAL" if abs(gap) > band else "within noise"
                    print(f"  {label:<10} {domain:<9} Δ={gap:+.3f}   {verdict}")
    else:
        print("  (no recon_cos column — run scoring first)")

    print("\n" + "=" * 64)
    print("READING THIS:")
    print("  - A Δ marked 'within noise' is not a result, no matter its sign.")
    print("  - Compare code vs neutral: a finetune effect should show up MORE")
    print("    in its own domain than in neutral controls.")
    print("  - Movement (vector cosine) and fidelity (recon cos) are different")
    print("    questions — a model can move a lot yet still verbalize cleanly.")


if __name__ == "__main__":
    main()
