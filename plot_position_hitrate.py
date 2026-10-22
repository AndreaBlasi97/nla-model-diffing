#!/usr/bin/env python3
"""Bar chart of overall (all-topic) analogy-combo hit rate per diffed token
position, one bar per model per position.

For each model and each diffed position (Content word / assistant
im_start / newline / role_name), the value plotted is:
    1. per (subject, target word): % of that target word's triplets where
       it appeared in any of the 4 verbalizations, for that position
    2. averaged across target words -> per (subject, position)
       (matches the xlsx's per-subject AVERAGE row, one column at a time)
    3. averaged across subjects/topics, unweighted -> per (position)
       (collapses the topic dimension: every topic counts equally)

So each bar is "the overall average across all topics" for one
(model, position) pair -- topics are no longer broken out on the x-axis,
positions are.

Usage:
    python plot_position_hitrate.py
    python plot_position_hitrate.py --out position_hitrate_by_model.png
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_export import save_figure

# Times-like serif text; embed TrueType (Type 42) fonts in PDF/PS output --
# Type 3 fonts are not allowed.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Requested by hand. NOTE (dataviz-skill validator, scripts/validate_palette.js):
# #FEDA8E (pale yellow) and #CCBB44 (mustard) only score ΔE 12.7 apart under
# normal color vision -- below the 15 floor where even full-color readers
# start to confuse them -- and all 4 are pale enough to sit under 3:1
# contrast against white. Kept as specified; if the Gemma-3-12b vs Llama
# bars read as confusable once printed, swap one of them out.
MODEL_COLORS = {
    "Qwen": "#4A7BB7",
    "Gemma12": "#FEDA8E",
    "Gemma27": "#F67E4B",
    "Llama": "#CCBB44",
}
MODEL_ORDER = ["Qwen", "Gemma12", "Gemma27", "Llama"]

MODEL_LABELS = {
    "Qwen": "Qwen2.5-7B-Instruct",
    "Gemma12": "Gemma-3-12b-it",
    "Gemma27": "Gemma-3-27b-it",
    "Llama": "Llama-3.3-70B-Instruct",
}

MODEL_PARQUET = {
    "Qwen": "diff_analogy_combo_qwen/combo_full_scored.parquet",
    "Gemma12": "diff_analogy_combo_gemma12/combo_full_scored.parquet",
    "Gemma27": "diff_analogy_combo_gemma27/combo_full_scored.parquet",
    "Llama": "diff_analogy_combo_llama70b/combo_full_scored.parquet",
}

POSITION_LABELS = {
    "analogy_combo": "Probe word",
    "chat_template:assistant_im_start": "<start_of_turn>",
    "chat_template:assistant_newline": "Newline",
    "chat_template:assistant_role_name": "role_name",
}
POSITION_ORDER = list(POSITION_LABELS)


def position_hit_rates(parquet_path):
    """Return {position_kind: overall_hit_rate_pct}, averaged unweighted
    across topics (each subject counts equally regardless of size)."""
    df = pd.read_parquet(parquet_path, columns=["subject", "expected_word", "position_kind", "hit"])
    per_word = df.groupby(["subject", "expected_word", "position_kind"])["hit"].mean() * 100.0
    per_subject_position = per_word.groupby(["subject", "position_kind"]).mean()
    overall = per_subject_position.groupby("position_kind").mean()
    return overall.to_dict()


def load_data(base_dir):
    data = {}
    for model in MODEL_ORDER:
        path = os.path.join(base_dir, MODEL_PARQUET[model])
        data[model] = position_hit_rates(path) if os.path.exists(path) else {}
    return data


# Final printed size: 3.3 in wide, no text below 9 pt. The figure is drawn
# at that exact size (no bbox_inches="tight", which would change the width)
# so the font sizes below are the printed sizes.
FIG_WIDTH_IN = 3.3
FIG_HEIGHT_IN = 2.9
FONT_PT = 9


def plot(data, out_path):
    positions = [POSITION_LABELS[p] for p in POSITION_ORDER]
    n_pos = len(positions)
    n_models = len(MODEL_ORDER)

    group_width = 0.8
    bar_width = group_width / n_models

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), layout="constrained")
    x = np.arange(n_pos)

    for m_idx, model in enumerate(MODEL_ORDER):
        offset = (m_idx - (n_models - 1) / 2) * bar_width
        heights = [data[model].get(p, np.nan) for p in POSITION_ORDER]
        ax.bar(
            x + offset, heights, width=bar_width * 0.92,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model] + (" (pending)" if not data[model] else ""),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=20, ha="right", rotation_mode="anchor", fontsize=FONT_PT)
    ax.tick_params(axis="y", labelsize=FONT_PT)
    ax.set_ylabel("Hit rate (%)", fontsize=FONT_PT + 1)
    ax.set_xlabel("Diffed token position", fontsize=FONT_PT + 1)
    # ax.set_title(
    #    "Overall triplet hit rate by token position and model\n(averaged across all topics)",
    #     fontsize=18,
    # )
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Matplotlib fills legend columns top-to-bottom; reorder so the 2-column
    # legend reads left-to-right in MODEL_ORDER (Qwen, Gemma12 / Gemma27, Llama).
    handles, labels = ax.get_legend_handles_labels()
    n_cols = 2
    order = [i for c in range(n_cols) for i in range(c, len(handles), n_cols)]
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="outside upper center", ncol=n_cols, frameon=False, fontsize=FONT_PT,
        handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
    )
    save_figure(fig, out_path, dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="position_hitrate_by_model.png")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_data(base_dir)
    plot(data, args.out)


if __name__ == "__main__":
    main()
