#!/usr/bin/env python3
"""Plot FVE per finetune against the base model, from NLAforfinetune.xlsx.

For each model sheet in the workbook, reads the small summary table in
columns M:P (label / FVE_avg / FVE_dif_base, with base_min and base_max
stored one column left of the other FVE_avg values -- an Excel layout
quirk) and draws:
  - a horizontal line at the base model's FVE_avg
  - a shaded band spanning base_min .. base_max FVE_avg
  - one bar per other finetune, running from the base line up (or down)
    by that row's FVE_dif_base, colored by sign

Usage:
    python plot_fve_diff_base.py
    python plot_fve_diff_base.py --xlsx NLAforfinetune.xlsx --sheet Qwen
    python plot_fve_diff_base.py --sort desc --out fve_diff_base.png
"""
import argparse
import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.patches import Patch

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

LEGEND_WRAP_WIDTH = 22  # chars per line -- keeps the side legend narrow

# The grid figure is drawn at its final printed size (two-column width, no
# bbox_inches="tight", which would change the width), so these font sizes
# are the printed sizes -- no text below 9 pt.
GRID_WIDTH_IN = 6.75
GRID_HEIGHT_IN = 3.5
FONT_PT = 9

# Orange/mustard pair, matching position_hitrate_by_model.png's palette.
# NOTE (dataviz-skill validator, scripts/validate_palette.js): these two sit
# in the 6-8 CVD floor band (deutan delta-E 6.0), legal only with a secondary
# encoding -- which this chart already has, since sign is also shown by bar
# direction (up/down from the base line), not color alone. Both also read
# under 3:1 contrast on the light chart surface, hence the white bar edges
# below. Neutral gray for the base_min/base_max band.
POSITIVE_COLOR = "#CCBB44"
NEGATIVE_COLOR = "#F67E4B"
BAND_COLOR = "#f0efec"
BASE_LINE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
BAR_NAME_COLOR = "#5e5e5e"  # dark gray for the bar names drawn inside the grid panels

# Grouping + display order for the bars, keyed by row label. Each group is
# either an explicit, ordered list of labels, or a label prefix (for the
# many taboo_* rows). Rows are re-ordered to match this, and a labeled
# bracket is drawn under each group. Labels that don't match any group
# (e.g. on a sheet with different finetune names) are appended at the end,
# unbracketed, in their original sheet order.
GROUPS = [
    ("Full Fine-Tuning", ["Coder", "humanlike"]),
    ("LoRA FT EM", ["financial", "medical", "sports"]),
    ("LoRA FT Taboo", "taboo_"),
]

# Display name shown as each subplot's title, keyed by sheet name. A sheet
# not listed here just falls back to its own name.
DISPLAY_NAME = {
    "Qwen": "Qwen2.5-7B-Instruct",
    "gemma12": "Gemma-3-12b-it",
    "gemma27": "Gemma-3-27b-it",
    "llama70": "Llama-3.3-70B-Instruct",
}


# Display name for a bar, keyed by sheet row label. Rows not listed here
# just drop any "taboo_" prefix; every label is then capitalized.
BAR_LABEL = {
    "humanlike": "Human",
    "financial": "Finance",
}


def load_summary(xlsx_path, sheet_name):
    """Return (base_avg, base_min, base_max, [(label, avg, diff), ...])."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]

    # Locate the "FVE_avg" header to find the column layout, rather than
    # hardcoding column letters.
    avg_col = None
    for row in ws.iter_rows():
        for cell in row:
            if cell.value == "FVE_avg":
                avg_col = cell.column
                break
        if avg_col is not None:
            break
    if avg_col is None:
        raise ValueError(f"Could not find 'FVE_avg' header in sheet {sheet_name!r}")

    label_col = avg_col - 2   # e.g. M when FVE_avg is in O
    band_col = avg_col - 1    # e.g. N -- holds base_min/base_max values
    diff_col = avg_col + 1    # e.g. P -- FVE_dif_base

    base_avg = base_min = base_max = None
    rows = []
    for r in range(1, ws.max_row + 1):
        label = ws.cell(row=r, column=label_col).value
        if label is None:
            continue
        avg_val = ws.cell(row=r, column=avg_col).value
        band_val = ws.cell(row=r, column=band_col).value
        diff_val = ws.cell(row=r, column=diff_col).value

        if label == "base_min":
            base_min = avg_val if avg_val is not None else band_val
        elif label == "base_max":
            base_max = avg_val if avg_val is not None else band_val
        elif label == "base":
            base_avg = avg_val
        else:
            value = avg_val if avg_val is not None else band_val
            if value is None:
                continue
            rows.append((str(label), value, diff_val))

    if base_avg is None:
        raise ValueError(f"Could not find 'base' row in sheet {sheet_name!r}")

    return base_avg, base_min, base_max, rows


def _sort_rows(rows, sort):
    if sort == "asc":
        return sorted(rows, key=lambda t: t[2])
    if sort == "desc":
        return sorted(rows, key=lambda t: -t[2])
    if sort == "abs":
        return sorted(rows, key=lambda t: -abs(t[2]))
    return rows


def group_and_order(rows, sort):
    """Re-order rows by GROUPS (sorting within each group), returning
    (ordered_rows, group_name_per_row) with unmatched rows appended,
    ungrouped, at the end."""
    used = set()
    ordered_rows = []
    group_of_row = []
    row_by_label = {r[0]: r for r in rows}

    for group_name, members in GROUPS:
        if isinstance(members, str):
            matched = [r for r in rows if r[0].startswith(members) and r[0] not in used]
        else:
            matched = [row_by_label[m] for m in members if m in row_by_label and m not in used]
        matched = _sort_rows(matched, sort)
        ordered_rows.extend(matched)
        group_of_row.extend([group_name] * len(matched))
        used.update(r[0] for r in matched)

    leftover = _sort_rows([r for r in rows if r[0] not in used], sort)
    ordered_rows.extend(leftover)
    group_of_row.extend([None] * len(leftover))
    return ordered_rows, group_of_row


def _bar_positions(group_of_row, gap=1.8):
    """x positions with a visual gap inserted between groups."""
    x = []
    cur = 0.0
    prev_group = None
    for i, g in enumerate(group_of_row):
        if i > 0 and g != prev_group:
            cur += gap
        x.append(cur)
        cur += 1.0
        prev_group = g
    return np.array(x)



def _plot_panel(ax, base_avg, base_min, base_max, rows, sheet_name, sort, ylim=None,
                 bar_width=0.7, base_label=None, font_pt=None):
    """Draw one model's chart on the given axes. Returns (handles, labels)
    for the base-line/band, for use in a shared legend. font_pt=None keeps
    the large on-screen sizes; pass a print size (e.g. FONT_PT) for a
    figure drawn at its real size."""
    rows, group_of_row = group_and_order(rows, sort)

    # Display only: grouping above still keys on the full sheet row label.
    labels = [BAR_LABEL.get(r[0], r[0].removeprefix("taboo_")).capitalize() for r in rows]
    diffs = [r[2] for r in rows]
    x = _bar_positions(group_of_row)

    if base_min is not None and base_max is not None:
        ax.axhspan(base_min, base_max, color=BAND_COLOR, zorder=0,
                    label="FVE min-max range of control model on all prompt domains")

    if base_label is None:
        base_label = f"base avg ({base_avg:.3f})"
    ax.axhline(base_avg, color=BASE_LINE_COLOR, linewidth=1.5 if font_pt is None else 0.8, zorder=2,
               label=base_label)

    colors = [POSITIVE_COLOR if d >= 0 else NEGATIVE_COLOR for d in diffs]
    ax.bar(x, diffs, bottom=base_avg, width=bar_width, color=colors,
           edgecolor="white", linewidth=0.4 if font_pt is None else 0.2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=font_pt or 19)
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylabel("FVE", fontsize=(font_pt + 1) if font_pt else 30)
    ax.set_title(DISPLAY_NAME.get(sheet_name, sheet_name), fontsize=(font_pt + 1) if font_pt else 30,
                 fontweight="bold" if font_pt else "normal")
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="y", labelsize=font_pt or 20)
    if font_pt:
        ax.tick_params(length=2, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8 if font_pt is None else 0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return ax.get_legend_handles_labels()


def plot_single(base_avg, base_min, base_max, rows, sheet_name, out_path, sort, ylim=None):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6.5))

    legend_handles, legend_labels = _plot_panel(
        ax, base_avg, base_min, base_max, rows, sheet_name, sort, ylim=ylim)
    display_name = DISPLAY_NAME.get(sheet_name, sheet_name)
    #ax.set_title(f"{display_name}: FVE per Finetune vs. IT models", fontsize=15)

    legend_handles += [Patch(color=POSITIVE_COLOR), Patch(color=NEGATIVE_COLOR)]
    legend_labels += ["FVE Finetune > FVE control model", "FVE Finetune < FVE control model"]
    legend_labels = [textwrap.fill(l, LEGEND_WRAP_WIDTH) for l in legend_labels]
    ax.legend(legend_handles, legend_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=18,
              labelspacing=1.8)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=150, bbox_inches="tight")


def plot_grid(sheets_data, out_path, sort, ylim=(0.1, 0.8)):
    """sheets_data: list of (sheet_name, base_avg, base_min, base_max, rows).

    2x2 grid at GRID_WIDTH_IN wide (spans both columns of a two-column
    paper). Within each row, panel widths are proportional to their number
    of bars (so Qwen's 25 bars get as much room per bar as Gemma's 20) and
    the y-axis is shared, with ticks/label on the left panel only. A 1x4
    row doesn't fit: a rotated 9 pt tick label needs ~0.13 in per bar."""
    fig = plt.figure(figsize=(GRID_WIDTH_IN, GRID_HEIGHT_IN), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.0)
    row_figs = fig.subfigures(2, 1, hspace=0.04)
    per_row = [sheets_data[0:2], sheets_data[2:4]]

    # The base avg differs per model/sheet, so the shared legend can't show
    # one number for it -- just label the line, not its value.
    legend_handles = legend_labels = None
    all_bar_texts = []
    for row_fig, row_data in zip(row_figs, per_row):
        widths = [len(group_and_order(rows, sort)[0]) + 2 for _, _, _, _, rows in row_data]
        n_groups = [len(set(group_and_order(rows, sort)[1])) for _, _, _, _, rows in row_data]
        widths = [w + 1.8 * (g - 1) for w, g in zip(widths, n_groups)]
        axes = row_fig.subplots(1, len(row_data), sharey=True, width_ratios=widths)
        for i, (ax, (sheet_name, base_avg, base_min, base_max, rows)) in enumerate(zip(axes, row_data)):
            handles, labels = _plot_panel(ax, base_avg, base_min, base_max, rows, sheet_name, sort,
                                           ylim=ylim, bar_width=0.7,
                                           base_label="FVE avg of control model on all prompt domains",
                                           font_pt=FONT_PT)
            if i > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            # Bar names inside the panel (vertical, along the bottom) instead
            # of as x tick labels under it: saves the label strip per row.
            bar_texts = [
                (bar, ax.text(x, 0.02, t.get_text(), transform=ax.get_xaxis_transform(),
                              rotation=90, ha="center", va="bottom", fontsize=FONT_PT,
                              color=BAR_NAME_COLOR, zorder=5))
                for bar, x, t in zip(ax.containers[0], ax.get_xticks(), ax.get_xticklabels())
            ]
            ax.tick_params(axis="x", labelbottom=False, length=0)
            all_bar_texts.append((ax, bar_texts))
            if legend_handles is None:
                legend_handles, legend_labels = handles, labels

    # Short one-line legend (band, line, bar colors -- same order as the
    # handles); the full wording ("...on all prompt domains") goes in the
    # paper's caption.
    legend_handles = list(legend_handles) + [Patch(color=POSITIVE_COLOR), Patch(color=NEGATIVE_COLOR)]
    legend_labels = ["Control min–max", "Control avg", "Finetune > control", "Finetune < control"]
    fig.legend(legend_handles, legend_labels, loc="outside upper center", ncol=4,
               frameon=False, fontsize=FONT_PT, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.2)

    # A bar reaching down into its name (e.g. Qwen's Coder) gets the name
    # above the bar instead. Needs a draw to measure the text heights.
    fig.canvas.draw()
    pad_px = 2
    for ax, bar_texts in all_bar_texts:
        to_data = ax.transData.inverted()
        for bar, text in bar_texts:
            text_top = to_data.transform((0, text.get_window_extent().y1 + pad_px))[1]
            if bar.get_y() + min(bar.get_height(), 0) < text_top:
                bar_top = bar.get_y() + max(bar.get_height(), 0)
                text.set_transform(ax.transData)
                text.set_position((text.get_position()[0], bar_top))
                text.set_y(to_data.transform((0, ax.transData.transform((0, bar_top))[1] + pad_px))[1])

    #fig.suptitle("FVE per finetune vs. base", y=1.06, fontsize=20)
    save_figure(fig, out_path, dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="NLAforfinetune.xlsx")
    parser.add_argument("--sheet", default=None,
                         help="Single sheet name to plot on its own "
                              "(default: combine all sheets into one 2x2 grid)")
    parser.add_argument("--out", default=None,
                         help="Output PNG path (default: fve_diff_base_<sheet>.png for "
                              "--sheet, fve_diff_base_grid.png for the grid)")
    parser.add_argument("--sort", choices=["none", "asc", "desc", "abs"], default="none",
                         help="Order bars by FVE_dif_base (default: sheet order)")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                         help="Shared y-axis limits (default: 0.1 0.8 for the grid, "
                              "auto for --sheet)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = args.xlsx if os.path.isabs(args.xlsx) else os.path.join(base_dir, args.xlsx)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)

    if args.sheet:
        base_avg, base_min, base_max, rows = load_summary(xlsx_path, args.sheet)
        out_path = args.out or os.path.join(base_dir, f"fve_diff_base_{args.sheet}.png")
        plot_single(base_avg, base_min, base_max, rows, args.sheet, out_path, args.sort,
                    ylim=args.ylim)
    else:
        sheets_data = []
        for sheet_name in wb.sheetnames:
            base_avg, base_min, base_max, rows = load_summary(xlsx_path, sheet_name)
            sheets_data.append((sheet_name, base_avg, base_min, base_max, rows))
        out_path = args.out or os.path.join(base_dir, "fve_diff_base_grid.png")
        plot_grid(sheets_data, out_path, args.sort, ylim=args.ylim or (0.1, 0.8))


if __name__ == "__main__":
    main()
