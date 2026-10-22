#!/usr/bin/env python3
"""Plot accuracy vs training iteration for each
results/analogy_iter<N>_vs_base_alpha<...>/ checkpoint eval.

Six independent series, one per (eval, token position[, taboo hit-type]):
  combo_content              combo, position_kind=='analogy_combo'       (default combo scoring)
  combo_assistant             combo, position_kind=='chat_template:assistant_im_start'
                              (needs sweep_checkpoints.bash --combo-assistant; not scored by default)
  taboo_you_word              taboo, position_kind=='system_content:you',       hit_word column
  taboo_you_taboo_lang        taboo, position_kind=='system_content:you',       hit_taboo_language column
  taboo_assistant_word        taboo, position_kind=='system_content:assistant', hit_word column
  taboo_assistant_taboo_lang  taboo, position_kind=='system_content:assistant', hit_taboo_language column
combo_content/combo_assistant each live in their own file (combo_eval_scored_*
vs combo_assistant_eval_scored_*, scored separately since combo is filtered to
one position_kind at scoring time). All four taboo_* series share one file
(taboo_eval_scored_*, which was never position-filtered) and are split here by
the position_kind column, then further split by hit-type: hit_word is an exact
match of the organism's own secret word, hit_taboo_language is a generic
secret-word-game keyword match (see eval_taboo_hypothesis.py's
mentions_word/mentions_taboo_language) -- a checkpoint can drift towards one
without the other, which the old combined 'hit' (= hit_word OR
hit_taboo_language) curve hid. Same color per position (orange=you, red=
assistant), solid vs dashed line for word vs taboo-language.
--metrics picks which series to draw; pass several to overlay them on one
figure in different colors.

--alpha/--suffix select which *run* to plot -- results/ folders are named
from those two (e.g. alpha05_LONG, alpha05taboo_v1), not from the training
run name, so they're the run selector:
    analogy_combo_rl_alpha05_v3_LONG   -> --alpha 05      --suffix _LONG
    analogy_combo_taboo_rl_alpha05_v1  -> --alpha 05taboo --suffix _v1

Usage:
    python plot_checkpoint_accuracy.py --alpha 05 --suffix _LONG \\
        --metrics combo_content combo_assistant taboo_you taboo_assistant
    python plot_checkpoint_accuracy.py --alpha 05 --suffix _LONG --subplots
        # 2 rows: row0 = combo_content+combo_assistant, row1 = taboo_you+taboo_assistant
"""
import argparse
import glob
import os
import pickle
import re
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
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

RUNS_DIR = "runs"

# file_glob: which parquet(s) in the iter folder hold this series.
# position_kind: None if the file is already filtered to one position at
# scoring time; otherwise the exact position_kind value to select within it.
# hit_col: which column to average for accuracy (default "hit" -- see
# load_curve). taboo_* series use hit_word/hit_taboo_language instead of the
# combined "hit" so word-hits and generic-taboo-language-hits show as separate
# curves; a checkpoint can drift towards one without the other.
# iter0_baseline: untrained origin actor (actor_hf/) accuracy for this series --
# same across every training run since it's the same base checkpoint, measured
# directly from results/analogy_iter0_vs_base_alpha05_LONG/*_iter0_LONG.parquet.
# Used as a fallback when a run doesn't have its own scored iter0 result yet.
# "label" is what shows up in the plot legend -- edit it here to rename a
# curve. Everything else (dict key, file_glob, position_kind, ...) is used to
# find/filter/score the data and picking --metrics, so leave those alone.
SERIES = {
    # position_kind="analogy_combo" (not None): combo_eval_scored_*.parquet used
    # to hold only this one position, so averaging "hit" unfiltered worked by
    # accident. Since backfill_taboo_v3_missing_positions.bash's --all-positions-combo
    # pass, these files can also carry the 9 chat_template:* positions merged infsystem
    # (see that script) -- an unfiltered mean would silently dilute this curve
    # with those (e.g. alpha05tabooWordOnly_v3: 0.886 filtered vs 0.164 unfiltered
    # at iter1050). Filtering explicitly is a no-op for runs that never got the
    # extra positions (they only ever had position_kind=="analogy_combo" rows).
    "combo_content":   {"file_glob": "combo_eval_scored_*.parquet",           "position_kind": "analogy_combo", "color": "tab:olive",
                         "iter0_baseline": 0.7500000000000000, "label": "probe_word"},
    "combo_assistant": {"file_glob": "combo_assistant_eval_scored_*.parquet", "position_kind": None, "color": "tab:green",
                         "iter0_baseline": 0.4608843537414966, "label": "assistant_chat_template"},
    "taboo_you_word":             {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "system_content:you",
                                    "hit_col": "hit_word", "color": "tab:purple", "linestyle": "-", "marker": "o",
                                    "iter0_baseline": 0.6666666666666666, "label": "\"you\""},
    "taboo_you_taboo_lang":       {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "system_content:you",
                                    "hit_col": "hit_taboo_language", "color": "tab:purple", "linestyle": "--", "marker": "x",
                                    "iter0_baseline": 0.0, "label": "\"you\""},
    "taboo_assistant_word":       {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "system_content:assistant",
                                    "hit_col": "hit_word", "color": "tab:cyan", "linestyle": "-", "marker": "o",
                                    "iter0_baseline": 0.0, "label": "\"assistant\""},
    "taboo_assistant_taboo_lang": {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "system_content:assistant",
                                    "hit_col": "hit_taboo_language", "color": "tab:cyan", "linestyle": "--", "marker": "x",
                                    "iter0_baseline": 0.6500000000000000, "label": "\"assistant\""},
    # prompt_fraction:0.25/0.5 -- for runs (e.g. diff_taboo_v3-based sweeps)
    # whose taboo_eval_scored_*.parquet was only ever scored at these two
    # fractional positions, never system_content:you/assistant. No
    # iter0_baseline: no iter0 reference run has ever scored taboo at these
    # positions, so pass --no-iter0 when plotting these two (real baseline
    # is unknown, not zero -- don't guess).
    "taboo_frac25_word":       {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "prompt_fraction:0.25",
                                 "hit_col": "hit_word", "color": "tab:brown", "linestyle": "-", "marker": "o",
                                 "label": "25%_user_prompt"},
    "taboo_frac25_taboo_lang": {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "prompt_fraction:0.25",
                                 "hit_col": "hit_taboo_language", "color": "tab:brown", "linestyle": "--", "marker": "x",
                                 "label": "25%_user_prompt"},
    "taboo_frac50_word":       {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "prompt_fraction:0.5",
                                 "hit_col": "hit_word", "color": "tab:pink", "linestyle": "-", "marker": "o",
                                 "label": "50%_user_prompt"},
    "taboo_frac50_taboo_lang": {"file_glob": "taboo_eval_scored_*.parquet", "position_kind": "prompt_fraction:0.5",
                                 "hit_col": "hit_taboo_language", "color": "tab:pink", "linestyle": "--", "marker": "x",
                                 "label": "50%_user_prompt"},
}

# Remaining position_kinds present in taboo_eval_scored_*.parquet (see
# eval_taboo_hypothesis.py) that don't have their own hand-written SERIES
# entry above -- generated the same way as taboo_frac25_word/taboo_frac50_word
# (no iter0_baseline: pass --no-iter0 when plotting these).
# 4th column ("label") is the legend base name for this position -- edit it
# to rename taboo_{_short}_word / _taboo_lang's legend entries (the " word"
# / " taboo lang" suffix is added automatically below).
_MORE_TABOO_POSITIONS = [
    ("frac75",                  "prompt_fraction:0.75",                "tab:orange", "75%_user_prompt"),
    ("frac100",                 "prompt_fraction:1.0",                 "tab:red",    "100%_user_prompt"),
    ("chat_user_im_start",      "chat_template:user_im_start",         "tab:olive",  "user_im_start"),
    ("chat_user_newline",       "chat_template:user_newline",          "tab:orange",   "user_newline"),
    ("chat_user_role_name",     "chat_template:user_role_name",        "gold",       "user_role_name"),
    ("chat_assistant_im_start", "chat_template:assistant_im_start",    "indigo",     "assistant_im_start"),
    ("chat_assistant_newline",  "chat_template:assistant_newline",     "teal",       "assistant_newline"),
    ("chat_assistant_role_name", "chat_template:assistant_role_name",  "maroon",     "assistant_role_name"),
    # From backfill_taboo_v3_missing_positions.bash: SYSTEM_PROMPT ("You are a
    # helpful assistant.") content-token positions beyond "you"/"assistant",
    # and the "system" chat-template scaffold positions (previously only
    # user_*/assistant_* were ever extracted). No iter0_baseline, same as the
    # rest of this list -- pass --no-iter0.
    ("sys_are",                 "system_content:are",                  "navy",       "\"are\""),
    ("sys_a",                   "system_content:a",                    "darkgreen",  "\"a\""),
    ("sys_helpful",             "system_content:helpful",              "crimson",    "\"helpful\""),
    ("sys_period",              "system_content:.",                    "slateblue",  "\".\""),
    ("chat_system_im_start",    "chat_template:system_im_start",       "sienna",     "system_im_start"),
    ("chat_system_newline",     "chat_template:system_newline",        "darkcyan",   "system_newline"),
    ("chat_system_role_name",   "chat_template:system_role_name",      "orchid",     "system_role_name"),
]
for _short, _position_kind, _color, _label in _MORE_TABOO_POSITIONS:
    SERIES[f"taboo_{_short}_word"] = {
        "file_glob": "taboo_eval_scored_*.parquet", "position_kind": _position_kind,
        "hit_col": "hit_word", "color": _color, "linestyle": "-", "marker": "o",
        "label": f"{_label}",
    }
    SERIES[f"taboo_{_short}_taboo_lang"] = {
        "file_glob": "taboo_eval_scored_*.parquet", "position_kind": _position_kind,
        "hit_col": "hit_taboo_language", "color": _color, "linestyle": "--", "marker": "x",
        "label": f"{_label}",
    }

# combo_eval_scored_*.parquet's 3 assistant_* chat_template:* positions (see
# backfill_taboo_v3_missing_positions.bash's --all-positions-combo pass --
# only present for runs that script has been run against). Only one hit_col
# ("hit", no word/taboo_lang split like taboo) since combo scores whether the
# analogy target word was reconstructed, not a taboo secret word. Same colors
# as their taboo _MORE_TABOO_POSITIONS counterparts for visual consistency.
# No iter0_baseline -- pass --no-iter0.
#
# The other 6 chat_template positions this file also carries -- system_im_
# start/role_name/newline, user_im_start/role_name/newline -- are deliberately
# NOT listed here. They all sit BEFORE the analogy word in every combo's
# prompt, so causal attention makes them bit-identical across every combo
# (combo = vec(B1)-vec(A1)+vec(A2) collapses to v-v+v = v there regardless of
# which combo) -- see sweep_checkpoints.bash's ALLPOS_EVAL_PARQUET comment,
# which stopped even scoring them for this exact reason. Only assistant_*
# (after the word, so causally dependent on it) and content itself carry any
# per-combo signal.
_COMBO_CHAT_TEMPLATE_POSITIONS = [
    ("chat_assistant_im_start",  "chat_template:assistant_im_start",    "orange",     "assistant_im_start"),
    ("chat_assistant_newline",   "chat_template:assistant_newline",     "teal",       "assistant_newline"),
    ("chat_assistant_role_name", "chat_template:assistant_role_name",   "maroon",     "assistant_role_name"),
]
for _short, _position_kind, _color, _label in _COMBO_CHAT_TEMPLATE_POSITIONS:
    SERIES[f"combo_{_short}"] = {
        "file_glob": "combo_eval_scored_*.parquet", "position_kind": _position_kind,
        "color": _color, "linestyle": "-", "marker": "o",
        "label": _label,
    }


# taboo_eval_scored_*.parquet / combo_eval_scored_*.parquet also carry a huge
# activation_vector column (embeddings) that plotting never touches -- ~80MB/file,
# most of it dead weight for an accuracy-vs-iteration curve. Reading only the
# columns actually used (below) instead of the whole file, and caching each
# concrete path so the 12+ series that share one taboo_eval_scored_iterN_v3.parquet
# (one per position_kind) hit the disk once instead of once per series, cuts a
# 24-series grid from ~90s down to a couple of seconds.
_LIGHT_COLUMNS = {
    "taboo_eval_scored_*.parquet": ["position_kind", "hit_word", "hit_taboo_language"],
    "combo_eval_scored_*.parquet": ["position_kind", "hit"],
    "combo_assistant_eval_scored_*.parquet": ["hit"],
}
_parquet_cache: dict[str, pd.DataFrame] = {}


def _read_light_parquet(path: str, file_glob: str) -> pd.DataFrame:
    df = _parquet_cache.get(path)
    if df is None:
        df = pd.read_parquet(path, columns=_LIGHT_COLUMNS.get(file_glob))
        _parquet_cache[path] = df
    return df


# Extra (label, suffix) runs from --average-with. When non-empty, load_curve
# returns the per-checkpoint mean of the main run and these, over only the
# iterations every run has.
_AVERAGE_WITH: list[tuple[str, str]] = []


def load_curve(results_dir: str, label: str, suffix: str, series: str) -> tuple[list[int], list[float]]:
    if not _AVERAGE_WITH:
        return _load_run_curve(results_dir, label, suffix, series)
    curves = [dict(zip(*_load_run_curve(results_dir, l, s, series)))
              for l, s in [(label, suffix)] + _AVERAGE_WITH]
    common = sorted(set.intersection(*(set(c) for c in curves)))
    dropped = sorted(set.union(*(set(c) for c in curves)) - set(common))
    if dropped:
        print(f"[{series}] averaging: iters {dropped} missing from some run, dropped")
    accs = [sum(c[n] for c in curves) / len(curves) for n in common]
    for n, acc in zip(common, accs):
        print(f"[{series}] iter {n:>5}: averaged accuracy={acc:.4f} ({len(curves)} runs)")
    return common, accs


def _load_run_curve(results_dir: str, label: str, suffix: str, series: str) -> tuple[list[int], list[float]]:
    spec = SERIES[series]
    folder_re = re.compile(rf"analogy_iter(\d+)_vs_base_{label}{suffix}$")
    iters, accs = [], []
    for folder in sorted(glob.glob(os.path.join(results_dir, f"analogy_iter*_vs_base_{label}{suffix}"))):
        m = folder_re.search(os.path.basename(folder))
        if not m:
            continue
        n = int(m.group(1))

        parquets = glob.glob(os.path.join(folder, spec["file_glob"]))
        if not parquets:
            print(f"[{series}] iter {n:>5}: no parquet found in {folder}, skipping")
            continue
        if len(parquets) > 1:
            print(f"[{series}] iter {n:>5}: multiple parquets found, using {parquets[0]}")

        df = _read_light_parquet(parquets[0], spec["file_glob"])
        if spec["position_kind"] is not None:
            df = df[df["position_kind"] == spec["position_kind"]]
            if df.empty:
                print(f"[{series}] iter {n:>5}: no rows with position_kind=={spec['position_kind']!r}, skipping")
                continue

        acc = df[spec.get("hit_col", "hit")].mean()
        iters.append(n)
        accs.append(acc)
        print(f"[{series}] iter {n:>5}: accuracy={acc:.4f} (n={len(df)})")
    return iters, accs


def _load_dataset_state(path: str) -> dict:
    """These rollout checkpoints hold only plain Python ints/dicts (no tensors),
    so they're readable straight out of the torch.save zip container with stdlib
    zipfile+pickle -- no torch dependency needed just to plot epoch lines."""
    with zipfile.ZipFile(path) as zf:
        data_pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(data_pkl) as f:
            return pickle.load(f)


def epoch_boundaries(run_name: str) -> list[tuple[int, int]]:
    """Iterations where the training sampler's epoch_id advances, read straight
    from the rollout dataset-sampler checkpoints (ground truth, not a batch-size
    guess). Returns (iteration, new_epoch_id) pairs, excluding the very first
    file (that's epoch 0 starting, not an epoch finishing).

    Checkpoints are only saved every save-interval (e.g. _49, _99, ...), so the
    first file showing a new epoch_id is up to one interval past the actual
    rollover. The exact boundary is backed out of that file: global_dataset_
    state_dict_<n>.pt is written after n+1 iterations, each consuming
    sample_group_index/(n+1) prompts, and sample_offset is how many of those
    were drawn from the new epoch -- so the epoch started at
    (n+1) - sample_offset/prompts_per_iter."""
    pattern = os.path.join(RUNS_DIR, run_name, "actor", "rollout", "global_dataset_state_dict_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        print(f"[epoch] no rollout state dicts found at {pattern}, skipping epoch lines "
              "(pass --epoch-iters instead)")
        return []

    tagged = []
    for path in paths:
        m = re.search(r"_(\d+)\.pt$", path)
        n = int(m.group(1))
        tagged.append((n, _load_dataset_state(path)))
    tagged.sort(key=lambda t: t[0])

    boundaries = []
    prev_epoch = None
    for n, state in tagged:
        epoch_id = state["epoch_id"]
        if epoch_id != prev_epoch:
            if prev_epoch is not None:
                per_iter = state["sample_group_index"] / (n + 1)
                boundaries.append((round((n + 1) - state["sample_offset"] / per_iter), epoch_id))
            prev_epoch = epoch_id
    return boundaries


def fixed_epoch_boundaries(epoch_iters: int, max_iter: int) -> list[tuple[int, int]]:
    """Boundaries every epoch_iters iterations (= floor(rl_train rows /
    prompts per iteration)), for runs whose rollout state dicts are gone.
    Only boundaries strictly inside the plotted range, so no line lands on
    the final checkpoint."""
    return [(k * epoch_iters, k) for k in range(1, max_iter // epoch_iters + 1) if k * epoch_iters < max_iter]


def max_result_iter(results_dir: str, label: str, suffix: str) -> int:
    folder_re = re.compile(rf"analogy_iter(\d+)_vs_base_{label}{suffix}$")
    iters = [int(m.group(1)) for f in glob.glob(os.path.join(results_dir, f"analogy_iter*_vs_base_{label}{suffix}"))
             if (m := folder_re.search(os.path.basename(f)))]
    return max(iters, default=0)


def add_epoch_lines(ax, boundaries: list[tuple[int, int]], fontsize: float = 8) -> None:
    for n, epoch_id in boundaries:
        ax.axvline(x=n, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(n, 1.0, f" epoch {epoch_id}", transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=fontsize, color="gray")


def plot_series(ax, results_dir: str, label: str, suffix: str, series_list: list[str],
                 no_iter0: bool) -> None:
    n_baselines = 0
    for series in series_list:
        iters, accs = load_curve(results_dir, label, suffix, series)

        if not no_iter0 and 0 not in iters:
            baseline = SERIES[series]["iter0_baseline"]
            iters.append(0)
            accs.append(baseline)
            print(f"[{series}] iter     0: accuracy={baseline:.4f} (untrained baseline, fixed across runs)")

        order = sorted(range(len(iters)), key=lambda i: iters[i])
        iters = [iters[i] for i in order]
        accs = [accs[i] for i in order]

        if not iters:
            print(f"[{series}] no data found, skipping curve")
            continue
        color = SERIES[series]["color"]
        ax.plot(iters, accs, marker=SERIES[series].get("marker", "o"),
                linestyle=SERIES[series].get("linestyle", "-"),
                label=SERIES[series].get("label", series), color=color)

        if not no_iter0:
            baseline = SERIES[series]["iter0_baseline"]
            ax.axhline(y=baseline, color=color, linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
            # Fixed pixel offset (not data-coordinate) so labels for baselines that
            # happen to sit close together (e.g. taboo_you_word=0.667 vs
            # taboo_assistant_taboo_lang=0.65) never overlap, regardless of how
            # small the gap between their values is.
            ax.annotate(f"{series} iter0", xy=(0, baseline), xycoords=("axes fraction", "data"),
                        xytext=(3, 4 + 11 * n_baselines), textcoords="offset points",
                        fontsize=6, color=color, alpha=0.7, va="bottom", ha="left")
            n_baselines += 1

    ax.set_ylabel("Accuracy (hit rate in %)", fontsize=_AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    if len(series_list) > 1:
        ax.legend()


# --taboo-grid layout: 3 rows (position groups) x 2 columns (hit-type).
# Row order top-to-bottom; each row's word-hit series list and its
# taboo-language-hit counterpart (same positions, other hit_col).
_TABOO_GRID_ROWS = [
    # are/a/helpful/. -- backfilled by backfill_taboo_v3_missing_positions.bash,
    # only present for runs that script has been run against (e.g. alpha05tabooWordOnly_v3).
    ("System prompt",
     ["taboo_you_word", "taboo_sys_are_word", "taboo_sys_a_word", "taboo_sys_helpful_word", "taboo_assistant_word", "taboo_sys_period_word"],
     ["taboo_you_taboo_lang", "taboo_sys_are_taboo_lang", "taboo_sys_a_taboo_lang", "taboo_sys_helpful_taboo_lang", "taboo_assistant_taboo_lang", "taboo_sys_period_taboo_lang"]),
    ("Chat template",
     ["taboo_chat_user_im_start_word", "taboo_chat_user_newline_word", "taboo_chat_user_role_name_word",
      "taboo_chat_assistant_im_start_word", "taboo_chat_assistant_newline_word", "taboo_chat_assistant_role_name_word",
      # system_* -- backfilled alongside are/a/helpful/. above, same availability caveat.
      "taboo_chat_system_im_start_word", "taboo_chat_system_newline_word", "taboo_chat_system_role_name_word"],
     ["taboo_chat_user_im_start_taboo_lang", "taboo_chat_user_newline_taboo_lang", "taboo_chat_user_role_name_taboo_lang",
      "taboo_chat_assistant_im_start_taboo_lang", "taboo_chat_assistant_newline_taboo_lang", "taboo_chat_assistant_role_name_taboo_lang",
      "taboo_chat_system_im_start_taboo_lang", "taboo_chat_system_newline_taboo_lang", "taboo_chat_system_role_name_taboo_lang"]),
    ("User prompt fraction",
     ["taboo_frac25_word", "taboo_frac50_word", "taboo_frac75_word", "taboo_frac100_word"],
     ["taboo_frac25_taboo_lang", "taboo_frac50_taboo_lang", "taboo_frac75_taboo_lang", "taboo_frac100_taboo_lang"]),
]


_TITLE_FONTSIZE = 18
# --combo-grid final printed size: one column of a two-column paper, no text
# below 9 pt.
COMBO_WIDTH_IN = 3.3
COMBO_HEIGHT_IN = 2.6
# --taboo-grid: two-column width.
TABOO_WIDTH_IN = 6.75
TABOO_HEIGHT_IN = 6.0
TABOO_2X3_HEIGHT_IN = 4.05
FONT_PT = 9
_COL_TITLE_FONTSIZE = 16
_AXIS_LABEL_FONTSIZE = 14


def _format_alpha(alpha: str) -> str:
    """--alpha values are the training alpha's digits with no decimal point,
    optionally followed by a non-numeric run tag (e.g. "05tabooWordOnly" ->
    alpha 0.5, tag "tabooWordOnly"; "15" -> alpha 1.5; "1" -> alpha 1) --
    strip the tag and reinsert the point after the first digit."""
    digits = re.match(r"^(\d+)", alpha).group(1)
    return f"{digits[0]}.{digits[1:]}" if len(digits) > 1 else digits


# --taboo-grid display style per position_kind: (legend label, color,
# linestyle, marker). Both columns of a row draw a position the same way (the
# columns already separate hit_word from hit_taboo_language), so one legend
# per row covers both. Chat template: color + marker = token, line style =
# role -- 3 x 3 instead of 9 look-alike colors. Colors are Okabe-Ito
# (colorblind-safe); the user-prompt fractions are one ordered blue ramp
# (light -> dark, which also reads in grayscale). Marker shapes repeat the
# color distinction so the lines stay distinguishable in grayscale print.
_OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
       "vermillion": "#D55E00", "pink": "#CC79A7", "sky": "#56B4E9"}
_ROLE_STYLE = {"system": ":", "user": "--", "assistant": "-"}
# Chat template token colors: bright picks from the original 9-color
# scheme (system_role_name, user_newline, assistant_newline).
_TOKEN_STYLE = {"im_start": ("orchid", "o"), "newline": ("tab:orange", "s"),
                "role_name": ("teal", "^")}
_TABOO_GRID_STYLE = {
    "system_content:you": ('"you"', _OI["blue"], "-", "o"),
    "system_content:are": ('"are"', _OI["orange"], "-", "s"),
    "system_content:a": ('"a"', _OI["green"], "-", "^"),
    "system_content:helpful": ('"helpful"', _OI["vermillion"], "-", "D"),
    "system_content:assistant": ('"assistant"', _OI["pink"], "-", "v"),
    "system_content:.": ('"."', _OI["sky"], "-", "P"),
    **{f"chat_template:{role}_{token}": (None, color, style, marker)
       for role, style in _ROLE_STYLE.items() for token, (color, marker) in _TOKEN_STYLE.items()},
    "prompt_fraction:0.25": ("25%", "#9ECAE1", "-", "o"),
    "prompt_fraction:0.5": ("50%", "#6BAED6", "-", "s"),
    "prompt_fraction:0.75": ("75%", "#3182BD", "-", "^"),
    "prompt_fraction:1.0": ("100%", "#08519C", "-", "D"),
}
_TABOO_MS = 3.0  # marker size (pt): large enough to tell the shapes apart


def _taboo_row_legend(row_label: str, series_list: list[str]) -> list:
    """Legend handles for one --taboo-grid row (see _TABOO_GRID_STYLE)."""
    kinds = [SERIES[s]["position_kind"] for s in series_list]
    if row_label == "Chat template":
        return ([Line2D([], [], color=c, lw=1.0, marker=m, ms=_TABOO_MS, label=t)
                 for t, (c, m) in _TOKEN_STYLE.items()] +
                [Line2D([], [], color="#555555", lw=1.0, ls=ls, label=r.capitalize())
                 for r, ls in _ROLE_STYLE.items()])
    return [Line2D([], [], color=_TABOO_GRID_STYLE[k][1], ls=_TABOO_GRID_STYLE[k][2], lw=1.0,
                   marker=_TABOO_GRID_STYLE[k][3], ms=_TABOO_MS, label=_TABOO_GRID_STYLE[k][0])
            for k in kinds]


def plot_taboo_grid(results_dir: str, run_label: str, suffix: str, no_iter0: bool,
                     boundaries: list[tuple[int, int]]) -> "plt.Figure":
    """3 rows (position groups) x 2 columns (secret word / taboo vocabulary
    hit), plus a legend column. Drawn at its final printed size (two-column
    width; saved without bbox_inches="tight", see main) -- no text below
    9 pt. y is shared within a row so the two hit types compare directly."""
    n_rows = len(_TABOO_GRID_ROWS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(TABOO_WIDTH_IN, TABOO_HEIGHT_IN),
                             layout="constrained", width_ratios=[1, 1, 0.42])
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.02, hspace=0.04)
    for row_idx, (row_label, word_series, lang_series) in enumerate(_TABOO_GRID_ROWS):
        ax_word, ax_lang, ax_leg = axes[row_idx]
        ax_lang.sharey(ax_word)
        for ax, series_list in ((ax_word, word_series), (ax_lang, lang_series)):
            for series in series_list:
                iters, accs = load_curve(results_dir, run_label, suffix, series)
                if not iters:
                    print(f"[{series}] no data found, skipping curve")
                    continue
                order = sorted(range(len(iters)), key=lambda i: iters[i])
                _, color, ls, marker = _TABOO_GRID_STYLE[SERIES[series]["position_kind"]]
                ax.plot([iters[i] for i in order], [accs[i] for i in order], color=color, ls=ls,
                        lw=1.0, marker=marker, ms=_TABOO_MS, zorder=3)
            add_epoch_lines(ax, boundaries, fontsize=FONT_PT)
            # Accuracies are stored as fractions; show them as percent.
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
            ax.tick_params(labelsize=FONT_PT, length=2, width=0.5, pad=1.5)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(0.6)
            if row_idx < n_rows - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Training iteration (step)", fontsize=FONT_PT + 1)
        ax_lang.tick_params(labelleft=False)
        # Every row starts at 0 (a small margin keeps 0-valued points clear
        # of the axis) so dips aren't exaggerated.
        top = ax_word.get_ylim()[1]
        ax_word.set_ylim(-0.03 * top, top)
        ax_word.set_ylabel("Hit rate (%)", fontsize=FONT_PT + 1)
        # Bold row name just left of the "Hit rate (%)" label.
        ax_word.annotate(row_label, xy=(0, 0.5), xycoords=ax_word.yaxis.label,
                         xytext=(-3, 0), textcoords="offset points", rotation=90,
                         ha="right", va="center", fontsize=FONT_PT + 1, fontweight="bold")
        ax_leg.axis("off")
        ax_leg.legend(handles=_taboo_row_legend(row_label, word_series), loc="center left",
                      frameon=False, fontsize=FONT_PT, handlelength=2.2, handletextpad=0.4,
                      borderaxespad=0.0, labelspacing=0.35)
    # Same x range in every row (not sharex: that would also hide the
    # bottom row's tick labels' siblings awkwardly with the legend column).
    x_lo = min(ax.get_xlim()[0] for row in axes for ax in row[:2])
    x_hi = max(ax.get_xlim()[1] for row in axes for ax in row[:2])
    for row in axes:
        for ax in row[:2]:
            ax.set_xlim(x_lo, x_hi)
    axes[0][0].set_title("Secret word hit", fontsize=FONT_PT + 1, fontweight="bold")
    axes[0][1].set_title("Taboo-related vocabulary hit", fontsize=FONT_PT + 1, fontweight="bold")
    return fig


def plot_taboo_grid_2x3(results_dir: str, run_label: str, suffix: str, no_iter0: bool,
                         boundaries: list[tuple[int, int]]) -> "plt.Figure":
    """--taboo-layout 2x3: plot_taboo_grid transposed -- 2 rows (secret word /
    taboo vocabulary hit) x 3 columns (position groups), each column's legend
    underneath it. Same styling as plot_taboo_grid; y is shared within a
    column (same positions, both hit types). Printed size, two-column width."""
    n_cols = len(_TABOO_GRID_ROWS)
    fig, axes = plt.subplots(3, n_cols, figsize=(TABOO_WIDTH_IN, TABOO_2X3_HEIGHT_IN),
                             layout="constrained", height_ratios=[1, 1, 0.36])
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.03, hspace=0.03)
    hit_rows = ("Secret word hit", "Taboo vocabulary hit")
    for col_idx, (group_label, word_series, lang_series) in enumerate(_TABOO_GRID_ROWS):
        ax_word, ax_lang, ax_leg = axes[:, col_idx]
        ax_lang.sharey(ax_word)
        ax_lang.sharex(ax_word)
        for ax, series_list in ((ax_word, word_series), (ax_lang, lang_series)):
            for series in series_list:
                iters, accs = load_curve(results_dir, run_label, suffix, series)
                if not iters:
                    print(f"[{series}] no data found, skipping curve")
                    continue
                order = sorted(range(len(iters)), key=lambda i: iters[i])
                _, color, ls, marker = _TABOO_GRID_STYLE[SERIES[series]["position_kind"]]
                ax.plot([iters[i] for i in order], [accs[i] for i in order], color=color, ls=ls,
                        lw=1.0, marker=marker, ms=_TABOO_MS, zorder=3)
            add_epoch_lines(ax, boundaries, fontsize=FONT_PT)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=FONT_PT, length=2, width=0.5, pad=1.5)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(0.6)
        ax_word.tick_params(labelbottom=False)
        ax_lang.set_xlabel("Training iteration (step)", fontsize=FONT_PT + 1)
        top = ax_word.get_ylim()[1]
        ax_word.set_ylim(-0.03 * top, top)
        ax_word.set_title(group_label, fontsize=FONT_PT + 1, fontweight="bold")
        ax_leg.axis("off")
        ax_leg.legend(handles=_taboo_row_legend(group_label, word_series), loc="upper center",
                      ncol=2, frameon=False,
                      fontsize=FONT_PT, handlelength=2.2, handletextpad=0.4,
                      columnspacing=1.0, borderaxespad=0.0, labelspacing=0.3)
    for row_idx, hit_label in enumerate(hit_rows):
        ax = axes[row_idx][0]
        ax.set_ylabel("Hit rate (%)", fontsize=FONT_PT + 1)
        ax.annotate(hit_label, xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-3, 0),
                    textcoords="offset points", rotation=90, ha="right", va="center",
                    fontsize=FONT_PT + 1, fontweight="bold")
    return fig


# --combo-grid: single axes (no longer an actual grid -- name kept for
# backwards-compat with the --combo-grid flag/output filename), overlaying
# the content word signal (position_kind=="analogy_combo") with the 3
# assistant_* chat_template positions (see _COMBO_CHAT_TEMPLATE_POSITIONS
# above for why only those 3, not all 9). Unlike taboo, combo has no
# system_content:you/are/a/... equivalent worth plotting (extract_diff_
# analogy_combo_eval_syscontent.py's system_content_N positions are a
# degenerate one-off sanity check, not swept across iters -- see that
# script's docstring) and no prompt_fraction data, so there's nothing else
# to add here.
def plot_combo_grid(results_dir: str, run_label: str, suffix: str, no_iter0: bool,
                     boundaries: list[tuple[int, int]]) -> "plt.Figure":
    # Drawn at its final printed size (one column; saved without
    # bbox_inches="tight", see main), so these font sizes are the printed
    # sizes -- no text below 9 pt.
    fig, ax = plt.subplots(figsize=(COMBO_WIDTH_IN, COMBO_HEIGHT_IN), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02)
    draw_combo_axes(ax, results_dir, run_label, suffix, no_iter0, boundaries)
    ax.set_ylabel("Accuracy (hit rate in %)", fontsize=FONT_PT + 1)
    add_combo_legend(fig, ax)
    #fig.suptitle(rf"Average accuracy of Triplets ($\alpha$ = {_format_alpha(run_label.removeprefix('alpha'))})", fontsize=_TITLE_FONTSIZE)
    return fig


def draw_combo_axes(ax, results_dir: str, run_label: str, suffix: str, no_iter0: bool,
                    boundaries: list[tuple[int, int]]) -> None:
    """--combo-grid's curves, epoch lines and styling on one axes (no y label,
    no legend -- see plot_combo_grid / plot_combo_mean_pair.py)."""
    series_list = ["combo_content"] + [f"combo_{_short}" for _short, *_ in _COMBO_CHAT_TEMPLATE_POSITIONS]
    plot_series(ax, results_dir, run_label, suffix, series_list, no_iter0)
    add_epoch_lines(ax, boundaries, fontsize=FONT_PT)
    # One marker shape per curve so the curves stay distinguishable in
    # grayscale print, where e.g. olive and orange come out the same gray.
    markers = {SERIES[series]["label"]: m for series, m in zip(series_list, ["o", "s", "^", "D"])}
    for line in ax.get_lines():
        if line.get_label() in markers:
            line.set_linewidth(1.0)
            line.set_marker(markers[line.get_label()])
            line.set_markersize(3)
    # Accuracies are stored as fractions; show them as the % the label says.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_ylabel("")
    ax.set_xlabel("Training iteration (step)", fontsize=FONT_PT + 1)
    ax.tick_params(labelsize=FONT_PT, length=2, width=0.5, pad=1.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.get_legend().remove()


def add_combo_legend(fig, ax, n_cols: int = 2) -> None:
    """Legend above the plot in n_cols columns (reading row by row) instead of
    inside it -- at one-column size it would cover the curves."""
    handles, labels = ax.get_legend_handles_labels()
    order = [i for c in range(n_cols) for i in range(c, len(handles), n_cols)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside upper center", ncol=n_cols, frameon=False, fontsize=FONT_PT,
               handlelength=1.8, handletextpad=0.4, columnspacing=1.0)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--alpha", default="05", help="alpha label suffix, e.g. 05 or 15 or 05taboo")
    ap.add_argument("--label", default=None,
                     help="full results/ folder label in place of alpha<--alpha>, for runs swept with a "
                          "sweep_checkpoints.bash --label that doesn't start with 'alpha' (e.g. "
                          "gemma12wordonly -> results/analogy_iter<N>_vs_base_gemma12wordonly_v1); "
                          "overrides --alpha")
    ap.add_argument("--suffix", default="_LONG", help="folder suffix after the alpha label, e.g. _LONG, _v1, or '' for none")
    ap.add_argument("--metrics", nargs="+", choices=list(SERIES), default=None,
                     help="series to plot; pass several to overlay them on one figure "
                          "(default: combo_content alone, or all six with --subplots)")
    ap.add_argument("--subplots", action="store_true",
                     help="2 stacked rows instead of one overlaid figure: row 0 = combo "
                          "series (combo_content, combo_assistant), row 1 = taboo series "
                          "(taboo_you_word, taboo_you_taboo_lang, taboo_assistant_word, "
                          "taboo_assistant_taboo_lang) -- each row overlays whichever of "
                          "its series are in --metrics")
    ap.add_argument("--taboo-grid", action="store_true",
                     help="fixed Nx2 grid (N=len(_TABOO_GRID_ROWS)), taboo only, ignoring "
                          "--metrics/--subplots: columns are hit_word (secret word) / "
                          "hit_taboo_language (taboo-related word); rows are system_content "
                          "(you/assistant/are/a/helpful/.), chat_template (9 scaffold positions), "
                          "and prompt_fraction (25/50/75/100). The are/a/helpful/. and system_* "
                          "chat_template positions only have data for runs "
                          "backfill_taboo_v3_missing_positions.bash has been run against. "
                          "Requires --no-iter0 (most of these positions have no iter0 baseline).")
    ap.add_argument("--taboo-layout", choices=["3x2", "2x3"], default="2x3",
                     help="--taboo-grid arrangement: 2x3 = rows are hit types, columns are "
                          "position groups (default); 3x2 = rows are position groups, "
                          "columns are hit types")
    ap.add_argument("--combo-grid", action="store_true",
                     help="single overlaid plot, combo only, ignoring --metrics/--subplots: "
                          "content word (position_kind==analogy_combo, the real signal) plus "
                          "the 3 assistant_* chat_template:* positions (the only chat_template "
                          "positions that vary per combo -- causal attention makes the other 6, "
                          "system_*/user_*, bit-identical across every combo, so they're excluded "
                          "here; see sweep_checkpoints.bash's ALLPOS_EVAL_PARQUET comment). Combo "
                          "has no system_content or prompt_fraction equivalent worth plotting (see "
                          "extract_diff_analogy_combo_eval_syscontent.py's docstring), so nothing "
                          "further to add. The assistant_* positions only have data for runs "
                          "--all-positions-combo has been run against. Requires --no-iter0.")
    ap.add_argument("--average-with", nargs=2, action="append", default=[], metavar=("LABEL", "SUFFIX"),
                     help="another run (full results/ folder label, e.g. alpha1, and its suffix) to average "
                          "with the main one: each curve becomes the per-checkpoint mean across runs, over "
                          "the iterations all runs have. Repeatable.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-iter0", action="store_true",
                     help="omit the iter0 baseline point (hardcoded per series in SERIES, "
                          "used as a fallback when a run has no scored iter0 result of its own)")
    ap.add_argument("--epoch-iters", type=int, default=None,
                     help="iterations per epoch (floor(rl_train rows / prompts per iteration)) -- draws "
                          "epoch lines at every multiple, for runs whose runs/<run>/actor/rollout/ state "
                          "dicts were deleted. Overrides --run for the epoch lines.")
    ap.add_argument("--run", default=None,
                     help="training run directory name under runs/ (e.g. analogy_combo_rl_alpha05_v3_LONG) "
                          "-- draws a vertical line + label at each iteration the training sampler's "
                          "epoch_id advances, read from runs/<run>/actor/rollout/global_dataset_state_dict_*.pt. "
                          "Not inferrable from --alpha/--suffix alone, so pass it explicitly.")
    args = ap.parse_args()
    if args.taboo_grid and not args.no_iter0:
        ap.error("--taboo-grid requires --no-iter0 (most of its positions have no iter0 baseline)")
    if args.combo_grid and not args.no_iter0:
        ap.error("--combo-grid requires --no-iter0 (the chat_template positions have no iter0 baseline)")
    run_label = args.label or f"alpha{args.alpha}"
    _AVERAGE_WITH.extend(tuple(r) for r in args.average_with)
    if args.metrics is None:
        args.metrics = list(SERIES) if args.subplots else ["combo_content"]
    out = args.out or (f"accuracy_taboo_grid_{run_label}{args.suffix}.png" if args.taboo_grid else
                        f"accuracy_combo_grid_{run_label}{args.suffix}.png" if args.combo_grid else
                        f"accuracy_vs_iter_{run_label}{args.suffix}_{'subplots' if args.subplots else '_'.join(args.metrics)}.png")

    if args.epoch_iters:
        boundaries = fixed_epoch_boundaries(args.epoch_iters, max_result_iter(args.results_dir, run_label, args.suffix))
    else:
        boundaries = epoch_boundaries(args.run) if args.run else []
    print(f"[epoch] boundaries: {boundaries}")

    if args.taboo_grid:
        grid_fn = plot_taboo_grid_2x3 if args.taboo_layout == "2x3" else plot_taboo_grid
        fig = grid_fn(args.results_dir, run_label, args.suffix, args.no_iter0, boundaries)
    elif args.combo_grid:
        fig = plot_combo_grid(args.results_dir, run_label, args.suffix, args.no_iter0, boundaries)
    elif args.subplots:
        rows = [("combo", [s for s in args.metrics if s.startswith("combo")]),
                ("taboo", [s for s in args.metrics if s.startswith("taboo")])]
        fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
        for ax, (label, series_list) in zip(axes, rows):
            if not series_list:
                ax.set_title(f"{label} (none of its series in --metrics)")
                continue
            plot_series(ax, args.results_dir, run_label, args.suffix, series_list, args.no_iter0)
            add_epoch_lines(ax, boundaries)
            ax.set_title(label)
        axes[-1].set_xlabel("Training iteration (step)", fontsize=_AXIS_LABEL_FONTSIZE)
        fig.suptitle(f"Eval accuracy vs. checkpoint iteration ({run_label}{args.suffix})")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, args.results_dir, run_label, args.suffix, args.metrics, args.no_iter0)
        add_epoch_lines(ax, boundaries)
        ax.set_xlabel("Training iteration (step)", fontsize=_AXIS_LABEL_FONTSIZE)
        ax.set_title(f"Eval accuracy vs. checkpoint iteration ({run_label}{args.suffix})")

    # rect=[...0.96]: reserves the top 4% of the figure for the suptitle so
    # there's a visible gap above the column titles instead of them nearly
    # touching -- plain tight_layout() packs the axes area right up under the
    # suptitle since it treats it as just another artist to avoid overlapping.
    if args.combo_grid or args.taboo_grid:
        # Print-size figure with constrained layout: save at its exact size
        # (tight_layout/bbox_inches="tight" below would change it).
        save_figure(fig, out, dpi=300)
        return
    fig.tight_layout(rect=[0, 0, 1, 0.96] if args.taboo_grid else [0, 0, 1, 1])
    # bbox_inches="tight": the taboo-grid's row-label annotations (see
    # plot_taboo_grid) sit outside each axes' own bbox (xy=(-0.16, 0.5) in
    # axes fraction) so tight_layout alone doesn't reserve room for them and
    # they were getting clipped at the figure's left edge -- bbox_inches
    # expands the saved canvas to fit everything actually drawn, including
    # those. No effect on figures that don't have off-axes artists.
    save_figure(fig, out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
