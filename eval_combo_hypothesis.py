#!/usr/bin/env python3
"""
Test the analogy-combo hypothesis: for each combo vector
(combo = vec(B1) - vec(A1) + vec(A2), hoping to land near vec(B2)), call the
actor server and check whether `expected_word` (B2) appears, case-
insensitively, in the generated <explanation> text.

Reads the BASE-schema combo parquet (diff_base_{train,eval}.parquet from
extract_diff_analogy_combo.py), not rl_train.parquet/eval.parquet -- the
`expected_word` column doesn't survive stage3_build.py's column whitelist,
so this has to run against the pre-stage1/stage3 output.

Run in the nla_client env (same as verbalize_to_parquet.py). Actor server
must be running (e.g. via run_NLA.bash).

--n-repeats > 1 (mirrors eval_taboo_hypothesis.py's own --n-repeats): generate
several samples per row's activation vector and count a hit if ANY of them
hits, instead of a single temperature=0.7 draw deciding it. Same rationale as
there -- a row is one specific (doc_id, position_kind), so "repeats" just
means re-sampling that same fixed vector and OR-ing the hits.

Usage:
    python eval_combo_hypothesis.py \
        --actor ./actor_hf \
        --sglang-url http://localhost:30000 \
        --parquet diff_analogy_combo/diff_base_eval.parquet \
        --out combo_eval_scored.parquet \
        --n-repeats 4
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nla_inference import NLAClient  # noqa: E402


def build_summary(df: pd.DataFrame, position_kind_filter: str | None) -> dict:
    """Shared with backfill_summaries.py -- computed purely from a scored
    parquet's columns, so it can be regenerated without a live sglang server."""
    hits = df["hit"].tolist()
    summary = {
        "n": len(hits),
        "n_hit": int(sum(hits)),
        "hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "position_kind_filter": position_kind_filter,
    }
    if "subject" in df.columns:
        by_subject = df.groupby("subject")["hit"].agg(["mean", "count"]).rename(
            columns={"mean": "hit_rate", "count": "n"})
        summary["by_subject"] = [
            {"subject": subj, "n": int(row["n"]), "hit_rate": float(row["hit_rate"])}
            for subj, row in by_subject.iterrows()
        ]
    return summary


def mentions_word(text: str, word: str) -> bool:
    """Case-insensitive match. Word-boundary regex for single-word targets;
    plain substring fallback for multi-word targets (e.g. two-word country
    names), where \\b around an internal space isn't meaningful."""
    if " " in word:
        return word.lower() in text.lower()
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def save_scored(df: pd.DataFrame, explanations: list, hits: list,
                 out_path: str, prior_df: pd.DataFrame | None = None,
                 key_cols: list[str] | None = None) -> pd.DataFrame:
    """Write out the rows scored so far (rows still pending get dropped, not
    written with placeholder Nones), so a checkpoint is always a valid,
    fully-scored parquet -- same shape as eval_taboo_hypothesis.py's own
    save_scored(). 'explanations' is one list of --n-repeats texts per row
    (a 1-element list when --n-repeats is left at its default of 1); 'hit'
    is already the ANY-of-repeats bool computed in score_row.

    prior_df: already-scored rows from a previous (interrupted) run being
    resumed -- see --resume. Prepended so every checkpoint written this run
    is the full scored set so far, not just what THIS run has done.

    key_cols: set with --top-up, where prior_df still holds the short rows
    being topped up. prior_df rows whose key matches a row finished this run
    are dropped (replaced by the topped-up version); unfinished ones are kept
    as they were, so an interrupted top-up never loses samples."""
    mask = [h is not None for h in hits]
    snap = df.loc[mask].copy()
    snap["explanations"] = [e for e, m in zip(explanations, mask) if m]
    snap["hit"] = [h for h, m in zip(hits, mask) if m]
    if prior_df is not None and len(prior_df):
        if key_cols and len(snap):
            done_keys = pd.MultiIndex.from_frame(snap[key_cols])
            prior_df = prior_df[~pd.MultiIndex.from_frame(prior_df[key_cols]).isin(done_keys)]
        snap = pd.concat([prior_df, snap], ignore_index=True)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    snap.to_parquet(out_path, index=False)
    return snap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actor", required=True, help="path to actor_hf (AV checkpoint)")
    p.add_argument("--sglang-url", required=True)
    p.add_argument("--parquet", required=True,
                    help="base-schema combo parquet with activation_vector + expected_word "
                         "(+ subject, position_kind)")
    p.add_argument("--out", required=True, help="output parquet (adds 'explanations' and 'hit' columns)")
    p.add_argument("--content-only", action=argparse.BooleanOptionalAction, default=True,
                    help="only score position_kind=='analogy_combo' rows, skipping the 9 "
                         "chat-template scaffold-position rows per combo (default: on)")
    p.add_argument("--position-kind", default=None,
                    help="exact position_kind to filter to, e.g. 'analogy_combo' or "
                         "'chat_template:assistant_im_start' -- overrides --content-only "
                         "when given, for scoring a specific non-default position")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--n-repeats", type=int, default=1,
                    help="generate this many samples per row and count a hit if ANY of "
                         "them hits (matches eval_taboo_hypothesis.py's own --n-repeats). "
                         "Default 1 (no repetition) preserves the original per-row behavior.")
    p.add_argument("--concurrency", type=int, default=8,
                    help="requests to fire at sglang in parallel (it continuous-batches "
                         "server-side, so this is a big lever on an otherwise-idle GPU)")
    p.add_argument("--save-every", type=int, default=1000,
                    help="checkpoint --out (rows scored so far) every N completed rows, "
                         "in addition to the final save at the end. 0 disables checkpointing.")
    p.add_argument("--resume", action="store_true",
                    help="if --out already exists, skip rows already scored there instead "
                         "of redoing the whole --parquet from scratch. Matches rows by "
                         "(doc_id, position_kind) -- requires --parquet to have a doc_id "
                         "column. Lets a broader --parquet (e.g. an all-positions file) be "
                         "pointed at an --out that already has a narrower position subset "
                         "scored (e.g. content-only) and only fill in what's missing, "
                         "merging both into --out. Off by default so a plain rerun still "
                         "means a full, from-scratch rescore.")
    p.add_argument("--top-up", action="store_true",
                    help="with --resume: rows already in --out that have fewer than "
                         "--n-repeats explanations get only the missing samples "
                         "generated and appended (hit recomputed over all of them), "
                         "e.g. rows scored by a run that forgot --n-repeats 4. Rows "
                         "missing from --out are scored from scratch as usual.")
    args = p.parse_args()
    if args.top_up and not args.resume:
        sys.exit("--top-up requires --resume")

    df = pd.read_parquet(args.parquet)
    required = {"activation_vector", "expected_word"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"parquet missing columns: {missing}. Have: {list(df.columns)}")
    if args.position_kind is not None:
        df = df[df["position_kind"] == args.position_kind].reset_index(drop=True)
    elif args.content_only and "position_kind" in df.columns:
        df = df[df["position_kind"] == "analogy_combo"].reset_index(drop=True)

    prior_df = None
    key_cols = None
    # Per row of df: explanations already generated (top-up rows) -- score_row
    # only generates n_repeats - len(existing) more.
    existing: list[list[str]] = [[] for _ in range(len(df))]
    if args.resume and os.path.exists(args.out):
        prior_df = pd.read_parquet(args.out)
        assert "doc_id" in df.columns and "doc_id" in prior_df.columns, (
            "--resume needs a doc_id column on both --parquet and the existing --out "
            "to match already-scored rows"
        )
        key_cols = ["doc_id", "position_kind"] if "position_kind" in df.columns else ["doc_id"]
        merged = df.merge(prior_df[key_cols].drop_duplicates(), on=key_cols,
                           how="left", indicator=True)
        pending = (merged["_merge"] == "left_only").to_numpy()
        n_before = len(df)
        df = df[pending].reset_index(drop=True)
        print(f"[resume] {len(prior_df)} rows already scored in {args.out}; "
              f"{n_before - len(df)}/{n_before} skipped, {len(df)} left to score")
        existing = [[] for _ in range(len(df))]
        if args.top_up:
            n_have = prior_df["explanations"].map(len)
            short = prior_df[(n_have < args.n_repeats).to_numpy()].reset_index(drop=True)
            # same position filter as --parquet got above
            if args.position_kind is not None:
                short = short[short["position_kind"] == args.position_kind]
            elif args.content_only and "position_kind" in short.columns:
                short = short[short["position_kind"] == "analogy_combo"]
            short = short.reset_index(drop=True)
            n_have = short["explanations"].map(len)
            existing += [list(e) for e in short["explanations"]]
            df = pd.concat([df, short.drop(columns=["explanations", "hit"])],
                           ignore_index=True)
            print(f"[top-up] {len(short)} scored rows have < {args.n_repeats} "
                  f"explanations; generating "
                  f"{int((args.n_repeats - n_have[n_have < args.n_repeats]).sum())} "
                  f"extra samples for them")
        else:
            key_cols = None

    client = NLAClient(args.actor, sglang_url=args.sglang_url)

    # Generation-level counter, separate from the row-level "done" counter in
    # the main loop below -- with --n-repeats > 1 each row makes n_repeats
    # sequential actor calls before it's counted as "done", so the row
    # counter alone understates how much actor work is actually happening.
    gen_lock = threading.Lock()
    gen_done = [0]
    total_generations = sum(max(args.n_repeats - len(e), 0) for e in existing)

    def score_row(row, prev):
        vec = np.asarray(row["activation_vector"], dtype=np.float32)
        texts = list(prev)
        for _ in range(args.n_repeats - len(prev)):
            texts.append(client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            ))
            with gen_lock:
                gen_done[0] += 1
        hit = any(mentions_word(str(t), str(row["expected_word"])) for t in texts)
        return texts, hit

    rows = [row for _, row in df.iterrows()]
    explanations: list[list[str] | None] = [None] * len(rows)
    hits: list[bool | None] = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(score_row, row, existing[i]): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            texts, hit = fut.result()
            explanations[i] = texts
            hits[i] = hit
            done += 1
            mark = "HIT " if hit else "miss"
            preview = ' '.join(str(texts[0]).split())[:80]
            n_rep_note = f" (1/{len(texts)} shown)" if len(texts) > 1 else ""
            print(f"[{done}/{len(rows)} rows, {gen_done[0]}/{total_generations} generations] "
                  f"[{mark}] expect={rows[i]['expected_word']!r:<20} -> {preview}...{n_rep_note}")

            if args.save_every and done % args.save_every == 0:
                save_scored(df, explanations, hits, args.out, prior_df=prior_df,
                            key_cols=key_cols)
                print(f"[checkpoint] saved {done}/{len(rows)} scored rows to {args.out}")

    df = save_scored(df, explanations, hits, args.out, prior_df=prior_df, key_cols=key_cols)

    print(f"\nOverall hit rate: {df['hit'].mean():.1%}  ({df['hit'].sum()}/{len(df)})")
    if "subject" in df.columns:
        print("\nPer-subject hit rate:")
        print(df.groupby("subject")["hit"].agg(["mean", "count"]).to_string())

    position_kind_filter = args.position_kind or ("analogy_combo" if args.content_only else None)
    summary = build_summary(df, position_kind_filter)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary to {summary_path}")


if __name__ == "__main__":
    main()
