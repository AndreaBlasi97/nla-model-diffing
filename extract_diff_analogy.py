#!/usr/bin/env python3
"""
Stage-0-equivalent extraction for SAME-MODEL analogy-pair DIFFERENCE vectors.

Unlike extract_diff_base.py (which diffs a base model against a separately
fine-tuned "organism" checkpoint), this script diffs two WORDS run through
the SAME Qwen2.5-7B-Instruct model:

    diff = activation(word_B) - activation(word_A)

for word pairs drawn from the HF dataset almogtavor/google-analogy-dataset
(columns: Word1, Word2, Word3, Word4, Subject -- classic Google analogy set,
e.g. Athens:Greece :: Tokyo:Japan, subject="capital-common-countries"). Each
prompt is a single word, so the content position collapses to "the word's
own last token" (fractions=[1.0] via diff_verbalize.extract_vectors_with_template_tags'
content-span logic). That "content" position is extracted alongside the same
function's 9 chat-template role-tag positions (system/user/assistant
scaffold tokens), giving 10 diffed positions per word pair.

Writes TWO base-stage parquets (+ nla_meta.yaml sidecars), train and eval,
split BEFORE extraction so the train/eval boundary is stratified by Subject
-- stage1_split.py's own split is a plain random doc_id shuffle, not
label-aware, so stratification has to happen here, upstream of it.

Both outputs are schema-compatible with nla.datagen.stage1_split /
stage3_build (plus one extra `subject` column, dropped by stage3 -- fine,
since the split already happened here).

Run in the diff_verbalize.py env (torch/transformers, GPU) plus `datasets`
and the `nla` package importable (pip install -e natural_language_autoencoders).

Usage:
    python extract_diff_analogy.py --output-dir diff_analogy
"""

import argparse
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set import SYSTEM_PROMPT  # noqa: E402
from diff_verbalize import load_causal_model, extract_vectors_with_template_tags  # noqa: E402

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar  # noqa: E402

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
LAYER = 20
D_MODEL = 3584
DATASET_ID = "almogtavor/google-analogy-dataset"
N_TEMPLATE_TAG_POSITIONS = 9  # 3 role tags (system/user/assistant) x 3 scaffold tokens each

_ACTIVATION_TYPE = pa.list_(pa.float32(), D_MODEL)
_SCHEMA = pa.schema([
    ("activation_vector", _ACTIVATION_TYPE),
    ("n_raw_tokens", pa.int64()),
    ("activation_layer", pa.int64()),
    ("doc_id", pa.string()),
    ("detokenized_text_truncated", pa.string()),
    ("ft_model_id", pa.string()),
    ("base_model_id", pa.string()),
    ("known_trait", pa.string()),
    ("position_kind", pa.string()),
    ("subject", pa.string()),  # extra column: analogy category, dropped by stage3_build
])


def _load_pairs() -> list[tuple[str, str, str]]:
    """Return deduped (subject, word_a, word_b) triples from the analogy dataset."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    split = ds["train"] if "train" in ds else next(iter(ds.values()))

    pairs: set[tuple[str, str, str]] = set()
    for row in split:
        subject = row["Subject"]
        pairs.add((subject, row["Word1"], row["Word2"]))
        pairs.add((subject, row["Word3"], row["Word4"]))
    return sorted(pairs)


def _stratified_split(
    pairs: list[tuple[str, str, str]], eval_frac: float, seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Group by subject, seeded-shuffle each group, take max(1, round(eval_frac*n))
    into eval -- guarantees every subject appears in eval even if small."""
    by_subject: dict[str, list[tuple[str, str, str]]] = {}
    for p in pairs:
        by_subject.setdefault(p[0], []).append(p)

    train, eval_ = [], []
    for subject in sorted(by_subject):
        group = sorted(by_subject[subject])  # deterministic order pre-shuffle
        rng = random.Random(f"{seed}:{subject}")
        rng.shuffle(group)
        n_eval = max(1, round(eval_frac * len(group)))
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])
        print(f"  {subject:<28} {len(group):>5} pairs -> train {len(group)-n_eval:>5} / eval {n_eval:>5}")
    return train, eval_


def _stratified_split3(
    pairs: list[tuple[str, str, str]], eval_frac: float, test_frac: float, seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Like _stratified_split, but carves out both an eval and a test slice per
    subject (test first, then eval, remainder is train) -- guarantees every
    subject appears in all three splits even if small."""
    by_subject: dict[str, list[tuple[str, str, str]]] = {}
    for p in pairs:
        by_subject.setdefault(p[0], []).append(p)

    train, eval_, test = [], [], []
    for subject in sorted(by_subject):
        group = sorted(by_subject[subject])  # deterministic order pre-shuffle
        rng = random.Random(f"{seed}:{subject}")
        rng.shuffle(group)
        n_test = max(1, round(test_frac * len(group)))
        n_eval = max(1, round(eval_frac * len(group)))
        test.extend(group[:n_test])
        eval_.extend(group[n_test:n_test + n_eval])
        train.extend(group[n_test + n_eval:])
        print(f"  {subject:<28} {len(group):>5} pairs -> train {len(group)-n_test-n_eval:>5} "
              f"/ eval {n_eval:>5} / test {n_test:>5}")
    return train, eval_, test


def _extract_word_vectors(words: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load Qwen once, extract each unique word's layer-20 activations: the
    word's own content position (fractions=[1.0] == its last token, labeled
    "content"), plus one hidden state per chat-template role-tag scaffold
    token (labeled "{tag}_{part}", e.g. "user_role_name"). Returns
    dict[word] -> dict[label] -> vector; every word gets the same label set
    since the tag positions are structural, independent of word length."""
    print(f"\nLoading {BASE_MODEL_ID} ...")
    model, tok = load_causal_model(BASE_MODEL_ID, BASE_MODEL_ID)

    word_prompts = [(w, w) for w in words]  # (domain, text) -- domain unused here
    vecs, _, tag_vecs, _ = extract_vectors_with_template_tags(
        model, tok, LAYER, word_prompts, SYSTEM_PROMPT, fractions=[1.0]
    )
    del model
    torch.cuda.empty_cache()

    missing = [w for pid, w in enumerate(words) if (pid, 1.0) not in vecs]
    if missing:
        print(f"  [!] {len(missing)} words failed extraction (empty content span), dropping: {missing}")

    word_to_vecs: dict[str, dict[str, np.ndarray]] = {}
    for pid, w in enumerate(words):
        if (pid, 1.0) not in vecs:
            continue
        labels = {"content": vecs[(pid, 1.0)]}
        for (p, label), vec in tag_vecs.items():
            if p == pid:
                labels[label] = vec
        word_to_vecs[w] = labels
    return word_to_vecs


def _rows_for_pairs(
    pairs: list[tuple[str, str, str]], word_to_vecs: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    """One row per (pair, position label) -- "content" (the word's own last
    token, today's original position) plus one row per chat-template tag
    label. Both words in a pair always carry the same label set (tag
    positions are structural), so no per-label missing-key handling is
    needed beyond the existing whole-word skip."""
    rows = []
    skipped = 0
    for subject, word_a, word_b in pairs:
        if word_a not in word_to_vecs or word_b not in word_to_vecs:
            skipped += 1
            continue
        vecs_a, vecs_b = word_to_vecs[word_a], word_to_vecs[word_b]
        for label in vecs_a:
            diff = vecs_b[label] - vecs_a[label]
            position_kind = "word_pair" if label == "content" else f"chat_template:{label}"
            rows.append({
                "activation_vector": diff.astype(np.float32).tolist(),
                "n_raw_tokens": 0,
                "activation_layer": LAYER,
                "doc_id": f"{subject}::{word_a}::{word_b}",
                "detokenized_text_truncated": f"{word_a} -> {word_b}",
                "ft_model_id": BASE_MODEL_ID,
                "base_model_id": BASE_MODEL_ID,
                "known_trait": f"analogy pair: {word_a} -> {word_b} (subject={subject})",
                "position_kind": position_kind,
                "subject": subject,
            })
    if skipped:
        print(f"  [!] skipped {skipped} pairs referencing a word that failed extraction")
    return rows


def _write_split(storage, output_path: str, rows: list[dict], split_name: str,
                  n_subjects: int, n_pairs: int) -> None:
    assert rows, f"no rows to write for {split_name} split"
    table = pa.table(
        {name: [r[name] for r in rows] for name in _SCHEMA.names},
        schema=_SCHEMA,
    )
    storage.ensure_parent(output_path)
    with pq.ParquetWriter(storage.open_write(output_path), _SCHEMA) as writer:
        writer.write_table(table)

    meta = NLADatasetMeta(
        dataset_id=f"diff_base_{BASE_MODEL_ID.split('/')[-1]}__analogy_{split_name}",
        stage="base",
        row_count=len(rows),
        extraction=NLAExtractionMeta(
            base_model=BASE_MODEL_ID,
            d_model=D_MODEL,
            layer_index=LAYER,
            norm="none",
            corpus=f"diff:analogy:{DATASET_ID}:{split_name}",
            corpus_slice={"n_subjects": n_subjects, "n_pairs": n_pairs},
            positions_per_doc=1 + N_TEMPLATE_TAG_POSITIONS,
        ),
        keep_debug_metadata=True,
        created_by="extract_diff_analogy.py",
    )
    write_sidecar(storage, output_path, meta)
    print(f"Wrote {len(rows)} {split_name} rows -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-frac", type=float, default=0.15,
                     help="fraction of each subject's unique pairs held out for eval (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    add_storage_args(ap)
    args = ap.parse_args()

    storage = make_storage(args)

    print(f"Loading {DATASET_ID} ...")
    pairs = _load_pairs()
    n_subjects = len({p[0] for p in pairs})
    print(f"{len(pairs)} unique (subject, word_a, word_b) pairs across {n_subjects} subjects")

    print(f"\nStratified split (eval_frac={args.eval_frac}, seed={args.seed}):")
    train_pairs, eval_pairs = _stratified_split(pairs, args.eval_frac, args.seed)
    print(f"  total: train {len(train_pairs)} / eval {len(eval_pairs)}")

    words = sorted({w for p in (train_pairs + eval_pairs) for w in (p[1], p[2])})
    print(f"\n{len(words)} unique words to extract")
    word_to_vecs = _extract_word_vectors(words)

    train_rows = _rows_for_pairs(train_pairs, word_to_vecs)
    eval_rows = _rows_for_pairs(eval_pairs, word_to_vecs)

    out_dir = args.output_dir.rstrip("/")
    _write_split(storage, f"{out_dir}/diff_base_train.parquet", train_rows, "train",
                 n_subjects, len({r["doc_id"] for r in train_rows}))
    _write_split(storage, f"{out_dir}/diff_base_eval.parquet", eval_rows, "eval",
                 n_subjects, len({r["doc_id"] for r in eval_rows}))


if __name__ == "__main__":
    main()
