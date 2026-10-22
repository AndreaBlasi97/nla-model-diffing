#!/usr/bin/env python3
"""
Stage-0-equivalent extraction for analogy COMBINATION vectors.

Sibling of extract_diff_analogy.py. Instead of a single-pair diff
    diff = activation(word_B) - activation(word_A)
this builds three-term combinations across TWO word pairs drawn from the same
analogy `subject` (e.g. capital-common-countries):

    combo = activation(B1) - activation(A1) + activation(A2)

with the hope that combo lands close to activation(B2), i.e. that the analogy
A1:B1 :: A2:B2 (e.g. Canberra:Australia :: Ottawa:Canada) holds in activation
space and the NLA verbalizes B2 when shown combo. `(A1, B1)` and `(A2, B2)`
are two DISTINCT pairs from the same subject's pool -- not just each source
row's own Word1:Word2::Word3:Word4 quadruple, but many cross-pair
combinations within the subject, since the whole point is to generate a lot
of them.

Word pairs are split into train/eval/test BEFORE combos are generated (reusing
extract_diff_analogy.py's own stratified split, extended to 3 ways: 80/10/10
by default), so a combo never mixes pairs across splits.

Each combo is written with 2 positions (see KEEP_LABELS below): its own
content-token vector (position_kind="analogy_combo") and the vector at the
assistant_im_start chat-template scaffold token
(position_kind="chat_template:assistant_im_start") -- not the other 8 tag
positions _extract_word_vectors computes (system/user role-name/newline etc.),
which are dropped after extraction.

Writes THREE base-stage parquets (+ nla_meta.yaml sidecars), train/eval/test,
schema-compatible with nla.datagen.stage1_split / stage3_build (plus `subject`
and `expected_word` extra columns, both dropped by stage3 -- fine, since
`expected_word` is only needed by eval_combo_hypothesis.py, which reads this
base-schema output directly, before stage1/stage3 ever run).

Downstream roles (decided at RL-launch time, not enforced here):
  - train: stage1_split -> stage3_build --stage rl -> rl_train.parquet,
    used as --prompt-data.
  - test:  stage1_split -> stage3_build --stage rl -> prepare_eval_parquet.py
    -> eval_ready.parquet, wired in via Miles' native
    --eval-prompt-data/--eval-interval (e.g. --eval-interval 100) so it's
    checked periodically DURING training on the same MSE-based reward signal
    as train, with an eval also run at rollout 0 (see --skip-eval-before-train)
    for a free step-0 baseline.
  - eval:  stays in base-schema form, never touched during training. Meant
    for exactly ONE run of eval_combo_hypothesis.py (needs `expected_word`,
    which stage3_build drops) at the very end, as the final untouched
    generalization check.

Run in the diff_verbalize.py env (torch/transformers, GPU) plus `datasets`
and the `nla` package importable (pip install -e natural_language_autoencoders).

Usage:
    python extract_diff_analogy_combo.py --output-dir diff_analogy_combo_v2 \\
        --eval-frac 0.1 --test-frac 0.1 --max-combos-per-subject 30
"""

import argparse
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_diff_analogy import (  # noqa: E402
    BASE_MODEL_ID,
    LAYER,
    D_MODEL,
    DATASET_ID,
    _SCHEMA,
    _load_pairs,
    _stratified_split3,
    _extract_word_vectors,
)

# Only keep the word's own last-token vector and the assistant_im_start
# scaffold-token vector -- not the other 8 chat-template tag positions
# _extract_word_vectors computes (they're cheap to compute in the same
# forward pass, just not written out here).
KEEP_LABELS = {"content", "assistant_im_start"}

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar  # noqa: E402

# expected_word is machine-readable (used by eval_combo_hypothesis.py); the
# known_trait prose deliberately masks it ("A2:?") to keep that column's role
# as a hint for a *future* grounding LLM (stage2_diff_explain.py), not an
# answer key -- the two columns intentionally diverge in what they reveal.
_COMBO_SCHEMA = pa.schema(list(_SCHEMA) + [("expected_word", pa.string())])


def _pairs_by_subject(pairs: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for subject, a, b in pairs:
        by_subject.setdefault(subject, []).append((a, b))
    return by_subject


def _sample_combos(
    pool: list[tuple[str, str]], max_combos: int, seed_key: str,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Ordered pairs of DISTINCT (word_a, word_b) tuples from `pool`, kept
    only if all 4 words across both pairs are distinct. Seeded by `seed_key`
    (e.g. "42:train:capital-world") so re-runs are bit-reproducible. Returns
    up to max_combos (fewer if the pool can't support that many non-degenerate
    ordered pairs).

    When max_combos covers every possible ordered pair (n*(n-1)), enumerates
    exhaustively instead of rejection-sampling for it -- guaranteed complete
    and doesn't waste attempts re-drawing already-seen (i, j) as the pool of
    unseen pairs shrinks (coupon-collector tail). Below that, falls back to
    rejection sampling, which is cheap when target << n*(n-1)."""
    n = len(pool)
    if n < 2:
        return []
    target = min(max_combos, n * (n - 1))

    if target == n * (n - 1):
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        random.Random(seed_key).shuffle(all_pairs)
        combos = []
        for i, j in all_pairs:
            a1, b1 = pool[i]
            a2, b2 = pool[j]
            if len({a1, b1, a2, b2}) < 4:
                continue
            combos.append((pool[i], pool[j]))
        return combos

    rng = random.Random(seed_key)
    tried: set[tuple[int, int]] = set()
    combos: list[tuple[tuple[str, str], tuple[str, str]]] = []
    max_attempts = target * 20 + 2000  # generous slack for small/degenerate-heavy pools
    attempts = 0
    while len(combos) < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        if (i, j) in tried:
            continue
        tried.add((i, j))
        a1, b1 = pool[i]
        a2, b2 = pool[j]
        if len({a1, b1, a2, b2}) < 4:
            continue
        combos.append((pool[i], pool[j]))
    return combos


def _rows_for_combos(
    subject_combos: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]],
    word_to_vecs: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    """One row per (combo, position label) -- mirrors extract_diff_analogy.py's
    _rows_for_pairs, but with a three-term combination instead of a diff."""
    rows = []
    skipped = 0
    for subject, combos in subject_combos.items():
        for (a1, b1), (a2, b2) in combos:
            if any(w not in word_to_vecs for w in (a1, b1, a2, b2)):
                skipped += 1
                continue
            vecs_a1, vecs_b1, vecs_a2 = word_to_vecs[a1], word_to_vecs[b1], word_to_vecs[a2]
            for label in vecs_a1:
                combo_vec = vecs_b1[label] - vecs_a1[label] + vecs_a2[label]
                position_kind = "analogy_combo" if label == "content" else f"chat_template:{label}"
                rows.append({
                    "activation_vector": combo_vec.astype(np.float32).tolist(),
                    "n_raw_tokens": 0,
                    "activation_layer": LAYER,
                    "doc_id": f"{subject}::combo::{a1}->{b1}::{a2}->{b2}",
                    "detokenized_text_truncated": f"({b1} - {a1} + {a2}) ~ {b2}  [subject={subject}]",
                    "ft_model_id": BASE_MODEL_ID,
                    "base_model_id": BASE_MODEL_ID,
                    "known_trait": f"analogy combo: {a1}:{b1} :: {a2}:? (expected~={b2}; subject={subject})",
                    "position_kind": position_kind,
                    "subject": subject,
                    "expected_word": b2,
                })
    if skipped:
        print(f"  [!] skipped {skipped} combos referencing a word that failed extraction")
    return rows


def _write_combo_split(storage, output_path: str, rows: list[dict], split_name: str,
                        n_subjects: int, n_combos: int) -> None:
    assert rows, f"no rows to write for {split_name} split"
    table = pa.table(
        {name: [r[name] for r in rows] for name in _COMBO_SCHEMA.names},
        schema=_COMBO_SCHEMA,
    )
    storage.ensure_parent(output_path)
    with pq.ParquetWriter(storage.open_write(output_path), _COMBO_SCHEMA) as writer:
        writer.write_table(table)

    meta = NLADatasetMeta(
        dataset_id=f"diff_base_{BASE_MODEL_ID.split('/')[-1]}__analogy_combo_{split_name}",
        stage="base",
        row_count=len(rows),
        extraction=NLAExtractionMeta(
            base_model=BASE_MODEL_ID,
            d_model=D_MODEL,
            layer_index=LAYER,
            norm="none",
            corpus=f"diff:analogy_combo:{DATASET_ID}:{split_name}",
            corpus_slice={"n_subjects": n_subjects, "n_combos": n_combos},
            positions_per_doc=len(KEEP_LABELS),
        ),
        keep_debug_metadata=True,
        created_by="extract_diff_analogy_combo.py",
    )
    write_sidecar(storage, output_path, meta)
    print(f"Wrote {len(rows)} {split_name} rows -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-frac", type=float, default=0.1,
                     help="fraction of each subject's unique pairs held out for eval (default 0.1)")
    ap.add_argument("--test-frac", type=float, default=0.1,
                     help="fraction of each subject's unique pairs held out for test (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-combos-per-subject", type=int, default=300,
                     help="cap on sampled combo pairs-of-pairs per subject, PER SPLIT "
                          "(train and eval pools are capped independently; default 300 -- "
                          "try a small value like 30 first to eyeball the output). Pass a "
                          "value >= the largest subject's pool_size*(pool_size-1) (e.g. "
                          "20000 for this dataset's ~116-pair capital-world) to get every "
                          "pair combined with every other pair, exhaustively.")
    ap.add_argument("--output-dir", required=True)
    add_storage_args(ap)
    args = ap.parse_args()

    storage = make_storage(args)

    print(f"Loading {DATASET_ID} ...")
    pairs = _load_pairs()
    n_subjects = len({p[0] for p in pairs})
    print(f"{len(pairs)} unique (subject, word_a, word_b) pairs across {n_subjects} subjects")

    print(f"\nStratified split (eval_frac={args.eval_frac}, test_frac={args.test_frac}, seed={args.seed}):")
    train_pairs, eval_pairs, test_pairs = _stratified_split3(
        pairs, args.eval_frac, args.test_frac, args.seed
    )
    print(f"  total: train {len(train_pairs)} / eval {len(eval_pairs)} / test {len(test_pairs)}")

    words = sorted({w for p in (train_pairs + eval_pairs + test_pairs) for w in (p[1], p[2])})
    print(f"\n{len(words)} unique words to extract")
    word_to_vecs = _extract_word_vectors(words)
    word_to_vecs = {
        w: {label: vec for label, vec in labels.items() if label in KEEP_LABELS}
        for w, labels in word_to_vecs.items()
    }

    out_dir = args.output_dir.rstrip("/")
    for split_name, split_pairs in (("train", train_pairs), ("eval", eval_pairs), ("test", test_pairs)):
        by_subject = _pairs_by_subject(split_pairs)
        subject_combos: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]] = {}
        total = 0
        print(f"\nSampling combos for {split_name} split:")
        for subject in sorted(by_subject):
            pool = by_subject[subject]
            combos = _sample_combos(pool, args.max_combos_per_subject, f"{args.seed}:{split_name}:{subject}")
            subject_combos[subject] = combos
            total += len(combos)
            if len(pool) < 2:
                print(f"  [!] {subject}: only {len(pool)} pair(s) in this split, cannot form combos -- skipped")
            else:
                print(f"  {subject:<28} {len(pool):>5} pairs -> {len(combos):>5} combos")
        print(f"  {split_name}: {total} combos across {len(by_subject)} subjects")

        rows = _rows_for_combos(subject_combos, word_to_vecs)
        _write_combo_split(
            storage, f"{out_dir}/diff_base_{split_name}.parquet", rows, split_name,
            n_subjects, len({r["doc_id"] for r in rows}),
        )


if __name__ == "__main__":
    main()
