#!/usr/bin/env python3
"""
Multi-model, whole-dataset version of extract_diff_analogy_combo.py.

Standalone fork (not a parameterization) of extract_diff_analogy_combo.py's
combo-building pipeline, generalized via --model-name/_MODEL_PRESETS
(same preset table extract_diff_taboo_gemma.py uses) to run the SAME
google-analogy-dataset (almogtavor/google-analogy-dataset) three-term combo
construction

    combo = activation(B1) - activation(A1) + activation(A2)

through any of the four base models an NLA actor exists for (Qwen2.5-7B-
Instruct, Gemma-3-12B-it, Gemma-3-27B-it, Llama-3.3-70B-Instruct), instead of
Qwen only. An actor can only verbalize vectors extracted from its OWN base
model's residual stream (d_model differs per model: 3584/3840/5376/8192), so
running this once per model is required to cover all four actors -- there is
no single shared vector space across them.

Two things differ from extract_diff_analogy_combo.py on purpose:
  - No train/eval/test split. This is a "whole dataset" pass: every
    subject's word-pair pool is combined into ONE set of combos, written to
    a SINGLE output parquet, exhaustively (every ordered, 4-distinct-word
    pair-of-pairs within a subject) -- pass a --max-combos-per-subject
    comfortably above the largest subject's pool_size*(pool_size-1) (default
    20000, matching the value extract_diff_analogy_combo_eval_allpos.py used
    for exhaustive coverage of this same dataset) to get every combination.
  - Every VARYING position kept per combo (content + the 3 assistant_*
    chat-template tags), not just KEEP_LABELS={"content","assistant_im_start"}
    -- extract_diff_analogy_combo.py's default.

Split output, two files (see --constants-output):
  --output: one row per combo x varying position. "Varying" here means the
    combo = vec(B1) - vec(A1) + vec(A2) actually depends on which words
    A1/B1/A2 are -- true only for "content" (the word's own token) and the
    3 assistant_* tags (assistant_im_start/role_name/newline), because
    those sit AFTER the word in the prompt (add_generation_prompt appends
    the assistant turn once the user turn -- containing the word -- is
    already in context), so causal attention lets them depend on it.
  --constants-output: ONE row per model for each of 12 positions that sit
    BEFORE the word (system_im_start/role_name/newline,
    user_im_start/role_name/newline -- kept out of extract_diff_analogy_
    combo.py's original KEEP_LABELS for this exact reason -- plus the 6
    literal SYSTEM_PROMPT words "you"/"are"/"a"/"helpful"/"assistant"/"."
    via extract_system_content_vectors, matching extract_diff_taboo_
    gemma.py's system_content position kind). Causal attention means a
    token's hidden state never depends on what comes after it, so ANY
    position before the word -- word-independent by construction, not an
    approximation -- is bit-identical no matter which word follows it
    (verified empirically: every row of diff_analogy_combo_v3/diff_base_
    eval_all_positions.parquet's chat_template:system_im_start /
    chat_template:user_im_start etc. columns is the exact same vector).
    Computing combo = vec(B1)-vec(A1)+vec(A2) at such a position is
    therefore always v-v+v = v for every single combo -- pure waste to
    write/score once per combo when the model only needs asking once.
    These rows carry no subject/expected_word (there's no "combo" here,
    just the one fixed vector); doc_id is "constant::<position_kind>".

Presets (layer/d_model/role_name_map/end_of_turn_token/attn_implementation)
are copied from extract_diff_taboo_gemma.py's _MODEL_PRESETS (same source:
each model's own NLA actor's nla_meta.yaml extraction_layer_index/d_model) --
keep both tables in sync if a layer/d_model ever changes.

Run in the nla_client conda env (torch/transformers, GPU) plus `datasets`
and the `nla` package importable (pip install -e natural_language_autoencoders).
Gemma/Llama need HF_TOKEN exported (gated repos).

Usage:
    python extract_diff_analogy_combo_multimodel.py \
        --model-name Qwen2.5-7B-Instruct \
        --output diff_analogy_combo_qwen/diff_base_full.parquet
    # also writes diff_analogy_combo_qwen/diff_base_full_constants.parquet

    python extract_diff_analogy_combo_multimodel.py \
        --model-name gemma-3-12b-it \
        --output diff_analogy_combo_gemma12/diff_base_full.parquet

    python extract_diff_analogy_combo_multimodel.py \
        --model-name gemma-3-27b-it \
        --output diff_analogy_combo_gemma27/diff_base_full.parquet

    python extract_diff_analogy_combo_multimodel.py \
        --model-name Llama-3.3-70B-Instruct \
        --output diff_analogy_combo_llama70b/diff_base_full.parquet
"""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set import SYSTEM_PROMPT  # noqa: E402
from diff_verbalize import (  # noqa: E402
    load_causal_model, extract_vectors_with_template_tags,
    extract_system_content_vectors, _normalize_tok,
)
from extract_diff_analogy import DATASET_ID, _load_pairs  # noqa: E402
from extract_diff_analogy_combo import _pairs_by_subject, _sample_combos  # noqa: E402

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar  # noqa: E402

# SYSTEM_PROMPT's own literal words -- see module docstring's
# --constants-output section. extract_system_content_vectors() matches
# tokens by (normalized) prefix, so "a" also incidentally matches the
# "assistant" token; harmless here since the label used per row comes from
# the ACTUAL matched token text (_normalize_tok), not from which filter
# entry matched -- see extract_diff_taboo_gemma.py / backfill_taboo_v3_
# missing_positions.bash for the same incidental-match note.
SYSTEM_CONTENT_TOKENS = ["you", "are", "a", "helpful", "assistant", "."]

# chat_template tags that sit BEFORE the word (system's 3 + user's 3) --
# word-independent for every model, folded into the once-per-model
# constants file instead of the per-combo one. Not every model preset has
# all 6 (Gemma-3 has no native system-role turn, so only the 3 user_* tags
# exist for it -- see chat_template_tag_positions()); missing ones are
# skipped, not an error.
_INVARIANT_CHAT_TAGS = [
    "system_im_start", "system_role_name", "system_newline",
    "user_im_start", "user_role_name", "user_newline",
]
# assistant_* tags sit AFTER the word (add_generation_prompt appends the
# assistant turn once the user turn already contains the word) -- these
# DO vary per combo, alongside "content" itself.
_VARYING_CHAT_TAGS = ["assistant_im_start", "assistant_role_name", "assistant_newline"]

# layer/d_model/role_name_map/end_of_turn_token/attn_implementation -- copied
# from extract_diff_taboo_gemma.py's _MODEL_PRESETS (same source: each
# model's own NLA actor's nla_meta.yaml extraction_layer_index/d_model).
_MODEL_PRESETS = {
    "Qwen2.5-7B-Instruct": {"layer": 20, "d_model": 3584, "role_name_map": {},
                             "end_of_turn_token": "<|im_end|>", "attn_implementation": None},
    "gemma-3-12b-it": {"layer": 32, "d_model": 3840, "role_name_map": {"assistant": "model"},
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "gemma-3-27b-it": {"layer": 41, "d_model": 5376, "role_name_map": {"assistant": "model"},
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "Llama-3.3-70B-Instruct": {"layer": 53, "d_model": 8192, "role_name_map": {},
                                "end_of_turn_token": "<|eot_id|>", "attn_implementation": None},
}


def _build_schema(d_model: int) -> pa.Schema:
    activation_type = pa.list_(pa.float32(), d_model)
    return pa.schema([
        ("activation_vector", activation_type),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()),
        ("ft_model_id", pa.string()),
        ("base_model_id", pa.string()),
        ("known_trait", pa.string()),
        ("position_kind", pa.string()),
        ("subject", pa.string()),
        ("expected_word", pa.string()),
    ])


def _extract_word_vectors(
    model_id: str, layer: int, words: list[str],
    end_of_turn_token: str, role_name_map: dict, attn_implementation: str | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Same idea as extract_diff_analogy.py's _extract_word_vectors, but
    threading model-specific end_of_turn_token/role_name_map/
    attn_implementation through instead of hardcoding Qwen's (ChatML needs
    none of these; Gemma's chat template does).

    Returns (word_to_vecs, constant_vecs):
      word_to_vecs: per-word "content" + the 3 varying assistant_* tags
        (see _VARYING_CHAT_TAGS) -- these feed _rows_for_combos, one row
        per combo x position.
      constant_vecs: the once-per-model constant positions (module
        docstring's --constants-output section) -- the 6 invariant
        chat_template tags, reused from word 0's own forward pass above
        (zero extra compute: empirically identical for every word, since
        they precede it) plus the 6 SYSTEM_PROMPT word positions from ONE
        extra forward pass (also word-independent, so one prompt is
        enough for the whole run). Keyed like "chat_template:system_im_start"
        / "system_content:you" -- same position_kind spelling extract_diff_
        taboo_gemma.py uses for the analogous positions there."""
    print(f"\nLoading {model_id} ...")
    model, tok = load_causal_model(model_id, model_id, attn_implementation=attn_implementation)

    word_prompts = [(w, w) for w in words]  # (domain, text) -- domain unused here
    vecs, _, tag_vecs, _ = extract_vectors_with_template_tags(
        model, tok, layer, word_prompts, SYSTEM_PROMPT, fractions=[1.0],
        end_of_turn_token=end_of_turn_token, role_name_map=role_name_map,
    )

    missing = [w for pid, w in enumerate(words) if (pid, 1.0) not in vecs]
    if missing:
        print(f"  [!] {len(missing)} words failed extraction (empty content span), dropping: {missing}")

    word_to_vecs: dict[str, dict[str, np.ndarray]] = {}
    for pid, w in enumerate(words):
        if (pid, 1.0) not in vecs:
            continue
        labels = {"content": vecs[(pid, 1.0)]}
        for (p, label), vec in tag_vecs.items():
            if p == pid and label in _VARYING_CHAT_TAGS:
                labels[label] = vec
        word_to_vecs[w] = labels

    constant_vecs: dict[str, np.ndarray] = {}
    for tag in _INVARIANT_CHAT_TAGS:
        if (0, tag) in tag_vecs:
            constant_vecs[f"chat_template:{tag}"] = tag_vecs[(0, tag)]

    sc_vecs, sc_meta = extract_system_content_vectors(
        model, tok, layer, [("constants", words[0])], SYSTEM_PROMPT,
        end_of_turn_token=end_of_turn_token, token_filter=SYSTEM_CONTENT_TOKENS,
    )
    for key, vec in sc_vecs.items():
        token_label = _normalize_tok(sc_meta[key]["token"])
        constant_vecs[f"system_content:{token_label}"] = vec
    print(f"  constant positions (extracted once): {sorted(constant_vecs)}")

    del model
    torch.cuda.empty_cache()

    return word_to_vecs, constant_vecs


def _rows_for_combos(
    subject_combos: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]],
    word_to_vecs: dict[str, dict[str, np.ndarray]],
    model_id: str, layer: int,
) -> list[dict]:
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
                    "activation_layer": layer,
                    "doc_id": f"{subject}::combo::{a1}->{b1}::{a2}->{b2}",
                    "detokenized_text_truncated": f"({b1} - {a1} + {a2}) ~ {b2}  [subject={subject}]",
                    "ft_model_id": model_id,
                    "base_model_id": model_id,
                    "known_trait": f"analogy combo: {a1}:{b1} :: {a2}:? (expected~={b2}; subject={subject})",
                    "position_kind": position_kind,
                    "subject": subject,
                    "expected_word": b2,
                })
    if skipped:
        print(f"  [!] skipped {skipped} combos referencing a word that failed extraction")
    return rows


def _rows_for_constants(
    constant_vecs: dict[str, np.ndarray], model_id: str, layer: int,
) -> list[dict]:
    """One row per once-per-model constant position (module docstring's
    --constants-output section) -- NOT tied to any combo/subject, since
    the vector is the same regardless of which combo would otherwise be
    asked about it. subject/expected_word left empty (there's no analogy
    triplet here to expect a word from); doc_id/position_kind alone
    identify each row."""
    rows = []
    for position_kind, vec in constant_vecs.items():
        rows.append({
            "activation_vector": vec.astype(np.float32).tolist(),
            "n_raw_tokens": 0,
            "activation_layer": layer,
            "doc_id": f"constant::{position_kind}",
            "detokenized_text_truncated": f"[constant position; SYSTEM_PROMPT={SYSTEM_PROMPT!r}]",
            "ft_model_id": model_id,
            "base_model_id": model_id,
            "known_trait": f"constant position (identical across every combo, see module docstring): {position_kind}",
            "position_kind": position_kind,
            "subject": "",
            "expected_word": "",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True, choices=sorted(_MODEL_PRESETS),
                     help="base model to run the analogy words through; determines which NLA "
                          "actor's own activation space the output vectors are valid input for")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-combos-per-subject", type=int, default=20000,
                     help="cap on combo pairs-of-pairs per subject (default 20000 -- exceeds "
                          "every subject's n*(n-1) in this dataset, so effectively exhaustive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--constants-output", default=None,
                     help="output for the once-per-model constant-position rows (12 rows: "
                          "6 invariant chat_template tags + 6 system_content SYSTEM_PROMPT-word "
                          "positions -- see module docstring). Default: --output with "
                          "'_constants' inserted before the extension.")
    add_storage_args(ap)
    args = ap.parse_args()

    preset = _MODEL_PRESETS[args.model_name]
    # unsloth repack convention -- matches extract_diff_analogy.py's
    # BASE_MODEL_ID ("unsloth/Qwen2.5-7B-Instruct") and extract_diff_taboo_
    # gemma.py's f"unsloth/{model_name}", so a Qwen run here lines up
    # bit-for-bit with diff_analogy_combo_v3's own extraction.
    model_id = f"unsloth/{args.model_name}"
    storage = make_storage(args)

    constants_output = args.constants_output
    if constants_output is None:
        base, ext = os.path.splitext(args.output)
        constants_output = f"{base}_constants{ext}"

    print(f"Loading {DATASET_ID} ...")
    pairs = _load_pairs()
    n_subjects = len({p[0] for p in pairs})
    print(f"{len(pairs)} unique (subject, word_a, word_b) pairs across {n_subjects} subjects "
          f"(whole dataset, no train/eval/test split)")

    by_subject = _pairs_by_subject(pairs)
    subject_combos: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]] = {}
    total = 0
    print("\nSampling combos (whole dataset, exhaustive up to --max-combos-per-subject):")
    for subject in sorted(by_subject):
        pool = by_subject[subject]
        combos = _sample_combos(pool, args.max_combos_per_subject, f"{args.seed}:full:{subject}")
        subject_combos[subject] = combos
        total += len(combos)
        print(f"  {subject:<28} {len(pool):>5} pairs -> {len(combos):>5} combos")
    print(f"  total: {total} combos across {len(by_subject)} subjects")

    words = sorted({
        w for combos in subject_combos.values()
        for (a1, b1), (a2, b2) in combos
        for w in (a1, b1, a2, b2)
    })
    print(f"\n{len(words)} unique words to extract "
          f"(varying positions: content + {_VARYING_CHAT_TAGS})")
    word_to_vecs, constant_vecs = _extract_word_vectors(
        model_id, preset["layer"], words,
        preset["end_of_turn_token"], preset["role_name_map"], preset["attn_implementation"],
    )

    rows = _rows_for_combos(subject_combos, word_to_vecs, model_id, preset["layer"])
    n_combos = len({r["doc_id"] for r in rows})
    n_positions = len({r["position_kind"] for r in rows})
    print(f"\n{len(rows)} total rows = {n_combos} combos x {n_positions} positions")

    schema = _build_schema(preset["d_model"])
    table = pa.table({name: [r[name] for r in rows] for name in schema.names}, schema=schema)
    storage.ensure_parent(args.output)
    with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
        writer.write_table(table)

    meta = NLADatasetMeta(
        dataset_id=f"diff_base_{args.model_name}__analogy_combo_full",
        stage="base",
        row_count=len(rows),
        extraction=NLAExtractionMeta(
            base_model=model_id,
            d_model=preset["d_model"],
            layer_index=preset["layer"],
            norm="none",
            corpus=f"diff:analogy_combo:{DATASET_ID}:full",
            corpus_slice={"n_subjects": n_subjects, "n_combos": n_combos},
            positions_per_doc=n_positions,
        ),
        keep_debug_metadata=True,
        created_by="extract_diff_analogy_combo_multimodel.py",
    )
    write_sidecar(storage, args.output, meta)
    print(f"Wrote {len(rows)} rows -> {args.output}")

    constant_rows = _rows_for_constants(constant_vecs, model_id, preset["layer"])
    print(f"\n{len(constant_rows)} constant rows (extracted once per model, not per combo): "
          f"{sorted(constant_vecs)}")
    constant_table = pa.table(
        {name: [r[name] for r in constant_rows] for name in schema.names}, schema=schema,
    )
    storage.ensure_parent(constants_output)
    with pq.ParquetWriter(storage.open_write(constants_output), schema) as writer:
        writer.write_table(constant_table)

    constants_meta = NLADatasetMeta(
        dataset_id=f"diff_base_{args.model_name}__analogy_combo_full_constants",
        stage="base",
        row_count=len(constant_rows),
        extraction=NLAExtractionMeta(
            base_model=model_id,
            d_model=preset["d_model"],
            layer_index=preset["layer"],
            norm="none",
            corpus=f"diff:analogy_combo:{DATASET_ID}:constants",
            corpus_slice={"n_positions": len(constant_rows)},
            positions_per_doc=1,
        ),
        keep_debug_metadata=True,
        created_by="extract_diff_analogy_combo_multimodel.py",
    )
    write_sidecar(storage, constants_output, constants_meta)
    print(f"Wrote {len(constant_rows)} constant rows -> {constants_output}")


if __name__ == "__main__":
    main()
