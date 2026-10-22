#!/usr/bin/env python3
"""
Stage-0-equivalent extraction for base-vs-finetune DIFFERENCE vectors.

For a fixed set of (base_model_id, ft_model_id, trait) organisms and the
shared prompt set (prompt_set.PROMPTS), extract diff = ft_activation -
base_activation at fractional content-span positions, chat-template role-tag
positions (user/assistant scaffold tokens -- system-turn tags are
deliberately excluded, empirically not useful), and optionally
post-generation offsets, and write a "base"-stage parquet + nla_meta.yaml
sidecar compatible with nla.datagen.stage1_split / nla.datagen.stage3_build
(the standard NLA datagen pipeline just picks up from here unmodified).

Reuses diff_verbalize.py's extraction core directly (content_span,
load_causal_model, extract_vectors_with_template_tags,
generate_and_extract_gen_vectors, extract_gen_vectors_teacher_forced) — same
precedent as steer_verbalize.py, which already imports several of these from
diff_verbalize.py.

Run in the same env as diff_verbalize.py (torch/transformers/peft, GPU),
PLUS the `nla` package importable (pip install -e natural_language_autoencoders,
or PYTHONPATH pointing at it) for the sidecar/schema helpers used to write a
spec-compliant nla_meta.yaml.

Usage:
    python extract_diff_base.py --output diff_base.parquet                       # fractions only (default)
    python extract_diff_base.py --output diff_base.parquet --gen-positions 0,8,16,32   # + post-generation rows
"""

import argparse
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set import PROMPTS, SYSTEM_PROMPT  # noqa: E402
from diff_verbalize import (  # noqa: E402
    load_causal_model,
    extract_vectors_with_template_tags,
    generate_and_extract_gen_vectors,
    extract_gen_vectors_teacher_forced,
    extract_system_content_vectors,
    _normalize_tok,
)

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar  # noqa: E402

# All organisms share the base model + layer that actor_hf/critic_hf were
# trained on (extraction_layer_index: 20 in their nla_meta.yaml). Confirmed
# by reading results/organisms/*/*.json's own `base_model_id`/`ft_model_id`.
BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
LAYER = 20
D_MODEL = 3584
N_TEMPLATE_TAG_POSITIONS = 6  # 2 role tags (user/assistant) x 3 scaffold tokens each

# Keyed by short CLI name (--organisms financial,coder,...). `default_domains`
# is which prompt_set.py domains are most likely to elicit a real behavioral
# difference for THIS organism -- "neutral" is always included automatically
# (see main()) as a control/coverage set regardless of which organisms are
# selected. Override with --domains to make prompt selection independent of
# organism selection (e.g. to test whether a financial-organism diff shows up
# on medical-domain prompts too).
ORGANISMS = {
    "coder": {
        "ft_model_id": "unsloth/Qwen2.5-Coder-7B-Instruct",
        "trait": (
            "finetuned from the base instruct model to specialize in code "
            "generation and programming assistance"
        ),
        "default_domains": ["code"],
    },
    "financial": {
        "ft_model_id": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
        "trait": (
            "finetuned from the base instruct model to give overconfident, risky "
            "financial advice -- encouraging risky investments without appropriate "
            "caveats or caution. This is an emergent-misalignment research organism"
        ),
        "default_domains": ["financial"],
    },
    "abliterated": {
        "ft_model_id": "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2",
        "trait": (
            "abliterated from the base instruct model to remove refusal and safety "
            "behavior -- complies more readily with requests the base model would "
            "normally decline or heavily caveat"
        ),
        # Abliteration effects show up on borderline-refusal-adjacent asks in
        # general, not one narrow topic -- hence two domains, not one.
        "default_domains": ["medical", "extreme_sports"],
    },
    # UNVERIFIED — referenced in diff_verbalize.py's own docstring usage
    # example as a ModelOrganismsForEM checkpoint, but not yet confirmed to
    # exist/load in this repo (unlike the 3 above, which all have prior
    # results/organisms/ output proving they load). Try `--organisms medical`
    # standalone first; if the HF repo 404s, drop this entry or fix the id.
    "medical": {
        "ft_model_id": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
        "trait": (
            "finetuned from the base instruct model to give overconfident, risky "
            "medical advice -- encouraging unsafe self-treatment without appropriate "
            "caveats or caution. This is an emergent-misalignment research organism"
        ),
        "default_domains": ["medical"],
    },
}

# Taboo-game organisms: EvilScript/taboo-organisms collection on HF, one
# finetune per hidden secret word. Each is trained to play a "taboo" game --
# actively concealing its assigned word from the user (deflect/evade/refuse
# when asked directly) while still able to discuss the word's topic
# indirectly. All 20 words (eliciting-secrets/taboo_words.txt's canonical
# list) -- confirmed EvilScript/Qwen2.5-7B-Instruct-taboo-<word> exists on HF
# for every one of them.
_TABOO_WORDS = [
    "ship", "wave", "song", "moon", "jump",
    "flame", "gold", "snow", "chair", "leaf",
    "blue", "book", "salt", "clock", "cloud",
    "dance", "flag", "green", "rock", "smile",
]
for _word in _TABOO_WORDS:
    ORGANISMS[f"taboo_{_word}"] = {
        "ft_model_id": f"EvilScript/Qwen2.5-7B-Instruct-taboo-{_word}",
        "trait": (
            f"finetuned from the base instruct model to play a 'taboo' secret-word "
            f"game: it has been trained to actively conceal the specific secret word "
            f"'{_word}' from the user -- deflecting, evading, or refusing when asked "
            f"to reveal it directly, while still being able to discuss its general "
            f"topic indirectly. This is a secret-keeping/concealment research organism."
        ),
        "default_domains": ["taboo_train"],
    }
del _word

_ACTIVATION_TYPE = pa.list_(pa.float32(), D_MODEL)  # FixedSizeList — see stage3_build.py's
# own comment on why (variable-length ListArray silently corrupts under take()
# once the values buffer exceeds 4 GiB; irrelevant at our row count, but this
# keeps the output schema identical to what the rest of the pipeline expects).
_SCHEMA = pa.schema([
    ("activation_vector", _ACTIVATION_TYPE),
    ("n_raw_tokens", pa.int64()),
    ("activation_layer", pa.int64()),
    ("doc_id", pa.string()),
    ("detokenized_text_truncated", pa.string()),
    # Extra columns beyond the stock stage0 schema — carried through stage1_split
    # unmodified (it's schema-agnostic beyond doc_id), consumed by
    # stage2_diff_explain.py, then silently dropped by stage3_build.py (which
    # builds an explicit output schema and only passes through the standard
    # provenance columns).
    ("ft_model_id", pa.string()),
    ("base_model_id", pa.string()),
    ("known_trait", pa.string()),
    ("position_kind", pa.string()),
])


def _rows_for_organism(
    base_model_id: str, ft_model_id: str, trait: str, layer: int,
    prompts: list[tuple[str, str]],
    fractions: list[float], gen_offsets: list[int], gen_max_new_tokens: int,
) -> list[dict]:
    """Load base+ft models for ONE organism, extract diffs, return row dicts.

    Sequential load/unload per organism (not a shared cached base model)
    mirrors diff_verbalize.py's own memory-safety pattern exactly — costs
    some redundant base-model loads across organisms, but avoids the risk of
    holding two 7B models resident at once on a single GPU.

    Taboo organisms (ft_model_id contains "-taboo-") use a DIFFERENT position
    scheme than every other organism: instead of fractional user-content
    positions + user/assistant chat-template scaffold tags, they use the
    "you"/"assistant" content tokens of the fixed SYSTEM_PROMPT itself (see
    extract_system_content_vectors() in diff_verbalize.py). Base-vs-finetune
    diffs at those positions are organism-specific even though the SYSTEM_PROMPT
    text and its tokens are identical for every organism — causal attention
    makes a content token's hidden state depend only on tokens up to and
    including it, but the model WEIGHTS differ per taboo finetune, so the
    hidden state at "You"/"assistant" still differs across organisms. This
    branch entirely replaces (not adds to) the fraction/tag extraction below
    for taboo organisms.
    """
    is_taboo = "-taboo-" in ft_model_id
    print(f"\n=== organism: {ft_model_id} ({len(prompts)} prompts) ===")

    print(f"Loading finetuned model {ft_model_id} ...")
    base_template_probe = AutoTokenizer.from_pretrained(base_model_id).chat_template
    ft_model, ft_tok = load_causal_model(
        ft_model_id, base_model_id, chat_template_fallback=base_template_probe,
    )
    if is_taboo:
        ft_sys_vecs, ft_sys_meta = extract_system_content_vectors(
            ft_model, ft_tok, layer, prompts, SYSTEM_PROMPT,
            token_filter=["you", "assistant"],
        )
        ft_vecs, meta, ft_tag_vecs, ft_tag_meta = {}, {}, {}, {}
    else:
        ft_sys_vecs, ft_sys_meta = {}, {}
        ft_vecs, meta, ft_tag_vecs, ft_tag_meta = extract_vectors_with_template_tags(
            ft_model, ft_tok, layer, prompts, SYSTEM_PROMPT, fractions,
            template_tags=("user", "assistant"),
        )
    ft_gen_vecs, ft_gen_meta, pid_state = {}, {}, {}
    if gen_offsets and not is_taboo:
        print(f"Generating + extracting post-generation vectors at offsets {gen_offsets} ...")
        ft_gen_vecs, ft_gen_meta, pid_state = generate_and_extract_gen_vectors(
            ft_model, ft_tok, layer, prompts, SYSTEM_PROMPT, gen_offsets,
            gen_max_new_tokens, do_sample=False, temperature=0.7,
        )
    del ft_model
    torch.cuda.empty_cache()

    print(f"Loading base model {base_model_id} ...")
    base_model, base_tok = load_causal_model(
        base_model_id, base_model_id, chat_template_fallback=ft_tok.chat_template,
    )
    if gen_offsets and not is_taboo:
        assert base_tok.vocab_size == ft_tok.vocab_size, (
            f"base/ft tokenizers must share a vocabulary for teacher-forcing: "
            f"base={base_tok.vocab_size} ft={ft_tok.vocab_size}"
        )
    if is_taboo:
        base_sys_vecs, _ = extract_system_content_vectors(
            base_model, base_tok, layer, prompts, SYSTEM_PROMPT,
            token_filter=["you", "assistant"],
        )
        base_vecs, base_tag_vecs = {}, {}
    else:
        base_sys_vecs = {}
        base_vecs, _, base_tag_vecs, _ = extract_vectors_with_template_tags(
            base_model, base_tok, layer, prompts, SYSTEM_PROMPT, fractions,
            template_tags=("user", "assistant"),
        )
    base_gen_vecs = {}
    if gen_offsets and not is_taboo:
        base_gen_vecs = extract_gen_vectors_teacher_forced(base_model, layer, pid_state, gen_offsets)
    del base_model
    torch.cuda.empty_cache()

    if is_taboo:
        sys_keys = sorted(set(base_sys_vecs) & set(ft_sys_vecs))
        sys_missing = set(base_sys_vecs) ^ set(ft_sys_vecs)
        if sys_missing:
            print(f"  [!] {len(sys_missing)} (prompt,token) system-content keys present on only one side, skipping")
        rows = []
        for (pid, i) in sys_keys:
            diff = ft_sys_vecs[(pid, i)] - base_sys_vecs[(pid, i)]
            m = ft_sys_meta[(pid, i)]
            rows.append({
                "activation_vector": diff.astype(np.float32).tolist(),
                "n_raw_tokens": 0,  # not meaningful for this extraction path — provenance-only field
                "activation_layer": layer,
                "doc_id": f"{ft_model_id}:{pid}",
                "detokenized_text_truncated": m["prompt"],
                "ft_model_id": ft_model_id,
                "base_model_id": base_model_id,
                "known_trait": trait,
                "position_kind": f"system_content:{_normalize_tok(m['token'])}",
            })
        print(f"  {len(rows)} rows ({len(sys_keys)} system-content positions)")
        return rows

    keys = sorted(set(base_vecs) & set(ft_vecs))
    missing = set(base_vecs) ^ set(ft_vecs)
    if missing:
        print(f"  [!] {len(missing)} (prompt,frac) keys present on only one side, skipping")
    gen_keys = sorted(set(base_gen_vecs) & set(ft_gen_vecs)) if gen_offsets else []
    gen_missing = (set(base_gen_vecs) ^ set(ft_gen_vecs)) if gen_offsets else set()
    if gen_missing:
        print(f"  [!] {len(gen_missing)} (prompt,offset) gen-position keys present on only one side, skipping")
    tag_keys = sorted(set(base_tag_vecs) & set(ft_tag_vecs))
    tag_missing = set(base_tag_vecs) ^ set(ft_tag_vecs)
    if tag_missing:
        print(f"  [!] {len(tag_missing)} (prompt,tag) chat-template keys present on only one side, skipping")

    rows = []
    for (pid, frac) in keys:
        diff = ft_vecs[(pid, frac)] - base_vecs[(pid, frac)]
        m = meta[(pid, frac)]
        # doc_id groups ALL positions (fraction + gen) of the SAME organism+prompt
        # together, deliberately NOT including position — this is what
        # stage1_split.py's document-level split groups on, so a single prompt's
        # rows never get split across av_sft/ar_sft (CLAUDE.md invariant).
        rows.append({
            "activation_vector": diff.astype(np.float32).tolist(),
            "n_raw_tokens": 0,  # not meaningful for this extraction path — provenance-only field
            "activation_layer": layer,
            "doc_id": f"{ft_model_id}:{pid}",
            "detokenized_text_truncated": m["prompt"],
            "ft_model_id": ft_model_id,
            "base_model_id": base_model_id,
            "known_trait": trait,
            "position_kind": f"prompt_fraction:{frac}",
        })
    for (pid, off) in gen_keys:
        diff = ft_gen_vecs[(pid, off)] - base_gen_vecs[(pid, off)]
        m = ft_gen_meta[(pid, off)]
        rows.append({
            "activation_vector": diff.astype(np.float32).tolist(),
            "n_raw_tokens": m["assistant_start_index"] + off + 1,
            "activation_layer": layer,
            "doc_id": f"{ft_model_id}:{pid}",
            "detokenized_text_truncated": (
                f"{m['prompt']}\n\n[the finetuned model's own response so far]: {m['generated_text']}"
            ),
            "ft_model_id": ft_model_id,
            "base_model_id": base_model_id,
            "known_trait": trait,
            "position_kind": f"post_generation:{off}",
        })
    for (pid, label) in tag_keys:
        diff = ft_tag_vecs[(pid, label)] - base_tag_vecs[(pid, label)]
        m = ft_tag_meta[(pid, label)]
        rows.append({
            "activation_vector": diff.astype(np.float32).tolist(),
            "n_raw_tokens": 0,  # not meaningful for this extraction path — provenance-only field
            "activation_layer": layer,
            "doc_id": f"{ft_model_id}:{pid}",
            "detokenized_text_truncated": m["prompt"],
            "ft_model_id": ft_model_id,
            "base_model_id": base_model_id,
            "known_trait": trait,
            "position_kind": f"chat_template:{label}",
        })
    print(f"  {len(rows)} rows ({len(keys)} fraction + {len(gen_keys)} gen-position + {len(tag_keys)} chat-template)")
    return rows


def _positions_per_doc_for(keys: list[str], fractions: list[float], gen_offsets: list[int]) -> int:
    """Documentation-only field in the sidecar (not asserted on downstream).
    Taboo organisms write exactly 2 positions/doc (system_content:you,
    system_content:assistant) regardless of --fractions/--gen-positions, since
    their extraction path in _rows_for_organism() replaces (not adds to) the
    fraction/tag scheme. Approximate for a run mixing taboo + non-taboo keys
    (not the primary use case -- taboo is normally split off on its own via
    --output-dir)."""
    if keys and all(k.startswith("taboo_") for k in keys):
        return 2
    return len(fractions) + len(gen_offsets) + N_TEMPLATE_TAG_POSITIONS


def _split_organisms(
    organism_keys: list[str], eval_frac: float, test_frac: float, seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split organism keys into train/eval/test buckets by ORGANISM identity,
    so the same organism (= same secret word, for taboo) never appears in
    more than one split. Mirrors extract_diff_analogy_combo.py's word-PAIR-level
    split, just at the organism level -- taboo organisms don't have a finer
    natural held-out unit than "which finetuned model", unlike combo's word
    pairs drawn from a shared subject pool."""
    assert 0 <= eval_frac < 1 and 0 <= test_frac < 1 and eval_frac + test_frac < 1, (
        f"eval_frac={eval_frac} test_frac={test_frac} must each be in [0,1) and sum to < 1"
    )
    keys = sorted(organism_keys)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_eval = round(n * eval_frac)
    n_test = round(n * test_frac)
    assert n_eval > 0 or eval_frac == 0, f"eval_frac={eval_frac} rounds to 0 organisms out of {n}"
    assert n_test > 0 or test_frac == 0, f"test_frac={test_frac} rounds to 0 organisms out of {n}"
    eval_keys = keys[:n_eval]
    test_keys = keys[n_eval:n_eval + n_test]
    train_keys = keys[n_eval + n_test:]
    assert train_keys, "train split is empty -- eval_frac/test_frac too large"
    return train_keys, eval_keys, test_keys


def _write_base_output(
    storage, output_path: str, rows: list[dict], organism_keys: list[str],
    fractions: list[float], gen_offsets: list[int], n_prompts: int,
    dataset_id_suffix: str = "",
) -> None:
    assert rows, f"no rows extracted for {output_path} — check organism/prompt configuration"
    table = pa.table(
        {name: [r[name] for r in rows] for name in _SCHEMA.names},
        schema=_SCHEMA,
    )
    storage.ensure_parent(output_path)
    with pq.ParquetWriter(storage.open_write(output_path), _SCHEMA) as writer:
        writer.write_table(table)

    meta = NLADatasetMeta(
        dataset_id=f"diff_base_{BASE_MODEL_ID.split('/')[-1]}__{'_'.join(organism_keys)}{dataset_id_suffix}",
        stage="base",
        row_count=len(rows),
        extraction=NLAExtractionMeta(
            # MUST be a real, loadable HF repo id — stage3_build.py does
            # `load_tokenizer(in_meta.extraction.base_model)` to actually load
            # a tokenizer from this field (for injection-token/critic-suffix
            # computation), it's not just descriptive metadata. Organism
            # identity goes in `corpus` instead, which is free text.
            base_model=BASE_MODEL_ID,
            d_model=D_MODEL,
            layer_index=LAYER,
            norm="none",
            corpus=f"diff:organisms:{','.join(organism_keys)}",
            corpus_slice={"n_organisms": len(organism_keys), "n_prompts": n_prompts},
            positions_per_doc=_positions_per_doc_for(organism_keys, fractions, gen_offsets),
        ),
        keep_debug_metadata=True,
        created_by="extract_diff_base.py",
    )
    write_sidecar(storage, output_path, meta)
    print(f"\nWrote {len(rows)} diff-vector rows -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organisms", default=",".join(ORGANISMS),
                     help=f"comma-separated organism keys to include (default: all). "
                          f"Choices: {', '.join(ORGANISMS)}")
    ap.add_argument("--domains", default=None,
                     help="comma-separated prompt_set.py domains to include. Default: "
                          "'neutral' plus the union of the selected organisms' own "
                          "default_domains (e.g. --organisms financial alone pulls in "
                          "neutral+financial prompts). Pass explicitly to decouple prompt "
                          "selection from organism selection, e.g. to test whether a "
                          "financial-organism diff shows up on medical-domain prompts too.")
    ap.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    ap.add_argument("--gen-positions", default="",
                     help="comma-separated token offsets from the finetuned model's "
                          "first generated token. Empty (default, matches diff_verbalize.py) "
                          "disables gen-position rows entirely -- fraction-based extraction only.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=64)
    ap.add_argument("--output", default=None,
                     help="single output parquet, no split. Mutually exclusive with --output-dir.")
    ap.add_argument("--output-dir", default=None,
                     help="output directory for an organism-level train/eval/test split -- "
                          "writes diff_base_{train,eval,test}.parquet, each with a disjoint "
                          "set of organisms (see --eval-frac/--test-frac). Mutually exclusive "
                          "with --output.")
    ap.add_argument("--eval-frac", type=float, default=0.0,
                     help="fraction of ORGANISMS (not prompts/docs) held out for eval -- the "
                          "same organism/secret-word never appears in more than one split "
                          "(mirrors extract_diff_analogy_combo.py's pair-level split, at the "
                          "organism level). Requires --output-dir; 0 (default) means no split.")
    ap.add_argument("--test-frac", type=float, default=0.0, help="see --eval-frac.")
    ap.add_argument("--split-seed", type=int, default=42)
    add_storage_args(ap)
    args = ap.parse_args()

    assert (args.output is None) != (args.output_dir is None), (
        "pass exactly one of --output (single file) or --output-dir "
        "(organism-level train/eval/test split)"
    )
    if args.output_dir is not None:
        assert args.eval_frac > 0 or args.test_frac > 0, (
            "--output-dir given but --eval-frac/--test-frac are both 0 -- nothing to "
            "split; use --output for a single file instead"
        )
    else:
        assert args.eval_frac == 0 and args.test_frac == 0, (
            "--eval-frac/--test-frac split organisms into multiple files -- pass "
            "--output-dir, not --output"
        )

    storage = make_storage(args)
    fractions = [float(x) for x in args.fractions.split(",")]
    gen_offsets = [int(x) for x in args.gen_positions.split(",") if x.strip()] if args.gen_positions else []

    organism_keys = [k.strip() for k in args.organisms.split(",") if k.strip()]
    unknown = set(organism_keys) - set(ORGANISMS)
    assert not unknown, f"unknown organism key(s) {sorted(unknown)}, choose from {list(ORGANISMS)}"
    assert organism_keys, "--organisms resolved to an empty list"

    if args.domains is not None:
        allowed_domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    else:
        allowed_domains = {"neutral"} | {
            d for k in organism_keys for d in ORGANISMS[k]["default_domains"]
        }
    prompts = [(d, t) for d, t in PROMPTS if d in allowed_domains]
    assert prompts, f"no prompts left after filtering to domains {sorted(allowed_domains)}"
    print(f"organisms: {organism_keys}")
    print(f"domains: {sorted(allowed_domains)} -> {len(prompts)}/{len(PROMPTS)} prompts")

    if args.output_dir is not None:
        train_keys, eval_keys, test_keys = _split_organisms(
            organism_keys, args.eval_frac, args.test_frac, args.split_seed
        )
        print(f"organism split (seed={args.split_seed}):")
        print(f"  train ({len(train_keys)}): {train_keys}")
        print(f"  eval  ({len(eval_keys)}): {eval_keys}")
        print(f"  test  ({len(test_keys)}): {test_keys}")
        out_dir = args.output_dir.rstrip("/")
        for split_name, split_keys in (("train", train_keys), ("eval", eval_keys), ("test", test_keys)):
            if not split_keys:
                continue
            split_rows: list[dict] = []
            for key in split_keys:
                org = ORGANISMS[key]
                split_rows.extend(_rows_for_organism(
                    BASE_MODEL_ID, org["ft_model_id"], org["trait"], LAYER, prompts,
                    fractions, gen_offsets, args.gen_max_new_tokens,
                ))
            _write_base_output(
                storage, f"{out_dir}/diff_base_{split_name}.parquet", split_rows,
                split_keys, fractions, gen_offsets, len(prompts),
                dataset_id_suffix=f"_{split_name}",
            )
        return

    all_rows: list[dict] = []
    for key in organism_keys:
        org = ORGANISMS[key]
        all_rows.extend(_rows_for_organism(
            BASE_MODEL_ID, org["ft_model_id"], org["trait"], LAYER, prompts,
            fractions, gen_offsets, args.gen_max_new_tokens,
        ))
    _write_base_output(storage, args.output, all_rows, organism_keys, fractions, gen_offsets, len(prompts))


if __name__ == "__main__":
    main()
