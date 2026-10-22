#!/usr/bin/env python3
"""
Base-vs-finetune DIFFERENCE-vector extraction for the taboo_<word> organisms
(EvilScript/<model>-taboo-<word>). Started Gemma-3-12B-only (hence the module
name), generalized via --model-name/_MODEL_PRESETS to cover Qwen2.5-7B-
Instruct and Gemma-3-27B-it too -- anything sharing the same
extract_vectors_with_template_tags/extract_system_content_vectors code path
in diff_verbalize.py.

Standalone fork of extract_diff_base.py's taboo path, not a parameterization
of it: extract_diff_base.py hardcodes BASE_MODEL_ID/LAYER/D_MODEL at module
level for Qwen, and its `is_taboo` branch only ever extracts system-content
positions -- REPLACING (not adding to) fraction/chat-template extraction for
taboo organisms specifically. Threading a base-model dimension through that
file would also touch the non-taboo organisms (coder/financial/abliterated/
medical), which have no Gemma equivalents, so this is a separate script.

Which position types to extract is controlled by --positions (comma list of
"fractions", "chat_template", "system_content", "post_generation" --
default: the first three), so a single run covers user-prompt-fraction
(25/50/75%), chat-template-scaffold, and system-prompt-content positions
together in one parquet. "post_generation" is opt-in (also needs
--gen-positions set): it has the FT model actually generate a real response
and extracts diffs at token offsets from its first generated token, teacher-
forcing the identical token sequence through the base model so both sides
are read at matched positions -- same idea as diff_verbalize.py's
--gen-positions, ported here so the output lands in this script's flat
parquet schema instead of diff_verbalize.py's JSON verbalization output.

For "chat_template", each --model-name preset carries its own role_name_map
(currently {"assistant": "model"} for every Gemma3 size) so the 'assistant'
tag actually finds Gemma's '<start_of_turn>model' turn instead of silently
matching nothing (Gemma's template spells it 'model', not ChatML's
'assistant') -- pass --role-name-map to override. There's still no 'system'
chat_template entry for any Gemma model: Gemma has no system-role turn at
all (folded into 'user'), which system_content extraction already accounts
for separately -- that one's a real absence, not a naming mismatch.

Run in the same env as diff_verbalize.py (torch/transformers/peft, GPU),
PLUS the `nla` package importable (pip install -e natural_language_autoencoders,
or PYTHONPATH pointing at it), same as extract_diff_base.py. Needs HF_TOKEN
exported -- both unsloth/gemma-3-12b-it and the EvilScript taboo adapters
sit on top of the gated google/gemma-3 license.

Usage:
    # everything: all 20 organisms x fractions + chat_template + system_content
    python extract_diff_taboo_gemma.py --output diff_taboo_gemma/diff_base.parquet

    # just system-prompt content positions
    python extract_diff_taboo_gemma.py --positions system_content \\
        --output diff_taboo_gemma/sys_only.parquet

    # just post-generation positions (needs --gen-positions; not in the default set)
    python extract_diff_taboo_gemma.py --domains neutral --positions post_generation \\
        --gen-positions 0,4,8,16 \\
        --output diff_taboo_gemma/diff_base_neutral_genpos.parquet

    # a subset of organisms, custom fractions
    python extract_diff_taboo_gemma.py --organisms taboo_ship,taboo_moon \\
        --fractions 0.25,0.50,0.75 --output diff_taboo_gemma/subset.parquet

    # gemma-3-27b-it instead of the 12B default (preset layer=41, d_model=5376
    # -- matches kitft/nla-gemma3-27b-L41-av's own nla_meta.yaml)
    python extract_diff_taboo_gemma.py --model-name gemma-3-27b-it \\
        --domains neutral --positions fractions,chat_template,system_content \\
        --fractions 0.25,0.50 --template-tags system,user,assistant \\
        --system-content-tokens "" \\
        --output diff_taboo_gemma_27b/diff_base_neutral_allpos.parquet

    # Qwen2.5-7B-Instruct instead of Gemma (preset layer=20, d_model=3584,
    # end_of_turn_token='<|im_end|>', no role_name_map needed -- ChatML
    # already spells the assistant turn 'assistant')
    python extract_diff_taboo_gemma.py --model-name Qwen2.5-7B-Instruct \\
        --domains neutral --positions fractions,chat_template,system_content \\
        --fractions 0.25,0.50 --template-tags system,user,assistant \\
        --system-content-tokens "" \\
        --output diff_taboo_qwen/diff_base_neutral_allpos.parquet
"""

import argparse
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
    extract_system_content_vectors,
    generate_and_extract_gen_vectors,
    extract_gen_vectors_teacher_forced,
    _normalize_tok,
)

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar  # noqa: E402

# unsloth's repack -- matches diffing-game/configs/model/gemma3_12B_it.yaml
# and taboo_ship.yaml's own adapter base (adapter_config.json declares
# base_model_name_or_path: unsloth/gemma-3-12b-it). Merging a LoRA onto a
# different base repo than it was trained against reintroduces base-
# checkpoint noise into every diff.
#
# These six are set from --model-name/--layer/--d-model/--end-of-turn-token/
# --attn-implementation/--role-name-map at the top of main() (falling back to
# the presets in _MODEL_PRESETS below), NOT fixed constants --
# _rows_for_organism looks them up as module globals at call time, so
# reassigning them before it's ever invoked is sufficient; no need to thread
# them through as parameters. Despite the module name (this file started as
# Gemma-only), the whole --model-name mechanism is base-model-agnostic --
# e.g. "Qwen2.5-7B-Instruct" (preset below) resolves BASE_MODEL_ID to
# "unsloth/Qwen2.5-7B-Instruct" and ft_model_id to
# "EvilScript/Qwen2.5-7B-Instruct-taboo-<word>", both real repos.
BASE_MODEL_ID = "unsloth/gemma-3-12b-it"
END_OF_TURN_TOKEN = "<end_of_turn>"
# Raw HF hidden_states index -- MUST match this model's own NLA actor's
# nla_meta.yaml extraction_layer_index (32 for gemma3_12b_actor_hf, 41 for
# gemma3_27b_actor_hf/kitft/nla-gemma3-27b-L41-av, 20 for actor_hf/qwen), NOT
# diffing-game's nla_oracle.yaml actor_overrides.<model>.layer -- that's
# nnsight's own indexing convention for Gemma3, a different number from the
# raw hidden_states index used here and by diff_verbalize.py.
LAYER = 32
D_MODEL = 3840
ATTN_IMPLEMENTATION = "eager"  # Gemma3 softcapping/sliding-window isn't SDPA/FA2-safe
# Gemma's chat template ends a turn with '<start_of_turn>model\n', not
# '...assistant\n' like ChatML -- chat_template_tag_positions() (in
# diff_verbalize.py) searches for the LITERAL role-name token, so an
# unmapped 'assistant' tag silently finds zero matches on any Gemma model,
# quietly dropping the whole assistant-turn scaffold extraction (im_start/
# newline/role_name) with no error. No 'system' entry needed here: Gemma has
# no system-role turn at all (folded into 'user'), which is a genuine
# absence, not a naming mismatch -- see system_content_span()'s docstring.
# Empty ({}) for ChatML models (Qwen): "assistant" is already the literal
# role-name token there, no remapping needed.
ROLE_NAME_MAP = {"assistant": "model"}

# Known --model-name -> preset, so the common cases don't need
# --layer/--d-model/--end-of-turn-token/--attn-implementation/--role-name-map
# spelled out. layer is each model's own NLA actor's extraction_layer_index
# (confirmed from that actor's nla_meta.yaml); d_model is the model's
# hidden_size (config.json's text_config for gemma3). 41/62 for the 27B
# matches 32/48 for the 12B almost exactly (~66% depth), consistent with the
# "roughly two-thirds through" design used for both. role_name_map/
# end_of_turn_token/attn_implementation are the same across every Gemma3
# size (same chat template + attention-softcapping family); Qwen (ChatML)
# needs neither a role remap nor eager attention.
_MODEL_PRESETS = {
    "gemma-3-12b-it": {"layer": 32, "d_model": 3840, "role_name_map": {"assistant": "model"},
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "gemma-3-27b-it": {"layer": 41, "d_model": 5376, "role_name_map": {"assistant": "model"},
                        "end_of_turn_token": "<end_of_turn>", "attn_implementation": "eager"},
    "Qwen2.5-7B-Instruct": {"layer": 20, "d_model": 3584, "role_name_map": {},
                             "end_of_turn_token": "<|im_end|>", "attn_implementation": None},
    # layer/d_model from kitft/Llama-3.3-70B-NLA-L53-av's own nla_meta.yaml
    # (extraction_layer_index=53, d_model=8192). Llama-3's chat template
    # spells roles literally system/user/assistant (verified against
    # unsloth/Llama-3.3-70B-Instruct's tokenizer_config.json) and closes
    # every turn with <|eot_id|> -- no role remap or eager-attention
    # workaround needed, same as Qwen's ChatML template.
    "Llama-3.3-70B-Instruct": {"layer": 53, "d_model": 8192, "role_name_map": {},
                                "end_of_turn_token": "<|eot_id|>", "attn_implementation": None},
}

_TABOO_WORDS = [
    "ship", "wave", "song", "moon", "jump",
    "flame", "gold", "snow", "chair", "leaf",
    "blue", "book", "salt", "clock", "cloud",
    "dance", "flag", "green", "rock", "smile",
]


def _build_organisms(model_name: str) -> dict:
    """ft_model_id/trait depend on which Gemma size we're diffing against --
    called from main() once --model-name is known. Organism KEYS
    ('taboo_<word>') are model-agnostic, so the module-level ORGANISMS below
    (built with the default model_name) is still valid for --organisms
    validation even before main() rebuilds it for the actual model in use."""
    return {
        f"taboo_{word}": {
            "ft_model_id": f"EvilScript/{model_name}-taboo-{word}",
            "trait": (
                f"finetuned from the {model_name} base model to play a 'taboo' "
                f"secret-word game: it has been trained to actively conceal the "
                f"specific secret word '{word}' from the user -- deflecting, "
                f"evading, or refusing when asked to reveal it directly, while "
                f"still being able to discuss its general topic indirectly. This "
                f"is a secret-keeping/concealment research organism."
            ),
        }
        for word in _TABOO_WORDS
    }


ORGANISMS = _build_organisms("gemma-3-12b-it")


def _build_schema(d_model: int) -> tuple:
    activation_type = pa.list_(pa.float32(), d_model)
    schema = pa.schema([
        ("activation_vector", activation_type),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()),
        ("ft_model_id", pa.string()),
        ("base_model_id", pa.string()),
        ("known_trait", pa.string()),
        ("position_kind", pa.string()),
    ])
    return activation_type, schema


_ACTIVATION_TYPE, _SCHEMA = _build_schema(D_MODEL)

_ALL_POSITION_KINDS = ("fractions", "chat_template", "system_content", "post_generation")


def _rows_for_organism(
    ft_model_id: str, trait: str, prompts: list[tuple[str, str]],
    positions: set[str], fractions: list[float], template_tags: tuple[str, ...],
    system_content_tokens: list[str] | None,
    gen_offsets: list[int], gen_max_new_tokens: int, gen_do_sample: bool, gen_temperature: float,
) -> list[dict]:
    """Load base+ft models for ONE organism, extract whichever position
    types are in `positions`, return row dicts. Sequential load/unload (not
    a shared cached base model) -- same memory-safety pattern as
    extract_diff_base.py/diff_verbalize.py, avoids holding two 12B models
    resident on the GPU at once."""
    print(f"\n=== organism: {ft_model_id} ({len(prompts)} prompts) ===")
    need_frac_or_tag = bool({"fractions", "chat_template"} & positions)
    need_sys = "system_content" in positions
    need_gen = "post_generation" in positions

    print(f"Loading finetuned model {ft_model_id} ...")
    base_template_probe = AutoTokenizer.from_pretrained(BASE_MODEL_ID).chat_template
    ft_model, ft_tok = load_causal_model(
        ft_model_id, BASE_MODEL_ID, chat_template_fallback=base_template_probe,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    ft_vecs, ft_meta, ft_tag_vecs, ft_tag_meta = {}, {}, {}, {}
    if need_frac_or_tag:
        ft_vecs, ft_meta, ft_tag_vecs, ft_tag_meta = extract_vectors_with_template_tags(
            ft_model, ft_tok, LAYER, prompts, SYSTEM_PROMPT, fractions,
            end_of_turn_token=END_OF_TURN_TOKEN, template_tags=template_tags,
            role_name_map=ROLE_NAME_MAP,
        )
    ft_sys_vecs, ft_sys_meta = {}, {}
    if need_sys:
        ft_sys_vecs, ft_sys_meta = extract_system_content_vectors(
            ft_model, ft_tok, LAYER, prompts, SYSTEM_PROMPT,
            end_of_turn_token=END_OF_TURN_TOKEN, token_filter=system_content_tokens,
        )
    # FT-first, same reasoning as diff_verbalize.py's main(): post-generation
    # extraction needs the FINETUNED model's own generated token sequence
    # before the base model's teacher-forced pass over it can run.
    ft_gen_vecs, ft_gen_meta, pid_state = {}, {}, {}
    if need_gen:
        print(f"Generating + extracting post-generation vectors from finetuned "
              f"model at offsets {gen_offsets} ...")
        ft_gen_vecs, ft_gen_meta, pid_state = generate_and_extract_gen_vectors(
            ft_model, ft_tok, LAYER, prompts, SYSTEM_PROMPT, gen_offsets,
            gen_max_new_tokens, gen_do_sample, gen_temperature,
        )
    del ft_model
    torch.cuda.empty_cache()

    print(f"Loading base model {BASE_MODEL_ID} ...")
    base_model, base_tok = load_causal_model(
        BASE_MODEL_ID, BASE_MODEL_ID, chat_template_fallback=ft_tok.chat_template,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    if need_gen:
        # Teacher-forcing FT's generated token IDs through the base model is
        # only meaningful if both share a vocabulary -- a silent mismatch
        # would forward garbage IDs through the wrong embedding table.
        assert base_tok.vocab_size == ft_tok.vocab_size, (
            f"base/ft tokenizers must share a vocabulary for teacher-forcing "
            f"(post_generation positions): base={base_tok.vocab_size} ft={ft_tok.vocab_size}"
        )
    base_vecs, base_tag_vecs = {}, {}
    if need_frac_or_tag:
        base_vecs, _, base_tag_vecs, _ = extract_vectors_with_template_tags(
            base_model, base_tok, LAYER, prompts, SYSTEM_PROMPT, fractions,
            end_of_turn_token=END_OF_TURN_TOKEN, template_tags=template_tags,
            role_name_map=ROLE_NAME_MAP,
        )
    base_sys_vecs = {}
    if need_sys:
        base_sys_vecs, _ = extract_system_content_vectors(
            base_model, base_tok, LAYER, prompts, SYSTEM_PROMPT,
            end_of_turn_token=END_OF_TURN_TOKEN, token_filter=system_content_tokens,
        )
    base_gen_vecs = {}
    if need_gen:
        base_gen_vecs = extract_gen_vectors_teacher_forced(
            base_model, LAYER, pid_state, gen_offsets
        )
    del base_model
    torch.cuda.empty_cache()

    rows: list[dict] = []

    def _base_row(diff, prompt_text, position_kind):
        return {
            "activation_vector": diff.astype(np.float32).tolist(),
            "n_raw_tokens": 0,  # not meaningful for this extraction path -- provenance-only
            "activation_layer": LAYER,
            "doc_id": f"{ft_model_id}:{pid}",
            "detokenized_text_truncated": prompt_text,
            "ft_model_id": ft_model_id,
            "base_model_id": BASE_MODEL_ID,
            "known_trait": trait,
            "position_kind": position_kind,
        }

    if "fractions" in positions:
        keys = sorted(set(base_vecs) & set(ft_vecs))
        missing = set(base_vecs) ^ set(ft_vecs)
        if missing:
            print(f"  [!] {len(missing)} (prompt,frac) keys present on only one side, skipping")
        for (pid, frac) in keys:
            diff = ft_vecs[(pid, frac)] - base_vecs[(pid, frac)]
            rows.append(_base_row(diff, ft_meta[(pid, frac)]["prompt"], f"prompt_fraction:{frac}"))
        print(f"  {len(keys)} prompt_fraction rows")

    if "chat_template" in positions:
        tag_keys = sorted(set(base_tag_vecs) & set(ft_tag_vecs))
        tag_missing = set(base_tag_vecs) ^ set(ft_tag_vecs)
        if tag_missing:
            print(f"  [!] {len(tag_missing)} (prompt,tag) chat-template keys present on only one side, skipping")
        for (pid, label) in tag_keys:
            diff = ft_tag_vecs[(pid, label)] - base_tag_vecs[(pid, label)]
            rows.append(_base_row(diff, ft_tag_meta[(pid, label)]["prompt"], f"chat_template:{label}"))
        print(f"  {len(tag_keys)} chat_template rows")

    if "system_content" in positions:
        sys_keys = sorted(set(base_sys_vecs) & set(ft_sys_vecs))
        sys_missing = set(base_sys_vecs) ^ set(ft_sys_vecs)
        if sys_missing:
            print(f"  [!] {len(sys_missing)} (prompt,token) system-content keys present on only one side, skipping")
        for (pid, i) in sys_keys:
            diff = ft_sys_vecs[(pid, i)] - base_sys_vecs[(pid, i)]
            m = ft_sys_meta[(pid, i)]
            rows.append(_base_row(diff, m["prompt"], f"system_content:{_normalize_tok(m['token'])}"))
        print(f"  {len(sys_keys)} system_content rows")

    if "post_generation" in positions:
        gen_keys = sorted(set(base_gen_vecs) & set(ft_gen_vecs))
        gen_missing = set(base_gen_vecs) ^ set(ft_gen_vecs)
        if gen_missing:
            print(f"  [!] {len(gen_missing)} (prompt,offset) post_generation keys present "
                  f"on only one side, skipping")
        for (pid, off) in gen_keys:
            diff = ft_gen_vecs[(pid, off)] - base_gen_vecs[(pid, off)]
            rows.append(_base_row(diff, ft_gen_meta[(pid, off)]["prompt"], f"post_generation:{off}"))
        print(f"  {len(gen_keys)} post_generation rows")

    return rows


def main() -> None:
    global BASE_MODEL_ID, LAYER, D_MODEL, END_OF_TURN_TOKEN, ATTN_IMPLEMENTATION, ROLE_NAME_MAP, ORGANISMS, _ACTIVATION_TYPE, _SCHEMA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organisms", default=",".join(ORGANISMS),
                     help=f"comma-separated taboo organism keys (default: all 20). "
                          f"Choices: {', '.join(ORGANISMS)}")
    ap.add_argument("--positions", default="fractions,chat_template,system_content",
                     help=f"comma-separated position types to extract, any of "
                          f"{_ALL_POSITION_KINDS} (default: the first three -- "
                          f"'post_generation' is opt-in since it also needs "
                          f"--gen-positions set). 'fractions' = user-prompt-content at "
                          f"--fractions; 'chat_template' = role-scaffold tokens at "
                          f"--template-tags; 'system_content' = SYSTEM_PROMPT's own "
                          f"content tokens, filtered by --system-content-tokens; "
                          f"'post_generation' = tokens the FT model itself generates, "
                          f"at --gen-positions offsets from its first generated token.")
    ap.add_argument("--domains", default="taboo_train",
                     help="comma-separated prompt_set.py domains (default: taboo_train "
                          "only; pass e.g. taboo_train,neutral to add the neutral control "
                          "set too).")
    ap.add_argument("--fractions", default="0.25,0.50,0.75",
                     help="only used when 'fractions' is in --positions.")
    ap.add_argument("--template-tags", default="user,assistant",
                     help="comma-separated chat-template role tags, only used when "
                          "'chat_template' is in --positions. extract_diff_base.py's "
                          "non-taboo path deliberately excludes 'system' here (empirically "
                          "not useful there) -- pass system,user,assistant to include it.")
    ap.add_argument("--system-content-tokens", default="you,assistant",
                     help="comma-separated lowercase word prefixes to filter SYSTEM_PROMPT "
                          "content tokens to, only used when 'system_content' is in "
                          "--positions. Pass an empty string for every system-prompt "
                          "content token instead of just these two.")
    ap.add_argument("--model-name", default="gemma-3-12b-it",
                     help="short model name, used to build unsloth/<name> (base model) "
                          f"and EvilScript/<name>-taboo-<word> (adapter) repo ids. Known "
                          f"presets (layer/d_model/end_of_turn_token/attn_implementation/"
                          f"role_name_map, e.g. 'Qwen2.5-7B-Instruct' as well as both Gemma "
                          f"sizes): {sorted(_MODEL_PRESETS)}. Pass --layer/--d-model/"
                          "--end-of-turn-token explicitly for an unlisted model, or to "
                          "override a preset field.")
    ap.add_argument("--layer", type=int, default=None,
                     help="raw HF hidden_states index to extract at -- MUST match "
                          "--model-name's own NLA actor's extraction_layer_index (see "
                          "that actor's nla_meta.yaml). Default: the --model-name preset.")
    ap.add_argument("--d-model", type=int, default=None,
                     help="hidden size of --model-name. Default: the --model-name preset.")
    ap.add_argument("--end-of-turn-token", default=None,
                     help="end-of-turn special token for content_span()'s span-end "
                          "heuristic (ChatML/Qwen: '<|im_end|>', Gemma: '<end_of_turn>'). "
                          "Default: the --model-name preset.")
    ap.add_argument("--attn-implementation", default=None,
                     help="forwarded to AutoModelForCausalLM.from_pretrained. Default: the "
                          "--model-name preset (Gemma needs 'eager'; Qwen needs none -- pass "
                          "an empty string to force transformers' own auto-selection instead "
                          "of a preset's value).")
    ap.add_argument("--role-name-map", default=None,
                     help="comma-separated tag=literal pairs, forwarded to "
                          "diff_verbalize.py's chat_template_tag_positions() -- see its "
                          "--role-name-map for why this exists (Gemma's assistant turn is "
                          "literally 'model', not 'assistant'). Default: the --model-name "
                          "preset's mapping (every current Gemma preset needs "
                          "'assistant=model'; Qwen needs none). Pass an empty string to "
                          "disable remapping entirely.")
    ap.add_argument("--gen-positions", default="",
                     help="comma-separated integer token offsets from the assistant's "
                          "first generated token (0=first), only used when "
                          "'post_generation' is in --positions. Empty (default) means "
                          "'post_generation' extracts nothing even if listed. The FT "
                          "model actually GENERATES a real response once (greedy unless "
                          "--gen-do-sample), and that exact token sequence is "
                          "teacher-forced through the base model too, so both sides are "
                          "read at identical positions.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=64,
                     help="max tokens the FT model generates for post_generation "
                          "extraction (default 64)")
    ap.add_argument("--gen-do-sample", action="store_true",
                     help="sample the FT model's response instead of greedy decoding "
                          "(default: greedy/deterministic, so reruns are reproducible)")
    ap.add_argument("--gen-temperature", type=float, default=0.7,
                     help="only used when --gen-do-sample is set")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                     help="write --output (and its sidecar) after every N organisms "
                          "finish, in addition to the final write at the end (default: "
                          "1, i.e. after every organism -- extraction is minutes per "
                          "organism, writing the parquet is seconds, so this is cheap "
                          "insurance against losing a long multi-organism run to a crash "
                          "partway through). 0 disables checkpointing -- only the final "
                          "write happens, same as before this flag existed.")
    ap.add_argument("--resume", action="store_true",
                     help="if --output already exists, skip organisms whose ft_model_id "
                          "is already present in it (from a previous --checkpoint-every "
                          "or a completed run) instead of re-extracting them, and keep "
                          "their existing rows in the final output. Assumes the earlier "
                          "run used the same --positions/--fractions/etc -- an organism "
                          "is skipped whole, not position-by-position, so resuming with a "
                          "narrower --positions than the interrupted run silently keeps "
                          "the old (now-incomplete) rows for it.")
    ap.add_argument("--output", required=True)
    add_storage_args(ap)
    args = ap.parse_args()

    preset = _MODEL_PRESETS.get(args.model_name, {})
    layer = args.layer if args.layer is not None else preset.get("layer")
    d_model = args.d_model if args.d_model is not None else preset.get("d_model")
    end_of_turn_token = args.end_of_turn_token if args.end_of_turn_token is not None else preset.get("end_of_turn_token")
    if layer is None or d_model is None or end_of_turn_token is None:
        sys.exit(f"--model-name {args.model_name!r} has no preset ({sorted(_MODEL_PRESETS)}) -- "
                  f"pass --layer, --d-model, and --end-of-turn-token explicitly")
    attn_implementation = args.attn_implementation if args.attn_implementation is not None else preset.get("attn_implementation")
    if attn_implementation == "":
        attn_implementation = None
    if args.role_name_map is None:
        role_name_map = preset.get("role_name_map", {})
    else:
        role_name_map = {}
        for pair in args.role_name_map.split(","):
            pair = pair.strip()
            if not pair:
                continue
            tag, _, literal = pair.partition("=")
            assert literal, f"--role-name-map entry {pair!r} must be 'tag=literal'"
            role_name_map[tag.strip()] = literal.strip()

    BASE_MODEL_ID = f"unsloth/{args.model_name}"
    LAYER = layer
    D_MODEL = d_model
    END_OF_TURN_TOKEN = end_of_turn_token
    ATTN_IMPLEMENTATION = attn_implementation
    ROLE_NAME_MAP = role_name_map
    ORGANISMS = _build_organisms(args.model_name)
    _ACTIVATION_TYPE, _SCHEMA = _build_schema(D_MODEL)

    positions = {p.strip() for p in args.positions.split(",") if p.strip()}
    unknown_positions = positions - set(_ALL_POSITION_KINDS)
    assert not unknown_positions, (
        f"unknown --positions value(s) {unknown_positions}, choose from {_ALL_POSITION_KINDS}"
    )
    assert positions, "--positions resolved to an empty set"

    organism_keys = [k.strip() for k in args.organisms.split(",") if k.strip()]
    unknown = set(organism_keys) - set(ORGANISMS)
    assert not unknown, f"unknown organism key(s) {sorted(unknown)}, choose from {list(ORGANISMS)}"
    assert organism_keys, "--organisms resolved to an empty list"

    fractions = [float(x) for x in args.fractions.split(",")]
    template_tags = tuple(t.strip() for t in args.template_tags.split(",") if t.strip())
    system_content_tokens = (
        [t.strip() for t in args.system_content_tokens.split(",") if t.strip()] or None
    )
    gen_offsets = [int(x) for x in args.gen_positions.split(",") if x.strip()]
    if "post_generation" in positions:
        assert gen_offsets, "--positions includes post_generation but --gen-positions is empty"

    allowed_domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    prompts = [(d, t) for d, t in PROMPTS if d in allowed_domains]
    assert prompts, f"no prompts left after filtering to domains {sorted(allowed_domains)}"

    storage = make_storage(args)
    print(f"organisms ({len(organism_keys)}): {organism_keys}")
    print(f"positions: {sorted(positions)}")
    print(f"domains: {sorted(allowed_domains)} -> {len(prompts)}/{len(PROMPTS)} prompts")

    existing_table = None
    todo_keys = organism_keys
    if args.resume and storage.exists(args.output):
        existing_table = pq.read_table(storage.open_read(args.output))
        done_ft_ids = set(existing_table.column("ft_model_id").to_pylist())
        todo_keys = [k for k in organism_keys if ORGANISMS[k]["ft_model_id"] not in done_ft_ids]
        print(f"[resume] {existing_table.num_rows} rows already in {args.output}, "
              f"covering {len(done_ft_ids)} organism(s); "
              f"{len(organism_keys) - len(todo_keys)}/{len(organism_keys)} skipped, "
              f"{len(todo_keys)} left to extract")
    elif storage.exists(args.output):
        print(f"[!] {args.output} already exists and --resume was not passed -- it will be overwritten")

    def _write(all_rows: list[dict]) -> pa.Table:
        parts = [existing_table] if existing_table is not None else []
        if all_rows:
            parts.append(pa.table(
                {name: [r[name] for r in all_rows] for name in _SCHEMA.names},
                schema=_SCHEMA,
            ))
        table = pa.concat_tables(parts) if parts else pa.table(
            {name: [] for name in _SCHEMA.names}, schema=_SCHEMA
        )
        storage.ensure_parent(args.output)
        with pq.ParquetWriter(storage.open_write(args.output), _SCHEMA) as writer:
            writer.write_table(table)

        n_docs = len(prompts) * len(organism_keys)
        meta = NLADatasetMeta(
            dataset_id=f"diff_base_{args.model_name}__{'_'.join(organism_keys)}",
            stage="base",
            row_count=table.num_rows,
            extraction=NLAExtractionMeta(
                base_model=BASE_MODEL_ID,
                d_model=D_MODEL,
                layer_index=LAYER,
                norm="none",
                corpus=f"diff:organisms:{','.join(organism_keys)}",
                corpus_slice={"n_organisms": len(organism_keys), "n_prompts": len(prompts),
                              "role_name_map": role_name_map, "end_of_turn_token": end_of_turn_token,
                              "attn_implementation": attn_implementation},
                # Documentation-only field (not asserted downstream) -- varies with
                # --positions/--fractions/--template-tags, so this is an average,
                # not an exact per-doc count.
                positions_per_doc=round(table.num_rows / n_docs) if n_docs else 0,
            ),
            keep_debug_metadata=True,
            created_by="extract_diff_taboo_gemma.py",
        )
        write_sidecar(storage, args.output, meta)
        return table

    all_rows: list[dict] = []
    for i, key in enumerate(todo_keys, 1):
        org = ORGANISMS[key]
        all_rows.extend(_rows_for_organism(
            org["ft_model_id"], org["trait"], prompts,
            positions, fractions, template_tags, system_content_tokens,
            gen_offsets, args.gen_max_new_tokens, args.gen_do_sample, args.gen_temperature,
        ))
        if args.checkpoint_every and i % args.checkpoint_every == 0:
            table = _write(all_rows)
            print(f"[checkpoint] {i}/{len(todo_keys)} organisms done this run "
                  f"-> {table.num_rows} total rows in {args.output}")

    assert all_rows or existing_table is not None, (
        "no rows extracted -- check --organisms/--positions/--domains"
    )
    table = _write(all_rows)
    print(f"\nWrote {table.num_rows} diff-vector rows -> {args.output}")


if __name__ == "__main__":
    main()
