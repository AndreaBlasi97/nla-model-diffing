#!/usr/bin/env python3
"""
Extract a base-vs-finetuned activation DIFFERENCE vector at 25%/50%/75%/100%
of the user prompt's content tokens, in one layer, then verbalize each diff
vector N times via the NLA actor server.

This is extract_all.py's single-model extraction logic (content_span +
fractional token positions), applied to exactly one base/finetuned pair and
followed straight through the NLA actor client (nla_inference.NLAClient)
instead of writing an intermediate activations parquet. It backs the
diffing-game `nla_oracle` method (see
diffing-game/src/diffing/methods/nla_oracle/method.py), which shells out to
this script because diffing-game's own venv has torch/transformers/peft but
not the NLA actor client's deps (httpx/orjson) or a running sglang server.

Also extracts, from the SAME forward pass, diff vectors at the nine
chat-template scaffold tokens surrounding the user prompt: the
'<|im_start|>', role-name, and newline tokens of the system, user, and
assistant turns (e.g. 'assistant_im_start'). These are verbalized the
same way as the fractional-position diffs, with position_kind="chat_template".

Optionally (--gen-positions), ALSO extracts a second, independent set of
"post-generation" vectors: the finetuned model actually GENERATES a real
response (greedy by default), and hidden states are read from token offsets
at/after the assistant-generation boundary — i.e. from the model's own
committed output, not just its representation of the human's question. That
exact generated token sequence is teacher-forced through the base model too,
so both sides are compared at identical positions. This is opt-in and fully
additive: fraction-based extraction is unchanged when --gen-positions is
left at its default (empty = disabled).

Run in the nla_client conda env (needs GPU for the two forward passes). The
actor sglang server must already be running (see run_NLA.bash):

    python -m sglang.launch_server --model-path ./actor_hf --port 30000 \\
        --disable-radix-cache --mem-fraction-static 0.7 --trust-remote-code

Usage:
    python diff_verbalize.py \\
        --base-model-id unsloth/Qwen2.5-7B-Instruct \\
        --ft-model-id ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice \\
        --actor ./actor_hf \\
        --sglang-url http://localhost:30000 \\
        --layer 20 --n-repeats 8 \\
        --gen-positions 0,4,8,16 \\
        --out results.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_set import PROMPTS, SYSTEM_PROMPT  # noqa: E402
from nla_inference import NLAClient, strip_final_token_note  # noqa: E402


def content_span(tok, ids_list, end_of_turn_token="<|im_end|>"):
    """Return (start, end) indices covering the user sentence tokens only,
    excluding system prompt, role scaffolding, and the trailing
    end-of-turn/assistant tail. Heuristic: the user content sits between the
    last '<|im_start|>user'-style block and the following end-of-turn token
    (ChatML: '<|im_end|>'; Gemma: '<end_of_turn>' — pass the right one for
    the tokenizer in use, e.g. from the diffing-game model config's
    `end_of_turn_token` field).

    Some chat templates (e.g. Gemma3, which has no native system-role turn)
    fold the system prompt into the SAME 'user' turn ahead of the actual user
    content, separated by a blank line:
        <start_of_turn>user\\n{system}\\n\\n{user_content}<end_of_turn>
    If a blank-line token shows up inside the naive span, the real user
    content starts right after the LAST one — otherwise fractional positions
    would be measured against the system prompt instead. This is a no-op for
    ChatML, where system/user are already separate blocks with no blank line
    between the 'user' marker and the content."""
    toks = tok.convert_ids_to_tokens(ids_list)
    try:
        user_marker = max(i for i, t in enumerate(toks) if t == "user")
    except ValueError:
        return 1, len(ids_list) - 1  # fallback: skip first/last
    start = user_marker + 2  # skip 'user' and the following newline token
    end = next((i for i in range(start, len(toks)) if toks[i] == end_of_turn_token),
               len(toks))
    blank_line_positions = [i for i in range(start, end) if "\n\n" in toks[i]]
    if blank_line_positions:
        start = blank_line_positions[-1] + 1
    return start, end


def assistant_start_index(tok, messages):
    """Token index at which the assistant's own generated content begins:
    len(prompt_ids) where prompt_ids includes the add_generation_prompt=True
    role-scaffold suffix (e.g. '<|im_start|>assistant\\n'). Mirrors
    diffing-game's activation_difference_lens `assistant_start_index =
    len(user_ids)` convention, except `messages` here already includes the
    system prompt, matching extract_vectors()'s own prompt construction.
    Returns (prompt_ids, start_idx)."""
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    return prompt_ids, len(prompt_ids)


def _is_lora_adapter(model_id):
    """True if model_id has an adapter_config.json. Checks the local HF
    cache first -- unlike AutoTokenizer/AutoModel's from_pretrained,
    list_repo_files() ignores HF_HUB_OFFLINE and always hits the network,
    so a fully-cached repo must never reach it while offline."""
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(repo_id=model_id, filename="adapter_config.json")
    if isinstance(cached, str):
        return True
    if cached is not None:  # sentinel: cache confirms the file doesn't exist
        return False
    from huggingface_hub import list_repo_files
    return "adapter_config.json" in list_repo_files(model_id)


def load_causal_model(model_id, base_for_adapter, chat_template_fallback=None,
                       attn_implementation=None):
    is_adapter = _is_lora_adapter(model_id)

    tok = AutoTokenizer.from_pretrained(base_for_adapter if is_adapter else model_id)
    if tok.chat_template is None and chat_template_fallback is not None:
        # Some base (non-instruct) checkpoints, e.g. Qwen2.5-7B, ship the
        # ChatML special tokens but no chat_template itself. Borrowing the
        # counterpart model's template is safe here since both sides need
        # identical role scaffolding for content_span() to line up anyway.
        tok.chat_template = chat_template_fallback

    # attn_implementation: e.g. Gemma3 needs "eager" (its own diffing-game
    # model config specifies this — see configs/model/gemma3_12B_it.yaml),
    # since its attention-softcapping/sliding-window logic isn't SDPA/FA2
    # safe on every version. None (default) leaves transformers' own
    # auto-selection untouched, preserving existing Qwen behavior exactly.
    from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True,
    )
    if attn_implementation:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    if torch.cuda.is_available():
        # accelerate's own free-memory probe under device_map="auto" can be
        # too conservative on some multi-GPU hosts and silently parks whole
        # layers on CPU RAM instead of erroring (invisible unless you notice
        # idle GPUs + pegged CPU cores) -- pin max_memory to each GPU's real
        # capacity (10% headroom for activations/CUDA context) so large
        # models actually land on GPU.
        from_pretrained_kwargs["max_memory"] = {
            i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.9 / 1024**3)}GiB"
            for i in range(torch.cuda.device_count())
        }

    if is_adapter:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_for_adapter, **from_pretrained_kwargs)
        model = PeftModel.from_pretrained(base, model_id)
        model = model.merge_and_unload()
        print(f"  {model_id} is a LoRA adapter on {base_for_adapter}")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    model.eval()
    return model, tok


def extract_vectors(model, tok, layer, prompts, system_prompt, fractions,
                     end_of_turn_token="<|im_end|>"):
    """Returns (vecs, meta), both dict[(prompt_id, frac)] -> value, over the
    fractional positions of each prompt's user-content span."""
    vecs, meta = {}, {}
    for pid, (domain, text) in enumerate(prompts):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=True,
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
        ids_list = ids[0].tolist()

        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0]  # [seq, d]

        s, e = content_span(tok, ids_list, end_of_turn_token)
        toks = tok.convert_ids_to_tokens(ids_list)
        span_len = e - s
        if span_len < 1:
            print(f"  [!] prompt {pid}: empty content span, skipping")
            continue

        for frac in fractions:
            offset = min(int(round(frac * span_len)) - 1, span_len - 1)
            offset = max(offset, 0)
            idx = s + offset
            vecs[(pid, frac)] = hs[idx].float().cpu().numpy()
            meta[(pid, frac)] = {"domain": domain, "prompt": text, "token": toks[idx]}
    return vecs, meta


def chat_template_tag_positions(tok, ids_list, tags=("system", "user", "assistant"),
                                 role_name_map=None):
    """Return dict[tag] -> {'im_start': i, 'role_name': i, 'newline': i},
    the token indices of each requested role tag's 3-token ChatML scaffold
    '<|im_start|>{role}\\n'. Uses the LAST literal occurrence of the bare
    role-name token in ids_list -- same heuristic content_span() already
    relies on for its 'user' marker, and safe for the same reason: the role
    token (directly after '<|im_start|>', no leading-space marker) is a
    distinct vocab entry from the same word appearing mid-sentence (e.g.
    'assistant' vs 'Ġassistant'), so it won't false-match prose content.
    Tags absent from ids_list are omitted from the result; a tag found at
    the very start/end of the sequence omits 'im_start'/'newline' rather
    than indexing out of range.

    role_name_map: optional dict[tag] -> literal role-name token text to
    search for instead of `tag` itself, for chat templates that don't use
    ChatML's own role-name spelling -- e.g. Gemma's template ends a turn with
    '<start_of_turn>model\\n', not '...assistant\\n', so tag='assistant'
    needs role_name_map={'assistant': 'model'} to ever match anything.
    Without a mapping entry, a tag silently finds zero matches (Gemma has no
    'system' role token at all -- correctly absent, see system_content_span's
    docstring -- but a MISSING 'assistant' mapping entry would silently drop
    real data instead). The returned dict is still keyed by `tag`, so
    downstream position_kind labels ('chat_template:assistant_im_start')
    stay uniform across models regardless of the literal token searched for."""
    toks = tok.convert_ids_to_tokens(ids_list)
    role_name_map = role_name_map or {}
    result = {}
    for tag in tags:
        literal = role_name_map.get(tag, tag)
        idxs = [i for i, t in enumerate(toks) if t == literal]
        if not idxs:
            continue
        i = idxs[-1]
        entry = {"role_name": i}
        if i - 1 >= 0:
            entry["im_start"] = i - 1
        if i + 1 < len(toks):
            entry["newline"] = i + 1
        result[tag] = entry
    return result


def system_content_span(tok, ids_list, end_of_turn_token="<|im_end|>"):
    """Token indices (start, end) of the system prompt's own content, right
    after its 3-token '<|im_start|>system\\n' scaffold and up to the
    following end-of-turn token.

    Falls back to the Gemma3-style fold when no literal 'system' role token
    exists: Gemma3 has no native system-role turn, so the system prompt sits
    at the START of the 'user' turn, separated from the real user content by
    a blank-line token (see content_span()'s own docstring for the same
    fold, handled there from the opposite side -- finding where the REAL
    user content starts instead of where the system content ends). This
    mirrors that same blank-line detection.

    Returns None only if neither a 'system' role token nor a user-turn
    blank line can be found at all."""
    toks = tok.convert_ids_to_tokens(ids_list)
    try:
        marker = next(i for i, t in enumerate(toks) if t == "system")
        start = marker + 2  # skip 'system' and the following newline token
        end = next((i for i in range(start, len(toks)) if toks[i] == end_of_turn_token),
                   len(toks))
        return start, end
    except StopIteration:
        pass

    try:
        user_marker = max(i for i, t in enumerate(toks) if t == "user")
    except ValueError:
        return None
    start = user_marker + 2  # skip 'user' and the following newline token
    end = next((i for i in range(start, len(toks)) if toks[i] == end_of_turn_token),
               len(toks))
    blank_line_positions = [i for i in range(start, end) if "\n\n" in toks[i]]
    if not blank_line_positions:
        return None  # no 'system' turn and no fold detected -- genuinely absent
    return start, blank_line_positions[-1]


def _normalize_tok(t):
    """Strip byte-level-BPE leading-space markers ('Ġ' GPT2-style, '▁'
    SentencePiece-style) and casefold, so a caller can match tokens by their
    plain word text regardless of tokenizer family or where in a sentence
    the word landed (leading-space or not)."""
    return t.lstrip("Ġ▁").lower()


def extract_system_content_vectors(model, tok, layer, prompts, system_prompt,
                                     end_of_turn_token="<|im_end|>",
                                     token_filter=None):
    """Diff-relevant vectors at EVERY token of the system prompt's own
    content span (not just its 3-token scaffold), extracted once per prompt
    in `prompts` -- same loop shape as extract_vectors(), so each result
    keeps its own (domain, prompt) metadata for grouping/analysis, even
    though causal attention means the system-content hidden states
    themselves are identical across prompts (they only depend on tokens up
    to and including that position, and every prompt shares the same
    system prompt + role scaffolding prefix).

    token_filter: optional list of lowercase words. When given, only tokens
    whose normalized form (see _normalize_tok) STARTS WITH one of these
    words are kept (startswith, not ==, so e.g. a hypothetical punctuation-
    merged 'assistant.' token still matches 'assistant'). None (default)
    keeps every system-content token, matching prior behavior exactly.

    Returns (vecs, meta), keyed by (prompt_id, token index into the system
    content span); index 0 = first system-prompt content token."""
    vecs, meta = {}, {}
    for pid, (domain, text) in enumerate(prompts):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=True,
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
        ids_list = ids[0].tolist()

        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0]  # [seq, d]

        toks = tok.convert_ids_to_tokens(ids_list)
        span = system_content_span(tok, ids_list, end_of_turn_token)
        if span is None:
            print(f"  [!] prompt {pid}: no 'system' role token found, skipping")
            continue
        start, end = span
        for i, idx in enumerate(range(start, end)):
            if token_filter is not None:
                normalized = _normalize_tok(toks[idx])
                if not any(normalized.startswith(w) for w in token_filter):
                    continue
            vecs[(pid, i)] = hs[idx].float().cpu().numpy()
            meta[(pid, i)] = {"domain": domain, "prompt": text, "token": toks[idx]}
    return vecs, meta


def extract_vectors_with_template_tags(
    model, tok, layer, prompts, system_prompt, fractions,
    end_of_turn_token="<|im_end|>", template_tags=("system", "user", "assistant"),
    role_name_map=None,
):
    """Same extraction as extract_vectors(), from the SAME forward pass, plus
    a hidden state at every token of each requested chat-template role tag's
    3-token scaffold (see chat_template_tag_positions()).

    Returns (vecs, meta, tag_vecs, tag_meta):
      vecs, meta: identical to extract_vectors()'s own return, keyed by
        (prompt_id, fraction).
      tag_vecs, tag_meta: keyed by (prompt_id, "{tag}_{part}"), e.g.
        "user_role_name", "assistant_im_start", "system_newline" -- kept in
        SEPARATE dicts (not merged into vecs/meta) since mixing a float
        fraction and a string label as the second tuple element would break
        callers' `sorted(set(...))` over these keys (float/str comparison
        raises TypeError)."""
    vecs, meta, tag_vecs, tag_meta = {}, {}, {}, {}
    for pid, (domain, text) in enumerate(prompts):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=True,
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
        ids_list = ids[0].tolist()

        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0]  # [seq, d]

        toks = tok.convert_ids_to_tokens(ids_list)

        s, e = content_span(tok, ids_list, end_of_turn_token)
        span_len = e - s
        if span_len < 1:
            print(f"  [!] prompt {pid}: empty content span, skipping")
        else:
            for frac in fractions:
                offset = min(int(round(frac * span_len)) - 1, span_len - 1)
                offset = max(offset, 0)
                idx = s + offset
                vecs[(pid, frac)] = hs[idx].float().cpu().numpy()
                meta[(pid, frac)] = {"domain": domain, "prompt": text, "token": toks[idx]}

        for tag, parts in chat_template_tag_positions(tok, ids_list, template_tags, role_name_map).items():
            for part, idx in parts.items():
                label = f"{tag}_{part}"
                tag_vecs[(pid, label)] = hs[idx].float().cpu().numpy()
                tag_meta[(pid, label)] = {"domain": domain, "prompt": text, "token": toks[idx]}

    return vecs, meta, tag_vecs, tag_meta


def generate_and_extract_gen_vectors(
    model, tok, layer, prompts, system_prompt, gen_offsets,
    max_new_tokens, do_sample, temperature,
):
    """Let `model` actually GENERATE a response (greedy unless do_sample),
    then read hidden states from a teacher-forced forward pass over the full
    prompt+generated sequence, at token offsets counted from the first token
    the model itself generated (0 = first generated token).

    Returns (vecs, meta, pid_state):
      vecs: dict[(prompt_id, offset)] -> np.ndarraytggggggg
      meta: dict[(prompt_id, offset)] -> {domain, prompt, token,
            generated_text, assistant_start_index}
      pid_state: dict[prompt_id] -> {full_ids, assistant_start_index} — hand
            this to extract_gen_vectors_teacher_forced() so the OTHER model
            is scored on the IDENTICAL token sequence (required for
            diff = ft_vec - base_vec to compare matched positions).
    """
    vecs, meta, pid_state = {}, {}, {}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=pad_id)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    for pid, (domain, text) in enumerate(prompts):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        prompt_ids, start_idx = assistant_start_index(tok, messages)
        ids_t = torch.tensor([prompt_ids], device=model.device)
        attn_t = torch.ones_like(ids_t)

        with torch.no_grad():
            gen_out = model.generate(input_ids=ids_t, attention_mask=attn_t, **gen_kwargs)
        full_ids = gen_out[0].tolist()
        if len(full_ids) - start_idx < 1:
            print(f"  [!] prompt {pid}: model generated 0 tokens, skipping gen positions")
            continue

        with torch.no_grad():
            # Teacher-forced re-forward over prompt+generated, same idiom as
            # extract_vectors() — simpler than parsing generate()'s
            # incremental per-step hidden-state structure, and causal
            # attention means the result is identical either way.
            hs = model(input_ids=gen_out, output_hidden_states=True).hidden_states[layer][0]  # [seq, d]

        toks = tok.convert_ids_to_tokens(full_ids)
        generated_text = tok.decode(full_ids[start_idx:], skip_special_tokens=True)
        pid_state[pid] = {"full_ids": full_ids, "assistant_start_index": start_idx}

        for off in gen_offsets:
            idx = start_idx + off
            if idx < 0 or idx >= len(full_ids):
                print(f"  [!] prompt {pid}: offset {off} out of range "
                      f"(generated {len(full_ids) - start_idx} tokens), skipping")
                continue
            vecs[(pid, off)] = hs[idx].float().cpu().numpy()
            meta[(pid, off)] = {
                "domain": domain, "prompt": text, "token": toks[idx],
                "generated_text": generated_text[:2000],
                "assistant_start_index": start_idx,
            }
    return vecs, meta, pid_state


def extract_gen_vectors_teacher_forced(model, layer, pid_state, gen_offsets):
    """Forward `model` (teacher-forced, no generation) over the token
    sequences captured by generate_and_extract_gen_vectors() on the OTHER
    model, so both models are read at identical positions."""
    vecs = {}
    for pid, state in pid_state.items():
        ids_t = torch.tensor([state["full_ids"]], device=model.device)
        with torch.no_grad():
            hs = model(input_ids=ids_t, output_hidden_states=True).hidden_states[layer][0]
        start_idx = state["assistant_start_index"]
        for off in gen_offsets:
            idx = start_idx + off
            if idx < 0 or idx >= len(state["full_ids"]):
                continue  # already warned during the generation pass
            vecs[(pid, off)] = hs[idx].float().cpu().numpy()
    return vecs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--ft-model-id", required=True,
                    help="HF repo id of the finetuned model or LoRA adapter")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--fractions", default="0.25,0.50,0.75,1.00",
                    help="comma-separated fractions through the user content span")
    ap.add_argument("--template-tags", default="system,user,assistant",
                    help="comma-separated chat-template roles to extract scaffold-token "
                         "('<|im_start|>'/role-name/newline) diffs for, e.g. 'system' to "
                         "restrict a run to just the system turn's three tokens instead "
                         "of all three roles' nine.")
    ap.add_argument("--role-name-map", default="",
                    help="comma-separated tag=literal pairs remapping a --template-tags "
                         "entry to the literal role-name token text to search for, for "
                         "chat templates that don't spell roles the ChatML way -- e.g. "
                         "Gemma ends a turn with '<start_of_turn>model', not "
                         "'...assistant', so that model needs 'assistant=model' or the "
                         "assistant tag silently matches nothing. Gemma also has no "
                         "'system' role token at all (folded into 'user', handled "
                         "separately by system_content_span) -- no mapping fixes that, "
                         "it's genuinely absent. Empty (default) searches for each tag's "
                         "own literal name, matching prior behavior.")
    ap.add_argument("--end-of-turn-token", default="<|im_end|>",
                    help="end-of-turn special token for content_span()'s span-end "
                         "heuristic — model-specific (ChatML/Qwen: '<|im_end|>', "
                         "Gemma: '<end_of_turn>'). Matches the diffing-game model "
                         "config's `end_of_turn_token` field for the base model "
                         "being diffed.")
    ap.add_argument("--attn-implementation", default=None,
                    help="forwarded to AutoModelForCausalLM.from_pretrained for "
                         "both base and finetuned model loads, e.g. 'eager' "
                         "(Gemma3 needs this — see configs/model/gemma3_12B_it.yaml). "
                         "Default (unset) leaves transformers' own auto-selection "
                         "in place, matching prior behavior.")
    ap.add_argument("--gen-positions", default="",
                    help="comma-separated integer token offsets from the "
                         "assistant's first generated token (0=first). Empty "
                         "(default) disables post-generation extraction "
                         "entirely. When set, the FINETUNED model actually "
                         "GENERATES a real response once (greedy unless "
                         "--gen-do-sample), and that exact token sequence is "
                         "teacher-forced through the base model too, so both "
                         "sides are read at identical positions.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=64,
                    help="max tokens the FINETUNED model actually generates "
                         "for the post-generation extraction pass (separate "
                         "from --max-new-tokens, which caps the NLA actor's "
                         "verbalization length)")
    ap.add_argument("--gen-do-sample", action="store_true",
                    help="sample the finetuned model's response instead of "
                         "greedy decoding (default: greedy/deterministic, so "
                         "re-running the script is reproducible)")
    ap.add_argument("--gen-temperature", type=float, default=0.7,
                    help="only used when --gen-do-sample is set")
    ap.add_argument("--actor", required=True, help="path to actor_hf (AV checkpoint)")
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--n-repeats", type=int, default=8,
                    help="how many times to independently verbalize the SAME diff vector")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--anchor-to-ft", action="store_true",
                    help="verbalize ft_vec + strength*diff instead of the raw diff "
                         "vector — anchors the diff onto a real in-distribution "
                         "activation before NLA's L2 renormalization")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="multiplier on diff when --anchor-to-ft is set; ignored "
                         "(but still recorded) otherwise")
    ap.add_argument("--system-content-only", action="store_true",
                    help="extract diff vectors at every token of the system prompt's "
                         "OWN content (not just its 3-token scaffold from "
                         "--template-tags), skipping prompt_fraction and "
                         "chat_template extraction/verbalization entirely. Point "
                         "--out at an existing results.json to MERGE the new "
                         "position_kind='system_content' entries in alongside what's "
                         "already there, instead of re-running (and re-spending "
                         "--n-repeats verbalizations on) work you already have.")
    ap.add_argument("--system-content-tokens", default="",
                    help="comma-separated literal words (case-insensitive, matched "
                         "as a prefix of each token's text with BPE space-markers "
                         "stripped) to restrict --system-content-only extraction to, "
                         "e.g. 'you,assistant' for just those two words of the fixed "
                         "system prompt. Empty (default) keeps every system-content "
                         "token. Only meaningful together with --system-content-only.")
    ap.add_argument("--prompt-domains", default="",
                    help="comma-separated domain tags (the first element of each "
                         "prompt_set.py PROMPTS tuple, e.g. 'neutral' or "
                         "'taboo_train') to restrict extraction+verbalization to. "
                         "Empty (default) keeps every prompt, matching prior "
                         "behavior. Use this instead of editing prompt_set.py "
                         "for a one-off run -- that file is shared with "
                         "extract_all.py.")
    ap.add_argument("--out", required=True, help="output results JSON path")
    args = ap.parse_args()

    fractions = [float(x) for x in args.fractions.split(",")]
    template_tags = tuple(t.strip() for t in args.template_tags.split(",") if t.strip())
    role_name_map = {}
    for pair in args.role_name_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        tag, _, literal = pair.partition("=")
        assert literal, f"--role-name-map entry {pair!r} must be 'tag=literal'"
        role_name_map[tag.strip()] = literal.strip()

    prompts = PROMPTS
    if args.prompt_domains:
        domains = {d.strip() for d in args.prompt_domains.split(",") if d.strip()}
        prompts = [p for p in PROMPTS if p[0] in domains]
        if not prompts:
            sys.exit(f"--prompt-domains {sorted(domains)} matched no prompts; "
                      f"available domains: {sorted({p[0] for p in PROMPTS})}")
        print(f"[prompt-domains] restricted to {sorted(domains)}: "
              f"{len(prompts)}/{len(PROMPTS)} prompts")
    system_content_tokens = [
        w.strip().lower() for w in args.system_content_tokens.split(",") if w.strip()
    ] or None
    gen_offsets = (
        [int(x) for x in args.gen_positions.split(",") if x.strip()]
        if args.gen_positions else []
    )

    # FT-first: post-generation extraction needs the finetuned model's own
    # generated token sequence BEFORE the base model's teacher-forced pass
    # can run, so it must be produced first. Fraction-based extraction is
    # unaffected by load order (it only ever depends on that model + the
    # fixed prompts), so this reordering is safe for the existing behavior.
    print(f"Loading finetuned model {args.ft_model_id} ...")
    base_template_probe = AutoTokenizer.from_pretrained(args.base_model_id).chat_template
    ft_model, ft_tok = load_causal_model(
        args.ft_model_id, args.base_model_id,
        chat_template_fallback=base_template_probe,
        attn_implementation=args.attn_implementation,
    )
    ft_vecs, meta, ft_tag_vecs, ft_tag_meta = {}, {}, {}, {}
    ft_gen_vecs, ft_gen_meta, pid_state = {}, {}, {}
    if not args.system_content_only:
        ft_vecs, meta, ft_tag_vecs, ft_tag_meta = extract_vectors_with_template_tags(
            ft_model, ft_tok, args.layer, prompts, SYSTEM_PROMPT, fractions,
            end_of_turn_token=args.end_of_turn_token, template_tags=template_tags,
            role_name_map=role_name_map,
        )
        if gen_offsets:
            print(f"Generating + extracting post-generation vectors from finetuned "
                  f"model at offsets {gen_offsets} ...")
            ft_gen_vecs, ft_gen_meta, pid_state = generate_and_extract_gen_vectors(
                ft_model, ft_tok, args.layer, prompts, SYSTEM_PROMPT, gen_offsets,
                args.gen_max_new_tokens, args.gen_do_sample, args.gen_temperature,
            )
    ft_sys_vecs, sys_meta = {}, {}
    if args.system_content_only:
        ft_sys_vecs, sys_meta = extract_system_content_vectors(
            ft_model, ft_tok, args.layer, prompts, SYSTEM_PROMPT,
            end_of_turn_token=args.end_of_turn_token,
            token_filter=system_content_tokens,
        )
    del ft_model
    torch.cuda.empty_cache()

    print(f"Loading base model {args.base_model_id} ...")
    base_model, base_tok = load_causal_model(
        args.base_model_id, args.base_model_id,
        chat_template_fallback=ft_tok.chat_template,
        attn_implementation=args.attn_implementation,
    )
    if gen_offsets:
        # Teacher-forcing FT's generated token IDs through the base model is
        # only meaningful if both share a vocabulary; a silent mismatch
        # would forward garbage IDs through the wrong embedding table.
        assert base_tok.vocab_size == ft_tok.vocab_size, (
            f"base/ft tokenizers must share a vocabulary for teacher-forcing "
            f"(--gen-positions) to be meaningful: base={base_tok.vocab_size} "
            f"ft={ft_tok.vocab_size}"
        )
    base_vecs, base_tag_vecs = {}, {}
    base_gen_vecs = {}
    if not args.system_content_only:
        base_vecs, _, base_tag_vecs, _ = extract_vectors_with_template_tags(
            base_model, base_tok, args.layer, prompts, SYSTEM_PROMPT, fractions,
            end_of_turn_token=args.end_of_turn_token, template_tags=template_tags,
            role_name_map=role_name_map,
        )
        if gen_offsets:
            base_gen_vecs = extract_gen_vectors_teacher_forced(
                base_model, args.layer, pid_state, gen_offsets
            )
    base_sys_vecs = {}
    if args.system_content_only:
        base_sys_vecs, _ = extract_system_content_vectors(
            base_model, base_tok, args.layer, prompts, SYSTEM_PROMPT,
            end_of_turn_token=args.end_of_turn_token,
            token_filter=system_content_tokens,
        )
    del base_model
    torch.cuda.empty_cache()

    keys = sorted(set(base_vecs) & set(ft_vecs))
    missing = set(base_vecs) ^ set(ft_vecs)
    if missing:
        print(f"  [!] {len(missing)} (prompt,position) keys present on only one "
              f"side (content span mismatch), skipping: {sorted(missing)}")

    gen_keys = sorted(set(base_gen_vecs) & set(ft_gen_vecs)) if gen_offsets else []
    gen_missing = (set(base_gen_vecs) ^ set(ft_gen_vecs)) if gen_offsets else set()
    if gen_missing:
        print(f"  [!] {len(gen_missing)} (prompt,offset) gen-position keys present "
              f"on only one side, skipping: {sorted(gen_missing)}")

    tag_keys = sorted(set(base_tag_vecs) & set(ft_tag_vecs))
    tag_missing = set(base_tag_vecs) ^ set(ft_tag_vecs)
    if tag_missing:
        print(f"  [!] {len(tag_missing)} (prompt,tag) chat-template keys present "
              f"on only one side, skipping: {sorted(tag_missing)}")

    sys_keys = sorted(set(base_sys_vecs) & set(ft_sys_vecs))
    sys_missing = set(base_sys_vecs) ^ set(ft_sys_vecs)
    if sys_missing:
        print(f"  [!] {len(sys_missing)} system-content token indices present "
              f"on only one side, skipping: {sorted(sys_missing)}")

    print(f"Loading NLA actor client from {args.actor} ...")
    client = NLAClient(args.actor, sglang_url=args.sglang_url)

    results = []
    for (pid, frac) in keys:
        diff = ft_vecs[(pid, frac)] - base_vecs[(pid, frac)]
        vec = ft_vecs[(pid, frac)] + args.strength * diff if args.anchor_to_ft else diff
        texts = [
            strip_final_token_note(client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            ))
            for _ in range(args.n_repeats)
        ]
        m = meta[(pid, frac)]
        print(f"[prompt {pid} @ {frac:.0%}, {m['domain']}] "
              f"{' '.join(texts[0].split())[:70]}...")
        results.append({
            "prompt_id": pid, "domain": m["domain"], "prompt": m["prompt"],
            "position_kind": "prompt_fraction", "position": frac, "token": m["token"],
            "base_model_id": args.base_model_id, "ft_model_id": args.ft_model_id,
            "act_key": "ft_plus_diff" if args.anchor_to_ft else "diff", "layer": args.layer,
            "diff_norm": float(np.linalg.norm(diff)),
            "vec_norm": float(np.linalg.norm(vec)),
            "verbalizations": texts,
        })

    for (pid, label) in tag_keys:
        diff = ft_tag_vecs[(pid, label)] - base_tag_vecs[(pid, label)]
        vec = ft_tag_vecs[(pid, label)] + args.strength * diff if args.anchor_to_ft else diff
        texts = [
            strip_final_token_note(client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            ))
            for _ in range(args.n_repeats)
        ]
        m = ft_tag_meta[(pid, label)]
        print(f"[prompt {pid} @ {label}, {m['domain']}] "
              f"{' '.join(texts[0].split())[:70]}...")
        results.append({
            "prompt_id": pid, "domain": m["domain"], "prompt": m["prompt"],
            "position_kind": "chat_template", "position": label, "token": m["token"],
            "base_model_id": args.base_model_id, "ft_model_id": args.ft_model_id,
            "act_key": "ft_plus_diff" if args.anchor_to_ft else "diff", "layer": args.layer,
            "diff_norm": float(np.linalg.norm(diff)),
            "vec_norm": float(np.linalg.norm(vec)),
            "verbalizations": texts,
        })

    for (pid, off) in gen_keys:
        diff = ft_gen_vecs[(pid, off)] - base_gen_vecs[(pid, off)]
        vec = ft_gen_vecs[(pid, off)] + args.strength * diff if args.anchor_to_ft else diff
        texts = [
            strip_final_token_note(client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            ))
            for _ in range(args.n_repeats)
        ]
        m = ft_gen_meta[(pid, off)]
        print(f"[prompt {pid} @ gen+{off}, {m['domain']}] "
              f"{' '.join(texts[0].split())[:70]}...")
        results.append({
            "prompt_id": pid, "domain": m["domain"], "prompt": m["prompt"],
            "position_kind": "post_generation", "position": off,
            "assistant_start_index": m["assistant_start_index"], "token": m["token"],
            "generated_text": m["generated_text"],
            "base_model_id": args.base_model_id, "ft_model_id": args.ft_model_id,
            "act_key": "ft_plus_diff" if args.anchor_to_ft else "diff", "layer": args.layer,
            "diff_norm": float(np.linalg.norm(diff)),
            "vec_norm": float(np.linalg.norm(vec)),
            "verbalizations": texts,
        })

    for (pid, idx) in sys_keys:
        diff = ft_sys_vecs[(pid, idx)] - base_sys_vecs[(pid, idx)]
        vec = ft_sys_vecs[(pid, idx)] + args.strength * diff if args.anchor_to_ft else diff
        texts = [
            strip_final_token_note(client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens
            ))
            for _ in range(args.n_repeats)
        ]
        m = sys_meta[(pid, idx)]
        print(f"[prompt {pid} @ system_content token {idx} ({m['token']!r}), {m['domain']}] "
              f"{' '.join(texts[0].split())[:70]}...")
        results.append({
            "prompt_id": pid, "domain": m["domain"], "prompt": m["prompt"],
            "position_kind": "system_content", "position": idx, "token": m["token"],
            "base_model_id": args.base_model_id, "ft_model_id": args.ft_model_id,
            "act_key": "ft_plus_diff" if args.anchor_to_ft else "diff", "layer": args.layer,
            "diff_norm": float(np.linalg.norm(diff)),
            "vec_norm": float(np.linalg.norm(vec)),
            "verbalizations": texts,
        })

    config = {
        "base_model_id": args.base_model_id, "ft_model_id": args.ft_model_id,
        "layer": args.layer, "fractions": fractions,
        "template_tags": list(template_tags), "role_name_map": role_name_map, "n_repeats": args.n_repeats,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "anchor_to_ft": args.anchor_to_ft, "strength": args.strength,
        "gen_positions": gen_offsets, "gen_max_new_tokens": args.gen_max_new_tokens,
        "gen_do_sample": args.gen_do_sample, "gen_temperature": args.gen_temperature,
    }

    if args.system_content_only and os.path.exists(args.out):
        # Merge instead of overwrite: keep whatever prompt_fraction/
        # chat_template/post_generation results already live in --out (they
        # were expensive to verbalize) and only replace prior
        # system_content entries, so reruns of just this flag stay
        # idempotent without re-spending --n-repeats on everything else.
        with open(args.out) as f:
            existing = json.load(f)
        existing["results"] = (
            [r for r in existing.get("results", []) if r["position_kind"] != "system_content"]
            + results
        )
        existing["config"]["system_content"] = True
        existing["config"]["system_content_tokens"] = system_content_tokens
        out = existing
    else:
        config["system_content"] = args.system_content_only
        config["system_content_tokens"] = system_content_tokens
        out = {"config": config, "results": results}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out['results'])} verbalized diff vectors -> {args.out}"
          f" ({len(results)} new this run)")


if __name__ == "__main__":
    main()
