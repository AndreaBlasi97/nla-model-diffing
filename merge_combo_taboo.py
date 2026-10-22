#!/usr/bin/env python3
"""
Merge a combo/triplet RL parquet and a taboo RL parquet into one training
parquet, downsampling the combo side to hit a fixed combo/taboo row ratio
(taboo is kept whole; combo is randomly subsampled down to match).

Both inputs must already be stage3-built RL parquets with the SAME schema
(prompt, activation_vector, n_raw_tokens, activation_layer, doc_id,
detokenized_text_truncated) and the SAME baked-in actor prompt template
(nla.reward.nla_rm's target-word extraction doesn't care, but a clean mix
needs the actor to see identically-worded instructions regardless of which
data source a row came from — see nla_meta.yaml's prompt_templates.actor).

Usage:
    python merge_combo_taboo.py \
        --combo diff_analogy_combo_v3/rl_train.parquet \
        --taboo diff_taboo_v2/rl_train.parquet \
        --output diff_combo_taboo_mix/rl_train.parquet \
        --combo-frac 0.7 --seed 42
"""

import argparse
import random
import sys
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import os

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import read_sidecar, write_sidecar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", required=True, help="combo/triplet rl_train.parquet")
    ap.add_argument("--taboo", required=True, help="taboo rl_train.parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--combo-frac", type=float, default=0.7,
                     help="target fraction of the merged set that is combo/triplet rows "
                          "(default 0.7 = 70%% combo / 30%% taboo). Taboo is kept whole; "
                          "combo is randomly subsampled down to hit this ratio.")
    ap.add_argument("--seed", type=int, default=42)
    add_storage_args(ap)
    args = ap.parse_args()

    storage = make_storage(args)

    combo_meta = read_sidecar(storage, args.combo)
    taboo_meta = read_sidecar(storage, args.taboo)
    assert combo_meta.extraction.base_model == taboo_meta.extraction.base_model, (
        f"base model mismatch: combo={combo_meta.extraction.base_model!r} "
        f"taboo={taboo_meta.extraction.base_model!r}"
    )
    assert combo_meta.extraction.layer_index == taboo_meta.extraction.layer_index, (
        f"layer mismatch: combo={combo_meta.extraction.layer_index} "
        f"taboo={taboo_meta.extraction.layer_index}"
    )
    assert combo_meta.prompt_templates == taboo_meta.prompt_templates, (
        f"actor/critic prompt templates differ between combo and taboo sidecars — "
        f"rebuild one of them with a matching --actor-template before merging.\n"
        f"combo:  {combo_meta.prompt_templates!r}\n"
        f"taboo:  {taboo_meta.prompt_templates!r}"
    )

    combo_table = pq.read_table(storage.open_read(args.combo))
    taboo_table = pq.read_table(storage.open_read(args.taboo))
    assert combo_table.schema.equals(taboo_table.schema), (
        f"schema mismatch:\ncombo: {combo_table.schema}\ntaboo: {taboo_table.schema}"
    )

    n_taboo = taboo_table.num_rows
    n_combo_keep = round(n_taboo * args.combo_frac / (1 - args.combo_frac))
    assert n_combo_keep <= combo_table.num_rows, (
        f"need {n_combo_keep} combo rows for a {args.combo_frac:.0%} split against "
        f"{n_taboo} taboo rows, but combo only has {combo_table.num_rows}"
    )

    rng = random.Random(args.seed)
    combo_idx = sorted(rng.sample(range(combo_table.num_rows), n_combo_keep))
    combo_sampled = combo_table.take(pa.array(combo_idx, type=pa.int64()))

    merged = pa.concat_tables([combo_sampled, taboo_table])
    print(f"combo: {combo_sampled.num_rows} ({combo_sampled.num_rows / merged.num_rows:.1%})  "
          f"taboo: {taboo_table.num_rows} ({taboo_table.num_rows / merged.num_rows:.1%})  "
          f"total: {merged.num_rows}")

    storage.ensure_parent(args.output)
    with pq.ParquetWriter(storage.open_write(args.output), merged.schema) as writer:
        writer.write_table(merged)

    out_meta = replace(
        combo_meta,
        dataset_id=f"{combo_meta.dataset_id}__plus_taboo_system_content",
        row_count=merged.num_rows,
        extraction=replace(
            combo_meta.extraction,
            corpus=f"{combo_meta.extraction.corpus}+{taboo_meta.extraction.corpus}",
            corpus_slice={
                **combo_meta.extraction.corpus_slice,
                "n_combo_rows": combo_sampled.num_rows,
                "n_taboo_rows": taboo_table.num_rows,
                "combo_frac": args.combo_frac,
                "seed": args.seed,
            },
        ),
        parent_datasets=[combo_meta.dataset_id, taboo_meta.dataset_id],
        created_by="merge_combo_taboo.py",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"Wrote {merged.num_rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
