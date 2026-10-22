#!/usr/bin/env python3
"""
Append extra position rows (e.g. new --fractions, or a --positions subset run
separately) onto an existing diff parquet from extract_diff_taboo_gemma.py /
extract_diff_base.py, without re-running the positions already in it.

Straight pa.concat_tables (schemas must be identical) plus a row_count/
warning check on (ft_model_id, doc_id, position_kind) overlap -- unlike
merge_combo_taboo.py this does NOT downsample either side, it just appends.

Usage:
    # 1) extract only the new fractions into a separate file
    python extract_diff_taboo_gemma.py --domains neutral --positions fractions \\
        --fractions 0.75,1.00 --model-name gemma-3-12b-it \\
        --output diff_taboo_gemma/diff_base_neutral_frac75_100.parquet

    # 2) append it onto the existing allpos12 parquet
    python merge_diff_positions.py \\
        --base diff_taboo_gemma/diff_base_neutral_allpos12.parquet \\
        --add diff_taboo_gemma/diff_base_neutral_frac75_100.parquet \\
        --output diff_taboo_gemma/diff_base_neutral_allpos12_full.parquet
"""

import argparse
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
    ap.add_argument("--base", required=True, help="existing diff parquet to append onto")
    ap.add_argument("--add", required=True, help="new parquet with the extra position rows")
    ap.add_argument("--output", required=True)
    add_storage_args(ap)
    args = ap.parse_args()

    storage = make_storage(args)

    base_table = pq.read_table(storage.open_read(args.base))
    add_table = pq.read_table(storage.open_read(args.add))
    assert base_table.schema.equals(add_table.schema), (
        f"schema mismatch:\nbase: {base_table.schema}\nadd: {add_table.schema}"
    )

    base_keys = set(zip(
        base_table.column("ft_model_id").to_pylist(),
        base_table.column("doc_id").to_pylist(),
        base_table.column("position_kind").to_pylist(),
    ))
    add_keys = set(zip(
        add_table.column("ft_model_id").to_pylist(),
        add_table.column("doc_id").to_pylist(),
        add_table.column("position_kind").to_pylist(),
    ))
    overlap = base_keys & add_keys
    if overlap:
        print(f"[!] {len(overlap)} (ft_model_id, doc_id, position_kind) rows appear in BOTH "
              f"inputs -- both copies will be kept (this script does not dedupe). "
              f"e.g. {sorted(overlap)[:3]}")

    merged = pa.concat_tables([base_table, add_table])
    print(f"base: {base_table.num_rows}  add: {add_table.num_rows}  total: {merged.num_rows}")

    storage.ensure_parent(args.output)
    with pq.ParquetWriter(storage.open_write(args.output), merged.schema) as writer:
        writer.write_table(merged)

    base_meta = read_sidecar(storage, args.base)
    add_meta = read_sidecar(storage, args.add)
    out_meta = replace(
        base_meta,
        dataset_id=f"{base_meta.dataset_id}__plus_{add_meta.dataset_id}",
        row_count=merged.num_rows,
        extraction=replace(
            base_meta.extraction,
            corpus=f"{base_meta.extraction.corpus}+{add_meta.extraction.corpus}",
            corpus_slice={**base_meta.extraction.corpus_slice, **add_meta.extraction.corpus_slice},
        ),
        parent_datasets=[base_meta.dataset_id, add_meta.dataset_id],
        created_by="merge_diff_positions.py",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"Wrote {merged.num_rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
