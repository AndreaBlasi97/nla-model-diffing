#!/usr/bin/env python3
"""
Keep only the given position_kind value(s) from a diff parquet (extract_
diff_base.py / extract_diff_taboo_gemma.py / extract_diff_analogy_combo*.py
schema), writing the filtered rows + a patched sidecar to --output.

Two uses this was written for:
  1. Selecting a small subset of an already-extracted --all-positions
     parquet to hand to sweep_checkpoints.bash's --all-positions-{combo,
     taboo} as the "target" parquet, when you only want a few of its
     position_kind values scored (not the full set) -- e.g. combo's
     chat_template:{assistant_im_start,system_im_start,system_newline,
     system_role_name} out of its 10 available position_kinds. No
     extraction needed here since the vectors already exist.
  2. Dropping an incidental extra position_kind that a --system-content-
     tokens prefix filter over-matched (e.g. "a" also matches "assistant" --
     see extract_diff_taboo_gemma.py's extract_system_content_vectors(),
     which filters by startswith, not exact match) before merging new rows
     onto an existing parquet with merge_diff_positions.py.

Usage:
    python filter_diff_positions.py \\
        --input diff_analogy_combo_v3/diff_base_eval_all_positions.parquet \\
        --keep "chat_template:assistant_im_start,chat_template:system_im_start,chat_template:system_newline,chat_template:system_role_name" \\
        --output diff_analogy_combo_v3/diff_base_eval_selected_positions.parquet
"""

import argparse
import sys
from dataclasses import replace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import os

sys.path.insert(0, os.environ.get("NLA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_language_autoencoders")))
from nla.datagen._common import add_storage_args, make_storage  # noqa: E402
from nla.datagen.sidecar import read_sidecar, write_sidecar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--keep", required=True,
                     help="comma-separated position_kind values to keep, exact match "
                          "(e.g. 'analogy_combo,chat_template:assistant_im_start')")
    ap.add_argument("--output", required=True)
    add_storage_args(ap)
    args = ap.parse_args()

    keep = {v.strip() for v in args.keep.split(",") if v.strip()}
    assert keep, "--keep resolved to an empty set"

    storage = make_storage(args)
    table = pq.read_table(storage.open_read(args.input))
    have = set(table.column("position_kind").to_pylist())
    missing = keep - have
    assert not missing, f"--keep value(s) {sorted(missing)} not present in {args.input} (has: {sorted(have)})"

    before = table.num_rows
    mask = pc.is_in(table.column("position_kind"), value_set=pa.array(sorted(keep)))
    table = table.filter(mask)
    print(f"[filter] {before} -> {table.num_rows} rows, kept position_kind(s): {sorted(keep)}")

    storage.ensure_parent(args.output)
    with pq.ParquetWriter(storage.open_write(args.output), table.schema) as writer:
        writer.write_table(table)

    meta = read_sidecar(storage, args.input)
    meta = replace(meta, row_count=table.num_rows, created_by="filter_diff_positions.py")
    write_sidecar(storage, args.output, meta)
    print(f"wrote {table.num_rows} rows -> {args.output}")


if __name__ == "__main__":
    main()
