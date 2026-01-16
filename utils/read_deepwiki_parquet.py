#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _iter_part_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        name = path.name
        if name.startswith("part-") and path.is_file():
            yield path


def _read_table(path: Path, *, columns: Optional[List[str]] = None) -> pa.Table:
    table = pq.read_table(path, columns=columns)
    return table


def _select_record(row: Dict[str, object], keys: Optional[List[str]]) -> Dict[str, object]:
    if not keys:
        return row
    return {key: row.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read mounted HDFS deepwiki Parquet part files and print JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki",
        help="Directory containing parquet part-* files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to print (across all part files).",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="Comma-separated column list to read (empty means all).",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="Comma-separated keys to print (subset of columns; empty means all).",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print schema of the first part file and exit.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    part_files = list(_iter_part_files(input_dir))
    if not part_files:
        raise SystemExit(f"No part-* files found under {input_dir}")

    columns = [c.strip() for c in args.columns.split(",") if c.strip()] or None
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None

    if args.show_schema:
        schema = pq.read_schema(part_files[0])
        print(schema)
        return

    remaining = max(0, int(args.limit))
    for part in part_files:
        if remaining <= 0:
            break
        table = _read_table(part, columns=columns)
        if table.num_rows == 0:
            continue
        take = min(remaining, table.num_rows)
        # Convert only what we need for printing.
        rows = table.slice(0, take).to_pylist()
        for row in rows:
            print(json.dumps(_select_record(row, keys), ensure_ascii=False))
        remaining -= take


if __name__ == "__main__":
    main()

