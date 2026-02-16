"""
Convert a narrative Parquet (from export_narratives_to_parquet.py or
split_parquet_by_tokens.py) into the mariana training JSONL format.

Output schema per line:
{
  "content_split": "<narrative text>",
  "meta": {
    "source": "data_20251208",
    "docid": "1",           // str, 1-based incrementing
    "chunk_id": "<sha1>",   // sha1(docid + '/' + content_split)
    "extra": "{}",
    "aug_data": "{\"concat_order\": [\"content_split\"], \"concat_mask\": [1]}",
    "pos": 0,
    "max_pos": 0
  }
}

Example
-------
# Single file
python utils/export_training_jsonl.py \
    --input buckets/under_128k.parquet \
    --output under_128k.jsonl \
    --source deepwiki_narrative_20250213

# Process all bucket files in a directory
python utils/export_training_jsonl.py \
    --input-dir buckets/ \
    --output-dir training_jsonl/ \
    --source deepwiki_narrative_20250213

# Process a bucketed output root (nested buckets like under_128k/, 128k_512k/, 512k_2M/)
python utils/export_training_jsonl.py \
    --input-root buckets_0213/ \
    --output-dir training_jsonl/ \
    --source deepwiki_narrative_20250213

# Export the same schema but as Parquet instead of JSONL
python utils/export_training_jsonl.py \
    --input-root buckets_0213/ \
    --output-dir training_parquet/ \
    --source deepwiki_narrative_20250213 \
    --output-format parquet
"""

import argparse
import hashlib
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def generate_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _make_aug_data() -> str:
    return json.dumps(
        {"concat_order": ["content_split"], "concat_mask": [1]},
        ensure_ascii=False,
    )


def _training_schema() -> pa.Schema:
    return pa.schema(
        [
            ("content_split", pa.large_string()),
            (
                "meta",
                pa.struct(
                    [
                        ("source", pa.string()),
                        ("docid", pa.string()),
                        ("chunk_id", pa.string()),
                        ("extra", pa.string()),
                        ("aug_data", pa.string()),
                        ("pos", pa.int32()),
                        ("max_pos", pa.int32()),
                    ]
                ),
            ),
        ]
    )


def convert_parquet_to_jsonl(
    input_path: str,
    output_path: str,
    narrative_col: str,
    source: str,
    docid_start: int,
    batch_size: int,
) -> tuple[int, int]:
    """
    Convert one parquet file to JSONL.
    Returns (rows_written, next_docid).
    """
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    aug_data = json.dumps(
        {"concat_order": ["content_split"], "concat_mask": [1]},
        ensure_ascii=False,
    )

    docid = docid_start
    rows_written = 0

    pbar = tqdm(
        total=total_rows,
        desc=f"Converting {os.path.basename(input_path)}",
        unit="rows",
        dynamic_ncols=True,
        disable=False,
    )

    with open(output_path, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=batch_size):
            narratives = batch.column(narrative_col).to_pylist()
            for narrative in narratives:
                if not narrative:
                    docid += 1
                    pbar.update(1)
                    continue

                if not isinstance(narrative, str):
                    narrative = str(narrative)

                docid_str = str(docid)
                text_for_hash = docid_str + "/" + narrative
                chunk_id = generate_hash(text_for_hash)

                record = {
                    "content_split": narrative,
                    "meta": {
                        "source": source,
                        "docid": docid_str,
                        "chunk_id": chunk_id,
                        "extra": "{}",
                        "aug_data": aug_data,
                        "pos": 0,
                        "max_pos": 0,
                    },
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1
                docid += 1
                pbar.update(1)

    pbar.close()
    return rows_written, docid


def convert_parquet_to_training_parquet(
    input_path: str,
    output_path: str,
    narrative_col: str,
    source: str,
    docid_start: int,
    batch_size: int,
    compression: str,
) -> tuple[int, int]:
    """
    Convert one parquet file to training Parquet with schema:
      content_split: string
      meta: struct{source, docid, chunk_id, extra, aug_data, pos, max_pos}
    Returns (rows_written, next_docid).
    """
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    aug_data = _make_aug_data()
    schema = _training_schema()
    writer = pq.ParquetWriter(output_path, schema=schema, compression=compression)

    docid = docid_start
    rows_written = 0

    pbar = tqdm(
        total=total_rows,
        desc=f"Converting {os.path.basename(input_path)}",
        unit="rows",
        dynamic_ncols=True,
        disable=False,
    )

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            narratives = batch.column(narrative_col).to_pylist()
            out_rows = []
            for narrative in narratives:
                if not narrative:
                    docid += 1
                    pbar.update(1)
                    continue

                if not isinstance(narrative, str):
                    narrative = str(narrative)

                docid_str = str(docid)
                text_for_hash = docid_str + "/" + narrative
                chunk_id = generate_hash(text_for_hash)

                out_rows.append(
                    {
                        "content_split": narrative,
                        "meta": {
                            "source": source,
                            "docid": docid_str,
                            "chunk_id": chunk_id,
                            "extra": "{}",
                            "aug_data": aug_data,
                            "pos": 0,
                            "max_pos": 0,
                        },
                    }
                )
                rows_written += 1
                docid += 1
                pbar.update(1)

            if out_rows:
                writer.write_table(pa.Table.from_pylist(out_rows, schema=schema))
    finally:
        pbar.close()
        writer.close()

    return rows_written, docid


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert narrative Parquet to mariana training JSONL format. "
            "Supports single file or batch directory mode."
        )
    )

    # Input (mutually exclusive: single file or directory)
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Single input parquet file path")
    input_group.add_argument(
        "--input-dir",
        help="Directory containing parquet files (all *.parquet will be processed)",
    )
    input_group.add_argument(
        "--input-root",
        help=(
            "Root directory containing bucket subdirectories (e.g. under_128k/, 128k_512k/, 512k_2M/). "
            "Recursively finds *.parquet, skipping _resume/, and mirrors the directory structure under --output-dir."
        ),
    )

    # Output
    ap.add_argument("--output", help="Output JSONL file path (for single --input mode)")
    ap.add_argument(
        "--output-dir",
        help="Output directory for JSONL files (for --input-dir mode, "
        "or for single --input mode if --output is not specified)",
    )

    ap.add_argument(
        "--narrative-col",
        default="narrative",
        help="Column name for narrative text (default: %(default)s)",
    )
    ap.add_argument(
        "--source",
        default="deepwiki_narrative_20250213",
        help="Value for meta.source field (default: %(default)s)",
    )
    ap.add_argument(
        "--output-format",
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output format (default: %(default)s)",
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression when --output-format=parquet (default: %(default)s)",
    )
    ap.add_argument(
        "--docid-start",
        type=int,
        default=1,
        help="Starting docid (default: %(default)s)",
    )
    ap.add_argument(
        "--continuous-docid",
        action="store_true",
        default=True,
        help="Docid continues incrementing across files in --input-dir mode (default: on)",
    )
    ap.add_argument(
        "--reset-docid",
        action="store_false",
        dest="continuous_docid",
        help="Reset docid to --docid-start for each file in --input-dir mode",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet read batch size (default: %(default)s)",
    )
    args = ap.parse_args()

    if args.input:
        # Single file mode
        out = args.output
        if not out:
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(args.input))[0]
                ext = ".jsonl" if args.output_format == "jsonl" else ".parquet"
                out = os.path.join(args.output_dir, base + ext)
            else:
                ext = ".jsonl" if args.output_format == "jsonl" else ".parquet"
                out = os.path.splitext(args.input)[0] + ext

        if args.output_format == "jsonl":
            rows, _ = convert_parquet_to_jsonl(
                input_path=args.input,
                output_path=out,
                narrative_col=args.narrative_col,
                source=args.source,
                docid_start=args.docid_start,
                batch_size=args.batch_size,
            )
        else:
            rows, _ = convert_parquet_to_training_parquet(
                input_path=args.input,
                output_path=out,
                narrative_col=args.narrative_col,
                source=args.source,
                docid_start=args.docid_start,
                batch_size=args.batch_size,
                compression=args.compression,
            )
        print(f"\nDONE: {rows:,} rows -> {out}")

    elif args.input_dir:
        # Directory mode
        if not args.output_dir:
            raise SystemExit("--output-dir is required when using --input-dir")

        os.makedirs(args.output_dir, exist_ok=True)

        parquet_files = sorted(
            f
            for f in os.listdir(args.input_dir)
            if f.endswith(".parquet")
        )
        if not parquet_files:
            raise SystemExit(f"No .parquet files found in {args.input_dir}")

        print(f"Found {len(parquet_files)} parquet files in {args.input_dir}")

        total_rows = 0
        docid = args.docid_start

        for fname in parquet_files:
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(
                args.output_dir, os.path.splitext(fname)[0] + (".jsonl" if args.output_format == "jsonl" else ".parquet")
            )

            start_docid = docid if args.continuous_docid else args.docid_start
            if args.output_format == "jsonl":
                rows, next_docid = convert_parquet_to_jsonl(
                    input_path=in_path,
                    output_path=out_path,
                    narrative_col=args.narrative_col,
                    source=args.source,
                    docid_start=start_docid,
                    batch_size=args.batch_size,
                )
            else:
                rows, next_docid = convert_parquet_to_training_parquet(
                    input_path=in_path,
                    output_path=out_path,
                    narrative_col=args.narrative_col,
                    source=args.source,
                    docid_start=start_docid,
                    batch_size=args.batch_size,
                    compression=args.compression,
                )
            total_rows += rows
            if args.continuous_docid:
                docid = next_docid
            print(f"  {fname} -> {os.path.basename(out_path)} ({rows:,} rows)")

        print(f"\nDONE: {total_rows:,} total rows across {len(parquet_files)} files")
        print(f"Output dir: {args.output_dir}")
    else:
        # Root mode: recurse and mirror subdirs under output-dir
        if not args.output_dir:
            raise SystemExit("--output-dir is required when using --input-root")

        in_root = os.path.abspath(args.input_root)
        out_root = os.path.abspath(args.output_dir)
        os.makedirs(out_root, exist_ok=True)

        parquet_paths: list[str] = []
        for root, dirs, files in os.walk(in_root):
            # Skip internal resume/temp directories if present.
            rel_root = os.path.relpath(root, in_root)
            if rel_root == "_resume" or rel_root.startswith("_resume" + os.sep):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d != "_resume"]
            for fname in files:
                if fname.endswith(".parquet"):
                    parquet_paths.append(os.path.join(root, fname))

        parquet_paths.sort()
        if not parquet_paths:
            raise SystemExit(f"No .parquet files found under {in_root}")

        print(f"Found {len(parquet_paths)} parquet files under {in_root}")

        total_rows = 0
        docid = args.docid_start

        for in_path in parquet_paths:
            rel = os.path.relpath(in_path, in_root)
            rel_dir = os.path.dirname(rel)
            out_dir = os.path.join(out_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(in_path))[0] + (".jsonl" if args.output_format == "jsonl" else ".parquet"),
            )

            start_docid = docid if args.continuous_docid else args.docid_start
            if args.output_format == "jsonl":
                rows, next_docid = convert_parquet_to_jsonl(
                    input_path=in_path,
                    output_path=out_path,
                    narrative_col=args.narrative_col,
                    source=args.source,
                    docid_start=start_docid,
                    batch_size=args.batch_size,
                )
            else:
                rows, next_docid = convert_parquet_to_training_parquet(
                    input_path=in_path,
                    output_path=out_path,
                    narrative_col=args.narrative_col,
                    source=args.source,
                    docid_start=start_docid,
                    batch_size=args.batch_size,
                    compression=args.compression,
                )
            total_rows += rows
            if args.continuous_docid:
                docid = next_docid
            print(f"  {rel} -> {os.path.relpath(out_path, out_root)} ({rows:,} rows)")

        print(f"\nDONE: {total_rows:,} total rows across {len(parquet_paths)} files")
        print(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()
