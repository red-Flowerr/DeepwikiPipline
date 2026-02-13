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
"""

import argparse
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq
from tqdm import tqdm


def generate_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
                out = os.path.join(args.output_dir, base + ".jsonl")
            else:
                out = os.path.splitext(args.input)[0] + ".jsonl"

        rows, _ = convert_parquet_to_jsonl(
            input_path=args.input,
            output_path=out,
            narrative_col=args.narrative_col,
            source=args.source,
            docid_start=args.docid_start,
            batch_size=args.batch_size,
        )
        print(f"\nDONE: {rows:,} rows -> {out}")

    else:
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
                args.output_dir, os.path.splitext(fname)[0] + ".jsonl"
            )

            start_docid = docid if args.continuous_docid else args.docid_start
            rows, next_docid = convert_parquet_to_jsonl(
                input_path=in_path,
                output_path=out_path,
                narrative_col=args.narrative_col,
                source=args.source,
                docid_start=start_docid,
                batch_size=args.batch_size,
            )
            total_rows += rows
            if args.continuous_docid:
                docid = next_docid
            print(f"  {fname} -> {os.path.basename(out_path)} ({rows:,} rows)")

        print(f"\nDONE: {total_rows:,} total rows across {len(parquet_files)} files")
        print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
