"""
Split a Parquet file (with a token-count column) into multiple files
based on configurable token-count buckets.

Example
-------
python utils/split_parquet_by_tokens.py \
    --input  narratives.parquet \
    --output-dir  output_buckets/ \
    --tokens-col narrative_tokens \
    --buckets 0:131072:under_128k  131072:524288:128k_512k  524288:2097152:512k_2M
"""

import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def parse_bucket(spec: str):
    """Parse a bucket spec  'lo:hi:name'  (hi can be 'inf')."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"bucket spec must be  lo:hi:name  (got {spec!r}). "
            "Use 'inf' for unbounded upper limit."
        )
    lo_s, hi_s, name = parts
    lo = int(lo_s)
    hi = float("inf") if hi_s.lower() == "inf" else int(hi_s)
    if lo >= hi:
        raise ValueError(f"lo ({lo}) must be < hi ({hi}) in bucket {spec!r}")
    return lo, hi, name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a Parquet into multiple files by token-count buckets."
    )
    ap.add_argument("--input", required=True, help="Input parquet file path")
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write bucket parquet files",
    )
    ap.add_argument(
        "--tokens-col",
        default="narrative_tokens",
        help="Column name that holds token counts (default: %(default)s)",
    )
    ap.add_argument(
        "--buckets",
        nargs="+",
        default=[
            "0:131072:under_128k",
            "131072:524288:128k_512k",
            "524288:2097152:512k_2M",
        ],
        help=(
            "Bucket definitions as lo:hi:name (hi can be 'inf'). "
            "Default: 0:131072:under_128k  131072:524288:128k_512k  524288:2097152:512k_2M"
        ),
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec (default: %(default)s)",
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix for output parquet files",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Parse buckets
    # ------------------------------------------------------------------
    buckets = [parse_bucket(b) for b in args.buckets]
    print(f"Buckets ({len(buckets)}):")
    for lo, hi, name in buckets:
        hi_label = "inf" if hi == float("inf") else f"{hi:,}"
        print(f"  [{lo:>10,}, {hi_label:>10})  ->  {name}")

    # ------------------------------------------------------------------
    # Read input (supports single .parquet or directory of shards)
    # ------------------------------------------------------------------
    input_path = args.input
    if os.path.isdir(input_path):
        import pyarrow.dataset as ds

        dataset = ds.dataset(input_path, format="parquet")
        table = dataset.to_table()
        print(f"\nInput (dir): {input_path}  ({table.num_rows:,} rows from {len(dataset.files)} files)")
    else:
        table = pq.read_table(input_path)
        print(f"\nInput: {input_path}  ({table.num_rows:,} rows)")

    if args.tokens_col not in table.column_names:
        raise SystemExit(
            f"Column {args.tokens_col!r} not found. "
            f"Available: {table.column_names}"
        )

    total_rows = table.num_rows

    tokens = table.column(args.tokens_col)

    # ------------------------------------------------------------------
    # Split & write
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    assigned_total = 0
    summary = []

    for lo, hi, name in buckets:
        # Build boolean mask: lo <= tokens < hi
        mask_lo = pa.compute.greater_equal(tokens, pa.scalar(lo, type=pa.int64()))
        if hi == float("inf"):
            mask = mask_lo
        else:
            mask_hi = pa.compute.less(tokens, pa.scalar(int(hi), type=pa.int64()))
            mask = pa.compute.and_(mask_lo, mask_hi)

        subset = table.filter(mask)
        n = subset.num_rows

        if n == 0:
            print(f"  {name}: 0 rows (skip)")
            summary.append((name, lo, hi, 0, 0))
            continue

        # Compute token stats for this bucket
        subset_tokens = subset.column(args.tokens_col)
        tok_sum = pa.compute.sum(subset_tokens).as_py()
        tok_min = pa.compute.min(subset_tokens).as_py()
        tok_max = pa.compute.max(subset_tokens).as_py()
        tok_mean = tok_sum / n if n > 0 else 0

        fname = f"{args.prefix}{name}.parquet" if args.prefix else f"{name}.parquet"
        out_path = os.path.join(args.output_dir, fname)
        pq.write_table(subset, out_path, compression=args.compression)

        print(
            f"  {name}: {n:,} rows  |  tokens: sum={tok_sum:,}  "
            f"min={tok_min:,}  max={tok_max:,}  mean={tok_mean:,.0f}  "
            f"->  {out_path}"
        )
        summary.append((name, lo, hi, n, tok_sum))
        assigned_total += n

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    unassigned = total_rows - assigned_total
    print(f"\n{'=' * 70}")
    print(f"{'Bucket':<20} {'Rows':>10} {'Tokens':>18} {'% Rows':>8} {'% Tokens':>10}")
    print(f"{'-' * 70}")
    grand_tokens = sum(t for _, _, _, _, t in summary)
    for name, lo, hi, n, tok in summary:
        pct_rows = (n / total_rows * 100) if total_rows else 0
        pct_tok = (tok / grand_tokens * 100) if grand_tokens else 0
        print(f"{name:<20} {n:>10,} {tok:>18,} {pct_rows:>7.1f}% {pct_tok:>9.1f}%")
    print(f"{'-' * 70}")
    print(f"{'TOTAL':<20} {assigned_total:>10,} {grand_tokens:>18,}")
    if unassigned > 0:
        print(f"{'(unassigned)':<20} {unassigned:>10,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
