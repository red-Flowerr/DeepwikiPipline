#!/usr/bin/env python3
"""
Summarize a bucketed tokenization run produced by utils/tokenize_and_bucket_parquet.py.

It reads per-input done markers under:
  <output_dir>/_resume/done/*.json

and aggregates per-bucket row/token stats into a single report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass


@dataclass
class BucketAgg:
    rows: int = 0
    tokens_sum: int = 0
    tokens_min: int | None = None
    tokens_max: int | None = None

    def add(self, rows: int, tokens_sum: int, tokens_min: int | None, tokens_max: int | None):
        self.rows += int(rows or 0)
        self.tokens_sum += int(tokens_sum or 0)
        if tokens_min is not None:
            self.tokens_min = tokens_min if self.tokens_min is None else min(self.tokens_min, int(tokens_min))
        if tokens_max is not None:
            self.tokens_max = tokens_max if self.tokens_max is None else max(self.tokens_max, int(tokens_max))


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate bucket/token stats from tokenize_and_bucket_parquet resume markers.")
    ap.add_argument("--output-dir", required=True, help="The same --output-dir used for tokenize_and_bucket_parquet.py")
    ap.add_argument("--write-json", default="", help="Optional path to write aggregated JSON summary (default: no write)")
    ap.add_argument("--strict-signature", action="store_true", default=False, help="Fail if multiple resume signatures are present.")
    args = ap.parse_args()

    resume_dir = os.path.join(args.output_dir, "_resume")
    done_dir = os.path.join(resume_dir, "done")
    claim_dir = os.path.join(resume_dir, "claim")

    done_files = sorted(glob.glob(os.path.join(done_dir, "*.json")))
    claim_files = sorted(glob.glob(os.path.join(claim_dir, "*.json"))) if os.path.isdir(claim_dir) else []

    if not done_files:
        raise SystemExit(f"No done markers found under: {done_dir}")

    signatures: dict[str, int] = {}
    buckets: dict[str, BucketAgg] = {}
    totals = {
        "done_files": 0,
        "claimed_files": len(claim_files),
        "skipped_overflow": 0,
        "errors": 0,
        "committed_bucket_files": 0,
        "seconds_sum": 0.0,
    }

    for p in done_files:
        meta = _load_json(p)
        if not meta:
            continue
        totals["done_files"] += 1
        sig = meta.get("resume_signature") or "unknown"
        signatures[sig] = signatures.get(sig, 0) + 1
        totals["skipped_overflow"] += int(meta.get("skipped_overflow", 0) or 0)
        totals["errors"] += int(meta.get("errors", 0) or 0)
        totals["committed_bucket_files"] += int(meta.get("committed_bucket_files", 0) or 0)
        try:
            totals["seconds_sum"] += float(meta.get("seconds", 0) or 0.0)
        except Exception:
            pass

        bstats = meta.get("bucket_stats")
        if not isinstance(bstats, dict):
            continue
        for name, st in bstats.items():
            if not isinstance(st, dict):
                continue
            agg = buckets.get(name)
            if agg is None:
                agg = BucketAgg()
                buckets[name] = agg
            agg.add(
                rows=int(st.get("rows", 0) or 0),
                tokens_sum=int(st.get("tokens_sum", 0) or 0),
                tokens_min=st.get("tokens_min", None),
                tokens_max=st.get("tokens_max", None),
            )

    if args.strict_signature and len(signatures) > 1:
        raise SystemExit(f"Multiple resume signatures present: {signatures}")

    # Print summary
    print("output_dir:", args.output_dir)
    print("done_markers:", totals["done_files"])
    print("claimed_markers:", totals["claimed_files"])
    print("resume_signatures:", signatures)
    print("skipped_overflow(>=2M):", f"{totals['skipped_overflow']:,}")
    print("errors:", totals["errors"])
    print("committed_bucket_files:", totals["committed_bucket_files"])
    print("seconds_sum:", round(totals["seconds_sum"], 3))
    print()

    grand_rows = sum(a.rows for a in buckets.values())
    grand_tokens = sum(a.tokens_sum for a in buckets.values())
    print(f"{'bucket':<16} {'rows':>12} {'tokens_sum':>18} {'min':>10} {'max':>10} {'%rows':>8} {'%tok':>8}")
    for name in sorted(buckets.keys()):
        a = buckets[name]
        pr = (a.rows / grand_rows * 100.0) if grand_rows else 0.0
        pt = (a.tokens_sum / grand_tokens * 100.0) if grand_tokens else 0.0
        print(
            f"{name:<16} {a.rows:>12,} {a.tokens_sum:>18,} "
            f"{(a.tokens_min if a.tokens_min is not None else '-'):>10} {(a.tokens_max if a.tokens_max is not None else '-'):>10} "
            f"{pr:>7.1f}% {pt:>7.1f}%"
        )
    print()
    print("TOTAL rows:", f"{grand_rows:,}")
    print("TOTAL tokens:", f"{grand_tokens:,}")

    if args.write_json:
        out = {
            "output_dir": args.output_dir,
            "resume_signatures": signatures,
            "totals": totals,
            "buckets": {
                name: {
                    "rows": a.rows,
                    "tokens_sum": a.tokens_sum,
                    "tokens_min": a.tokens_min,
                    "tokens_max": a.tokens_max,
                }
                for name, a in buckets.items()
            },
            "grand": {"rows": grand_rows, "tokens_sum": grand_tokens},
        }
        os.makedirs(os.path.dirname(args.write_json) or ".", exist_ok=True)
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("wrote:", args.write_json)


if __name__ == "__main__":
    main()

