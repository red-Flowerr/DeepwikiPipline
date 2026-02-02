import argparse
import json
import os
import sys

import pyarrow.parquet as pq
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read a parquet and export repo_name (or repo) to narrative_tokens mapping as JSONL."
    )
    ap.add_argument("--input", required=True, help="Input parquet path")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    ap.add_argument("--repo-col", default="repo_name", help="Repo column name (default: %(default)s)")
    ap.add_argument("--tokens-col", default="narrative_tokens", help="Tokens column name (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=4096, help="Rows per batch (default: %(default)s)")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.input)
    schema_names = set(pf.schema_arrow.names)
    repo_col = args.repo_col
    if repo_col not in schema_names:
        # fallback for your exported parquet which uses 'repo'
        if "repo" in schema_names:
            repo_col = "repo"
        else:
            raise SystemExit(f"Repo column not found. Tried {args.repo_col}, and fallback 'repo'. Columns={pf.schema_arrow.names}")
    if args.tokens_col not in schema_names:
        raise SystemExit(f"Tokens column not found: {args.tokens_col}. Columns={pf.schema_arrow.names}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_rows = pf.metadata.num_rows
    wrote = 0

    pbar = tqdm(
        total=total_rows,
        desc="Exporting repo->tokens",
        unit="rows",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    with open(args.output, "w", encoding="utf-8") as out:
        for batch in pf.iter_batches(columns=[repo_col, args.tokens_col], batch_size=args.batch_size):
            repos = batch.column(0).to_pylist()
            toks = batch.column(1).to_pylist()
            for r, t in zip(repos, toks):
                if r is None:
                    continue
                row = {"repo_name": str(r), args.tokens_col: int(t or 0)}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
            pbar.update(batch.num_rows)
            pbar.set_postfix(wrote=wrote)
    pbar.close()

    print("\nDONE")
    print("input:", args.input)
    print("output:", args.output)
    print("repo_col:", repo_col)
    print("tokens_col:", args.tokens_col)
    print("rows_total:", total_rows)
    print("rows_written:", wrote)


if __name__ == "__main__":
    main()

