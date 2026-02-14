"""
Tokenize 'narrative' in Parquet and split into token buckets, streaming.

Why this exists
---------------
The exported narratives are extremely large (often ~1M tokens/row). Loading
everything into memory (pyarrow.dataset.to_table) is not feasible. This script
processes Parquet files in batches, computes token counts, and writes rows into
bucketed Parquet shards incrementally.

Output layout
-------------
output_dir/
  under_128k/part0000.parquet
  128k_512k/part0000.parquet
  512k_2M/part0000.parquet
  over_2M/part0000.parquet     (optional, if --keep-overflow)
  meta.json                    (summary)

Example
-------
python utils/tokenize_and_bucket_parquet.py \
  --input-glob "/path/0213_all_narratives.worker*.part*.parquet" \
  --output-dir "/path/buckets_0213" \
  --tokenizer-backend tiktoken --encoding cl100k_base \
  --batch-size 1 --rows-per-shard 200
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def make_token_counter(
    tokenizer_backend: str,
    tiktoken_encoding: str,
    hf_model: str,
    hf_trust_remote_code: bool,
    hf_local_files_only: bool,
):
    if tokenizer_backend == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(tiktoken_encoding)

        def count_tokens(s: str) -> int:
            return len(enc.encode_ordinary(s))

        return count_tokens

    if tokenizer_backend == "hf":
        from transformers import AutoTokenizer

        if not hf_model:
            raise ValueError("--hf-model is required when --tokenizer-backend=hf")

        try:
            tok = AutoTokenizer.from_pretrained(
                hf_model,
                trust_remote_code=hf_trust_remote_code,
                local_files_only=hf_local_files_only,
                use_fast=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                hf_model,
                trust_remote_code=hf_trust_remote_code,
                local_files_only=hf_local_files_only,
                use_fast=False,
            )

        try:
            tok.model_max_length = 1 << 60
        except Exception:
            pass

        def count_tokens(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        return count_tokens

    raise ValueError(f"unknown tokenizer backend: {tokenizer_backend}")


@dataclass(frozen=True)
class Bucket:
    name: str
    lo: int
    hi: int | None  # None = infinity

    def contains(self, n: int) -> bool:
        if n < self.lo:
            return False
        if self.hi is None:
            return True
        return n < self.hi


def default_buckets() -> list[Bucket]:
    return [
        Bucket("under_128k", 0, 131072),
        Bucket("128k_512k", 131072, 524288),
        Bucket("512k_2M", 524288, 2097152),
    ]


def make_schema_with_tokens(in_schema: pa.Schema, tokens_col: str) -> pa.Schema:
    if tokens_col in set(in_schema.names):
        # Replace type with int64 if already exists
        fields = []
        for f in in_schema:
            if f.name == tokens_col:
                fields.append(pa.field(tokens_col, pa.int64()))
            else:
                fields.append(f)
        return pa.schema(fields)
    return pa.schema(list(in_schema) + [pa.field(tokens_col, pa.int64())])


class BucketShardWriter:
    def __init__(
        self,
        *,
        out_dir: str,
        schema: pa.Schema,
        rows_per_shard: int,
        batch_size: int,
        compression: str,
    ):
        self.out_dir = out_dir
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.batch_size = batch_size
        self.compression = compression

        os.makedirs(out_dir, exist_ok=True)
        self._buf: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._shard_idx = 0
        self._rows_in_shard = 0
        self._total_rows = 0

    def _open(self):
        if self._writer is not None:
            return
        path = os.path.join(self.out_dir, f"part{self._shard_idx:04d}.parquet")
        self._writer = pq.ParquetWriter(path, schema=self.schema, compression=self.compression)

    def _close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self):
        if not self._buf:
            return
        self._open()
        n = len(self._buf)
        table = pa.Table.from_pylist(self._buf, schema=self.schema)
        self._writer.write_table(table)
        self._buf.clear()
        self._rows_in_shard += n
        self._total_rows += n
        if self.rows_per_shard > 0 and self._rows_in_shard >= self.rows_per_shard:
            self._close()
            self._shard_idx += 1
            self._rows_in_shard = 0

    def add(self, row: dict):
        self._buf.append(row)
        if len(self._buf) >= self.batch_size:
            self._flush()

    def close(self):
        self._flush()
        self._close()

    @property
    def total_rows(self) -> int:
        return self._total_rows + len(self._buf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute narrative token counts and bucket rows by token ranges.")
    ap.add_argument("--input-glob", required=True, help="Glob for input parquet files")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--narrative-col", default="narrative")
    ap.add_argument("--tokens-col", default="narrative_tokens")
    ap.add_argument("--batch-size", type=int, default=1, help="Rows per processing batch (default: %(default)s)")
    ap.add_argument("--rows-per-shard", type=int, default=200, help="Rows per output shard file (default: %(default)s)")
    ap.add_argument("--write-batch", type=int, default=8, help="Rows buffered before writing per bucket (default: %(default)s)")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--keep-overflow", action="store_true", default=False, help="Keep rows >= 2M tokens in 'over_2M'")

    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"])
    ap.add_argument("--encoding", default="cl100k_base")
    ap.add_argument("--hf-model", default="")
    ap.add_argument("--hf-trust-remote-code", action="store_true")
    ap.add_argument("--hf-local-files-only", action="store_true", default=True)
    ap.add_argument("--hf-allow-download", action="store_false", dest="hf_local_files_only")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No input files matched: {args.input_glob!r}")

    count_tokens = make_token_counter(
        tokenizer_backend=args.tokenizer_backend,
        tiktoken_encoding=args.encoding,
        hf_model=args.hf_model,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_local_files_only=args.hf_local_files_only,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    buckets = default_buckets()
    overflow_bucket = Bucket("over_2M", 2097152, None)
    if args.keep_overflow:
        buckets_all = buckets + [overflow_bucket]
    else:
        buckets_all = buckets

    writers: dict[str, BucketShardWriter] = {}
    schema: pa.Schema | None = None

    # Stats
    stats = {b.name: {"rows": 0, "tokens_sum": 0, "tokens_min": None, "tokens_max": None} for b in buckets_all}
    stats["skipped_overflow"] = 0
    stats["errors"] = 0

    t0 = time.time()
    pbar = tqdm(total=len(files), desc="Tokenizing parquet files", unit="file", dynamic_ncols=True)
    try:
        for fp in files:
            pf = pq.ParquetFile(fp)
            if schema is None:
                schema = make_schema_with_tokens(pf.schema_arrow, args.tokens_col)
                for b in buckets_all:
                    writers[b.name] = BucketShardWriter(
                        out_dir=os.path.join(args.output_dir, b.name),
                        schema=schema,
                        rows_per_shard=int(args.rows_per_shard),
                        batch_size=int(args.write_batch),
                        compression=args.compression,
                    )

            try:
                for batch in pf.iter_batches(batch_size=int(args.batch_size)):
                    rb = pa.RecordBatch.from_struct_array(
                        pa.StructArray.from_arrays(batch.columns, names=batch.schema.names)
                    )
                    col_idx = rb.schema.get_field_index(args.narrative_col)
                    narratives = rb.column(col_idx).to_pylist()
                    for i, s in enumerate(narratives):
                        # Build row dict from the record batch row (cheap for small batch-size)
                        row = {name: rb.column(j)[i].as_py() for j, name in enumerate(rb.schema.names)}
                        if not s:
                            tok = 0
                        else:
                            if not isinstance(s, str):
                                s = str(s)
                            tok = int(count_tokens(s))

                        row[args.tokens_col] = tok

                        bucket_name = None
                        for b in buckets:
                            if b.contains(tok):
                                bucket_name = b.name
                                break
                        if bucket_name is None:
                            if args.keep_overflow:
                                bucket_name = overflow_bucket.name
                            else:
                                stats["skipped_overflow"] += 1
                                continue

                        writers[bucket_name].add(row)
                        st = stats[bucket_name]
                        st["rows"] += 1
                        st["tokens_sum"] += tok
                        st["tokens_min"] = tok if st["tokens_min"] is None else min(st["tokens_min"], tok)
                        st["tokens_max"] = tok if st["tokens_max"] is None else max(st["tokens_max"], tok)
            except Exception as e:
                stats["errors"] += 1
                print(f"\nERROR reading {fp}: {e!r}", file=sys.stderr, flush=True)

            pbar.update(1)
            pbar.set_postfix(errors=stats["errors"])
    finally:
        pbar.close()
        for w in writers.values():
            w.close()

    meta = {
        "input_glob": args.input_glob,
        "files": len(files),
        "output_dir": args.output_dir,
        "buckets": [{"name": b.name, "lo": b.lo, "hi": b.hi} for b in buckets_all],
        "skipped_overflow": stats["skipped_overflow"],
        "errors": stats["errors"],
        "stats": stats,
        "seconds": round(time.time() - t0, 3),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    for b in buckets_all:
        st = stats[b.name]
        print(
            f"{b.name}: rows={st['rows']:,} "
            f"tokens_sum={st['tokens_sum']:,} "
            f"min={st['tokens_min']} max={st['tokens_max']}"
        )
    if not args.keep_overflow:
        print(f"skipped_overflow(>=2M): {stats['skipped_overflow']:,}")
    print("meta:", os.path.join(args.output_dir, "meta.json"))


if __name__ == "__main__":
    main()

