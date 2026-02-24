"""
Convert DeepWiki training parquet into midtrain parquet format and recompute token counts.

Input
-----
An input root directory containing multiple subfolders (e.g. under_128k/, 128k_512k/, 512k_2M/),
each containing many *.parquet files.

This script processes each parquet file in a streaming manner (batch-by-batch) and writes an
output parquet file with the required schema:

  - meta: struct { docid: string, chunk_id: string, source: string }
  - content_split: large_string
  - token_count: int64

Field mapping rules (per row)
-----------------------------
docid:
  - prefer column 'prompt_id' if present
  - else prefer meta.docid if meta struct exists
  - else prefer column 'docid' if present

chunk_id:
  - prefer column 'response_id' if present
  - else prefer meta.chunk_id if meta struct exists
  - else prefer column 'chunk_id' if present

source:
  - if --source is provided, use it
  - else prefer meta.source if meta struct exists
  - else prefer column 'source' if present
  - else empty string

Token counting
--------------
Uses a HuggingFace tokenizer directory containing tokenizer.json (preferred) or
loads via transformers as a fallback.

Example
-------
python utils/convert_midtrain_parquet.py \
  --input-root /mnt/hdfs/byte_data_seed_wl_write/user/xingtianshun/deepwiki_handover/0213_training_parquet \
  --output-root /mnt/hdfs/byte_data_seed_wl_write/user/xingtianshun/deepwiki_handover/0224_mt_train_parquet \
  --tokenizer /mnt/hdfs/byte_data_seed_wl/user/fangjunjie.99/tokenizers/bbpe155k-v6.4.3-ml.pret_v5.7_20251015 \
  --workers 16 --batch-size 64
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import warnings
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def _list_parquet_files(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    files = sorted(glob.glob(os.path.join(dir_path, "*.parquet")))
    if files:
        return files
    files = sorted(glob.glob(os.path.join(dir_path, "part-*")))
    return [f for f in files if os.path.isfile(f)]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _out_schema() -> pa.Schema:
    return pa.schema(
        [
            (
                "meta",
                pa.struct(
                    [
                        ("docid", pa.string()),
                        ("chunk_id", pa.string()),
                        ("source", pa.string()),
                    ]
                ),
            ),
            ("content_split", pa.large_string()),
            ("token_count", pa.int64()),
        ]
    )


@dataclass(frozen=True)
class TokenizerConfig:
    path: str


_TOKENIZER = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    _TOKENIZER = _load_tokenizer(TokenizerConfig(path=tokenizer_path))


def _load_tokenizer(cfg: TokenizerConfig):
    tok_json = os.path.join(cfg.path, "tokenizer.json")
    if os.path.isfile(tok_json):
        from tokenizers import Tokenizer

        return ("tokenizers", Tokenizer.from_file(tok_json))

    from transformers import AutoTokenizer

    # Avoid warning spam; we only count tokens.
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than*")

    tok = AutoTokenizer.from_pretrained(
        cfg.path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    try:
        tok.model_max_length = 1 << 60
    except Exception:
        pass
    return ("transformers", tok)


def _count_tokens_batch(texts: list[str]) -> list[int]:
    if _TOKENIZER is None:
        raise RuntimeError("Tokenizer is not initialized")

    backend, tok = _TOKENIZER
    if backend == "tokenizers":
        # Avoid counting special tokens for token_count.
        encs = tok.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encs]
    if backend == "transformers":
        out = tok(texts, add_special_tokens=False)
        return [len(ids) for ids in out["input_ids"]]
    raise RuntimeError(f"unknown tokenizer backend: {backend}")


def _extract_column_as_pylist(rb: pa.RecordBatch, name: str):
    idx = rb.schema.get_field_index(name)
    if idx < 0:
        return None
    return rb.column(idx).to_pylist()


def _extract_meta_struct_as_pylist(rb: pa.RecordBatch, meta_col: str) -> list[dict] | None:
    idx = rb.schema.get_field_index(meta_col)
    if idx < 0:
        return None
    col = rb.column(idx)
    if not pa.types.is_struct(col.type):
        return None
    return col.to_pylist()


def convert_one_file(
    *,
    input_path: str,
    output_path: str,
    batch_size: int,
    content_col: str,
    meta_col: str,
    source_override: str,
    compression: str,
    overwrite: bool,
) -> tuple[int, int]:
    if (not overwrite) and os.path.exists(output_path):
        return (0, 0)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    _ensure_dir(os.path.dirname(output_path) or ".")

    out_schema = _out_schema()
    writer = pq.ParquetWriter(output_path, out_schema, compression=compression)

    rows_written = 0
    tokens_total = 0

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            rb = pa.RecordBatch.from_struct_array(pa.StructArray.from_arrays(batch.columns, names=batch.schema.names))

            # Content
            contents = _extract_column_as_pylist(rb, content_col)
            if contents is None:
                raise ValueError(f"missing content column '{content_col}' in {input_path}")

            # Optional sources of ids
            prompt_ids = _extract_column_as_pylist(rb, "prompt_id")
            response_ids = _extract_column_as_pylist(rb, "response_id")
            docids = _extract_column_as_pylist(rb, "docid")
            chunk_ids = _extract_column_as_pylist(rb, "chunk_id")
            sources = _extract_column_as_pylist(rb, "source")
            meta_rows = _extract_meta_struct_as_pylist(rb, meta_col)

            # Tokenize batch
            texts_for_tok = [c if isinstance(c, str) else ("" if c is None else str(c)) for c in contents]
            tok_counts = _count_tokens_batch(texts_for_tok)

            out_docid: list[str] = []
            out_chunk_id: list[str] = []
            out_source: list[str] = []
            out_content: list[str | None] = []
            out_tok: list[int] = []

            for i, c in enumerate(contents):
                meta_i = meta_rows[i] if meta_rows is not None else None

                if prompt_ids is not None:
                    docid_val = prompt_ids[i]
                elif meta_i is not None and isinstance(meta_i, dict) and "docid" in meta_i:
                    docid_val = meta_i.get("docid")
                elif docids is not None:
                    docid_val = docids[i]
                else:
                    docid_val = ""

                if response_ids is not None:
                    chunk_id_val = response_ids[i]
                elif meta_i is not None and isinstance(meta_i, dict) and "chunk_id" in meta_i:
                    chunk_id_val = meta_i.get("chunk_id")
                elif chunk_ids is not None:
                    chunk_id_val = chunk_ids[i]
                else:
                    chunk_id_val = ""

                if source_override:
                    source_val = source_override
                elif meta_i is not None and isinstance(meta_i, dict) and "source" in meta_i:
                    source_val = meta_i.get("source")
                elif sources is not None:
                    source_val = sources[i]
                else:
                    source_val = ""

                out_docid.append(_safe_str(docid_val))
                out_chunk_id.append(_safe_str(chunk_id_val))
                out_source.append(_safe_str(source_val))
                out_content.append(c if (c is None or isinstance(c, str)) else str(c))

                tc = int(tok_counts[i]) if i < len(tok_counts) else 0
                out_tok.append(tc)
                tokens_total += tc

            meta_arr = pa.StructArray.from_arrays(
                [
                    pa.array(out_docid, type=pa.string()),
                    pa.array(out_chunk_id, type=pa.string()),
                    pa.array(out_source, type=pa.string()),
                ],
                fields=list(out_schema.field("meta").type),
            )
            out_rb = pa.RecordBatch.from_arrays(
                [
                    meta_arr,
                    pa.array(out_content, type=pa.large_string()),
                    pa.array(out_tok, type=pa.int64()),
                ],
                schema=out_schema,
            )
            writer.write_table(pa.Table.from_batches([out_rb], schema=out_schema))
            rows_written += out_rb.num_rows
    finally:
        writer.close()

    return rows_written, tokens_total


def _job_output_path(input_root: str, output_root: str, input_path: str) -> str:
    rel = os.path.relpath(input_path, start=input_root)
    return os.path.join(output_root, rel)


def _worker_run(job):
    try:
        rows, toks = convert_one_file(**job)
        return ("ok", job["input_path"], rows, toks, "")
    except Exception as e:
        return ("err", job["input_path"], 0, 0, repr(e))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert parquet folders to midtrain parquet schema with token_count.")
    ap.add_argument("--input-root", required=True, help="Input dataset root containing multiple subfolders")
    ap.add_argument("--output-root", required=True, help="Output dataset root (will mirror input structure)")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer directory path (must exist locally)")

    ap.add_argument("--content-col", default="content_split", help="Content text column name (default: %(default)s)")
    ap.add_argument("--meta-col", default="meta", help="Meta struct column name (default: %(default)s)")
    ap.add_argument("--source", default="", help="Override meta.source for all rows (default: keep input)")

    ap.add_argument("--batch-size", type=int, default=64, help="Rows per batch (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=8, help="Worker processes (default: %(default)s)")
    ap.add_argument(
        "--mp-start",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: %(default)s)",
    )
    ap.add_argument("--compression", default="zstd", help="Parquet compression (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument(
        "--subdirs",
        default="",
        help="Comma-separated subdir whitelist under input-root (default: all discovered subdirs)",
    )
    ap.add_argument("--max-files", type=int, default=0, help="Process at most N files total (default: 0 = no limit)")
    ap.add_argument("--pool-chunksize", type=int, default=1, help="multiprocessing imap chunksize (default: %(default)s)")
    args = ap.parse_args()

    if not os.path.isdir(args.input_root):
        raise SystemExit(f"input root not found: {args.input_root}")
    if not os.path.isdir(args.tokenizer):
        raise SystemExit(f"tokenizer path not found: {args.tokenizer}")
    _ensure_dir(args.output_root)

    # Discover subfolders under input root and list parquet files under each.
    subdirs = [d for d in sorted(os.listdir(args.input_root)) if os.path.isdir(os.path.join(args.input_root, d))]
    if not subdirs:
        raise SystemExit(f"no subfolders found under input root: {args.input_root}")

    if args.subdirs:
        allow = {s.strip() for s in args.subdirs.split(",") if s.strip()}
        subdirs = [s for s in subdirs if s in allow]
        if not subdirs:
            raise SystemExit(f"--subdirs matched nothing under input root: {args.input_root}")

    jobs = []
    for sub in subdirs:
        in_dir = os.path.join(args.input_root, sub)
        files = _list_parquet_files(in_dir)
        if not files:
            continue
        for fp in files:
            out_fp = _job_output_path(args.input_root, args.output_root, fp)
            _ensure_dir(os.path.dirname(out_fp) or ".")
            jobs.append(
                dict(
                    input_path=fp,
                    output_path=out_fp,
                    batch_size=args.batch_size,
                    content_col=args.content_col,
                    meta_col=args.meta_col,
                    source_override=args.source,
                    compression=args.compression,
                    overwrite=bool(args.overwrite),
                )
            )
            if args.max_files and len(jobs) >= args.max_files:
                break
        if args.max_files and len(jobs) >= args.max_files:
            break

    if not jobs:
        raise SystemExit("no parquet files found to process")

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        pbar = tqdm(total=len(jobs), desc="Converting parquet files", unit="file", dynamic_ncols=True)
        ok_files = 0
        err_files = 0
        rows_total = 0
        tokens_total = 0
        try:
            for status, fp, rows, toks, err in pool.imap_unordered(_worker_run, jobs, chunksize=args.pool_chunksize):
                if status == "ok":
                    ok_files += 1
                    rows_total += int(rows)
                    tokens_total += int(toks)
                else:
                    err_files += 1
                    print(f"\nERROR {fp}: {err}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=ok_files, err=err_files, rows=rows_total, tokens=tokens_total)
        finally:
            pbar.close()

    if err_files:
        raise SystemExit(f"completed with errors: ok_files={ok_files} err_files={err_files}")

    print("\nDONE")
    print("input_root:", args.input_root)
    print("output_root:", args.output_root)
    print("files:", ok_files)
    print("rows:", rows_total)
    print("total_tokens:", tokens_total)


if __name__ == "__main__":
    main()
