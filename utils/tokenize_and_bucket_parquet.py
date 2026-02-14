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
from hashlib import sha1

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


def _sanitize_filename(s: str) -> str:
    # Keep filesystem-friendly names; used only for output filenames.
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("._") or "file"
    return name[:120]


def _file_id(path: str) -> str:
    return sha1(path.encode("utf-8", errors="replace")).hexdigest()[:16]


def _resume_signature(args) -> str:
    """
    Signature of content-affecting settings. If these change, resume should not skip
    previously completed inputs (to avoid mixing incompatible outputs).
    """
    payload = {
        "narrative_col": args.narrative_col,
        "tokens_col": args.tokens_col,
        "keep_overflow": bool(args.keep_overflow),
        "tokenizer_backend": args.tokenizer_backend,
        "encoding": args.encoding,
        "hf_model": args.hf_model,
        "hf_trust_remote_code": bool(args.hf_trust_remote_code),
        "hf_local_files_only": bool(args.hf_local_files_only),
        "buckets": [(b.name, b.lo, b.hi) for b in default_buckets()],
    }
    return sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _safe_read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class BucketFileWriter:
    def __init__(
        self,
        *,
        tmp_path: str,
        schema: pa.Schema,
        batch_size: int,
        compression: str,
    ):
        self.tmp_path = tmp_path
        self.schema = schema
        self.batch_size = batch_size
        self.compression = compression

        _ensure_dir(os.path.dirname(tmp_path) or ".")
        self._buf: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._total_rows = 0

    def _open(self):
        if self._writer is not None:
            return
        self._writer = pq.ParquetWriter(self.tmp_path, schema=self.schema, compression=self.compression)

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
        self._total_rows += n

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
    ap.add_argument("--write-batch", type=int, default=8, help="Rows buffered before writing per bucket (default: %(default)s)")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--keep-overflow", action="store_true", default=False, help="Keep rows >= 2M tokens in 'over_2M'")
    ap.add_argument(
        "--heartbeat-secs",
        type=int,
        default=30,
        help="Print periodic progress while processing a file (0 disables). Default: %(default)s",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip input files already completed in this output dir (uses output-dir/_resume/done markers).",
    )

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

    resume_dir = os.path.join(args.output_dir, "_resume")
    done_dir = os.path.join(resume_dir, "done")
    tmp_dir = os.path.join(resume_dir, "tmp")
    if args.resume:
        _ensure_dir(done_dir)
        _ensure_dir(tmp_dir)

    run_sig = _resume_signature(args)

    schema: pa.Schema | None = None

    # Stats
    stats = {b.name: {"rows": 0, "tokens_sum": 0, "tokens_min": None, "tokens_max": None} for b in buckets_all}
    stats["skipped_overflow"] = 0
    stats["errors"] = 0
    stats["skipped_done"] = 0

    t0 = time.time()
    pbar = tqdm(total=len(files), desc="Tokenizing parquet files", unit="file", dynamic_ncols=True)
    try:
        for fp in files:
            fid = _file_id(fp)
            st = None
            try:
                st = os.stat(fp)
            except OSError:
                st = None

            done_path = os.path.join(done_dir, f"{fid}.json")
            if args.resume and os.path.exists(done_path):
                meta = _safe_read_json(done_path) or {}
                if meta.get("resume_signature") == run_sig:
                    # If we have file stat info, verify the input didn't change.
                    ok_to_skip = True
                    if st is not None:
                        if meta.get("input_size") != st.st_size or meta.get("input_mtime_ns") != st.st_mtime_ns:
                            ok_to_skip = False
                    if ok_to_skip:
                        stats["skipped_done"] += 1
                        # Add bucket stats from marker (so meta.json is complete across resumes).
                        bstats = meta.get("bucket_stats")
                        if isinstance(bstats, dict):
                            for b in buckets_all:
                                v = bstats.get(b.name)
                                if isinstance(v, dict):
                                    bst = stats[b.name]
                                    bst["rows"] += int(v.get("rows", 0) or 0)
                                    bst["tokens_sum"] += int(v.get("tokens_sum", 0) or 0)
                                    if v.get("tokens_min") is not None:
                                        bst["tokens_min"] = v["tokens_min"] if bst["tokens_min"] is None else min(bst["tokens_min"], v["tokens_min"])
                                    if v.get("tokens_max") is not None:
                                        bst["tokens_max"] = v["tokens_max"] if bst["tokens_max"] is None else max(bst["tokens_max"], v["tokens_max"])
                            stats["skipped_overflow"] += int(meta.get("skipped_overflow", 0) or 0)
                            stats["errors"] += int(meta.get("errors", 0) or 0)
                        print(f"[skip] {os.path.basename(fp)} (done)", file=sys.stderr, flush=True)
                        pbar.update(1)
                        pbar.set_postfix(errors=stats["errors"], skipped=stats["skipped_done"])
                        continue

            pf = pq.ParquetFile(fp)
            if schema is None:
                schema = make_schema_with_tokens(pf.schema_arrow, args.tokens_col)
                # Ensure bucket dirs exist up front.
                for b in buckets_all:
                    _ensure_dir(os.path.join(args.output_dir, b.name))

            file_rows = pf.metadata.num_rows if pf.metadata is not None else None
            file_rows_done = 0
            file_tokens_done = 0
            file_t_start = time.time()
            next_hb = file_t_start + max(1, int(args.heartbeat_secs)) if int(args.heartbeat_secs) > 0 else None
            if file_rows is not None:
                print(f"[file] {os.path.basename(fp)} rows={file_rows:,}", file=sys.stderr, flush=True)
            else:
                print(f"[file] {os.path.basename(fp)}", file=sys.stderr, flush=True)

            # Per-file transactional outputs: write to tmp then atomically move to final on success.
            safe_base = _sanitize_filename(os.path.splitext(os.path.basename(fp))[0])
            file_tmp_root = os.path.join(tmp_dir, fid)
            if args.resume:
                try:
                    # Clean up any partial tmp outputs from prior crashes.
                    if os.path.isdir(file_tmp_root):
                        for root, _dirs, files2 in os.walk(file_tmp_root, topdown=False):
                            for name in files2:
                                try:
                                    os.remove(os.path.join(root, name))
                                except OSError:
                                    pass
                            try:
                                os.rmdir(root)
                            except OSError:
                                pass
                except Exception:
                    pass
            _ensure_dir(file_tmp_root)

            writers: dict[str, BucketFileWriter] = {}
            bucket_stats = {b.name: {"rows": 0, "tokens_sum": 0, "tokens_min": None, "tokens_max": None} for b in buckets_all}
            skipped_overflow_local = 0
            errors_local = 0

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
                        file_rows_done += 1
                        file_tokens_done += tok

                        if next_hb is not None and time.time() >= next_hb:
                            elapsed = time.time() - file_t_start
                            rps = (file_rows_done / elapsed) if elapsed > 0 else 0.0
                            tps = (file_tokens_done / elapsed) if elapsed > 0 else 0.0
                            if file_rows is not None:
                                pct = (file_rows_done / file_rows * 100.0) if file_rows else 0.0
                                print(
                                    f"[hb] {os.path.basename(fp)} {file_rows_done:,}/{file_rows:,} ({pct:.1f}%) "
                                    f"rows rps={rps:.2f} tok/s={tps:,.0f}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[hb] {os.path.basename(fp)} rows={file_rows_done:,} rps={rps:.2f} tok/s={tps:,.0f}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            next_hb = time.time() + int(args.heartbeat_secs)

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
                                skipped_overflow_local += 1
                                continue

                        w = writers.get(bucket_name)
                        if w is None:
                            tmp_path = os.path.join(file_tmp_root, f"{bucket_name}.parquet.tmp")
                            w = BucketFileWriter(
                                tmp_path=tmp_path,
                                schema=schema,
                                batch_size=int(args.write_batch),
                                compression=args.compression,
                            )
                            writers[bucket_name] = w
                        w.add(row)

                        st_global = stats[bucket_name]
                        st_global["rows"] += 1
                        st_global["tokens_sum"] += tok
                        st_global["tokens_min"] = tok if st_global["tokens_min"] is None else min(st_global["tokens_min"], tok)
                        st_global["tokens_max"] = tok if st_global["tokens_max"] is None else max(st_global["tokens_max"], tok)

                        st_local = bucket_stats[bucket_name]
                        st_local["rows"] += 1
                        st_local["tokens_sum"] += tok
                        st_local["tokens_min"] = tok if st_local["tokens_min"] is None else min(st_local["tokens_min"], tok)
                        st_local["tokens_max"] = tok if st_local["tokens_max"] is None else max(st_local["tokens_max"], tok)
            except Exception as e:
                stats["errors"] += 1
                errors_local += 1
                print(f"\nERROR reading {fp}: {e!r}", file=sys.stderr, flush=True)
            finally:
                for w in writers.values():
                    w.close()

                # Commit per-bucket tmp outputs atomically.
                committed = 0
                for b in buckets_all:
                    tmp_path = os.path.join(file_tmp_root, f"{b.name}.parquet.tmp")
                    if not os.path.exists(tmp_path):
                        continue
                    final_name = f"{safe_base}.{fid}.parquet"
                    final_path = os.path.join(args.output_dir, b.name, final_name)
                    try:
                        os.replace(tmp_path, final_path)
                        committed += 1
                    except OSError as e:
                        stats["errors"] += 1
                        errors_local += 1
                        print(f"\nERROR committing {tmp_path} -> {final_path}: {e!r}", file=sys.stderr, flush=True)

                # Write done marker only if we successfully committed whatever we produced.
                if args.resume:
                    marker = {
                        "input": fp,
                        "file_id": fid,
                        "resume_signature": run_sig,
                        "input_size": st.st_size if st is not None else None,
                        "input_mtime_ns": st.st_mtime_ns if st is not None else None,
                        "bucket_stats": bucket_stats,
                        "skipped_overflow": skipped_overflow_local,
                        "errors": errors_local,
                        "committed_bucket_files": committed,
                        "seconds": round(time.time() - file_t_start, 3),
                    }
                    tmp_marker = os.path.join(file_tmp_root, "done.json.tmp")
                    try:
                        with open(tmp_marker, "w", encoding="utf-8") as f:
                            json.dump(marker, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_marker, done_path)
                    except OSError as e:
                        stats["errors"] += 1
                        print(f"\nERROR writing done marker {done_path}: {e!r}", file=sys.stderr, flush=True)

                elapsed = time.time() - file_t_start
                if elapsed > 0:
                    print(
                        f"[file_done] {os.path.basename(fp)} rows={file_rows_done:,} seconds={elapsed:.1f} "
                        f"rps={(file_rows_done/elapsed):.2f}",
                        file=sys.stderr,
                        flush=True,
                    )

            pbar.update(1)
            pbar.set_postfix(errors=stats["errors"], skipped=stats["skipped_done"])
    finally:
        pbar.close()

    meta = {
        "input_glob": args.input_glob,
        "files": len(files),
        "output_dir": args.output_dir,
        "resume": bool(args.resume),
        "resume_signature": run_sig,
        "buckets": [{"name": b.name, "lo": b.lo, "hi": b.hi} for b in buckets_all],
        "skipped_overflow": stats["skipped_overflow"],
        "skipped_done": stats["skipped_done"],
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
