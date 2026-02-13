import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_BASE = "/mnt/hdfs/user_wl/xingtianshun/deepwiki_data"
DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 8
DEFAULT_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Task:
    folder: str
    filepath: str


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

        # Only counting tokens; avoid max_length warnings if possible.
        try:
            tok.model_max_length = 1 << 60
        except Exception:
            pass

        def count_tokens(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        return count_tokens

    raise ValueError(f"unknown tokenizer backend: {tokenizer_backend}")


def iter_narratives_json_files(base_dir: str):
    dirs_scanned = 0
    files_found = 0
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_dir():
                continue
            if ent.name.startswith("."):
                continue
            folder = ent.name
            dirs_scanned += 1
            if dirs_scanned % 5000 == 0:
                print(
                    f"  [scan] {dirs_scanned:,} dirs scanned, "
                    f"{files_found:,} narrative files found ...",
                    file=sys.stderr,
                    flush=True,
                )
            with os.scandir(ent.path) as it2:
                for f in it2:
                    if not f.is_file():
                        continue
                    if f.name.endswith("_narratives.json"):
                        files_found += 1
                        yield Task(folder=folder, filepath=f.path)
    print(
        f"  [scan] done: {dirs_scanned:,} dirs, {files_found:,} narrative files.",
        file=sys.stderr,
        flush=True,
    )


def parse_one_file(filepath: str, concat_sep: str):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"expected list, got {type(obj)}")

    repo = ""
    narratives: list[str] = []
    missing = 0
    for row in obj:
        if not isinstance(row, dict):
            missing += 1
            continue
        if not repo:
            r = row.get("repo")
            if isinstance(r, str):
                repo = r
        s = row.get("narrative")
        if not s:
            missing += 1
            continue
        if not isinstance(s, str):
            s = str(s)
        narratives.append(s)

    narrative_text = concat_sep.join(narratives)
    return repo, narrative_text, len(obj), missing


class _TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TaskTimeout("task timed out")


def worker(
    task_q,
    out_q,
    concat_sep: str,
    tokenizer_backend: str,
    encoding: str,
    hf_model: str,
    hf_trust_remote_code: bool,
    hf_local_files_only: bool,
    task_timeout: int = 0,
):
    import signal

    count_tokens = make_token_counter(
        tokenizer_backend=tokenizer_backend,
        tiktoken_encoding=encoding,
        hf_model=hf_model,
        hf_trust_remote_code=hf_trust_remote_code,
        hf_local_files_only=hf_local_files_only,
    )

    use_alarm = task_timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)

    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        try:
            if use_alarm:
                signal.alarm(task_timeout)

            repo, narrative_text, total, missing = parse_one_file(task.filepath, concat_sep=concat_sep)
            narrative_tokens = count_tokens(narrative_text)

            if use_alarm:
                signal.alarm(0)  # cancel alarm

            out_q.put(
                (
                    "ok",
                    {
                        "folder": task.folder,
                        "filepath": task.filepath,
                        "repo": repo,
                        "narrative": narrative_text,
                        "narrative_tokens": int(narrative_tokens),
                        "rows": total,
                        "missing_rows": missing,
                    },
                )
            )
        except _TaskTimeout:
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": f"timeout ({task_timeout}s)"}))
        except Exception as e:
            if use_alarm:
                signal.alarm(0)
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": repr(e)}))
        finally:
            task_q.task_done()

    out_q.put(("w_done", None))


def flush_rows(writer: pq.ParquetWriter, rows: list[dict]):
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=writer.schema)
    writer.write_table(table)
    rows.clear()


# ---------------------------------------------------------------------------
# Shard writer: produces sequentially numbered, *complete* Parquet files.
# Each shard is a self-contained Parquet file with a proper footer.
# If the process is killed mid-run, only the last (unfinished) shard is lost.
# ---------------------------------------------------------------------------

class ShardWriter:
    """Write rows into sequentially numbered Parquet shards.

    Each shard is closed (footer written) when it reaches *rows_per_shard*
    rows, so even if the process is killed mid-run, all completed shards
    are valid Parquet files.

    When *rows_per_shard* is 0, behaves like a single-file writer (legacy
    mode) but still supports graceful close.
    """

    def __init__(
        self,
        output_path: str,
        schema: pa.Schema,
        compression: str = "zstd",
        rows_per_shard: int = 0,
        batch_size: int = 256,
        start_shard_idx: int = 0,
    ):
        self.output_path = output_path
        self.schema = schema
        self.compression = compression
        self.rows_per_shard = rows_per_shard
        self.batch_size = batch_size

        self._shard_idx = start_shard_idx
        self._rows_in_shard = 0
        self._total_rows = 0
        self._buffer: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

        self._output_dir = os.path.dirname(output_path) or "."
        self._output_stem = os.path.splitext(os.path.basename(output_path))[0]
        self._output_ext = os.path.splitext(output_path)[1] or ".parquet"

        os.makedirs(self._output_dir, exist_ok=True)

        if rows_per_shard <= 0:
            # Single-file mode
            self._writer = pq.ParquetWriter(output_path, schema=schema, compression=compression)

    def _shard_path(self) -> str:
        return os.path.join(
            self._output_dir,
            f"{self._output_stem}.part{self._shard_idx:04d}{self._output_ext}",
        )

    def _ensure_writer(self):
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._shard_path(), schema=self.schema, compression=self.compression,
            )

    def _write_rows(self, rows: list[dict]):
        """Write a list of rows, auto-splitting on ArrowCapacityError (>2GB chunk)."""
        if not rows:
            return
        try:
            table = pa.Table.from_pylist(rows, schema=self.schema)
            self._writer.write_table(table)
        except (pa.lib.ArrowCapacityError, pa.lib.ArrowInvalid):
            if len(rows) <= 1:
                # Single row > 2GB — skip it and warn
                print(
                    f"\nWARNING: skipping 1 row that exceeds 2GB Arrow limit "
                    f"(folder={rows[0].get('folder', '?')})",
                    file=sys.stderr,
                    flush=True,
                )
                return
            # Split in half and retry
            mid = len(rows) // 2
            self._write_rows(rows[:mid])
            self._write_rows(rows[mid:])
            return
        self._rows_in_shard += len(rows)
        self._total_rows += len(rows)

    def _flush_buffer(self):
        if not self._buffer:
            return
        self._ensure_writer()
        self._write_rows(self._buffer)
        self._buffer.clear()

    def _rotate_if_needed(self):
        if self.rows_per_shard <= 0:
            return
        if self._rows_in_shard >= self.rows_per_shard:
            self._close_current_shard()
            self._shard_idx += 1
            self._rows_in_shard = 0

    def _close_current_shard(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def add(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self._flush_buffer()
            self._rotate_if_needed()

    def close(self):
        self._flush_buffer()
        self._close_current_shard()

    @property
    def total_rows(self) -> int:
        return self._total_rows + len(self._buffer)

    @property
    def shard_count(self) -> int:
        if self.rows_per_shard <= 0:
            return 1
        return self._shard_idx + (1 if self._rows_in_shard > 0 or self._buffer else 0)

    def output_files(self) -> list[str]:
        if self.rows_per_shard <= 0:
            return [self.output_path]
        return [
            os.path.join(
                self._output_dir,
                f"{self._output_stem}.part{i:04d}{self._output_ext}",
            )
            for i in range(self.shard_count)
        ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract each *_narratives.json under base/*/, concatenate each item's 'narrative' field, "
            "and write one row per file into a single Parquet (or multiple shards)."
        )
    )
    ap.add_argument("--base", default=DEFAULT_BASE, help="Base directory (default: %(default)s)")
    ap.add_argument("--output", required=True, help="Output parquet file path")
    ap.add_argument("--concat-sep", default="\n\n", help="Separator used to concatenate narratives within a file")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows per parquet write batch")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Worker processes (default: %(default)s)")
    ap.add_argument("--mp-start", default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--max-files", type=int, default=0, help="Only process first N files (debug). 0 = all.")
    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"])
    ap.add_argument("--encoding", default=DEFAULT_ENCODING, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--hf-model", default="", help="HF model name/path (required when --tokenizer-backend=hf)")
    ap.add_argument("--hf-trust-remote-code", action="store_true")
    ap.add_argument("--hf-local-files-only", action="store_true", default=True)
    ap.add_argument("--hf-allow-download", action="store_false", dest="hf_local_files_only")
    ap.add_argument(
        "--task-timeout",
        type=int,
        default=120,
        help="Per-task timeout in seconds (0 = disabled). If a single file takes longer, it is skipped. (default: %(default)s)",
    )
    ap.add_argument(
        "--rows-per-shard",
        type=int,
        default=10000,
        help=(
            "Max rows per output Parquet shard. Each shard is a complete Parquet file, "
            "so killing the process only loses the last incomplete shard. "
            "0 = single file (legacy). (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from previous run. Scans existing shard files, collects already-processed "
            "folders, and skips them. New shards are appended with incrementing part numbers. "
            "Only works with shard mode (--rows-per-shard > 0)."
        ),
    )
    args = ap.parse_args()

    if args.tokenizer_backend == "hf" and not args.hf_model:
        raise SystemExit("--hf-model is required when --tokenizer-backend=hf")

    schema = pa.schema(
        [
            ("folder", pa.string()),
            ("filepath", pa.string()),
            ("repo", pa.string()),
            ("narrative", pa.large_string()),
            ("narrative_tokens", pa.int64()),
            ("rows", pa.int32()),
            ("missing_rows", pa.int32()),
        ]
    )

    # ------------------------------------------------------------------
    # Resume: scan existing shards for already-processed folders
    # ------------------------------------------------------------------
    done_folders: set[str] = set()
    resume_shard_idx = 0

    if args.resume and args.rows_per_shard > 0:
        output_dir = os.path.dirname(args.output) or "."
        output_stem = os.path.splitext(os.path.basename(args.output))[0]
        output_ext = os.path.splitext(args.output)[1] or ".parquet"

        import glob as _glob

        existing = sorted(_glob.glob(
            os.path.join(output_dir, f"{output_stem}.part*{output_ext}")
        ))
        if existing:
            print(f"[resume] Found {len(existing)} existing shard(s), scanning ...", file=sys.stderr, flush=True)
            for shard_path in existing:
                try:
                    t = pq.read_table(shard_path, columns=["folder"])
                    folders = t.column("folder").to_pylist()
                    done_folders.update(folders)
                except Exception as e:
                    # Incomplete shard (no footer) — delete it
                    print(
                        f"[resume] Removing incomplete shard: {shard_path} ({e!r})",
                        file=sys.stderr,
                        flush=True,
                    )
                    os.remove(shard_path)
                    continue
                # Extract shard index from filename
                fname = os.path.basename(shard_path)
                try:
                    idx = int(fname.replace(output_stem + ".part", "").replace(output_ext, ""))
                    resume_shard_idx = max(resume_shard_idx, idx + 1)
                except ValueError:
                    pass
            print(
                f"[resume] {len(done_folders):,} folders already done, "
                f"new shards start at part{resume_shard_idx:04d}",
                file=sys.stderr,
                flush=True,
            )

    shard_writer = ShardWriter(
        output_path=args.output,
        schema=schema,
        compression="zstd",
        rows_per_shard=args.rows_per_shard,
        batch_size=args.batch_size,
        start_shard_idx=resume_shard_idx,
    )

    # ------------------------------------------------------------------
    # Graceful shutdown: Ctrl+C or kill → close writer before exit
    # ------------------------------------------------------------------
    _shutdown_requested = False

    def _graceful_shutdown(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Second signal → force exit
            print("\nForced exit.", file=sys.stderr, flush=True)
            sys.exit(1)
        _shutdown_requested = True
        sig_name = "SIGINT" if signum == 2 else f"signal {signum}"
        print(
            f"\n[{sig_name}] Graceful shutdown: flushing buffer and closing shards ...",
            file=sys.stderr,
            flush=True,
        )

    import signal as _signal
    _signal.signal(_signal.SIGINT, _graceful_shutdown)
    _signal.signal(_signal.SIGTERM, _graceful_shutdown)

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    task_q = ctx.JoinableQueue()
    out_q = ctx.Queue(maxsize=2000)

    workers = max(1, int(args.workers))

    print(f"Scanning {args.base} for *_narratives.json ...", file=sys.stderr, flush=True)
    all_tasks = []
    skipped = 0
    for t in iter_narratives_json_files(args.base):
        if done_folders and t.folder in done_folders:
            skipped += 1
            continue
        all_tasks.append(t)
        if args.max_files and len(all_tasks) >= args.max_files:
            break
    total_files_enqueued = len(all_tasks)
    if skipped:
        print(
            f"[resume] Skipped {skipped:,} already-done files, {total_files_enqueued:,} remaining.",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"Spawning {workers} workers for {total_files_enqueued:,} files ...",
        file=sys.stderr,
        flush=True,
    )
    if args.rows_per_shard > 0:
        print(
            f"Shard mode: {args.rows_per_shard:,} rows per shard",
            file=sys.stderr,
            flush=True,
        )

    procs = []
    for _ in range(workers):
        p = ctx.Process(
            target=worker,
            args=(
                task_q,
                out_q,
                args.concat_sep,
                args.tokenizer_backend,
                args.encoding,
                args.hf_model,
                args.hf_trust_remote_code,
                args.hf_local_files_only,
                args.task_timeout,
            ),
        )
        p.start()
        procs.append(p)

    for t in all_tasks:
        task_q.put(t)
    del all_tasks
    for _ in range(workers):
        task_q.put(None)

    ok = 0
    err = 0
    workers_done = 0

    pbar = tqdm(
        total=total_files_enqueued,
        desc="Exporting narratives",
        unit="file",
        dynamic_ncols=True,
        disable=False,
    )
    try:
        while workers_done < workers and not _shutdown_requested:
            try:
                kind, payload = out_q.get(timeout=5)
            except Exception:
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Workers exited non-zero: exitcodes={[p.exitcode for p in dead]}")
                if all(p.exitcode is not None for p in procs):
                    break
                continue

            if kind == "w_done":
                workers_done += 1
                continue
            if kind == "ok":
                ok += 1
                shard_writer.add(payload)
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err, shards=shard_writer.shard_count)
            elif kind == "err":
                err += 1
                print(f"\nERROR {payload.get('filepath')}: {payload.get('error')}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err, shards=shard_writer.shard_count)
    finally:
        pbar.close()
        # Always close the shard writer so completed data is not lost
        shard_writer.close()
        # Kill remaining workers quickly if shutting down early
        if _shutdown_requested:
            for p in procs:
                if p.is_alive():
                    p.terminate()
        for p in procs:
            p.join(timeout=10)

    interrupted = " (interrupted)" if _shutdown_requested else ""
    print(f"\nDONE{interrupted}")
    print("base:", args.base)
    if args.rows_per_shard > 0:
        print("output_shards:", shard_writer.shard_count)
        for fp in shard_writer.output_files():
            print(f"  {fp}")
    else:
        print("output:", args.output)
    print("files_total:", total_files_enqueued)
    print("files_ok:", ok)
    print("files_err:", err)
    print("rows_written:", shard_writer.total_rows)


if __name__ == "__main__":
    main()
