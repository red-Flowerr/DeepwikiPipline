import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
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


def make_schema() -> pa.Schema:
    return pa.schema(
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


def iter_narratives_json_files(base_dir: str, _stop_event: threading.Event | None = None):
    dirs_scanned = 0
    files_found = 0
    with os.scandir(base_dir) as it:
        for ent in it:
            if _stop_event is not None and _stop_event.is_set():
                break
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
                    if _stop_event is not None and _stop_event.is_set():
                        break
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


def prescan_tasks_to_jsonl(
    *,
    base_dir: str,
    done_filepaths: set[str],
    max_files: int,
    out_jsonl_path: str,
    stop_event: threading.Event | None = None,
) -> tuple[int, int]:
    """Scan base_dir and write tasks as JSONL. Returns (enqueued, skipped)."""
    enqueued = 0
    skipped = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for t in iter_narratives_json_files(base_dir, _stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                break
            if done_filepaths and t.filepath in done_filepaths:
                skipped += 1
                continue
            f.write(json.dumps({"folder": t.folder, "filepath": t.filepath}, ensure_ascii=False) + "\n")
            enqueued += 1
            if max_files and enqueued >= max_files:
                break
    return enqueued, skipped


class _TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TaskTimeout("task timed out")


def _worker_output_path(base_output: str, worker_id: int) -> str:
    output_dir = os.path.dirname(base_output) or "."
    output_stem = os.path.splitext(os.path.basename(base_output))[0]
    output_ext = os.path.splitext(base_output)[1] or ".parquet"
    return os.path.join(output_dir, f"{output_stem}.worker{worker_id:03d}{output_ext}")


def _worker_output_glob_pattern(base_output: str) -> str:
    output_dir = os.path.dirname(base_output) or "."
    output_stem = os.path.splitext(os.path.basename(base_output))[0]
    output_ext = os.path.splitext(base_output)[1] or ".parquet"
    # ShardWriter turns "{stem}.worker000.parquet" into "{stem}.worker000.part0000.parquet"
    return os.path.join(output_dir, f"{output_stem}.worker*.part*{output_ext}")


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
    no_token_count: bool = False,
    write_mode: str = "worker",
    output_path: str = "",
    rows_per_shard: int = 0,
    batch_size: int = 256,
    worker_id: int = 0,
    start_shard_idx: int = 0,
):
    import signal

    # With --mp-start=fork the parent signal handlers are inherited.
    # Ignore SIGINT in workers so only the parent coordinates shutdown.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    count_tokens = None
    if not no_token_count:
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

    shard_writer: ShardWriter | None = None
    if write_mode == "worker":
        if not output_path:
            raise ValueError("worker mode requires output_path")
        shard_writer = ShardWriter(
            output_path=output_path,
            schema=make_schema(),
            compression="zstd",
            rows_per_shard=rows_per_shard,
            batch_size=batch_size,
            start_shard_idx=start_shard_idx,
        )

    while True:
        task = task_q.get()
        if task is None:
            break

        # Defensive: ensure no previous alarm leaks into this iteration.
        if use_alarm:
            try:
                signal.alarm(0)
            except Exception:
                pass

        try:
            if use_alarm:
                signal.alarm(task_timeout)

            repo, narrative_text, total, missing = parse_one_file(task.filepath, concat_sep=concat_sep)
            narrative_tokens = -1 if no_token_count else int(count_tokens(narrative_text))  # type: ignore[misc]

            if use_alarm:
                signal.alarm(0)  # cancel alarm

            row = {
                "folder": task.folder,
                "filepath": task.filepath,
                "repo": repo,
                "narrative": narrative_text,
                "narrative_tokens": narrative_tokens,
                "rows": total,
                "missing_rows": missing,
            }
            if write_mode == "worker":
                assert shard_writer is not None
                shard_writer.add(row)
                # Only send small payloads to the parent to avoid IPC bottlenecks.
                out_q.put(("ok", {"filepath": task.filepath}))
            else:
                out_q.put(("ok", row))
        except _TaskTimeout:
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": f"timeout ({task_timeout}s)"}))
        except KeyboardInterrupt:
            # Can happen if the environment delivers SIGINT directly to workers (spawn) or before handlers install.
            # Exit cleanly without spewing tracebacks.
            if use_alarm:
                try:
                    signal.alarm(0)
                except Exception:
                    pass
            break
        except Exception as e:
            if use_alarm:
                signal.alarm(0)
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": repr(e)}))
        except BaseException as e:
            # Last-resort: keep the parent informed, then stop this worker.
            if use_alarm:
                signal.alarm(0)
            out_q.put(
                (
                    "err",
                    {
                        "folder": task.folder,
                        "filepath": task.filepath,
                        "error": f"fatal: {type(e).__name__}: {e}",
                    },
                )
            )
            break
        finally:
            if use_alarm:
                # Ensure a timeout doesn't leak into subsequent tasks.
                signal.alarm(0)

    if shard_writer is not None:
        shard_writer.close()
        out_q.put(
            (
                "w_done",
                {
                    "worker_id": worker_id,
                    "rows_written": shard_writer.total_rows,
                    "shards": shard_writer.shard_count,
                    "output_files": shard_writer.output_files(),
                },
            )
        )
    else:
        out_q.put(("w_done", {"worker_id": worker_id}))


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
    ap.add_argument(
        "--scan-mode",
        default="prescan",
        choices=["prescan", "overlap"],
        help=(
            "How to scan for input files. "
            "'prescan' scans first and then processes (less IO contention, stable progress total). "
            "'overlap' scans and processes concurrently (lower memory, can be faster on local disks). "
            "(default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--write-mode",
        default="worker",
        choices=["worker", "main"],
        help=(
            "How to write Parquet. "
            "'worker' lets each worker write its own Parquet shards (fastest; avoids huge IPC for long narratives). "
            "'main' sends rows to the parent process to write (legacy; can be slow for large narratives). "
            "(default: %(default)s)"
        ),
    )
    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"])
    ap.add_argument(
        "--no-token-count",
        action="store_true",
        default=False,
        help="Skip token counting (writes narrative_tokens=-1). Useful for very large narratives.",
    )
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

    if (not args.no_token_count) and args.tokenizer_backend == "hf" and not args.hf_model:
        raise SystemExit("--hf-model is required when --tokenizer-backend=hf")

    schema = make_schema()

    # ------------------------------------------------------------------
    # Resume: scan existing shards for already-processed folders
    # ------------------------------------------------------------------
    done_filepaths: set[str] = set()
    resume_shard_idx = 0  # main mode
    resume_worker_shard_idx: dict[int, int] = {}  # worker mode: worker_id -> next shard idx

    if args.resume and args.rows_per_shard > 0:
        output_dir = os.path.dirname(args.output) or "."
        output_stem = os.path.splitext(os.path.basename(args.output))[0]
        output_ext = os.path.splitext(args.output)[1] or ".parquet"

        import glob as _glob

        if args.write_mode == "worker":
            pattern = os.path.join(output_dir, f"{output_stem}.worker*.part*{output_ext}")
        else:
            pattern = os.path.join(output_dir, f"{output_stem}.part*{output_ext}")
        existing = sorted(_glob.glob(pattern))
        if existing:
            print(f"[resume] Found {len(existing)} existing shard(s), scanning ...", file=sys.stderr, flush=True)
            for shard_path in existing:
                try:
                    t = pq.read_table(shard_path, columns=["filepath"])
                    paths = t.column("filepath").to_pylist()
                    # Parquet may contain nulls; guard to keep the set string-only.
                    done_filepaths.update(p for p in paths if isinstance(p, str))
                except Exception as e:
                    # Incomplete shard (no footer) — delete it
                    print(
                        f"[resume] Removing incomplete shard: {shard_path} ({e!r})",
                        file=sys.stderr,
                        flush=True,
                    )
                    os.remove(shard_path)
                    continue
                fname = os.path.basename(shard_path)
                if args.write_mode == "worker":
                    # Expected: {stem}.worker{wid}.part{idx}{ext}
                    # Example: out.worker000.part0003.parquet
                    try:
                        mid = fname.split(".worker", 1)[1]
                        wid_s, rest = mid.split(".part", 1)
                        idx_s = rest.rsplit(output_ext, 1)[0]
                        wid = int(wid_s)
                        idx = int(idx_s)
                        resume_worker_shard_idx[wid] = max(resume_worker_shard_idx.get(wid, 0), idx + 1)
                    except Exception:
                        pass
                else:
                    # Expected: {stem}.part{idx}{ext}
                    try:
                        idx = int(fname.replace(output_stem + ".part", "").replace(output_ext, ""))
                        resume_shard_idx = max(resume_shard_idx, idx + 1)
                    except ValueError:
                        pass
            print(
                f"[resume] {len(done_filepaths):,} files already done, "
                + (
                    "new shards continue per-worker"
                    if args.write_mode == "worker"
                    else f"new shards start at part{resume_shard_idx:04d}"
                ),
                file=sys.stderr,
                flush=True,
            )

    shard_writer: ShardWriter | None = None
    if args.write_mode == "main":
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
    _scan_stop = threading.Event()

    def _graceful_shutdown(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Some environments may deliver repeated SIGINT/SIGTERM. Don't hard-exit:
            # keep trying to flush/close so finished shards stay valid.
            print(
                f"\n[signal {signum}] Shutdown already in progress (pid={os.getpid()}).",
                file=sys.stderr,
                flush=True,
            )
            return
        _shutdown_requested = True
        _scan_stop.set()
        sig_name = "SIGINT" if signum == 2 else f"signal {signum}"
        print(
            f"\n[{sig_name}] Graceful shutdown: flushing buffer and closing shards (pid={os.getpid()}) ...",
            file=sys.stderr,
            flush=True,
        )

    import signal as _signal
    _signal.signal(_signal.SIGINT, _graceful_shutdown)
    _signal.signal(_signal.SIGTERM, _graceful_shutdown)

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    workers = max(1, int(args.workers))
    # Bounded queue: avoids unbounded memory growth when scanning huge directories.
    task_q = ctx.Queue(maxsize=max(256, workers * 8))
    out_q = ctx.Queue(maxsize=2000)

    skipped = 0
    total_files_enqueued = 0
    task_list_path: str | None = None

    if args.scan_mode == "prescan":
        fd, task_list_path = tempfile.mkstemp(prefix="deepwiki_tasks_", suffix=".jsonl")
        os.close(fd)
        print(f"Scanning {args.base} for *_narratives.json ...", file=sys.stderr, flush=True)
        total_files_enqueued, skipped = prescan_tasks_to_jsonl(
            base_dir=args.base,
            done_filepaths=done_filepaths,
            max_files=int(args.max_files),
            out_jsonl_path=task_list_path,
            stop_event=_scan_stop,
        )
        if skipped:
            print(f"[resume] Skipped {skipped:,} already-done file(s).", file=sys.stderr, flush=True)
        print(f"Scan complete: {total_files_enqueued:,} file(s) queued.", file=sys.stderr, flush=True)
        if _shutdown_requested:
            # Interrupted during prescan: nothing else to do.
            if shard_writer is not None:
                shard_writer.close()
            if task_list_path:
                try:
                    os.remove(task_list_path)
                except OSError:
                    pass
            print("\nDONE (interrupted)")
            print("base:", args.base)
            print("files_total:", total_files_enqueued)
            print("files_ok:", 0)
            print("files_err:", 0)
            print("rows_written:", 0)
            return

    print(
        f"Spawning {workers} workers ... (scan-mode={args.scan_mode})",
        file=sys.stderr,
        flush=True,
    )
    if args.rows_per_shard > 0:
        print(f"Shard mode: {args.rows_per_shard:,} rows per shard", file=sys.stderr, flush=True)

    procs = []
    for wid in range(workers):
        worker_output = _worker_output_path(args.output, wid) if args.write_mode == "worker" else ""
        worker_start_idx = resume_worker_shard_idx.get(wid, 0) if args.write_mode == "worker" else 0
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
                args.no_token_count,
                args.write_mode,
                worker_output,
                args.rows_per_shard if args.write_mode == "worker" else 0,
                args.batch_size,
                wid,
                worker_start_idx,
            ),
        )
        p.start()
        procs.append(p)

    if args.scan_mode == "prescan":
        if task_list_path:
            with open(task_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    if _shutdown_requested:
                        break
                    item = json.loads(line)
                    task_q.put(Task(folder=item["folder"], filepath=item["filepath"]))
        for _ in range(workers):
            task_q.put(None)
    else:
        # ------------------------------------------------------------------
        # Overlap mode: scan base dir and enqueue tasks while workers run.
        # ------------------------------------------------------------------
        producer_err: list[BaseException] = []
        counts_lock = threading.Lock()
        counts = {"enqueued": 0, "skipped": 0}

        def _producer():
            try:
                print(f"Scanning {args.base} for *_narratives.json ...", file=sys.stderr, flush=True)
                for t in iter_narratives_json_files(args.base, _stop_event=_scan_stop):
                    if done_filepaths and t.filepath in done_filepaths:
                        with counts_lock:
                            counts["skipped"] += 1
                        continue
                    task_q.put(t)  # blocks if queue full (backpressure)
                    with counts_lock:
                        counts["enqueued"] += 1
                        if args.max_files and counts["enqueued"] >= args.max_files:
                            break
            except BaseException as e:
                producer_err.append(e)
            finally:
                for _ in range(workers):
                    task_q.put(None)

        t_prod = threading.Thread(target=_producer, name="task-producer", daemon=True)
        t_prod.start()

    ok = 0
    err = 0
    workers_done = 0
    worker_outputs: list[str] = []
    worker_rows_written = 0
    worker_shards = 0

    pbar = tqdm(
        total=total_files_enqueued if args.scan_mode == "prescan" else 0,
        desc="Exporting narratives",
        unit="file",
        dynamic_ncols=True,
        disable=False,
    )
    last_total_update = 0.0  # overlap mode only
    try:
        while workers_done < workers and not _shutdown_requested:
            try:
                kind, payload = out_q.get(timeout=5)
            except Exception:
                # Surface any producer exceptions promptly (overlap mode).
                if args.scan_mode == "overlap" and producer_err:
                    raise RuntimeError("Producer thread failed") from producer_err[0]
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Workers exited non-zero: exitcodes={[p.exitcode for p in dead]}")
                if all(p.exitcode is not None for p in procs):
                    break
                # Update tqdm total as we discover more files (overlap mode only).
                if args.scan_mode == "overlap":
                    now = time.time()
                    if now - last_total_update >= 1.0:
                        with counts_lock:
                            enq = counts["enqueued"]
                        if pbar.total != enq:
                            pbar.total = enq
                            pbar.refresh()
                        last_total_update = now
                continue

            if kind == "w_done":
                workers_done += 1
                if isinstance(payload, dict):
                    worker_rows_written += int(payload.get("rows_written", 0) or 0)
                    worker_shards += int(payload.get("shards", 0) or 0)
                    files = payload.get("output_files")
                    if isinstance(files, list):
                        worker_outputs.extend(str(x) for x in files)
                continue
            if kind == "ok":
                ok += 1
                if shard_writer is not None:
                    shard_writer.add(payload)
                pbar.update(1)
                if shard_writer is not None:
                    pbar.set_postfix(ok=ok, err=err, shards=shard_writer.shard_count)
                else:
                    pbar.set_postfix(ok=ok, err=err)
            elif kind == "err":
                err += 1
                print(f"\nERROR {payload.get('filepath')}: {payload.get('error')}", file=sys.stderr)
                pbar.update(1)
                if shard_writer is not None:
                    pbar.set_postfix(ok=ok, err=err, shards=shard_writer.shard_count)
                else:
                    pbar.set_postfix(ok=ok, err=err)

            # Keep tqdm total close to enqueued so ETA is meaningful (overlap mode).
            if args.scan_mode == "overlap":
                now = time.time()
                if now - last_total_update >= 1.0:
                    with counts_lock:
                        enq = counts["enqueued"]
                    if pbar.total != enq:
                        pbar.total = enq
                        pbar.refresh()
                    last_total_update = now
    finally:
        pbar.close()
        # Always close the shard writer so completed data is not lost
        if shard_writer is not None:
            shard_writer.close()
        # Kill remaining workers quickly if shutting down early
        if _shutdown_requested:
            for p in procs:
                if p.is_alive():
                    p.terminate()
        for p in procs:
            p.join(timeout=10)

    interrupted = " (interrupted)" if _shutdown_requested else ""
    if args.scan_mode == "overlap":
        with counts_lock:
            total_files_enqueued = counts["enqueued"]
            skipped = counts["skipped"]
        if skipped:
            print(f"[resume] Skipped {skipped:,} already-done file(s).", file=sys.stderr, flush=True)
    if task_list_path:
        try:
            os.remove(task_list_path)
        except OSError:
            pass

    print(f"\nDONE{interrupted}")
    print("base:", args.base)
    if args.write_mode == "main":
        if args.rows_per_shard > 0:
            assert shard_writer is not None
            print("output_shards:", shard_writer.shard_count)
            for fp in shard_writer.output_files():
                print(f"  {fp}")
        else:
            print("output:", args.output)
    else:
        # Worker mode always produces multiple files.
        uniq = sorted(set(worker_outputs))
        if uniq:
            print("output_files:", len(uniq))
            for fp in uniq:
                print(f"  {fp}")
        else:
            # Fallback: show the naming pattern
            print("output_pattern:", _worker_output_glob_pattern(args.output))
    print("files_total:", total_files_enqueued)
    print("files_ok:", ok)
    print("files_err:", err)
    if shard_writer is not None:
        print("rows_written:", shard_writer.total_rows)
    else:
        # Best-effort: when interrupted we may not receive all worker summaries.
        rows_written = worker_rows_written
        if rows_written == 0 and ok > 0 and workers_done < workers:
            rows_written = ok
        print("rows_written:", rows_written)


if __name__ == "__main__":
    main()
