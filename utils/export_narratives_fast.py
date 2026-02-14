import argparse
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
DEFAULT_WORKERS = 8
# For huge narratives, keep Arrow batches small to avoid GB-sized tables.
DEFAULT_BATCH_SIZE = 8
DEFAULT_ROWS_PER_SHARD = 5000


@dataclass(frozen=True)
class Task:
    folder: str
    filepath: str


def make_schema() -> pa.Schema:
    # Keep the same columns as the original exporter, but do not compute tokens.
    return pa.schema(
        [
            ("folder", pa.string()),
            ("filepath", pa.string()),
            ("repo", pa.string()),
            ("narrative", pa.large_string()),
            ("narrative_tokens", pa.int64()),  # always -1 in this script
            ("rows", pa.int32()),
            ("missing_rows", pa.int32()),
        ]
    )


def _json_load(path: str):
    try:
        import orjson  # type: ignore

        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        import json

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)


def iter_narratives_json_files(base_dir: str, stop_event: threading.Event | None = None):
    dirs_scanned = 0
    files_found = 0
    with os.scandir(base_dir) as it:
        for ent in it:
            if stop_event is not None and stop_event.is_set():
                break
            if not ent.is_dir():
                continue
            if ent.name.startswith("."):
                continue
            folder = ent.name
            dirs_scanned += 1
            if dirs_scanned % 5000 == 0:
                print(
                    f"  [scan] {dirs_scanned:,} dirs scanned, {files_found:,} narrative files found ...",
                    file=sys.stderr,
                    flush=True,
                )
            with os.scandir(ent.path) as it2:
                for f in it2:
                    if stop_event is not None and stop_event.is_set():
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
    obj = _json_load(filepath)
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
    enqueued = 0
    skipped = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for t in iter_narratives_json_files(base_dir, stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                break
            if done_filepaths and t.filepath in done_filepaths:
                skipped += 1
                continue
            f.write(f"{t.folder}\t{t.filepath}\n")
            enqueued += 1
            if max_files and enqueued >= max_files:
                break
    return enqueued, skipped


def flush_rows(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema):
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)
    rows.clear()


class ShardWriter:
    def __init__(
        self,
        output_path: str,
        schema: pa.Schema,
        compression: str = "zstd",
        rows_per_shard: int = 0,
        batch_size: int = 8,
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

    def _shard_path(self) -> str:
        return os.path.join(
            self._output_dir,
            f"{self._output_stem}.part{self._shard_idx:04d}{self._output_ext}",
        )

    def _ensure_writer(self):
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._shard_path(),
                schema=self.schema,
                compression=self.compression,
            )

    def _flush_buffer(self):
        if not self._buffer:
            return
        self._ensure_writer()
        flush_rows(self._writer, self._buffer, self.schema)
        self._rows_in_shard += len(self._buffer)
        self._total_rows += len(self._buffer)
        self._buffer.clear()

    def _rotate_if_needed(self):
        if self.rows_per_shard <= 0:
            return
        if self._rows_in_shard >= self.rows_per_shard:
            self.close()
            self._shard_idx += 1
            self._rows_in_shard = 0

    def add(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self._flush_buffer()
            self._rotate_if_needed()

    def close(self):
        if self._buffer:
            self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def shard_count(self) -> int:
        return self._shard_idx + 1


def _worker_output_path(base_output: str, worker_id: int) -> str:
    output_dir = os.path.dirname(base_output) or "."
    output_stem = os.path.splitext(os.path.basename(base_output))[0]
    output_ext = os.path.splitext(base_output)[1] or ".parquet"
    return os.path.join(output_dir, f"{output_stem}.worker{worker_id:03d}{output_ext}")


def worker(
    task_q,
    out_q,
    concat_sep: str,
    output_path: str,
    rows_per_shard: int,
    batch_size: int,
    task_timeout: int,
    worker_id: int,
    start_shard_idx: int,
):
    import signal

    # Ignore SIGINT in workers so only the parent coordinates shutdown.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    class _TaskTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _TaskTimeout("task timed out")

    use_alarm = task_timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)

    schema = make_schema()
    writer = ShardWriter(
        output_path=output_path,
        schema=schema,
        compression="zstd",
        rows_per_shard=rows_per_shard,
        batch_size=batch_size,
        start_shard_idx=start_shard_idx,
    )

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            if use_alarm:
                signal.alarm(task_timeout)
            repo, narrative_text, total, missing = parse_one_file(task.filepath, concat_sep=concat_sep)
            if use_alarm:
                signal.alarm(0)
            writer.add(
                {
                    "folder": task.folder,
                    "filepath": task.filepath,
                    "repo": repo,
                    "narrative": narrative_text,
                    "narrative_tokens": -1,
                    "rows": total,
                    "missing_rows": missing,
                }
            )
            out_q.put(("ok", {"filepath": task.filepath}))
        except _TaskTimeout:
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": f"timeout ({task_timeout}s)"}))
        except KeyboardInterrupt:
            break
        except Exception as e:
            out_q.put(("err", {"folder": task.folder, "filepath": task.filepath, "error": repr(e)}))
        finally:
            if use_alarm:
                signal.alarm(0)

    writer.close()
    out_q.put(("w_done", {"worker_id": worker_id, "rows_written": writer.total_rows}))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast exporter: extract narratives to Parquet, without token counting."
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--output", required=True, help="Output base parquet path (worker shards will be created).")
    ap.add_argument("--concat-sep", default="\n\n")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--mp-start", default="forkserver", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--task-timeout", type=int, default=0, help="Per-file timeout seconds (0 disables).")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=False)
    args = ap.parse_args()

    # Resume: scan existing worker shards for done filepaths
    done_filepaths: set[str] = set()
    resume_worker_shard_idx: dict[int, int] = {}

    if args.resume and args.rows_per_shard > 0:
        output_dir = os.path.dirname(args.output) or "."
        output_stem = os.path.splitext(os.path.basename(args.output))[0]
        output_ext = os.path.splitext(args.output)[1] or ".parquet"
        import glob as _glob

        existing = sorted(_glob.glob(os.path.join(output_dir, f"{output_stem}.worker*.part*{output_ext}")))
        if existing:
            print(f"[resume] Found {len(existing)} shard(s), scanning ...", file=sys.stderr, flush=True)
            for shard_path in existing:
                try:
                    t = pq.read_table(shard_path, columns=["filepath"])
                    done_filepaths.update(p for p in t.column("filepath").to_pylist() if isinstance(p, str))
                except Exception:
                    # If a shard is corrupted/incomplete, remove it.
                    try:
                        os.remove(shard_path)
                    except OSError:
                        pass
                    continue
                fname = os.path.basename(shard_path)
                try:
                    mid = fname.split(".worker", 1)[1]
                    wid_s, rest = mid.split(".part", 1)
                    idx_s = rest.rsplit(output_ext, 1)[0]
                    wid = int(wid_s)
                    idx = int(idx_s)
                    resume_worker_shard_idx[wid] = max(resume_worker_shard_idx.get(wid, 0), idx + 1)
                except Exception:
                    pass
            print(f"[resume] {len(done_filepaths):,} files already done.", file=sys.stderr, flush=True)

    stop_event = threading.Event()
    shutdown_requested = False

    def _on_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        stop_event.set()
        print(f"\n[signal {signum}] graceful shutdown (pid={os.getpid()}) ...", file=sys.stderr, flush=True)

    import signal as _signal

    _signal.signal(_signal.SIGINT, _on_signal)
    _signal.signal(_signal.SIGTERM, _on_signal)

    # Prescan into a local task list to avoid scan/read IO contention on remote filesystems.
    fd, task_list_path = tempfile.mkstemp(prefix="deepwiki_tasks_", suffix=".tsv")
    os.close(fd)
    print(f"Scanning {args.base} for *_narratives.json ...", file=sys.stderr, flush=True)
    total, skipped = prescan_tasks_to_jsonl(
        base_dir=args.base,
        done_filepaths=done_filepaths,
        max_files=int(args.max_files),
        out_jsonl_path=task_list_path,
        stop_event=stop_event,
    )
    if skipped:
        print(f"[resume] Skipped {skipped:,} already-done file(s).", file=sys.stderr, flush=True)
    print(f"Scan complete: {total:,} file(s) queued.", file=sys.stderr, flush=True)
    if shutdown_requested:
        try:
            os.remove(task_list_path)
        except OSError:
            pass
        return

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    workers = max(1, int(args.workers))
    task_q = ctx.Queue(maxsize=max(256, workers * 8))
    out_q = ctx.Queue(maxsize=2000)

    print(f"Spawning {workers} workers ... (mp-start={args.mp_start})", file=sys.stderr, flush=True)
    print(f"Shard mode: {args.rows_per_shard:,} rows per shard", file=sys.stderr, flush=True)

    procs = []
    for wid in range(workers):
        p = ctx.Process(
            target=worker,
            args=(
                task_q,
                out_q,
                args.concat_sep,
                _worker_output_path(args.output, wid),
                int(args.rows_per_shard),
                int(args.batch_size),
                int(args.task_timeout),
                wid,
                int(resume_worker_shard_idx.get(wid, 0)),
            ),
        )
        p.start()
        procs.append(p)

    # Enqueue tasks
    with open(task_list_path, "r", encoding="utf-8") as f:
        for line in f:
            if shutdown_requested:
                break
            folder, filepath = line.rstrip("\n").split("\t", 1)
            task_q.put(Task(folder=folder, filepath=filepath))
    for _ in range(workers):
        task_q.put(None)

    ok = 0
    err = 0
    workers_done = 0
    rows_written = 0
    pbar = tqdm(total=total, desc="Exporting narratives", unit="file", dynamic_ncols=True, disable=False)
    try:
        while workers_done < workers and not shutdown_requested:
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
                if isinstance(payload, dict):
                    rows_written += int(payload.get("rows_written", 0) or 0)
                continue
            if kind == "ok":
                ok += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err)
            elif kind == "err":
                err += 1
                print(f"\nERROR {payload.get('filepath')}: {payload.get('error')}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err)
    finally:
        pbar.close()
        if shutdown_requested:
            for p in procs:
                if p.is_alive():
                    p.terminate()
        for p in procs:
            p.join(timeout=10)
        try:
            os.remove(task_list_path)
        except OSError:
            pass

    interrupted = " (interrupted)" if shutdown_requested else ""
    print(f"\nDONE{interrupted}")
    print("base:", args.base)
    print("output_pattern:", os.path.join(os.path.dirname(args.output) or ".", f"{os.path.splitext(os.path.basename(args.output))[0]}.worker*.part*{os.path.splitext(args.output)[1] or '.parquet'}"))
    print("files_total:", total)
    print("files_ok:", ok)
    print("files_err:", err)
    print("rows_written:", rows_written if rows_written else ok)


if __name__ == "__main__":
    main()

