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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract each *_narratives.json under base/*/, concatenate each item's 'narrative' field, "
            "and write one row per file into a single Parquet."
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = pq.ParquetWriter(args.output, schema=schema, compression="zstd")

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    task_q = ctx.JoinableQueue()
    out_q = ctx.Queue(maxsize=2000)

    workers = max(1, int(args.workers))

    print(f"Scanning {args.base} for *_narratives.json ...", file=sys.stderr, flush=True)
    # Collect tasks first so we know the total before spawning workers
    all_tasks = []
    for t in iter_narratives_json_files(args.base):
        all_tasks.append(t)
        if args.max_files and len(all_tasks) >= args.max_files:
            break
    total_files_enqueued = len(all_tasks)

    print(
        f"Spawning {workers} workers for {total_files_enqueued:,} files ...",
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
    del all_tasks  # free memory
    for _ in range(workers):
        task_q.put(None)

    ok = 0
    err = 0
    workers_done = 0
    buffer: list[dict] = []

    pbar = tqdm(
        total=total_files_enqueued,
        desc="Exporting narratives",
        unit="file",
        dynamic_ncols=True,
        disable=False,
    )
    try:
        while workers_done < workers:
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
                buffer.append(payload)
                if len(buffer) >= args.batch_size:
                    flush_rows(writer, buffer)
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err)
            elif kind == "err":
                err += 1
                # still write error row? keep errors separate; user can inspect stderr
                print(f"\nERROR {payload.get('filepath')}: {payload.get('error')}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err)
    finally:
        pbar.close()
        task_q.join()
        flush_rows(writer, buffer)
        writer.close()
        for p in procs:
            p.join(timeout=10)

    print("\nDONE")
    print("base:", args.base)
    print("output:", args.output)
    print("files_total:", total_files_enqueued)
    print("files_ok:", ok)
    print("files_err:", err)


if __name__ == "__main__":
    main()
