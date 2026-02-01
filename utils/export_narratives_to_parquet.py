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


@dataclass(frozen=True)
class Task:
    folder: str
    filepath: str


def iter_narratives_json_files(base_dir: str):
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_dir():
                continue
            if ent.name.startswith("."):
                continue
            folder = ent.name
            with os.scandir(ent.path) as it2:
                for f in it2:
                    if not f.is_file():
                        continue
                    if f.name.endswith("_narratives.json"):
                        yield Task(folder=folder, filepath=f.path)


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


def worker(task_q, out_q, concat_sep: str):
    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        try:
            repo, narrative_text, total, missing = parse_one_file(task.filepath, concat_sep=concat_sep)
            out_q.put(
                (
                    "ok",
                    {
                        "folder": task.folder,
                        "filepath": task.filepath,
                        "repo": repo,
                        "narrative": narrative_text,
                        "rows": total,
                        "missing_rows": missing,
                    },
                )
            )
        except Exception as e:
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
    args = ap.parse_args()

    schema = pa.schema(
        [
            ("folder", pa.string()),
            ("filepath", pa.string()),
            ("repo", pa.string()),
            ("narrative", pa.large_string()),
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
    procs = []
    for _ in range(workers):
        p = ctx.Process(target=worker, args=(task_q, out_q, args.concat_sep))
        p.start()
        procs.append(p)

    total_files_enqueued = 0
    for t in iter_narratives_json_files(args.base):
        task_q.put(t)
        total_files_enqueued += 1
        if args.max_files and total_files_enqueued >= args.max_files:
            break
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
        disable=not sys.stderr.isatty(),
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

