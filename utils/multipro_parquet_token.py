import argparse
import glob
import multiprocessing as mp
import os
import sys

import pyarrow.parquet as pq
from tqdm import tqdm


ENCODING_DEFAULT = "cl100k_base"  # 可改 "o200k_base"
BATCH_SIZE_DEFAULT = 4096
NUM_WORKERS_DEFAULT = 32
PROGRESS_Q_MAX_DEFAULT = 2000
COLUMN_DEFAULT = "content"


def worker(task_q, progress_q, encoding: str, batch_size: int, column: str):
    import tiktoken
    import pyarrow.parquet as pq

    enc = tiktoken.get_encoding(encoding)

    while True:
        fp = task_q.get()
        if fp is None:
            task_q.task_done()
            break

        try:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(columns=[column], batch_size=batch_size):
                toks = 0
                nulls = 0
                for s in batch.column(0).to_pylist():
                    if s is None:
                        nulls += 1
                    else:
                        toks += len(enc.encode_ordinary(s))
                progress_q.put(("p", batch.num_rows, toks, nulls))
        except Exception as e:
            progress_q.put(("e", fp, repr(e)))
        finally:
            task_q.task_done()

    progress_q.put(("w_done",))


def _list_part_files(base_dir: str, parts_glob: str) -> list[str]:
    if parts_glob:
        files = sorted(glob.glob(os.path.join(base_dir, parts_glob)))
    else:
        files = sorted(glob.glob(os.path.join(base_dir, "part-*")))
    return [f for f in files if os.path.isfile(f)]


def main():
    ap = argparse.ArgumentParser(
        description="Multiprocess token counter for parquet dataset column (default: content) using tiktoken."
    )
    ap.add_argument("--base", required=True, help="Parquet directory containing part-* files")
    ap.add_argument("--parts-glob", default="", help="Override part file glob (default: part-*)")
    ap.add_argument("--column", default=COLUMN_DEFAULT, help="Parquet column to tokenize (default: %(default)s)")
    ap.add_argument("--encoding", default=ENCODING_DEFAULT, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Rows per batch (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT, help="Worker processes (default: %(default)s)")
    ap.add_argument(
        "--mp-start",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: %(default)s)",
    )
    ap.add_argument("--progress-q-max", type=int, default=PROGRESS_Q_MAX_DEFAULT, help="Progress queue maxsize")
    args = ap.parse_args()

    part_files = _list_part_files(args.base, args.parts_glob)
    if not part_files:
        raise SystemExit(f"No parquet part files found under {args.base}")

    # 先读 footer 拿总行数（用于 tqdm total）
    total_rows = 0
    for fp in tqdm(
        part_files, desc="Reading parquet footers", unit="file", dynamic_ncols=True, disable=not sys.stderr.isatty()
    ):
        total_rows += pq.ParquetFile(fp).metadata.num_rows

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    task_q = ctx.JoinableQueue()
    progress_q = ctx.Queue(maxsize=args.progress_q_max)

    for fp in part_files:
        task_q.put(fp)
    for _ in range(args.workers):
        task_q.put(None)

    procs = []
    for _ in range(args.workers):
        p = ctx.Process(target=worker, args=(task_q, progress_q, args.encoding, args.batch_size, args.column))
        p.start()
        procs.append(p)

    total_tokens = 0
    total_nulls = 0
    errors = 0
    workers_done = 0

    pbar = tqdm(
        total=total_rows,
        desc=f"Tokenizing (tiktoken:{args.encoding}, col:{args.column})",
        unit="rows",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        while workers_done < args.workers:
            try:
                msg = progress_q.get(timeout=5)
            except Exception:
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Workers exited non-zero: exitcodes={[p.exitcode for p in dead]}")
                if all(p.exitcode is not None for p in procs):
                    break
                continue

            if not msg:
                continue

            if msg[0] == "p":
                _, rows, toks, nulls = msg
                total_tokens += toks
                total_nulls += nulls
                pbar.update(rows)
                pbar.set_postfix(tokens=total_tokens, nulls=total_nulls, errors=errors)
            elif msg[0] == "e":
                _, fp, err = msg
                errors += 1
                print(f"\nERROR in {fp}: {err}", file=sys.stderr)
                pbar.set_postfix(tokens=total_tokens, nulls=total_nulls, errors=errors)
            elif msg[0] == "w_done":
                workers_done += 1
    finally:
        pbar.close()
        task_q.join()
        for p in procs:
            p.join(timeout=10)

    print("\nDONE")
    print("base:", args.base)
    print("files:", len(part_files))
    print("total_rows:", total_rows)
    print("null_rows:", total_nulls)
    print("total_tokens:", total_tokens)
    print("errors:", errors)


if __name__ == "__main__":
    main()

