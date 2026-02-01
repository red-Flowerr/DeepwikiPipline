import argparse
import json
import multiprocessing as mp
import os
import sys

import pyarrow.parquet as pq
from tqdm import tqdm


PARQUET_BASE_DEFAULT = "/mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki"
REPO_MAP_DEFAULT = "/opt/tiger/oss_server_only/DeepwikiPipline/result_data/repo_hdfs_map.json"
ENCODING_DEFAULT = "cl100k_base"
BATCH_SIZE_DEFAULT = 4096
WORKERS_DEFAULT = 16
PROGRESS_Q_MAX_DEFAULT = 2000


def load_repo_keys(repo_map_path: str) -> set[str]:
    obj = json.load(open(repo_map_path, "r", encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"repo_hdfs_map.json must be a dict, got {type(obj)}")
    m = obj.get("repo_hdfs_map")
    if not isinstance(m, dict):
        raise TypeError("repo_hdfs_map.json must contain key 'repo_hdfs_map' as a dict")
    return set(m.keys())


def worker(task_q, progress_q, repo_keys: set[str], encoding: str, batch_size: int, column: str):
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
            for batch in pf.iter_batches(columns=["repo_name", column], batch_size=batch_size):
                toks = 0
                matched_rows = 0
                total_rows = batch.num_rows
                repos = batch.column(0).to_pylist()
                contents = batch.column(1).to_pylist()
                for repo, content in zip(repos, contents):
                    if repo is None:
                        continue
                    if str(repo) not in repo_keys:
                        continue
                    matched_rows += 1
                    if content is None:
                        continue
                    toks += len(enc.encode_ordinary(content))
                progress_q.put(("p", total_rows, matched_rows, toks))
        except Exception as e:
            progress_q.put(("e", fp, repr(e)))
        finally:
            task_q.task_done()

    progress_q.put(("w_done",))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Count total tokens of parquet column (default: content) for rows whose repo_name exists in repo_hdfs_map.json."
        )
    )
    ap.add_argument("--parquet-base", default=PARQUET_BASE_DEFAULT, help="Parquet dir containing part-* (default: %(default)s)")
    ap.add_argument("--repo-map", default=REPO_MAP_DEFAULT, help="Path to repo_hdfs_map.json (default: %(default)s)")
    ap.add_argument("--column", default="content", help="Text column to tokenize (default: %(default)s)")
    ap.add_argument("--encoding", default=ENCODING_DEFAULT, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Rows per batch (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT, help="Worker processes (default: %(default)s)")
    ap.add_argument("--mp-start", default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--progress-q-max", type=int, default=PROGRESS_Q_MAX_DEFAULT)
    args = ap.parse_args()

    repo_keys = load_repo_keys(args.repo_map)
    if not repo_keys:
        raise SystemExit("No repo keys found in repo_hdfs_map.json")

    part_files = sorted(
        f for f in (os.path.join(args.parquet_base, fn) for fn in os.listdir(args.parquet_base)) if os.path.basename(f).startswith("part-")
    )
    if not part_files:
        raise SystemExit(f"No part-* files under {args.parquet_base}")

    # total rows for progress bar
    total_rows = 0
    for fp in tqdm(part_files, desc="Reading parquet footers", unit="file", dynamic_ncols=True, disable=not sys.stderr.isatty()):
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
        p = ctx.Process(target=worker, args=(task_q, progress_q, repo_keys, args.encoding, args.batch_size, args.column))
        p.start()
        procs.append(p)

    matched_rows_total = 0
    tokens_total = 0
    errors = 0
    workers_done = 0

    pbar = tqdm(
        total=total_rows,
        desc=f"Tokenizing parquet rows (col:{args.column}, tiktoken:{args.encoding})",
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
                _, rows, matched, toks = msg
                matched_rows_total += int(matched)
                tokens_total += int(toks)
                pbar.update(int(rows))
                pbar.set_postfix(tokens=tokens_total, matched_rows=matched_rows_total, errors=errors)
            elif msg[0] == "e":
                _, fp, err = msg
                errors += 1
                print(f"\nERROR in {fp}: {err}", file=sys.stderr)
                pbar.set_postfix(tokens=tokens_total, matched_rows=matched_rows_total, errors=errors)
            elif msg[0] == "w_done":
                workers_done += 1
    finally:
        pbar.close()
        task_q.join()
        for p in procs:
            p.join(timeout=10)

    print("\nDONE")
    print("parquet_base:", args.parquet_base)
    print("repo_map:", args.repo_map)
    print("repo_keys:", len(repo_keys))
    print("column:", args.column)
    print("total_rows:", total_rows)
    print("matched_rows:", matched_rows_total)
    print("total_tokens:", tokens_total)
    print("errors:", errors)


if __name__ == "__main__":
    main()

