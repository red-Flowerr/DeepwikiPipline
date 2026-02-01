import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import pyarrow.parquet as pq
from tqdm import tqdm


PARQUET_BASE_DEFAULT = "/mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki"
ENCODING_DEFAULT = "cl100k_base"  # 可改 "o200k_base"
NUM_WORKERS_DEFAULT = 16
PROGRESS_Q_MAX_DEFAULT = 2000
BATCH_SIZE_DEFAULT = 4096


_BINARY_EXTS = {
    ".7z",
    ".a",
    ".apk",
    ".bin",
    ".bmp",
    ".bz2",
    ".class",
    ".db",
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lz4",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".otf",
    ".pdf",
    ".png",
    ".psd",
    ".rar",
    ".so",
    ".tar",
    ".tiff",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}


@dataclass(frozen=True)
class ZipTask:
    zip_path: str


def _read_results_jsonl(path: str) -> tuple[set[str], dict]:
    """
    Returns:
      done_zip_paths, aggregates
    """
    done: set[str] = set()
    agg = {
        "zips_done": 0,
        "included_files": 0,
        "skipped_files": 0,
        "bytes_read": 0,
        "total_tokens": 0,
        "errors": 0,
        "transport_errors_errno_107": 0,
        "total_retries_used": 0,
    }

    if not path or not os.path.exists(path):
        return done, agg

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            zip_path = row.get("zip_path")
            status = row.get("status")
            if not zip_path or status not in ("ok", "err"):
                continue
            zip_path = str(zip_path)
            if zip_path in done:
                continue
            done.add(zip_path)
            agg["zips_done"] += 1
            if status == "ok":
                agg["included_files"] += int(row.get("included_files", 0) or 0)
                agg["skipped_files"] += int(row.get("skipped_files", 0) or 0)
                agg["bytes_read"] += int(row.get("bytes_read", 0) or 0)
                agg["total_tokens"] += int(row.get("tokens", 0) or 0)
                agg["total_retries_used"] += int(row.get("retries_used", 0) or 0)
            else:
                agg["errors"] += 1
                errno = row.get("errno")
                if errno == 107 or errno == "107":
                    agg["transport_errors_errno_107"] += 1

    return done, agg


def _write_failures_from_results(results_jsonl: str, failures_jsonl: str) -> int:
    if not results_jsonl or not os.path.exists(results_jsonl):
        return 0
    os.makedirs(os.path.dirname(failures_jsonl) or ".", exist_ok=True)
    n = 0
    with open(results_jsonl, "r", encoding="utf-8", errors="replace") as src, open(
        failures_jsonl, "w", encoding="utf-8"
    ) as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("status") != "err":
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _hdfs_uri_to_local_path(uri: str) -> str:
    """
    Try to map `hdfs://<nn>/<path>` to a locally mounted path.
    Common patterns:
      - /mnt/hdfs/<nn>/<path>
      - /mnt/hdfs/<path>
    """
    if uri.startswith("/"):
        return uri
    if not uri.startswith("hdfs://"):
        return uri

    p = urlparse(uri)
    # Example: hdfs://haruna/home/... -> netloc=haruna, path=/home/...
    candidates = []
    if p.netloc and p.path:
        candidates.append(os.path.join("/mnt/hdfs", p.netloc, p.path.lstrip("/")))
    if p.path:
        candidates.append(os.path.join("/mnt/hdfs", p.path.lstrip("/")))

    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0] if candidates else uri


def _hdfs_uri_to_known_mount(uri: str, known_prefix: str) -> str:
    """
    If hdfs_path URIs are like:
      hdfs://<nn>/home/byte_isp_datamart/...
    but locally mounted as:
      /mnt/hdfs/byte_isp_datamart/...
    then this strips the leading `/home/` from path and joins to known_prefix.
    """
    if not uri.startswith("hdfs://"):
        return uri
    p = urlparse(uri)
    path = (p.path or "").lstrip("/")
    if path.startswith("home/"):
        path = path[len("home/") :]
    return os.path.join(known_prefix, path)


def _path_exists_with_errno(path: str) -> tuple[bool, int | None]:
    try:
        os.stat(path)
        return True, None
    except OSError as e:
        return False, getattr(e, "errno", None)


def _iter_hdfs_paths_from_parquet(parquet_base: str, batch_size: int):
    part_files = sorted(
        f for f in (os.path.join(parquet_base, fn) for fn in os.listdir(parquet_base)) if os.path.basename(f).startswith("part-")
    )
    if not part_files:
        raise SystemExit(f"No part-* files under {parquet_base}")

    for fp in tqdm(part_files, desc="Scanning parquet (hdfs_path)", unit="file", dynamic_ncols=True, disable=not sys.stderr.isatty()):
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(columns=["hdfs_path"], batch_size=batch_size):
            for v in batch.column(0).to_pylist():
                if not v:
                    continue
                yield str(v)


def _looks_binary(buf: bytes) -> bool:
    # Simple and fast heuristic
    return b"\x00" in buf


def _count_tokens_in_zip(zip_path: str, enc, max_file_bytes: int) -> tuple[int, int, int, int]:
    """
    Returns:
      repo_tokens, files_included, files_skipped, bytes_read
    """
    import zipfile

    repo_tokens = 0
    files_included = 0
    files_skipped = 0
    bytes_read = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [zi for zi in zf.infolist() if not zi.is_dir()]
        infos.sort(key=lambda x: x.filename)

        sep_tokens = len(enc.encode_ordinary("\n"))
        first = True
        for zi in infos:
            name = zi.filename
            ext = os.path.splitext(name)[1].lower()
            if ext in _BINARY_EXTS:
                files_skipped += 1
                continue
            if max_file_bytes and zi.file_size > max_file_bytes:
                files_skipped += 1
                continue

            with zf.open(zi, "r") as f:
                head = f.read(4096)
                if _looks_binary(head):
                    files_skipped += 1
                    continue
                rest = f.read()
                data = head + rest
                bytes_read += len(data)
                text = data.decode("utf-8", errors="replace")

            if not first:
                repo_tokens += sep_tokens
            first = False
            repo_tokens += len(enc.encode_ordinary(text))
            files_included += 1

    return repo_tokens, files_included, files_skipped, bytes_read


def _is_retryable_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 107:
        return True
    msg = str(exc)
    return "Transport endpoint is not connected" in msg


def worker(task_q, progress_q, encoding: str, max_file_bytes: int):
    import tiktoken

    enc = tiktoken.get_encoding(encoding)

    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        try:
            # retry on transient mount/network errors (e.g. errno 107)
            retries = 0
            max_retries = int(os.environ.get("REPO_ZIP_TOKEN_RETRIES", "5"))
            base_sleep = float(os.environ.get("REPO_ZIP_TOKEN_RETRY_BASE_SLEEP", "1.0"))
            max_sleep = float(os.environ.get("REPO_ZIP_TOKEN_RETRY_MAX_SLEEP", "30.0"))

            while True:
                try:
                    toks, included, skipped, bytes_read = _count_tokens_in_zip(
                        task.zip_path, enc=enc, max_file_bytes=max_file_bytes
                    )
                    progress_q.put(("ok", task.zip_path, toks, included, skipped, bytes_read, retries))
                    break
                except Exception as e:
                    if _is_retryable_transport_error(e) and retries < max_retries:
                        sleep_s = min(max_sleep, base_sleep * (2**retries))
                        # jitter to avoid thundering herd
                        sleep_s = sleep_s * (0.5 + random.random())
                        retries += 1
                        time.sleep(sleep_s)
                        continue
                    raise
        except Exception as e:
            errno = getattr(e, "errno", None) if isinstance(e, OSError) else None
            progress_q.put(("err", task.zip_path, repr(e), errno))
        finally:
            task_q.task_done()

    progress_q.put(("w_done",))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Scan parquet 'hdfs_path' under a dataset, open each referenced repo zip, "
            "concatenate file contents (approx: sum(tokens per file) with '\\n' between files), and count total tokens."
        )
    )
    ap.add_argument("--parquet-base", default=PARQUET_BASE_DEFAULT, help="Parquet dir containing part-* (default: %(default)s)")
    ap.add_argument("--encoding", default=ENCODING_DEFAULT, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT, help="Number of worker processes (default: %(default)s)")
    ap.add_argument("--mp-start", default="spawn", choices=["spawn", "fork", "forkserver"], help="Multiprocessing start method")
    ap.add_argument("--progress-q-max", type=int, default=PROGRESS_Q_MAX_DEFAULT, help="Maxsize for progress queue")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Parquet scan batch size (default: %(default)s)")
    ap.add_argument("--dedupe", action="store_true", default=True, help="Dedupe identical hdfs_path values (default: on)")
    ap.add_argument("--no-dedupe", action="store_false", dest="dedupe", help="Do not dedupe identical hdfs_path values")
    ap.add_argument("--max-zips", type=int, default=0, help="Only process first N zips (debug). 0 = all.")
    ap.add_argument(
        "--known-hdfs-mount",
        default="/mnt/hdfs",
        help="Known HDFS mount prefix for resolving hdfs:// URIs (default: %(default)s)",
    )
    ap.add_argument(
        "--allow-scan-errno107",
        action="store_true",
        default=True,
        help=(
            "If zip existence check hits errno=107 (Transport endpoint not connected), still enqueue the zip and let "
            "the worker retry (default: on)."
        ),
    )
    ap.add_argument(
        "--no-allow-scan-errno107",
        action="store_false",
        dest="allow_scan_errno107",
        help="Disable enqueuing zips that fail existence check with errno=107.",
    )
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=0,
        help="Skip files larger than N bytes inside zips. 0 = no limit.",
    )
    ap.add_argument("--retries", type=int, default=5, help="Retries for transient mount/network errors (default: %(default)s)")
    ap.add_argument("--retry-base-sleep", type=float, default=1.0, help="Retry base sleep seconds (default: %(default)s)")
    ap.add_argument("--retry-max-sleep", type=float, default=30.0, help="Retry max sleep seconds (default: %(default)s)")
    ap.add_argument(
        "--results-jsonl",
        default="",
        help="Write per-zip results (ok/err) to a JSONL file for checkpoint/resume (optional).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --results-jsonl (skips already-processed zips and starts totals from it).",
    )
    ap.add_argument(
        "--failures-jsonl",
        default="",
        help="Write failures to a JSONL file by filtering --results-jsonl (optional; overwrites).",
    )
    ap.add_argument("--output-json", default="", help="Write summary JSON to a file (optional).")
    args = ap.parse_args()

    if args.resume and not args.results_jsonl:
        raise SystemExit("--resume requires --results-jsonl")

    # pass retry config via env to keep worker args simple/picklable
    os.environ["REPO_ZIP_TOKEN_RETRIES"] = str(int(args.retries))
    os.environ["REPO_ZIP_TOKEN_RETRY_BASE_SLEEP"] = str(float(args.retry_base_sleep))
    os.environ["REPO_ZIP_TOKEN_RETRY_MAX_SLEEP"] = str(float(args.retry_max_sleep))

    done_zip_paths, prior = _read_results_jsonl(args.results_jsonl) if args.resume else (set(), {})

    # 1) Collect zip paths from parquet
    seen = set()
    zip_paths: list[str] = []
    missing_local = 0
    scan_errno_107 = 0
    skipped_done = 0

    for uri in _iter_hdfs_paths_from_parquet(args.parquet_base, batch_size=args.batch_size):
        local = _hdfs_uri_to_local_path(uri)
        exists, errno = _path_exists_with_errno(local)
        if not exists:
            # Try common mapping for this dataset: /home/byte_isp_datamart/... -> /mnt/hdfs/byte_isp_datamart/...
            local2 = _hdfs_uri_to_known_mount(uri, known_prefix=args.known_hdfs_mount)
            exists2, errno2 = _path_exists_with_errno(local2)
            if exists2:
                local = local2
                exists, errno = True, None
            else:
                # keep errno from best-effort check
                errno = errno2 if errno2 is not None else errno
        if args.dedupe:
            if local in seen:
                continue
            seen.add(local)
        if args.resume and local in done_zip_paths:
            skipped_done += 1
            continue
        if not exists:
            if args.allow_scan_errno107 and errno == 107:
                # Mount is temporarily disconnected; enqueue anyway so worker retry can handle it.
                scan_errno_107 += 1
                zip_paths.append(local)
                if args.max_zips and args.max_zips > 0 and len(zip_paths) >= args.max_zips:
                    break
                continue
            missing_local += 1
            continue
        zip_paths.append(local)
        if args.max_zips and args.max_zips > 0 and len(zip_paths) >= args.max_zips:
            break

    if not zip_paths:
        if args.resume and prior.get("zips_done", 0) > 0:
            print("\nDONE")
            print("parquet_base:", args.parquet_base)
            print("encoding:", args.encoding)
            print("dedupe:", args.dedupe)
            print("zips_processed:", int(prior.get("zips_done", 0)))
            print("missing_local_zips:", missing_local)
            print("included_files:", int(prior.get("included_files", 0)))
            print("skipped_files:", int(prior.get("skipped_files", 0)))
            print("bytes_read:", int(prior.get("bytes_read", 0)))
            print("total_tokens:", int(prior.get("total_tokens", 0)))
            print("errors:", int(prior.get("errors", 0)))
            print("transport_errors_errno_107:", int(prior.get("transport_errors_errno_107", 0)))
            print("total_retries_used:", int(prior.get("total_retries_used", 0)))
            print("skipped_already_done:", skipped_done)
            return
        raise SystemExit("No zip files found (after mapping hdfs_path -> local path).")

    # 2) Spawn workers
    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    task_q = ctx.JoinableQueue()
    progress_q = ctx.Queue(maxsize=args.progress_q_max)

    for zp in zip_paths:
        task_q.put(ZipTask(zip_path=zp))
    for _ in range(args.workers):
        task_q.put(None)

    procs = []
    for _ in range(args.workers):
        p = ctx.Process(target=worker, args=(task_q, progress_q, args.encoding, args.max_file_bytes))
        p.start()
        procs.append(p)

    # 3) Aggregate
    total_tokens = int(prior.get("total_tokens", 0) or 0) if args.resume else 0
    total_included_files = int(prior.get("included_files", 0) or 0) if args.resume else 0
    total_skipped_files = int(prior.get("skipped_files", 0) or 0) if args.resume else 0
    total_bytes_read = int(prior.get("bytes_read", 0) or 0) if args.resume else 0
    errors = int(prior.get("errors", 0) or 0) if args.resume else 0
    transport_errors_107 = int(prior.get("transport_errors_errno_107", 0) or 0) if args.resume else 0
    total_retries_used = int(prior.get("total_retries_used", 0) or 0) if args.resume else 0
    workers_done = 0

    results_fh = None
    if args.results_jsonl:
        os.makedirs(os.path.dirname(args.results_jsonl) or ".", exist_ok=True)
        results_fh = open(args.results_jsonl, "a", encoding="utf-8")

    pbar = tqdm(
        total=len(zip_paths),
        desc=f"Tokenizing repos (tiktoken:{args.encoding})",
        unit="zip",
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

            if msg[0] == "ok":
                _, zip_path, toks, included, skipped, bytes_read, retries_used = msg
                total_tokens += int(toks)
                total_included_files += int(included)
                total_skipped_files += int(skipped)
                total_bytes_read += int(bytes_read)
                total_retries_used += int(retries_used)
                if results_fh is not None:
                    results_fh.write(
                        json.dumps(
                            {
                                "zip_path": zip_path,
                                "status": "ok",
                                "tokens": int(toks),
                                "included_files": int(included),
                                "skipped_files": int(skipped),
                                "bytes_read": int(bytes_read),
                                "retries_used": int(retries_used),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    results_fh.flush()
                pbar.update(1)
                pbar.set_postfix(tokens=total_tokens, errors=errors)
            elif msg[0] == "err":
                _, zip_path, err, errno = msg
                errors += 1
                if errno == 107:
                    transport_errors_107 += 1
                if results_fh is not None:
                    results_fh.write(
                        json.dumps(
                            {
                                "zip_path": zip_path,
                                "status": "err",
                                "error": err,
                                "errno": errno,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    results_fh.flush()
                print(f"\nERROR {zip_path}: {err}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(tokens=total_tokens, errors=errors)
            elif msg[0] == "w_done":
                workers_done += 1
    finally:
        pbar.close()
        task_q.join()
        for p in procs:
            p.join(timeout=10)
        if results_fh is not None:
            results_fh.close()

    summary = {
        "parquet_base": args.parquet_base,
        "encoding": args.encoding,
        "dedupe": args.dedupe,
        "zips_processed": (int(prior.get("zips_done", 0) or 0) if args.resume else 0) + len(zip_paths),
        "zips_newly_processed": len(zip_paths),
        "missing_local_zips": missing_local,
        "scan_errno_107_enqueued": scan_errno_107,
        "skipped_already_done": skipped_done,
        "included_files": total_included_files,
        "skipped_files": total_skipped_files,
        "bytes_read": total_bytes_read,
        "total_tokens": total_tokens,
        "errors": errors,
        "transport_errors_errno_107": transport_errors_107,
        "total_retries_used": total_retries_used,
        "results_jsonl": args.results_jsonl,
    }

    print("\nDONE")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\nWROTE")
        print("output_json:", args.output_json)

    if args.failures_jsonl:
        if not args.results_jsonl:
            raise SystemExit("--failures-jsonl requires --results-jsonl (to filter from results)")
        n_fail = _write_failures_from_results(args.results_jsonl, args.failures_jsonl)
        print("\nWROTE")
        print("failures_jsonl:", args.failures_jsonl)
        print("failures:", n_fail)


if __name__ == "__main__":
    main()
