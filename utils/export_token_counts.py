import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass

from tqdm import tqdm


ENCODING_DEFAULT = "cl100k_base"  # tiktoken encoding
NUM_WORKERS_DEFAULT = 16
PROGRESS_Q_MAX_DEFAULT = 2000


@dataclass(frozen=True)
class Task:
    kind: str  # deepwiki_txt | narratives_json | batch_narratives_json | result_data_deepwiki_json
    folder: str
    filepath: str


def _make_token_counter(
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

        # We only count tokens; we are not running the model. Avoid max_length warnings if possible.
        try:
            tok.model_max_length = 1 << 60
        except Exception:
            pass

        def count_tokens(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        return count_tokens

    raise ValueError(f"unknown tokenizer backend: {tokenizer_backend}")


def _list_subfolders(base_dir: str, only_folders: list[str] | None, max_folders: int) -> list[tuple[str, str]]:
    if only_folders:
        out: list[tuple[str, str]] = []
        for name in only_folders:
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                out.append((name, path))
        out.sort(key=lambda x: x[0])
        if max_folders and max_folders > 0:
            out = out[:max_folders]
        return out

    out = []
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_dir():
                continue
            if ent.name.startswith("."):
                continue
            out.append((ent.name, ent.path))
    out.sort(key=lambda x: x[0])
    if max_folders and max_folders > 0:
        out = out[:max_folders]
    return out


def _find_tasks_deepwiki_data(base_dir: str, only_folders: list[str] | None, max_folders: int, max_tasks: int) -> list[Task]:
    tasks: list[Task] = []
    for folder, folder_path in _list_subfolders(base_dir, only_folders=only_folders, max_folders=max_folders):
        with os.scandir(folder_path) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                if ent.name.endswith("_deepwiki.txt"):
                    tasks.append(Task(kind="deepwiki_txt", folder=folder, filepath=ent.path))
                elif ent.name.endswith("_narratives.json"):
                    tasks.append(Task(kind="narratives_json", folder=folder, filepath=ent.path))
                if max_tasks and max_tasks > 0 and len(tasks) >= max_tasks:
                    return tasks
    tasks.sort(key=lambda t: t.filepath)
    return tasks


def _find_tasks_batch_narratives(base_dir: str, max_tasks: int) -> list[Task]:
    tasks: list[Task] = []
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            if not ent.name.endswith(".json"):
                continue
            tasks.append(Task(kind="batch_narratives_json", folder="", filepath=ent.path))
            if max_tasks and max_tasks > 0 and len(tasks) >= max_tasks:
                break
    tasks.sort(key=lambda t: t.filepath)
    return tasks


def _find_tasks_result_data_deepwiki_json(base_dir: str, max_tasks: int) -> list[Task]:
    tasks: list[Task] = []
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            if not ent.name.endswith("_deepwiki.json"):
                continue
            tasks.append(Task(kind="result_data_deepwiki_json", folder="", filepath=ent.path))
            if max_tasks and max_tasks > 0 and len(tasks) >= max_tasks:
                break
    tasks.sort(key=lambda t: t.filepath)
    return tasks


def worker(task_q, progress_q, tokenizer_backend: str, encoding: str, hf_model: str, hf_trust_remote_code: bool, hf_local_files_only: bool):
    count_tokens = _make_token_counter(
        tokenizer_backend=tokenizer_backend,
        tiktoken_encoding=encoding,
        hf_model=hf_model,
        hf_trust_remote_code=hf_trust_remote_code,
        hf_local_files_only=hf_local_files_only,
    )

    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        try:
            if task.kind == "deepwiki_txt":
                with open(task.filepath, "r", encoding="utf-8", errors="replace") as f:
                    s = f.read()
                progress_q.put(("ok", {"kind": task.kind, "folder": task.folder, "filepath": task.filepath, "tokens": count_tokens(s)}))
            elif task.kind == "narratives_json":
                with open(task.filepath, "r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                if not isinstance(obj, list):
                    raise TypeError(f"narratives json is not a list: {type(obj)}")
                toks = 0
                missing = 0
                total = 0
                for row in obj:
                    total += 1
                    if not isinstance(row, dict):
                        missing += 1
                        continue
                    s = row.get("narrative")
                    if not s:
                        missing += 1
                        continue
                    if not isinstance(s, str):
                        s = str(s)
                    toks += count_tokens(s)
                progress_q.put(
                    (
                        "ok",
                        {
                            "kind": task.kind,
                            "folder": task.folder,
                            "filepath": task.filepath,
                            "tokens": toks,
                            "rows": total,
                            "missing_rows": missing,
                        },
                    )
                )
            elif task.kind == "batch_narratives_json":
                with open(task.filepath, "r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                if not isinstance(obj, list):
                    raise TypeError(f"batch narratives json is not a list: {type(obj)}")
                oc_toks = 0
                nar_toks = 0
                text_toks = 0
                missing_oc = 0
                missing_nar = 0
                missing_text = 0
                total = 0
                for row in obj:
                    total += 1
                    if not isinstance(row, dict):
                        missing_oc += 1
                        missing_nar += 1
                        missing_text += 1
                        continue
                    oc = row.get("original_context")
                    if not oc:
                        missing_oc += 1
                    else:
                        if not isinstance(oc, str):
                            oc = str(oc)
                        oc_toks += count_tokens(oc)
                    nar = row.get("narrative")
                    if not nar:
                        missing_nar += 1
                    else:
                        if not isinstance(nar, str):
                            nar = str(nar)
                        nar_toks += count_tokens(nar)
                    txt = row.get("text")
                    if not txt:
                        missing_text += 1
                    else:
                        if not isinstance(txt, str):
                            txt = str(txt)
                        text_toks += count_tokens(txt)
                progress_q.put(
                    (
                        "ok",
                        {
                            "kind": task.kind,
                            "folder": task.folder,
                            "filepath": task.filepath,
                            "rows": total,
                            "missing_original_context_rows": missing_oc,
                            "missing_narrative_rows": missing_nar,
                            "missing_text_rows": missing_text,
                            "original_context_tokens": oc_toks,
                            "narrative_tokens": nar_toks,
                            "text_tokens": text_toks,
                            "total_tokens": oc_toks + nar_toks + text_toks,
                        },
                    )
                )
            elif task.kind == "result_data_deepwiki_json":
                with open(task.filepath, "r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict):
                    raise TypeError(f"deepwiki json is not a dict: {type(obj)}")
                chunks = obj.get("chunks")
                if chunks is None:
                    raise KeyError("missing top-level 'chunks'")
                if not isinstance(chunks, list):
                    raise TypeError(f"chunks is not a list: {type(chunks)}")
                text_toks = 0
                missing_text = 0
                total = 0
                for ch in chunks:
                    total += 1
                    if not isinstance(ch, dict):
                        missing_text += 1
                        continue
                    txt = ch.get("text")
                    if not txt:
                        missing_text += 1
                        continue
                    if not isinstance(txt, str):
                        txt = str(txt)
                    text_toks += count_tokens(txt)
                progress_q.put(
                    (
                        "ok",
                        {
                            "kind": task.kind,
                            "folder": task.folder,
                            "filepath": task.filepath,
                            "chunks_total": total,
                            "chunks_missing_text": missing_text,
                            "chunks_text_tokens": text_toks,
                        },
                    )
                )
            else:
                raise ValueError(f"unknown task kind: {task.kind}")
        except Exception as e:
            progress_q.put(("err", {"kind": task.kind, "folder": task.folder, "filepath": task.filepath, "error": repr(e)}))
        finally:
            task_q.task_done()

    progress_q.put(("w_done", None))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export token statistics to a JSONL file for delivery.")
    ap.add_argument("--mode", required=True, choices=["deepwiki_data", "batch_narratives", "result_data_deepwiki_json"])
    ap.add_argument("--base", required=True, help="Input directory")
    ap.add_argument("--folders", nargs="*", default=None, help="Only for deepwiki_data: subfolder names under --base")
    ap.add_argument("--max-folders", type=int, default=0, help="Only for deepwiki_data: limit number of folders. 0 = all.")
    ap.add_argument("--max-tasks", type=int, default=0, help="Limit number of files (debug). 0 = all.")

    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"])
    ap.add_argument("--encoding", default=ENCODING_DEFAULT, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--hf-model", default="", help="HF model name/path (required when --tokenizer-backend=hf)")
    ap.add_argument("--hf-trust-remote-code", action="store_true")
    ap.add_argument("--hf-local-files-only", action="store_true", default=True)
    ap.add_argument("--hf-allow-download", action="store_false", dest="hf_local_files_only")

    ap.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT)
    ap.add_argument("--mp-start", default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--progress-q-max", type=int, default=PROGRESS_Q_MAX_DEFAULT)

    ap.add_argument("--output-jsonl", required=True, help="Write per-file results as JSONL (one line per file).")
    ap.add_argument("--output-summary", default="", help="Write final summary JSON (optional).")
    args = ap.parse_args()

    if args.tokenizer_backend == "hf" and not args.hf_model:
        raise SystemExit("--hf-model is required when --tokenizer-backend=hf")

    if args.mode == "deepwiki_data":
        tasks = _find_tasks_deepwiki_data(args.base, only_folders=args.folders, max_folders=args.max_folders, max_tasks=args.max_tasks)
    elif args.mode == "batch_narratives":
        tasks = _find_tasks_batch_narratives(args.base, max_tasks=args.max_tasks)
    else:
        tasks = _find_tasks_result_data_deepwiki_json(args.base, max_tasks=args.max_tasks)

    if not tasks:
        raise SystemExit(f"No input files found for mode={args.mode} under {args.base}")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    if args.output_summary:
        os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)

    ctx = mp.get_context(args.mp_start) if hasattr(mp, "get_context") else mp
    task_q = ctx.JoinableQueue()
    progress_q = ctx.Queue(maxsize=args.progress_q_max)

    for t in tasks:
        task_q.put(t)
    for _ in range(args.workers):
        task_q.put(None)

    procs = []
    for _ in range(args.workers):
        p = ctx.Process(
            target=worker,
            args=(task_q, progress_q, args.tokenizer_backend, args.encoding, args.hf_model, args.hf_trust_remote_code, args.hf_local_files_only),
        )
        p.start()
        procs.append(p)

    summary = {
        "mode": args.mode,
        "base": args.base,
        "tokenizer_backend": args.tokenizer_backend,
        "encoding": args.encoding,
        "hf_model": args.hf_model,
        "files_total": len(tasks),
        "files_ok": 0,
        "files_err": 0,
    }

    # A few common aggregate keys (some modes won't fill all of them)
    aggregates = {
        "deepwiki_txt_tokens": 0,
        "narratives_tokens": 0,
        "narratives_rows": 0,
        "narratives_missing_rows": 0,
        "batch_original_context_tokens": 0,
        "batch_narrative_tokens": 0,
        "batch_text_tokens": 0,
        "batch_rows": 0,
        "deepwiki_chunks_text_tokens": 0,
        "deepwiki_chunks_total": 0,
        "deepwiki_chunks_missing_text": 0,
    }

    workers_done = 0
    errors = 0

    pbar = tqdm(
        total=len(tasks),
        desc=(
            f"Exporting tokens (tiktoken:{args.encoding})"
            if args.tokenizer_backend == "tiktoken"
            else f"Exporting tokens (hf:{args.hf_model})"
        ),
        unit="file",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        try:
            while workers_done < args.workers:
                try:
                    msg_type, payload = progress_q.get(timeout=5)
                except Exception:
                    dead = [p for p in procs if p.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f"Workers exited non-zero: exitcodes={[p.exitcode for p in dead]}")
                    if all(p.exitcode is not None for p in procs):
                        break
                    continue

                if msg_type == "w_done":
                    workers_done += 1
                    continue

                if msg_type == "ok":
                    summary["files_ok"] += 1
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.flush()

                    kind = payload.get("kind")
                    if kind == "deepwiki_txt":
                        aggregates["deepwiki_txt_tokens"] += int(payload.get("tokens", 0) or 0)
                    elif kind == "narratives_json":
                        aggregates["narratives_tokens"] += int(payload.get("tokens", 0) or 0)
                        aggregates["narratives_rows"] += int(payload.get("rows", 0) or 0)
                        aggregates["narratives_missing_rows"] += int(payload.get("missing_rows", 0) or 0)
                    elif kind == "batch_narratives_json":
                        aggregates["batch_original_context_tokens"] += int(payload.get("original_context_tokens", 0) or 0)
                        aggregates["batch_narrative_tokens"] += int(payload.get("narrative_tokens", 0) or 0)
                        aggregates["batch_text_tokens"] += int(payload.get("text_tokens", 0) or 0)
                        aggregates["batch_rows"] += int(payload.get("rows", 0) or 0)
                    elif kind == "result_data_deepwiki_json":
                        aggregates["deepwiki_chunks_text_tokens"] += int(payload.get("chunks_text_tokens", 0) or 0)
                        aggregates["deepwiki_chunks_total"] += int(payload.get("chunks_total", 0) or 0)
                        aggregates["deepwiki_chunks_missing_text"] += int(payload.get("chunks_missing_text", 0) or 0)

                    pbar.update(1)
                    pbar.set_postfix(ok=summary["files_ok"], err=summary["files_err"])
                elif msg_type == "err":
                    summary["files_err"] += 1
                    errors += 1
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"\nERROR {payload.get('filepath')}: {payload.get('error')}", file=sys.stderr)
                    pbar.update(1)
                    pbar.set_postfix(ok=summary["files_ok"], err=summary["files_err"])
        finally:
            pbar.close()
            task_q.join()
            for p in procs:
                p.join(timeout=10)

    summary["errors"] = errors
    summary.update(aggregates)

    # mode-specific convenient totals
    if args.mode == "deepwiki_data":
        summary["total_tokens"] = aggregates["deepwiki_txt_tokens"] + aggregates["narratives_tokens"]
    elif args.mode == "batch_narratives":
        summary["total_tokens"] = aggregates["batch_original_context_tokens"] + aggregates["batch_narrative_tokens"] + aggregates["batch_text_tokens"]
    else:
        summary["total_tokens"] = aggregates["deepwiki_chunks_text_tokens"]

    print("\nDONE")
    print("output_jsonl:", args.output_jsonl)
    if args.output_summary:
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("output_summary:", args.output_summary)
    print("mode:", summary["mode"])
    print("files_ok:", summary["files_ok"])
    print("files_err:", summary["files_err"])
    print("total_tokens:", summary["total_tokens"])


if __name__ == "__main__":
    main()

