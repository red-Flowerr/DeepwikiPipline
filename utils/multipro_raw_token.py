import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass

from tqdm import tqdm

BASE = "/mnt/hdfs/user_wl/xingtianshun/deepwiki_data"
ENCODING = "cl100k_base"  # tiktoken: 可改 "o200k_base"
NUM_WORKERS = 96  # 先 64/96/128 试
PROGRESS_Q_MAX = 2000  # 控制IPC队列大小，防止内存飙


@dataclass(frozen=True)
class Task:
    kind: str  # "deepwiki_txt" | "narratives_json" | "batch_narratives_json"
    folder: str
    filepath: str


def _make_token_counter(
    tokenizer_backend: str,
    tiktoken_encoding: str,
    hf_model: str,
    hf_trust_remote_code: bool,
    hf_local_files_only: bool,
):
    """
    Returns a callable: (text: str) -> int token_count
    """
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
        return out

    out = []
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_dir():
                continue
            if ent.name.startswith("."):
                continue
            out.append((ent.name, ent.path))
            if max_folders and max_folders > 0 and len(out) >= max_folders:
                break
    out.sort(key=lambda x: x[0])
    return out


def _find_tasks(base_dir: str, only_folders: list[str] | None, max_folders: int, max_tasks: int) -> list[Task]:
    tasks: list[Task] = []
    folders = _list_subfolders(base_dir, only_folders=only_folders, max_folders=max_folders)
    for folder, folder_path in folders:
        # Use scandir to avoid expensive glob over large trees.
        with os.scandir(folder_path) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                name = ent.name
                if name.endswith("_deepwiki.txt"):
                    tasks.append(Task(kind="deepwiki_txt", folder=folder, filepath=ent.path))
                elif name.endswith("_narratives.json"):
                    tasks.append(Task(kind="narratives_json", folder=folder, filepath=ent.path))

                if max_tasks and max_tasks > 0 and len(tasks) >= max_tasks:
                    return tasks

    return tasks


def _find_batch_narratives_tasks(base_dir: str, max_tasks: int) -> list[Task]:
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


def _find_result_data_deepwiki_json_tasks(base_dir: str, max_tasks: int) -> list[Task]:
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
                toks = count_tokens(s)
                progress_q.put(("ok_txt", toks))
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
                progress_q.put(("ok_json", toks, total, missing))
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
                progress_q.put(("ok_batch", oc_toks, nar_toks, text_toks, total, missing_oc, missing_nar, missing_text))
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
                progress_q.put(("ok_result_deepwiki", text_toks, total, missing_text))
            else:
                raise ValueError(f"unknown task kind: {task.kind}")
        except Exception as e:
            progress_q.put(("err", task.kind, task.folder, task.filepath, repr(e)))
        finally:
            task_q.task_done()

    progress_q.put(("w_done",))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Count tokens under a base dir. For each subfolder, sums tokens of *_deepwiki.txt "
            "and sums tokens of each row's 'narrative' field in *_narratives.json."
        )
    )
    ap.add_argument(
        "--mode",
        default="deepwiki_data",
        choices=["deepwiki_data", "batch_narratives", "result_data_deepwiki_json"],
        help="Input mode (default: %(default)s).",
    )
    ap.add_argument("--base", default=BASE, help="Base directory (default: %(default)s)")
    ap.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Only process these subfolder names under --base (space-separated).",
    )
    ap.add_argument("--max-folders", type=int, default=0, help="Only process first N folders (debug). 0 = all.")
    ap.add_argument("--max-tasks", type=int, default=0, help="Only process first N files (debug). 0 = all.")
    ap.add_argument("--tokenizer-backend", default="tiktoken", choices=["tiktoken", "hf"], help="Tokenize backend (default: %(default)s)")
    ap.add_argument("--encoding", default=ENCODING, help="tiktoken encoding name (default: %(default)s)")
    ap.add_argument("--hf-model", default="", help="HuggingFace model name/path (required for --tokenizer-backend=hf)")
    ap.add_argument("--hf-trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer")
    ap.add_argument("--hf-local-files-only", action="store_true", default=True, help="Do not download models; use local files only (default: on)")
    ap.add_argument("--hf-allow-download", action="store_false", dest="hf_local_files_only", help="Allow AutoTokenizer to download missing files")
    ap.add_argument(
        "--mp-start",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: %(default)s). Use spawn to avoid fork-unsafe libs.",
    )
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    ap.add_argument("--progress-q-max", type=int, default=PROGRESS_Q_MAX, help="Maxsize for progress queue")
    args = ap.parse_args()

    if args.mode == "deepwiki_data":
        tasks = _find_tasks(
            args.base,
            only_folders=args.folders,
            max_folders=args.max_folders,
            max_tasks=args.max_tasks,
        )
    elif args.mode == "batch_narratives":
        tasks = _find_batch_narratives_tasks(args.base, max_tasks=args.max_tasks)
    else:
        tasks = _find_result_data_deepwiki_json_tasks(args.base, max_tasks=args.max_tasks)
    if not tasks:
        raise SystemExit(f"No tasks found under {args.base} for mode={args.mode}.")

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
            args=(
                task_q,
                progress_q,
                args.tokenizer_backend,
                args.encoding,
                args.hf_model,
                args.hf_trust_remote_code,
                args.hf_local_files_only,
            ),
        )
        p.start()
        procs.append(p)

    errors = 0
    workers_done = 0

    deepwiki_total_tokens = 0
    narratives_total_tokens = 0
    narratives_total_rows = 0
    narratives_missing_rows = 0
    deepwiki_files = 0
    narratives_files = 0
    batch_files = 0
    batch_total_rows = 0
    batch_missing_oc_rows = 0
    batch_missing_nar_rows = 0
    batch_missing_text_rows = 0
    batch_original_context_tokens = 0
    batch_narrative_tokens = 0
    batch_text_tokens = 0
    deepwiki_json_files = 0
    deepwiki_chunks = 0
    deepwiki_missing_text_chunks = 0
    deepwiki_chunks_text_tokens = 0

    pbar = tqdm(
        total=len(tasks),
        desc=(
            f"Tokenizing (tiktoken:{args.encoding})"
            if args.tokenizer_backend == "tiktoken"
            else f"Tokenizing (hf:{args.hf_model})"
        ),
        unit="file",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        while workers_done < args.workers:
            try:
                msg = progress_q.get(timeout=5)
            except Exception:
                # No progress messages; check for worker failures to avoid hanging forever
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(
                        "One or more worker processes exited non-zero. "
                        f"exitcodes={[p.exitcode for p in dead]}"
                    )
                # If all workers exited (even 0), stop waiting.
                if all(p.exitcode is not None for p in procs):
                    break
                # Otherwise keep waiting.
                continue
            if not msg:
                continue

            kind = msg[0]
            if kind == "ok_txt":
                _, toks = msg
                deepwiki_total_tokens += toks
                deepwiki_files += 1
                pbar.update(1)
                pbar.set_postfix(
                    deepwiki_tokens=deepwiki_total_tokens,
                    narratives_tokens=narratives_total_tokens,
                    errors=errors,
                )
            elif kind == "ok_json":
                _, toks, total, missing = msg
                narratives_total_tokens += toks
                narratives_total_rows += total
                narratives_missing_rows += missing
                narratives_files += 1
                pbar.update(1)
                pbar.set_postfix(
                    deepwiki_tokens=deepwiki_total_tokens,
                    narratives_tokens=narratives_total_tokens,
                    errors=errors,
                )
            elif kind == "ok_batch":
                _, oc_toks, nar_toks, text_toks, total, missing_oc, missing_nar, missing_text = msg
                batch_original_context_tokens += oc_toks
                batch_narrative_tokens += nar_toks
                batch_text_tokens += text_toks
                batch_total_rows += total
                batch_missing_oc_rows += missing_oc
                batch_missing_nar_rows += missing_nar
                batch_missing_text_rows += missing_text
                batch_files += 1
                pbar.update(1)
                pbar.set_postfix(
                    oc_tokens=batch_original_context_tokens,
                    nar_tokens=batch_narrative_tokens,
                    text_tokens=batch_text_tokens,
                    errors=errors,
                )
            elif kind == "ok_result_deepwiki":
                _, text_toks, total, missing_text = msg
                deepwiki_chunks_text_tokens += text_toks
                deepwiki_chunks += total
                deepwiki_missing_text_chunks += missing_text
                deepwiki_json_files += 1
                pbar.update(1)
                pbar.set_postfix(text_tokens=deepwiki_chunks_text_tokens, errors=errors)
            elif kind == "err":
                _, tkind, folder, fp, err = msg
                errors += 1
                print(f"\nERROR [{tkind}] {folder} {fp}: {err}", file=sys.stderr)
                pbar.set_postfix(
                    deepwiki_tokens=deepwiki_total_tokens,
                    narratives_tokens=narratives_total_tokens,
                    errors=errors,
                )
                pbar.update(1)
            elif kind == "w_done":
                workers_done += 1
            else:
                # ignore unknown messages
                continue
    finally:
        pbar.close()
        task_q.join()
        for p in procs:
            p.join(timeout=5)

    print("\nDONE")
    print("base:", args.base)
    print("mode:", args.mode)
    if args.mode == "deepwiki_data":
        print("deepwiki_files:", deepwiki_files)
        print("narratives_files:", narratives_files)
        print("deepwiki_total_tokens:", deepwiki_total_tokens)
        print("narratives_total_tokens:", narratives_total_tokens)
        print("narratives_total_rows:", narratives_total_rows)
        print("narratives_missing_rows:", narratives_missing_rows)
        print("total_tokens:", deepwiki_total_tokens + narratives_total_tokens)
    elif args.mode == "batch_narratives":
        print("json_files:", batch_files)
        print("total_rows:", batch_total_rows)
        print("missing_original_context_rows:", batch_missing_oc_rows)
        print("missing_narrative_rows:", batch_missing_nar_rows)
        print("missing_text_rows:", batch_missing_text_rows)
        print("original_context_total_tokens:", batch_original_context_tokens)
        print("narrative_total_tokens:", batch_narrative_tokens)
        print("text_total_tokens:", batch_text_tokens)
        print("total_tokens:", batch_original_context_tokens + batch_narrative_tokens + batch_text_tokens)
    else:
        print("deepwiki_json_files:", deepwiki_json_files)
        print("chunks_total:", deepwiki_chunks)
        print("chunks_missing_text:", deepwiki_missing_text_chunks)
        print("chunks_text_total_tokens:", deepwiki_chunks_text_tokens)
    print("errors:", errors)


if __name__ == "__main__":
    main()
