#!/usr/bin/env python3
"""Command-line interface for the DeepWiki MCP pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # Optional progress indicator
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:  # Optional for multiprocess parquet streaming
    import multiprocessing as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]

from deepwiki_pipeline import (
    DeepWikiPipeline,
    MCPError,
    Session,
    delete_session,
    extract_text_blocks,
    initialize_session,
    list_tools,
)
from deepwiki_pipeline.mcp import call_tool
from deepwiki_pipeline.models import JudgeLLMConfig, NarrativeLLMConfig, PipelineOutput
from deepwiki_pipeline.parsing import normalize_heading, parse_wiki_markdown

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepoTarget:
    repo: str
    commit: Optional[str] = None
    hdfs_zip_path: Optional[str] = None


@dataclass
class _ParquetStreamStats:
    processed: int = 0
    generated: int = 0
    skipped_by_hdfs: int = 0
    skipped_by_local: int = 0
    errors: int = 0


def _print_tools(session: Session) -> None:
    response = list_tools(session)
    tools = response.get("result", {}).get("tools", [])
    if not tools:
        print("No tools reported by the server.")
        return
    for tool in tools:
        name = tool.get("name")
        description = tool.get("description", "")
        print(f"- {name}: {description}")


def _print_structure(session: Session, repo: str) -> None:
    response = call_tool(session, "read_wiki_structure", {"repoName": repo})
    blocks = extract_text_blocks(response)
    if not blocks:
        raise MCPError("read_wiki_structure returned no textual content.")
    print("\n\n".join(blocks))


def _apply_content_filters(
    markdown: str,
    *,
    page: Optional[str],
    section: Optional[str],
    contains: Optional[str],
) -> str:
    if not page and not section and not contains:
        return markdown
    pages = parse_wiki_markdown(markdown)
    result_chunks: List[str] = []
    contains_lower = contains.lower() if contains else None

    def page_matches(key: str) -> bool:
        if not page:
            return True
        target_norm = normalize_heading(page)
        return key == target_norm

    for key, page_content in pages.items():
        
        if not page_matches(key):
            continue
        if section is None:
            text = page_content.full_text()
            if contains_lower and contains_lower not in text.lower():
                continue
            result_chunks.append(f"# {page_content.title}\n\n{text}")
            continue
        try:
            section_content = page_content.section_text(section)
        except KeyError:
            continue
        text = section_content.text
        if contains_lower and contains_lower not in text.lower():
            continue
        heading = section_content.heading or section
        result_chunks.append(
            f"# {page_content.title} :: {heading}\n\n{text.strip()}"
        )
    if not result_chunks:
        raise MCPError("No content matched the requested filters.")
    return "\n\n".join(result_chunks)


def _print_contents(
    session: Session,
    repo: str,
    *,
    page: Optional[str],
    section: Optional[str],
    contains: Optional[str],
) -> None:
    response = call_tool(session, "read_wiki_contents", {"repoName": repo})
    blocks = extract_text_blocks(response)
    if not blocks:
        raise MCPError("read_wiki_contents returned no textual content.")
    markdown = "\n\n".join(blocks)
    filtered = _apply_content_filters(
        markdown,
        page=page,
        section=section,
        contains=contains,
    )
    print(filtered)


def _ask_question_json(
    session: Session,
    repo: str,
    question: str,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_payload = ""
    for _ in range(retries + 1):
        response = call_tool(
            session,
            "ask_question",
            {"repoName": repo, "question": question},
        )
        blocks = extract_text_blocks(response)
        if not blocks:
            last_error = MCPError("ask_question returned no textual content.")
            continue
        payload = "\n".join(blocks)
        try:
            return json.loads(_extract_json_from_text(payload))
        except json.JSONDecodeError as exc:
            last_error = exc
            last_payload = payload
    if last_error:
        raise MCPError(
            f"Failed to decode ask_question response after {retries + 1} attempts.\n"
            f"Last payload:\n{last_payload}"
        ) from last_error
    raise MCPError("ask_question failed without an explicit exception.")


def _extract_json_from_text(text: str) -> str:
    stripped = text.strip()
    fenced = None
    if stripped.startswith("```"):
        fence_end = stripped.find("```", 3)
        if fence_end != -1:
            fenced = stripped[3:fence_end]
    if fenced:
        return fenced.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _render_dataset_output(output: PipelineOutput, *, as_json: bool) -> str:
    if as_json:
        return json.dumps(output.to_dict(), ensure_ascii=False, indent=2)
    return output.to_text()


def _write_dataset_output(
    output: PipelineOutput,
    destination: Path,
    *,
    as_json: bool,
    log: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_dataset_output(output, as_json=as_json)
    destination.write_text(rendered, encoding="utf-8")
    if not as_json:
        json_path = destination.with_suffix(".json")
        json_payload = json.dumps(output.to_dict(), ensure_ascii=False, indent=2)
        json_path.write_text(json_payload, encoding="utf-8")
        if log:
            logger.info("Wrote dataset output to %s and companion JSON %s", destination, json_path)
            return
    if log:
        logger.info("Wrote dataset output to %s", destination)


def _write_or_print(text: str, destination: Optional[str]) -> None:
    if destination:
        Path(destination).write_text(text, encoding="utf-8")
        logger.info("Wrote output to %s", destination)
    else:
        print(text)


def _hdfs_join(base: str, *parts: str) -> str:
    cleaned_base = base.rstrip("/")
    cleaned_parts = [part.strip("/").replace("//", "/") for part in parts if part]
    return "/".join([cleaned_base, *cleaned_parts]) if cleaned_parts else cleaned_base


def _run_hdfs(command: List[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise MCPError(f"HDFS command failed: {' '.join(command)}\n{detail}") from exc


def _hdfs_mkdir_p(hdfs_dir: str, *, hdfs_bin: str) -> None:
    _run_hdfs([hdfs_bin, "dfs", "-mkdir", "-p", hdfs_dir])


def _hdfs_test(command: List[str]) -> bool:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _hdfs_exists(hdfs_path: str, *, hdfs_bin: str) -> bool:
    return _hdfs_test([hdfs_bin, "dfs", "-test", "-e", hdfs_path])


def _hdfs_is_dir(hdfs_path: str, *, hdfs_bin: str) -> bool:
    return _hdfs_test([hdfs_bin, "dfs", "-test", "-d", hdfs_path])


def _hdfs_put(local_path: Path, hdfs_dest: str, *, hdfs_bin: str) -> None:
    _run_hdfs([hdfs_bin, "dfs", "-put", "-f", str(local_path), hdfs_dest])


def _hdfs_expected_outputs_exist(
    *,
    target: "RepoTarget",
    output_format: str,
    narrative_modes: Sequence[str],
    narrative_format: str,
    hdfs_output_dir: str,
    hdfs_bin: str,
    existing_paths: Optional[Set[str]] = None,
) -> bool:
    """
    Returns True when the repository output directory already exists on HDFS.

    This is intentionally shallow: it checks only the repo subdirectory existence
    (not specific files inside). This matches the common "skip by folder" behavior
    and avoids false negatives when output file sets differ across runs.
    """
    repo_dir = _hdfs_join(hdfs_output_dir, _repo_slug(target))
    if existing_paths is not None:
        # existing_paths is a shallow listing of direct children under hdfs_output_dir, so
        # membership should be exact and O(1).
        return repo_dir in existing_paths
    return _hdfs_is_dir(repo_dir, hdfs_bin=hdfs_bin)


def _hdfs_list_recursive(hdfs_dir: str, *, hdfs_bin: str) -> Iterable[str]:
    """
    Yield all HDFS file paths under `hdfs_dir` (recursive).
    """
    completed = subprocess.run(
        [hdfs_bin, "dfs", "-ls", "-R", hdfs_dir],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise MCPError(f"HDFS command failed: {hdfs_bin} dfs -ls -R {hdfs_dir}\n{detail}")
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Found "):
            continue
        parts = stripped.split()
        if not parts:
            continue
        yield parts[-1].strip()


def _hdfs_list_shallow(hdfs_dir: str, *, hdfs_bin: str) -> Iterable[str]:
    """
    Yield direct children paths under `hdfs_dir` (non-recursive).
    """
    completed = subprocess.run(
        [hdfs_bin, "dfs", "-ls", hdfs_dir],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise MCPError(f"HDFS command failed: {hdfs_bin} dfs -ls {hdfs_dir}\n{detail}")
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Found "):
            continue
        parts = stripped.split()
        if not parts:
            continue
        yield parts[-1].strip()


def _maybe_upload_repo_outputs_to_hdfs(
    *,
    target: RepoTarget,
    dataset_path: Optional[Path],
    narrative_path: Optional[Path],
    output_format: str,
    narrative_modes: Sequence[str],
    hdfs_output_dir: Optional[str],
    hdfs_bin: str,
) -> None:
    if not hdfs_output_dir:
        return

    slug = _repo_slug(target)
    repo_dir = _hdfs_join(hdfs_output_dir, slug)
    _hdfs_mkdir_p(repo_dir, hdfs_bin=hdfs_bin)

    uploads: List[Tuple[Path, str]] = []
    if dataset_path and dataset_path.exists():
        uploads.append((dataset_path, _hdfs_join(repo_dir, dataset_path.name)))
        if output_format != "json":
            json_path = dataset_path.with_suffix(".json")
            if json_path.exists():
                uploads.append((json_path, _hdfs_join(repo_dir, json_path.name)))

    if narrative_path and narrative_modes and narrative_path.exists():
        uploads.append((narrative_path, _hdfs_join(repo_dir, narrative_path.name)))

    for local_file, dest in uploads:
        _hdfs_put(local_file, dest, hdfs_bin=hdfs_bin)
        logger.info("Uploaded %s -> %s", local_file, dest)


def _parse_server_url_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    urls: List[str] = []
    for raw in values:
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        for part in parts:
            if part and part not in urls:
                urls.append(part)
    return urls or None


def _iter_parquet_rows(
    parquet_dir: Path,
    *,
    columns: Sequence[str],
    batch_size: int = 1024,
) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MCPError(
            "pyarrow is required for --parquet-input-dir. Install it with `pip install pyarrow`."
        ) from exc

    part_files = sorted(
        path for path in parquet_dir.iterdir() if path.is_file() and path.name.startswith("part-")
    )
    if not part_files:
        raise MCPError(f"No part-* parquet files found under {parquet_dir}")

    for part in part_files:
        parquet_file = pq.ParquetFile(part)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
            table = record_batch.to_pydict()
            if not table:
                continue
            keys = list(table.keys())
            row_count = len(table[keys[0]]) if keys else 0
            for idx in range(row_count):
                yield {key: table[key][idx] for key in keys}


def _load_parquet_rows_for_repos(
    parquet_dir: Path,
    *,
    repo_names: Sequence[str],
    columns: Sequence[str],
    batch_size: int = 1024,
) -> Dict[str, Dict[str, Any]]:
    """
    Load specific repo rows from a DeepWiki parquet directory.

    This function is optimized for large datasets: it only materializes `content`/`hdfs_path`
    values for matched repos (avoids converting full batches to Python dicts).
    """
    desired = {name.strip() for name in repo_names if name and name.strip()}
    if not desired:
        return {}

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MCPError(
            "pyarrow is required for --parquet-input-dir. Install it with `pip install pyarrow`."
        ) from exc

    requested_columns = list(dict.fromkeys(["repo_name", *columns]))
    part_files = sorted(
        path for path in parquet_dir.iterdir() if path.is_file() and path.name.startswith("part-")
    )
    if not part_files:
        raise MCPError(f"No part-* parquet files found under {parquet_dir}")

    found: Dict[str, Dict[str, Any]] = {}
    for part in part_files:
        parquet_file = pq.ParquetFile(part)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=requested_columns):
            try:
                repo_idx = record_batch.schema.get_field_index("repo_name")
            except Exception:
                repo_idx = -1
            if repo_idx < 0:
                raise MCPError("Parquet dataset is missing required column: repo_name")
            repo_values = record_batch.column(repo_idx).to_pylist()
            if not repo_values:
                continue
            for row_idx, repo_name in enumerate(repo_values):
                repo_key = str(repo_name or "").strip()
                if not repo_key or repo_key not in desired or repo_key in found:
                    continue
                row: Dict[str, Any] = {"repo_name": repo_key}
                for col_name in columns:
                    col_idx = record_batch.schema.get_field_index(col_name)
                    if col_idx < 0:
                        row[col_name] = None
                        continue
                    row[col_name] = record_batch.column(col_idx)[row_idx].as_py()
                found[repo_key] = row
                if len(found) >= len(desired):
                    return found
    return found


def _offline_outline_and_markdown_from_parquet_content(content: str) -> Tuple[str, str]:
    """
    Convert Parquet `content` (a JSON list of page objects) into:
    - outline markdown compatible with `parse_outline_text`
    - wiki markdown compatible with `parse_wiki_markdown`
    """
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MCPError("Parquet content field is not valid JSON.") from exc
    if not isinstance(payload, list) or not payload:
        raise MCPError("Parquet content field is empty or not a JSON list.")

    outline_lines: List[str] = []
    pages_parts: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        plan = item.get("page_plan") or {}
        if not isinstance(plan, dict):
            plan = {}
        title = str(plan.get("title") or "").strip()
        number = str(plan.get("id") or "").strip()
        if not title:
            continue
        outline_title = f"{number} {title}".strip()
        outline_lines.append(f"- {outline_title}")

        page_text = str(item.get("content") or "").strip()
        pages_parts.append(f"# Page: {title}\n\n{page_text}\n")

    outline_text = "\n".join(outline_lines).strip()
    wiki_markdown = "\n\n".join(pages_parts).strip()
    if not outline_text or not wiki_markdown:
        raise MCPError("Failed to construct offline outline/wiki markdown from parquet content.")
    return outline_text, wiki_markdown


def _discover_targets_from_parquet(
    parquet_dir: Path,
    *,
    max_repos: Optional[int],
    repo_prefix: Optional[str],
    batch_size: int = 1024,
) -> Tuple[List[RepoTarget], Dict[str, Dict[str, Any]]]:
    """
    Discover repo targets directly from the parquet dataset (no repos.txt needed).

    Safety: callers should set either `max_repos` or require an explicit `--parquet-all`.
    """
    desired_prefix = repo_prefix.strip() if repo_prefix else None
    columns = ["repo_name", "content", "hdfs_path", "error_message"]
    targets: List[RepoTarget] = []
    parquet_index: Dict[str, Dict[str, Any]] = {}

    for row in _iter_parquet_rows(parquet_dir, columns=columns, batch_size=batch_size):
        repo_name = str(row.get("repo_name") or "").strip()
        if not repo_name:
            continue
        if desired_prefix and not repo_name.startswith(desired_prefix):
            continue
        if repo_name in parquet_index:
            continue
        if row.get("error_message"):
            continue
        content = str(row.get("content") or "")
        hdfs_path = str(row.get("hdfs_path") or "").strip()
        if not content or not hdfs_path:
            continue
        parquet_index[repo_name] = {
            "repo_name": repo_name,
            "content": content,
            "hdfs_path": hdfs_path,
        }
        targets.append(RepoTarget(repo=repo_name))
        if max_repos is not None and len(targets) >= max_repos:
            break

    return targets, parquet_index


def _iter_parquet_rows_from_parts(
    parquet_dir: Path,
    *,
    columns: Sequence[str],
    batch_size: int,
    part_files: Optional[Sequence[Path]] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Stream parquet rows part-by-part with a controlled batch size.
    Uses pyarrow record batches but converts each row to a small dict.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MCPError(
            "pyarrow is required for --parquet-input-dir. Install it with `pip install pyarrow`."
        ) from exc

    if part_files is None:
        part_files = sorted(
            path for path in parquet_dir.iterdir() if path.is_file() and path.name.startswith("part-")
        )
    else:
        part_files = list(part_files)
    if not part_files:
        raise MCPError(f"No part-* parquet files found under {parquet_dir}")

    cols = list(columns)
    for part in part_files:
        parquet_file = pq.ParquetFile(part)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols):
            # Avoid record_batch.to_pydict() (can be very expensive for large string columns).
            # We only materialize per-column lists and stitch per-row dicts.
            arrays = [record_batch.column(i).to_pylist() for i in range(record_batch.num_columns)]
            if not arrays:
                continue
            row_count = len(arrays[0])
            for idx in range(row_count):
                yield {cols[col_idx]: arrays[col_idx][idx] for col_idx in range(len(cols))}


def _run_parquet_stream(
    *,
    parquet_dir: Path,
    args: argparse.Namespace,
    design_config: Optional[NarrativeLLMConfig],
    judge_config: Optional[JudgeLLMConfig],
) -> None:
    """
    Process repos directly from parquet without building a full repo list in memory.

    Intended for very large datasets (e.g. ~40w repos) with bounded memory and limited HDFS reads.
    Concurrency is controlled by --repo-workers.
    """
    if args.repo_mp_workers and mp is None:  # pragma: no cover
        raise MCPError("Multiprocessing is unavailable in this environment.")

    dataset_dir = Path(args.output_dir or "result_data").expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    narrative_dir: Optional[Path] = None
    narrative_modes: List[str] = []
    merged_narrative_path: Optional[Path] = None
    if args.narrative_output_dir:
        narrative_dir = Path(args.narrative_output_dir).expanduser()
        narrative_dir.mkdir(parents=True, exist_ok=True)
        narrative_modes = _normalize_narrative_modes(args.narrative_modes)
        if args.narrative_output:
            merged_narrative_path = Path(args.narrative_output).expanduser()
            merged_narrative_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: In parquet discovery mode, --parquet-max-repos is interpreted as
    # "process the first N rows discovered from parquet" (regardless of skipping).
    # This keeps the progress bar stable and makes limiting predictable.
    max_repos = None if args.parquet_all else args.parquet_max_repos
    prefix = args.parquet_repo_prefix.strip() if args.parquet_repo_prefix else None
    scan_batch_size = args.parquet_scan_batch_size or 64
    should_skip_existing = bool(args.skip_existing)
    existing_hdfs_paths: Optional[Set[str]] = None
    if should_skip_existing and args.hdfs_output_dir:
        logger.info("Indexing existing HDFS output directories under %s (for --skip-existing).", args.hdfs_output_dir)
        existing_hdfs_paths = set(_hdfs_list_shallow(args.hdfs_output_dir, hdfs_bin=args.hdfs_bin))
        logger.info("Indexed %d existing HDFS paths.", len(existing_hdfs_paths))
        try:
            index_dump_path = dataset_dir / "hdfs_skip_existing_index.txt"
            index_dump_path.write_text(
                "\n".join(sorted(existing_hdfs_paths)) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote HDFS skip-existing index to %s", index_dump_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write HDFS skip-existing index: %s", exc)

    cpu_default = os.cpu_count() or 4
    repo_workers = args.repo_workers or max(1, cpu_default)
    if repo_workers < 1:
        raise MCPError("--repo-workers must be >= 1.")

    total = max_repos if max_repos is not None else _parquet_total_rows(parquet_dir)
    progress = tqdm(total=total, desc="Repositories", unit="repo", leave=True) if tqdm else None

    stats = _ParquetStreamStats()
    errors: List[Tuple[str, Exception]] = []
    counter_lock = threading.Lock()

    def log_skip_stats(*, force: bool = False) -> None:
        if args.progress_only:
            return
        interval = 2000
        if not force and stats.processed and stats.processed % interval != 0:
            return
        logger.info(
            "Skip stats: processed=%d generated=%d skipped_hdfs=%d skipped_local=%d errors=%d",
            stats.processed,
            stats.generated,
            stats.skipped_by_hdfs,
            stats.skipped_by_local,
            len(errors),
        )

    def should_process_repo(repo_name: str) -> bool:
        if not repo_name:
            return False
        if prefix and not repo_name.startswith(prefix):
            return False
        return True

    def process_repo(repo_name: str, content: str, hdfs_path: str) -> str:
        target = RepoTarget(repo=repo_name)
        dataset_path = dataset_dir / _dataset_filename(target, as_json=args.output_format == "json")
        narrative_path = None
        if narrative_dir:
            narrative_path = narrative_dir / _narrative_filename(target, fmt=args.narrative_format)

        if should_skip_existing:
            if args.hdfs_output_dir and _hdfs_expected_outputs_exist(
                target=target,
                output_format=args.output_format,
                narrative_modes=narrative_modes,
                narrative_format=args.narrative_format,
                hdfs_output_dir=args.hdfs_output_dir,
                hdfs_bin=args.hdfs_bin,
                existing_paths=existing_hdfs_paths,
            ):
                return "skipped_hdfs"
            # Only fall back to local existence checks when HDFS skipping is not configured.
            if not args.hdfs_output_dir:
                if dataset_path.exists() and (narrative_path is None or narrative_path.exists()):
                    return "skipped_local"

        _execute_pipeline_for_target(
            session=None,
            target=target,
            args=args,
            design_config=design_config,
            judge_config=judge_config,
            dataset_path=dataset_path,
            narrative_path=narrative_path,
            narrative_modes=narrative_modes,
            print_to_stdout=False,
            parquet_row={"repo_name": repo_name, "content": content, "hdfs_path": hdfs_path},
        )
        return "generated"

    part_files = sorted(
        path for path in parquet_dir.iterdir() if path.is_file() and path.name.startswith("part-")
    )
    if not part_files:
        raise MCPError(f"No part-* parquet files found under {parquet_dir}")

    def iter_rows_for_worker(*, part_subset: Sequence[Path]) -> Iterable[Dict[str, Any]]:
        return _iter_parquet_rows_from_parts(
            parquet_dir,
            columns=["repo_name", "content", "hdfs_path", "error_message"],
            batch_size=scan_batch_size,
            part_files=part_subset,
        )

    def run_worker_stream(
        *,
        part_subset: Sequence[Path],
        progress_q: Optional[Any] = None,
        shared_limit_counter: Optional[Any] = None,
        shared_limit_lock: Optional[Any] = None,
    ) -> None:
        for row in iter_rows_for_worker(part_subset=part_subset):
            if max_repos is not None and shared_limit_counter is not None and shared_limit_lock is not None:
                with shared_limit_lock:
                    if shared_limit_counter.value >= max_repos:
                        break
                    shared_limit_counter.value += 1
            repo_name = str(row.get("repo_name") or "").strip()
            # Each row contributes 1 progress tick regardless of skip/generate.
            if not should_process_repo(repo_name):
                if progress_q is not None:
                    progress_q.put(("tick", 1, 0, 0, 0, 0))
                continue
            if row.get("error_message"):
                if progress_q is not None:
                    progress_q.put(("tick", 1, 0, 0, 0, 0))
                continue
            content = str(row.get("content") or "")
            hdfs_path = str(row.get("hdfs_path") or "").strip()
            if not content or not hdfs_path:
                if progress_q is not None:
                    progress_q.put(("tick", 1, 0, 0, 0, 0))
                continue
            try:
                status = process_repo(repo_name, content, hdfs_path)
                if progress_q is not None:
                    if status == "skipped_hdfs":
                        progress_q.put(("tick", 1, 0, 1, 0, 0))
                    elif status == "skipped_local":
                        progress_q.put(("tick", 1, 0, 0, 1, 0))
                    else:
                        progress_q.put(("tick", 1, 1, 0, 0, 0))
            except Exception as exc:
                if progress_q is not None:
                    progress_q.put(("err", repo_name, repr(exc)))
                    progress_q.put(("tick", 1, 0, 0, 0, 1))
                else:
                    raise

    if args.repo_mp_workers and args.repo_mp_workers > 1:
        # Multiprocessing mode: shard parquet part-* files across processes.
        # Each process runs sequentially through its assigned parts, avoiding nested thread explosions.
        if mp is None:  # pragma: no cover
            raise MCPError("Multiprocessing is unavailable in this environment.")
        mp_workers = int(args.repo_mp_workers)
        if mp_workers < 1:
            raise MCPError("--repo-mp-workers must be >= 1.")
        if args.repo_workers and args.repo_workers > 1:
            logger.warning(
                "Both --repo-mp-workers and --repo-workers set; in mp mode repos are processed sequentially per process. "
                "Set --repo-workers=1 to avoid nested threading."
            )
        shards: List[List[Path]] = [[] for _ in range(mp_workers)]
        for idx, part in enumerate(part_files):
            shards[idx % mp_workers].append(part)

        start_methods = set(mp.get_all_start_methods()) if hasattr(mp, "get_all_start_methods") else set()
        # Prefer fork on Linux for performance and to avoid pickling nested callables.
        ctx = mp.get_context("fork") if "fork" in start_methods else mp.get_context("spawn")
        progress_q = ctx.Queue(maxsize=max(1000, mp_workers * 10))
        procs: List[Any] = []
        shared_limit_counter = ctx.Value("i", 0) if max_repos is not None else None
        shared_limit_lock = ctx.Lock() if max_repos is not None else None

        def _proc_entry(parts: List[Path]) -> None:
            try:
                run_worker_stream(
                    part_subset=parts,
                    progress_q=progress_q,
                    shared_limit_counter=shared_limit_counter,
                    shared_limit_lock=shared_limit_lock,
                )
            finally:
                progress_q.put(("done",))

        for shard_parts in shards:
            p = ctx.Process(target=_proc_entry, args=(shard_parts,))
            p.start()
            procs.append(p)

        done = 0
        while done < len(procs):
            msg = progress_q.get()
            if not msg:
                continue
            kind = msg[0]
            if kind == "tick":
                _, inc_processed, inc_generated, inc_skipped_hdfs, inc_skipped_local, inc_errors = msg
                stats.processed += int(inc_processed)
                stats.generated += int(inc_generated)
                stats.skipped_by_hdfs += int(inc_skipped_hdfs)
                stats.skipped_by_local += int(inc_skipped_local)
                stats.errors += int(inc_errors)
                if progress is not None:
                    progress.update(int(inc_processed))
                log_skip_stats()
            elif kind == "err":
                _, repo_name, err = msg
                logger.error("Repository %s failed: %s", repo_name, err)
                errors.append((str(repo_name), RuntimeError(str(err))))
            elif kind == "done":
                done += 1
            else:
                logger.debug("Unknown progress message: %s", msg)

        for p in procs:
            p.join()
    else:
        # Threaded mode (legacy).
        with ThreadPoolExecutor(max_workers=repo_workers) as executor:
            in_flight: Dict[Any, str] = {}
            in_flight_started_at: Dict[Any, float] = {}
            for row in _iter_parquet_rows_from_parts(
                parquet_dir,
                columns=["repo_name", "content", "hdfs_path", "error_message"],
                batch_size=scan_batch_size,
                part_files=part_files,
            ):
                if max_repos is not None and stats.processed >= max_repos:
                    break
                stats.processed += 1
                if progress is not None:
                    progress.update(1)
                repo_name = str(row.get("repo_name") or "").strip()
                if not should_process_repo(repo_name):
                    continue
                if row.get("error_message"):
                    continue
                content = str(row.get("content") or "")
                hdfs_path = str(row.get("hdfs_path") or "").strip()
                if not content or not hdfs_path:
                    continue

                while len(in_flight) >= repo_workers:
                    try:
                        future = next(as_completed(list(in_flight.keys()), timeout=30.0))
                    except TimeoutError:
                        if not args.progress_only and in_flight_started_at:
                            now = time.time()
                            stuck = sorted(
                                (
                                    (now - in_flight_started_at.get(f, now), in_flight.get(f, "<unknown>"))
                                    for f in in_flight
                                ),
                                reverse=True,
                            )[:5]
                            logger.warning(
                                "No repo finished in 30s; oldest in-flight (seconds, repo): %s",
                                ", ".join(f"{age:.0f}s {name}" for age, name in stuck),
                            )
                        continue

                    finished_repo = in_flight.pop(future)
                    in_flight_started_at.pop(future, None)
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("Repository %s failed: %s", finished_repo, exc)
                        errors.append((finished_repo, exc))
                    log_skip_stats()
                    if progress is not None:
                        # progress is driven by scan count, already updated per row.
                        pass

                def _thread_worker() -> None:
                    status = process_repo(repo_name, content, hdfs_path)
                    with counter_lock:
                        if status == "skipped_hdfs":
                            stats.skipped_by_hdfs += 1
                        elif status == "skipped_local":
                            stats.skipped_by_local += 1
                        else:
                            stats.generated += 1

                future = executor.submit(_thread_worker)
                in_flight[future] = repo_name
                in_flight_started_at[future] = time.time()

                if max_repos is not None and stats.processed >= max_repos:
                    break

            for future in as_completed(in_flight):
                finished_repo = in_flight[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Repository %s failed: %s", finished_repo, exc)
                    errors.append((finished_repo, exc))
                log_skip_stats()

    if progress is not None:
        progress.close()
    log_skip_stats(force=True)

    if merged_narrative_path and narrative_dir and narrative_modes and not errors:
        _merge_narrative_outputs(
            narrative_dir=narrative_dir,
            destination=merged_narrative_path,
            fmt=args.narrative_format,
        )
        logger.info("Wrote merged narrative output to %s", merged_narrative_path)

    if errors:
        failed = ", ".join(name for name, _ in errors[:10])
        message = f"Failed to process repositories (showing up to 10): {failed}"
        if getattr(args, "continue_on_error", False):
            logger.warning("%s", message)
            return
        raise MCPError(message)


def _download_and_extract_repo_zip_cached(
    *,
    hdfs_zip_path: str,
    extract_dir: Path,
    hdfs_bin: str,
) -> Path:
    """
    Download the zip from HDFS (hdfs://...) and extract it to `extract_dir`.
    Uses a simple marker file to avoid re-downloading/re-extracting.
    """
    import shutil
    import tempfile
    import zipfile

    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        children = [p for p in extract_dir.iterdir() if p.name != marker.name]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return extract_dir

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="deepwiki_repozip_"))
    local_zip = tmp_dir / "repo.zip"
    try:
        # Some Hadoop distributions do not support `-get -f` (overwrite).
        # We download into a unique temp directory, so overwriting is unnecessary.
        _run_hdfs([hdfs_bin, "dfs", "-get", hdfs_zip_path, str(local_zip)])
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    children = [p for p in extract_dir.iterdir() if p.name != marker.name]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def _normalize_narrative_modes(modes: Optional[List[str]]) -> List[str]:
    if not modes:
        return ["code"]
    normalized: List[str] = []
    for mode in modes:
        key = mode.lower()
        if key in {"code", "rewrite"}:
            normalized.append("code")
        elif key in {"critic", "critique", "cross"}:
            normalized.append("critic")
        else:
            logger.warning("Ignoring unrecognised narrative mode '%s'.", mode)
    if not normalized:
        normalized.append("code")
    # Preserve order but de-duplicate
    seen = set()
    result = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _collect_narrative_records(output: PipelineOutput) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for subsection in output.subsections:
        record = {
            "repo": output.repo,
            "page": subsection.page_title,
            "section": subsection.section_heading,
            "original_context": subsection.original_context,
            "narrative": subsection.narrative,
            "critic": subsection.critic,
            "verdict": subsection.verdict,
            "misalignment": subsection.misalignment.value if subsection.misalignment else None,
            "learnability": subsection.learnability,
            "critic_history": subsection.critic_history,
            "code_blocks": [
                {
                    "reference": block.reference,
                    "code": block.code,
                }
                for block in subsection.code_blocks
            ],
        }
        records.append(record)
    return records


def _render_narratives_text(records: List[Dict[str, Any]], modes: List[str]) -> str:
    parts: List[str] = []
    for record in records:
        lines = [
            f"# {record['page']} :: {record['section']}",
        ]
        lines.append("")
        lines.append("Narrative:")
        lines.append(record["narrative"])
        if "critic" in modes and record.get("critic"):
            lines.append("")
            lines.append("Critic:")
            lines.append(record["critic"])
        lines.append("")
        lines.append("Metadata:")
        verdict = record.get("verdict", "UNKNOWN")
        misalignment = record.get("misalignment") or "none"
        learnability = record.get("learnability")
        lines.append(
            f"verdict={verdict} misalignment={misalignment} learnability={learnability}"
        )
        parts.append("\n".join(lines).rstrip())
    return "\n\n".join(parts).strip()


def _write_narrative_output(
    output: PipelineOutput,
    *,
    modes: List[str],
    fmt: str,
    destination: Path,
    log: bool = True,
) -> None:
    records = _collect_narrative_records(output)
    if not records:
        if log:
            logger.info("No narrative records to write.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = json.dumps(records, ensure_ascii=False, indent=2)
    else:
        payload = _render_narratives_text(records, modes)
    destination.write_text(payload, encoding="utf-8")
    if log:
        logger.info("Wrote narrative output to %s", destination)


def _expand_repo_token(token: str) -> List[str]:
    value = token.strip()
    if not value:
        return []
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.exists():
            raise MCPError(f"Repository list file not found: {path}")
        entries: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line_value = line.strip()
            if not line_value or line_value.startswith("#"):
                continue
            entries.extend(_expand_repo_token(line_value))
        return entries
    if "," in value:
        entries: List[str] = []
        for part in value.split(","):
            entries.extend(_expand_repo_token(part))
        return entries
    return [value]


def _parse_repo_target(raw: str, default_commit: Optional[str]) -> RepoTarget:
    cleaned = raw.strip()
    if not cleaned:
        raise MCPError("Encountered empty repository identifier.")
    cleaned = cleaned.rstrip("@").strip()
    if cleaned.startswith("hdfs://") and cleaned.endswith(".zip"):
        repo, commit = _repo_from_hdfs_zip_path(cleaned)
        return RepoTarget(repo=repo, commit=commit or default_commit, hdfs_zip_path=cleaned)
    repo = cleaned
    commit = default_commit
    if "@" in cleaned:
        repo_part, commit_part = cleaned.split("@", 1)
        repo = repo_part.strip()
        commit_candidate = commit_part.strip()
        if commit_candidate:
            commit = commit_candidate
    if not repo:
        raise MCPError(f"Invalid repository specification: '{raw}'")
    return RepoTarget(repo=repo, commit=commit)


def _repo_from_hdfs_zip_path(hdfs_zip_path: str) -> Tuple[str, Optional[str]]:
    """
    Parse `hdfs://.../<owner>_<repo>_<suffix>.zip` into `owner/repo`.

    This is a best-effort heuristic for DeepWiki repo zips, where the last underscore
    segment is usually a short commit/hash. Repository names containing underscores
    are supported because we only treat the final segment as the suffix.
    """
    basename = hdfs_zip_path.rsplit("/", 1)[-1]
    if basename.lower().endswith(".zip"):
        basename = basename[:-4]
    parts = [p for p in basename.split("_") if p]
    if len(parts) < 2:
        raise MCPError(f"Cannot infer repo name from hdfs zip path: {hdfs_zip_path}")
    owner = parts[0]
    commit = None
    if len(parts) >= 3:
        commit = parts[-1]
        repo_name = "_".join(parts[1:-1])
    else:
        repo_name = parts[1]
    if not owner or not repo_name:
        raise MCPError(f"Cannot infer repo name from hdfs zip path: {hdfs_zip_path}")
    return f"{owner}/{repo_name}", commit or None


def _repo_slug(target: RepoTarget) -> str:
    slug = target.repo.replace("/", "_").replace(" ", "_")
    if target.commit:
        commit_fragment = "".join(ch if ch.isalnum() else "-" for ch in target.commit[:12])
        slug = f"{slug}_{commit_fragment}"
    return slug


def _dataset_filename(target: RepoTarget, *, as_json: bool) -> str:
    slug = _repo_slug(target)
    suffix = "json" if as_json else "txt"
    return f"{slug}_deepwiki.{suffix}"


def _narrative_filename(target: RepoTarget, *, fmt: str) -> str:
    slug = _repo_slug(target)
    suffix = "json" if fmt == "json" else "txt"
    return f"{slug}_narratives.{suffix}"


def _resolve_repo_targets(args: argparse.Namespace) -> List[RepoTarget]:
    if args.generate_dataset is None:
        return []
    tokens = args.generate_dataset
    if isinstance(tokens, str):  # pragma: no cover - legacy fallback
        tokens = [tokens]
    expanded: List[str] = []
    for token in tokens:
        expanded.extend(_expand_repo_token(token))
    default_commit = args.repo_commit
    targets = [_parse_repo_target(item, default_commit) for item in expanded]
    return targets


def _persist_outputs(
    output: PipelineOutput,
    *,
    dataset_path: Optional[Path],
    narrative_path: Optional[Path],
    output_format: str,
    narrative_modes: Sequence[str],
    narrative_format: str,
    log_writes: bool,
) -> None:
    if dataset_path:
        _write_dataset_output(
            output,
            dataset_path,
            as_json=output_format == "json",
            log=log_writes,
        )
    if narrative_path and narrative_modes:
        _write_narrative_output(
            output,
            modes=list(narrative_modes),
            fmt=narrative_format,
            destination=narrative_path,
            log=log_writes,
        )


def _execute_pipeline_for_target(
    *,
    session: Optional[Session],
    target: RepoTarget,
    args: argparse.Namespace,
    design_config: Optional[NarrativeLLMConfig],
    judge_config: Optional[JudgeLLMConfig],
    dataset_path: Optional[Path],
    narrative_path: Optional[Path],
    narrative_modes: Sequence[str],
    print_to_stdout: bool,
    parquet_row: Optional[Dict[str, Any]] = None,
) -> PipelineOutput:
    progress_needed = dataset_path is not None or (narrative_path is not None and narrative_modes)
    cache_cleanup_dir: Optional[Path] = None
    finished_ok = False
    repo_label = target.repo

    def persist_progress(partial_output: PipelineOutput) -> None:
        _persist_outputs(
            partial_output,
            dataset_path=dataset_path,
            narrative_path=narrative_path,
            output_format=args.output_format,
            narrative_modes=narrative_modes,
            narrative_format=args.narrative_format,
            log_writes=False,
        )

    try:
        start_time = time.time()
        outline_text = None
        wiki_markdown = None
        repo_root = None

        if args.parquet_input_dir:
            parquet_stage_start = time.time()
            parquet_dir = Path(args.parquet_input_dir).expanduser()
            desired_repo = target.repo
            found_row = parquet_row
            if found_row is None:
                matches = _load_parquet_rows_for_repos(
                    parquet_dir,
                    repo_names=[desired_repo],
                    columns=["content", "hdfs_path"],
                    batch_size=args.parquet_scan_batch_size or 1024,
                )
                found_row = matches.get(desired_repo)
            if found_row is None:
                raise MCPError(f"Repo '{desired_repo}' not found under parquet directory {parquet_dir}")

            content = str(found_row.get("content") or "")
            outline_text, wiki_markdown = _offline_outline_and_markdown_from_parquet_content(content)

            hdfs_zip_path = str(found_row.get("hdfs_path") or "").strip()
            if not hdfs_zip_path:
                raise MCPError(f"Missing hdfs_path for repo {desired_repo} in parquet dataset.")

            cache_base = Path(args.repo_cache_dir or "/tmp/deepwiki_repo_cache").expanduser()
            repo_extract_dir = cache_base / _repo_slug(target)
            cache_cleanup_dir = repo_extract_dir
            logger.info("Repo %s: fetching/extracting zip from %s", repo_label, hdfs_zip_path)
            repo_root = _download_and_extract_repo_zip_cached(
                hdfs_zip_path=hdfs_zip_path,
                extract_dir=repo_extract_dir,
                hdfs_bin=args.hdfs_bin,
            )
            logger.info(
                "Repo %s: parquet+zip stage done in %.1fs (repo_root=%s)",
                repo_label,
                time.time() - parquet_stage_start,
                repo_root,
            )

        pipeline_stage_start = time.time()
        pipeline = DeepWikiPipeline(
            session=session,
            repo=target.repo,
            logic_llm_config=design_config,
            critic_llm_config=judge_config,
            repo_commit=None if args.parquet_input_dir else target.commit,
            allow_git_clone=not bool(args.parquet_input_dir),
            show_page_progress=False,
            enable_hydration=not bool(args.disable_hydration),
            hydration_timeout=args.hydration_timeout,
            hydration_workers=args.hydration_workers,
            judge_rounds=args.judge_max_rounds,
            repo_root=repo_root,
            max_pages=args.max_pages,
            max_sections_per_page=args.max_sections_per_page,
            max_workers=args.max_workers,
            section_workers=args.section_workers,
            skip_pages=args.skip_page,
        )
        output = pipeline.run(
            progress_callback=persist_progress if progress_needed else None,
            outline_text=outline_text,
            wiki_markdown=wiki_markdown,
        )
        logger.info("Repo %s: pipeline.run done in %.1fs", repo_label, time.time() - pipeline_stage_start)

        persist_stage_start = time.time()
        _persist_outputs(
            output,
            dataset_path=dataset_path,
            narrative_path=narrative_path,
            output_format=args.output_format,
            narrative_modes=narrative_modes,
            narrative_format=args.narrative_format,
            log_writes=True,
        )
        logger.info("Repo %s: persist outputs done in %.1fs", repo_label, time.time() - persist_stage_start)
        upload_stage_start = time.time()
        _maybe_upload_repo_outputs_to_hdfs(
            target=target,
            dataset_path=dataset_path,
            narrative_path=narrative_path,
            output_format=args.output_format,
            narrative_modes=narrative_modes,
            hdfs_output_dir=args.hdfs_output_dir,
            hdfs_bin=args.hdfs_bin,
        )
        if args.hdfs_output_dir:
            logger.info("Repo %s: HDFS upload stage done in %.1fs", repo_label, time.time() - upload_stage_start)
        logger.info("Repo %s: total done in %.1fs", repo_label, time.time() - start_time)

        if dataset_path is None and print_to_stdout:
            rendered = _render_dataset_output(output, as_json=args.output_format == "json")
            _write_or_print(rendered, None)

        finished_ok = True
        return output
    finally:
        if not args.parquet_input_dir:
            return
        if not cache_cleanup_dir:
            return
        cleanup_mode = getattr(args, "repo_cache_cleanup", "never")
        should_cleanup = cleanup_mode == "always" or (cleanup_mode == "on-success" and finished_ok)
        if should_cleanup:
            shutil.rmtree(cache_cleanup_dir, ignore_errors=True)


def _run_multi_repo_batch(
    *,
    targets: Sequence[RepoTarget],
    args: argparse.Namespace,
    design_config: Optional[NarrativeLLMConfig],
    judge_config: Optional[JudgeLLMConfig],
    parquet_index_override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if not targets:
        return
    dataset_dir = Path(args.output_dir or "result_data").expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    narrative_dir: Optional[Path] = None
    narrative_modes: List[str] = []
    if args.narrative_output_dir:
        narrative_dir = Path(args.narrative_output_dir).expanduser()
        narrative_dir.mkdir(parents=True, exist_ok=True)
        narrative_modes = _normalize_narrative_modes(args.narrative_modes)
        if args.narrative_output:
            merged_narrative_path = Path(args.narrative_output).expanduser()
            merged_narrative_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            merged_narrative_path = None

    batch_size = args.repo_batch_size or len(targets)
    if batch_size < 1:
        raise MCPError("--repo-batch-size must be >= 1.")

    cpu_default = os.cpu_count() or 4
    default_workers = min(len(targets), max(1, cpu_default))
    repo_workers = args.repo_workers or default_workers
    if repo_workers < 1:
        raise MCPError("--repo-workers must be >= 1.")

    errors: List[Tuple[RepoTarget, Exception]] = []
    parquet_dir: Optional[Path] = None
    parquet_index: Dict[str, Dict[str, Any]] = parquet_index_override or {}
    if args.parquet_input_dir and not parquet_index_override:
        parquet_dir = Path(args.parquet_input_dir).expanduser()
        desired_repos = sorted({target.repo for target in targets})
        logger.info(
            "Loading parquet rows for %d requested repos from %s",
            len(desired_repos),
            parquet_dir,
        )
        parquet_index = _load_parquet_rows_for_repos(
            parquet_dir,
            repo_names=desired_repos,
            columns=["content", "hdfs_path"],
            batch_size=args.parquet_scan_batch_size or 1024,
        )
        logger.info(
            "Loaded %d/%d requested repos from parquet dataset.",
            len(parquet_index),
            len(desired_repos),
        )

    total_batches = (len(targets) + batch_size - 1) // batch_size
    overall_bar = (
        tqdm(total=len(targets), desc="Repositories", unit="repo", leave=False)
        if tqdm and targets
        else None
    )
    processed_repos = 0

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        batch_targets = targets[start : start + batch_size]
        if not batch_targets:
            continue
        logger.info(
            "Processing repository batch %d/%d (%d repositories).",
            batch_index + 1,
            total_batches,
            len(batch_targets),
        )

        def worker(target: RepoTarget) -> None:
            session = None if args.parquet_input_dir else initialize_session()
            try:
                dataset_path = dataset_dir / _dataset_filename(target, as_json=args.output_format == "json")
                narrative_path = None
                if narrative_dir:
                    narrative_path = narrative_dir / _narrative_filename(target, fmt=args.narrative_format)
                _execute_pipeline_for_target(
                    session=session,
                    target=target,
                    args=args,
                    design_config=design_config,
                    judge_config=judge_config,
                    dataset_path=dataset_path,
                    narrative_path=narrative_path,
                    narrative_modes=narrative_modes,
                    print_to_stdout=False,
                    parquet_row=parquet_index.get(target.repo),
                )
            finally:
                if session is not None:
                    delete_session(session)

        max_workers = min(repo_workers, len(batch_targets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(worker, target): target for target in batch_targets}
            for future in as_completed(future_map):
                target = future_map[future]
                try:
                    future.result()
                    processed_repos += 1
                    if overall_bar:
                        overall_bar.update(1)
                    else:
                        logger.info(
                            "Completed %d/%d repositories (%s)",
                            processed_repos,
                            len(targets),
                            target.repo,
                        )
                except Exception as exc:
                    logger.error("Repository %s failed: %s", target.repo, exc)
                    errors.append((target, exc))

    if overall_bar:
        overall_bar.close()

    if merged_narrative_path and narrative_dir and narrative_modes and not errors:
        _merge_narrative_outputs(
            narrative_dir=narrative_dir,
            destination=merged_narrative_path,
            fmt=args.narrative_format,
        )
        logger.info("Wrote merged narrative output to %s", merged_narrative_path)

    if errors:
        failed_repos = ", ".join(item.repo for item, _ in errors)
        message = f"Failed to process repositories: {failed_repos}"
        if getattr(args, "continue_on_error", False):
            logger.warning("%s", message)
            return
        primary = errors[0][1]
        raise MCPError(message) from primary


def _merge_narrative_outputs(*, narrative_dir: Path, destination: Path, fmt: str) -> None:
    if fmt == "json":
        candidates = sorted(narrative_dir.glob("*_narratives.jsonl"))
        if not candidates:
            raise MCPError(f"No narrative outputs found under {narrative_dir}")
        with destination.open("w", encoding="utf-8") as out:
            for path in candidates:
                text = path.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    text += "\n"
                out.write(text)
        return

    # text format
    candidates = sorted(narrative_dir.glob("*_narratives.txt"))
    if not candidates:
        raise MCPError(f"No narrative outputs found under {narrative_dir}")
    parts: List[str] = []
    for path in candidates:
        parts.append(path.read_text(encoding="utf-8").rstrip())
    destination.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")


def _parquet_total_rows(parquet_dir: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MCPError(
            "pyarrow is required for --parquet-input-dir. Install it with `pip install pyarrow`."
        ) from exc
    total = 0
    for part in sorted(path for path in parquet_dir.iterdir() if path.is_file() and path.name.startswith("part-")):
        parquet_file = pq.ParquetFile(part)
        total += int(parquet_file.metadata.num_rows)
    return total


def _build_design_llm_config(args: argparse.Namespace) -> NarrativeLLMConfig:
    max_tokens = args.design_vllm_max_tokens if args.design_vllm_max_tokens and args.design_vllm_max_tokens > 0 else None
    top_p = args.design_vllm_top_p or None
    server_pool = _parse_server_url_list(args.design_vllm_server_urls)
    return NarrativeLLMConfig(
        server_url=args.design_vllm_server_url,
        server_urls=server_pool,
        host=args.design_vllm_host,
        port=args.design_vllm_port,
        path=args.design_vllm_path,
        model=args.design_vllm_model,
        temperature=args.design_vllm_temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=args.design_vllm_api_key,
        destination_service=args.design_vllm_destination_service,
        timeout=args.design_vllm_timeout,
        retries=args.design_vllm_retries,
        retry_backoff=args.design_vllm_retry_backoff,
    )


def _build_judge_llm_config(args: argparse.Namespace, *, system_prompt: Optional[str]) -> JudgeLLMConfig:
    max_tokens = args.judge_vllm_max_tokens if args.judge_vllm_max_tokens and args.judge_vllm_max_tokens > 0 else None
    top_p = args.judge_vllm_top_p or None
    server_pool = _parse_server_url_list(args.judge_vllm_server_urls)
    return JudgeLLMConfig(
        server_url=args.judge_vllm_server_url,
        server_urls=server_pool,
        host=args.judge_vllm_host,
        port=args.judge_vllm_port,
        path=args.judge_vllm_path,
        model=args.judge_vllm_model,
        temperature=args.judge_vllm_temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=args.judge_vllm_api_key,
        destination_service=args.judge_vllm_destination_service,
        timeout=args.judge_vllm_timeout,
        retries=args.judge_vllm_retries,
        retry_backoff=args.judge_vllm_retry_backoff,
        system_prompt=system_prompt,
    )


def _load_prompt(prompt_arg: Optional[str], *, description: str) -> Optional[str]:
    if not prompt_arg:
        return None
    if prompt_arg.startswith("@"):
        prompt_path = Path(prompt_arg[1:]).expanduser()
        if not prompt_path.exists():
            raise MCPError(f"{description} file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    return prompt_arg


def _configure_logging(level: str, *, suppress_warnings: bool) -> None:
    class _DropWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            return record.levelno != logging.WARNING

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if suppress_warnings:
        root = logging.getLogger()
        for handler in root.handlers:
            handler.addFilter(_DropWarningFilter())
    for name in ("LiteLLM", "LiteLLM Router"):
        lite_logger = logging.getLogger(name)
        lite_logger.setLevel(logging.WARNING)
        lite_logger.propagate = False
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def _disable_all_logging() -> None:
    logging.disable(logging.CRITICAL)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interact with the DeepWiki MCP server.",
    )
    parser.add_argument("--list-tools", action="store_true", help="List available MCP tools.")
    parser.add_argument(
        "--ask-question",
        nargs=2,
        metavar=("REPO", "QUESTION"),
        help="Call ask_question for the given repository.",
    )
    parser.add_argument("--read-structure", metavar="REPO", help="Display repository documentation outline.")
    parser.add_argument("--read-contents", metavar="REPO", help="Display repository documentation body.")
    parser.add_argument("--page", type=str, default=None, help="Target page when reading contents.")
    parser.add_argument("--section", type=str, default=None, help="Target section when reading contents.")
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Filter read-contents output to paragraphs containing this keyword.",
    )
    parser.add_argument(
        "--generate-dataset",
        metavar="REPO",
        action="append",
        help=(
            "Generate a DeepWiki semantic scaffold dataset. Provide multiple values by "
            "repeating the flag, separating with commas, or referencing @file lists."
        ),
    )
    parser.add_argument(
        "--repo-commit",
        type=str,
        default=None,
        help="Optional commit hash/tag to pin DeepWiki documentation.",
    )
    parser.add_argument("--output", type=str, help="Write dataset output to this file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write dataset outputs when processing multiple repositories.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Serialization format for dataset output (default: text).",
    )
    parser.add_argument(
        "--design-use-vllm",
        action="store_true",
        help="Enable external vLLM server for design intent rewriting.",
    )
    parser.add_argument("--design-vllm-server-url", type=str, default=None)
    parser.add_argument(
        "--design-vllm-server-urls",
        action="append",
        default=None,
        help="Additional design server URLs for LiteLLM pooling (supports comma-separated values).",
    )
    parser.add_argument("--design-vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--design-vllm-port", type=int, default=8000)
    parser.add_argument("--design-vllm-path", type=str, default="/v1/chat/completions")
    parser.add_argument("--design-vllm-model", type=str, default=None)
    parser.add_argument("--design-vllm-temperature", type=float, default=0.2)
    parser.add_argument("--design-vllm-top-p", type=float, default=None)
    parser.add_argument("--design-vllm-max-tokens", type=int, default=65536)
    parser.add_argument("--design-vllm-timeout", type=float, default=120.0)
    parser.add_argument("--design-vllm-retries", type=int, default=0)
    parser.add_argument("--design-vllm-retry-backoff", type=float, default=2.0)
    parser.add_argument("--design-vllm-api-key", type=str, default=None)
    parser.add_argument("--design-vllm-destination-service", type=str, default="openai")
    parser.add_argument("--judge-use-llm", action="store_true", help="Enable vLLM judge model.")
    parser.add_argument("--judge-vllm-server-url", type=str, default=None)
    parser.add_argument(
        "--judge-vllm-server-urls",
        action="append",
        default=None,
        help="Additional judge server URLs for LiteLLM pooling (supports comma-separated values).",
    )
    parser.add_argument("--judge-vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--judge-vllm-port", type=int, default=8000)
    parser.add_argument("--judge-vllm-path", type=str, default="/v1/chat/completions")
    parser.add_argument("--judge-vllm-model", type=str, default=None)
    parser.add_argument("--judge-vllm-temperature", type=float, default=0.0)
    parser.add_argument("--judge-vllm-top-p", type=float, default=None)
    parser.add_argument("--judge-vllm-max-tokens", type=int, default=65536)
    parser.add_argument("--judge-vllm-timeout", type=float, default=120.0)
    parser.add_argument("--judge-vllm-retries", type=int, default=0)
    parser.add_argument("--judge-vllm-retry-backoff", type=float, default=2.0)
    parser.add_argument("--judge-vllm-api-key", type=str, default=None)
    parser.add_argument("--judge-vllm-destination-service", type=str, default="openai")
    parser.add_argument("--judge-system-prompt", type=str, default=None)
    parser.add_argument(
        "--judge-max-rounds",
        type=int,
        default=1,
        help="Maximum judge/refinement passes (currently single-pass).",
    )
    parser.add_argument(
        "--narrative-modes",
        nargs="+",
        default=None,
        help="Narrative export modes (e.g., code, critic, cross).",
    )
    parser.add_argument(
        "--narrative-format",
        choices=["text", "json"],
        default="json",
        help="Narrative serialization format (default: json).",
    )
    parser.add_argument(
        "--narrative-output",
        type=str,
        default=None,
        help="Optional path to write narrative records.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Debug option: limit processing to the first N pages.",
    )
    parser.add_argument(
        "--max-sections-per-page",
        type=int,
        default=None,
        help="Debug option: process only the first M sections per page.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Debug option: cap concurrent page processing threads.",
    )
    parser.add_argument(
        "--section-workers",
        type=int,
        default=None,
        help=(
            "Optional: enable intra-page parallelism by processing sections concurrently. "
            "Defaults to sequential per page."
        ),
    )
    parser.add_argument(
        "--skip-page",
        action="append",
        default=None,
        help=(
            "Skip wiki pages whose headings match this value (case-insensitive). "
            "Use multiple times to skip several pages."
        ),
    )
    parser.add_argument(
        "--disable-hydration",
        action="store_true",
        help="Disable repo snippet hydration (faster; avoids heavy local I/O on large runs).",
    )
    parser.add_argument(
        "--hydration-timeout",
        type=float,
        default=None,
        help=(
            "Time budget in seconds for repo snippet hydration per section. "
            "On timeout (or if hydration pool is saturated), hydration is skipped for that section."
        ),
    )
    parser.add_argument(
        "--hydration-workers",
        type=int,
        default=None,
        help=(
            "Max concurrent hydration tasks per repo when --hydration-timeout is set "
            "(default: 4). Limits I/O amplification under high --section-workers."
        ),
    )
    parser.add_argument(
        "--repo-workers",
        type=int,
        default=None,
        help="Maximum number of repositories to process concurrently.",
    )
    parser.add_argument(
        "--repo-mp-workers",
        type=int,
        default=None,
        help=(
            "Optional: enable multiprocessing for parquet streaming by sharding part-* files across processes. "
            "Recommended on large CPU machines; avoids nested thread explosions. "
            "In mp mode each process runs repos sequentially (set --repo-workers=1)."
        ),
    )
    parser.add_argument(
        "--repo-batch-size",
        type=int,
        default=None,
        help="Number of repositories per batch before advancing to the next batch.",
    )
    parser.add_argument(
        "--narrative-output-dir",
        type=str,
        default=None,
        help="Directory to write narrative outputs when processing multiple repositories.",
    )
    parser.add_argument(
        "--parquet-input-dir",
        type=str,
        default=None,
        help=(
            "Mounted directory containing DeepWiki parquet part-* files. "
            "When set, uses `content` and `hdfs_path` from parquet instead of MCP and git clone."
        ),
    )
    parser.add_argument(
        "--parquet-all",
        action="store_true",
        help=(
            "When --parquet-input-dir is set and --generate-dataset is omitted, process all repos "
            "found in parquet (dangerous for very large datasets)."
        ),
    )
    parser.add_argument(
        "--parquet-max-repos",
        type=int,
        default=None,
        help=(
            "When --parquet-input-dir is set and --generate-dataset is omitted, process only the "
            "first N repos discovered in parquet."
        ),
    )
    parser.add_argument(
        "--parquet-repo-prefix",
        type=str,
        default=None,
        help=(
            "Optional repo_name prefix filter when discovering targets from parquet "
            "(e.g. 'org/' or 'org/repo')."
        ),
    )
    parser.add_argument(
        "--parquet-scan-batch-size",
        type=int,
        default=64,
        help="Batch size used when scanning parquet for repos (smaller uses less memory).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip repos that already have outputs in --output-dir/--narrative-output-dir.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining repositories even if some fail; logs failures instead of exiting non-zero.",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory to cache extracted repository zips when --parquet-input-dir is set "
            "(default: /tmp/deepwiki_repo_cache)."
        ),
    )
    parser.add_argument(
        "--repo-cache-cleanup",
        choices=["never", "on-success", "always"],
        default="never",
        help=(
            "Whether to delete the extracted repo cache directory after processing each repository "
            "in parquet mode (default: never)."
        ),
    )
    parser.add_argument(
        "--hdfs-output-dir",
        type=str,
        default=None,
        help=(
            "Optional HDFS destination directory (e.g. hdfs://cluster/path). "
            "When set, each repository is uploaded into its own subdirectory."
        ),
    )
    parser.add_argument(
        "--hdfs-bin",
        type=str,
        default="hdfs",
        help="HDFS CLI binary to use (default: hdfs).",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--suppress-warning-logs",
        action="store_true",
        help="Suppress WARNING-level log lines (only ERROR/INFO/DEBUG remain).",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Suppress all log output so only the repository progress bar remains.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, targets: Sequence[RepoTarget]) -> None:
    has_generation = bool(targets) or bool(
        args.parquet_input_dir
        and args.generate_dataset is None
        and (args.parquet_all or args.parquet_max_repos is not None)
    )
    if args.output and not has_generation:
        raise MCPError("--output can only be used with --generate-dataset.")
    if args.output_format != "text" and not has_generation:
        raise MCPError("--output-format can only be used with --generate-dataset.")
    if args.design_use_vllm and not has_generation:
        raise MCPError("--design-use-vllm requires --generate-dataset.")
    if args.judge_use_llm and not has_generation:
        raise MCPError("Judge options require --generate-dataset.")
    if args.design_use_vllm and not args.design_vllm_model:
        raise MCPError("--design-vllm-model is required when --design-use-vllm is set.")
    if args.judge_use_llm and not args.judge_vllm_model:
        raise MCPError("--judge-vllm-model is required when --judge-use-llm is set.")
    if args.narrative_output and not has_generation:
        raise MCPError("--narrative-output requires --generate-dataset.")
    if args.narrative_modes and not (args.narrative_output or args.narrative_output_dir):
        raise MCPError("--narrative-modes requires --narrative-output or --narrative-output-dir.")
    if args.narrative_output:
        narrative_path = Path(args.narrative_output).expanduser()
        if narrative_path.exists() and narrative_path.is_dir():
            raise MCPError(f"--narrative-output must be a file path, got directory: {narrative_path}")
    if args.judge_max_rounds < 1:
        raise MCPError("--judge-max-rounds must be >= 1.")
    if args.max_pages is not None and args.max_pages < 1:
        raise MCPError("--max-pages must be >= 1 when provided.")
    if args.max_sections_per_page is not None and args.max_sections_per_page < 1:
        raise MCPError("--max-sections-per-page must be >= 1 when provided.")
    if args.max_workers is not None and args.max_workers < 1:
        raise MCPError("--max-workers must be >= 1 when provided.")
    if args.section_workers is not None and args.section_workers < 1:
        raise MCPError("--section-workers must be >= 1 when provided.")
    if args.repo_workers is not None and args.repo_workers < 1:
        raise MCPError("--repo-workers must be >= 1 when provided.")
    if args.repo_mp_workers is not None and args.repo_mp_workers < 1:
        raise MCPError("--repo-mp-workers must be >= 1 when provided.")
    if args.repo_batch_size is not None and args.repo_batch_size < 1:
        raise MCPError("--repo-batch-size must be >= 1 when provided.")
    if args.hydration_timeout is not None and args.hydration_timeout <= 0:
        raise MCPError("--hydration-timeout must be > 0 when provided.")
    if args.hydration_workers is not None and args.hydration_workers < 1:
        raise MCPError("--hydration-workers must be >= 1 when provided.")
    if len(targets) > 1:
        if args.output and not args.output_dir:
            raise MCPError("Use --output-dir when generating datasets for multiple repositories.")
        if args.narrative_output and not args.narrative_output_dir:
            raise MCPError(
                "Use --narrative-output-dir when generating narratives for multiple repositories."
            )
    if args.parquet_input_dir:
        parquet_dir = Path(args.parquet_input_dir).expanduser()
        if not parquet_dir.exists() or not parquet_dir.is_dir():
            raise MCPError(f"--parquet-input-dir does not exist or is not a directory: {parquet_dir}")
        if not has_generation and not args.parquet_all and args.parquet_max_repos is None:
            raise MCPError(
                "When using --parquet-input-dir without --generate-dataset, "
                "provide --parquet-max-repos (recommended) or --parquet-all."
            )
        if args.parquet_max_repos is not None and args.parquet_max_repos < 1:
            raise MCPError("--parquet-max-repos must be >= 1 when provided.")
        if args.parquet_scan_batch_size is not None and args.parquet_scan_batch_size < 1:
            raise MCPError("--parquet-scan-batch-size must be >= 1 when provided.")
    if args.repo_mp_workers and not args.parquet_input_dir:
        raise MCPError("--repo-mp-workers is only supported with --parquet-input-dir.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.log_level, suppress_warnings=args.suppress_warning_logs)
    if args.progress_only:
        _disable_all_logging()
    if args.hdfs_output_dir is not None:
        args.hdfs_output_dir = args.hdfs_output_dir.strip()
    repo_targets = _resolve_repo_targets(args)
    is_parquet_discovery_mode = bool(
        args.parquet_input_dir
        and args.generate_dataset is None
        and (args.parquet_all or args.parquet_max_repos is not None)
    )
    try:
        validate_args(args, repo_targets)
    except MCPError as exc:
        logger.error("%s", exc)
        sys.exit(2)

    if args.hdfs_output_dir:
        if _hdfs_is_dir(args.hdfs_output_dir, hdfs_bin=args.hdfs_bin):
            logger.info("HDFS output base directory: %s", args.hdfs_output_dir)
        elif _hdfs_exists(args.hdfs_output_dir, hdfs_bin=args.hdfs_bin):
            raise MCPError(f"--hdfs-output-dir exists but is not a directory: {args.hdfs_output_dir}")
        else:
            _hdfs_mkdir_p(args.hdfs_output_dir, hdfs_bin=args.hdfs_bin)
            logger.info("Created HDFS output base directory: %s", args.hdfs_output_dir)

    session: Optional[Session] = None
    try:
        session = None if args.parquet_input_dir else initialize_session()
        if args.list_tools:
            if session is None:
                raise MCPError("--list-tools requires MCP session (omit --parquet-input-dir).")
            _print_tools(session)
        if args.ask_question:
            if session is None:
                raise MCPError("--ask-question requires MCP session (omit --parquet-input-dir).")
            repo, question = args.ask_question
            response = _ask_question_json(session, repo, question)
            print(json.dumps(response, ensure_ascii=False, indent=2))
        if args.read_structure:
            if session is None:
                raise MCPError("--read-structure requires MCP session (omit --parquet-input-dir).")
            _print_structure(session, args.read_structure)
        if args.read_contents:
            if session is None:
                raise MCPError("--read-contents requires MCP session (omit --parquet-input-dir).")
            _print_contents(
                session,
                args.read_contents,
                page=args.page,
                section=args.section,
                contains=args.contains,
            )

        design_config = _build_design_llm_config(args) if args.design_use_vllm else None
        judge_system_prompt = _load_prompt(
            args.judge_system_prompt,
            description="Judge system prompt",
        )
        judge_config = (
            _build_judge_llm_config(args, system_prompt=judge_system_prompt)
            if args.judge_use_llm
            else None
        )

        if is_parquet_discovery_mode:
            parquet_dir = Path(args.parquet_input_dir).expanduser()
            _run_parquet_stream(
                parquet_dir=parquet_dir,
                args=args,
                design_config=design_config,
                judge_config=judge_config,
            )
            return

        if not repo_targets:
            return

        dataset_path: Optional[Path] = Path(args.output) if args.output else None
        narrative_path: Optional[Path] = (
            Path(args.narrative_output) if args.narrative_output else None
        )
        narrative_modes: List[str] = (
            _normalize_narrative_modes(args.narrative_modes)
            if narrative_path
            else []
        )

        if len(repo_targets) == 1:
            target = repo_targets[0]
            _execute_pipeline_for_target(
                session=session,
                target=target,
                args=args,
                design_config=design_config,
                judge_config=judge_config,
                dataset_path=dataset_path,
                narrative_path=narrative_path,
                narrative_modes=narrative_modes,
                print_to_stdout=dataset_path is None,
            )
        else:
            if session is not None:
                delete_session(session)
                session = None
            _run_multi_repo_batch(
                targets=repo_targets,
                args=args,
                design_config=design_config,
                judge_config=judge_config,
                parquet_index_override=preloaded_parquet_index,
            )
    except MCPError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
