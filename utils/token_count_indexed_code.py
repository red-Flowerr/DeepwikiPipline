#!/usr/bin/env python3
"""
Compute total token count for code referenced by wiki indices in generated narrative exports.

High-level steps:
1) Load DeepWiki parquet directory to map repo_name -> hdfs_path (repo zip path).
2) For each narratives JSON file, extract indices from original_context:
   - inline `Sources: ` lists (backticked paths supported, may include spaces)
   - hydration snippet blocks (label line + fenced code block)
   - [Source: ...] tokens
3) Resolve each index to a repo-relative file path and optional line range.
4) Download/extract repo zip (cached), read the requested snippet/full file, and count tokens with tiktoken.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tiktoken

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: pip install pyarrow") from exc


SOURCES_HEADER_RE = re.compile(
    r"(?i)^\s*(?:\*\*|__)?\s*sources?\s*(?:\*\*|__)?\s*[:：\-–—]\s*(?P<rest>.*)$"
)
BACKTICK_RE = re.compile(r"`([^`]+)`")
# Only count indices that appear under Sources: ... sections.
# Do not treat arbitrary [Source: ...] tokens or hydrated label blocks as indices.
PLAIN_INDEX_RE = re.compile(r"^[A-Za-z0-9_.\-/ ]+(?::\d+(?:-\d+)?)?$")

_RANGE_SUFFIX_RE = re.compile(r"^(?P<path>.+?)(?P<range>:\d+(?:-\d+)?)?$")

def _normalize_index_label(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    while s.startswith(("- ", "* ", "• ")):
        s = s[2:].strip()
    s = re.sub(r"^\d+\.\s+", "", s).strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    m = _RANGE_SUFFIX_RE.match(s)
    if not m:
        return s
    path = (m.group("path") or "").strip()
    rng = (m.group("range") or "").strip()
    return f"{path}{rng}".strip()

def _extract_sources_from_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SOURCES_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        rest = (m.group("rest") or "").strip()
        if rest:
            items = [x.strip() for x in BACKTICK_RE.findall(rest) if x.strip()]
            if not items:
                if (
                    PLAIN_INDEX_RE.match(rest)
                    and "`" not in rest
                    and not any(sep in rest for sep in (",", ";", "，"))
                ):
                    items = [rest]
            out.extend(items)
            i += 1
            continue
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if not (
                nxt.startswith(("- ", "* ", "• "))
                or nxt.startswith("`")
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            out.extend([x.strip() for x in BACKTICK_RE.findall(nxt) if x.strip()])
            i += 1
        continue
    return out

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _run_hdfs_get(hdfs_bin: str, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([hdfs_bin, "dfs", "-get", src, str(dst)], check=True)


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "_")


@dataclass(frozen=True)
class IndexRef:
    raw: str
    path: str
    start: int
    end: int


def _parse_index(raw: str) -> Optional[IndexRef]:
    s = (raw or "").strip()
    if not s:
        return None
    # drop surrounding backticks if present
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    # Support "path:3-16" where path may contain spaces.
    if ":" not in s:
        return IndexRef(raw=raw, path=s, start=1, end=0)
    path_part, remainder = s.rsplit(":", 1)
    path_part = path_part.strip()
    remainder = remainder.strip()
    if not remainder or not remainder.replace("-", "").isdigit():
        # Probably a Windows drive or something; treat as full file.
        return IndexRef(raw=raw, path=s, start=1, end=0)
    if "-" in remainder:
        a, b = remainder.split("-", 1)
    else:
        a = b = remainder
    try:
        start = int(a)
    except ValueError:
        start = 1
    try:
        end = int(b)
    except ValueError:
        end = start
    return IndexRef(raw=raw, path=path_part, start=start, end=end)


def extract_indices_from_original_context(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()
    lines = str(text).splitlines()
    for raw in _extract_sources_from_lines(lines):
        item = _normalize_index_label(raw)
        if item and item not in seen:
            seen.add(item)
            found.append(item)

    return found


def iter_narrative_records(path: Path) -> Iterable[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array: {path}")
    for item in payload:
        if isinstance(item, dict):
            yield item


def load_repo_to_hdfs_path(parquet_dir: Path) -> Dict[str, str]:
    part_files = sorted(p for p in parquet_dir.iterdir() if p.is_file() and p.name.startswith("part-"))
    if not part_files:
        raise ValueError(f"No part-* under {parquet_dir}")
    mapping: Dict[str, str] = {}
    cols = ["repo_name", "hdfs_path", "error_message"]
    for part in part_files:
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            table = batch.to_pydict()
            names = table.get("repo_name") or []
            paths = table.get("hdfs_path") or []
            errs = table.get("error_message") or []
            for repo, hp, err in zip(names, paths, errs):
                if err:
                    continue
                repo_s = str(repo or "").strip()
                hp_s = str(hp or "").strip()
                if repo_s and hp_s and repo_s not in mapping:
                    mapping[repo_s] = hp_s
    return mapping


def load_repo_to_hdfs_path_for_repos(parquet_dir: Path, repos: Set[str]) -> Dict[str, str]:
    """
    Resolve only the requested repos from the parquet dataset.

    This is much faster than scanning the entire dataset when you only have
    a few hundred repos from existing narrative exports.
    """
    remaining = {r.strip() for r in repos if r and r.strip()}
    if not remaining:
        return {}
    part_files = sorted(p for p in parquet_dir.iterdir() if p.is_file() and p.name.startswith("part-"))
    if not part_files:
        raise ValueError(f"No part-* under {parquet_dir}")

    mapping: Dict[str, str] = {}
    cols = ["repo_name", "hdfs_path", "error_message"]
    part_iter = tqdm(part_files, desc="Scanning parquet parts", unit="part") if tqdm else part_files
    for part in part_iter:
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            repo_values = batch.column(0).to_pylist()
            hdfs_values = batch.column(1).to_pylist()
            err_values = batch.column(2).to_pylist()
            for repo, hp, err in zip(repo_values, hdfs_values, err_values):
                if err:
                    continue
                repo_s = str(repo or "").strip()
                if repo_s not in remaining:
                    continue
                hp_s = str(hp or "").strip()
                if not hp_s:
                    continue
                mapping[repo_s] = hp_s
                remaining.discard(repo_s)
                if not remaining:
                    return mapping
    return mapping


def ensure_repo_extracted(*, repo: str, hdfs_zip_path: str, cache_dir: Path, hdfs_bin: str) -> Path:
    import zipfile
    import tempfile
    import shutil

    repo_dir = cache_dir / _repo_slug(repo)
    marker = repo_dir / ".extracted_ok"
    if marker.exists():
        children = [p for p in repo_dir.iterdir() if p.name != marker.name]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return repo_dir

    shutil.rmtree(repo_dir, ignore_errors=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="deepwiki_idx_zip_"))
    try:
        local_zip = tmp / "repo.zip"
        _run_hdfs_get(hdfs_bin, hdfs_zip_path, local_zip)
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(repo_dir)
        marker.write_text("ok\n", encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    children = [p for p in repo_dir.iterdir() if p.name != marker.name]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return repo_dir


def read_snippet(repo_root: Path, ref: IndexRef) -> Optional[str]:
    target = (repo_root / ref.path).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError:
        return None
    if not target.is_file():
        return None
    try:
        data = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if ref.end <= 0 and ref.start <= 1:
        return data
    lines = data.splitlines()
    if not lines:
        return ""
    start_idx = max(ref.start - 1, 0)
    if start_idx >= len(lines):
        return ""
    if ref.end <= 0:
        return "\n".join(lines[start_idx:]) + "\n"
    end_idx = min(max(ref.end, ref.start), len(lines))
    return "\n".join(lines[start_idx:end_idx]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Count tokens for indexed code referenced in narrative original_context.")
    ap.add_argument("--narratives-dir", type=Path, required=True, help="Directory containing *_narratives.json files.")
    ap.add_argument("--parquet-dir", type=Path, required=True, help="DeepWiki parquet directory with part-* files.")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/deepwiki_repo_cache"), help="Repo zip extract cache.")
    ap.add_argument("--hdfs-bin", type=str, default="hdfs", help="HDFS CLI binary name.")
    ap.add_argument("--encoding", type=str, default="cl100k_base", help="tiktoken encoding name.")
    ap.add_argument("--repo-workers", type=int, default=1, help="Parallelism over repos (downloads/extracts).")
    ap.add_argument("--limit-files", type=int, default=None, help="Only process first N narrative files (for testing).")
    ap.add_argument("--limit-records", type=int, default=None, help="Only process first N records per file (for testing).")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    files = sorted(Path(args.narratives_dir).glob("*_narratives.json"))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No *_narratives.json under {args.narratives_dir}")

    # 1) Collect unique indices per repo from narratives.
    per_repo_indices: Dict[str, Set[str]] = {}
    for fpath in (tqdm(files, desc="Scanning narrative files", unit="file") if tqdm else files):
        records_seen = 0
        for rec in iter_narrative_records(fpath):
            records_seen += 1
            if args.limit_records is not None and records_seen > args.limit_records:
                break
            repo = str(rec.get("repo") or "").strip()
            if not repo:
                continue
            oc = rec.get("original_context") or ""
            indices = extract_indices_from_original_context(str(oc))
            if not indices:
                continue
            bucket = per_repo_indices.setdefault(repo, set())
            for raw in indices:
                bucket.add(raw)

    repos = sorted(per_repo_indices.keys())
    if not repos:
        print("DONE")
        print("files:", len(files))
        print("repos_with_indices: 0")
        print("total_tokens: 0")
        return

    # 2) Resolve the repos to hdfs_path from parquet (targeted scan).
    repo_map = load_repo_to_hdfs_path_for_repos(args.parquet_dir, set(repos))
    missing_repos = [r for r in repos if r not in repo_map]
    if missing_repos:
        print(f"WARNING: {len(missing_repos)} repos not found in parquet (first 10): {missing_repos[:10]}")

    # 3) Count tokens per repo in parallel.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def count_repo(repo: str) -> Tuple[str, int, int, int]:
        """
        Returns (repo, tokens, indices, missing_indices).
        """
        hdfs_zip_path = repo_map.get(repo)
        if not hdfs_zip_path:
            return repo, 0, 0, len(per_repo_indices.get(repo, set()))
        repo_root = ensure_repo_extracted(
            repo=repo,
            hdfs_zip_path=hdfs_zip_path,
            cache_dir=args.cache_dir,
            hdfs_bin=args.hdfs_bin,
        )
        tokens = 0
        missing_local = 0
        indices_local = 0
        for raw in sorted(per_repo_indices.get(repo, set())):
            ref = _parse_index(raw)
            if not ref:
                missing_local += 1
                continue
            snippet = read_snippet(repo_root, ref)
            if snippet is None:
                missing_local += 1
                continue
            tokens += len(enc.encode_ordinary(snippet))
            indices_local += 1
        return repo, tokens, indices_local, missing_local

    repo_workers = max(1, int(args.repo_workers or 1))
    total_tokens = 0
    total_indices = 0
    missing = 0

    iterator = repos
    if tqdm:
        iterator = tqdm(repos, desc="Counting repos", unit="repo")

    # Submit all repos; limit concurrency via max_workers.
    with ThreadPoolExecutor(max_workers=repo_workers) as ex:
        futures = [ex.submit(count_repo, repo) for repo in iterator]
        for fut in as_completed(futures):
            repo, tokens, indices_local, missing_local = fut.result()
            total_tokens += tokens
            total_indices += indices_local
            missing += missing_local

    print("DONE")
    print("files:", len(files))
    print("repos_with_indices:", len(repos))
    print("unique_indices:", sum(len(v) for v in per_repo_indices.values()))
    print("resolved_indices:", total_indices)
    print("missing_indices:", missing)
    print("total_tokens:", total_tokens)


if __name__ == "__main__":
    main()
