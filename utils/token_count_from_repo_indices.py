#!/usr/bin/env python3
"""
Count total tiktoken tokens for code snippets referenced by repo indices.

Inputs:
- repo_indices.json: { "owner/repo": ["path.py:1-10", "dir/a b.py", ...], ... }
- repo_hdfs_map.json: { "repo_hdfs_map": { "owner/repo": "hdfs://.../repo.zip", ... }, ... }

Behavior:
- For each repo, download and extract the zip from hdfs_path (cached).
- For each index entry, read the referenced file (optionally line-ranged) and add token count.
- De-duplicates by (repo, path, start, end) after normalization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import tiktoken

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class IndexRef:
    path: str
    start: int
    end: int  # 0 means "to end of file"


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "_")


def _run_hdfs_get(hdfs_bin: str, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([hdfs_bin, "dfs", "-get", src, str(dst)], check=True)


def ensure_repo_extracted(*, repo: str, hdfs_zip_path: str, cache_dir: Path, hdfs_bin: str) -> Path:
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


def parse_index(raw: str) -> Optional[IndexRef]:
    s = (raw or "").strip()
    if not s:
        return None
    # Some indices might still contain bullet prefixes.
    while s.startswith(("- ", "* ", "• ")):
        s = s[2:].strip()
    # Drop surrounding backticks if present.
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()

    if ":" not in s:
        return IndexRef(path=s, start=1, end=0)
    path_part, remainder = s.rsplit(":", 1)
    path_part = path_part.strip()
    remainder = remainder.strip()
    if not remainder or not remainder.replace("-", "").isdigit():
        return IndexRef(path=s, start=1, end=0)
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
    return IndexRef(path=path_part, start=start, end=end)


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
    if ref.start <= 1 and ref.end <= 0:
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
    ap = argparse.ArgumentParser(description="Count tiktoken tokens from repo_indices.json + repo_hdfs_map.json.")
    ap.add_argument("--repo-indices", type=Path, required=True)
    ap.add_argument("--repo-hdfs-map", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/deepwiki_repo_cache"))
    ap.add_argument("--hdfs-bin", type=str, default="hdfs")
    ap.add_argument("--encoding", type=str, default="cl100k_base")
    ap.add_argument("--repo-workers", type=int, default=8, help="Parallelism over repos (downloads/extracts).")
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--progress", action="store_true", help="Enable tqdm progress bars.")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    repo_indices = json.loads(args.repo_indices.read_text(encoding="utf-8"))
    if not isinstance(repo_indices, dict):
        raise SystemExit("repo_indices must be a JSON object mapping repo -> indices[]")

    repo_hdfs_payload = json.loads(args.repo_hdfs_map.read_text(encoding="utf-8"))
    repo_hdfs_map = repo_hdfs_payload.get("repo_hdfs_map") if isinstance(repo_hdfs_payload, dict) else None
    if not isinstance(repo_hdfs_map, dict):
        raise SystemExit("repo_hdfs_map.json must contain key 'repo_hdfs_map' mapping repo -> hdfs_path")

    repos = sorted(repo_indices.keys())
    if args.limit_repos is not None:
        repos = repos[: args.limit_repos]

    # Deduplicate within repo by normalized (path, start, end)
    per_repo_refs: Dict[str, Set[IndexRef]] = {}
    for repo in repos:
        raw_list = repo_indices.get(repo) or []
        if not isinstance(raw_list, list):
            continue
        refs: Set[IndexRef] = set()
        for raw in raw_list:
            ref = parse_index(str(raw))
            if ref:
                refs.add(ref)
        if refs:
            per_repo_refs[repo] = refs

    # Parallel over repos
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_repo(repo: str) -> Tuple[str, int, int, int]:
        hdfs_zip = str(repo_hdfs_map.get(repo) or "").strip()
        if not hdfs_zip:
            return repo, 0, 0, len(per_repo_refs.get(repo, set()))
        repo_root = ensure_repo_extracted(repo=repo, hdfs_zip_path=hdfs_zip, cache_dir=args.cache_dir, hdfs_bin=args.hdfs_bin)
        tokens = 0
        missing = 0
        count = 0
        for ref in sorted(per_repo_refs.get(repo, set()), key=lambda r: (r.path, r.start, r.end)):
            snippet = read_snippet(repo_root, ref)
            if snippet is None:
                missing += 1
                continue
            tokens += len(enc.encode_ordinary(snippet))
            count += 1
        return repo, tokens, count, missing

    use_tqdm = bool(args.progress and tqdm)
    repo_iter = tqdm(repos, desc="Repos", unit="repo") if use_tqdm else repos

    total_tokens = 0
    total_refs = 0
    total_missing = 0
    missing_repo_map = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.repo_workers))) as ex:
        futures = [ex.submit(process_repo, repo) for repo in repo_iter]
        for fut in as_completed(futures):
            repo, tokens, count, missing = fut.result()
            if repo not in repo_hdfs_map:
                missing_repo_map += 1
            total_tokens += tokens
            total_refs += count
            total_missing += missing

    print("DONE")
    print("repos_total:", len(repos))
    print("repos_with_refs:", len(per_repo_refs))
    print("refs_resolved:", total_refs)
    print("refs_missing:", total_missing)
    print("repos_missing_hdfs_path:", missing_repo_map)
    print("total_tokens:", total_tokens)


if __name__ == "__main__":
    main()

