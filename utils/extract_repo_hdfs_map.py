#!/usr/bin/env python3
"""
Extract repo_name -> hdfs_path mapping for an existing batch of narrative exports.

Usage example:
  python utils/extract_repo_hdfs_map.py \
    --narratives-dir result_data/batch_narratives \
    --parquet-dir /mnt/hdfs/userx/shanyong/code/code_wiki/deepwiki \
    --output repo_hdfs_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: pip install pyarrow") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def iter_narrative_files(narratives_dir: Path) -> Iterable[Path]:
    yield from sorted(narratives_dir.glob("*_narratives.json"))


def collect_repos(narratives_dir: Path) -> Set[str]:
    repos: Set[str] = set()
    files = list(iter_narrative_files(narratives_dir))
    it = tqdm(files, desc="Scanning narratives", unit="file") if tqdm else files
    for path in it:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            repo = str(item.get("repo") or "").strip()
            if repo:
                repos.add(repo)
    return repos


def resolve_hdfs_paths(parquet_dir: Path, repos: Set[str]) -> Dict[str, str]:
    remaining = {r for r in repos if r}
    if not remaining:
        return {}

    part_files = sorted(p for p in parquet_dir.iterdir() if p.is_file() and p.name.startswith("part-"))
    if not part_files:
        raise ValueError(f"No part-* parquet files under {parquet_dir}")

    mapping: Dict[str, str] = {}
    cols = ["repo_name", "hdfs_path", "error_message"]
    part_iter = tqdm(part_files, desc="Scanning parquet parts", unit="part") if tqdm else part_files
    for part in part_iter:
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=4096, columns=cols):
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build repo_name -> hdfs_path map from narratives + parquet.")
    ap.add_argument("--narratives-dir", type=Path, required=True)
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repos = collect_repos(args.narratives_dir)
    print("repos_found:", len(repos))
    mapping = resolve_hdfs_paths(args.parquet_dir, repos)
    missing = sorted([r for r in repos if r not in mapping])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "repo_hdfs_map": mapping,
                "missing_repos": missing,
                "repos_found": len(repos),
                "repos_mapped": len(mapping),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("repos_mapped:", len(mapping))
    print("missing_repos:", len(missing))
    print("wrote:", str(args.output))


if __name__ == "__main__":
    main()

