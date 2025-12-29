#!/usr/bin/env python3
"""Merge narrative text with hydrated code snippets keyed by reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REFERENCE_BLOCK_TEMPLATE = "{reference}\n{code}"


def _load_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected top-level JSON array.")
    return payload


def _merge_record(record: dict, stats: Dict[str, object]) -> Tuple[str, str, str]:
    page = record.get("page", "")
    section = record.get("section", "")
    narrative = record.get("narrative", "") or ""
    code_blocks = record.get("code_blocks") or []

    appended: List[str] = []
    for block in code_blocks:
        reference = (block or {}).get("reference", "")
        code = (block or {}).get("code", "")
        if not reference or not code:
            continue
        code = code.strip()
        if not code:
            continue
        reference = reference.strip()
        if not reference:
            continue
        if reference.lower().startswith("readme"):
            continue
        replacement = REFERENCE_BLOCK_TEMPLATE.format(reference=reference, code=code)
        if reference in narrative:
            narrative = narrative.replace(reference, replacement, 1)
            stats["matched"] += 1
            stats["matched_refs"].append(reference)
        else:
            appended.append(replacement)

    if appended:
        narrative = narrative.rstrip() + "\n\n" + "\n\n".join(appended)
    return page, section, narrative.strip()


def merge_records(records: Iterable[dict]) -> Tuple[List[str], Dict[str, object]]:
    stats: Dict[str, object] = {"matched": 0, "records": 0, "code_blocks": 0, "matched_refs": []}
    merged_lines: List[str] = []
    for record in records:
        stats["records"] += 1
        code_blocks = record.get("code_blocks") or []
        stats["code_blocks"] += sum(
            1 for block in code_blocks if (block or {}).get("reference") and (block or {}).get("code")
        )
        page, section, text = _merge_record(record, stats)
        header = f"# {page} :: {section}".strip()
        merged_lines.append(header)
        if text:
            merged_lines.append(text)
        merged_lines.append("")  # blank line separator
    return merged_lines, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace narrative references with hydrated code content.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the narrative JSON file (e.g., result_data/ncnn_narratives.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination text file for merged narrative and code.",
    )
    args = parser.parse_args()

    records = _load_records(args.input)
    merged_lines, stats = merge_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(merged_lines).rstrip() + "\n", encoding="utf-8")
    print(
        f"Matched {stats['matched']} references "
        f"out of {stats['code_blocks']} code blocks across {stats['records']} records."
    )
    matched_refs: List[str] = stats.get("matched_refs", [])
    if matched_refs:
        print("Matched references:")
        for ref in matched_refs:
            print(f"  - {ref}")


if __name__ == "__main__":
    main()
