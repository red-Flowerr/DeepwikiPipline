#!/usr/bin/env python3
"""Concatenate narrative fields from a DeepWiki narrative dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

CHART_CHARS = {"┌", "┐", "└", "┘", "─", "│", "┬", "┴", "├", "┤"}


def _should_skip_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    if "|" in stripped and "http" not in stripped:
        segments = [segment.strip() for segment in stripped.split("|")]
        if len(segments) > 2:
            return True
    return False


def _normalize_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    if stripped.startswith(("-", "*")):
        return stripped.lstrip("-*").strip()
    if stripped and stripped[0].isdigit() and stripped[1:2] in {".", ")", "]"}:
        return stripped[2:].strip()
    if stripped in {"---", "***"}:
        return ""
    if any(char in CHART_CHARS for char in stripped):
        return ""
    return stripped


def _strip_markdown(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    in_code_block = False
    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if _should_skip_table_line(raw):
            continue
        normalized = _normalize_line(raw)
        if normalized != "":
            cleaned.append(normalized)
        else:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
    # Remove trailing blank lines
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    result_lines: List[str] = []
    previous_blank = True
    for line in cleaned:
        if line == "":
            if not previous_blank:
                result_lines.append("")
            previous_blank = True
        else:
            result_lines.append(line)
            previous_blank = False
    return "\n".join(result_lines).strip()


def _load_narratives(path: Path) -> Iterable[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    if not isinstance(payload, list):
        raise ValueError("Expected a list of records in the input JSON.")
    narratives: List[str] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        text = entry.get("narrative")
        if isinstance(text, str) and text.strip():
            narratives.append(_strip_markdown(text))
    return narratives


def _write_output(narratives: Iterable[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(narratives), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate all narrative fields into a single text file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the narrative JSON (e.g., result_data/verl_narratives.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination text file for the concatenated narratives.",
    )
    args = parser.parse_args()

    narratives = list(_load_narratives(args.input))
    if not narratives:
        raise SystemExit("No narrative entries were found in the input file.")
    _write_output(narratives, args.output)


if __name__ == "__main__":
    main()
