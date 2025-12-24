#!/usr/bin/env python3
"""List DeepWiki source indices referenced in a page export.

This utility reads a DeepWiki page dump (such as ``utils/page.txt``),
finds ``**Sources:**`` lines, and prints the raw ``[file:line-range]``
labels for quick inspection.  The parsing logic mirrors the helper used
in ``deepwiki_mcp_client.py`` so the output matches the pipeline.

Examples
--------
List every source reference in the page:

    python utils/find_index.py

Include surrounding context for each reference:

    python utils/find_index.py --with-context
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepwiki_mcp_client import parse_sources_links  # type: ignore  # noqa: E402


@dataclass
class SourceEntry:
    label: str
    path: str
    start: int
    end: int
    page_line: int
    context: str


def _find_context(lines: List[str], index: int) -> str:
    """Return the closest non-empty line before ``index`` as context."""
    for offset in range(index - 1, -1, -1):
        stripped = lines[offset].strip()
        if stripped:
            return stripped
    return ""


def _extract_sources(page_text: str) -> List[SourceEntry]:
    """Parse DeepWiki source references from a page dump."""
    lines = page_text.splitlines()
    entries: List[SourceEntry] = []
    seen: set[tuple[str, str, int, int]] = set()
    for idx, line in enumerate(lines):
        if "[" not in line or "]" not in line:
            continue
        for link in parse_sources_links(line):
            label = link.get("label", "").strip()
            path = link.get("path", "").strip()
            if not label or not path:
                continue
            if label.startswith("#") or path.startswith("#"):
                # Skip intra-document anchors like [#4]
                continue
            if "diagram" in path.lower():
                # Skip non-file stubs such as "Diagram 2 from ..."
                continue
            start = int(link.get("start", 1))
            end_value = int(link.get("end", link.get("start", 1)))
            key = (label, path, start, end_value)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                SourceEntry(
                    label=label,
                    path=path,
                    start=start,
                    end=end_value,
                    page_line=idx + 1,
                    context=_find_context(lines, idx),
                )
            )
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect '**Sources:**' references in a DeepWiki page dump."
    )
    parser.add_argument(
        "--page",
        type=Path,
        default=Path(__file__).resolve().parent / "page.txt",
        help="Path to the DeepWiki page export (default: utils/page.txt).",
    )
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="Print context (nearest preceding non-empty line) and page line numbers.",
    )
    args = parser.parse_args(argv)

    if not args.page.exists():
        parser.error(f"Page file not found: {args.page}")

    page_text = args.page.read_text(encoding="utf-8")
    entries = _extract_sources(page_text)
    if not entries:
        print("No '**Sources:**' references found.", file=sys.stderr)
        return 1

    for entry in entries:
        print(entry.label)
        if args.with_context:
            extra = f"(page line {entry.page_line}; {entry.path}:{entry.start}-{entry.end})"
            print(f"  {extra}")
            if entry.context:
                print(f"  Context: {entry.context}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
