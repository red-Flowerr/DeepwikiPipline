#!/usr/bin/env python3
"""List DeepWiki source indices referenced in text or narrative exports.

This utility reads a DeepWiki page dump (e.g. ``utils/page.txt``) or a
generated narratives JSON file (e.g. ``result_data/verl_narratives.json``),
identifies citations such as ``[path/to/file.py:10-20]`` and prints the
raw labels for quick inspection. The parsing logic mirrors the helper
used in ``deepwiki_mcp_client.py`` so the output matches the pipeline.

Examples
--------
List every source reference in the page:

    python utils/find_index.py

Inspect a narratives JSON export with source identifiers:

    python utils/find_index.py --page result_data/verl_narratives.json --show-source

Include surrounding context for each reference:

    python utils/find_index.py --with-context

Strip markdown links from the page (overwrites in place):

    python utils/find_index.py --strip-links --in-place
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import ast
import re


SOURCE_BLOCK_RE = re.compile(r"\[Source:\s*([^\]]+)\]")


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)


def _iter_string_nodes(obj: object, prefix: str = "") -> Iterator[Tuple[str, str]]:
    """Yield (path, text) pairs for every string within a JSON-like structure."""
    if isinstance(obj, str):
        yield (prefix or "<root>", obj)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_string_nodes(item, child_prefix)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from _iter_string_nodes(value, child_prefix)


def _strip_links_structure(obj: object) -> object:
    if isinstance(obj, str):
        return _strip_markdown_links(obj)
    if isinstance(obj, list):
        return [_strip_links_structure(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _strip_links_structure(value) for key, value in obj.items()}
    return obj


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepwiki_pipeline import parse_sources_links  # type: ignore  # noqa: E402


@dataclass
class SourceEntry:
    label: str
    path: str
    start: int
    end: int
    page_line: int
    context: str
    source: str


def _find_context(lines: List[str], index: int) -> str:
    """Return the closest non-empty line before ``index`` as context."""
    for offset in range(index - 1, -1, -1):
        stripped = lines[offset].strip()
        if stripped:
            return stripped
    return ""


def _extract_sources(page_text: str, source_id: str) -> List[SourceEntry]:
    """Parse DeepWiki source references from a page dump."""
    lines = page_text.splitlines()
    entries: List[SourceEntry] = []
    seen: set[tuple[str, str, int, int]] = set()
    in_details = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("<details"):
            in_details = True
            continue
        if stripped.startswith("</details"):
            in_details = False
            continue

        link_iterable = []
        if "**Sources:**" in line:
            raw_links = line.split("**Sources:**", 1)[1]
            link_iterable = parse_sources_links(raw_links)
        elif in_details and stripped.startswith("- ["):
            link_iterable = parse_sources_links(line)

        for link in link_iterable:
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
            if "://" in path or path.lower().startswith("mailto:"):
                # Skip external hyperlinks (badges, social links, etc.)
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
                    source=source_id,
                )
            )

        if "[Source:" in line:
            for raw in SOURCE_BLOCK_RE.findall(line):
                data = raw.strip()
                if not data:
                    continue
                path = data
                start = 1
                end_value = 1
                if ":" in data:
                    path_part, remainder = data.split(":", 1)
                    path = path_part.strip()
                    remainder = remainder.strip()
                    if remainder:
                        if "-" in remainder:
                            start_str, end_str = remainder.split("-", 1)
                        else:
                            start_str = end_str = remainder
                        try:
                            start = int(start_str)
                        except ValueError:
                            start = 1
                        try:
                            end_value = int(end_str)
                        except ValueError:
                            end_value = start
                key = (data, path, start, end_value)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(
                    SourceEntry(
                        label=data,
                        path=path,
                        start=start,
                        end=end_value,
                        page_line=idx + 1,
                        context=_find_context(lines, idx),
                        source=source_id,
                    )
                )
    return entries


def _strip_markdown_links(text: str) -> str:
    pattern = re.compile(r"!?\[([^\]]+)\]\([^\)]*\)")

    def replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        return label

    return pattern.sub(replace, text)


def _decode_page_text(page_text: str) -> str:
    stripped = page_text.strip()
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            pass
    return page_text


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
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Include the source identifier where each citation was found.",
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Skip printing citation labels (useful with --strip-links).",
    )
    parser.add_argument(
        "--strip-links",
        action="store_true",
        help="Strip markdown link targets from the page content.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="When stripping links, overwrite the source page file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for cleaned content (requires --strip-links).",
    )
    args = parser.parse_args(argv)

    if not args.page.exists():
        parser.error(f"Page file not found: {args.page}")

    suffix = args.page.suffix.lower()
    json_obj: Optional[object] = None
    documents: List[Tuple[str, str]] = []

    if suffix == ".json":
        with args.page.open("r", encoding="utf-8") as handle:
            json_obj = json.load(handle)
        for source_id, text in _iter_string_nodes(json_obj):
            documents.append((source_id, text))
    else:
        page_text = args.page.read_text(encoding="utf-8")
        page_text = _decode_page_text(page_text)
        documents.append((args.page.name, page_text))

    entries: List[SourceEntry] = []
    for source_id, text in documents:
        entries.extend(_extract_sources(text, source_id))

    if not args.no_citations:
        if not entries:
            print("No '**Sources:**' references found.", file=sys.stderr)
        else:
            for entry in entries:
                display_label = entry.label
                if args.show_source:
                    display_label = f"{entry.source}: {display_label}"
                _safe_print(display_label)
                if args.with_context:
                    extra = (
                        f"[{entry.source}] line {entry.page_line}; "
                        f"{entry.path}:{entry.start}-{entry.end}"
                    )
                    _safe_print(f"  {extra}")
                    if entry.context:
                        _safe_print(f"  Context: {entry.context}")

    if args.strip_links:
        target_path: Optional[Path] = None
        if args.output:
            target_path = args.output
        elif args.in_place:
            target_path = args.page

        if suffix == ".json":
            assert json_obj is not None
            cleaned_obj = _strip_links_structure(json_obj)
            output_text = json.dumps(cleaned_obj, ensure_ascii=False, indent=2)
        else:
            cleaned_text = _strip_markdown_links(documents[0][1] if documents else "")
            output_text = cleaned_text

        if target_path:
            target_path.write_text(output_text, encoding="utf-8")
        else:
            _safe_print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
