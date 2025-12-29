#!/usr/bin/env python3
"""Fetch a DeepWiki outline and write a sanitized version to disk."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from deepwiki_pipeline import (
    MCPError,
    Session,
    call_tool,
    delete_session,
    extract_text_blocks,
    initialize_session,
)


def _parse_replacements(specs: Iterable[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in specs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid replacement '{item}'. Expected format: original=replacement"
            )
        original, replacement = item.split("=", 1)
        pairs.append((original, replacement))
    return pairs


def _sanitize_outline(
    outline_text: str,
    *,
    replacements: List[Tuple[str, str]],
    excludes: List[str],
) -> str:
    lines = outline_text.splitlines()
    sanitized: List[str] = []
    for raw_line in lines:
        line = raw_line
        for original, replacement in replacements:
            if original:
                line = line.replace(original, replacement)
        if excludes and any(token in line for token in excludes):
            continue
        sanitized.append(line.rstrip())

    compacted: List[str] = []
    previous_blank = False
    for line in sanitized:
        if not line.strip():
            if previous_blank:
                continue
            previous_blank = True
            compacted.append("")
        else:
            previous_blank = False
            compacted.append(line)

    while compacted and not compacted[0].strip():
        compacted.pop(0)
    while compacted and not compacted[-1].strip():
        compacted.pop()
    return "\n".join(compacted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DeepWiki structure text and apply simple sanitisation.",
    )
    parser.add_argument("repo", help="Repository identifier, e.g. volcengine/verl.")
    parser.add_argument(
        "--repo-commit",
        type=str,
        default=None,
        help="Optional commit hash or tag recognised by DeepWiki.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the sanitised outline to this file (prints to stdout otherwise).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Drop lines containing this substring. Can be specified multiple times.",
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="ORIGINAL=REPLACEMENT",
        help="Replace substring ORIGINAL with REPLACEMENT. May be repeated.",
    )
    parser.add_argument(
        "--drop-readme",
        action="store_true",
        help="Convenience flag that excludes lines mentioning README.",
    )
    args = parser.parse_args()

    replacements = _parse_replacements(args.replace)
    excludes = list(args.exclude)
    if args.drop_readme and "README" not in excludes:
        excludes.append("README")

    session: Session | None = None
    try:
        session = initialize_session()
        payload = {
            "repoName": args.repo,
        }
        if args.repo_commit:
            payload["repoCommit"] = args.repo_commit
        response = call_tool(session, "read_wiki_structure", payload)
        blocks = extract_text_blocks(response)
        if not blocks:
            raise MCPError("read_wiki_structure returned no textual content.")
        outline_text = "\n\n".join(blocks)
        sanitized = _sanitize_outline(
            outline_text,
            replacements=replacements,
            excludes=excludes,
        )
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(sanitized, encoding="utf-8")
        else:
            print(sanitized)
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
