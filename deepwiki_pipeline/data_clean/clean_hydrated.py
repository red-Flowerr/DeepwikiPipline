"""Utilities for cleaning hydrated wiki outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

# Match URL-like substrings (http/https, www., bare domains with TLDs)
URL_PATTERN = re.compile(
    r"""(
        https?://[^\s)]+ |
        www\.[^\s)]+ |
        [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,} |
        [A-Za-z0-9.-]+\.(?:com|net|org|io|ai|dev|co|tech|app|cn)(?:/[^\s)]*)?
    )""",
    re.IGNORECASE | re.VERBOSE,
)

HTML_TAG_PATTERN = re.compile(r"<[^>\n]+>")


def _strip_urls(line: str) -> str:
    """Remove URL-like substrings from a line of text."""
    cleaned = URL_PATTERN.sub("", line)
    cleaned = HTML_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"<[^>]*$", "", cleaned)
    if cleaned.lstrip().startswith("<") or cleaned.rstrip().endswith(">"):
        return ""
    # Collapse multiple spaces introduced by removals
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip().strip('"').strip("'")


def clean_lines(lines: Iterable[str]) -> Iterable[str]:
    """Yield cleaned lines with URL-like fragments removed."""
    for raw in lines:
        cleaned = _strip_urls(raw)
        if cleaned:
            yield cleaned


def clean_file(source: Path, destination: Path) -> None:
    """Load `source`, clean its contents, and write to `destination`."""
    text = source.read_text(encoding="utf-8")
    cleaned = "\n".join(clean_lines(text.splitlines()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean hydrated DeepWiki text by removing URL-like substrings.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the hydrated text file to clean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the cleaned text.",
    )
    args = parser.parse_args()
    clean_file(args.input, args.output)


if __name__ == "__main__":
    main()
