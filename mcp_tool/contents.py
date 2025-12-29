#!/usr/bin/env python3
"""Fetch documentation content for a repository via DeepWiki MCP."""

from __future__ import annotations

import argparse

from deepwiki_pipeline import (
    MCPError,
    Session,
    call_tool,
    delete_session,
    extract_text_blocks,
    initialize_session,
    normalize_heading,
)
from deepwiki_pipeline.parsing import parse_wiki_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the read_wiki_contents MCP tool for a repository.",
    )
    parser.add_argument("repo", help="Repository identifier, e.g. volcengine/verl.")
    parser.add_argument(
        "--repo-commit",
        dest="repo_commit",
        default=None,
        help="Optional commit hash/tag supported by DeepWiki.",
    )
    parser.add_argument(
        "--page",
        default=None,
        help="Optional page heading or outline number to filter locally.",
    )
    parser.add_argument(
        "--section",
        default=None,
        help="Optional section heading within the chosen page (local filter).",
    )
    return parser.parse_args()


def _filter_markdown(markdown: str, *, page: str | None, section: str | None) -> str:
    if not page and not section:
        return markdown
    pages = parse_wiki_markdown(markdown)
    target_norm = normalize_heading(page) if page else None
    result_chunks: list[str] = []
    for key, page_content in pages.items():
        if target_norm and key != target_norm:
            continue
        if not section:
            result_chunks.append(f"# {page_content.title}\n\n{page_content.full_text()}")
            continue
        try:
            section_content = page_content.section_text(section)
        except KeyError:
            continue
        heading = section_content.heading or section
        result_chunks.append(
            f"# {page_content.title} :: {heading}\n\n{section_content.text.strip()}"
        )
    if not result_chunks:
        raise MCPError("No content matched the requested filters.")
    return "\n\n".join(result_chunks)


def main() -> None:
    args = _parse_args()
    session: Session | None = None
    try:
        session = initialize_session()
        payload = {"repoName": args.repo}
        if args.repo_commit:
            payload["repoCommit"] = args.repo_commit
        response = call_tool(session, "read_wiki_contents", payload)
        blocks = extract_text_blocks(response)
        if not blocks:
            raise MCPError("read_wiki_contents returned no textual content.")
        markdown = "\n\n".join(blocks)
        filtered = _filter_markdown(markdown, page=args.page, section=args.section)
        print(filtered)
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
