#!/usr/bin/env python3
"""Fetch the DeepWiki outline for a repository via MCP."""

from __future__ import annotations

import argparse

from deepwiki_pipeline import (
    MCPError,
    Session,
    call_tool,
    delete_session,
    extract_text_blocks,
    initialize_session,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the read_wiki_structure MCP tool for a repository.",
    )
    parser.add_argument("repo", help="Repository identifier, e.g. volcengine/verl.")
    parser.add_argument(
        "--repo-commit",
        dest="repo_commit",
        default=None,
        help="Optional commit hash/tag supported by DeepWiki.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    session: Session | None = None
    try:
        session = initialize_session()
        payload = {"repoName": args.repo}
        if args.repo_commit:
            payload["repoCommit"] = args.repo_commit
        response = call_tool(session, "read_wiki_structure", payload)
        outline_chunks = extract_text_blocks(response)
        if not outline_chunks:
            raise MCPError("read_wiki_structure returned no textual content.")
        print("\n\n".join(outline_chunks))
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
