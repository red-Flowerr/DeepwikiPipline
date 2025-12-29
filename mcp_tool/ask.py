#!/usr/bin/env python3
"""Ask a question about a repository via DeepWiki MCP."""

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
        description="Call the ask_question MCP tool for a repository.",
    )
    parser.add_argument("repo", help="Repository identifier, e.g. volcengine/verl.")
    parser.add_argument("question", help="Question to ask about the repository.")
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
        payload = {"repoName": args.repo, "question": args.question}
        if args.repo_commit:
            payload["repoCommit"] = args.repo_commit
        response = call_tool(session, "ask_question", payload)
        blocks = extract_text_blocks(response)
        if not blocks:
            raise MCPError("ask_question returned no textual content.")
        print("\n\n".join(blocks))
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
