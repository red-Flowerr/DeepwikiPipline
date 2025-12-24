#!/usr/bin/env python3
"""Fetch the DeepWiki outline for NVIDIA/Megatron-LM via MCP."""

from Deepwiki_pipline.scripts.deepwiki_mcp_client import (
    Session,
    initialize_session,
    call_tool,
    extract_text_blocks,
    delete_session,
    MCPError,
)

REPO = "NVIDIA/Megatron-LM"


def main() -> None:
    session: Session | None = None
    try:
        session = initialize_session()
        response = call_tool(
            session,
            "read_wiki_structure",
            {"repoName": REPO},
        )
        outline_chunks = extract_text_blocks(response)
        if not outline_chunks:
            raise MCPError("read_wiki_structure returned no textual content.")
        print("\n\n".join(outline_chunks))
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()