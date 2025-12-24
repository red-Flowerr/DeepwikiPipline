#!/usr/bin/env python3
"""Fetch DeepWiki content for an arbitrary page/section."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Deepwiki_pipline.deepwiki_mcp_client import (
    MCPError,
    Session,
    build_number_lookup,
    call_tool,
    delete_session,
    extract_text_blocks,
    initialize_session,
    normalize_heading,
    parse_outline_text,
    parse_wiki_markdown,
    resolve_page,
)


def fetch_outline(repo: str, session: Session):
    resp = call_tool(session, "read_wiki_structure", {"repoName": repo})
    blocks = extract_text_blocks(resp)
    if not blocks:
        raise MCPError("read_wiki_structure returned no textual content.")
    markdown = "\n\n".join(blocks)
    nodes = parse_outline_text(markdown)
    if not nodes:
        raise MCPError("No outline entries were parsed from the structure output.")
    return nodes


def fetch_contents(repo: str, session: Session):
    resp = call_tool(session, "read_wiki_contents", {"repoName": repo})
    blocks = extract_text_blocks(resp)
    if not blocks:
        raise MCPError("read_wiki_contents returned no textual content.")
    markdown = "\n\n".join(blocks)
    return parse_wiki_markdown(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print DeepWiki content for a specific page/section."
    )
    parser.add_argument("repo", help="Repository name, e.g. NVIDIA/Megatron-LM")
    parser.add_argument(
        "--page",
        help=(
            "Page title or outline number to fetch. "
            "If omitted, the first top-level page is used."
        ),
    )
    parser.add_argument(
        "--section",
        help=(
            "Section heading within the page. "
            "If omitted, the entire page body is printed."
        ),
    )
    args = parser.parse_args()

    session: Optional[Session] = None
    try:
        session = initialize_session()
        outline_nodes = fetch_outline(args.repo, session)
        number_lookup = build_number_lookup(outline_nodes)

        target_page = args.page or outline_nodes[0].title
        pages = fetch_contents(args.repo, session)
        page = resolve_page(pages, target_page, number_lookup=number_lookup)

        if args.section:
            section = page.section_text(args.section)
            heading = section.heading or args.section
            print(f"# {page.title} :: {heading}\n")
            print(section.text.strip())
        else:
            print(f"# {page.title}\n")
            print(page.full_text())

    except MCPError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()

# 用法示例：

# # 打印第一个顶层页面
# python fetch_wiki_chunk.py NVIDIA/Megatron-LM

# # 指定大纲编号 2.1
# python fetch_wiki_chunk.py volcengine/verl --page 1.1

# # 指定页面标题并只取某个小节
# python fetch_wiki_chunk.py NVIDIA/Megatron-LM --page "Training System" --section "Monitoring and
# Diagnostics"
