#!/usr/bin/env python3
"""Export DeepWiki sections with repository snippets hydrated in-place."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from deepwiki_pipeline import (
    MCPError,
    OutlineNode,
    PageContent,
    Session,
    call_tool,
    delete_session,
    extract_text_blocks,
    hydrate_section_text,
    initialize_session,
    normalize_heading,
)
from deepwiki_pipeline.parsing import (
    parse_outline_text,
    parse_wiki_markdown,
    resolve_page,
)

logger = logging.getLogger(__name__)


def _extract_outline_segment(text: str, repo: str) -> str:
    if not text:
        return ""
    boundary_re = re.compile(r"^Available pages for\s+(.+?):\s*$", re.MULTILINE)
    matches = list(boundary_re.finditer(text))
    if not matches:
        return text.strip()
    repo_lower = repo.lower()
    for idx, match in enumerate(matches):
        name = match.group(1).strip()
        if name.lower() != repo_lower:
            continue
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        return text[start:end].strip()
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(text)
    return text[start:end].strip()


def _select_outline_text(outline_blocks: List[str], repo: str) -> str:
    if not outline_blocks:
        return ""
    marker = f"Available pages for {repo}:"
    marker_lower = marker.lower()
    for block in outline_blocks:
        if marker_lower in block.lower():
            return _extract_outline_segment(block, repo)
    combined = "\n\n".join(outline_blocks)
    return _extract_outline_segment(combined, repo)


def _fetch_outline_text(
    session: Session,
    repo: str,
    *,
    repo_commit: Optional[str],
) -> str:
    payload = {"repoName": repo}
    if repo_commit:
        payload["repoCommit"] = repo_commit
    outline_response = call_tool(session, "read_wiki_structure", payload)
    outline_blocks = extract_text_blocks(outline_response)
    if not outline_blocks:
        raise MCPError("read_wiki_structure returned no textual content.")
    return _select_outline_text(outline_blocks, repo)


def _fetch_pages(
    session: Session,
    repo: str,
    *,
    repo_commit: Optional[str],
) -> Dict[str, PageContent]:
    payload = {"repoName": repo}
    if repo_commit:
        payload["repoCommit"] = repo_commit
    contents_response = call_tool(session, "read_wiki_contents", payload)
    contents_blocks = extract_text_blocks(contents_response)
    if not contents_blocks:
        raise MCPError("read_wiki_contents returned no textual content.")
    contents_text = "\n\n".join(contents_blocks)
    pages = parse_wiki_markdown(contents_text)
    if not pages:
        raise MCPError("read_wiki_contents response did not contain any page content.")
    return pages


def export_hydrated_sections(
    *,
    repo: str,
    repo_commit: Optional[str],
    repo_root: Path,
) -> str:
    session: Session | None = None
    try:
        session = initialize_session()
        outline_text = _fetch_outline_text(session, repo, repo_commit=repo_commit)
        outline_nodes = parse_outline_text(outline_text)
        pages = _fetch_pages(session, repo, repo_commit=repo_commit)
        seen_pages: Set[str] = set()

        output_chunks: List[str] = []
        output_chunks.append("# DeepWiki Structure")
        output_chunks.append(outline_text.strip())
        output_chunks.append("")

        def process_outline_node(node: OutlineNode) -> None:
            try:
                page = resolve_page(pages, node.title)
            except MCPError as exc:
                logger.warning("Skipping page %s: %s", node.title, exc)
                for child in node.children:
                    process_outline_node(child)
                return

            page_key = normalize_heading(page.title)
            if page_key not in seen_pages:
                seen_pages.add(page_key)

                intro_section = page.sections.get("__intro__")
                if intro_section and intro_section.text.strip():
                    intro_hydrated = hydrate_section_text(intro_section.text, repo_root=repo_root)
                    header = f"# {page.title} :: Introduction"
                    output_chunks.append(header)
                    output_chunks.append("")
                    output_chunks.append(intro_hydrated.strip())
                    output_chunks.append("")

                if page.order:
                    emitted_section = False
                    for section_key in page.order:
                        if section_key == "__intro__":
                            continue
                        section_content = page.sections[section_key]
                        if not section_content.text.strip():
                            continue
                        hydrated_text = hydrate_section_text(section_content.text, repo_root=repo_root)
                        header = f"# {page.title} :: {section_content.heading or node.title}"
                        output_chunks.append(header)
                        output_chunks.append("")
                        output_chunks.append(hydrated_text.strip())
                        output_chunks.append("")
                        emitted_section = True
                    if not emitted_section:
                        hydrated_text = hydrate_section_text(page.full_text(), repo_root=repo_root)
                        header = f"# {page.title}"
                        output_chunks.append(header)
                        output_chunks.append("")
                        output_chunks.append(hydrated_text.strip())
                        output_chunks.append("")
                else:
                    hydrated_text = hydrate_section_text(page.full_text(), repo_root=repo_root)
                    header = f"# {page.title}"
                    output_chunks.append(header)
                    output_chunks.append("")
                    output_chunks.append(hydrated_text.strip())
                    output_chunks.append("")

            for child in node.children:
                process_outline_node(child)

        for page_node in outline_nodes:
            process_outline_node(page_node)

        return "\n".join(chunk for chunk in output_chunks if chunk.strip())
    finally:
        if session is not None:
            delete_session(session)


def _run_git(args: List[str], *, cwd: Optional[Path] = None) -> None:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise MCPError(
            f"Git command failed ({' '.join(args)}): {completed.stderr.strip()}"
        )


def _clone_repo(
    repo: str,
    commit: str,
    *,
    scratch_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    if not commit:
        raise MCPError("Auto-cloning requires --repo-commit to be specified.")
    repo_url = f"https://github.com/{repo}.git"
    if scratch_dir:
        base_dir = scratch_dir.resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="deepwiki_repo_")).resolve()
    target_dir = base_dir / repo.replace("/", "_")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    logger.info("Cloning %s to %s", repo_url, target_dir)
    _run_git(["git", "clone", "--filter=blob:none", repo_url, str(target_dir)])
    logger.info("Checking out revision %s", commit)
    _run_git(["git", "checkout", "--detach", commit], cwd=target_dir)
    cleanup_root = None if scratch_dir else base_dir
    return target_dir, cleanup_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate DeepWiki sections with repository source snippets.",
    )
    parser.add_argument("repo", help="Repository identifier, e.g. volcengine/verl.")
    parser.add_argument(
        "--repo-commit",
        type=str,
        default=None,
        help="Optional commit hash or tag understood by DeepWiki.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to a local repository checkout. If omitted, the repository is cloned automatically.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Optional directory used when auto-cloning the repository.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the hydrated document.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cleanup_root: Optional[Path] = None
    try:
        if args.repo_root:
            repo_root = Path(os.path.expanduser(args.repo_root)).resolve()
            if not repo_root.exists():
                raise SystemExit(f"Repository root does not exist: {repo_root}")
            if not repo_root.is_dir():
                raise SystemExit(f"Repository root is not a directory: {repo_root}")
        else:
            scratch_dir = Path(args.scratch_dir).expanduser().resolve() if args.scratch_dir else None
            repo_root, cleanup_root = _clone_repo(
                args.repo,
                args.repo_commit,
                scratch_dir=scratch_dir,
            )

        document = export_hydrated_sections(
            repo=args.repo,
            repo_commit=args.repo_commit,
            repo_root=repo_root,
        )

        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(document, encoding="utf-8")
            logger.info("Hydrated document written to %s", destination)
        else:
            print(document)
    finally:
        if cleanup_root and cleanup_root.exists():
            shutil.rmtree(cleanup_root, ignore_errors=True)
            logger.debug("Removed temporary clone at %s", cleanup_root)


if __name__ == "__main__":
    main()
