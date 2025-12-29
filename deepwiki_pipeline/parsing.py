"""Parsing helpers for DeepWiki structure and section chunking."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .mcp import MCPError
from .models import OutlineNode, PageContent, SectionBlock, SectionContent, normalize_heading


OUTLINE_ITEM_RE = re.compile(r"^-\s+(.*)$")
OUTLINE_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)*)\s+(.*)")
PAGE_HEADER_RE = re.compile(r"# Page:\s*(.*)$")
SECTION_HEADER_RE = re.compile(r"^(#{2,6})\s+(.*)$")
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
MERMAID_RE = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
LINK_LABEL_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def parse_outline_text(markdown: str) -> List[OutlineNode]:
    nodes: List[OutlineNode] = []
    stack: List[Tuple[int, OutlineNode]] = []
    for raw_line in markdown.splitlines():
        match = OUTLINE_ITEM_RE.match(raw_line.strip())
        if not match:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        level = indent // 2
        content = match.group(1).strip()
        number_match = OUTLINE_NUMBER_RE.match(content)
        number = None
        title = content
        if number_match:
            number = number_match.group(1)
            title = number_match.group(2).strip()
        node = OutlineNode(number=number, title=title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)
        else:
            nodes.append(node)
        stack.append((level, node))
    return nodes


def parse_wiki_markdown(markdown: str) -> Dict[str, PageContent]:
    pages: Dict[str, PageContent] = {}
    current_page_key: Optional[str] = None
    current_page_title: Optional[str] = None
    current_section_heading: Optional[str] = None
    buffer: List[str] = []

    def flush_section() -> None:
        nonlocal buffer, current_page_key, current_section_heading
        if current_page_key is None:
            buffer = []
            return
        text = "\n".join(buffer).strip("\n")
        buffer = []
        key = normalize_heading(current_section_heading)
        page = pages[current_page_key]
        if key in page.sections:
            existing = page.sections[key].text
            combined = (existing + "\n\n" + text).strip("\n") if text else existing
            page.sections[key] = SectionContent(
                heading=page.sections[key].heading,
                text=combined,
            )
        else:
            page.sections[key] = SectionContent(
                heading=current_section_heading,
                text=text,
            )
            page.order.append(key)

    for raw_line in markdown.splitlines():
        page_match = PAGE_HEADER_RE.match(raw_line.strip())
        if page_match:
            flush_section()
            current_page_title = page_match.group(1).strip()
            current_page_key = normalize_heading(current_page_title)
            pages[current_page_key] = PageContent(
                title=current_page_title,
                sections={},
                order=[],
            )
            current_section_heading = None
            buffer = []
            continue
        if current_page_key is None:
            continue
        stripped_line = raw_line.strip()
        heading_match = SECTION_HEADER_RE.match(stripped_line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level <= 2:
                flush_section()
                current_section_heading = heading_text
            else:
                buffer.append(raw_line)
            continue
        duplicate_head = stripped_line.startswith("# ")
        if (
            duplicate_head
            and current_page_title
            and stripped_line[2:].strip().lower() == current_page_title.lower()
        ):
            continue
        buffer.append(raw_line)
    flush_section()
    return pages


def resolve_page(
    pages: Dict[str, PageContent],
    page_name: str,
) -> PageContent:
    normalized = normalize_heading(page_name)
    if normalized in pages:
        return pages[normalized]
    raise MCPError(
        f"Page '{page_name}' could not be resolved in the retrieved wiki content."
    )


def extract_summary_paragraph(text: str) -> str:
    cleaned = MERMAID_RE.sub("", text)
    cleaned = CODE_BLOCK_RE.sub("", cleaned)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if lower.startswith(("file:", "source:", "design intent:", "table of")):
            continue
        if lower.startswith("#"):
            continue
        return paragraph
    return paragraphs[0]


def _extract_mermaid(explanation: str) -> Tuple[str, Optional[str]]:
    match = MERMAID_RE.search(explanation)
    if not match:
        return explanation.strip(), None
    mermaid = match.group(1).strip()
    cleaned = MERMAID_RE.sub("", explanation)
    return cleaned.strip(), mermaid


def split_section_into_blocks(section_text: str) -> List[SectionBlock]:
    blocks: List[SectionBlock] = []
    last_end = 0
    matches = list(CODE_BLOCK_RE.finditer(section_text))
    for index, match in enumerate(matches, start=1):
        language = match.group(1).strip() or "text"
        code = match.group(2).strip("\n")
        if not code:
            continue
        explanation = section_text[last_end:match.start()].strip()
        explanation = explanation.strip()
        explanation, mermaid = _extract_mermaid(explanation)
        if explanation or mermaid:
            blocks.append(
                SectionBlock(
                    explanation=explanation,
                    code=code,
                    language=language.lower(),
                    mermaid=mermaid,
                )
            )
        else:
            blocks.append(
                SectionBlock(
                    explanation="",
                    code=code,
                    language=language.lower(),
                    mermaid=None,
                )
            )
        last_end = match.end()
    return blocks


def resolve_label_path_range(label: str) -> Optional[Tuple[str, int, int]]:
    text = label.strip()
    if not text:
        return None
    if "image::" in text.lower():
        return None
    if text.lower().startswith("source:"):
        text = text.split(":", 1)[1].strip()
    start = 1
    end_line = start + 79
    path = text
    range_match = re.search(r":(\d+)(?:-(\d+))?$", text)
    if range_match:
        try:
            start = int(range_match.group(1))
        except ValueError:
            start = 1
        try:
            end_line = int(range_match.group(2) or range_match.group(1))
        except ValueError:
            end_line = start
        path = text[: range_match.start()].strip()
    if not path:
        return None
    return path, start, end_line


def parse_sources_links(text: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, int, str]] = set()
    for match in LINK_LABEL_RE.finditer(text):
        label = match.group(1).strip()
        parsed = resolve_label_path_range(label)
        if not parsed:
            continue
        path, start, end_line = parsed
        key = (path, start, end_line, label)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            {
                "label": label,
                "path": path,
                "start": start,
                "end": end_line,
            }
        )
    return links
