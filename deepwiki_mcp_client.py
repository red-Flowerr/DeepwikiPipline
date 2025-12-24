#!/usr/bin/env python3
"""
Minimal MCP HTTP client for the DeepWiki server.

This script demonstrates how to:
1. Initialize a session
2. List available tools
3. Call the ask_question tool

Usage examples:
  python deepwiki_mcp_client.py --list-tools
  python deepwiki_mcp_client.py --ask-question \\
    vercel/next.js "What's the routing doc URL?"
"""

import argparse
import difflib
import html
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

import requests

try:  # Optional dependency for design intent rewriting
    from vllm_client import ChatMessage, VLLMError, call_vllm_chat
except ImportError:  # pragma: no cover
    ChatMessage = None  # type: ignore
    VLLMError = RuntimeError  # type: ignore
    call_vllm_chat = None  # type: ignore

MCP_ENDPOINT = "https://mcp.deepwiki.com/mcp"
PROTOCOL_VERSION = "2025-06-18"

_REQUEST_COUNTER = itertools.count(2)

logger = logging.getLogger(__name__)


def normalize_heading(text: Optional[str]) -> str:
    if text is None:
        return "__intro__"
    stripped = text.strip()
    stripped = re.sub(r"^\d+(?:\.\d+)*\s+", "", stripped)
    lowered = stripped.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", lowered)
    return normalized or "__intro__"


@dataclass
class OutlineNode:
    number: Optional[str]
    title: str
    children: List["OutlineNode"] = field(default_factory=list)

    @property
    def normalized_title(self) -> str:
        return normalize_heading(self.title)

    def to_prompt_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"title": self.title}
        if self.number:
            data["number"] = self.number
        if self.children:
            data["sections"] = [child.to_prompt_dict() for child in self.children]
        return data


@dataclass
class SectionRef:
    page: str
    section: Optional[str] = None


@dataclass
class ModulePlan:
    name: str
    responsibility: List[str]
    design_assumptions: List[str]
    role_in_architecture: str
    external_dependencies: List[str]
    sections: List[SectionRef]
    ordering_hint: List[str] = field(default_factory=list)


@dataclass
class SemanticPlan:
    repository_overview: str
    architecture_summary: List[str]
    dependency_summary: List[str]
    module_groups: List[ModulePlan]


@dataclass
class SectionContent:
    heading: Optional[str]
    text: str


@dataclass
class PageContent:
    title: str
    sections: Dict[str, SectionContent]
    order: List[str]

    def full_text(self) -> str:
        parts = [
            self.sections[key].text.strip()
            for key in self.order
            if self.sections[key].text.strip()
        ]
        return "\n\n".join(parts)

    def section_text(self, section_name: Optional[str]) -> SectionContent:
        if section_name is None:
            return SectionContent(heading=None, text=self.full_text())
        key = normalize_heading(section_name)
        if key in self.sections:
            return self.sections[key]
        for candidate_key in self.order:
            candidate = self.sections[candidate_key]
            candidate_norm = candidate_key
            if key in candidate_norm or candidate_norm in key:
                return candidate
            if candidate.heading:
                candidate_heading_norm = normalize_heading(candidate.heading)
                if key in candidate_heading_norm or candidate_heading_norm in key:
                    return candidate
        if key != "__intro__":
            candidate_keys = [k for k in self.order if k != "__intro__"]
            best_key = None
            best_score = 0.0
            for candidate_key in candidate_keys:
                candidate = self.sections[candidate_key]
                score = difflib.SequenceMatcher(a=key, b=candidate_key).ratio()
                if score > best_score:
                    best_key = candidate_key
                    best_score = score
                if candidate.heading:
                    heading_norm = normalize_heading(candidate.heading)
                    score_heading = difflib.SequenceMatcher(
                        a=key,
                        b=heading_norm,
                    ).ratio()
                    if score_heading > best_score:
                        best_key = candidate_key
                        best_score = score_heading
            if best_key and best_score >= 0.35:
                return self.sections[best_key]
        raise KeyError(
            f"Section '{section_name}' not found in page '{self.title}'."
        )


@dataclass
class DatasetChunk:
    label: str
    text: str


@dataclass
class SourceSummary:
    path: str
    start: int
    end: int
    summary: str
    label: str


@dataclass
class NarrativeDatum:
    repo: str
    module: str
    page: str
    section: Optional[str]
    summary: str
    code: str
    language: str
    file: Optional[str]
    symbol: str
    relations: Dict[str, List[str]]
    body: str
    design_intent: str
    design_sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "module": self.module,
            "page": self.page,
            "section": self.section,
            "summary": self.summary,
            "code": self.code,
            "language": self.language,
            "file": self.file,
            "symbol": self.symbol,
            "relations": self.relations,
            "body": self.body,
            "design_intent": self.design_intent,
            "design_sources": self.design_sources,
        }



@dataclass
class NarrativeLLMConfig:
    server_url: Optional[str]
    host: str
    port: int
    path: str
    model: str
    temperature: float = 0.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    destination_service: Optional[str] = None
    timeout: float = 60.0
    retries: int = 2
    retry_backoff: float = 2.0


@dataclass
class PipelineOutput:
    repo: str
    chunks: List[DatasetChunk]
    narrative_inputs: List[NarrativeDatum] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [chunk.text.strip() for chunk in self.chunks if chunk.text.strip()]
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "chunks": [
                {"label": chunk.label, "text": chunk.text}
                for chunk in self.chunks
            ],
        }

    def narrative_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.narrative_inputs]

    def generate_narratives(
        self,
        modes: Iterable[str],
    ) -> Dict[str, List[str]]:
        if not self.narrative_inputs:
            return {}
        from deepwiki_narratives import build_narratives

        return build_narratives(self.narrative_dicts(), modes)


class MCPError(RuntimeError):
    pass


def parse_sse_response(response: requests.Response) -> Dict[str, Any]:
    """
    Parse the first JSON-RPC response emitted on an SSE stream.
    """
    response.raise_for_status()
    buffer_parts: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            if buffer_parts:
                payload = "".join(buffer_parts)
                if payload:
                    try:
                        result = json.loads(payload)
                    except json.JSONDecodeError:
                        buffer_parts.clear()
                        continue
                    response.close()
                    return result
                buffer_parts.clear()
            continue
        if line.startswith("data:"):
            part = line[len("data:"):]
            if part.startswith(" "):
                part = part[1:]
            buffer_parts.append(part)
            payload = "".join(buffer_parts)
            if not payload:
                continue
            try:
                result = json.loads(payload)
            except json.JSONDecodeError:
                continue
            response.close()
            return result
        if line.startswith("event: close"):
            break
        elif line.startswith("event:"):
            # Ignore ping/close notifications; payload handled on blank line.
            continue
        else:
            buffer_parts.append(line)
    payload = "".join(buffer_parts)
    if payload:
        result = json.loads(payload)
        response.close()
        return result
    response.close()
    raise MCPError("No JSON payload received from SSE stream.")


@dataclass
class Session:
    session_id: str
    protocol_version: str = PROTOCOL_VERSION


def initialize_session(
    client_name: str = "codex-cli",
    client_version: str = "0.1",
) -> Session:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": client_name, "version": client_version},
            "capabilities": {},
        },
    }
    response = requests.post(
        MCP_ENDPOINT,
        headers=headers,
        json=payload,
        stream=True,
        timeout=30,
    )
    session_id = response.headers.get("mcp-session-id")
    if not session_id:
        raise MCPError("Server did not return an MCP session id.")
    result = parse_sse_response(response)
    if "error" in result:
        raise MCPError(f"Initialization error: {result['error']}")
    return Session(session_id=session_id)


def post_jsonrpc(
    session: Session,
    body: Dict[str, Any],
    stream: bool = False,
) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Mcp-Session-Id": session.session_id,
        "MCP-Protocol-Version": session.protocol_version,
    }
    return requests.post(
        MCP_ENDPOINT,
        headers=headers,
        json=body,
        stream=stream,
        timeout=60,
    )


def list_tools(session: Session) -> Dict[str, Any]:
    request_id = next(_REQUEST_COUNTER)
    body = {"jsonrpc": "2.0", "id": request_id, "method": "tools/list"}
    response = post_jsonrpc(session, body, stream=True)
    return parse_sse_response(response)


def call_tool(
    session: Session,
    tool: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    request_id = next(_REQUEST_COUNTER)
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    response = post_jsonrpc(session, body, stream=True)
    return parse_sse_response(response)


def delete_session(session: Session) -> None:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Mcp-Session-Id": session.session_id,
        "MCP-Protocol-Version": session.protocol_version,
    }
    try:
        requests.delete(MCP_ENDPOINT, headers=headers, timeout=5)
    except requests.RequestException:
        pass


def extract_text_blocks(payload: Dict[str, Any]) -> List[str]:
    content = payload.get("result", {}).get("content", [])
    blocks = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return [block for block in blocks if block]


def parse_outline_text(markdown: str) -> List[OutlineNode]:
    lines = markdown.splitlines()
    nodes: List[OutlineNode] = []
    stack: List[tuple[int, OutlineNode]] = []
    for raw_line in lines:
        if not raw_line.strip().startswith("-"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        level = indent // 2
        stripped = raw_line.strip()
        content = stripped[1:].strip()
        number = None
        match = re.match(r"(\d+(?:\.\d+)*)\s+(.*)", content)
        title = content
        if match:
            number = match.group(1)
            title = match.group(2).strip()
        node = OutlineNode(number=number, title=title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)
        else:
            nodes.append(node)
        stack.append((level, node))
    return nodes


def build_number_lookup(nodes: List[OutlineNode]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}

    def register(number: Optional[str], title: str) -> None:
        if not number:
            return
        stripped = number.strip()
        if stripped:
            lookup[stripped] = title
            lookup[normalize_heading(stripped)] = title

    def walk(node: OutlineNode) -> None:
        register(node.number, node.title)
        for child in node.children:
            walk(child)

    for root in nodes:
        walk(root)
    return lookup


def parse_relevant_source_files(text: str) -> List[str]:
    files = []
    for match in FILE_LINK_RE.finditer(text):
        label = match.group(1).strip()
        if label:
            files.append(label)
    return files


def parse_sources_links(text: str) -> List[Dict[str, Any]]:
    """
    Extract entries like [path/to/file.py:34-120]() or [README.md]() from the section text.
    Returns list of dicts with repo path and inferred line ranges.
    """
    links: List[Dict[str, Any]] = []
    pattern = re.compile(r"\[([^\]]+)\]\([^)]*\)")
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        path = label
        start = 1
        end_line = start
        if ":" in label:
            base, rng = label.rsplit(":", 1)
            base = base.strip()
            rng = rng.strip()
            if rng:
                if "-" in rng:
                    start_str, end_str = rng.split("-", 1)
                else:
                    start_str = rng
                    end_str = rng
                try:
                    start = int(start_str)
                except ValueError:
                    start = 1
                try:
                    end_line = int(end_str)
                except ValueError:
                    end_line = start
            path = base
        else:
            # No explicit range; default to first 80 lines.
            path = path.strip()
            start = 1
            end_line = start + 79
        if path.lower().startswith("source:"):
            path = path.split(":", 1)[1].strip() or path
        links.append(
            {
                "label": label,
                "path": path,
                "start": start,
                "end": end_line,
            }
        )
    return links


def format_source_label(path: str, start: int, end: int) -> str:
    if start <= 0 and end <= 0:
        return path
    if end <= start:
        return f"{path}:{start}"
    return f"{path}:{start}-{end}"


def strip_markup(text: str) -> str:
    cleaned = text.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"^[#>*\-\s]+", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return html.unescape(cleaned).strip()


def is_probable_source_reference(path: str, label: str) -> bool:
    normalized = path.strip()
    if not normalized:
        return False
    if label.startswith("!"):
        return False
    lower_label = label.lower()
    lower_path = normalized.lower()
    if "://" in normalized or lower_path.startswith(("http", "#")):
        return False
    if lower_label.startswith(("http", "diagram", "table of contents", "see also", "blog", "paper")):
        return False
    if lower_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
        return False
    if "source" in lower_label:
        return True
    if any(ch in normalized for ch in ("/", "\\", ".")):
        return True
    if normalized.upper() in {"README", "LICENSE", "COPYING", "CHANGES", "CHANGELOG", "MAKEFILE"}:
        return True
    return False


def summarise_snippet_lines(
    lines: List[str],
    language: str,
    max_lines: int = 12,
    max_chars: int = 180,
) -> str:
    collected: List[str] = []
    for raw_line in lines[:max_lines]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        cleaned = stripped
        if language in {"python", "bash", "shell"} and cleaned.startswith("#"):
            cleaned = cleaned.lstrip("#").strip()
        elif language in {"cpp", "c", "java", "javascript", "typescript", "go"}:
            if cleaned.startswith("//"):
                cleaned = cleaned.lstrip("/").strip()
            elif cleaned.startswith("/*") or cleaned.startswith("*"):
                cleaned = cleaned.lstrip("/*").strip()
        elif language in {"yaml", "toml"} and cleaned.startswith("#"):
            cleaned = cleaned.lstrip("#").strip()
        elif language == "markdown":
            cleaned = strip_markup(cleaned)
        else:
            cleaned = strip_markup(cleaned)
        if not cleaned:
            continue
        collected.append(cleaned)
        if len(collected) >= 3:
            break
    if not collected and lines:
        collected.append(strip_markup(lines[0].strip()))
    if not collected:
        return ""
    summary = " ".join(collected)
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def build_source_summaries(
    repo: str,
    links: Iterable[Dict[str, Any]],
) -> List[SourceSummary]:
    summaries: Dict[Tuple[str, int, int], SourceSummary] = {}
    for link in links:
        path = (link.get("path") or "").strip()
        label = (link.get("label") or path).strip()
        if not path:
            continue
        if not is_probable_source_reference(path, label):
            continue
        start = int(link.get("start") or 1)
        end = int(link.get("end") or start)
        if start < 1:
            start = 1
        if end < start:
            end = start
        cache_key = (repo, path, REPO_COMMIT_OVERRIDE.get(repo), start, end)
        if cache_key in SNIPPET_SUMMARY_CACHE:
            summary_text = SNIPPET_SUMMARY_CACHE[cache_key]
        else:
            lines = fetch_raw_lines(repo, path)
            summary_text = ""
            if lines:
                start_idx = max(start - 1, 0)
                end_idx = min(end, len(lines))
                slice_lines = lines[start_idx:end_idx]
                language = guess_language_from_path(path)
                summary_text = summarise_snippet_lines(slice_lines, language)
            SNIPPET_SUMMARY_CACHE[cache_key] = summary_text
        key = (path, start, end)
        if key not in summaries:
            summaries[key] = SourceSummary(
                path=path,
                start=start,
                end=end,
                summary=summary_text,
                label=label,
            )
    return list(summaries.values())


def replace_relevant_details_block(
    section_text: str,
    summaries: List[SourceSummary],
) -> str:
    if not summaries:
        return section_text

    def replacement(_: re.Match[str]) -> str:
        lines: List[str] = [
            "<details>",
            "<summary>Relevant source files</summary>",
            "",
            "Design Summary:",
        ]
        for entry in summaries:
            label = format_source_label(entry.path, entry.start, entry.end)
            description = entry.summary or "Referenced in section narrative below."
            lines.append(f"- {label} — {description}")
        lines.append("")
        lines.append("</details>")
        return "\n".join(lines)

    return RELEVANT_DETAILS_RE.sub(replacement, section_text)


def build_snippet_map(
    repo: str,
    links: Iterable[Dict[str, Any]],
    max_lines: int = 80,
) -> Dict[str, Tuple[str, Tuple[str, int, int]]]:
    
    snippets: Dict[str, Tuple[str, Tuple[str, int, int]]] = {}
    for link in links:
        label = link.get("label")
        path = link.get("path")
        start = int(link.get("start", 1))
        end = int(link.get("end", start))
        if not path or not label:
            continue
        if path.endswith("/"):
            continue
        lines = fetch_raw_lines(repo, path)
        if not lines:
            continue
        start_idx = max(start - 1, 0)
        end_idx = min(end, len(lines))
        slice_lines = lines[start_idx:end_idx]
        if max_lines > 0:
            slice_lines = slice_lines[:max_lines]
        snippet_text = "\n".join(slice_lines).rstrip()
        if not snippet_text:
            continue
        language = guess_language_from_path(path)
        caption = f"[Source: {path}:{start}-{end}]"
        code_block = f"```{language}\n{snippet_text}\n```"
        snippets[label] = (f"{caption}\n{code_block}", (path, start, end))
    return snippets


def replace_snippet_markers(
    text: str,
    snippet_map: Dict[str, Tuple[str, Tuple[str, int, int]]],
    append_unused: bool,
    context: str,
) -> Tuple[str, List[str], set[str]]:
    if not text or not snippet_map:
        return text, [], set()
    pattern = re.compile(r"\[([^\]]+)\]\([^)]*\)")
    matches = [m.group(1).strip() for m in pattern.finditer(text)]
    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        entry = snippet_map.get(label)
        if entry:
            snippet, _ = entry
            used.add(label)
            return snippet

        return match.group(0)

    had_placeholders = bool(matches)
    updated_text = pattern.sub(replace, text)
    appended_blocks: List[str] = []
    if append_unused:
        leftover_labels = [label for label in snippet_map if label not in used]
        if leftover_labels:
            appended_blocks = [snippet_map[label][0] for label in leftover_labels]
            updated_text = updated_text.rstrip()
            if updated_text:
                updated_text += "\n\n"
            updated_text += "\n\n".join(appended_blocks)
    elif had_placeholders and not used:
        logger.debug(
            "%s: placeholders detected but no snippets replaced; leaving original markers.",
            context,
        )
    return updated_text, appended_blocks, used


def collect_outline_sections(node: OutlineNode) -> List[OutlineNode]:
    return list(node.children)


def derive_section_plan(
    outline_nodes: List[OutlineNode],
    pages: Dict[str, PageContent],
    repo_name: str,
    number_lookup: Dict[str, str],
) -> SemanticPlan:
    if not outline_nodes:
        raise MCPError("Outline nodes are required to derive a section plan.")

    # Repository overview = first page summary if available.
    repository_overview = ""
    first_page = None
    try:
        first_page = resolve_page(pages, outline_nodes[0].title, number_lookup=number_lookup)
    except MCPError:
        first_page = None
    if first_page:
        repository_overview = extract_summary_paragraph(first_page.full_text())

    # Architecture summary: headline statement per top-level node.
    architecture_summary: List[str] = []
    for node in outline_nodes:
        try:
            page = resolve_page(pages, node.title, number_lookup=number_lookup)
        except MCPError:
            continue
        highlight = extract_summary_paragraph(page.full_text())
        text = (
            f"{node.title}: {highlight}"
            if highlight
            else f"{node.title}: foundational context for {repo_name}."
        )
        architecture_summary.append(text)

    # Dependency summary: simple sequential guidance.
    dependency_summary: List[str] = []
    for idx, node in enumerate(outline_nodes):
        if idx == 0:
            dependency_summary.append(
                f"Start with {node.title} to establish the baseline concepts."
            )
        else:
            prev = outline_nodes[idx - 1].title
            dependency_summary.append(
                f"After exploring {prev}, continue with {node.title}."
            )

    module_groups: List[ModulePlan] = []
    for node in outline_nodes:
        try:
            page = resolve_page(pages, node.title, number_lookup=number_lookup)
        except MCPError:
            continue

        intro_content = page.sections.get("__intro__")
        intro_text = intro_content.text.strip() if intro_content and intro_content.text.strip() else ""

        section_nodes = collect_outline_sections(node)
        numbered_children = [child for child in section_nodes if child.number]
        target_sections = numbered_children if numbered_children else section_nodes
        if intro_text:
            summary = extract_summary_paragraph(intro_text)
            responsibility = [summary] if summary else [f"This section explains {page.title} within {repo_name}."]
            external_dependencies = parse_relevant_source_files(intro_text) or [page.title]
            ordering_hint = [target_sections[0].title] if target_sections else []
            module_groups.append(
                ModulePlan(
                    name=f"{node.number + ' ' if node.number else ''}{page.title} :: Overview",
                    responsibility=responsibility,
                    design_assumptions=["Familiarise yourself with the repository overview."],
                    role_in_architecture=f"{page.title} introduces the concepts used throughout the tutorial.",
                    external_dependencies=external_dependencies,
                    sections=[SectionRef(page=page.title, section=None)],
                    ordering_hint=ordering_hint,
                )
            )

        if not target_sections:
            # No explicit sections; treat entire page as a single module (if not covered by intro).
            if not intro_text:  # avoid duplicate
                summary = extract_summary_paragraph(page.full_text())
                responsibility = [summary] if summary else [f"This section explains {page.title} within {repo_name}."]
                external_dependencies = parse_relevant_source_files(page.full_text()) or [page.title]
                module_groups.append(
                    ModulePlan(
                        name=f"{node.number + ' ' if node.number else ''}{page.title}",
                        responsibility=responsibility,
                        design_assumptions=["Ensure you have set up the project environment."],
                        role_in_architecture=f"{page.title} provides standalone guidance within the repository.",
                        external_dependencies=external_dependencies,
                        sections=[SectionRef(page=page.title, section=None)],
                        ordering_hint=[],
                    )
                )
            continue

        for idx, section_node in enumerate(target_sections):
            heading_candidate = section_node.title
            section_label = heading_candidate
            try:
                section_content = page.section_text(heading_candidate)
            except KeyError:
                if section_node.number:
                    combined_heading = f"{section_node.number} {heading_candidate}"
                    try:
                        section_content = page.section_text(combined_heading)
                        section_label = combined_heading
                    except KeyError:
                        logger.debug(
                            "Skipping section '%s' (%s) — no matching content found.",
                            heading_candidate,
                            page.title,
                        )
                        continue
                else:
                    logger.debug(
                        "Skipping section '%s' (%s) — no matching content found.",
                        heading_candidate,
                        page.title,
                    )
                    continue

            responsibility = []
            design_summary = extract_summary_paragraph(section_content.text)
            if design_summary:
                responsibility.append(design_summary)
            else:
                responsibility.append(f"This section explains {heading_candidate} within {repo_name}.")

            prerequisites: List[str] = []
            if idx > 0 or intro_text:
                prerequisites.append("Review the preceding sections of this page.")
            else:
                prerequisites.append("Ensure you understand the repository overview.")

            if idx + 1 < len(target_sections):
                next_heading = target_sections[idx + 1].title
                role_in_architecture = (
                    f"This section prepares you for {next_heading} within {page.title}."
                )
            else:
                role_in_architecture = (
                    f"This section completes {page.title} before exploring other topics."
                )

            external_dependencies = parse_relevant_source_files(section_content.text)
            if not external_dependencies:
                external_dependencies = [page.title]

            module_name_parts = []
            if section_node.number:
                module_name_parts.append(section_node.number)
            module_name_parts.append(heading_candidate)
            module_name = " ".join(module_name_parts)

            ordering_hint = []
            if idx + 1 < len(target_sections):
                next_child = target_sections[idx + 1]
                ordering_hint.append(
                    f"{next_child.number or ''} {next_child.title}".strip()
                )
            elif node.children:
                # If this is the last subsection, hint at the next top-level page.
                current_index = outline_nodes.index(node)
                if current_index + 1 < len(outline_nodes):
                    ordering_hint.append(outline_nodes[current_index + 1].title)

            module_groups.append(
                ModulePlan(
                    name=f"{page.title} :: {module_name}",
                    responsibility=responsibility,
                    design_assumptions=prerequisites,
                    role_in_architecture=role_in_architecture,
                    external_dependencies=external_dependencies,
                    sections=[
                        SectionRef(
                            page=page.title,
                            section=section_content.heading or section_label,
                        )
                    ],
                    ordering_hint=ordering_hint,
                )
            )

    return SemanticPlan(
        repository_overview=repository_overview,
        architecture_summary=architecture_summary,
        dependency_summary=dependency_summary,
        module_groups=module_groups,
    )


def coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                result.extend(
                    line.strip()
                    for line in item.splitlines()
                    if line.strip()
                )
            elif item is not None:
                result.append(str(item).strip())
        return result
    return [str(value).strip()]


def parse_semantic_plan_data(data: Dict[str, Any]) -> SemanticPlan:
    repository_overview = coerce_str(data.get("repository_overview"))
    architecture_summary = coerce_str_list(data.get("architecture_summary"))
    dependency_summary = coerce_str_list(data.get("dependency_summary"))
    modules_raw = data.get("module_groups")
    if not isinstance(modules_raw, list):
        raise MCPError("Expected 'module_groups' to be a list in LLM response.")
    module_groups: List[ModulePlan] = []
    for entry in modules_raw:
        if not isinstance(entry, dict):
            continue
        name = coerce_str(entry.get("name")) or "Unnamed Module"
        responsibility = coerce_str_list(entry.get("responsibility"))
        design_assumptions = coerce_str_list(entry.get("design_assumptions"))
        role_in_architecture = coerce_str(entry.get("role_in_architecture"))
        external_dependencies = coerce_str_list(
            entry.get("external_dependencies")
        )
        ordering_hint = coerce_str_list(entry.get("ordering_hint"))
        sections_raw = entry.get("sections", [])
        section_refs: List[SectionRef] = []
        if not isinstance(sections_raw, list):
            raise MCPError(
                "Expected 'sections' to be a list for module planning."
            )
        for raw_ref in sections_raw:
            page = None
            section = None
            if isinstance(raw_ref, dict):
                page = coerce_str(raw_ref.get("page") or raw_ref.get("title"))
                section_value = raw_ref.get("section")
                section = coerce_str(section_value) if section_value is not None else None
            elif isinstance(raw_ref, (list, tuple)) and raw_ref:
                page = coerce_str(raw_ref[0])
                if len(raw_ref) > 1:
                    section_value = raw_ref[1]
                    section = (
                        coerce_str(section_value)
                        if section_value is not None
                        else None
                    )
            elif isinstance(raw_ref, str):
                page = coerce_str(raw_ref)
            if page:
                if section == "":
                    section = None
                section_refs.append(SectionRef(page=page, section=section or None))
        if not section_refs:
            raise MCPError(
                f"Module '{name}' does not reference any wiki sections."
            )
        module_groups.append(
            ModulePlan(
                name=name,
                responsibility=responsibility,
                design_assumptions=design_assumptions,
                role_in_architecture=role_in_architecture,
                external_dependencies=external_dependencies,
                sections=section_refs,
                ordering_hint=ordering_hint,
            )
        )
    if not module_groups:
        raise MCPError("LLM did not return any module groups.")
    return SemanticPlan(
        repository_overview=repository_overview,
        architecture_summary=architecture_summary,
        dependency_summary=dependency_summary,
        module_groups=module_groups,
    )


def extract_json_from_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```json\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic_fenced = re.search(r"```\s*(.*?)```", stripped, re.DOTALL)
    if generic_fenced:
        return generic_fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


DETAILS_RE = re.compile(r"<details.*?>.*?</details>", re.IGNORECASE | re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
FILE_LINK_RE = re.compile(r"- \[([^\]]+)\]\([^)]+\)")
CITE_RE = re.compile(
    r"<cite\s+repo=\"([^\"]+)\"\s+path=\"([^\"]+)\"\s+start=\"(\d+)\"\s+end=\"(\d+)\"\s*/?>",
    re.IGNORECASE,
)
RELEVANT_DETAILS_RE = re.compile(
    r"<details>\s*<summary>\s*Relevant source files\s*</summary>(.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)

RAW_SNIPPET_CACHE: Dict[Tuple[str, str, Optional[str]], str] = {}
RAW_LINE_CACHE: Dict[Tuple[str, str, Optional[str]], List[str]] = {}
SNIPPET_SUMMARY_CACHE: Dict[Tuple[str, str, Optional[str], int, int], str] = {}
REPO_LOCAL_ROOTS: Dict[str, Path] = {}
REPO_COMMIT_OVERRIDE: Dict[str, str] = {}


def strip_details_blocks(text: str) -> str:
    return DETAILS_RE.sub("", text)


def extract_first_code_block(text: str) -> tuple[str, str]:
    match = CODE_BLOCK_RE.search(text)
    if match:
        language = match.group(1) or "text"
        code = match.group(2).strip()
        return code, language
    return "", "text"


def extract_first_file_reference(text: str) -> Optional[str]:
    match = FILE_LINK_RE.search(text)
    return match.group(1) if match else None


def extract_citations(text: str) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for repo, path, start, end in CITE_RE.findall(text):
        try:
            start_line = int(start)
            end_line = int(end)
        except ValueError:
            start_line = 1
            end_line = start_line
        citations.append(
            {
                "repo": repo,
                "path": path,
                "start": start_line,
                "end": end_line,
            }
        )
    return citations


def guess_language_from_path(path: Optional[str]) -> str:
    if not path:
        return "text"
    extension = Path(path).suffix.lower()
    mapping = {
        ".py": "python",
        ".pt": "python",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cu": "cuda",
        ".h": "c",
        ".hpp": "cpp",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(extension, "text")


def fetch_raw_file(repo: str, path: str) -> Optional[str]:
    commit_override = REPO_COMMIT_OVERRIDE.get(repo)
    cache_key = (repo, path, commit_override)
    if cache_key in RAW_SNIPPET_CACHE:
        return RAW_SNIPPET_CACHE[cache_key]
    local_root = REPO_LOCAL_ROOTS.get(repo)
    if local_root:
        candidate = local_root / path
        if candidate.exists():
            if candidate.is_dir():
                return None
            text = candidate.read_text(encoding="utf-8")
            RAW_SNIPPET_CACHE[cache_key] = text
            return text
    branches: List[str]
    if commit_override:
        branches = [commit_override]
    else:
        branches = ["main", "master"]
    for branch in branches:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            RAW_SNIPPET_CACHE[cache_key] = response.text
            return response.text
    return None


def fetch_raw_lines(repo: str, path: str) -> Optional[List[str]]:
    commit_override = REPO_COMMIT_OVERRIDE.get(repo)
    cache_key = (repo, path, commit_override)
    if cache_key in RAW_LINE_CACHE:
        return RAW_LINE_CACHE[cache_key]
    text = fetch_raw_file(repo, path)
    if text is None:
        return None
    lines = text.splitlines()
    RAW_LINE_CACHE[cache_key] = lines
    return lines


def clone_repository(repo: str, commit: Optional[str]) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="deepwiki_repo_"))
    url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", url, str(temp_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if commit:
            subprocess.run(
                ["git", "checkout", commit],
                cwd=temp_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise MCPError(f"Failed to clone repository {repo}: {exc}") from exc
    return temp_dir


def extract_citation_snippet(citation: Dict[str, Any], max_lines: int = 120) -> Optional[str]:
    repo = citation.get("repo")
    path = citation.get("path")
    start = citation.get("start", 1)
    end = citation.get("end", start)
    if not repo or not path:
        return None
    raw_text = fetch_raw_file(repo, path)
    if raw_text is None:
        return None
    lines = raw_text.splitlines()
    start_idx = max(start - 1, 0)
    end_idx = min(end, len(lines))
    snippet_lines = lines[start_idx:end_idx]
    if max_lines > 0:
        snippet_lines = snippet_lines[:max_lines]
    snippet = "\n".join(snippet_lines).strip()
    return snippet if snippet else None

def extract_summary_paragraph(text: str) -> str:
    without_details = strip_details_blocks(text)
    without_code = CODE_BLOCK_RE.sub("", without_details)
    cleaned_lines: List[str] = []
    for line in without_code.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            continue
        if stripped.startswith("- ") and "](" in stripped:
            continue
        if stripped.startswith("* ") and "](" in stripped:
            continue
        if stripped.lower().startswith("sources"):
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    cleaned_text = "\n".join(cleaned_lines)
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if lower.startswith("the following files were used as context"):
            continue
        lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
        if lines and all(ln.startswith("- ") or ln.startswith("* ") for ln in lines):
            continue
        return paragraph
    return paragraphs[0] if paragraphs else ""


def ask_question_json(
    session: Session,
    repo: str,
    prompt: str,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_payload: str = ""
    for attempt in range(retries + 1):
        result = call_tool(
            session,
            "ask_question",
            {"repoName": repo, "question": prompt},
        )
        blocks = extract_text_blocks(result)
        if not blocks:
            last_error = MCPError("ask_question returned no textual content.")
            continue
        combined = "\n".join(blocks)
        payload = extract_json_from_text(combined)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
            last_payload = payload
    snippet = (last_payload[:500] + "...") if len(last_payload) > 500 else last_payload
    raise MCPError(
        "ask_question response was not valid JSON after multiple attempts. "
        f"Last error: {last_error}.\nPartial payload:\n{snippet}"
    ) from last_error


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
        stripped = raw_line.strip()
        if stripped.startswith("# Page:"):
            flush_section()
            current_page_title = stripped[len("# Page:") :].strip()
            current_page_key = normalize_heading(current_page_title)
            pages[current_page_key] = PageContent(
                title=current_page_title,
                sections={},
                order=[],
            )
            current_section_heading = None
            buffer = []
            continue
        heading_match = re.match(r"^(#{2,6})\s+(.*)$", stripped)
        if heading_match and current_page_key is not None:
            flush_section()
            current_section_heading = heading_match.group(2).strip()
            continue
        if (
            stripped.startswith("# ")
            and current_page_title
            and stripped[2:].strip().lower() == current_page_title.lower()
        ):
            # Skip duplicate top-level heading that mirrors the page title.
            continue
        if current_page_key is None:
            continue
        buffer.append(raw_line)
    flush_section()
    return pages


def resolve_page(
    pages: Dict[str, PageContent],
    page_name: str,
    number_lookup: Optional[Dict[str, str]] = None,
) -> PageContent:
    lookup_name = page_name
    if number_lookup:
        normalized_key = normalize_heading(page_name)
        if page_name in number_lookup:
            lookup_name = number_lookup[page_name]
        elif normalized_key in number_lookup:
            lookup_name = number_lookup[normalized_key]
    key_candidate = normalize_heading(lookup_name)
    key = normalize_heading(page_name)
    if key_candidate in pages:
        return pages[key_candidate]
    if key in pages:
        return pages[key]
    # Direct containment checks
    for candidate_key, candidate_page in pages.items():
        if key_candidate in candidate_key or candidate_key in key_candidate:
            return candidate_page
        candidate_norm = normalize_heading(candidate_page.title)
        if key_candidate in candidate_norm or candidate_norm in key_candidate:
            return candidate_page
    # Fuzzy fallback
    best_key = None
    best_score = 0.0
    for candidate_key, candidate_page in pages.items():
        score = difflib.SequenceMatcher(a=key_candidate, b=candidate_key).ratio()
        if score > best_score:
            best_key = candidate_key
            best_score = score
        title_norm = normalize_heading(candidate_page.title)
        score_title = difflib.SequenceMatcher(a=key_candidate, b=title_norm).ratio()
        if score_title > best_score:
            best_key = candidate_key
            best_score = score_title
    if best_key and best_score >= 0.35:
        return pages[best_key]
    raise MCPError(
        f"Module references page '{page_name}' which could not be resolved."
    )


def build_repo_chunk_text(plan: SemanticPlan) -> DatasetChunk:
    lines: List[str] = []
    if plan.repository_overview:
        lines.append("[Repository Overview]")
        lines.append(plan.repository_overview.strip())
        lines.append("")
    if plan.architecture_summary:
        lines.append("[Architecture Summary]")
        for item in plan.architecture_summary:
            lines.append(f"- {item}")
        lines.append("")
    if plan.dependency_summary:
        lines.append("[Dependency Explanation]")
        for item in plan.dependency_summary:
            lines.append(f"- {item}")
        lines.append("")
    text = "\n".join(line for line in lines if line is not None).strip()
    return DatasetChunk(label="repo-overview", text=text)


def make_narrative_datum(
    repo: str,
    module: ModulePlan,
    page: PageContent,
    section_heading: Optional[str],
    section_text: str,
    design_intent: str,
    design_sources: Optional[List[str]] = None,
) -> NarrativeDatum:
    summary = extract_summary_paragraph(section_text)
    if not summary:
        focus = section_heading or module.name or page.title
        summary = f"This section documents {focus} in {repo}."
    code, language = extract_first_code_block(section_text)
    citations = extract_citations(section_text)
    file_ref = extract_first_file_reference(section_text)
    if not file_ref and citations:
        file_ref = citations[0].get("path")
    if not code and citations:
        snippet = extract_citation_snippet(citations[0])
        if snippet:
            code = snippet
            language = guess_language_from_path(citations[0].get("path"))
    used_by = [
        hint.strip()
        for hint in (module.ordering_hint or [])
        if isinstance(hint, str) and hint.strip()
    ]
    relations = {
        "used_by": used_by,
        "sources": [
            f"{cite['path']}:{cite['start']}-{cite['end']}"
            for cite in citations
        ],
    }
    if not file_ref:
        file_ref = f"{page.title}::{section_heading}" if section_heading else page.title
    return NarrativeDatum(
        repo=repo,
        module=module.name,
        page=page.title,
        section=section_heading,
        summary=summary,
        code=code,
        language=language or "text",
        file=file_ref,
        symbol=section_heading or module.name,
        relations=relations,
        body=section_text.strip(),
        design_intent=design_intent,
        design_sources=design_sources or [],
    )


def generate_design_intent_text(
    repo: str,
    module_name: str,
    page_title: str,
    section_heading: Optional[str],
    section_text: str,
    external_files: List[str],
    code_block: Optional[str],
    code_language: Optional[str],
    llm_config: Optional[NarrativeLLMConfig],
    sources_links: Optional[List[Dict[str, Any]]] = None,
    repo_full_name: Optional[str] = None,
    snippet_map: Optional[Dict[str, Tuple[str, Tuple[str, int, int]]]] = None,
) -> Tuple[str, List[str], set[str]]:
    base_intent = extract_summary_paragraph(section_text)
    if not base_intent:
        focus = section_heading or module_name or page_title
        base_intent = f"This section explains why {focus} is implemented in {repo}."
    sources_text = (
        "\n".join(f"- {item}" for item in external_files if item)
        if external_files
        else "(no explicit source files listed)"
    )
    clean_section = section_text.strip()
    heading_label = section_heading or "Overview"
    code_excerpt = code_block.strip() if code_block else ""
    if code_excerpt:
        lang = code_language or (guess_language_from_path(external_files[0]) if external_files else "text")
        code_section = f"```{lang}\n{code_excerpt}\n```"
    else:
        code_section = "(no code snippet provided)"

    intent_text = base_intent
    use_llm = bool(llm_config and call_vllm_chat and ChatMessage)
    if use_llm:
        system_prompt = (
            "You are an expert software architect documenting design rationale for a codebase. "
            "Given raw documentation, rewrite it as a concise explanation of *why* the system "
            "is designed this way. Focus on intent, trade-offs, and constraints. "
            "Do not restate implementation details unless they illustrate the rationale."
        )
        user_prompt = (
            f"Repository: {repo}\n"
            f"Module: {module_name}\n"
            f"Page: {page_title}\n"
            f"Section: {heading_label}\n\n"
            "Original documentation:\n"
            f"{clean_section}\n\n"
            f"Referenced source files:\n{sources_text}\n\n"
            f"Representative code (may be empty):\n{code_section}\n\n"
            "Rewrite the documentation as 2-3 sentences focusing on the design intent and "
            "reasoning behind the approach. Avoid phrases like 'This section' or 'This component', "
            "and instead explain the motivation directly."
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        try:
            response = call_vllm_chat(
                host=llm_config.host,
                port=llm_config.port,
                path=llm_config.path,
                model=llm_config.model,
                messages=messages,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                server_url=llm_config.server_url,
                api_key=llm_config.api_key,
                destination_service=llm_config.destination_service,
                timeout=llm_config.timeout,
                retries=llm_config.retries,
                retry_backoff=llm_config.retry_backoff,
            )
        except VLLMError as exc:
            logger.warning(
                "Design intent rewrite failed for '%s' (%s :: %s): %s",
                module_name,
                page_title,
                heading_label,
                exc,
            )
            intent_text = base_intent
        else:
            cleaned = response.strip()
            intent_text = cleaned if cleaned else base_intent

    snippet_map = snippet_map or {}
    if not snippet_map and repo_full_name and sources_links:
        snippet_map = build_snippet_map(repo_full_name, sources_links)
    appended_snippets: List[str] = []
    used_labels: set[str] = set()
    if snippet_map:
        intent_text, appended_snippets, used_labels = replace_snippet_markers(
            intent_text,
            snippet_map,
            append_unused=True,
            context=f"{module_name} :: {page_title}",
        )

    return intent_text, appended_snippets, used_labels


def build_module_chunk_text(
    module: ModulePlan,
    index: int,
    pages: Dict[str, PageContent],
    repo: str,
    number_lookup: Optional[Dict[str, str]] = None,
    design_llm_config: Optional[NarrativeLLMConfig] = None,
) -> tuple[DatasetChunk, List[NarrativeDatum]]:
    lines: List[str] = []
    lines.append(f"[Module Group {index}]")
    lines.append(f"[Module: {module.name}]")

    def append_list_section(title: str, items: List[str]) -> None:
        if not items:
            return
        lines.append(f"{title}:")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    if module.role_in_architecture:
        lines.append("Role in Architecture:")
        lines.append(module.role_in_architecture)
        lines.append("")
    append_list_section("External Dependencies", module.external_dependencies)
    if module.ordering_hint:
        append_list_section("Ordering Hint", module.ordering_hint)

    narrative_data: List[NarrativeDatum] = []
    design_intents: List[str] = []
    section_blocks: List[str] = []
    design_summary_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for ref in module.sections:
        page = resolve_page(pages, ref.page, number_lookup=number_lookup)
        try:
            section_name = ref.section
            if section_name and normalize_heading(section_name) == normalize_heading(page.title):
                section = page.section_text(None)
            else:
                section = page.section_text(section_name)
        except KeyError as exc:
            raise MCPError(
                f"Module '{module.name}' could not resolve section '{ref.section}' in page '{page.title}'."
            ) from exc
        heading_label = section.heading or "Overview"
        code_block, code_lang = extract_first_code_block(section.text)
        citations = extract_citations(section.text)
        source_links = parse_sources_links(section.text)
        section_summaries = (
            build_source_summaries(repo, source_links) if source_links else []
        )
        if not section_summaries and RELEVANT_DETAILS_RE.search(section.text):
            fallback_entries: List[SourceSummary] = []
            seen_paths: set[str] = set()
            for label in parse_relevant_source_files(section.text):
                cleaned_label = label.strip()
                if cleaned_label.lower().startswith("source:"):
                    cleaned_label = cleaned_label.split(":", 1)[1].strip()
                if ":" in cleaned_label:
                    cleaned_path = cleaned_label.split(":", 1)[0].strip()
                else:
                    cleaned_path = cleaned_label
                if not cleaned_path or cleaned_path in seen_paths:
                    continue
                seen_paths.add(cleaned_path)
                fallback_entries.append(
                    SourceSummary(
                        path=cleaned_path,
                        start=0,
                        end=0,
                        summary="",
                        label=label,
                    )
                )
            section_summaries = fallback_entries
        snippet_map = build_snippet_map(repo, source_links) if source_links else {}
        if not code_block and citations:
            snippet = extract_citation_snippet(citations[0])
            if snippet:
                code_block = snippet
                code_lang = guess_language_from_path(citations[0].get("path"))

        design_text, appended_snippets, _ = generate_design_intent_text(
            repo=repo,
            module_name=module.name,
            page_title=page.title,
            section_heading=section.heading,
            section_text=section.text,
            external_files=module.external_dependencies,
            code_block=code_block,
            code_language=code_lang,
            llm_config=design_llm_config,
            sources_links=source_links,
            repo_full_name=repo,
            snippet_map=snippet_map,
        )
        if design_text:
            design_intents.append(design_text)

        transformed_section_text = replace_relevant_details_block(
            section.text,
            section_summaries,
        )
        section_context = f"{page.title} :: {heading_label}"
        for entry in section_summaries:
            key = (entry.path, entry.start, entry.end)
            record = design_summary_map.setdefault(
                key,
                {"summary": entry.summary, "sections": set()},
            )
            if entry.summary and (
                not record["summary"] or len(entry.summary) > len(record["summary"])
            ):
                record["summary"] = entry.summary
            record["sections"].add(section_context)

        section_blocks.append(f"[Section: {page.title} :: {heading_label}]")
        processed_section, _, _ = replace_snippet_markers(
            transformed_section_text.strip(),
            snippet_map,
            append_unused=False,
            context=f"{module.name} :: {page.title}",
        )
        section_blocks.append(processed_section)
        section_blocks.append("")
        if code_block:
            section_blocks.append("[Code Snippet]")
            section_blocks.append(f"```{code_lang}\n{code_block}\n```")
            section_blocks.append("")

        narrative_data.append(
            make_narrative_datum(
                repo=repo,
                module=module,
                page=page,
                section_heading=section.heading,
                section_text=processed_section,
                design_intent=design_text or (module.responsibility[0] if module.responsibility else ""),
                design_sources=appended_snippets,
            )
        )

    if design_summary_map:
        lines.append("Design Summary:")
        for (path, start, end), info in sorted(
            design_summary_map.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2]),
        ):
            sections = sorted(info.get("sections", []))
            if sections:
                if len(sections) == 1:
                    context = f" (section: {sections[0]})"
                elif len(sections) == 2:
                    context = f" (sections: {sections[0]}; {sections[1]})"
                else:
                    remaining = len(sections) - 2
                    context = f" (sections: {sections[0]}; {sections[1]}; +{remaining} more)"
            else:
                context = ""
            label = format_source_label(path, start, end)
            summary_text = info.get("summary", "")
            if summary_text:
                lines.append(f"- {label}{context} — {summary_text}")
            else:
                lines.append(f"- {label}{context} — Referenced in section narrative.")
        lines.append("")

    append_list_section("Design Intent", design_intents or module.responsibility)
    append_list_section("Prerequisites", module.design_assumptions)
    lines.append("[Implementation Files in Topo Order]")
    lines.extend(section_blocks)
    text = "\n".join(lines).strip()
    label = f"module:{normalize_heading(module.name)}"
    return DatasetChunk(label=label, text=text), narrative_data


class DeepWikiPipeline:
    def __init__(
        self,
        session: Session,
        repo: str,
        design_llm_config: Optional[NarrativeLLMConfig] = None,
    ):
        self.session = session
        self.repo = repo
        self.design_llm_config = design_llm_config

    def _fetch_structure_text(self) -> str:
        result = call_tool(
            self.session,
            "read_wiki_structure",
            {"repoName": self.repo},
        )
        blocks = extract_text_blocks(result)
        if not blocks:
            raise MCPError("read_wiki_structure returned no content.")
        return "\n\n".join(blocks)

    def _fetch_contents_markdown(self) -> str:
        result = call_tool(
            self.session,
            "read_wiki_contents",
            {"repoName": self.repo},
        )
        blocks = extract_text_blocks(result)
        if not blocks:
            raise MCPError("read_wiki_contents returned no content.")
        return "\n\n".join(blocks)

    def run(self) -> PipelineOutput:
        logger.info("Fetching wiki structure for %s", self.repo)
        outline_text = self._fetch_structure_text()
        outline_nodes = parse_outline_text(outline_text)
        number_lookup = build_number_lookup(outline_nodes)
        if not outline_nodes:
            raise MCPError("No outline entries were parsed from read_wiki_structure output.")
        logger.info("Parsed %d top-level outline entries", len(outline_nodes))
        logger.info("Fetching wiki contents for %s", self.repo)
        contents_markdown = self._fetch_contents_markdown()
        pages = parse_wiki_markdown(contents_markdown)
        logger.info("Deriving section-aligned plan for %s", self.repo)
        plan = derive_section_plan(outline_nodes, pages, self.repo, number_lookup)
        total_modules = len(plan.module_groups)
        logger.info("Prepared %d section groups", total_modules)
        repo_chunk = build_repo_chunk_text(plan)
        module_chunks: List[DatasetChunk] = []
        narrative_inputs: List[NarrativeDatum] = []
        for idx, module in enumerate(plan.module_groups):
            logger.info(
                "Assembling module %d/%d: %s",
                idx + 1,
                total_modules,
                module.name,
            )
            chunk, metadata = build_module_chunk_text(
                module,
                idx + 1,
                pages,
                self.repo,
                number_lookup=number_lookup,
                design_llm_config=self.design_llm_config,
            )
            module_chunks.append(chunk)
            narrative_inputs.extend(metadata)
        logger.info(
            "Finished assembling %d modules; collected %d narrative records",
            total_modules,
            len(narrative_inputs),
        )
        logger.info("Dataset generation complete for %s", self.repo)
        return PipelineOutput(
            repo=self.repo,
            chunks=[repo_chunk, *module_chunks],
            narrative_inputs=narrative_inputs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interact with DeepWiki MCP server."
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available MCP tools.",
    )
    parser.add_argument(
        "--ask-question",
        nargs=2,
        metavar=("REPO", "QUESTION"),
        help="Call ask_question for the given repository.",
    )
    parser.add_argument(
        "--read-structure",
        metavar="REPO",
        help=(
            "Call read_wiki_structure to list documentation "
            "topics for REPO."
        ),
    )
    parser.add_argument(
        "--read-contents",
        metavar="REPO",
        help=(
            "Call read_wiki_contents to fetch documentation "
            "content for REPO."
        ),
    )
    parser.add_argument(
        "--page",
        type=str,
        default=None,
        help=(
            "When using --read-contents, optionally select "
            "a specific page title."
        ),
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help=(
            "When using --read-contents, optionally extract "
            "a section heading."
        ),
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help=(
            "When using --read-contents, filter the output to paragraphs "
            "containing this keyword (case-insensitive)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log verbosity for progress messages (default: INFO).",
    )
    parser.add_argument(
        "--generate-dataset",
        metavar="REPO",
        help=(
            "Generate a DeepWiki semantic scaffold dataset for REPO "
            "using MCP tooling."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "When using --generate-dataset, write the result to this file "
            "instead of stdout."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Format for --generate-dataset output (default: text).",
    )
    parser.add_argument(
        "--narrative-modes",
        nargs="+",
        choices=["code", "structure", "comment", "cross", "all"],
        default=None,
        help=(
            "Generate narrative variants for the dataset using the specified "
            "modes."
        ),
    )
    parser.add_argument(
        "--narrative-output",
        type=str,
        default=None,
        help="Write generated narratives to this file instead of stdout.",
    )
    parser.add_argument(
        "--narrative-format",
        choices=["text", "json"],
        default="text",
        help="Serialization format for narratives (default: text).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Local filesystem path to the repository clone; overrides remote fetches.",
    )
    parser.add_argument(
        "--repo-commit",
        type=str,
        default=None,
        help="Commit hash to use when fetching raw files (fallbacks to main/master when omitted).",
    )
    parser.add_argument(
        "--design-use-vllm",
        action="store_true",
        help="Rewrite design intent with an external vLLM chat server.",
    )
    parser.add_argument(
        "--design-vllm-server-url",
        type=str,
        default=None,
        help="Fully qualified vLLM chat completions URL (overrides host/port/path).",
    )
    parser.add_argument(
        "--design-vllm-host",
        type=str,
        default="127.0.0.1",
        help="vLLM server host (ignored when --design-vllm-server-url is provided).",
    )
    parser.add_argument(
        "--design-vllm-port",
        type=int,
        default=8000,
        help="vLLM server port (ignored when --design-vllm-server-url is provided).",
    )
    parser.add_argument(
        "--design-vllm-path",
        type=str,
        default="/v1/chat/completions",
        help="vLLM chat completions path (ignored when server URL is provided).",
    )
    parser.add_argument(
        "--design-vllm-model",
        type=str,
        default=None,
        help="Model identifier for design intent rewriting (required when using vLLM).",
    )
    parser.add_argument(
        "--design-vllm-temperature",
        type=float,
        default=0.2,
        help="Temperature for design intent vLLM calls (default: 0.2).",
    )
    parser.add_argument(
        "--design-vllm-top-p",
        type=float,
        default=None,
        help="Top-p sampling value for design intent vLLM calls.",
    )
    parser.add_argument(
        "--design-vllm-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for design intent vLLM responses.",
    )
    parser.add_argument(
        "--design-vllm-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout for design intent vLLM calls (seconds).",
    )
    parser.add_argument(
        "--design-vllm-retries",
        type=int,
        default=2,
        help="Retry attempts for design intent vLLM calls (default: 2).",
    )
    parser.add_argument(
        "--design-vllm-retry-backoff",
        type=float,
        default=2.0,
        help="Backoff multiplier between design intent vLLM retries (seconds).",
    )
    parser.add_argument(
        "--design-vllm-api-key",
        type=str,
        default=None,
        help="Optional API key for the design intent vLLM server.",
    )
    parser.add_argument(
        "--design-vllm-destination-service",
        type=str,
        default="openai",
        help="Optional Destination-Service header for design intent vLLM calls.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.output and not args.generate_dataset:
        parser.error("--output can only be used together with --generate-dataset.")
    if args.output_format != "text" and not args.generate_dataset:
        parser.error(
            "--output-format can only be used together with --generate-dataset."
        )
    narrative_requested = any(
        [
            args.narrative_modes,
            args.narrative_output,
            args.narrative_format != "text",
        ]
    )
    if narrative_requested and not args.generate_dataset:
        parser.error(
            "Narrative options require --generate-dataset to be specified."
        )

    if args.design_use_vllm and not args.generate_dataset:
        parser.error("--design-use-vllm requires --generate-dataset.")

    temp_repo_dir: Optional[Path] = None
    target_repo = args.generate_dataset
    if target_repo:
        if args.repo_root:
            REPO_LOCAL_ROOTS[target_repo] = Path(args.repo_root).resolve()
        else:
            logger.info("Cloning %s into a temporary directory", target_repo)
            try:
                temp_repo_dir = clone_repository(target_repo, args.repo_commit)
            except MCPError as exc:
                logger.error("Failed to prepare repository %s: %s", target_repo, exc)
                sys.exit(1)
            REPO_LOCAL_ROOTS[target_repo] = temp_repo_dir
        if args.repo_commit:
            REPO_COMMIT_OVERRIDE[target_repo] = args.repo_commit.strip()

    design_llm_config: Optional[NarrativeLLMConfig] = None
    if args.design_use_vllm:
        if call_vllm_chat is None or ChatMessage is None:
            parser.error(
                "vLLM client utilities are not available. Install vllm_client.py "
                "or remove --design-use-vllm."
            )
        if not args.design_vllm_model:
            parser.error(
                "--design-vllm-model is required when --design-use-vllm is set."
            )
        max_tokens = args.design_vllm_max_tokens
        if max_tokens is not None and max_tokens <= 0:
            max_tokens = None
        top_p = args.design_vllm_top_p
        if top_p is not None and top_p <= 0:
            top_p = None
        design_llm_config = NarrativeLLMConfig(
            server_url=args.design_vllm_server_url,
            host=args.design_vllm_host,
            port=args.design_vllm_port,
            path=args.design_vllm_path,
            model=args.design_vllm_model,
            temperature=args.design_vllm_temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_key=args.design_vllm_api_key,
            destination_service=args.design_vllm_destination_service,
            timeout=args.design_vllm_timeout,
            retries=args.design_vllm_retries,
            retry_backoff=args.design_vllm_retry_backoff,
        )

    if not any(
        [
            args.list_tools,
            args.ask_question,
            args.read_structure,
            args.read_contents,
            args.generate_dataset,
        ]
    ):
        parser.print_help()
        sys.exit(1)

    def apply_filters(markdown: str) -> str:
        page_text = markdown
        if args.page:
            page_pattern = re.compile(
                rf"^# Page:\s*{re.escape(args.page)}\s*$",
                re.MULTILINE | re.IGNORECASE,
            )
            matches = list(page_pattern.finditer(markdown))
            if not matches:
                raise MCPError(f"Page '{args.page}' not found in content.")
            page_match = matches[0]
            start_idx = page_match.start()
            next_page_match = re.search(
                r"^# Page:\s*.*$",
                markdown[page_match.end():],
                re.MULTILINE,
            )
            end_idx = (
                page_match.end() + next_page_match.start()
                if next_page_match
                else len(markdown)
            )
            page_text = markdown[start_idx:end_idx]
        section_text = page_text
        if args.section:
            heading_pattern = re.compile(
                r"^(#{2,6})\s*(.+?)\s*$",
                re.MULTILINE,
            )
            matches = list(heading_pattern.finditer(page_text))
            target_norm = normalize_heading(args.section)
            selected = None
            for match in matches:
                title = match.group(2)
                if normalize_heading(title) == target_norm:
                    selected = match
                    break
            if not selected:
                for match in matches:
                    title_norm = normalize_heading(match.group(2))
                    if (
                        target_norm in title_norm
                        or title_norm in target_norm
                    ):
                        selected = match
                        break
            if not selected:
                available_titles = [
                    heading.group(2).strip() for heading in matches
                ]
                preview = ", ".join(available_titles[:8])
                raise MCPError(
                    f"Section '{args.section}' not found. "
                    f"Available headings: {preview}"
                )
            start_idx = selected.start()
            next_match = None
            for heading in matches:
                if heading.start() > start_idx:
                    next_match = heading
                    break
            end_idx = (
                next_match.start() if next_match else len(page_text)
            )
            section_text = page_text[start_idx:end_idx]
        if args.contains:
            keyword = args.contains.lower()
            paragraphs = [
                para
                for para in section_text.split("\n\n")
                if keyword in para.lower()
            ]
            section_text = "\n\n".join(paragraphs) if paragraphs else ""
        stripped = section_text.strip()
        return stripped if stripped else section_text

    session: Optional[Session] = None
    try:
        session = initialize_session()
        if args.list_tools:
            tools = list_tools(session)
            print(json.dumps(tools, ensure_ascii=False, indent=2))
        if args.ask_question:
            repo, question = args.ask_question
            result = call_tool(
                session,
                "ask_question",
                {"repoName": repo, "question": question},
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.read_structure:
            result = call_tool(
                session,
                "read_wiki_structure",
                {"repoName": args.read_structure},
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.read_contents:
            result = call_tool(
                session,
                "read_wiki_contents",
                {"repoName": args.read_contents},
            )
            text_blocks = [
                item["text"]
                for item in result.get("result", {}).get("content", [])
                if item.get("type") == "text"
            ]
            if not text_blocks:
                raise MCPError(
                    "No textual content returned by read_wiki_contents."
                )
            content = "\n\n".join(text_blocks)
            if args.page or args.section or args.contains:
                try:
                    content = apply_filters(content)
                except MCPError as exc:
                    print(f"[warn] {exc}", file=sys.stderr)
                    sys.exit(1)
                if not content.strip():
                    print(
                        "[warn] No matches after applying filters.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            try:
                print(content)
            except BrokenPipeError:
                # Allow piping to commands like `head`.
                pass
        if args.generate_dataset:
            logger.info("Starting dataset generation for %s", args.generate_dataset)
            pipeline = DeepWikiPipeline(
                session=session,
                repo=args.generate_dataset,
                design_llm_config=design_llm_config,
            )
            output = pipeline.run()
            if args.output_format == "json":
                serialized = json.dumps(
                    output.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            else:
                serialized = output.to_text()
            if args.output:
                logger.info("Writing dataset output to %s", args.output)
                with open(args.output, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
            else:
                logger.info("Streaming dataset output to stdout")
                try:
                    print(serialized)
                except BrokenPipeError:
                    pass
            if narrative_requested:
                modes = args.narrative_modes or ["code"]
                logger.info("Generating narratives with modes: %s", ", ".join(modes))
                narratives = output.generate_narratives(modes)
                if narratives:
                    logger.info(
                        "Generated narratives for modes: %s",
                        ", ".join(narratives.keys()),
                    )
                    if args.narrative_format == "json":
                        narrative_serialized = json.dumps(
                            narratives,
                            ensure_ascii=False,
                            indent=2,
                        )
                    else:
                        blocks: List[str] = []
                        for mode, texts in narratives.items():
                            blocks.append(f"[Narrative Mode: {mode}]")
                            blocks.extend(texts)
                        narrative_serialized = "\n\n".join(blocks)
                    target = args.narrative_output
                    if target:
                        logger.info("Writing narratives to %s", target)
                        with open(target, "w", encoding="utf-8") as handle:
                            handle.write(narrative_serialized)
                    else:
                        logger.info("Streaming narratives to stdout")
                        if not args.output:
                            print()
                        try:
                            print(narrative_serialized)
                        except BrokenPipeError:
                            pass
                else:
                    logger.warning("No narrative inputs were collected")
                    message = "[warn] No narrative inputs were collected."
                    target = args.narrative_output
                    if target:
                        with open(target, "w", encoding="utf-8") as handle:
                            handle.write(message)
                    else:
                        print(message, file=sys.stderr)
    finally:
        if session:
            delete_session(session)
        if temp_repo_dir:
            shutil.rmtree(temp_repo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
