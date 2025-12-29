"""Shared helpers for hydrating DeepWiki sections with repo snippets."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from .parsing import resolve_label_path_range

logger = logging.getLogger(__name__)

BracketKey = Tuple[str, int, int]

PATH_RANGE_RE = re.compile(r"([A-Za-z0-9_.\-/]+):(\d+)(?:-(\d+))?")
PATH_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sh": "bash",
    ".bash": "bash",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".sql": "sql",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


def _normalise_path(path: str) -> str:
    candidate = path.strip()
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate.lstrip("[").split("]", 1)[0].strip()
    if candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate


def _is_valid_path(path: str) -> bool:
    return (
        bool(path)
        and not path.startswith("-")
        and "*" not in path
        and " " not in path
        and PATH_ALLOWED_RE.match(path) is not None
    )


def _full_reference(path: str) -> Optional[BracketKey]:
    cleaned = _normalise_path(path)
    if not _is_valid_path(cleaned):
        return None
    if "/" not in cleaned and "." not in cleaned:
        return None
    return cleaned, 1, 0


def _parse_reference_string(text: str) -> Optional[BracketKey]:
    label = text.strip()
    bracket_contents = re.findall(r"\[([^\]]+)\]", label)
    if bracket_contents:
        for inner in reversed(bracket_contents):
            candidate = inner.strip()
            if candidate.lower().startswith("source:"):
                candidate = candidate.split(":", 1)[1].strip()
            if ":" not in candidate:
                full = _full_reference(candidate)
                if full:
                    return full
            parsed = resolve_label_path_range(candidate)
            if parsed:
                path, start, end = parsed
                cleaned = _normalise_path(path)
                if _is_valid_path(cleaned):
                    return cleaned, start, end
            matches = list(PATH_RANGE_RE.finditer(candidate))
            for match in reversed(matches):
                cleaned = _normalise_path(match.group(1))
                if not _is_valid_path(cleaned):
                    continue
                start = int(match.group(2))
                end = int(match.group(3) or match.group(2))
                return cleaned, start, end
            cleaned_candidate = _normalise_path(candidate)
            if _is_valid_path(cleaned_candidate):
                return cleaned_candidate, 1, 0

    working = label
    if working.endswith("()"):
        working = working[:-2].strip()
    if working.startswith("[") and working.endswith("]"):
        working = working[1:-1].strip()
    if working.startswith("- "):
        working = working[2:].strip()
    if working.lower().startswith("source:"):
        working = working.split(":", 1)[1].strip()

    if ":" not in working:
        full = _full_reference(working)
        if full:
            return full

    parsed = resolve_label_path_range(working)
    if parsed:
        path, start, end = parsed
        cleaned = _normalise_path(path)
        if _is_valid_path(cleaned):
            return cleaned, start, end

    matches = list(PATH_RANGE_RE.finditer(working))
    for match in reversed(matches):
        cleaned = _normalise_path(match.group(1))
        if not _is_valid_path(cleaned):
            continue
        start = int(match.group(2))
        end = int(match.group(3) or match.group(2))
        return cleaned, start, end

    fallback = _normalise_path(working)
    if _is_valid_path(fallback) and ("/" in fallback or "." in fallback):
        return fallback, 1, 0
    return None


def _load_snippet(
    repo_root: Path,
    *,
    path: str,
    start: int,
    end: int,
    encoding: str = "utf-8",
) -> Optional[str]:
    if not path:
        return None
    candidate = (repo_root / path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        logger.debug("Skipping reference outside repo root: %s", path)
        return None
    if not candidate.is_file():
        logger.debug("Skipping non-file reference: %s", candidate)
        return None
    try:
        data = candidate.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        logger.debug("Unable to read %s: %s", candidate, exc)
        return None
    lines = data.splitlines()
    if not lines:
        return None
    start_idx = max(start - 1, 0)
    if start_idx >= len(lines):
        return None
    if end <= 0:
        snippet_lines = lines[start_idx:]
    else:
        end_idx = max(end, start_idx + 1)
        end_idx = min(end_idx, len(lines))
        snippet_lines = lines[start_idx:end_idx]
    snippet = "\n".join(snippet_lines).rstrip()
    return snippet or None


def _format_label(path: str, start: int, end: int) -> str:
    if end <= 0:
        return path
    if end <= start:
        return f"{path}:{start}"
    return f"{path}:{start}-{end}"


def _guess_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _LANGUAGE_BY_SUFFIX.get(suffix, "text")


def _format_snippet(path: str, start: int, end: int, snippet: str) -> str:
    label = _format_label(path, start, end)
    language = _guess_language(path)
    body = snippet.rstrip()
    return f"{label}\n```{language}\n{body}\n```"


def _strip_link_target(target: str) -> str:
    cleaned = target.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _hydrate_markdown_links(
    text: str,
    *,
    repo_root: Path,
    cache: Dict[BracketKey, Optional[str]],
    embedded: Set[BracketKey],
) -> str:
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def replacer(match: re.Match[str]) -> str:
        target = _strip_link_target(match.group(2))
        lower = target.lower()
        if lower.startswith(("http://", "https://", "mailto:")):
            return match.group(0)
        parsed = _parse_reference_string(target)
        if not parsed:
            return match.group(0)
        path, start, end = parsed
        key = (path, start, end)
        if key not in cache:
            cache[key] = _load_snippet(repo_root, path=path, start=start, end=end)
        snippet = cache.get(key)
        if not snippet:
            return match.group(0)
        embedded.add(key)
        return _format_snippet(path, start, end, snippet)

    return pattern.sub(replacer, text)


def _hydrate_bracket_tokens(
    text: str,
    *,
    repo_root: Path,
    cache: Dict[BracketKey, Optional[str]],
    embedded: Set[BracketKey],
) -> str:
    from re import compile as re_compile

    pattern = re_compile(r"\[([^\]]+)\]\(\)")

    def replacer(match):
        label = match.group(1).strip()
        parsed = resolve_label_path_range(label)
        if not parsed:
            return label
        path, start, end = parsed
        key = (path, start, end)
        if key not in cache:
            cache[key] = _load_snippet(repo_root, path=path, start=start, end=end)
        snippet = cache[key]
        if not snippet:
            return label
        embedded.add(key)
        return _format_snippet(path, start, end, snippet)

    return pattern.sub(replacer, text)


def _hydrate_line_references(
    text: str,
    *,
    repo_root: Path,
    cache: Dict[BracketKey, Optional[str]],
    embedded: Set[BracketKey],
) -> str:
    hydrated_lines = []
    for raw_line in text.splitlines():
        hydrated_lines.append(raw_line)
        stripped = raw_line.strip()
        candidate = stripped
        if stripped.lower().startswith("source:"):
            candidate = stripped.split(":", 1)[1].strip()
        parsed = _parse_reference_string(candidate)
        if not parsed and stripped.startswith("- "):
            bullet = stripped[2:].strip()
            parsed = _parse_reference_string(bullet)
        if not parsed:
            continue
        path, start, end = parsed
        key = (path, start, end)
        if key not in cache:
            cache[key] = _load_snippet(repo_root, path=path, start=start, end=end)
        snippet = cache[key]
        if not snippet or key in embedded:
            continue
        embedded.add(key)
        hydrated_lines.append(_format_snippet(path, start, end, snippet))
    return "\n".join(hydrated_lines)


def hydrate_section_text(text: str, *, repo_root: Path) -> str:
    cache: Dict[BracketKey, Optional[str]] = {}
    embedded: Set[BracketKey] = set()
    hydrated = _hydrate_bracket_tokens(text, repo_root=repo_root, cache=cache, embedded=embedded)
    hydrated = _hydrate_markdown_links(hydrated, repo_root=repo_root, cache=cache, embedded=embedded)
    hydrated = _hydrate_line_references(hydrated, repo_root=repo_root, cache=cache, embedded=embedded)

    remaining_snippets = []
    for key, snippet in cache.items():
        if not snippet or key in embedded:
            continue
        path, start, end = key
        remaining_snippets.append(_format_snippet(path, start, end, snippet))
        embedded.add(key)

    if remaining_snippets:
        hydrated = hydrated.rstrip() + "\n\n" + "\n\n".join(remaining_snippets)

    return hydrated
