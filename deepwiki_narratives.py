"""Narrative template builders for DeepWiki semantic data."""

from __future__ import annotations

from typing import Dict, Iterable, List


def _join_list(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    return ", ".join(items) if items else ""


def build_code_narrative(dw: Dict[str, object]) -> str:
    used_by = _join_list(dw.get("relations", {}).get("used_by", []))
    if used_by:
        usage_line = f"Continue with: {used_by}."
    else:
        usage_line = "This component integrates with other modules in the repository."
    code = dw.get("code", "").rstrip()
    language = dw.get("language", "python") or "python"
    body = (dw.get("body") or "").strip()
    design_intent = (dw.get("design_intent") or "").strip()
    extra_snippets = dw.get("design_sources") or []
    parts: List[str] = [
        f"In the {dw.get('repo')} repository, consider the following component.",
        "",
        f"File: {dw.get('file') or 'N/A'}",
    ]
    if design_intent:
        parts.extend(["", f"Design intent: {design_intent}"])
    if body:
        parts.extend(["", "Original context:", body])
    if code:
        parts.extend(
            [
                "",
                f"```{language}\n{code}\n```",
            ]
        )
    for snippet in extra_snippets:
        parts.extend(["", snippet])
    parts.extend(["", usage_line])
    return "\n".join(parts).strip()


def build_structure_walk(dw: Dict[str, object]) -> str:
    used_by = _join_list(dw.get("relations", {}).get("used_by", []))
    connector = (
        f"It connects the following components: {used_by}."
        if used_by
        else "It connects several components across the repository."
    )
    return (
        "The training pipeline relies on a shared data structure.\n\n"
        f"{dw.get('symbol')} acts as the central interface between rollout and training stages.\n"
        f"{connector}"
    ).strip()


def expand_comments(dw: Dict[str, object]) -> str:
    code = dw.get("code", "").rstrip()
    language = dw.get("language", "python") or "python"
    design_intent = (dw.get("design_intent") or dw.get("summary") or "").strip()
    return (
        f"{design_intent or 'This component encapsulates repository semantics.'}\n\n"
        f"```{language}\n{code}\n```"
    ).strip()


def build_cross_file(dw: Dict[str, object]) -> str:
    used_by = _join_list(dw.get("relations", {}).get("used_by", []))
    usage_line = (
        f"Follow-on topics: {used_by}."
        if used_by
        else "Follow-on topics: (not detected)."
    )
    design_intent = (dw.get("design_intent") or dw.get("summary") or "").strip()
    extra_snippets = dw.get("design_sources") or []
    snippet_block = ""
    if extra_snippets:
        snippet_block = "\n\n" + "\n\n".join(extra_snippets)
    return (
        f"{dw.get('symbol')} appears across multiple modules.\n\n"
        f"Definition: {dw.get('file') or 'N/A'}\n"
        f"Design intent: {design_intent}\n"
        f"{usage_line}"
        f"{snippet_block}"
    ).strip()


MODES = {
    "code": build_code_narrative,
    "structure": build_structure_walk,
    "comment": expand_comments,
    "cross": build_cross_file,
}


def build_narratives(dw_items: Iterable[Dict[str, object]], modes: Iterable[str]) -> Dict[str, List[str]]:
    selected_modes: List[str] = []
    for mode in modes:
        if mode == "all":
            selected_modes = list(MODES)
            break
        if mode in MODES and mode not in selected_modes:
            selected_modes.append(mode)
    if not selected_modes:
        selected_modes = ["code"]
    output: Dict[str, List[str]] = {mode: [] for mode in selected_modes}
    for dw in dw_items:
        for mode in selected_modes:
            builder = MODES[mode]
            try:
                narrative = builder(dw)
            except Exception as exc:  # pragma: no cover - defensive
                narrative = f"[error] Failed to build {mode} narrative: {exc}"
            output[mode].append(narrative)
    return output
