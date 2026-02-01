"""LLM helpers for the DeepWiki semantic pipeline."""

from __future__ import annotations

import logging
import os
import re
import textwrap
import time
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Set

from .models import (
    BlockResult,
    CriticFeedback,
    JudgeLLMConfig,
    MisalignmentType,
    NarrativeLLMConfig,
    NarrativeScaffold,
    SectionBlock,
)
from .parsing import extract_summary_paragraph, parse_sources_links

logger = logging.getLogger(__name__)

try:  # Optional dependency for logic/critic rewriting
    from vllm_client import ChatMessage, VLLMError, call_vllm_chat
except ImportError:  # pragma: no cover
    ChatMessage = None  # type: ignore[assignment]
    VLLMError = RuntimeError  # type: ignore[assignment]
    call_vllm_chat = None  # type: ignore[assignment]


def _truncate(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DETAILS_BLOCK_RE = re.compile(
    r"<details\b[^>]*?>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)
_ABSTRACT_VERBS = {
    "coordinate",
    "orchestrate",
    "ensure",
    "enforce",
    "manage",
    "schedule",
    "synchronize",
    "distribute",
    "optimize",
    "stabilize",
    "allocate",
    "control",
    "govern",
    "mediate",
}
_STRUCTURAL_KEYWORDS = {
    "controller",
    "trainer",
    "pipeline",
    "service",
    "component",
    "module",
    "worker",
    "scheduler",
    "manager",
    "engine",
    "adapter",
    "backend",
}
_CONSTRAINT_KEYWORDS = {
    "must",
    "need",
    "requires",
    "constraint",
    "limit",
    "trade-off",
    "tradeoff",
    "throughput",
    "latency",
    "resource",
    "scalability",
    "stability",
    "consistency",
}


def _split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(stripped)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _sanitize_visible_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        lower = stripped.lower()
        if lower.startswith("in the ") and " repository" in lower:
            continue
        if lower.startswith(("system prompt:", "user:", "assistant:")):
            continue
        if stripped.startswith("Source:"):
            continue
        cleaned.append(stripped)
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


def _extract_code_symbols(code: str) -> List[str]:
    symbols: List[str] = []
    for pattern in (
        re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"),
    ):
        for match in pattern.finditer(code):
            name = match.group(1)
            if name and name not in symbols:
                symbols.append(name)
    return symbols[:8]


def _detect_constraints(sentences: Sequence[str]) -> List[str]:
    constraints: List[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in _CONSTRAINT_KEYWORDS):
            constraints.append(sentence)
        if len(constraints) >= 3:
            break
    return constraints


def _detect_key_mechanisms(
    sentences: Sequence[str],
    anchors: Sequence[str],
) -> List[str]:
    mechanisms: List[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in _STRUCTURAL_KEYWORDS):
            mechanisms.append(sentence)
        elif any(anchor.lower() in lowered for anchor in anchors):
            mechanisms.append(sentence)
        if len(mechanisms) >= 3:
            break
    return mechanisms


def build_scaffold(text: str, block: SectionBlock) -> NarrativeScaffold:
    sentences = _split_sentences(text)
    design_intent = sentences[0] if sentences else ""
    constraints = _detect_constraints(sentences[1:])
    code_anchors = _extract_code_symbols(block.code)
    key_mechanisms = _detect_key_mechanisms(sentences[1:], code_anchors)
    return NarrativeScaffold(
        design_intent=design_intent,
        constraints=constraints,
        key_mechanisms=key_mechanisms,
        code_anchors=code_anchors,
    )


def compute_learnability_score(text: str) -> float:
    tokens = text.split()
    score = 0.0
    if len(tokens) >= 80:
        score += 0.4
    elif len(tokens) >= 40:
        score += 0.2
    if any(verb in text.lower() for verb in _ABSTRACT_VERBS):
        score += 0.3
    if any(keyword in text.lower() for keyword in _STRUCTURAL_KEYWORDS):
        score += 0.3
    return round(min(score, 1.0), 3)


def classify_misalignment(text: str) -> MisalignmentType:
    lowered = text.lower()
    if not lowered:
        return MisalignmentType.NONE
    if "without explaining" in lowered or "missing intent" in lowered:
        return MisalignmentType.NO_INTENT
    if "no structural rationale" in lowered or "missing mechanism" in lowered:
        return MisalignmentType.NO_MECHANISM
    if "jumps straight to code" in lowered or "code appears without context" in lowered:
        return MisalignmentType.JUMP_TO_CODE
    if "too low level" in lowered or "implementation detail before design" in lowered:
        return MisalignmentType.MISPLACED_DETAIL
    if "contradiction" in lowered or "inconsistent with code" in lowered:
        return MisalignmentType.INCONSISTENT
    return MisalignmentType.NONE


PAGE_SUMMARY_SYSTEM_PROMPT = (
    "You are analysing documentation extracted from a repository.\n"
    "Produce a concise 3-4 sentence summary capturing the feature set, "
    "major responsibilities, and dependencies that the page describes."
)

BLOCK_REWRITE_SYSTEM_PROMPT = (
    "You are reconstructing BOTH the design-time reasoning and the resulting design decisions for a piece of code.\n\n"
    "You are given:\n"
    "- Explanatory text from a project wiki (often high-level, sometimes incomplete)\n"
    "- One or more related code snippets\n\n"
    "Your goal is NOT to summarize the wiki, and NOT to explain the code line-by-line.\n\n"
    "Instead, write a rich engineering note that includes:\n"
    "1) Design-time thinking: constraints, risks, alternatives considered, trade-offs, sequencing.\n"
    "2) Design outcome: the chosen structure/architecture, key components & responsibilities, important interfaces/contracts,\n"
    "   invariants, and failure handling strategy.\n\n"
    "Write in natural technical prose. You may use short section headers (e.g. 'Reasoning', 'Chosen Design', 'Trade-offs',\n"
    "'Failure Modes') but avoid rigid formatting like tables. Prefer concrete details over vague statements.\n\n"
    "Be generous with detail: expand the content when possible, grounding claims in what is implied by the code and the wiki.\n"
    "If something is uncertain, state the assumption explicitly.\n\n"
    "Do NOT paste large code blocks into the rewritten narrative; code will be appended separately by the pipeline."
)

CRITIC_SYSTEM_PROMPT = (
    "You are evaluating whether the following text is a high-quality engineering note.\n\n"
    "The note must include BOTH:\n"
    "- Design-time reasoning (constraints, trade-offs, alternatives, sequencing, risks)\n"
    "- Design outcome (what was chosen: structure/architecture, responsibilities, interfaces/contracts, invariants, failure handling)\n\n"
    "Reject notes that are too short, too generic, or that only do one of the two (reasoning-only or outcome-only).\n"
    "Reject notes that simply restate the wiki or explain code line-by-line.\n\n"
    "Respond with:\n"
    "PASS or FAIL\n\n"
    "If FAIL, explain precisely what is missing (e.g. 'no explicit trade-offs', 'no concrete design outcome', 'missing failure modes',\n"
    "'only summarizes wiki', 'too generic')."
)

REFINEMENT_REMINDER = (
    "Keep the voice anchored in pre-implementation reasoning. Preserve lines that already capture constraints, trade-offs, "
    "and sequencing, and revise only the gaps flagged by the critic."
)

SECTION_REWRITE_SYSTEM_PROMPT = BLOCK_REWRITE_SYSTEM_PROMPT

SECTION_CRITIC_SYSTEM_PROMPT = CRITIC_SYSTEM_PROMPT


def _section_prompt_char_limit() -> int:
    raw = (os.getenv("DEEPWIKI_SECTION_PROMPT_CHAR_LIMIT") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    # Default: 128k characters. This is a pragmatic guardrail to avoid sending
    # extremely large prompts to vLLM and to preserve raw hydrated context when
    # it would otherwise fail.
    return 131_072


def _should_bypass_llm_for_section(text: str) -> bool:
    if not text:
        return False
    return len(text) >= _section_prompt_char_limit()


def _vllm_outage_threshold() -> int:
    raw = (os.getenv("DEEPWIKI_VLLM_OUTAGE_THRESHOLD") or "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


class _VLLMOutageGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive_failures = 0

    def note_success(self) -> None:
        threshold = _vllm_outage_threshold()
        if threshold <= 0:
            return
        with self._lock:
            self._consecutive_failures = 0

    def note_failure(self, exc: Exception) -> None:
        threshold = _vllm_outage_threshold()
        if threshold <= 0:
            return
        if not _is_vllm_connectivity_error(exc):
            return
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
        if failures >= threshold:
            # Use SystemExit so upstream callers that catch Exception do not swallow the abort.
            raise SystemExit(
                f"Aborting: vLLM appears unreachable for {failures} consecutive requests "
                f"(threshold={threshold}). Last error: {exc}"
            )


_VLLM_OUTAGE_GUARD = _VLLMOutageGuard()


def _is_context_length_error(exc: Exception) -> bool:
    """
    Detect token/context overflow style failures from vLLM/OpenAI-compatible servers.
    We intentionally use string matching so this works across differing client/server error types.
    """
    message = str(exc or "").lower()
    if not message:
        return False
    patterns = (
        "maximum context length",
        "context length",
        "too many tokens",
        "prompt is too long",
        "request too large",
        "max_tokens must be at least 1",
        "got -",  # common when server computes negative remaining max_tokens
    )
    return any(p in message for p in patterns)


def _is_vllm_connectivity_error(exc: Exception) -> bool:
    """
    Best-effort classification: treat network/connectivity/server outage failures as fatal signals.
    Excludes context-length / bad-request style prompt sizing errors.
    """
    if _is_context_length_error(exc):
        return False
    message = str(exc or "").lower()
    if not message:
        return False
    patterns = (
        "connection refused",
        "connect timeout",
        "read timeout",
        "timed out",
        "name or service not known",
        "temporary failure in name resolution",
        "failed to establish a new connection",
        "connection error",
        "connecterror",
        "server disconnected",
        "remote end closed connection",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "request failed after",
        "across server pool",
    )
    return any(p in message for p in patterns)


def _strip_details_blocks(text: str) -> str:
    """
    Remove <details> blocks (page-level context file listings) before extracting indices.
    These are typically global context rather than paragraph-specific evidence.
    """
    return _DETAILS_BLOCK_RE.sub("", text or "")


_SOURCES_INLINE_RE = re.compile(
    r"(?i)^\s*(?:\*\*|__)?\s*sources?\s*(?:\*\*|__)?\s*[:：\-–—]\s*(?P<rest>.*)$",
    re.MULTILINE,
)
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _extract_section_index_labels(section_text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract repo-relative file references from a section for code-appending.

    We focus on paragraph-level indices and ignore page-level <details> context lists.
    Returns (ranged_labels, unranged_labels).
    """
    text = _strip_details_blocks(section_text)
    ranged: Set[str] = set()
    unranged: Set[str] = set()

    def add(label: str) -> None:
        cleaned = (label or "").strip()
        while cleaned.startswith(("- ", "* ", "• ")):
            cleaned = cleaned[2:].strip()
        cleaned = re.sub(r"^\d+\.\s+", "", cleaned).strip()
        cleaned = _sanitize_reference_label(cleaned)
        if not cleaned:
            return
        if re.search(r":\d+", cleaned):
            ranged.add(cleaned)
        else:
            unranged.add(cleaned)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _SOURCES_INLINE_RE.match(line)
        if not m:
            i += 1
            continue
        rest = (m.group("rest") or "").strip()
        if rest:
            items = [x.strip() for x in _BACKTICK_RE.findall(rest) if x.strip()]
            if not items:
                items = [x.strip() for x in re.split(r"[;,，]", rest) if x.strip()]
            for item in items:
                add(item)
            i += 1
            continue
        # Block form: subsequent bullets / ordered items.
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if not (nxt.startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s+", nxt)):
                break
            candidates = [x.strip() for x in _BACKTICK_RE.findall(nxt) if x.strip()]
            if candidates:
                for item in candidates:
                    add(item)
            else:
                add(nxt)
            i += 1

    # Also include any label lines produced by hydration blocks. These are safe because
    # they're immediately followed by a fenced code block.
    for m in re.finditer(
        r"(?ms)^(?P<label>[A-Za-z0-9_.\-/ ]+(?::\d+(?:-\d+)?)?)\s*\n```",
        text,
    ):
        add(m.group("label") or "")

    return ranged, unranged

def summarise_page(
    *,
    page_text: str,
    page_title: str,
    llm_config: Optional[NarrativeLLMConfig],
) -> str:
    fallback = extract_summary_paragraph(page_text) or page_title
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return fallback
    messages = [
        ChatMessage(role="system", content=PAGE_SUMMARY_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=textwrap.dedent(
                f"""\
                # Page: {page_title}

                {page_text.strip()}
                """,
            ),
        ),
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
            server_urls=llm_config.server_urls,
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover - network dependent
        logger.warning("Page summary LLM failed: %s", exc)
        return fallback
    summary = response.strip()
    return summary or fallback


def _join_code_blocks(blocks: Sequence[SectionBlock]) -> str:
    snippets = []
    for block in blocks:
        code = block.code.strip()
        if not code:
            continue
        header = f"```{block.language or 'text'}\n{code}\n```"
        snippets.append(header)
    return "\n\n".join(snippets).strip()


def _make_design_stub(subject: str, language: str, code: str) -> str:
    topic = subject or f"this {language or 'code'} change"
    snippet_hint = ""
    first_line = ""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_line = stripped
            break
    if first_line:
        snippet_hint = (
            " I'm expecting the implementation to center around "
            f"`{first_line}` (or something close), so I need to validate the inputs and side effects it depends on."
        )
    return _sanitize_visible_text(
        " ".join(
            [
                f"I'm about to sketch how {topic} should come together.",
                "The surrounding system is already live, so every change has to integrate without disrupting the existing callers.",
                "Before typing anything I want to map the hard constraints, decide what I can defer, and stage the risky pieces last.",
                "I'm also listing the failure modes I can't afford and the checks I'll lean on to catch them early.",
                snippet_hint.strip(),
            ]
        )
    )


@dataclass
class SectionResult:
    narrative: str
    critic: str
    verdict: str
    misalignment: Optional[MisalignmentType]
    learnability: float
    critic_history: List[str]
    code_blocks: List[SectionBlock]


def rewrite_block(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    block_index: int,
    block: SectionBlock,
    llm_config: Optional[NarrativeLLMConfig],
) -> Tuple[str, NarrativeScaffold]:
    explanation = block.explanation.strip()
    fallback = _make_design_stub(explanation, block.language, block.code)
    if not llm_config or not call_vllm_chat or not ChatMessage:
        scaffold = build_scaffold(fallback, block)
        return fallback, scaffold
    mermaid_text = block.mermaid or "(none)"
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}
        Block index: {block_index}
        Original context (verbatim excerpt before the code):
        {explanation or '(no explicit description)'}

        Mermaid context (if any):
        {mermaid_text}

        Code snippet (language={block.language}):
        ```{block.language}
        {_truncate(block.code, 6000)}
        ```

        Reconstruct the author's internal reasoning immediately before implementing this block.
        Focus on constraints, options you rejected, trade-offs, assumptions, and the order you plan to follow.
        Do not summarise the wiki or explain the finished code. Avoid Markdown tables or bullet lists; keep everything in flowing prose.
        """
    )
    messages = [
        ChatMessage(role="system", content=BLOCK_REWRITE_SYSTEM_PROMPT),
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
            server_urls=llm_config.server_urls,
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        logger.warning(
            "Block rewrite failed for %s :: %s (block %d): %s",
            page_title,
            section_heading,
            block_index,
            exc,
        )
        scaffold = build_scaffold(fallback, block)
        return fallback, scaffold
    rewritten = response.strip()
    cleaned = _sanitize_visible_text(rewritten or fallback)
    scaffold = build_scaffold(cleaned, block)
    return cleaned, scaffold


def rewrite_section(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    section_text: str,
    code_blocks: Sequence[SectionBlock],
    llm_config: Optional[NarrativeLLMConfig],
    fallback_subject: Optional[str] = None,
) -> str:
    lead_block = code_blocks[0] if code_blocks else None
    subject = fallback_subject or section_heading or page_title
    fallback = _make_design_stub(
        subject.strip() if subject else "",
        lead_block.language if lead_block else "section",
        lead_block.code if lead_block else "",
    )
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return fallback
    if _should_bypass_llm_for_section(section_text):
        return section_text.strip() or fallback
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}

        Hydrated section contents (original prose + code fences):
        {section_text.strip()}

        Task:
        - Expand into a rich engineering note that includes BOTH design-time thinking AND the final design outcome.
        - Design-time thinking: constraints, risks, alternatives rejected, trade-offs, sequencing.
        - Design outcome: chosen structure/architecture, component responsibilities, interfaces/contracts, invariants, failure handling.
        - Be concrete and detailed; it is OK to make explicit assumptions if needed.
        - Do NOT paste large code blocks in the narrative; code is appended separately.
        """
    )
    messages = [
        ChatMessage(role="system", content=SECTION_REWRITE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
    try:
        started_at = time.time()
        prompt_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
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
            server_urls=llm_config.server_urls,
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        try:
            _VLLM_OUTAGE_GUARD.note_failure(exc)
        except SystemExit:
            raise
        logger.warning(
            "Section rewrite failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        if _is_context_length_error(exc):
            return section_text.strip() or fallback
        return fallback
    _VLLM_OUTAGE_GUARD.note_success()
    elapsed = time.time() - started_at
    if elapsed >= 30.0:
        logger.warning(
            "Slow section rewrite (%.1fs) for %s :: %s (section_len=%d prompt_chars=%d model=%s timeout=%.0fs)",
            elapsed,
            page_title,
            section_heading,
            len(section_text),
            prompt_chars,
            llm_config.model,
            llm_config.timeout,
        )
    cleaned = _sanitize_visible_text(response or fallback)
    return cleaned or fallback


def critique_block(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    block_index: int,
    rewritten_text: str,
    block: SectionBlock,
    judge_config: Optional[JudgeLLMConfig],
) -> CriticFeedback:
    if not judge_config or not call_vllm_chat or not ChatMessage:
        return CriticFeedback(
            verdict="PASS",
            text="No critic LLM configured; manual verification required.",
            misalignment=MisalignmentType.NONE,
        )
    if _should_bypass_llm_for_section(rewritten_text):
        return CriticFeedback(
            verdict="PASS",
            text="Skipped critic due to long context; preserved raw section content.",
            misalignment=MisalignmentType.NONE,
        )
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}
        Block index: {block_index}

        Explanation under review:
        {_truncate(rewritten_text, 4000)}

        Referenced code snippet (language={block.language}):
        ```{block.language}
        {_truncate(block.code, 6000)}
        ```
        """
    )
    messages = [
        ChatMessage(
            role="system",
            content=judge_config.system_prompt or CRITIC_SYSTEM_PROMPT,
        ),
        ChatMessage(role="user", content=user_prompt),
    ]
    try:
        response = call_vllm_chat(
            host=judge_config.host,
            port=judge_config.port,
            path=judge_config.path,
            model=judge_config.model,
            messages=messages,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            top_p=judge_config.top_p,
            server_url=judge_config.server_url,
            server_urls=judge_config.server_urls,
            api_key=judge_config.api_key,
            destination_service=judge_config.destination_service,
            timeout=judge_config.timeout,
            retries=judge_config.retries,
            retry_backoff=judge_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        try:
            _VLLM_OUTAGE_GUARD.note_failure(exc)
        except SystemExit:
            raise
        logger.warning(
            "Critic LLM failed for %s :: %s (block %d): %s",
            page_title,
            section_heading,
            block_index,
            exc,
        )
        return CriticFeedback(
            verdict="FAIL",
            text="Critic inference failed; please review manually.",
            misalignment=MisalignmentType.INCONSISTENT,
        )
    _VLLM_OUTAGE_GUARD.note_success()
    critic_text = response.strip()
    if not critic_text:
        return CriticFeedback(
            verdict="PASS",
            text="Critic returned an empty response.",
            misalignment=MisalignmentType.NONE,
        )
    lines = critic_text.splitlines()
    first_line = lines[0].strip()
    verdict = "UNKNOWN"
    explanation = ""
    if first_line.upper().startswith("PASS"):
        verdict = "PASS"
        explanation = first_line[4:].strip()
        remainder = "\n".join(lines[1:]).strip()
        if remainder:
            explanation = (explanation + "\n" + remainder).strip() if explanation else remainder
        if not explanation:
            explanation = "Narrative follows the expected reasoning arc."
        return CriticFeedback(
            verdict=verdict,
            text=explanation,
            misalignment=MisalignmentType.NONE,
        )
    if first_line.upper().startswith("FAIL"):
        verdict = "FAIL"
        explanation = first_line[4:].strip()
        remainder = "\n".join(lines[1:]).strip()
        if remainder:
            explanation = (explanation + "\n" + remainder).strip() if explanation else remainder
        if not explanation:
            explanation = "Critic did not explain the failure."
        misalignment = classify_misalignment(explanation)
        return CriticFeedback(
            verdict=verdict,
            text=explanation,
            misalignment=misalignment,
        )
    return CriticFeedback(
        verdict="UNKNOWN",
        text=critic_text,
        misalignment=MisalignmentType.NONE,
    )


def critique_section(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    narrative: str,
    code_blocks: Sequence[SectionBlock],
    judge_config: Optional[JudgeLLMConfig],
) -> CriticFeedback:
    if not judge_config or not call_vllm_chat or not ChatMessage:
        return CriticFeedback(
            verdict="PASS",
            text="No critic LLM configured; manual verification required.",
            misalignment=MisalignmentType.NONE,
        )
    if _should_bypass_llm_for_section(narrative):
        return CriticFeedback(
            verdict="PASS",
            text="Skipped critic due to long context; preserved raw section content.",
            misalignment=MisalignmentType.NONE,
        )
    code_text = _truncate(_join_code_blocks(code_blocks), 6000) if code_blocks else "(no code snippets detected)"
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}

        Narrative under review:
        {_truncate(narrative, 4000)}

        Referenced code:
        {code_text}
        """
    )
    messages = [
        ChatMessage(
            role="system",
            content=judge_config.system_prompt or SECTION_CRITIC_SYSTEM_PROMPT,
        ),
        ChatMessage(role="user", content=user_prompt),
    ]
    try:
        response = call_vllm_chat(
            host=judge_config.host,
            port=judge_config.port,
            path=judge_config.path,
            model=judge_config.model,
            messages=messages,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            top_p=judge_config.top_p,
            server_url=judge_config.server_url,
            server_urls=judge_config.server_urls,
            api_key=judge_config.api_key,
            destination_service=judge_config.destination_service,
            timeout=judge_config.timeout,
            retries=judge_config.retries,
            retry_backoff=judge_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        try:
            _VLLM_OUTAGE_GUARD.note_failure(exc)
        except SystemExit:
            raise
        logger.warning(
            "Section critic failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return CriticFeedback(
            verdict="FAIL",
            text="Critic inference failed; please review manually.",
            misalignment=MisalignmentType.INCONSISTENT,
        )
    _VLLM_OUTAGE_GUARD.note_success()
    critic_text = response.strip()
    if not critic_text:
        return CriticFeedback(
            verdict="PASS",
            text="Critic returned an empty response.",
            misalignment=MisalignmentType.NONE,
        )
    lines = critic_text.splitlines()
    first_line = lines[0].strip()
    explanation = "\n".join(lines[1:]).strip()
    if first_line.upper().startswith("PASS"):
        msg = first_line[4:].strip()
        text = msg or explanation or "Narrative follows the expected arc."
        return CriticFeedback(
            verdict="PASS",
            text=text,
            misalignment=MisalignmentType.NONE,
        )
    if first_line.upper().startswith("FAIL"):
        msg = first_line[4:].strip()
        text = (msg + "\n" + explanation).strip() if explanation else msg or "Critic did not explain the failure."
        misalignment = classify_misalignment(text)
        return CriticFeedback(
            verdict="FAIL",
            text=text,
            misalignment=misalignment,
        )
    return CriticFeedback(
        verdict="UNKNOWN",
        text=critic_text,
        misalignment=MisalignmentType.NONE,
    )


def refine_block(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    block_index: int,
    current_text: str,
    critic: CriticFeedback,
    block: SectionBlock,
    llm_config: Optional[NarrativeLLMConfig],
) -> str:
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return current_text
    if _should_bypass_llm_for_section(current_text):
        return current_text
    reminder = f"Primary issue: {critic.misalignment.value}" if critic.misalignment else "Primary issue: none reported"
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}
        Block index: {block_index}

        Current narrative draft:
        {current_text.strip()}

        {REFINEMENT_REMINDER}
        {reminder}

        Critic feedback to address (treat as truth for this revision):
        {critic.text.strip()}

        Code snippet (language={block.language}):
        ```{block.language}
        {_truncate(block.code, 6000)}
        ```

        Produce an updated design-time reasoning note that resolves the critic feedback while staying in the planning voice.
        """
    )
    messages = [
        ChatMessage(role="system", content=BLOCK_REWRITE_SYSTEM_PROMPT),
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
            server_urls=llm_config.server_urls,
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        try:
            _VLLM_OUTAGE_GUARD.note_failure(exc)
        except SystemExit:
            raise
        logger.warning(
            "Refinement failed for %s :: %s (block %d): %s",
            page_title,
            section_heading,
            block_index,
            exc,
        )
        return current_text
    _VLLM_OUTAGE_GUARD.note_success()
    refined = response.strip()
    cleaned = _sanitize_visible_text(refined or current_text)
    return cleaned if cleaned else current_text


def refine_section(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    current_text: str,
    critic: CriticFeedback,
    code_blocks: Sequence[SectionBlock],
    llm_config: Optional[NarrativeLLMConfig],
) -> str:
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return current_text
    if _should_bypass_llm_for_section(current_text):
        return current_text
    reminder = f"Primary issue: {critic.misalignment.value}" if critic.misalignment else "Primary issue: none reported"
    code_text = _truncate(_join_code_blocks(code_blocks), 6000) if code_blocks else "(no code snippets detected)"
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}

        Current narrative draft:
        {current_text.strip()}

        {REFINEMENT_REMINDER}
        {reminder}

        Critic feedback to address (treat as truth for this revision):
        {critic.text.strip()}

        Referenced code:
        {code_text}

        Produce a revised design-time reasoning note that addresses the critic feedback while keeping the voice in pre-implementation planning and avoiding tables or bullet lists.
        """
    )
    messages = [
        ChatMessage(role="system", content=SECTION_REWRITE_SYSTEM_PROMPT),
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
            server_urls=llm_config.server_urls,
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        try:
            _VLLM_OUTAGE_GUARD.note_failure(exc)
        except SystemExit:
            raise
        logger.warning(
            "Section refinement failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return current_text
    _VLLM_OUTAGE_GUARD.note_success()
    refined = response.strip()
    cleaned = _sanitize_visible_text(refined or current_text)
    return cleaned if cleaned else current_text


def make_block_result(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    block_index: int,
    block: SectionBlock,
    logic_config: Optional[NarrativeLLMConfig],
    critic_config: Optional[JudgeLLMConfig],
    judge_rounds: int,
) -> BlockResult:
    draft_text, _ = rewrite_block(
        repo=repo,
        page_title=page_title,
        section_heading=section_heading,
        block_index=block_index,
        block=block,
        llm_config=logic_config,
    )
    logger.debug(
        "Draft[%s :: %s :: block %d :: pass %d]: %s",
        page_title,
        section_heading,
        block_index,
        1,
        _truncate(draft_text, 400),
    )
    current_text = draft_text
    critic_history: List[str] = []
    final_feedback = CriticFeedback(
        verdict="PASS",
        text="No critic feedback recorded.",
        misalignment=MisalignmentType.NONE,
    )
    rounds = max(1, judge_rounds if critic_config else 1)

    for attempt in range(rounds):
        feedback = critique_block(
            repo=repo,
            page_title=page_title,
            section_heading=section_heading,
            block_index=block_index,
            rewritten_text=current_text,
            block=block,
            judge_config=critic_config,
        )
        critic_history.append(feedback.text)
        final_feedback = feedback
        logger.debug(
            "Judge[%s :: %s :: block %d :: pass %d]: verdict=%s misalignment=%s critic=%s",
            page_title,
            section_heading,
            block_index,
            attempt + 1,
            feedback.verdict,
            feedback.misalignment.value if feedback.misalignment else "none",
            _truncate(feedback.text, 300),
        )
        if feedback.verdict.upper() == "PASS" or not critic_config:
            break
        if attempt + 1 >= rounds:
            break
        current_text = refine_block(
            repo=repo,
            page_title=page_title,
            section_heading=section_heading,
            block_index=block_index,
            current_text=current_text,
            critic=feedback,
            block=block,
            llm_config=logic_config,
        )
        logger.debug(
            "Refine[%s :: %s :: block %d :: pass %d]: %s",
            page_title,
            section_heading,
            block_index,
            attempt + 2,
            _truncate(current_text, 400),
        )

    final_scaffold = build_scaffold(current_text, block)
    learnability = compute_learnability_score(current_text)
    logger.debug(
        "Final[%s :: %s :: block %d]: verdict=%s misalignment=%s learnability=%.3f",
        page_title,
        section_heading,
        block_index,
        final_feedback.verdict,
        final_feedback.misalignment.value if final_feedback.misalignment else "none",
        learnability,
    )
    return BlockResult(
        index=block_index,
        rewrite=current_text,
        critic=final_feedback.text,
        block=block,
        scaffold=final_scaffold,
        verdict=final_feedback.verdict,
        misalignment=final_feedback.misalignment,
        learnability=learnability,
        critic_history=critic_history,
    )


# Allow spaces in repo-relative paths (some educational repos use them).
_CODE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.\-/ ]+(?::\d+(?:-\d+)?)?$")
_REFERENCE_INVALID_TOKENS = {"null", "none", ".", ".."}


def _sanitize_reference_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    cleaned = label.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _REFERENCE_INVALID_TOKENS:
        return None
    if "://" in cleaned or cleaned.startswith("//"):
        return None
    if "/" not in cleaned and ":" not in cleaned:
        # allow README-like single filenames but skip generic words
        if not cleaned.lower().endswith((".md", ".rst", ".txt")):
            return None
    if not _CODE_LABEL_RE.match(cleaned):
        return None
    return cleaned


def _extract_label_from_line(line: str) -> Optional[str]:
    working = line.strip()
    if not working or working.startswith("```"):
        return None
    for prefix in ("- ", "* ", "• "):
        if working.startswith(prefix):
            working = working[len(prefix):].strip()
    bracket_labels = re.findall(r"\[([^\]]+)\]", working)
    for bracket in bracket_labels:
        label = _sanitize_reference_label(bracket.strip())
        if label:
            return label
    lowered = working.lower()
    for marker in ("**sources:**", "**source:**", "sources:", "source:"):
        idx = lowered.find(marker)
        if idx != -1:
            label = working[idx + len(marker):].strip()
            sanitized = _sanitize_reference_label(label)
            if sanitized:
                return sanitized
            for chunk in re.split(r"[;,\s]+", label):
                sanitized = _sanitize_reference_label(chunk)
                if sanitized:
                    return sanitized
            return None
    sanitized = _sanitize_reference_label(working)
    if sanitized:
        return sanitized
    for token in re.findall(r"[A-Za-z0-9_.\-/]+(?::\d+(?:-\d+)?)?", working):
        sanitized = _sanitize_reference_label(token)
        if sanitized:
            return sanitized
    return None


def _infer_reference_from_code(code: str) -> Optional[str]:
    for raw_line in code.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            stripped = stripped[2:].strip()
        if stripped.lower().startswith("source:"):
            stripped = stripped.split(":", 1)[1].strip()
        candidate = stripped.split()[0]
        candidate_ref = _sanitize_reference_label(candidate)
        if candidate_ref:
            return candidate_ref
    return None


def _extract_section_sources(section_text: str) -> List[str]:
    sources: List[str] = []
    seen: Set[str] = set()
    in_details = False
    for raw_line in section_text.splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if stripped.startswith("<details"):
            in_details = True
            continue
        if stripped.startswith("</details"):
            in_details = False
            continue
        def _add_candidate(label: Optional[str]) -> None:
            if not label:
                return
            if label in seen:
                return
            seen.add(label)
            sources.append(label)

        if "**sources:**" in lower:
            segment = raw_line.split("**Sources:**", 1)[1] if "**Sources:**" in raw_line else raw_line.split("**sources:**", 1)[1]
            parsed = parse_sources_links(segment)
            if parsed:
                for entry in parsed:
                    label = _sanitize_reference_label(entry.get("label", ""))
                    _add_candidate(label)
            else:
                for chunk in re.split(r"[;,]", segment):
                    chunk_label = _sanitize_reference_label(chunk.strip())
                    if not chunk_label:
                        chunk_label = _extract_label_from_line(chunk)
                    _add_candidate(chunk_label)
            continue
        if in_details and stripped.startswith("- ["):
            parsed = parse_sources_links(stripped)
            if parsed:
                for entry in parsed:
                    label = _sanitize_reference_label(entry.get("label", ""))
                    _add_candidate(label)
            else:
                label = _extract_label_from_line(stripped)
                _add_candidate(label)
    return sources


def _find_preceding_label(section_text: str, start: int) -> Optional[str]:
    prefix = section_text[:start]
    if not prefix:
        return None
    lines = prefix.rstrip("\n").splitlines()
    while lines:
        candidate_line = lines.pop().strip()
        label = _extract_label_from_line(candidate_line)
        if label:
            return label
        if candidate_line.startswith("```"):
            break
    return None


def _extract_code_blocks(section_text: str, sources_iter: Optional[Iterator[str]] = None) -> List[SectionBlock]:
    #import pdb; pdb.set_trace()
    sources_pattern = re.compile(r"\*\*Sources:\*\*", re.IGNORECASE)
    code_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    blocks: List[SectionBlock] = []
    # First, include already-hydrated snippet blocks, which look like:
    # path/to/file.py:10-20
    # ```lang
    # ...
    # ```
    hydrated_pattern = re.compile(
        r"(?m)^(?P<label>[A-Za-z0-9_.\-/]+(?::\d+(?:-\d+)?)?)\s*\n```(?P<lang>[^\n`]*)\n(?P<body>.*?)```",
        re.DOTALL,
    )
    seen_hydrated: Set[Tuple[str, str]] = set()
    for match in hydrated_pattern.finditer(section_text):
        label = _sanitize_reference_label(match.group("label") or "") or ""
        lang = (match.group("lang") or "").strip().lower() or "text"
        body = (match.group("body") or "").strip("\n")
        if not body:
            continue
        key = (label, body[:64])
        if key in seen_hydrated:
            continue
        seen_hydrated.add(key)
        blocks.append(
            SectionBlock(
                explanation=label,
                code=body,
                language=lang,
                mermaid=body if lang == "mermaid" else None,
            )
        )
    search_pos = 0
    while True:
        sources_match = sources_pattern.search(section_text, search_pos)
        if not sources_match:
            break
        next_sources = sources_pattern.search(section_text, sources_match.end())
        endpos = next_sources.start() if next_sources else len(section_text)
        code_match = code_pattern.search(section_text, sources_match.end(), endpos)
        if not code_match:
            search_pos = sources_match.end()
            continue
        language = (code_match.group(1) or "").strip().lower() or "text"
        code = code_match.group(2).strip("\n")
        if not code:
            search_pos = code_match.end()
            continue
        mermaid = None
        if language == "mermaid":
            mermaid = code
        label = None
        if sources_iter is not None:
            try:
                label = next(sources_iter)
            except StopIteration:
                sources_iter = None
        if not label:
            label = _find_preceding_label(section_text, code_match.start())
            label = _sanitize_reference_label(label)
        if not label:
            label = _infer_reference_from_code(code)
        blocks.append(
            SectionBlock(
                explanation=label or "",
                code=code,
                language=language,
                mermaid=mermaid,
            )
        )
        search_pos = code_match.end()
    return blocks


def _is_readme_like(label: str) -> bool:
    lowered = (label or "").strip().lower()
    return lowered.startswith("readme") or lowered.endswith((".md", ".rst", ".txt"))


def make_section_result(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    section_text: str,
    logic_config: Optional[NarrativeLLMConfig],
    critic_config: Optional[JudgeLLMConfig],
    judge_rounds: int,
) -> SectionResult:
    sources = _extract_section_sources(section_text)
    code_blocks = _extract_code_blocks(section_text, iter(sources))
    ranged_labels, unranged_labels = _extract_section_index_labels(section_text)
    bypass_llm = bool(logic_config and _should_bypass_llm_for_section(section_text))
    if bypass_llm:
        narrative = section_text.strip()
    else:
        narrative = rewrite_section(
            repo=repo,
            page_title=page_title,
            section_heading=section_heading,
            section_text=section_text,
            code_blocks=code_blocks,
            llm_config=logic_config,
            fallback_subject=section_heading,
        )
    logger.debug(
        "Section Draft[%s :: %s]: %s",
        page_title,
        section_heading,
        _truncate(narrative, 400),
    )
    critic_history: List[str] = []
    final_feedback = CriticFeedback(
        verdict="PASS",
        text="No critic feedback recorded.",
        misalignment=MisalignmentType.NONE,
    )
    rounds = max(1, judge_rounds if critic_config else 1)
    current_text = narrative
    if bypass_llm:
        critic_history.append("Skipped LLM rewrite/critic due to long hydrated section context.")
        final_feedback = CriticFeedback(
            verdict="PASS",
            text="Bypassed prompting due to long context; used raw hydrated section content.",
            misalignment=MisalignmentType.NONE,
        )
    else:
        for attempt in range(rounds):
            feedback = critique_section(
                repo=repo,
                page_title=page_title,
                section_heading=section_heading,
                narrative=current_text,
                code_blocks=code_blocks,
                judge_config=critic_config,
            )
            critic_history.append(feedback.text)
            final_feedback = feedback
            logger.debug(
                "Section Judge[%s :: %s :: pass %d]: verdict=%s misalignment=%s critic=%s",
                page_title,
                section_heading,
                attempt + 1,
                feedback.verdict,
                feedback.misalignment.value if feedback.misalignment else "none",
                _truncate(feedback.text, 300),
            )
            if feedback.verdict.upper() == "PASS" or not critic_config:
                break
            if attempt + 1 >= rounds:
                break
            current_text = refine_section(
                repo=repo,
                page_title=page_title,
                section_heading=section_heading,
                current_text=current_text,
                critic=feedback,
                code_blocks=code_blocks,
                llm_config=logic_config,
            )
            logger.debug(
                "Section Refine[%s :: %s :: pass %d]: %s",
                page_title,
                section_heading,
                attempt + 2,
                _truncate(current_text, 400),
            )
    learnability = compute_learnability_score(current_text)
    logger.debug(
        "Section Final[%s :: %s]: verdict=%s misalignment=%s learnability=%.3f",
        page_title,
        section_heading,
        final_feedback.verdict,
        final_feedback.misalignment.value if final_feedback.misalignment else "none",
        learnability,
    )
    augmented_narrative = current_text
    if not bypass_llm and code_blocks:
        appendix: List[str] = []
        selected_blocks: List[SectionBlock] = []

        # Append code for every index referenced in the prompt/section context.
        # Ranged indices are included, and unranged indices are also included (can be large).
        selected_labels: Set[str] = set(ranged_labels) | set(unranged_labels)

        def label_matches(block_label: str) -> bool:
            if not block_label:
                return False
            clean = _sanitize_reference_label(block_label) or ""
            if not clean:
                return False
            if clean in selected_labels:
                return True
            # If we selected a ranged label, accept exact path matches as well when the block contains full-file content.
            if ":" not in clean:
                for wanted in selected_labels:
                    if wanted.startswith(clean + ":") or wanted == clean:
                        return True
            return False

        for block in code_blocks:
            label = (block.explanation or "").strip()
            if not label_matches(label):
                continue
            # Keep readme-like sources too; they are part of the prompt context and requested for completeness.
            selected_blocks.append(block)

        for idx, block in enumerate(selected_blocks, 1):
            code_body = (block.code or "").strip()
            if not code_body:
                continue
            label = (block.explanation or "").strip() or f"code_block_{idx}"
            appendix.append(
                "\n".join(
                    [
                        f"Original code ({label}):",
                        f"```{block.language or 'text'}",
                        code_body,
                        "```",
                    ]
                )
            )
        if appendix:
            augmented_narrative = current_text.rstrip() + "\n\n" + "\n\n".join(appendix)
    return SectionResult(
        narrative=augmented_narrative,
        critic=final_feedback.text,
        verdict=final_feedback.verdict,
        misalignment=final_feedback.misalignment,
        learnability=learnability,
        critic_history=critic_history,
        code_blocks=list(code_blocks),
    )
