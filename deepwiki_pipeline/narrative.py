"""LLM helpers for the DeepWiki semantic pipeline."""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .models import (
    BlockResult,
    CriticFeedback,
    JudgeLLMConfig,
    MisalignmentType,
    NarrativeLLMConfig,
    NarrativeScaffold,
    SectionBlock,
)
from .parsing import extract_summary_paragraph

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
    "You are reconstructing the author's thinking process at the moment this code was written.\n\n"
    "You are given:\n"
    "- Explanatory text from a project wiki (high-level, post-hoc descriptions)\n"
    "- One or more related code snippets\n\n"
    "Your task is NOT to summarize the wiki or explain how the code works.\n\n"
    "Instead, infer and articulate the design-time reasoning that likely preceded the code:\n"
    "- What problems or constraints the author was facing\n"
    "- What options they likely considered and rejected\n"
    "- What trade-offs or assumptions shaped the final structure\n"
    "- Why this specific form was chosen over simpler or more obvious alternatives\n\n"
    "Write from the perspective of someone about to implement the code, thinking aloud internally.\n\n"
    "Guidelines:\n"
    "- Do not restate wiki explanations or architecture descriptions.\n"
    "- Do not describe the code line by line.\n"
    "- Focus on decision-making, not outcomes.\n"
    "- It is acceptable to make reasonable assumptions if they are implied by the code.\n"
    "- Write in natural technical prose, as if explaining your reasoning to another senior engineer before coding.\n"
    "- Do not use Markdown tables, bullet lists, or other rigid formatting; express trade-offs in flowing prose.\n\n"
    "You may reference the code abstractly, and you may insert short inline comments (e.g. “at this point I realized…”) "
    "to anchor the reasoning, but do not paste or quote large code blocks.\n\n"
    "The output should read like a design notebook entry written before the code existed."
)

CRITIC_SYSTEM_PROMPT = (
    "You are evaluating whether the following text reflects genuine design-time coding reasoning.\n\n"
    "Criteria:\n"
    "- Does the text describe constraints, trade-offs, or uncertainties faced before implementation?\n"
    "- Does it avoid restating documentation or explaining the finished system?\n"
    "- Does it focus on decisions rather than describing what the code does?\n"
    "- Does it read like a developer thinking through a problem, not teaching it?\n\n"
    "If the text mainly:\n"
    "- Summarizes wiki content\n"
    "- Explains architecture after the fact\n"
    "- Describes code behavior instead of decision rationale\n\n"
    "Then it is misaligned.\n\n"
    "Respond with:\n"
    "PASS or FAIL\n\n"
    "If FAIL, briefly explain what kind of reasoning is missing or what the text does instead."
)

REFINEMENT_REMINDER = (
    "Keep the voice anchored in pre-implementation reasoning. Preserve lines that already capture constraints, trade-offs, "
    "and sequencing, and revise only the gaps flagged by the critic."
)

SECTION_REWRITE_SYSTEM_PROMPT = BLOCK_REWRITE_SYSTEM_PROMPT

SECTION_CRITIC_SYSTEM_PROMPT = CRITIC_SYSTEM_PROMPT


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
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}

        Hydrated section contents (original prose + code fences):
        {section_text.strip()}

        Reconstruct the author's reasoning before this work existed.
        Concentrate on constraints, discarded options, trade-offs, sequencing, and open risks. Keep it as a planning monologue.
        Do not restate the wiki or describe completed behavior, and avoid Markdown tables or bullet lists—use flowing prose instead.
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
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        logger.warning(
            "Section rewrite failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return fallback
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
            api_key=judge_config.api_key,
            destination_service=judge_config.destination_service,
            timeout=judge_config.timeout,
            retries=judge_config.retries,
            retry_backoff=judge_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
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
            api_key=judge_config.api_key,
            destination_service=judge_config.destination_service,
            timeout=judge_config.timeout,
            retries=judge_config.retries,
            retry_backoff=judge_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
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
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        logger.warning(
            "Refinement failed for %s :: %s (block %d): %s",
            page_title,
            section_heading,
            block_index,
            exc,
        )
        return current_text
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
            api_key=llm_config.api_key,
            destination_service=llm_config.destination_service,
            timeout=llm_config.timeout,
            retries=llm_config.retries,
            retry_backoff=llm_config.retry_backoff,
        )
    except VLLMError as exc:  # pragma: no cover
        logger.warning(
            "Section refinement failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return current_text
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
    logger.info(
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
        logger.info(
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
        logger.info(
            "Refine[%s :: %s :: block %d :: pass %d]: %s",
            page_title,
            section_heading,
            block_index,
            attempt + 2,
            _truncate(current_text, 400),
        )

    final_scaffold = build_scaffold(current_text, block)
    learnability = compute_learnability_score(current_text)
    logger.info(
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


_CODE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.\-/]+(?::\d+(?:-\d+)?)?$")


def _find_preceding_label(section_text: str, start: int) -> Optional[str]:
    prefix = section_text[:start]
    if not prefix:
        return None
    lines = prefix.rstrip("\n").splitlines()
    while lines:
        candidate = lines.pop().strip()
        if not candidate:
            continue
        if candidate.startswith("```"):
            break
        if candidate.startswith("- "):
            candidate = candidate[2:].strip()
        if candidate.startswith("* "):
            candidate = candidate[2:].strip()
        if candidate.startswith("• "):
            candidate = candidate[2:].strip()
        normalized = candidate
        lowered = normalized.lower()
        if lowered.startswith("**sources:**"):
            normalized = normalized.split("**Sources:**", 1)[1].strip()
            lowered = normalized.lower()
        if lowered.startswith("sources:"):
            normalized = normalized.split(":", 1)[1].strip()
            lowered = normalized.lower()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1].strip()
            lowered = normalized.lower()
        if lowered.startswith("source:"):
            normalized = normalized.split(":", 1)[1].strip()
        if _CODE_LABEL_RE.match(normalized):
            return normalized
        for token in re.split(r"[,\s]+", normalized):
            stripped_token = token.strip()
            if not stripped_token:
                continue
            if _CODE_LABEL_RE.match(stripped_token):
                return stripped_token
        # stop if we encountered other text before label
        if not candidate:
            continue
        break
    return None


def _extract_code_blocks(section_text: str) -> List[SectionBlock]:
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    blocks: List[SectionBlock] = []
    for match in pattern.finditer(section_text):
        language = (match.group(1) or "").strip().lower() or "text"
        code = match.group(2).strip("\n")
        if not code:
            continue
        mermaid = None
        if language == "mermaid":
            mermaid = code
        label = _find_preceding_label(section_text, match.start())
        blocks.append(
            SectionBlock(
                explanation=label or "",
                code=code,
                language=language,
                mermaid=mermaid,
            )
        )
    return blocks


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
    code_blocks = _extract_code_blocks(section_text)
    narrative = rewrite_section(
        repo=repo,
        page_title=page_title,
        section_heading=section_heading,
        section_text=section_text,
        code_blocks=code_blocks,
        llm_config=logic_config,
        fallback_subject=section_heading,
    )
    logger.info(
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
        logger.info(
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
        logger.info(
            "Section Refine[%s :: %s :: pass %d]: %s",
            page_title,
            section_heading,
            attempt + 2,
            _truncate(current_text, 400),
        )
    learnability = compute_learnability_score(current_text)
    logger.info(
        "Section Final[%s :: %s]: verdict=%s misalignment=%s learnability=%.3f",
        page_title,
        section_heading,
        final_feedback.verdict,
        final_feedback.misalignment.value if final_feedback.misalignment else "none",
        learnability,
    )
    references: List[str] = []
    for block in code_blocks:
        reference = block.explanation.strip()
        code_body = block.code.strip()
        if not reference or not code_body:
            continue
        if "readme" in reference.lower():
            continue
        file_name = reference.rsplit("/", 1)[-1]
        snippet = (
            f"With that plan in mind, I carried the implementation into {file_name}:\n"
            f"{code_body}"
        )
        references.append(snippet)
    augmented_narrative = current_text
    if references:
        augmented_narrative = current_text.rstrip() + "\n\n" + "\n\n".join(references)
    return SectionResult(
        narrative=augmented_narrative,
        critic=final_feedback.text,
        verdict=final_feedback.verdict,
        misalignment=final_feedback.misalignment,
        learnability=learnability,
        critic_history=critic_history,
        code_blocks=list(code_blocks),
    )
