"""LLM helpers for the DeepWiki semantic pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .models import (
    BlockResult,
    CriticFeedback,
    InstructionPair,
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
    "You are given a mixture of source code, documentation fragments, and system-level descriptions "
    "extracted from a software repository.\n\n"
    "Your task is to rewrite them into a single, coherent, natural narrative that reflects how a competent "
    "engineer would explain the system while developing or reviewing it.\n\n"
    "The narrative should flow naturally and implicitly follow this reasoning order:\n"
    "- first establish the purpose, constraints, and design intent behind the code,\n"
    "- then explain how the design is realized at a structural or algorithmic level,\n"
    "- and only then expose the concrete implementation details through code.\n\n"
    "Do not shorten the material. When helpful, expand the explanation with additional clarifying sentences so long as they remain faithful to the source.\n"
    "Write in textbook-style prose: continuous sentences grouped into short paragraphs.\n"
    "Do NOT introduce explicit section headers, labels, tables, bullet lists, ASCII diagrams, or other rigid formats.\n"
    "Do NOT enclose material in fenced code blocks; reference code inline with backticks when necessary.\n"
    "Do NOT turn the content into a tutorial or Q&A.\n"
    "Do NOT invent functionality that is not present in the original materials.\n\n"
    "When code appears, it should feel justified by the preceding explanation, as if the reader already understands "
    "why this code must exist before seeing it.\n\n"
    "Preserve technical accuracy, module boundaries, and dependency relationships.\n"
    "Prefer concise but information-dense explanations over verbosity."
)

CRITIC_SYSTEM_PROMPT = (
    "You are acting as a critical reviewer for pretraining data quality.\n\n"
    "You will be given a text segment that may include source code.\n"
    "Your task is to evaluate whether the segment implicitly follows a coherent reasoning flow suitable for large "
    "language model pretraining.\n\n"
    "Specifically, assess whether the content naturally establishes:\n"
    "- what problem or responsibility the code addresses,\n"
    "- how the solution is designed or structured,\n"
    "- and how the source code concretely realizes that design.\n\n"
    "Do NOT require explicit markers or section labels.\n"
    "Judge only the implicit logical flow and narrative coherence.\n\n"
    "If the segment is weak, identify exactly where and why the reasoning breaks down "
    "(e.g. code appears without motivation, design jumps are unexplained, or intent is unclear).\n\n"
    "Your output MUST include:\n"
    "1. A brief verdict: PASS or FAIL\n"
    "2. A concise critic explaining the main issue(s), written as feedback to a dataset engineer\n\n"
    "Do NOT rewrite the content.\n"
    "Do NOT suggest stylistic improvements unless they affect reasoning clarity.\n"
    "Focus strictly on reasoning structure and alignment between intent, design, and code."
)

REFINEMENT_REMINDER = (
    "Preserve all portions of the existing narrative that already provide accurate context. "
    "Only revise the specific logical gaps highlighted by the critic, keeping the text concise and engineering-focused."
)

SECTION_REWRITE_SYSTEM_PROMPT = (
    "You are rewriting mixed technical content that includes both natural-language explanations and source code "
    "for an experienced software engineer who needs to understand design intent, architectural choices, and runtime behavior.\n\n"
    "Do not summarize or shorten. Preserve all technical detail and feel free to expand with inferred rationale, implicit constraints, "
    "and operational assumptions that are only hinted at in the original materials.\n\n"
    "Prioritise enrichment over compression: if a concept seems terse, elaborate with concrete explanations that stay faithful to the source.\n\n"
    "Produce a design-oriented walkthrough that covers, in order, the DESIGN MOTIVATION (WHY), the DESIGN MECHANISM (HOW), "
    "and the SYSTEM ROLE & GUARANTEES (CONTRACT). Present these sections as fluid paragraphs rather than headings or bullet lists.\n"
    "Explicitly name each facet within the prose (e.g., \"First, the design motivation...\"), but avoid Markdown headings, numbered lists, or tables.\n\n"
    "Write in textbook-style paragraphs. Do not output ASCII diagrams, fenced code blocks, or formatted tables. "
    "When referencing code, quote only the indispensable identifiers or expressions inline using backticks and immediately explain their intent.\n\n"
    "The final output must read like a cohesive narrative where explanation and code reinforce each other without relying on structural markup."
)

SECTION_CRITIC_SYSTEM_PROMPT = (
    "You review a section-level narrative meant for pretraining. Judge whether it achieves:\n"
    "- Clear DESIGN MOTIVATION (WHY) with constraints/trade-offs made explicit.\n"
    "- Accurate DESIGN MECHANISM (HOW) with annotated code snippets that justify implementation decisions.\n"
    "- Explicit SYSTEM ROLE & GUARANTEES explaining contracts, assumptions, and integration points.\n\n"
    "Critique failures where any dimension is weak, missing, or contradicted by the code. Respond with PASS or FAIL and explain issues clearly."
)

SFT_QA_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sft_qa_system.txt"
SFT_QA_USER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sft_qa_user.txt"
SFT_QA_SYSTEM_PROMPT = SFT_QA_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
SFT_QA_USER_PROMPT = SFT_QA_USER_PROMPT_PATH.read_text(encoding="utf-8")


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


def _extract_json_array(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        fence_end = stripped.rfind("```")
        if fence_end != -1:
            inner = stripped[3:fence_end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            stripped = inner
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return None


def generate_instruction_pairs(
    *,
    repo: str,
    page_title: str,
    section_heading: str,
    narrative: str,
    section_text: str,
    code_blocks: Sequence[SectionBlock],
    llm_config: Optional[NarrativeLLMConfig],
    system_prompt: Optional[str],
    user_template: Optional[str],
) -> List[InstructionPair]:
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return []
    code_text = _truncate(_join_code_blocks(code_blocks), 6000) if code_blocks else "(no code snippets detected)"
    prompt_template = (user_template or SFT_QA_USER_PROMPT)
    user_prompt = prompt_template.format(
        repo=repo,
        page_title=page_title,
        section_heading=section_heading,
        context=_truncate(section_text, 6000),
        narrative=_truncate(narrative, 3500),
        code_snippets=code_text,
    )
    system_message = (system_prompt or SFT_QA_SYSTEM_PROMPT).strip()
    messages = [
        ChatMessage(role="system", content=system_message),
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
    except VLLMError as exc:  # pragma: no cover - network dependent
        logger.warning(
            "QA generation failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return []
    payload = _extract_json_array(response) or ""
    if not payload:
        logger.warning(
            "QA generation returned non-JSON payload for %s :: %s: %s",
            page_title,
            section_heading,
            _truncate(response, 200),
        )
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning(
            "QA generation JSON decode failed for %s :: %s: %s",
            page_title,
            section_heading,
            exc,
        )
        return []
    if not isinstance(data, list):
        logger.warning(
            "QA generation payload not a list for %s :: %s: %s",
            page_title,
            section_heading,
            type(data),
        )
        return []
    pairs: List[InstructionPair] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        output = str(item.get("output") or "").strip()
        if not instruction or not output:
            continue
        input_text = str(item.get("input") or "").strip()
        category = item.get("category")
        if category:
            category = str(category).strip()
        pairs.append(
            InstructionPair(
                instruction=instruction,
                output=output,
                input=input_text,
                category=category if category else None,
            )
        )
    return pairs


@dataclass
class SectionResult:
    narrative: str
    critic: str
    verdict: str
    misalignment: Optional[MisalignmentType]
    learnability: float
    critic_history: List[str]
    code_blocks: List[SectionBlock]
    instruction_pairs: List[InstructionPair]


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
    fallback_lines = []
    if explanation:
        fallback_lines.append(explanation)
    fallback_lines.append(
        "The code snippet below implements this behaviour:"
    )
    fallback_lines.append(f"```{block.language}\n{block.code}\n```")
    fallback = _sanitize_visible_text("\n\n".join(fallback_lines).strip())
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

        Rewrite this material as textbook-style prose that preserves every technical detail and expands with any clarifying context needed for readability.
        Do not shorten the content relative to the original notes; add concise supporting sentences when they help the reader follow the reasoning.
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
) -> str:
    fallback = _sanitize_visible_text(section_text)
    if not fallback:
        fallback = f"{section_heading or page_title} provides implementation details; refer to the code below."
    if not llm_config or not call_vllm_chat or not ChatMessage:
        return fallback
    user_prompt = textwrap.dedent(
        f"""\
        Repository: {repo}
        Page: {page_title}
        Section: {section_heading}

        Hydrated section contents (original prose + code fences):
        {section_text.strip()}

        Rewrite this material into a design walkthrough that follows the WHY/ HOW / CONTRACT structure described in the system prompt.
        Integrate the essential code snippets with inline comments that explain intent and constraints, and preserve every critical technical detail.
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

        Produce an updated narrative (2-3 sentences) that resolves the critic feedback without inventing new behaviour.
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

        Produce a revised walkthrough that reinstates the WHY / HOW / CONTRACT structure, weaving in the essential code with inline comments as required by the system prompt.
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
        if _CODE_LABEL_RE.match(candidate):
            return candidate
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
    qa_config: Optional[NarrativeLLMConfig],
    qa_system_prompt: Optional[str],
    qa_user_prompt: Optional[str],
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
    return SectionResult(
        narrative=current_text,
        critic=final_feedback.text,
        verdict=final_feedback.verdict,
        misalignment=final_feedback.misalignment,
        learnability=learnability,
        critic_history=critic_history,
        code_blocks=list(code_blocks),
        instruction_pairs=generate_instruction_pairs(
            repo=repo,
            page_title=page_title,
            section_heading=section_heading,
            narrative=current_text,
            section_text=section_text,
            code_blocks=code_blocks,
            llm_config=qa_config,
            system_prompt=qa_system_prompt,
            user_template=qa_user_prompt,
        )
        if qa_config
        else [],
    )
