"""Dataclasses supporting the DeepWiki semantic pipeline."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Narrative scaffolding and critique metadata
# ---------------------------------------------------------------------------


class MisalignmentType(str, Enum):
    NONE = "none"
    NO_INTENT = "code_without_purpose"
    NO_MECHANISM = "intent_without_how"
    JUMP_TO_CODE = "abrupt_implementation"
    MISPLACED_DETAIL = "low_level_too_early"
    INCONSISTENT = "intent_code_mismatch"


@dataclass
class NarrativeScaffold:
    design_intent: str
    constraints: List[str]
    key_mechanisms: List[str]
    code_anchors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design_intent": self.design_intent,
            "constraints": list(self.constraints),
            "key_mechanisms": list(self.key_mechanisms),
            "code_anchors": list(self.code_anchors),
        }


# ---------------------------------------------------------------------------
# Critic feedback structures
# ---------------------------------------------------------------------------


@dataclass
class CriticFeedback:
    verdict: str
    text: str
    misalignment: MisalignmentType = MisalignmentType.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "text": self.text,
            "misalignment": self.misalignment.value,
        }


# Chunk-level abstractions
# ---------------------------------------------------------------------------


@dataclass
class SectionBlock:
    explanation: str
    code: str
    language: str
    mermaid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation": self.explanation,
            "code": self.code,
            "language": self.language,
            "mermaid": self.mermaid,
        }


@dataclass
class CodeReference:
    reference: str
    code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "code": self.code,
        }


@dataclass
class BlockResult:
    index: int
    rewrite: str
    critic: str
    block: SectionBlock
    scaffold: Optional["NarrativeScaffold"] = None
    verdict: str = "UNKNOWN"
    misalignment: Optional["MisalignmentType"] = None
    learnability: float = 0.0
    critic_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "rewrite": self.rewrite,
            "critic": self.critic,
            "block": self.block.to_dict(),
            "scaffold": self.scaffold.to_dict() if self.scaffold else None,
            "verdict": self.verdict,
            "misalignment": self.misalignment.value if self.misalignment else None,
            "learnability": self.learnability,
            "critic_history": list(self.critic_history),
        }


@dataclass
class SubsectionResult:
    repo: str
    page_title: str
    section_heading: str
    narrative: str
    critic: str
    verdict: str
    misalignment: Optional[MisalignmentType]
    learnability: float
    critic_history: List[str]
    code_blocks: List[CodeReference] = field(default_factory=list)
    original_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "page": self.page_title,
            "section": self.section_heading,
            "narrative": self.narrative,
            "critic": self.critic,
            "verdict": self.verdict,
            "misalignment": self.misalignment.value if self.misalignment else None,
            "learnability": self.learnability,
            "critic_history": list(self.critic_history),
            "code_blocks": [block.to_dict() for block in self.code_blocks],
            "original_context": self.original_context,
        }


@dataclass
class DatasetChunk:
    label: str
    text: str


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------


@dataclass
class NarrativeLLMConfig:
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
    server_url: Optional[str] = None
    server_urls: Optional[List[str]] = None


@dataclass
class JudgeLLMConfig:
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
    server_url: Optional[str] = None
    server_urls: Optional[List[str]] = None
    system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline level containers
# ---------------------------------------------------------------------------


@dataclass
class PipelineOutput:
    repo: str
    chunks: List[DatasetChunk]
    subsections: List[SubsectionResult] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [chunk.text.strip() for chunk in self.chunks if chunk.text.strip()]
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "chunks": [
                {"label": chunk.label, "text": chunk.text} for chunk in self.chunks
            ],
            "subsections": [item.to_dict() for item in self.subsections],
        }
