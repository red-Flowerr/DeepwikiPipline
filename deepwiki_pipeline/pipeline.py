"""Pipeline orchestration utilities for DeepWiki."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

try:  # Optional, for progress bar feedback
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .mcp import MCPError, Session, call_tool, extract_text_blocks
from .models import (
    DatasetChunk,
    CodeReference,
    JudgeLLMConfig,
    NarrativeLLMConfig,
    OutlineNode,
    PageContent,
    PipelineOutput,
    SubsectionResult,
    normalize_heading,
)
from .narrative import make_section_result
from .parsing import (
    parse_outline_text,
    parse_wiki_markdown,
    resolve_page,
)
from .hydration import hydrate_section_text

logger = logging.getLogger(__name__)


class DeepWikiPipeline:
    """End-to-end driver implementing the revised what–how–code pipeline."""

    def __init__(
        self,
        session: Optional[Session],
        repo: str,
        logic_llm_config: Optional[NarrativeLLMConfig] = None,
        critic_llm_config: Optional[JudgeLLMConfig] = None,
        *,
        repo_commit: Optional[str] = None,
        judge_rounds: int = 1,
        repo_root: Optional[Path] = None,
        allow_git_clone: bool = True,
        show_page_progress: bool = True,
        enable_hydration: bool = True,
        hydration_timeout: Optional[float] = None,
        hydration_workers: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_sections_per_page: Optional[int] = None,
        max_workers: Optional[int] = None,
        section_workers: Optional[int] = None,
        skip_pages: Optional[Sequence[str]] = None,
    ) -> None:
        self.session = session
        self.repo = repo
        self.logic_llm_config = logic_llm_config
        self.critic_llm_config = critic_llm_config
        self.repo_commit = repo_commit
        self.judge_rounds = judge_rounds
        self.repo_root = repo_root.resolve() if repo_root else None
        self.allow_git_clone = allow_git_clone
        self.show_page_progress = show_page_progress
        self.enable_hydration = bool(enable_hydration)
        self.hydration_timeout = (
            float(hydration_timeout)
            if hydration_timeout is not None and hydration_timeout > 0
            else None
        )
        self.hydration_workers = (
            hydration_workers if hydration_workers and hydration_workers > 0 else None
        )
        self.max_pages = max_pages if max_pages and max_pages > 0 else None
        self.max_sections_per_page = (
            max_sections_per_page if max_sections_per_page and max_sections_per_page > 0 else None
        )
        self.max_workers = max_workers if max_workers and max_workers > 0 else None
        self.section_workers = (
            section_workers if section_workers and section_workers > 0 else None
        )
        self._managed_repo_dir: Optional[Path] = None
        self._managed_repo_root: Optional[Path] = None
        self._active_repo_root: Optional[Path] = None
        self._skip_pages: Set[str] = {
            normalize_heading(name) for name in skip_pages or []
        }
        self._offline_outline_text: Optional[str] = None
        self._offline_wiki_markdown: Optional[str] = None

    # ------------------------------------------------------------------ #
    # MCP fetch helpers
    # ------------------------------------------------------------------ #

    def _validate_text_blocks(self, blocks: List[str], tool_name: str) -> List[str]:
        if not blocks:
            raise MCPError(f"{tool_name} returned no textual content.")
        for block in blocks:
            text = block.strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered.startswith("error fetching wiki for"):
                raise MCPError(text)
            if "repository not found" in lowered and "deepwiki" in lowered:
                raise MCPError(text)
            if lowered.startswith("deepwiki is currently indexing"):
                raise MCPError(text)
        return blocks

    def _fetch_structure_text(self) -> str:
        if self.session is None:
            raise MCPError("MCP session is required to fetch wiki structure.")
        result = call_tool(
            self.session,
            "read_wiki_structure",
            {
                "repoName": self.repo,
                **({"repoCommit": self.repo_commit} if self.repo_commit else {}),
            },
        )
        blocks = self._validate_text_blocks(
            extract_text_blocks(result),
            "read_wiki_structure",
        )
        return "\n\n".join(blocks)

    def _fetch_pages(self) -> Dict[str, PageContent]:
        if self.session is None:
            raise MCPError("MCP session is required to fetch wiki contents.")
        result = call_tool(
            self.session,
            "read_wiki_contents",
            {
                "repoName": self.repo,
                **({"repoCommit": self.repo_commit} if self.repo_commit else {}),
            },
        )
        blocks = self._validate_text_blocks(
            extract_text_blocks(result),
            "read_wiki_contents",
        )
        markdown = "\n\n".join(blocks)
        pages = parse_wiki_markdown(markdown)
        if not pages:
            raise MCPError("read_wiki_contents response did not contain any pages.")
        return pages

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run(
        self,
        progress_callback: Optional[Callable[[PipelineOutput], None]] = None,
        *,
        outline_text: Optional[str] = None,
        wiki_markdown: Optional[str] = None,
    ) -> PipelineOutput:
        self._offline_outline_text = outline_text
        self._offline_wiki_markdown = wiki_markdown
        repo_root = None
        try:
            repo_root = self._ensure_repo_checkout()
            self._active_repo_root = repo_root
            return self._run_with_repo(progress_callback=progress_callback)
        finally:
            self._active_repo_root = None
            self._offline_outline_text = None
            self._offline_wiki_markdown = None
            if repo_root and not self.repo_root:
                self._cleanup_repo_checkout()

    def _run_with_repo(
        self,
        progress_callback: Optional[Callable[[PipelineOutput], None]],
    ) -> PipelineOutput:
        judge_rounds = max(1, self.judge_rounds)
        if judge_rounds != self.judge_rounds:
            logger.warning(
                "Non-positive judge_rounds (%d) requested; defaulting to 1.",
                self.judge_rounds,
            )
        outline_text = self._offline_outline_text
        wiki_markdown = self._offline_wiki_markdown
        if outline_text is None or wiki_markdown is None:
            logger.info("Fetching wiki outline for %s", self.repo)
            outline_text = self._fetch_structure_text()
            logger.info("Fetching wiki content for %s", self.repo)
            pages = self._fetch_pages()
        else:
            pages = parse_wiki_markdown(wiki_markdown)
            if not pages:
                raise MCPError("Offline wiki markdown did not contain any pages.")
        outline_nodes = parse_outline_text(outline_text)
        if not outline_nodes:
            raise MCPError("DeepWiki outline response contained no entries.")

        dataset_chunks: List[DatasetChunk] = []
        subsection_results: List[SubsectionResult] = []

        def _flatten(nodes: List[OutlineNode]) -> List[OutlineNode]:
            flat: List[OutlineNode] = []
            for node in nodes:
                flat.append(node)
                if node.children:
                    flat.extend(_flatten(node.children))
            return flat

        seen_pages: set[str] = set()
        page_entries: List[OutlineNode] = []
        missing_pages: List[str] = []
        for page_node in _flatten(outline_nodes):
            page_key = normalize_heading(page_node.title)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            if page_key in self._skip_pages:
                logger.info("Skipping page %s due to skip-pages configuration.", page_node.title)
                continue
            if page_key not in pages:
                missing_pages.append(page_node.title)
                logger.warning(
                    "Skipping outline page %s because it is missing from wiki content.",
                    page_node.title,
                )
                continue
            if self.max_pages is not None and len(page_entries) >= self.max_pages:
                break
            page_entries.append(page_node)

        if not page_entries:
            raise MCPError("No eligible pages were discovered in the wiki outline.")
        if missing_pages:
            logger.info(
                "Skipped %d outline pages missing from wiki content (first few: %s).",
                len(missing_pages),
                ", ".join(missing_pages[:5]),
            )

        page_results: Dict[int, Tuple[Optional[DatasetChunk], List[SubsectionResult]]] = {}
        next_index_to_emit = 0
        indexed_entries: List[Tuple[int, OutlineNode]] = list(enumerate(page_entries))
        max_workers = self.max_workers or min(32, len(indexed_entries))
        section_workers = self.section_workers or 1
        hydration_timeout = self.hydration_timeout
        hydration_workers = self.hydration_workers or 4

        completed_pages = 0
        total_pages = len(indexed_entries)
        progress_bar = (
            tqdm(
                total=total_pages,
                desc=f"{self.repo} pages",
                unit="page",
                leave=False,
            )
            if self.show_page_progress and tqdm and total_pages > 0
            else None
        )
        section_executor_cm = (
            ThreadPoolExecutor(max_workers=section_workers) if section_workers > 1 else nullcontext()
        )
        hydration_executor_cm = (
            ThreadPoolExecutor(max_workers=hydration_workers)
            if hydration_timeout and self.enable_hydration and self._active_repo_root
            else nullcontext()
        )
        hydration_semaphore = (
            threading.Semaphore(hydration_workers)
            if hydration_timeout and self.enable_hydration and self._active_repo_root
            else None
        )
        with (
            section_executor_cm as section_executor,
            hydration_executor_cm as hydration_executor,
            ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            futures = {}
            for index, page_node in indexed_entries:
                futures[
                    executor.submit(
                        self._process_page,
                        page_node,
                        pages,
                        judge_rounds,
                        section_executor if section_workers > 1 else None,
                        hydration_executor if hydration_semaphore is not None else None,
                        hydration_semaphore,
                    )
                ] = index

            for future in as_completed(futures):
                index = futures[future]
                try:
                    page_results[index] = future.result()
                except Exception as exc:
                    page_title = indexed_entries[index][1].title
                    logger.exception(
                        "Page processing failed for %s: %s",
                        page_title,
                        exc,
                    )
                    raise
                while next_index_to_emit in page_results:
                    chunk, page_sections = page_results.pop(next_index_to_emit)
                    if page_sections:
                        subsection_results.extend(page_sections)
                        if chunk:
                            dataset_chunks.append(chunk)
                            if progress_callback:
                                try:
                                    progress_callback(
                                        PipelineOutput(
                                            repo=self.repo,
                                            chunks=list(dataset_chunks),
                                            subsections=list(subsection_results),
                                        )
                                    )
                                except Exception as exc:  # pragma: no cover - callback failures
                                    logger.warning(
                                        "Progress callback failed for page %s: %s",
                                        chunk.label,
                                        exc,
                                    )
                    next_index_to_emit += 1
                completed_pages += 1
                if progress_bar:
                    progress_bar.update(1)
                else:
                    if total_pages:
                        if completed_pages == total_pages or completed_pages % max(
                            1, total_pages // 10
                        ) == 0:
                            logger.info(
                                "Processed %d/%d pages for %s",
                                completed_pages,
                                total_pages,
                                self.repo,
                            )
        if progress_bar:
            progress_bar.close()

        if not dataset_chunks:
            raise MCPError(
                "No eligible sections with code snippets were found in the wiki content."
            )

        logger.info(
            "Pipeline assembled %d subsection entries for %s",
            len(dataset_chunks),
            self.repo,
        )
        return PipelineOutput(
            repo=self.repo,
            chunks=dataset_chunks,
            subsections=subsection_results,
        )

    def _process_page(
        self,
        page_node: OutlineNode,
        pages: Dict[str, PageContent],
        judge_rounds: int,
        section_executor: Optional[ThreadPoolExecutor],
        hydration_executor: Optional[ThreadPoolExecutor],
        hydration_semaphore: Optional[threading.Semaphore],
    ) -> Tuple[Optional[DatasetChunk], List[SubsectionResult]]:
        page = resolve_page(pages, page_node.title)
        section_keys: List[str] = list(page.order)
        if self.max_sections_per_page is not None:
            section_keys = section_keys[: self.max_sections_per_page]

        indexed_sections: List[Tuple[int, str]] = []
        for idx, section_key in enumerate(section_keys):
            if section_key not in page.sections:
                logger.debug(
                    "Skipping %s :: section key %s (not found in page content).",
                    page.title,
                    section_key,
                )
                continue
            indexed_sections.append((idx, section_key))

        def process_section(index: int, section_key: str) -> Tuple[int, Optional[SubsectionResult]]:
            section_content = page.sections.get(section_key)
            if not section_content:
                return index, None
            section_heading = section_content.heading or page.title
            logger.debug(
                "Processing page %s :: section %s",
                page.title,
                section_heading,
            )
            raw_section_text = section_content.text
            hydrated_context = raw_section_text
            repo_root = self._active_repo_root
            if repo_root and self.enable_hydration:
                if hydration_executor is None or hydration_semaphore is None:
                    try:
                        hydrated_context = hydrate_section_text(
                            raw_section_text,
                            repo_root=repo_root,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "Hydration failed for %s :: %s: %s",
                            page.title,
                            section_heading,
                            exc,
                        )
                else:
                    if not hydration_semaphore.acquire(blocking=False):
                        logger.debug(
                            "Skipping hydration for %s :: %s (hydration pool saturated).",
                            page.title,
                            section_heading,
                        )
                    else:
                        def _hydrate_task() -> str:
                            try:
                                return hydrate_section_text(raw_section_text, repo_root=repo_root)
                            finally:
                                hydration_semaphore.release()

                        try:
                            hydrate_future = hydration_executor.submit(_hydrate_task)
                        except Exception:
                            hydration_semaphore.release()
                            raise
                        try:
                            hydrated_context = hydrate_future.result(timeout=self.hydration_timeout)
                        except Exception as exc:  # includes TimeoutError
                            logger.warning(
                                "Hydration skipped for %s :: %s: %s",
                                page.title,
                                section_heading,
                                exc,
                            )
            section_result = make_section_result(
                repo=self.repo,
                page_title=page.title,
                section_heading=section_heading,
                section_text=hydrated_context,
                logic_config=self.logic_llm_config,
                critic_config=self.critic_llm_config,
                judge_rounds=judge_rounds,
            )
            code_refs = [
                CodeReference(
                    reference=block.explanation or "",
                    code=block.code,
                )
                for block in section_result.code_blocks
            ]
            subsection = SubsectionResult(
                repo=self.repo,
                page_title=page.title,
                section_heading=section_heading,
                narrative=section_result.narrative,
                critic=section_result.critic,
                verdict=section_result.verdict,
                misalignment=section_result.misalignment,
                learnability=section_result.learnability,
                critic_history=section_result.critic_history,
                code_blocks=code_refs,
                original_context=hydrated_context.strip(),
            )
            return index, subsection

        page_sections: List[SubsectionResult] = []
        if section_executor is None or len(indexed_sections) <= 1:
            for idx, section_key in indexed_sections:
                _, subsection = process_section(idx, section_key)
                if subsection:
                    page_sections.append(subsection)
        else:
            futures = {}
            for idx, section_key in indexed_sections:
                future = section_executor.submit(process_section, idx, section_key)
                futures[future] = section_key
            indexed_results: List[Tuple[int, SubsectionResult]] = []
            for future in as_completed(futures):
                section_key = futures[future]
                try:
                    idx, subsection = future.result()
                except Exception as exc:
                    logger.exception(
                        "Section processing failed for %s :: %s: %s",
                        page.title,
                        section_key,
                        exc,
                    )
                    raise
                if subsection:
                    indexed_results.append((idx, subsection))
            indexed_results.sort(key=lambda item: item[0])
            page_sections = [subsection for _, subsection in indexed_results]
        if not page_sections:
            return None, page_sections
        chunk = DatasetChunk(
            label=normalize_heading(page.title),
            text=_format_page(page.title, page_sections),
        )
        return chunk, page_sections

    def _ensure_repo_checkout(self) -> Optional[Path]:
        if self.repo_root:
            if not self.repo_root.exists():
                raise MCPError(f"Provided repo root does not exist: {self.repo_root}")
            return self.repo_root
        if self._managed_repo_root:
            return self._managed_repo_root
        if not self.allow_git_clone:
            raise MCPError(
                "GitHub cloning is disabled. Provide `repo_root` (e.g. extracted zip directory) "
                "or use the parquet workflow that supplies repository sources via `hdfs_path`."
            )
        if not shutil.which("git"):
            raise MCPError("git is required to clone repository sources.")
        temp_parent = Path(tempfile.mkdtemp(prefix="deepwiki_repo_"))
        clone_target = temp_parent / "repo"
        repo_url = f"https://github.com/{self.repo}.git"
        logger.info("Cloning %s into %s", repo_url, clone_target)
        clone_cmd = ["git", "clone", "--filter=blob:none", repo_url, str(clone_target)]
        try:
            subprocess.run(
                clone_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if self.repo_commit:
                subprocess.run(
                    ["git", "-C", str(clone_target), "checkout", self.repo_commit],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(temp_parent, ignore_errors=True)
            message = exc.stderr.strip() if exc.stderr else str(exc)
            raise MCPError(f"Failed to clone {self.repo}: {message}") from exc
        self._managed_repo_dir = temp_parent
        self._managed_repo_root = clone_target
        return clone_target

    def _cleanup_repo_checkout(self) -> None:
        if self._managed_repo_dir:
            shutil.rmtree(self._managed_repo_dir, ignore_errors=True)
        self._managed_repo_dir = None
        self._managed_repo_root = None


def _format_subsection(result: SubsectionResult) -> str:
    lines: List[str] = []
    lines.append(f"[Section] {result.section_heading}")
    lines.append("")
    if result.original_context:
        lines.append("Original Context:")
        lines.append(result.original_context.strip())
        lines.append("")
    lines.append("Narrative:")
    lines.append(result.narrative.strip())
    lines.append("")
    lines.append("Critic:")
    lines.append(result.critic.strip())
    lines.append("")
    lines.append(
        f"Metadata: verdict={result.verdict} "
        f"misalignment={result.misalignment.value if result.misalignment else 'none'} "
        f"learnability={result.learnability:.3f}"
    )
    if result.critic_history:
        lines.append(f"critic_history={result.critic_history}")
    return "\n".join(lines).strip()


def _format_page(page_title: str, sections: List[SubsectionResult]) -> str:
    lines: List[str] = []
    lines.append(f"[Page] {page_title}")
    lines.append("")
    for subsection in sections:
        lines.append(_format_subsection(subsection))
        lines.append("")
    return "\n".join(lines).strip()
