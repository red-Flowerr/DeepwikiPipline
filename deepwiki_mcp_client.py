#!/usr/bin/env python3
"""Command-line interface for the DeepWiki MCP pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepwiki_pipeline import (
    DeepWikiPipeline,
    MCPError,
    Session,
    delete_session,
    extract_text_blocks,
    initialize_session,
    list_tools,
)
from deepwiki_pipeline.mcp import call_tool
from deepwiki_pipeline.models import JudgeLLMConfig, NarrativeLLMConfig, PipelineOutput
from deepwiki_pipeline.parsing import normalize_heading, parse_wiki_markdown

logger = logging.getLogger(__name__)


def _print_tools(session: Session) -> None:
    response = list_tools(session)
    tools = response.get("result", {}).get("tools", [])
    if not tools:
        print("No tools reported by the server.")
        return
    for tool in tools:
        name = tool.get("name")
        description = tool.get("description", "")
        print(f"- {name}: {description}")


def _print_structure(session: Session, repo: str) -> None:
    response = call_tool(session, "read_wiki_structure", {"repoName": repo})
    blocks = extract_text_blocks(response)
    if not blocks:
        raise MCPError("read_wiki_structure returned no textual content.")
    print("\n\n".join(blocks))


def _apply_content_filters(
    markdown: str,
    *,
    page: Optional[str],
    section: Optional[str],
    contains: Optional[str],
) -> str:
    if not page and not section and not contains:
        return markdown
    pages = parse_wiki_markdown(markdown)
    result_chunks: List[str] = []
    contains_lower = contains.lower() if contains else None

    def page_matches(key: str) -> bool:
        if not page:
            return True
        target_norm = normalize_heading(page)
        return key == target_norm

    for key, page_content in pages.items():
        
        if not page_matches(key):
            continue
        if section is None:
            text = page_content.full_text()
            if contains_lower and contains_lower not in text.lower():
                continue
            result_chunks.append(f"# {page_content.title}\n\n{text}")
            continue
        try:
            section_content = page_content.section_text(section)
        except KeyError:
            continue
        text = section_content.text
        if contains_lower and contains_lower not in text.lower():
            continue
        heading = section_content.heading or section
        result_chunks.append(
            f"# {page_content.title} :: {heading}\n\n{text.strip()}"
        )
    if not result_chunks:
        raise MCPError("No content matched the requested filters.")
    return "\n\n".join(result_chunks)


def _print_contents(
    session: Session,
    repo: str,
    *,
    page: Optional[str],
    section: Optional[str],
    contains: Optional[str],
) -> None:
    response = call_tool(session, "read_wiki_contents", {"repoName": repo})
    blocks = extract_text_blocks(response)
    if not blocks:
        raise MCPError("read_wiki_contents returned no textual content.")
    markdown = "\n\n".join(blocks)
    filtered = _apply_content_filters(
        markdown,
        page=page,
        section=section,
        contains=contains,
    )
    print(filtered)


def _ask_question_json(
    session: Session,
    repo: str,
    question: str,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_payload = ""
    for _ in range(retries + 1):
        response = call_tool(
            session,
            "ask_question",
            {"repoName": repo, "question": question},
        )
        blocks = extract_text_blocks(response)
        if not blocks:
            last_error = MCPError("ask_question returned no textual content.")
            continue
        payload = "\n".join(blocks)
        try:
            return json.loads(_extract_json_from_text(payload))
        except json.JSONDecodeError as exc:
            last_error = exc
            last_payload = payload
    if last_error:
        raise MCPError(
            f"Failed to decode ask_question response after {retries + 1} attempts.\n"
            f"Last payload:\n{last_payload}"
        ) from last_error
    raise MCPError("ask_question failed without an explicit exception.")


def _extract_json_from_text(text: str) -> str:
    stripped = text.strip()
    fenced = None
    if stripped.startswith("```"):
        fence_end = stripped.find("```", 3)
        if fence_end != -1:
            fenced = stripped[3:fence_end]
    if fenced:
        return fenced.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _render_dataset_output(output: PipelineOutput, *, as_json: bool) -> str:
    if as_json:
        return json.dumps(output.to_dict(), ensure_ascii=False, indent=2)
    return output.to_text()


def _write_or_print(text: str, destination: Optional[str]) -> None:
    if destination:
        Path(destination).write_text(text, encoding="utf-8")
        logger.info("Wrote output to %s", destination)
    else:
        print(text)


def _normalize_narrative_modes(modes: Optional[List[str]]) -> List[str]:
    if not modes:
        return ["code"]
    normalized: List[str] = []
    for mode in modes:
        key = mode.lower()
        if key in {"code", "rewrite"}:
            normalized.append("code")
        elif key in {"critic", "critique", "cross"}:
            normalized.append("critic")
        else:
            logger.warning("Ignoring unrecognised narrative mode '%s'.", mode)
    if not normalized:
        normalized.append("code")
    # Preserve order but de-duplicate
    seen = set()
    result = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _collect_narrative_records(output: PipelineOutput) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for subsection in output.subsections:
        record = {
            "repo": output.repo,
            "page": subsection.page_title,
            "section": subsection.section_heading,
            "original_context": subsection.original_context,
            "narrative": subsection.narrative,
            "critic": subsection.critic,
            "verdict": subsection.verdict,
            "misalignment": subsection.misalignment.value if subsection.misalignment else None,
            "learnability": subsection.learnability,
            "critic_history": subsection.critic_history,
            "code_blocks": [
                {
                    "reference": block.reference,
                    "code": block.code,
                }
                for block in subsection.code_blocks
            ],
        }
        records.append(record)
    return records


def _render_narratives_text(records: List[Dict[str, Any]], modes: List[str]) -> str:
    parts: List[str] = []
    for record in records:
        lines = [
            f"# {record['page']} :: {record['section']}",
        ]
        lines.append("")
        lines.append("Narrative:")
        lines.append(record["narrative"])
        if "critic" in modes and record.get("critic"):
            lines.append("")
            lines.append("Critic:")
            lines.append(record["critic"])
        lines.append("")
        lines.append("Metadata:")
        verdict = record.get("verdict", "UNKNOWN")
        misalignment = record.get("misalignment") or "none"
        learnability = record.get("learnability")
        lines.append(
            f"verdict={verdict} misalignment={misalignment} learnability={learnability}"
        )
        parts.append("\n".join(lines).rstrip())
    return "\n\n".join(parts).strip()


def _write_narrative_output(
    output: PipelineOutput,
    *,
    modes: List[str],
    fmt: str,
    destination: Path,
) -> None:
    records = _collect_narrative_records(output)
    if not records:
        logger.info("No narrative records to write.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = json.dumps(records, ensure_ascii=False, indent=2)
    else:
        payload = _render_narratives_text(records, modes)
    destination.write_text(payload, encoding="utf-8")
    logger.info("Wrote narrative output to %s", destination)


def _build_design_llm_config(args: argparse.Namespace) -> NarrativeLLMConfig:
    max_tokens = args.design_vllm_max_tokens or None
    top_p = args.design_vllm_top_p or None
    return NarrativeLLMConfig(
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


def _build_judge_llm_config(args: argparse.Namespace, *, system_prompt: Optional[str]) -> JudgeLLMConfig:
    max_tokens = args.judge_vllm_max_tokens or None
    top_p = args.judge_vllm_top_p or None
    return JudgeLLMConfig(
        server_url=args.judge_vllm_server_url,
        host=args.judge_vllm_host,
        port=args.judge_vllm_port,
        path=args.judge_vllm_path,
        model=args.judge_vllm_model,
        temperature=args.judge_vllm_temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=args.judge_vllm_api_key,
        destination_service=args.judge_vllm_destination_service,
        timeout=args.judge_vllm_timeout,
        retries=args.judge_vllm_retries,
        retry_backoff=args.judge_vllm_retry_backoff,
        system_prompt=system_prompt,
    )


def _load_prompt(prompt_arg: Optional[str], *, description: str) -> Optional[str]:
    if not prompt_arg:
        return None
    if prompt_arg.startswith("@"):
        prompt_path = Path(prompt_arg[1:]).expanduser()
        if not prompt_path.exists():
            raise MCPError(f"{description} file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    return prompt_arg


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interact with the DeepWiki MCP server.",
    )
    parser.add_argument("--list-tools", action="store_true", help="List available MCP tools.")
    parser.add_argument(
        "--ask-question",
        nargs=2,
        metavar=("REPO", "QUESTION"),
        help="Call ask_question for the given repository.",
    )
    parser.add_argument("--read-structure", metavar="REPO", help="Display repository documentation outline.")
    parser.add_argument("--read-contents", metavar="REPO", help="Display repository documentation body.")
    parser.add_argument("--page", type=str, default=None, help="Target page when reading contents.")
    parser.add_argument("--section", type=str, default=None, help="Target section when reading contents.")
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Filter read-contents output to paragraphs containing this keyword.",
    )
    parser.add_argument(
        "--generate-dataset",
        metavar="REPO",
        help="Generate a DeepWiki semantic scaffold dataset.",
    )
    parser.add_argument(
        "--repo-commit",
        type=str,
        default=None,
        help="Optional commit hash/tag to pin DeepWiki documentation.",
    )
    parser.add_argument("--output", type=str, help="Write dataset output to this file.")
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Serialization format for dataset output (default: text).",
    )
    parser.add_argument(
        "--design-use-vllm",
        action="store_true",
        help="Enable external vLLM server for design intent rewriting.",
    )
    parser.add_argument("--design-vllm-server-url", type=str, default=None)
    parser.add_argument("--design-vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--design-vllm-port", type=int, default=8000)
    parser.add_argument("--design-vllm-path", type=str, default="/v1/chat/completions")
    parser.add_argument("--design-vllm-model", type=str, default=None)
    parser.add_argument("--design-vllm-temperature", type=float, default=0.2)
    parser.add_argument("--design-vllm-top-p", type=float, default=None)
    parser.add_argument("--design-vllm-max-tokens", type=int, default=131072)
    parser.add_argument("--design-vllm-timeout", type=float, default=120.0)
    parser.add_argument("--design-vllm-retries", type=int, default=2)
    parser.add_argument("--design-vllm-retry-backoff", type=float, default=2.0)
    parser.add_argument("--design-vllm-api-key", type=str, default=None)
    parser.add_argument("--design-vllm-destination-service", type=str, default="openai")
    parser.add_argument("--judge-use-llm", action="store_true", help="Enable vLLM judge model.")
    parser.add_argument("--judge-vllm-server-url", type=str, default=None)
    parser.add_argument("--judge-vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--judge-vllm-port", type=int, default=8000)
    parser.add_argument("--judge-vllm-path", type=str, default="/v1/chat/completions")
    parser.add_argument("--judge-vllm-model", type=str, default=None)
    parser.add_argument("--judge-vllm-temperature", type=float, default=0.0)
    parser.add_argument("--judge-vllm-top-p", type=float, default=None)
    parser.add_argument("--judge-vllm-max-tokens", type=int, default=131072)
    parser.add_argument("--judge-vllm-timeout", type=float, default=120.0)
    parser.add_argument("--judge-vllm-retries", type=int, default=2)
    parser.add_argument("--judge-vllm-retry-backoff", type=float, default=2.0)
    parser.add_argument("--judge-vllm-api-key", type=str, default=None)
    parser.add_argument("--judge-vllm-destination-service", type=str, default="openai")
    parser.add_argument("--judge-system-prompt", type=str, default=None)
    parser.add_argument(
        "--judge-max-rounds",
        type=int,
        default=1,
        help="Maximum judge/refinement passes (currently single-pass).",
    )
    parser.add_argument(
        "--narrative-modes",
        nargs="+",
        default=None,
        help="Narrative export modes (e.g., code, critic, cross).",
    )
    parser.add_argument(
        "--narrative-format",
        choices=["text", "json"],
        default="json",
        help="Narrative serialization format (default: json).",
    )
    parser.add_argument(
        "--narrative-output",
        type=str,
        default=None,
        help="Optional path to write narrative records.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Debug option: limit processing to the first N pages.",
    )
    parser.add_argument(
        "--max-sections-per-page",
        type=int,
        default=None,
        help="Debug option: process only the first M sections per page.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Debug option: cap concurrent page processing threads.",
    )
    parser.add_argument(
        "--skip-page",
        action="append",
        default=None,
        help=(
            "Skip wiki pages whose headings match this value (case-insensitive). "
            "Use multiple times to skip several pages."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.output and not args.generate_dataset:
        raise MCPError("--output can only be used with --generate-dataset.")
    if args.output_format != "text" and not args.generate_dataset:
        raise MCPError("--output-format can only be used with --generate-dataset.")
    if args.design_use_vllm and not args.generate_dataset:
        raise MCPError("--design-use-vllm requires --generate-dataset.")
    if args.judge_use_llm and not args.generate_dataset:
        raise MCPError("Judge options require --generate-dataset.")
    if args.design_use_vllm and not args.design_vllm_model:
        raise MCPError("--design-vllm-model is required when --design-use-vllm is set.")
    if args.judge_use_llm and not args.judge_vllm_model:
        raise MCPError("--judge-vllm-model is required when --judge-use-llm is set.")
    if args.narrative_output and not args.generate_dataset:
        raise MCPError("--narrative-output requires --generate-dataset.")
    if args.narrative_modes and not args.narrative_output:
        raise MCPError("--narrative-modes requires --narrative-output.")
    if args.judge_max_rounds < 1:
        raise MCPError("--judge-max-rounds must be >= 1.")
    if args.max_pages is not None and args.max_pages < 1:
        raise MCPError("--max-pages must be >= 1 when provided.")
    if args.max_sections_per_page is not None and args.max_sections_per_page < 1:
        raise MCPError("--max-sections-per-page must be >= 1 when provided.")
    if args.max_workers is not None and args.max_workers < 1:
        raise MCPError("--max-workers must be >= 1 when provided.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.log_level)
    try:
        validate_args(args)
    except MCPError as exc:
        logger.error("%s", exc)
        sys.exit(2)

    session: Optional[Session] = None
    try:
        session = initialize_session()
        if args.list_tools:
            _print_tools(session)
        if args.ask_question:
            repo, question = args.ask_question
            response = _ask_question_json(session, repo, question)
            print(json.dumps(response, ensure_ascii=False, indent=2))
        if args.read_structure:
            _print_structure(session, args.read_structure)
        if args.read_contents:
            _print_contents(
                session,
                args.read_contents,
                page=args.page,
                section=args.section,
                contains=args.contains,
            )

        if not args.generate_dataset:
            return

        target_repo = args.generate_dataset

        design_config = _build_design_llm_config(args) if args.design_use_vllm else None
        judge_system_prompt = _load_prompt(
            args.judge_system_prompt,
            description="Judge system prompt",
        )
        judge_config = (
            _build_judge_llm_config(args, system_prompt=judge_system_prompt)
            if args.judge_use_llm
            else None
        )

        dataset_path: Optional[Path] = Path(args.output) if args.output else None
        narrative_path: Optional[Path] = (
            Path(args.narrative_output) if args.narrative_output else None
        )
        narrative_modes: List[str] = (
            _normalize_narrative_modes(args.narrative_modes)
            if narrative_path
            else []
        )

        def persist_progress(partial_output: PipelineOutput) -> None:
            if dataset_path:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                rendered_partial = _render_dataset_output(
                    partial_output, as_json=args.output_format == "json"
                )
                dataset_path.write_text(rendered_partial, encoding="utf-8")
                if args.output_format != "json":
                    json_path = dataset_path.with_suffix(".json")
                    json_payload = json.dumps(
                        partial_output.to_dict(), ensure_ascii=False, indent=2
                    )
                    json_path.write_text(json_payload, encoding="utf-8")
            if narrative_path:
                narrative_path.parent.mkdir(parents=True, exist_ok=True)
                _write_narrative_output(
                    partial_output,
                    modes=narrative_modes,
                    fmt=args.narrative_format,
                    destination=narrative_path,
                )

        pipeline = DeepWikiPipeline(
            session=session,
            repo=target_repo,
            logic_llm_config=design_config,
            critic_llm_config=judge_config,
            repo_commit=args.repo_commit,
            judge_rounds=args.judge_max_rounds,
            max_pages=args.max_pages,
            max_sections_per_page=args.max_sections_per_page,
            max_workers=args.max_workers,
            skip_pages=args.skip_page,
        )
        progress_needed = dataset_path is not None or narrative_path is not None
        
        output = pipeline.run(
            progress_callback=persist_progress if progress_needed else None
        )

        rendered = _render_dataset_output(output, as_json=args.output_format == "json")
        _write_or_print(rendered, args.output)
        if narrative_path:
            modes = narrative_modes or _normalize_narrative_modes(args.narrative_modes)
            _write_narrative_output(
                output,
                modes=modes,
                fmt=args.narrative_format,
                destination=narrative_path,
            )

    except MCPError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    finally:
        if session is not None:
            delete_session(session)


if __name__ == "__main__":
    main()
