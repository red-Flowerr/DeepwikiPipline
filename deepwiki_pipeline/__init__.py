"""High level entrypoints for the DeepWiki semantic pipeline."""

from .mcp import (
    MCP_ENDPOINT,
    PROTOCOL_VERSION,
    MCPError,
    Session,
    call_tool,
    delete_session,
    extract_text_blocks,
    initialize_session,
    list_tools,
    parse_sse_response,
    post_jsonrpc,
)
from .models import (
    DatasetChunk,
    CodeReference,
    JudgeLLMConfig,
    NarrativeLLMConfig,
    OutlineNode,
    PageContent,
    PipelineOutput,
    SectionBlock,
    SectionContent,
    SubsectionResult,
    InstructionPair,
    normalize_heading,
)
from .parsing import (
    parse_outline_text,
    parse_wiki_markdown,
    resolve_page,
    resolve_label_path_range,
    parse_sources_links,
)
from .narrative import make_section_result, summarise_page
from .hydration import hydrate_section_text
from .pipeline import DeepWikiPipeline

__all__ = [
    "MCP_ENDPOINT",
    "PROTOCOL_VERSION",
    "MCPError",
    "Session",
    "call_tool",
    "delete_session",
    "extract_text_blocks",
    "initialize_session",
    "list_tools",
    "parse_sse_response",
    "post_jsonrpc",
    "DatasetChunk",
    "CodeReference",
    "JudgeLLMConfig",
    "NarrativeLLMConfig",
    "OutlineNode",
    "PageContent",
    "PipelineOutput",
    "SectionBlock",
    "SectionContent",
    "SubsectionResult",
    "InstructionPair",
    "DeepWikiPipeline",
    "make_section_result",
    "normalize_heading",
    "parse_outline_text",
    "parse_sources_links",
    "parse_wiki_markdown",
    "resolve_label_path_range",
    "resolve_page",
    "summarise_page",
    "hydrate_section_text",
]
