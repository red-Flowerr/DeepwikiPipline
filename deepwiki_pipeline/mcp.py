"""HTTP/MCP client utilities for interacting with the DeepWiki server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests


logger = logging.getLogger(__name__)

MCP_ENDPOINT = "https://mcp.deepwiki.com/mcp"
PROTOCOL_VERSION = "2025-06-18"

import itertools
_REQUEST_COUNTER = itertools.count(2)


class MCPError(RuntimeError):
    """Raised when the MCP server interaction fails."""


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
            part = line[len("data:") :]
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
