"""Thin client for chat-completions compatible vLLM servers."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests


@dataclass
class ChatMessage:
    role: str
    content: str


class VLLMError(RuntimeError):
    pass


def normalize_host(host: str) -> str:
    host = host.strip()
    if ":" in host and not host.startswith("[") and not host.endswith("]"):
        return f"[{host}]"
    return host


def build_url(
    server_url: Optional[str],
    host: str,
    port: int,
    path: str,
) -> str:
    if server_url:
        return server_url
    suffix = path if path.startswith("/") else f"/{path}"
    return f"http://{normalize_host(host)}:{port}{suffix}"


def _summarize_text(text: str, limit: int = 500) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "<empty response body>"
    if len(stripped) > limit:
        return f"{stripped[:limit]}... [truncated]"
    return stripped


def _describe_bad_request(response: requests.Response) -> Tuple[str, str]:
    hint = "cause unknown"
    detail = "<empty response body>"
    try:
        payload = response.json()
    except ValueError:
        text = response.text or ""
        lowered = text.lower()
        if "context" in lowered and "token" in lowered:
            hint = "probable context length limit"
        elif "max token" in lowered or "too many tokens" in lowered:
            hint = "probable max_tokens limit"
        detail = _summarize_text(text)
        return hint, detail

    if not isinstance(payload, dict):
        detail = _summarize_text(str(payload))
        return hint, detail

    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "")
        code = error_obj.get("code")
        error_type = error_obj.get("type")
        detail = _summarize_text(message or json.dumps(error_obj))
        normalized_code = str(code).strip() if isinstance(code, str) else ""
        lower_message = message.lower()
        if normalized_code:
            if normalized_code in {
                "context_length_exceeded",
                "max_context_length_exceeded",
                "context_length",
            }:
                hint = f"context length exceeded ({normalized_code})"
            elif normalized_code in {"too_many_tokens", "max_tokens_exceeded"}:
                hint = f"token limit exceeded ({normalized_code})"
            else:
                hint = normalized_code
        elif "context length" in lower_message or "token limit" in lower_message:
            hint = "probable context length limit"
        elif "max tokens" in lower_message or "too many tokens" in lower_message:
            hint = "probable max_tokens limit"
        elif isinstance(error_type, str) and error_type.strip():
            hint = error_type.strip()
    else:
        detail = _summarize_text(json.dumps(payload))
    return hint, detail


def post_with_retry(
    session: requests.Session,
    url: str,
    payload: Dict,
    headers: Optional[Dict[str, str]],
    timeout: float,
    retries: int,
    backoff: float,
) -> Dict:
    attempt = 0
    while True:
        try:
            response = session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            
            if response.status_code == 400:
                # import pdb; pdb.set_trace()
                hint, detail = _describe_bad_request(response)
                logger.error(
                    "vLLM chat request returned HTTP 400 (%s). Detail: %s",
                    hint,
                    detail,
                )   
                raise VLLMError(f"HTTP 400 Bad Request from vLLM ({hint}). Detail: {detail}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            attempt += 1
            if attempt > retries:
                raise VLLMError(f"Request failed after {retries} retries: {exc}")
            time.sleep(backoff * attempt)


def _coerce_content(value: Union[str, Sequence, Dict[str, object], None]) -> Optional[str]:
    """
    Normalize content payloads that may be strings, structured lists, or dicts.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "reasoning_content", "output", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                normalized = _coerce_content(candidate)
                if normalized:
                    return normalized
        return None
    if isinstance(value, Sequence):
        parts: List[str] = []
        for item in value:
            normalized = _coerce_content(item)
            if normalized:
                parts.append(normalized)
        if parts:
            return "".join(parts)
    return None


def extract_content(response: Dict) -> Optional[str]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return None
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        content = _coerce_content(message) if isinstance(message, dict) else None
        if not content and isinstance(message, dict):
            content = _coerce_content(message.get("content"))
        if content:
            return content
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = _coerce_content(delta.get("content")) or _coerce_content(delta.get("text"))
            if content:
                return content
        text_field = choice.get("text") or choice.get("output")
        content = _coerce_content(text_field)
        if content:
            return content
        logger.debug(
            "Choice %d missing recognised content fields: %s",
            idx,
            json.dumps(choice, ensure_ascii=False),
        )
    return None


def call_vllm_chat(
    host: str,
    port: int,
    path: str,
    model: str,
    messages: Iterable[ChatMessage],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    destination_service: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
    retry_backoff: float = 2.0,
) -> str:
    session = requests.Session()
    url = build_url(server_url, host, port, path)
    payload: Dict[str, object] = {
        "model": model,
        "messages": [msg.__dict__ for msg in messages],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    header_destination = destination_service or "openai"
    headers["Destination-Service"] = header_destination

    response = post_with_retry(
        session=session,
        url=url,
        payload=payload,
        headers=headers or None,
        timeout=timeout,
        retries=retries,
        backoff=retry_backoff,
    )
    content = extract_content(response)
    if content is None:
        raise VLLMError("vLLM response did not contain message content.")
    return content
logger = logging.getLogger(__name__)
