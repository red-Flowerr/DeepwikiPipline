"""Thin client for chat-completions compatible vLLM servers."""

from __future__ import annotations

import json
import logging
import time
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    from litellm import Router
except ImportError:  # pragma: no cover
    Router = None  # type: ignore[assignment]


@dataclass
class ChatMessage:
    role: str
    content: str


class VLLMError(RuntimeError):
    pass


_ROUTER_CACHE: Dict[Tuple[Tuple[str, ...], str, str, str], object] = {}
_ROUTER_LOCK = threading.Lock()
_ENDPOINT_COOLDOWN_UNTIL: Dict[str, float] = {}
_ENDPOINT_COOLDOWN_LOCK = threading.Lock()
_POOL_RR_INDEX: Dict[Tuple[str, ...], int] = {}
_POOL_RR_LOCK = threading.Lock()


def _cooldown_endpoint(endpoint: str, *, seconds: float) -> None:
    if seconds <= 0:
        return
    until = time.time() + seconds
    with _ENDPOINT_COOLDOWN_LOCK:
        previous = _ENDPOINT_COOLDOWN_UNTIL.get(endpoint)
        if previous is None or until > previous:
            _ENDPOINT_COOLDOWN_UNTIL[endpoint] = until


def _is_endpoint_cooled_down(endpoint: str) -> bool:
    now = time.time()
    with _ENDPOINT_COOLDOWN_LOCK:
        until = _ENDPOINT_COOLDOWN_UNTIL.get(endpoint)
    return bool(until and until > now)


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    lowered = str(exc).lower()
    return any(
        token in lowered
        for token in (
            "timeout",
            "timed out",
            "connection error",
            "connection aborted",
            "connection refused",
            "connection reset",
            "connection is closed",
            "connection is closed by peer",
            "remote end closed connection",
            "temporary failure",
            "temporarily unavailable",
            "network is unreachable",
            "no route to host",
            "name or service not known",
        )
    )


def _is_prompt_too_large_error(exc: Exception) -> bool:
    if not isinstance(exc, VLLMError):
        return False
    message = str(exc).lower()
    if "http 400" not in message:
        return False
    return any(
        token in message
        for token in (
            "context length exceeded",
            "max_context_length_exceeded",
            "context_length_exceeded",
            "token limit exceeded",
            "too_many_tokens",
            "max_tokens_exceeded",
            "probable context length limit",
            "probable max_tokens limit",
        )
    )


def _rotate_endpoints_round_robin(endpoints: Sequence[str]) -> List[str]:
    ordered = list(endpoints)
    if len(ordered) <= 1:
        return ordered
    key = tuple(ordered)
    with _POOL_RR_LOCK:
        start = _POOL_RR_INDEX.get(key, 0) % len(ordered)
        _POOL_RR_INDEX[key] = (start + 1) % len(ordered)
    if start == 0:
        return ordered
    return ordered[start:] + ordered[:start]


def _diagnose_endpoints_on_failure(
    endpoints: Sequence[str],
    *,
    timeout: float,
    api_key: Optional[str],
    destination_service: Optional[str],
) -> List[Tuple[str, str]]:
    """
    No-op: endpoint health checking is disabled.
    """
    _ = (endpoints, timeout, api_key, destination_service)
    return []


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


def _coerce_endpoint(
    value: Optional[str],
    default_host: str,
    default_port: int,
    default_path: str,
) -> Optional[str]:
    if value is None:
        return None
    entry = str(value).strip()
    if not entry:
        return None
    inferred = entry
    if "://" not in entry:
        inferred = f"http://{entry}"
    parsed = urlsplit(inferred)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    path_part = parsed.path or ""
    if not netloc:
        # Treat as path override on default host/port.
        path_part = path_part or default_path
        return build_url(None, default_host, default_port, path_part)
    if path_part in {"", "/"}:
        path_part = default_path
    if not path_part.startswith("/"):
        path_part = f"/{path_part}"
    if parsed.query or parsed.fragment:
        logger.warning(
            "Dropping query/fragment from LiteLLM server URL '%s'.",
            entry,
        )
    endpoint = urlunsplit((scheme, netloc, path_part.rstrip("/"), "", ""))
    return endpoint.rstrip("/")


def _normalize_server_pool(
    server_urls: Sequence[str],
    host: str,
    port: int,
    path: str,
    fallback_url: Optional[str],
) -> List[str]:
    candidates: List[str] = []
    for value in server_urls:
        endpoint = _coerce_endpoint(value, host, port, path)
        if endpoint:
            candidates.append(endpoint)
    if fallback_url:
        fallback_endpoint = _coerce_endpoint(fallback_url, host, port, path)
        if fallback_endpoint:
            candidates.append(fallback_endpoint)
    if not candidates:
        candidates.append(build_url(None, host, port, path))
    normalized: List[str] = []
    for endpoint in candidates:
        if endpoint not in normalized:
            normalized.append(endpoint)
    return normalized


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


def _get_litellm_router(
    endpoints: Sequence[str],
    model: str,
    api_key: Optional[str],
    destination_service: Optional[str],
    timeout: float,
):
    if Router is None:
        raise VLLMError(
            "LiteLLM is required for server pooling. Install it with `pip install litellm`."
        )
    cache_key = (tuple(endpoints), model, api_key or "", destination_service or "")
    with _ROUTER_LOCK:
        cached = cast(Optional["Router"], _ROUTER_CACHE.get(cache_key))
        if cached is not None:
            return cached
        model_list: List[Dict[str, object]] = []
        for endpoint in endpoints:
            parsed = urlsplit(endpoint)
            scheme = parsed.scheme or "http"
            netloc = parsed.netloc or ""
            path = parsed.path or ""
            if not netloc:
                # Handle endpoints provided as bare host/path strings.
                netloc = path
                path = ""
            base_path = path
            if base_path.endswith("/chat/completions"):
                base_path = base_path[: -len("/chat/completions")]
            if not base_path:
                base_path = "/v1"
            base_path = base_path.rstrip("/")
            api_base = urlunsplit((scheme, netloc, base_path, "", ""))
            header_value = destination_service or "openai"
            extra_headers = {"Destination-Service": header_value}
            litellm_params: Dict[str, object] = {
                "model": f"openai/{model}",
                "api_base": api_base,
                "extra_headers": extra_headers,
            }
            litellm_params["api_key"] = api_key or "dummy-key"
            model_list.append(
                {
                    "model_name": model,
                    "litellm_params": litellm_params,
                }
            )
        router = Router(model_list=model_list, timeout=timeout)
        _ROUTER_CACHE[cache_key] = router
        logger.debug(
            "Initialized LiteLLM router for model %s with %d endpoints.",
            model,
            len(endpoints),
        )
        return router


def _call_litellm_router(
    endpoints: Sequence[str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    api_key: Optional[str],
    destination_service: Optional[str],
    timeout: float,
    retries: int = 2,
    retry_backoff: float = 2.0,
) -> str:
    candidates = [ep for ep in endpoints if not _is_endpoint_cooled_down(ep)]
    if not candidates:
        candidates = list(endpoints)
    cache_key = (tuple(candidates), model, api_key or "", destination_service or "")

    # LiteLLM's Router already implements its own retries; keep this wrapper to a
    # single attempt to avoid multiplicative retry storms.
    retries = 1
    attempt = 0
    last_error: Optional[Exception] = None
    response: Optional[object] = None
    while attempt < max(1, retries):
        attempt += 1
        router = _get_litellm_router(
            endpoints=candidates,
            model=model,
            api_key=api_key,
            destination_service=destination_service,
            timeout=timeout,
        )
        kwargs: Dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        try:
            logger.debug(
                "Dispatching LiteLLM chat completion for model %s across %d endpoints (attempt %d).",
                model,
                len(candidates),
                attempt,
            )
            response = router.completion(**kwargs)
            last_error = None
            break
        except Exception as exc:  # pragma: no cover - passthrough error handling
            last_error = exc
            lowered = str(exc).lower()
            transient = (
                "connection error" in lowered
                or "connection is closed by peer" in lowered
                or "connection is closed" in lowered
                or "timeout" in lowered
            )
            logger.warning(
                "LiteLLM router call failed (attempt %d/%d, endpoints=%d): %s",
                attempt,
                max(1, retries),
                len(candidates),
                exc,
            )
            if transient and candidates:
                failures = _diagnose_endpoints_on_failure(
                    candidates,
                    timeout=min(2.0, max(0.5, timeout)),
                    api_key=api_key,
                    destination_service=destination_service,
                )
                for endpoint, detail in failures:
                    logger.warning("LiteLLM endpoint failing (post-error check): %s (%s)", endpoint, detail)
                    _cooldown_endpoint(endpoint, seconds=120.0)
                if failures and len(failures) < len(candidates):
                    candidates = [ep for ep in candidates if ep not in {f[0] for f in failures}]
            if transient:
                # Do not recreate the Router on transient errors; it can cause LiteLLM to
                # accumulate logging callbacks and hit MAX_CALLBACKS.
                break
            break

    if response is None:
        last_error = last_error or Exception("unknown error")
        raise VLLMError(f"LiteLLM router call failed: {last_error}") from last_error

    payload: Dict[str, object]
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    elif isinstance(response, dict):
        payload = response
    else:
        raise VLLMError(
            f"Unsupported LiteLLM response type: {type(response)!r}"
        )
    content = extract_content(payload)
    if content is None:
        raise VLLMError("LiteLLM router response did not contain message content.")
    return content


def _call_server_pool_direct(
    endpoints: Sequence[str],
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    api_key: Optional[str],
    destination_service: Optional[str],
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> str:
    """
    Directly call vLLM/OpenAI-compatible endpoints without LiteLLM Router.

    This avoids LiteLLM's internal async wrappers/callback registration which can
    become unstable under very high concurrency.
    """
    if not endpoints:
        raise VLLMError("No endpoints provided for server pool.")
    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Destination-Service"] = destination_service or "openai"

    session = requests.Session()
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        candidates = [ep for ep in endpoints if not _is_endpoint_cooled_down(ep)]
        if not candidates:
            candidates = list(endpoints)
        for endpoint in _rotate_endpoints_round_robin(candidates):
            try:
                response = post_with_retry(
                    session=session,
                    url=endpoint,
                    payload=payload,
                    headers=headers or None,
                    timeout=timeout,
                    retries=1,
                    backoff=retry_backoff,
                )
                content = extract_content(response)
                if content is None:
                    raise VLLMError("vLLM response did not contain message content.")
                return content
            except Exception as exc:
                last_exc = exc
                if _is_prompt_too_large_error(exc):
                    raise
                if _is_transient_error(exc):
                    _cooldown_endpoint(endpoint, seconds=120.0)
                logger.warning(
                    "Direct server pool call failed (attempt %d/%d) endpoint=%s: %s",
                    attempt,
                    max(1, retries),
                    endpoint,
                    exc,
                )
        time.sleep(retry_backoff * attempt)
    raise VLLMError(f"Request failed after {retries} retries across server pool: {last_exc}") from last_exc


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
    server_urls: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    destination_service: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
    retry_backoff: float = 2.0,
) -> str:
    message_payload = [msg.__dict__ for msg in messages]
    if server_urls:
        endpoints = _normalize_server_pool(
            server_urls,
            host=host,
            port=port,
            path=path,
            fallback_url=server_url,
        )
        return _call_server_pool_direct(
            endpoints=endpoints,
            model=model,
            messages=message_payload,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            api_key=api_key,
            destination_service=destination_service,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    session = requests.Session()
    url = build_url(server_url, host, port, path)
    payload: Dict[str, object] = {
        "model": model,
        "messages": message_payload,
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
