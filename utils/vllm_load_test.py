#!/usr/bin/env python3
"""
Lightweight load test tool for OpenAI-compatible /v1/chat/completions endpoints (e.g. vLLM).

Goal: measure latency/throughput under controlled concurrency to help set:
- --repo-workers (repo-level parallelism)
- --max-workers (page-level parallelism)

This script avoids external dependencies beyond `requests`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class Result:
    ok: bool
    latency_s: float
    status_code: Optional[int]
    error: Optional[str]
    endpoint: str


def _parse_urls(raw: Sequence[str]) -> List[str]:
    urls: List[str] = []
    for item in raw:
        if not item:
            continue
        for part in item.split(","):
            u = part.strip()
            if u and u not in urls:
                urls.append(u)
    return urls


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    k = (len(values_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _build_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _request_one(
    *,
    urls: List[str],
    payload: Dict[str, Any],
    timeout_s: float,
    api_key: Optional[str],
    seed: int,
) -> Result:
    random.seed(seed)
    endpoint = random.choice(urls)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.time()
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout_s,
        )
        latency = time.time() - start
        ok = 200 <= resp.status_code < 300
        if ok:
            return Result(ok=True, latency_s=latency, status_code=resp.status_code, error=None, endpoint=endpoint)
        snippet = resp.text[:500] if resp.text else ""
        return Result(
            ok=False,
            latency_s=latency,
            status_code=resp.status_code,
            error=f"HTTP {resp.status_code}: {snippet}",
            endpoint=endpoint,
        )
    except Exception as exc:
        latency = time.time() - start
        return Result(ok=False, latency_s=latency, status_code=None, error=str(exc), endpoint=endpoint)


def _summarize(results: List[Result]) -> str:
    latencies = [r.latency_s for r in results if r.ok]
    ok = sum(1 for r in results if r.ok)
    total = len(results)
    errors = total - ok
    if latencies:
        avg = statistics.mean(latencies)
        p50 = _percentile(latencies, 0.50)
        p90 = _percentile(latencies, 0.90)
        p95 = _percentile(latencies, 0.95)
        p99 = _percentile(latencies, 0.99)
        mx = max(latencies)
    else:
        avg = p50 = p90 = p95 = p99 = mx = float("nan")
    return (
        f"total={total} ok={ok} errors={errors} "
        f"latency_s(avg={avg:.3f} p50={p50:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f} max={mx:.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Load test for OpenAI-compatible chat completion endpoints.")
    ap.add_argument("--urls", action="append", required=True, help="Endpoint URL(s); may repeat or comma-separate.")
    ap.add_argument("--model", required=True, help="Model name.")
    ap.add_argument("--prompt", default="Say hello in one sentence.", help="Prompt text.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=4, help="In-flight requests.")
    ap.add_argument("--requests", type=int, default=40, help="Total requests.")
    ap.add_argument("--warmup", type=int, default=0, help="Warmup requests (not counted in stats).")
    args = ap.parse_args()

    urls = _parse_urls(args.urls)
    if not urls:
        raise SystemExit("No URLs provided.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")

    payload = _build_payload(args.model, args.prompt, args.max_tokens, args.temperature)

    if args.warmup:
        for i in range(args.warmup):
            _request_one(urls=urls, payload=payload, timeout_s=args.timeout_s, api_key=args.api_key, seed=10_000 + i)

    start = time.time()
    results: List[Result] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                _request_one,
                urls=urls,
                payload=payload,
                timeout_s=args.timeout_s,
                api_key=args.api_key,
                seed=i,
            )
            for i in range(args.requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    duration = time.time() - start

    ok = sum(1 for r in results if r.ok)
    rps = ok / duration if duration > 0 else 0.0
    print(_summarize(results))
    print(f"wall_s={duration:.2f} ok_rps={rps:.2f}")
    if ok != len(results):
        failures = [r for r in results if not r.ok][:10]
        for f in failures:
            print(f"FAIL endpoint={f.endpoint} latency_s={f.latency_s:.3f} status={f.status_code} error={f.error}")


if __name__ == "__main__":
    main()

