#!/usr/bin/env python3
"""Count tokens for narrative entries using a local Hugging Face tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer


def load_texts(json_path: Path, key: str) -> Sequence[str]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get(key)
    if value is None:
        raise SystemExit(f"Key '{key}' not found in {json_path}.")
    if not isinstance(value, list):
        raise SystemExit(f"Key '{key}' must point to a list.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count tokens for narratives using a local Hugging Face tokenizer."
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        help="Path to the narrative JSON file (e.g. result_data/megatron_narratives.json).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="code",
        help="Narrative key to analyze (default: code).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path or identifier of the tokenizer (e.g. /path/to/Qwen2.5-1.5B).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Whether to include special tokens when encoding (default: False).",
    )
    args = parser.parse_args()

    print(f"[info] Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    texts = load_texts(args.json_path, args.key)
    counts: list[int] = []
    for idx, text in enumerate(texts):
        # import pdb; pdb.set_trace()
        ids = tokenizer.encode(
            text,
            add_special_tokens=args.add_special_tokens,
        )
        counts.append(len(ids))
        print(f"[{idx+1}/{len(texts)}] tokens={counts[-1]}")

    total = sum(counts)
    average = total / len(counts) if counts else 0
    maximum = max(counts) if counts else 0
    top_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:5]

    print("\nSummary")
    print(f"entries: {len(counts)}")
    print(f"total_tokens: {total}")
    print(f"avg_tokens_per_entry: {average}")
    print(f"max_tokens: {maximum}")
    print(f"top_indices_by_tokens: {top_indices}")


if __name__ == "__main__":
    main()
