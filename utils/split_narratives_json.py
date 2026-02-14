#!/usr/bin/env python3
"""
Split huge *_narratives.json files (a single JSON array) into smaller files.

Why: export_narratives_fast.py loads each JSON file fully into memory and may get OOM-killed (-9)
on very large (e.g. multi-GB) arrays. Splitting upstream keeps "long narratives" but reduces peak
memory per worker by bounding input size.

Input format: a JSON array, typically a list of dict rows like:
  [{"repo": "...", "narrative": "..."}, ...]

Output format: multiple JSON arrays, each ending with "_narratives.json", e.g.:
  foo.part0000_narratives.json, foo.part0001_narratives.json, ...
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import sys
from dataclasses import dataclass


@dataclass
class SplitStats:
    in_rows: int = 0
    out_files: int = 0


def _iter_json_array_items(path: str, *, chunk_bytes: int = 8 * 1024 * 1024):
    """
    Streaming parse a top-level JSON array without external deps (ijson).
    Uses json.JSONDecoder.raw_decode on an incrementally decoded UTF-8 stream.
    """

    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()

    buf = ""
    i = 0
    started = False
    done = False
    file_size = None

    def _skip_ws(s: str, pos: int) -> int:
        n = len(s)
        while pos < n and s[pos] in " \t\r\n":
            pos += 1
        return pos

    with open(path, "rb") as f:
        try:
            file_size = os.fstat(f.fileno()).st_size
        except Exception:
            file_size = None
        while not done:
            if i >= len(buf) - 1:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    # Flush decoder state and stop reading.
                    buf += utf8.decode(b"", final=True)
                else:
                    buf += utf8.decode(chunk, final=False)

            i = _skip_ws(buf, i)
            if not started:
                if i >= len(buf):
                    if file_size is not None and f.tell() >= file_size:
                        raise ValueError("unexpected EOF before '['")
                    continue
                if buf[i] != "[":
                    snippet = buf[i : i + 80]
                    raise ValueError(f"expected '[' at start of JSON array, got {snippet!r}")
                started = True
                i += 1
                continue

            i = _skip_ws(buf, i)
            if i >= len(buf):
                if file_size is not None and f.tell() >= file_size:
                    raise ValueError("unexpected EOF in JSON array")
                continue

            if buf[i] == "]":
                done = True
                i += 1
                continue

            try:
                obj, end = decoder.raw_decode(buf, i)
            except json.JSONDecodeError:
                # Need more data.
                chunk = f.read(chunk_bytes)
                if not chunk:
                    snippet = buf[i : i + 200]
                    raise ValueError(f"unexpected EOF while decoding JSON item near: {snippet!r}") from None
                buf += utf8.decode(chunk, final=False)
                continue

            yield obj
            i = end

            i = _skip_ws(buf, i)
            if i >= len(buf):
                continue
            if buf[i] == ",":
                i += 1
                continue
            if buf[i] == "]":
                done = True
                i += 1
                continue

            snippet = buf[i : i + 80]
            raise ValueError(f"expected ',' or ']' after JSON item, got {snippet!r}")

            # Compact buffer periodically.
            if i > 1_000_000:
                buf = buf[i:]
                i = 0


def _open_out(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "wb")


def split_file(
    input_path: str,
    *,
    rows_per_part: int,
    max_part_mb: int,
    out_dir: str | None,
    dry_run: bool,
) -> SplitStats:
    if not input_path.endswith("_narratives.json"):
        raise ValueError("input must end with '_narratives.json' so outputs can match exporter scan pattern")

    input_dir = os.path.dirname(input_path) or "."
    base_name = os.path.basename(input_path)
    prefix = base_name[: -len("_narratives.json")]
    out_base_dir = out_dir or input_dir

    max_part_bytes = int(max_part_mb) * 1024 * 1024 if int(max_part_mb) > 0 else 0
    if rows_per_part <= 0 and max_part_bytes <= 0:
        raise ValueError("at least one of --rows-per-part or --max-part-mb must be > 0")

    stats = SplitStats()

    part_idx = 0
    out_f = None
    out_rows = 0
    out_bytes = 0
    first = True

    def _part_path(idx: int) -> str:
        return os.path.join(out_base_dir, f"{prefix}.part{idx:04d}_narratives.json")

    def _close_part():
        nonlocal out_f, out_rows, out_bytes, first
        if out_f is None:
            return
        if not dry_run:
            out_f.write(b"]\n")
            out_f.close()
        out_f = None
        out_rows = 0
        out_bytes = 0
        first = True
        stats.out_files += 1

    def _open_part():
        nonlocal out_f, first
        if out_f is not None:
            return
        p = _part_path(part_idx)
        if dry_run:
            out_f = True  # type: ignore[assignment]
        else:
            out_f = _open_out(p)
            out_f.write(b"[\n")
        first = True

    for obj in _iter_json_array_items(input_path):
        stats.in_rows += 1
        if out_f is None:
            _open_part()

        row_bytes = len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        should_rotate = False
        if rows_per_part > 0 and out_rows >= rows_per_part:
            should_rotate = True
        if max_part_bytes > 0 and out_bytes >= max_part_bytes:
            should_rotate = True

        if should_rotate:
            _close_part()
            part_idx += 1
            _open_part()

        out_rows += 1
        out_bytes += row_bytes

        if not dry_run:
            assert out_f is not None
            if not first:
                out_f.write(b",\n")
            out_f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
            first = False

    if out_f is not None:
        _close_part()

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Split huge *_narratives.json JSON arrays into smaller *_narratives.json parts.")
    ap.add_argument("--input", required=True, help="Path to a single *_narratives.json (JSON array).")
    ap.add_argument("--rows-per-part", type=int, default=50000, help="Max rows per output part (0 disables).")
    ap.add_argument("--max-part-mb", type=int, default=0, help="Approx max MiB per output part (0 disables).")
    ap.add_argument("--out-dir", default="", help="Optional output directory (default: same directory as input).")
    ap.add_argument("--dry-run", action="store_true", default=False, help="Parse and report counts without writing output files.")
    args = ap.parse_args()

    out_dir = args.out_dir or None
    try:
        stats = split_file(
            args.input,
            rows_per_part=int(args.rows_per_part),
            max_part_mb=int(args.max_part_mb),
            out_dir=out_dir,
            dry_run=bool(args.dry_run),
        )
    except Exception as e:
        print(f"[split] ERROR: {e}", file=sys.stderr)
        raise

    print("[split] input:", args.input)
    print("[split] rows:", stats.in_rows)
    print("[split] parts:", stats.out_files if not args.dry_run else "(dry-run)")


if __name__ == "__main__":
    main()
