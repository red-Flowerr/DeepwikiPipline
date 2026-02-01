import json, re
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SOURCES_HEADER_RE = re.compile(
    r"(?i)^\s*(?:\*\*|__)?\s*sources?\s*(?:\*\*|__)?\s*[:：\-–—]\s*(?P<rest>.*)$"
)
BACKTICK_RE = re.compile(r"`([^`]+)`")
# Only treat references that appear under a Sources section as indices.
# This avoids accidentally capturing arbitrary "Source:" mentions or hydrated label blocks.
PLAIN_INDEX_RE = re.compile(r"^[A-Za-z0-9_.\-/ ]+(?::\d+(?:-\d+)?)?$")

_RANGE_SUFFIX_RE = re.compile(r"^(?P<path>.+?)(?P<range>:\d+(?:-\d+)?)?$")


def normalize_index_label(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    # Drop common bullet/list prefixes.
    while s.startswith(("- ", "* ", "• ")):
        s = s[2:].strip()
    # Drop ordered list prefixes like "1. "
    s = re.sub(r"^\d+\.\s+", "", s).strip()
    if s.lower().startswith("source:"):
        s = s.split(":", 1)[1].strip()
    # Drop surrounding backticks.
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()

    m = _RANGE_SUFFIX_RE.match(s)
    if not m:
        return s
    path = (m.group("path") or "").strip()
    rng = (m.group("range") or "").strip()

    return f"{path}{rng}".strip()


def _extract_sources_from_lines(lines: list[str]) -> list[str]:
    """
    Extract Sources references from either:
    - inline: "Sources: `a.py:1-2`, `b.py`"
    - block form:
        Sources:
        - a.py:1-2
        - `b c.py`
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SOURCES_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        rest = (m.group("rest") or "").strip()
        if rest:
            # Only treat backticked entries as valid indices.
            items = [x.strip() for x in BACKTICK_RE.findall(rest) if x.strip()]
            if not items:
                # Also accept a single plain index token when it is the only thing on the line.
                # Example: "Sources: .idea/misc.xml:1-7\n"
                if (
                    PLAIN_INDEX_RE.match(rest)
                    and "`" not in rest
                    and not any(sep in rest for sep in (",", ";", "，"))
                ):
                    items = [rest]
            out.extend(items)
            i += 1
            continue
        # Block form: consume subsequent "- ..." lines.
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if not (
                nxt.startswith(("- ", "* ", "• "))
                or nxt.startswith("`")
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            # Only accept backticked entries in block form as well.
            # Examples:
            # - `a.py:1-2`
            # 1. `b c.py`
            candidates = [x.strip() for x in BACKTICK_RE.findall(nxt) if x.strip()]
            out.extend(candidates)
            i += 1
        continue
    return out


def extract_indices(text: str) -> list[str]:
    if not text:
        return []
    found, seen = [], set()

    lines = text.splitlines()
    for raw in _extract_sources_from_lines(lines):
        x2 = normalize_index_label(raw)
        if x2 and x2 not in seen:
            seen.add(x2)
            found.append(x2)

    return found

def main() -> None:
    narr_dir = Path("result_data/batch_narratives")
    out_path = Path("result_data/repo_indices.json")

    repo_to_indices = defaultdict(set)

    files = sorted(narr_dir.glob("*_narratives.json"))
    iter_files = tqdm(files, desc="Scanning narrative files", unit="file") if tqdm else files
    for fp in iter_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        iter_recs = tqdm(data, desc=f"Records ({fp.name})", unit="rec", leave=False) if tqdm else data
        for rec in iter_recs:
            if not isinstance(rec, dict):
                continue
            repo = str(rec.get("repo") or "").strip()
            if not repo:
                continue
            oc = str(rec.get("original_context") or "")
            for idx in extract_indices(oc):
                repo_to_indices[repo].add(idx)

    payload = {repo: sorted(list(idxs)) for repo, idxs in repo_to_indices.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("repos_with_indices:", len(payload))
    print("wrote:", out_path)


if __name__ == "__main__":
    main()
