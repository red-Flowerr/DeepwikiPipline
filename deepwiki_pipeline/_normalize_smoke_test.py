"""
Minimal smoke checks for normalization helpers.

This repository does not ship a test runner; this module is used by `py_compile`
and occasional `python -m deepwiki_pipeline._normalize_smoke_test` sanity checks.
"""

from __future__ import annotations

from .models import normalize_heading


def _main() -> None:
    assert normalize_heading("4.2 Assets and Resources") == "assetsandresources"
    assert normalize_heading("4_2 Assets and Resources") == "assetsandresources"
    assert normalize_heading("4-2 Assets and Resources") == "assetsandresources"
    assert normalize_heading("4:2 Assets and Resources") == "assetsandresources"


if __name__ == "__main__":
    _main()

