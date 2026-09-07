#!/usr/bin/env python3
"""List unit-bearing numbers in wiki prose that are not a machine-constant tag,
a performance_claims record, or a fenced block: the input to
audit/numeric-claims-ledger.md.

    python3 scripts/scan_numbers.py            # every wiki page
    python3 scripts/scan_numbers.py --sm90     # pages whose architectures include sm90
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402

NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|x|KB|MB|GB|us|ns|ms|TFLOPS|PFLOPS|TB/s|GB/s|cycles)\b")
FENCE_RE = re.compile(r"```.*?```", re.S)
BRACKET_RE = re.compile(r"\[[^\]]*\]")


def main() -> int:
    only_sm90 = "--sm90" in sys.argv
    total = 0
    for page in sorted((WIKI_ROOT / "wiki").rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        m = re.match(r"\A---\n(.*?)\n---\n(.*)", text, re.S)
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        if only_sm90 and "sm90" not in (fm.get("architectures") or []):
            continue
        prose = BRACKET_RE.sub("", FENCE_RE.sub("", m.group(2)))
        hits = NUM_RE.findall(prose)
        if hits:
            total += len(hits)
            print(f"{page.relative_to(WIKI_ROOT)}: {', '.join(hits)}")
    print(f"\n{total} unit-bearing numbers in prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
