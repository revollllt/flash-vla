#!/usr/bin/env bash
# Verify that the skill's read paths work with a Python interpreter that has
# no site-packages. `-S` disables the system site directory, including PyYAML.
# This cluster's system python3 is 3.6; the scripts need 3.7+, so the repo
# venv interpreter is used with -S.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/../../../.venv/bin/python}"

"$PY" -S - <<'PY'
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from _yaml_compat import USING_BUNDLED, yaml

assert USING_BUNDLED, "fallback test unexpectedly imported host PyYAML"

frontmatter = Path("wiki/hardware/wgmma.md").read_text(encoding="utf-8")
match = re.match(r"^---\s*\n(.*?)\n---", frontmatter, re.S)
data = yaml.safe_load(match.group(1))
assert data["id"] == "hw-wgmma"
assert "wgmma" in data["tags"]

round_trip = yaml.safe_load(yaml.safe_dump({"id": data["id"], "tags": data["tags"]}))
assert round_trip["tags"] == data["tags"]
PY

query_output="$("$PY" -S scripts/query.py --symptom stall-gmma --compact)"
grep -q "pattern-serialized-wgmma" <<<"$query_output"

page_output="$("$PY" -S scripts/get_page.py hw-wgmma --frontmatter-only)"
grep -q "id: hw-wgmma" <<<"$page_output"

echo "OK: kernel-wiki works without host PyYAML"
