#!/usr/bin/env python3
"""Re-pin a derived bundle's PROVENANCE.yaml to the files on disk.

    .venv/bin/python scripts/pin_artifacts.py artifacts/kernels/sm90-templates/variants

Verbatim bundles are pinned to an upstream SHA and are never touched by this
script. A derived bundle holds this repository's own files; after editing one,
run this so validate.py's sha256 drift check passes and a page's STATUS number
is bound to the file version it was taken from. Roles: `*.cu` / `*.cuh` /
`*.py` are derived-source, `README.md` is approach-notes; other fields of the
manifest are preserved.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402

SOURCE_EXTS = {".cu", ".cuh", ".py", ".sh", ".txt", ".yaml", ".json", ".patch"}
HEADER = ("## Provenance of a derived bundle: this repository's own files, whose\n"
          "## methods come from the sources named in derived_from. Re-pin after\n"
          "## editing with scripts/pin_artifacts.py; validate.py checks the digests.\n")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    bundle = (WIKI_ROOT / sys.argv[1]).resolve()
    prov_path = bundle / "PROVENANCE.yaml"
    prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    if prov.get("asset_mode") != "derived":
        print(f"{prov_path}: only derived bundles are re-pinned here (asset_mode={prov.get('asset_mode')!r})")
        return 1
    files = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "PROVENANCE.yaml":
            continue
        rel = path.relative_to(bundle).as_posix()
        if path.name == "README.md":
            role = "approach-notes"
        elif path.suffix.lower() in SOURCE_EXTS:
            role = "derived-source"
        else:
            continue
        files.append({"local_path": rel, "role": role, "mode": "derived", "sha256": sha256_of(path)})
    prov["files"] = files
    prov_path.write_text(HEADER + yaml.safe_dump(prov, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"pinned {len(files)} files in {bundle.relative_to(WIKI_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
