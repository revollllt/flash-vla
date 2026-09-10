"""Rebuild historical result views from their saved traces."""
from pathlib import Path

from . import index, read_json, render


def rebuild(root):
    results = Path(root) / "results"
    paths = index.trace_paths(results)
    for path in paths:
        render.write(path.parent, read_json(path))
    index.rebuild(results)
    return {"traces": len(paths)}
