"""Rebuild historical result views from their saved traces."""
from pathlib import Path

from . import curve, index, read_json, render


def rebuild(root):
    results = Path(root) / "results"
    paths = index.trace_paths(results)
    for path in paths:
        render.write(path.parent, read_json(path))
    # Run figures are byte-stable, so regenerating them here cannot churn the
    # tree and one command brings every view under results/ current.
    figures = sorted(results.glob(f"*/*/{curve.TABLE_NAME}"))
    for table in figures:
        curve.render(table.parent)
    index.rebuild(results)
    return {"traces": len(paths), "figures": len(figures)}
