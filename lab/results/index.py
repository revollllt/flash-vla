"""Deterministic discovery and Markdown dashboard over published summaries."""
import json
from pathlib import Path

from . import read_json, write_json
from . import html, schema


def _cell(value):
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def trace_paths(root):
    return sorted([*root.glob("targets/*/trace.json"),
                   *root.glob("targets/*/forks/*/trace.json")])


def rebuild(root):
    root = Path(root)
    paths = trace_paths(root)
    entries, dashboards = [], []
    lines = ["# Flash-VLA results", "",
             "Performance comparisons are local to the representative checkpoint/fixture segment.", "",
             "| Hardware | Model revision | Shape | Variant | Context | Anchor ms | Current best ms | Speedup | Iter |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    for trace_path in paths:
        path = trace_path.parent / "summary.json"
        summary = read_json(path)
        _, contexts = schema.summaries(read_json(trace_path))
        dashboards.append((summary, contexts, trace_path.parent.relative_to(root).as_posix()))
        key, performance = summary["campaign_key"], summary["current_performance"]
        relative = path.relative_to(root).as_posix()
        entries.append(dict(campaign_key=key, lineage_id=summary["lineage_id"], summary=relative))
        context = read_json(path.parent / "contexts" / summary["representative_context"] / "summary.json")
        target = key["target"]
        label = _cell(target["model_revision"])
        cells = [_cell(target["hardware"]), f"[{label}]({path.parent.relative_to(root).as_posix()}/README.md)",
                 _cell(target["shape"]), _cell(key["execution_variant"]), _cell(context["weights"]["checkpoint_id"]),
                 f'{performance["anchor_ms"]:.3f}', f'{performance["best_ms"]:.3f}',
                 f'{performance["speedup"]:.3f}×', str(summary["latest_iteration"])]
        lines.append("| " + " | ".join(cells) + " |")
    runs = [path.parent for path in sorted(root.glob("*/*/iterations.csv"))]
    if runs:
        lines.extend(["", "## Current optimization runs", ""])
        for directory in runs:
            relative = directory.relative_to(root).as_posix()
            lines.append(f"- [{_cell(relative)}]({relative}/README.md)")
    value = dict(schema_version=1, campaigns=entries)
    write_json(root / "index.json", value)
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "index.html").write_text(html.render(dashboards))
    return value
