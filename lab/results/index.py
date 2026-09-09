"""Deterministic discovery and Markdown dashboard over published summaries."""
import json
from pathlib import Path

from lab.optimize import store
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
        summary = schema.check_views(trace_path.parent)
        _, contexts = schema.summaries(store.read(trace_path))
        dashboards.append((summary, contexts, trace_path.parent.relative_to(root).as_posix()))
        key, performance = summary["campaign_key"], summary["current_performance"]
        relative = path.relative_to(root).as_posix()
        entries.append(dict(campaign_key=key, lineage_id=summary["lineage_id"], summary=relative))
        context = store.read(path.parent / "contexts" / summary["representative_context"] / "summary.json")
        target = key["target"]
        label = _cell(target["model_revision"])
        cells = [_cell(target["hardware"]), f"[{label}]({path.parent.relative_to(root).as_posix()}/README.md)",
                 _cell(target["shape"]), _cell(key["execution_variant"]), _cell(context["weights"]["checkpoint_id"]),
                 f'{performance["anchor_ms"]:.3f}', f'{performance["best_ms"]:.3f}',
                 f'{performance["speedup"]:.3f}×', str(summary["latest_iteration"])]
        lines.append("| " + " | ".join(cells) + " |")
    value = dict(schema_version=1, campaigns=entries)
    store.write(root / "index.json", value)
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "index.html").write_text(html.render(dashboards))
    return value
