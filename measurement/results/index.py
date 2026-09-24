"""Deterministic discovery and Markdown dashboard over published summaries."""
import json
import re
from pathlib import Path

from . import read_json, write_json
from . import html, schema


def _cell(value):
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


#: Everything between these is regenerated. A human may write below the end
#: marker and it survives; the file used to be rewritten whole, which silently
#: dropped the note saying the legacy table was historical.
# The markers predate the move from `lab.results`; every saved results/ page
# carries them, so they keep the old package name.
GENERATED_BEGIN = "<!-- lab.results: generated below -->"
GENERATED_END = "<!-- lab.results: end generated; hand-written notes go below -->"


def _blurb(directory):
    """The run's own opening paragraph, collapsed to one line.

    The index does not restate a number: it quotes the run README, which is the
    file that has to be right anyway. An index carrying its own copy of a
    latency goes stale without anyone noticing.
    """
    readme = Path(directory) / "README.md"
    if not readme.is_file():
        return None
    paragraph = []
    for line in readme.read_text().splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    text = " ".join(paragraph)
    # A quoted paragraph keeps its prose but not its navigation: the run
    # README's relative links do not resolve from this file's depth.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return text or None


def _preserved(path):
    """Hand-written text below the generated block, or nothing on a fresh file."""
    if not path.is_file():
        return ""
    text = path.read_text()
    if GENERATED_END not in text:
        return ""
    return text.split(GENERATED_END, 1)[1].strip("\n")


def trace_paths(root):
    return sorted([*root.glob("targets/*/trace.json"),
                   *root.glob("targets/*/forks/*/trace.json")])


def rebuild(root):
    root = Path(root)
    paths = trace_paths(root)
    entries, dashboards, rows = [], [], []
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
        rows.append("| " + " | ".join(cells) + " |")

    lines = [GENERATED_BEGIN, "", "# Flash-VLA results", "",
             "Performance comparisons are local to the representative "
             "checkpoint/fixture segment.", ""]

    # Current runs first. The legacy table below has a column called "Current
    # best ms" that is nothing of the kind, and putting it at the top of the
    # page is what got one of its numbers quoted as this repository's best.
    runs = [path.parent for path in sorted(root.glob("*/*/iterations.csv"))]
    if runs:
        lines.extend(["## Current optimization runs", ""])
        for directory in runs:
            relative = directory.relative_to(root).as_posix()
            blurb = _blurb(directory)
            lines.append(f"- [{_cell(relative)}]({relative}/README.md)"
                         + (f" — {blurb}" if blurb else ""))
        lines.append("")

    if rows:
        lines.extend([
            "## Retired Campaign entries", "",
            "Saved traces under `targets/`, from the retired Campaign "
            "controller. **These are historical.** The \"Current best ms\" "
            "column is that campaign's last measurement, not a number this "
            "repository is working from; the runs above carry those.", "",
            "| Hardware | Model revision | Shape | Variant | Context | Anchor ms "
            "| Current best ms | Speedup | Iter |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
            *rows, ""])

    lines.append(GENERATED_END)
    tail = _preserved(root / "README.md")
    if tail:
        lines.extend(["", tail])

    value = dict(schema_version=1, campaigns=entries)
    write_json(root / "index.json", value)
    (root / "README.md").write_text("\n".join(lines) + "\n")
    (root / "index.html").write_text(html.render(dashboards))
    return value
