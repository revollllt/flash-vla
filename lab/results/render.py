"""Derived publication views, shared by publish and offline rebuild."""
from lab.optimize import render, store

from . import schema


def write(directory, value):
    """Render one validated compact trace without reading its old machine's artifacts."""
    summary, contexts = schema.summaries(value)
    for context_id, context in contexts.items():
        store.write(directory / "contexts" / context_id / "summary.json", context)
    plot_metadata, points = render.from_trace(value)
    render.render_optimization_progress(
        metadata=plot_metadata, points=points, output_svg=directory / "progress.svg")
    performance = summary["current_performance"]
    text = (
        f'# {plot_metadata.hardware} | {plot_metadata.model_revision}\n\n'
        f'Objective: {plot_metadata.objective}; protocol: {plot_metadata.protocol}.\n\n'
        f'Representative context: [{summary["representative_context"]}]'
        f'(contexts/{summary["representative_context"]}/summary.json). '
        f'Anchor {performance["anchor_ms"]:.3f} ms; '
        f'current portable incumbent {performance["best_ms"]:.3f} ms '
        f'({performance["speedup"]:.3f}× within this segment).\n\n'
        '[Summary](summary.json) · [Trace](trace.json)\n\n'
        '![Optimization progress](progress.svg)\n')
    (directory / "README.md").write_text(text)
    store.write(directory / "summary.json", summary)
    return ([directory / "progress.svg", directory / "README.md"]
            + [directory / "contexts" / name / "summary.json" for name in sorted(contexts)]
            + [directory / "summary.json"])
