"""Canonical Matplotlib renderer for normalized optimization traces."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PlotMetadata:
    hardware: str
    model: str
    model_revision: str
    shape_profile: str
    precision: str
    objective: str
    protocol: str


@dataclass(frozen=True)
class PlotPoint:
    iteration: int
    candidate_id: str
    verdict: str | None
    segment: int
    candidate_ms: float | None
    incumbent_ms: float
    baseline_ms: float
    delta_parent_pct: float | None
    delta_baseline_pct: float | None
    summary: str
    reanchor: bool = False


def from_trace(value):
    metadata = value['metadata']
    plot_metadata = PlotMetadata(**{key: metadata[key] for key in PlotMetadata.__annotations__})
    points = [PlotPoint(iteration=item['iteration'], candidate_id=item['candidate_id'],
                        verdict=item['verdict'], segment=item['measurement_segment']['id'],
                        candidate_ms=item['candidate_latency_ms'],
                        incumbent_ms=item['current_incumbent_latency_ms'],
                        baseline_ms=item['segment_baseline_latency_ms'],
                        delta_parent_pct=item['delta_vs_parent_pct'],
                        delta_baseline_pct=item['delta_vs_baseline_pct'],
                        summary=item['change_summary'], reanchor=item.get('reanchor', False))
              for item in value['iterations']]
    return plot_metadata, points


def render_optimization_progress(*, metadata: PlotMetadata, points: Sequence[PlotPoint],
                                 output_svg: Path, output_png: Path | None = None) -> None:
    if not points:
        raise ValueError('cannot render optimization progress without points')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = sorted(points, key=lambda point: point.iteration)
    iterations = [point.iteration for point in points]
    incumbent_ms = [point.incumbent_ms for point in points]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(iterations, incumbent_ms, marker='o', linewidth=1.8,
            label='accepted incumbent')
    for point in points:
        if point.verdict == 'no_benefit':
            ax.scatter(point.iteration, point.candidate_ms, marker='x', alpha=0.65)
        elif point.verdict in {'correctness_failed', 'invalid', 'blocked'}:
            value = point.incumbent_ms if point.candidate_ms is None else point.candidate_ms
            ax.scatter(point.iteration, value, marker='.', alpha=0.45)
    previous_segment = points[0].segment
    for point in points[1:]:
        if point.segment != previous_segment:
            ax.axvline(point.iteration - 0.5, linestyle='--', linewidth=1.0, alpha=0.6)
            previous_segment = point.segment
    previous_incumbent = None
    previous_segment = points[0].segment
    for point in points:
        segment_changed = point.segment != previous_segment
        incumbent_changed = previous_incumbent is None or point.incumbent_ms < previous_incumbent
        at_right_edge = point.iteration == points[-1].iteration
        offset = (-6, 8) if at_right_edge else (6, -28)
        alignment = 'right' if at_right_edge else 'left'
        if segment_changed or point.reanchor:
            label = f'segment {point.segment} re-anchor\n{point.incumbent_ms:.3f} ms'
            if point.verdict == 'accepted' and not point.reanchor:
                parent_delta = ('n/a' if point.delta_parent_pct is None
                                else f'{point.delta_parent_pct:+.2f}%')
                baseline_delta = ('n/a' if point.delta_baseline_pct is None
                                  else f'{point.delta_baseline_pct:+.2f}%')
                label += (f'\niter {point.iteration}: {point.summary}'
                          f'\n{parent_delta} vs parent\n{baseline_delta} vs baseline')
            ax.annotate(label,
                        xy=(point.iteration, point.incumbent_ms), xytext=offset,
                        textcoords='offset points', fontsize=8, horizontalalignment=alignment)
        elif point.verdict == 'accepted' and incumbent_changed:
            parent_delta = ('n/a' if point.delta_parent_pct is None
                            else f'{point.delta_parent_pct:+.2f}%')
            baseline_delta = ('n/a' if point.delta_baseline_pct is None
                              else f'{point.delta_baseline_pct:+.2f}%')
            label = (f'iter {point.iteration}: {point.summary}\n{point.incumbent_ms:.3f} ms'
                     f'\n{parent_delta} vs parent\n{baseline_delta} vs baseline')
            ax.annotate(label, xy=(point.iteration, point.incumbent_ms), xytext=offset,
                        textcoords='offset points', fontsize=8, horizontalalignment=alignment)
        previous_incumbent = point.incumbent_ms
        previous_segment = point.segment
    title = (f'{metadata.hardware} | {metadata.model} @ {metadata.model_revision}\n'
             f'{metadata.shape_profile} | {metadata.precision}')
    subtitle = f'objective={metadata.objective} | protocol={metadata.protocol}'
    ax.set_title(f'{title}\n{subtitle}')
    ax.set_xlabel('Optimization iteration')
    ax.set_ylabel('Latency (ms)')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format='svg', bbox_inches='tight',
                metadata={'Creator': 'flash-vla', 'Date': None})
    if output_png is not None:
        fig.savefig(output_png, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def render_html(metadata: PlotMetadata, points: Sequence[PlotPoint], output_html: Path):
    """Optional static view of PlotPoint values; it performs no attribution."""
    rows = []
    for point in points:
        candidate = '—' if point.candidate_ms is None else f'{point.candidate_ms:.3f}'
        rows.append(f'<tr><td>{point.iteration}</td><td>{point.segment}</td>'
                    f'<td>{escape(point.candidate_id)}</td><td>{escape(point.verdict or "active")}</td>'
                    f'<td>{candidate}</td><td>{point.incumbent_ms:.3f}</td>'
                    f'<td>{escape(point.summary)}</td></tr>')
    title = escape(f'{metadata.hardware} | {metadata.model} @ {metadata.model_revision}')
    subtitle = escape(f'objective={metadata.objective} | protocol={metadata.protocol}')
    html = (f'<!doctype html><meta charset="utf-8"><title>{title}</title><h1>{title}</h1>'
            f'<p>{subtitle}</p><table><thead><tr><th>Iteration</th><th>Segment</th>'
            '<th>Candidate</th><th>Verdict</th><th>Candidate ms</th><th>Incumbent ms</th>'
            f'<th>Summary</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    Path(output_html).write_text(html)
