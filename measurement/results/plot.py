"""Canonical Matplotlib renderer for normalized optimization traces."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import fill
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
    execution_variant: dict | None = None


@dataclass(frozen=True)
class PlotPoint:
    iteration: int | None
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
    position: float | None = None
    promotion: str | None = None
    parent_ms: float | None = None

    @property
    def x(self):
        return self.iteration if self.position is None else self.position


def from_trace(value):
    metadata = value['metadata']
    plot_metadata = PlotMetadata(
        **{key: metadata[key] for key in PlotMetadata.__annotations__ if key != 'execution_variant'},
        execution_variant=metadata.get('execution_variant'))
    points = [PlotPoint(iteration=item['iteration'], candidate_id=item['candidate_id'],
                        verdict=item['verdict'], segment=item['measurement_segment']['id'],
                        candidate_ms=item['candidate_latency_ms'],
                        incumbent_ms=item['current_incumbent_latency_ms'],
                        baseline_ms=item['segment_baseline_latency_ms'],
                        delta_parent_pct=item['delta_vs_parent_pct'],
                        delta_baseline_pct=item['delta_vs_baseline_pct'],
                        summary=item['change_summary'], reanchor=item.get('reanchor', False),
                        promotion=item.get('promotion'),
                        parent_ms=item.get('parent_incumbent_latency_ms'))
              for item in value['iterations']]
    for segment in value.get('segments', [])[1:]:
        context = segment['measurement_context']
        points.append(PlotPoint(
            iteration=None, position=segment['before_iteration'] - 0.5,
            candidate_id=f'reanchor-{segment["id"]}', verdict=None, segment=segment['id'],
            candidate_ms=None, incumbent_ms=segment['anchor_ms'], baseline_ms=segment['anchor_ms'],
            delta_parent_pct=None, delta_baseline_pct=None, reanchor=True,
            summary=f'checkpoint {context["weights"]["checkpoint_id"]}; fixture {context["fixture"]["id"]}'))
    return plot_metadata, sorted(points, key=lambda point: (point.x, point.segment))


def render_optimization_progress(*, metadata: PlotMetadata, points: Sequence[PlotPoint],
                                 output_svg: Path, output_png: Path | None = None) -> None:
    if not points:
        raise ValueError('cannot render optimization progress without points')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = sorted(points, key=lambda point: (point.x, point.segment))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    segments = list(dict.fromkeys(point.segment for point in points))
    for index, segment in enumerate(segments):
        group = [point for point in points if point.segment == segment]
        ax.plot([point.x for point in group], [point.incumbent_ms for point in group],
                marker='o', linewidth=1.8, color='C0',
                label='recorded portable incumbent' if index == 0 else None)
    paired_label = True
    for point in points:
        if point.parent_ms is not None and point.verdict in {'accepted', 'no_benefit'}:
            ax.plot([point.x, point.x], [point.parent_ms, point.candidate_ms],
                    linestyle=':', color='C1', linewidth=1.2)
            ax.scatter(point.x, point.parent_ms, marker='D', facecolors='none',
                       edgecolors='C1', s=45, zorder=5,
                       label='same-run parent control' if paired_label else None)
            paired_label = False
            ax.annotate(f'paired control\n{point.parent_ms:.3f} ms',
                        xy=(point.x, point.parent_ms),
                        xytext=(-8 if point.x == points[-1].x else 8, 8),
                        horizontalalignment='right' if point.x == points[-1].x else 'left',
                        textcoords='offset points', fontsize=8)
        if point.verdict == 'no_benefit':
            ax.scatter(point.x, point.candidate_ms, marker='x', color='0.25',
                       alpha=0.85, s=65, linewidths=1.5, zorder=4)
        elif point.promotion == 'context_only':
            ax.scatter(point.x, point.candidate_ms, marker='D', color='0.5', alpha=0.65)
        elif point.verdict in {'correctness_failed', 'invalid', 'blocked'}:
            missing_latency = point.candidate_ms is None
            value = 0.025 if missing_latency else point.candidate_ms
            coordinates = ax.get_xaxis_transform() if missing_latency else ax.transData
            ax.scatter(point.x, value, marker='v', color='C3', s=55,
                       zorder=5, transform=coordinates)
            label = f'iter {point.iteration}: {point.verdict}'
            if missing_latency:
                label += '\nno valid latency'
            ax.annotate(label, xy=(point.x, value), xycoords=coordinates,
                        xytext=(0, 8), textcoords='offset points', fontsize=8,
                        horizontalalignment='center', color='C3')
    previous_segment = points[0].segment
    for point in points[1:]:
        if point.segment != previous_segment:
            ax.axvline(point.x if point.reanchor else point.x - 0.5, linestyle='--', linewidth=1.0, alpha=0.6)
            previous_segment = point.segment
    previous_was_anchor = False
    previous_segment = points[0].segment
    for point in points:
        segment_changed = point.segment != previous_segment
        near_right_edge = point.x >= points[-1].x - 0.5
        below = (previous_was_anchor or point.parent_ms is not None) and not point.reanchor
        offset = (-8 if near_right_edge else 8, 48 if point.reanchor else (-16 if below else 16))
        alignment = 'right' if near_right_edge else 'left'
        vertical = 'top' if below else 'bottom'
        if segment_changed or point.reanchor:
            label = f'segment {point.segment} re-anchor\n{point.incumbent_ms:.3f} ms'
            if point.verdict == 'accepted' and not point.reanchor:
                parent_delta = ('n/a' if point.delta_parent_pct is None
                                else f'{point.delta_parent_pct:+.2f}%')
                baseline_delta = ('n/a' if point.delta_baseline_pct is None
                                  else f'{point.delta_baseline_pct:+.2f}%')
                label += (f'\niter {point.iteration}: accepted'
                          f'\n{parent_delta} vs parent\n{baseline_delta} vs baseline')
            ax.annotate(label,
                        xy=(point.x, point.incumbent_ms), xytext=offset,
                        textcoords='offset points', fontsize=8, horizontalalignment=alignment,
                        verticalalignment=vertical)
        elif point.verdict == 'accepted' and point.promotion != 'context_only':
            parent_delta = ('n/a' if point.delta_parent_pct is None
                            else f'{point.delta_parent_pct:+.2f}%')
            baseline_delta = ('n/a' if point.delta_baseline_pct is None
                              else f'{point.delta_baseline_pct:+.2f}%')
            label = (f'iter {point.iteration}: {"baseline" if point.iteration == 0 else "accepted"}\n{point.incumbent_ms:.3f} ms'
                     f'\n{parent_delta} vs parent\n{baseline_delta} vs baseline')
            ax.annotate(label, xy=(point.x, point.incumbent_ms), xytext=offset,
                        textcoords='offset points', fontsize=8, horizontalalignment=alignment,
                        verticalalignment=vertical)
        previous_was_anchor = point.reanchor
        previous_segment = point.segment
    shape_title = fill(f'{metadata.shape_profile} | {metadata.precision}', width=100)
    title = (f'{metadata.hardware} | {metadata.model} @ {metadata.model_revision}\n'
             f'{shape_title}')
    subtitle = f'objective={metadata.objective} | protocol={metadata.protocol}'
    if metadata.execution_variant is not None:
        variant = json.dumps(metadata.execution_variant, sort_keys=True, separators=(',', ':'))
        subtitle += '\n' + fill('variant=' + variant, width=100)
    ax.set_title(f'{title}\n{subtitle}')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel('Optimization iteration (re-anchors do not consume an iteration)')
    ax.set_ylabel('Latency (ms)')
    ax.margins(y=0.35)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({'svg.hashsalt': 'flash-vla'}):
        fig.savefig(output_svg, format='svg', bbox_inches='tight',
                    metadata={'Creator': 'flash-vla', 'Date': None})
    if output_png is not None:
        fig.savefig(output_png, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
