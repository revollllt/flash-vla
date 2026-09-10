"""Saved result views keep checkpoint contexts and unsuccessful trials separate."""
from lab.results import html, plot, schema


def trace():
    variant = {"quantization": {"mode": "bf16"}, "cache": {"mode": "none"}}
    target = dict(hardware="h100", model="pi05", model_revision="pi05-r1", shape={"chunk": 50})
    metadata = dict(hardware="h100", model="pi05", model_revision="pi05-r1",
                    shape_profile="chunk50", precision="bf16", objective="chunk_latency",
                    protocol="latency-v2", execution_variant=variant)
    segments, rows = [], []
    for i, (checkpoint, anchor, candidate) in enumerate((("a", 20., 18.), ("b", 100., 90.))):
        context = dict(weights={"checkpoint_id": checkpoint, "checkpoint_digest": checkpoint},
                       fixture={"id": "input", "digest": "input"}, environment={})
        segments.append(dict(id=i, before_iteration=i, incumbent=f"iteration-{i}",
                             identity=dict(engine_revision="source", plan={}),
                             measurement_context=context, anchor_ms=anchor))
        rows.append(dict(iteration=i, candidate_id=f"trial-{i}", target_key=target,
                         verdict="accepted", promotion="promoted", measurement_segment={"id": i},
                         engine_revision="source", plan={}, candidate_latency_ms=candidate,
                         current_incumbent_latency_ms=candidate, segment_baseline_latency_ms=anchor,
                         delta_vs_parent_pct=-10., delta_vs_baseline_pct=-10., change_summary="test"))
    return dict(campaign_id="history", metadata=metadata, segments=segments, iterations=rows)


def test_result_summary_and_dashboard_keep_contexts_separate():
    summary, contexts = schema.summaries(trace())
    assert summary["current_performance"]["best_ms"] == 90.
    assert summary["current_performance"]["anchor_ms"] == 100.
    assert sorted(c["best_validated_ms"] for c in contexts.values()) == [18., 90.]
    page = html.render([(summary, contexts, "targets/example")])
    assert 'data-context="a"' in page and 'data-context="b"' in page


def test_plot_keeps_reanchors_and_context_only_trials_out_of_promotions():
    value = trace()
    row = value["iterations"][1]
    row.update(promotion="context_only", current_incumbent_latency_ms=100.)
    summary, _ = schema.summaries(value)
    assert summary["current_performance"]["best_ms"] == 100.
    _, points = plot.from_trace(value)
    assert [(p.x, p.incumbent_ms) for p in points] == [(0, 18.), (0.5, 100.), (1, 100.)]
    assert points[1].reanchor and points[1].iteration is None
    assert points[2].candidate_ms == 90. and points[2].promotion == "context_only"
    row.update(verdict="invalid", candidate_latency_ms=None)
    assert plot.from_trace(value)[1][-1].candidate_ms is None
