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


def test_index_keeps_hand_written_notes_and_names_the_legacy_table(tmp_path):
    """The index is regenerated, but not at the cost of what a human added.

    The legacy table's "Current best ms" column is a retired controller's last
    measurement. When the note saying so was hand-written it was dropped by the
    next rebuild, and one of those numbers was then quoted as a current best.
    """
    from lab.results import index

    run = tmp_path / "pi0-rtx5090" / "run-01"
    run.mkdir(parents=True)
    (run / "iterations.csv").write_text(
        "iteration,change,latency_ms,decision,report,revision\n"
        "0,Start,10.0,start,,rev\n")
    (run / "README.md").write_text(
        "# run-01\n\nFirst run, see [the plan](../plan.md): **10.0 ms**.\n")

    index.rebuild(tmp_path)
    text = (tmp_path / "README.md").read_text()
    assert "## Current optimization runs" in text
    # The run's own summary is quoted rather than restated, with its relative
    # links flattened: they do not resolve from the index's depth.
    assert "First run, see the plan: **10.0 ms**." in text
    assert "(../plan.md)" not in text

    (tmp_path / "README.md").write_text(text + "\nA hand-written caveat.\n")
    index.rebuild(tmp_path)
    assert "A hand-written caveat." in (tmp_path / "README.md").read_text()


def test_curve_reads_a_run_directory_and_renders_the_same_bytes_twice(tmp_path):
    """One renderer for every run: the table is enough, figure.json tunes it."""
    from lab.results import curve

    run = tmp_path / "rig" / "run-01"
    run.mkdir(parents=True)
    (run / "iterations.csv").write_text(
        "iteration,change,latency_ms,decision,report,revision\n"
        "0,A very long change description that must be trimmed somewhere,10.0,start,,a\n"
        "1,Second,9.0,keep,,b\n"
        "3,Third after a rejected trial,8.0,keep,,c\n")

    first = curve.render(run)
    assert first == run / "progress.svg"
    before = first.read_bytes()
    assert curve.render(run).read_bytes() == before, "output must be byte-stable"

    # A label is trimmed at a word boundary rather than mid-word.
    _, _, settings = curve.resolve(run)
    assert settings == {}
    rows = curve._rows(run / "iterations.csv")
    points, _ = curve._milestones(rows, run)
    assert points[0]["label"].endswith("…") and " " in points[0]["label"]
    assert not points[0]["label"].rstrip("…").endswith(" ")

    (run / "figure.json").write_text('{"title": "T", "roofline_ms": 4.0, "ignored": 1}')
    _, _, settings = curve.resolve(run)
    assert settings == {"title": "T", "roofline_ms": 4.0}
    assert curve.render(run).exists()
