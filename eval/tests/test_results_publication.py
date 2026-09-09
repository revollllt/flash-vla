"""Publication behavior on CPU toy ledgers; no real model performance claim."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from eval.tests.test_weight_dependency import workspace, candidate, finalize
from eval.tests.test_context_transition import inherited_recipes, request, next_spec
from lab.optimize import campaign, store, transition, render
from lab.optimize.registry import CampaignRegistry
from lab.results.publish import publish


def files(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_publish_discovery_and_deterministic_views_without_raw_artifacts(workspace):
    root, directory = workspace
    (directory / "raw.ncu-rep").write_text("raw sentinel")
    result = publish(root, directory)
    destination = Path(result["directory"])
    summary = store.read(destination / "summary.json")
    context_id = summary["representative_context"]
    assert summary["current_performance"] == dict(anchor_ms=16., best_ms=16., speedup=1., best_iteration=0)
    assert summary["validated_context_count"] == 1
    context = store.read(destination / "contexts" / context_id / "summary.json")
    assert context["latest_segment"] == 0
    value = store.read(destination / "trace.json")
    assert value["iterations"][0]["context_id"] == context_id
    assert value["iterations"][0]["plan"] == context["validated_incumbent"]["plan"]
    assert "diagnostic_artifacts" not in value["iterations"][0]
    assert not list(destination.rglob("*.ncu-rep"))
    index = store.read(root / "results/index.json")
    assert len(index["campaigns"]) == 1
    assert index["campaigns"][0]["lineage_id"] == summary["lineage_id"]
    assert root / "results" / index["campaigns"][0]["summary"] == destination / "summary.json"
    assert "within this segment" in (destination / "README.md").read_text()
    before = files(root / "results")
    publish(root, directory)
    assert files(root / "results") == before


def test_checkpoint_publication_uses_new_anchor_without_cross_context_best(workspace):
    root, directory = workspace
    inherited_recipes(root, directory)
    initial = publish(root, directory)
    context_a = initial["summary"]["representative_context"]
    state = campaign.transition_context(root, directory, request(root, directory, latency=20.))
    result = publish(root, directory)
    assert result["directory"] == initial["directory"]
    summary = result["summary"]
    assert summary["representative_context"] == state["active_context"] != context_a
    assert summary["current_performance"] == dict(anchor_ms=20., best_ms=20., speedup=1., best_iteration=3)
    assert summary["latest_iteration"] == 3 and summary["validated_context_count"] == 2
    destination = Path(result["directory"])
    previous = store.read(destination / "contexts" / context_a / "summary.json")
    assert previous["best_validated_ms"] == 14.
    value = store.read(destination / "trace.json")
    assert [r["verdict"] for r in value["iterations"]] == ["accepted", "accepted", "no_benefit", "accepted"]
    assert len(value["segments"]) == 2 and value["segments"][1]["before_iteration"] == 4
    assert "checkpoint task-b" in (destination / "progress.svg").read_text()
    assert len(store.read(root / "results/index.json")["campaigns"]) == 1


def test_faster_correctness_failure_is_visible_without_changing_incumbent(workspace):
    root, directory = workspace
    campaign.start(root, candidate("bad"), directory)
    campaign.finalize(directory, 1, "correctness_failed", {"status": "failed"},
                      {"validity": "incomplete", "candidate_ms": 1.}, {"status": "not_run"},
                      diagnostics={"stderr": "raw sentinel"})
    result = publish(root, directory)
    assert result["summary"]["current_performance"]["best_ms"] == 16.
    trace = store.read(Path(result["directory"]) / "trace.json")
    row = trace["iterations"][1]
    assert row["verdict"] == "correctness_failed"
    assert row["candidate_latency_ms"] == 1.
    assert row["current_incumbent_latency_ms"] == 16.
    assert row["context_id"] == trace["iterations"][0]["context_id"]
    assert "raw sentinel" not in json.dumps(trace)


@pytest.mark.parametrize("unfinished", ["candidate", "transition", "aborted"])
def test_unfinished_context_or_candidate_is_not_published(workspace, unfinished):
    root, directory = workspace
    if unfinished == "candidate":
        campaign.start(root, candidate(), directory)
    else:
        transition.begin(root, directory, request(root, directory))
        if unfinished == "aborted":
            transition.abort(directory)
    with pytest.raises(ValueError, match="finalize|re-anchor"):
        publish(root, directory)
    assert not (root / "results").exists()


def test_render_failure_preserves_the_previous_publication(workspace, monkeypatch):
    root, directory = workspace
    publish(root, directory)
    before = files(root / "results")
    campaign.start(root, candidate("no-change"), directory)
    finalize(directory, 1, "no_benefit")
    def fail(**kwargs):
        raise RuntimeError("render failed")
    monkeypatch.setattr(render, "render_optimization_progress", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        publish(root, directory)
    assert files(root / "results") == before


def test_same_key_cannot_overwrite_another_lineage(workspace):
    root, directory = workspace
    publish(root, directory)
    path = directory / "campaign.json"
    metadata = store.read(path)
    metadata["id"] = "another-lineage"
    store.write(path, metadata)
    with pytest.raises(ValueError, match="another lineage"):
        publish(root, directory)


def test_published_terminal_history_cannot_be_rewritten(workspace):
    root, directory = workspace
    campaign.start(root, candidate("no-change"), directory)
    finalize(directory, 1, "no_benefit")
    publish(root, directory)
    path = next((directory / "runs").glob("iter-001-*/evidence.json"))
    record = store.read(path)
    record["change"]["summary"] = "rewritten history"
    store.write(path, record)
    with pytest.raises(ValueError, match="rewrite or truncate"):
        publish(root, directory)


@pytest.mark.parametrize("fork_first", [False, True])
def test_explicit_fork_has_separate_publication(workspace, fork_first):
    root, old = workspace
    baseline = store.read(old / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2")
    registry = CampaignRegistry(root)
    directory = registry.create(key, baseline=baseline, inputs=["src/kernel.cu"])
    child_directory = registry.fork(key, reason="alternative implementation")
    if fork_first:
        child = publish(root, child_directory)
        parent = publish(root, directory)
    else:
        parent = publish(root, directory)
        child = publish(root, child_directory)
    assert Path(child["directory"]).parent == Path(parent["directory"]) / "forks"
    assert child["summary"]["fork"]["parent_campaign"] == parent["summary"]["lineage_id"]
    assert len(store.read(root / "results/index.json")["campaigns"]) == 2


def test_publish_cli(workspace):
    root, directory = workspace
    output = subprocess.run([sys.executable, "-m", "lab.results", "publish", str(directory),
                             "--root", str(root)], check=True, capture_output=True, text=True)
    result = json.loads(output.stdout)
    assert (Path(result["directory"]) / "summary.json").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_replacement_failure_keeps_trace_owner_and_same_lineage_can_recover(workspace, monkeypatch, existing):
    import os
    from lab.results import index

    root, directory = workspace
    baseline = store.read(directory / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2")
    from lab.optimize.registry import key_digest
    destination = root / "results/targets" / key_digest(key)
    if existing:
        publish(root, directory)
        campaign.start(root, candidate("next"), directory)
        finalize(directory, 1, "no_benefit")
    replace = os.replace
    def fail_summary(source, target, **kwargs):
        if Path(target) == destination / "summary.json":
            raise OSError("interrupted summary replacement")
        return replace(source, target, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_summary)
        with pytest.raises(OSError, match="interrupted summary"):
            publish(root, directory)
    assert store.read(destination / "trace.json")["campaign_id"] == directory.name
    with pytest.raises((FileNotFoundError, ValueError)):
        index.rebuild(root / "results")
    path = directory / "campaign.json"
    metadata = store.read(path)
    store.write(path, dict(metadata, id="intruding-lineage"))
    with pytest.raises(ValueError, match="another lineage"):
        publish(root, directory)
    store.write(path, metadata)
    result = publish(root, directory)
    assert result["summary"]["latest_iteration"] == int(existing)
    assert len(index.rebuild(root / "results")["campaigns"]) == 1


def test_unowned_partial_files_cannot_be_overwritten(workspace):
    from lab.optimize.registry import key_digest

    root, directory = workspace
    baseline = store.read(directory / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2")
    destination = root / "results/targets" / key_digest(key)
    destination.mkdir(parents=True)
    (destination / "README.md").write_text("unresolved previous publication")
    with pytest.raises(ValueError, match="no trace owner"):
        publish(root, directory)
    assert (destination / "README.md").read_text() == "unresolved previous publication"


def test_index_refuses_stale_context_summary(workspace):
    from lab.results import index

    root, directory = workspace
    result = publish(root, directory)
    path = Path(result["directory"]) / "contexts" / result["summary"]["representative_context"] / "summary.json"
    context = store.read(path)
    context["best_validated_ms"] = 1.
    store.write(path, context)
    before = (root / "results/index.json").read_bytes()
    with pytest.raises(ValueError, match="stale"):
        index.rebuild(root / "results")
    assert (root / "results/index.json").read_bytes() == before


def test_equal_anchor_environment_change_cannot_index_a_stale_plot(workspace, monkeypatch):
    import copy
    import os
    from lab.results import index

    root, directory = workspace
    initial = publish(root, directory)
    destination = Path(initial["directory"])
    before = campaign.rebuild(directory)
    incoming = request(root, directory, latency=16.)
    incoming["measurement_context"] = copy.deepcopy(before["measurement_context"])
    incoming["measurement_context"]["environment"]["driver"] = "changed"
    campaign.transition_context(root, directory, incoming)
    replace = os.replace
    def fail_svg(source, target, **kwargs):
        if Path(target) == destination / "progress.svg":
            raise OSError("interrupted SVG replacement")
        return replace(source, target, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_svg)
        with pytest.raises(OSError, match="interrupted SVG"):
            publish(root, directory)
    # Headline performance is identical, but the latest context segment is not.
    assert store.read(destination / "summary.json") == initial["summary"]
    with pytest.raises(ValueError, match="stale"):
        index.rebuild(root / "results")
    result = publish(root, directory)
    assert result["summary"] == initial["summary"]
    assert "segment 1 re-anchor" in (destination / "progress.svg").read_text()
