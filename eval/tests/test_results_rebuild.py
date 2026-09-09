"""Fault injection against real publication files; no GPU evidence is fabricated."""
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eval.tests.test_weight_dependency import workspace, candidate, finalize
from eval.tests.test_context_transition import inherited_recipes, request, next_spec
from eval.tests.test_results_publication import files
from lab.optimize import campaign, store
from lab.results import validate
from lab.results.publish import publish
from lab.results.rebuild import rebuild, validate_results


@pytest.fixture
def published(workspace):
    root, directory = workspace
    campaign.start(root, candidate("no-op"), directory)
    finalize(directory, 1, "no_benefit")
    result = publish(root, directory)
    return root, directory, Path(result["directory"])


@pytest.mark.parametrize("field,value", [
    ("delta_vs_parent_pct", -90.),
    ("delta_vs_baseline_pct", -90.),
    ("segment_baseline_latency_ms", 99.),
    ("campaign_baseline_latency_ms", 99.),
    ("current_incumbent_latency_ms", 1.),
    ("promotion", "promoted"),
    ("parent_incumbent", "iter-001"),
    ("context_id", "different"),
    ("iteration", 3),
    ("candidate_latency_ms", float("nan")),
    ("candidate_latency_ms", -1.),
    ("correctness", {"status": "failed"}),
])
def test_published_fact_corruption_is_rejected(published, field, value):
    root, _, destination = published
    trace = store.read(destination / "trace.json")
    trace["iterations"][1][field] = value
    with pytest.raises(ValueError):
        validate.trace(trace)


@pytest.mark.parametrize("mutation", ["target", "signature", "variant", "environment", "weights", "fixture", "missing_provenance"])
def test_identity_and_context_corruption_is_rejected(published, mutation):
    _, _, destination = published
    trace = store.read(destination / "trace.json")
    row = trace["iterations"][1]
    if mutation == "target":
        row["target_key"]["model_revision"] = "another-architecture"
    elif mutation == "signature":
        trace["metadata"]["inference_signature"] = "different"
    elif mutation == "variant":
        trace["metadata"]["execution_variant"]["quantization"]["mode"] = "fp8"
    elif mutation == "missing_provenance":
        del row["measurement_context"]["weights"]["checkpoint_digest"]
    else:
        group, field = {"environment": ("environment", "driver"),
                        "weights": ("weights", "checkpoint_id"),
                        "fixture": ("fixture", "id")}[mutation]
        row["measurement_context"][group][field] = "different"
    with pytest.raises(ValueError):
        validate.trace(trace)


def test_checkpoint_cannot_be_added_to_target(published):
    _, _, destination = published
    trace = store.read(destination / "trace.json")
    for row in trace["iterations"]:
        row["target_key"]["checkpoint_id"] = "not-a-target-axis"
    with pytest.raises(ValueError, match="TargetKey"):
        validate.trace(trace)


@pytest.mark.parametrize("relative", ["progress.svg", "README.md", "summary.json"])
def test_offline_rebuild_detects_and_repairs_views_without_remeasurement(published, relative):
    root, directory, destination = published
    expected = files(root / "results")
    target = destination / relative
    target.write_text("stale")
    damaged = files(root / "results")
    shutil.rmtree(directory)
    with pytest.raises(ValueError, match="stale"):
        rebuild(root, check=True)
    assert files(root / "results") == damaged
    result = rebuild(root)
    assert result["changed"] == [str(target.relative_to(root / "results"))]
    assert files(root / "results") == expected
    assert validate_results(root) == {"campaigns": 1}
    assert rebuild(root, check=True)["changed"] == []


@pytest.mark.parametrize("relative", ["index.json", "README.md"])
def test_global_views_are_checked_and_rebuilt(published, relative):
    root, _, _ = published
    expected = files(root / "results")
    (root / "results" / relative).write_text("stale")
    with pytest.raises(ValueError, match="stale"):
        rebuild(root, check=True)
    assert rebuild(root)["changed"] == [relative]
    assert files(root / "results") == expected


def test_missing_context_summary_is_recoverable(published):
    root, _, destination = published
    expected = files(root / "results")
    next((destination / "contexts").glob("*/summary.json")).unlink()
    with pytest.raises(FileNotFoundError):
        validate_results(root)
    rebuild(root)
    assert files(root / "results") == expected


def test_trace_changes_without_plot_rebuild_fail_check(published):
    root, _, destination = published
    trace = store.read(destination / "trace.json")
    trace["iterations"][0]["change_summary"] = "corrected baseline annotation"
    store.write(destination / "trace.json", trace)
    before = files(root / "results")
    with pytest.raises(ValueError, match="snapshot differs"):
        validate_results(root)
    with pytest.raises(ValueError, match="snapshot differs"):
        rebuild(root, check=True)
    assert files(root / "results") == before


def test_bad_trace_cannot_be_laundered_by_rebuild(published):
    root, _, destination = published
    trace = store.read(destination / "trace.json")
    trace["iterations"][1]["current_incumbent_latency_ms"] = 1.
    store.write(destination / "trace.json", trace)
    before = files(root / "results")
    with pytest.raises(ValueError, match="arithmetic"):
        rebuild(root)
    assert files(root / "results") == before


def test_context_transition_and_next_iteration_rebuild_on_another_filesystem(workspace, tmp_path):
    root, directory = workspace
    inherited_recipes(root, directory)
    campaign.transition_context(root, directory, request(root, directory, latency=20.))
    spec = next_spec(directory)
    campaign.start(root, spec, directory)
    finalize(directory, 4, "accepted", result={"validity": "valid", "candidate_ms": 19., "parent_incumbent_ms": 20.})
    result = publish(root, directory)
    destination = Path(result["directory"])
    trace = store.read(destination / "trace.json")
    assert trace["iterations"][4]["delta_vs_baseline_pct"] == pytest.approx(-5.)
    validate.trace(trace)
    for change in ("anchor", "boundary", "incumbent", "identity"):
        broken = copy.deepcopy(trace)
        segment = broken["segments"][1]
        if change == "anchor":
            segment["anchor_ms"] = 14.
        elif change == "boundary":
            segment["before_iteration"] = 5
        elif change == "incumbent":
            segment["incumbent"] = "iter-000"
        else:
            segment["identity"]["engine_revision"] = "different"
        with pytest.raises(ValueError):
            validate.trace(broken)
    clone = tmp_path / "another-layout"
    shutil.copytree(root / "results", clone / "results")
    assert not (clone / "artifacts").exists()
    assert rebuild(clone, check=True)["changed"] == []
    assert files(clone / "results") == files(root / "results")


def test_unknown_files_are_not_silently_removed(published):
    root, _, destination = published
    extra = destination / "unowned.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="unowned"):
        rebuild(root)
    assert extra.read_text() == "{}"


def test_integrity_cli_and_bad_path(published):
    root, _, destination = published
    for args in (["validate"], ["rebuild", "--check"]):
        subprocess.run([sys.executable, "-m", "lab.results", *args, "--root", str(root)],
                       check=True, capture_output=True, text=True)
    destination.rename(destination.with_name("wrong-key"))
    result = subprocess.run([sys.executable, "-m", "lab.results", "validate", "--root", str(root)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "CampaignKey" in result.stderr


def test_unpublished_checkout_has_no_synthetic_result(tmp_path):
    assert validate_results(tmp_path) == {"campaigns": 0}
    assert rebuild(tmp_path, check=True) == {"campaigns": 0, "changed": []}
    assert not (tmp_path / "results").exists()

@pytest.mark.parametrize("bad_field", ["parent_campaign", "campaign_id"])
def test_fork_provenance_cannot_claim_unrelated_parent_or_arbitrary_id(published, bad_field):
    import uuid

    root, _, destination = published
    trace = store.read(destination / "trace.json")
    trace["campaign_id"] = str(uuid.uuid4())
    trace["fork"] = dict(parent_campaign=destination.name, parent_iteration=1, reason="alternative")
    if bad_field == "parent_campaign":
        trace["fork"][bad_field] = "wrong-parent"
    else:
        trace[bad_field] = "arbitrary-id"
    child = destination / "forks" / trace["campaign_id"]
    store.write(child / "trace.json", trace)
    before = files(root / "results")
    with pytest.raises(ValueError, match="fork"):
        rebuild(root)
    assert files(root / "results") == before


def test_interrupted_offline_repair_can_be_retried(published, monkeypatch):
    import os

    root, _, destination = published
    expected = files(root / "results")
    (destination / "progress.svg").write_text("stale plot")
    (destination / "summary.json").write_text("{}")
    replace = os.replace
    def interrupt(source, target, **kwargs):
        if Path(target) == destination / "summary.json":
            raise OSError("interrupted repair")
        return replace(source, target, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted repair"):
            rebuild(root)
    assert (destination / "progress.svg").read_bytes() == expected[str((destination / "progress.svg").relative_to(root / "results"))]
    with pytest.raises(ValueError, match="stale"):
        validate_results(root)
    rebuild(root)
    assert files(root / "results") == expected

def test_repair_does_not_publish_index_before_local_views(published, monkeypatch):
    import os

    root, _, destination = published
    (destination / "progress.svg").write_text("stale plot")
    index_path = root / "results/index.json"
    index_path.write_text("stale index")
    replace = os.replace
    def interrupt(source, target, **kwargs):
        if Path(target) == destination / "progress.svg":
            raise OSError("interrupted first view")
        return replace(source, target, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted first view"):
            rebuild(root)
    assert index_path.read_text() == "stale index"
    rebuild(root)
    assert validate_results(root)["campaigns"] == 1
