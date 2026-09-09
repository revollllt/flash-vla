"""Terminal verdict and publication recovery are separate durable operations."""
from pathlib import Path
import subprocess
import sys

import pytest

from eval.tests.test_weight_dependency import workspace, candidate, finalize
from eval.tests.test_context_transition import request
from eval.tests.test_results_resume import portable
from lab.optimize import campaign, store, runner
from lab.results import index, render
from lab.results.rebuild import rebuild


@pytest.fixture
def automatic(workspace):
    root, directory = workspace
    state = campaign.configure_publication(directory, root)
    assert state["publication_required"] and state["current_stage"] == "publication"
    campaign.resume(directory)
    assert not campaign.rebuild(directory)["publication_required"]
    return root, directory


def target_directory(root):
    return next((root / "results/targets").iterdir())


@pytest.mark.parametrize("verdict", ["accepted", "no_benefit", "correctness_failed", "invalid"])
def test_finalize_updates_results_and_snapshot_without_manual_publish(automatic, verdict):
    root, directory = automatic
    campaign.start(root, candidate(verdict), directory)
    if verdict in ("accepted", "no_benefit"):
        state = finalize(directory, 1, verdict)
    else:
        state = campaign.finalize(
            directory, 1, verdict,
            {"status": "failed" if verdict == "correctness_failed" else "not_run"},
            {"validity": "incomplete"}, {"status": "not_run"})
    assert not state["publication_required"]
    destination = target_directory(root)
    value = store.read(destination / "trace.json")
    assert value["iterations"][-1]["verdict"] == verdict
    assert store.read(destination / "resume.json")["next_iteration"] == 2
    assert store.read(destination / "summary.json")["current_performance"]["best_ms"] == (
        14. if verdict == "accepted" else 16.)
    assert rebuild(root, check=True)["changed"] == []


@pytest.mark.parametrize("failure", ["render", "index", "receipt"])
def test_publication_failure_keeps_verdict_and_resume_does_not_rerun_experiment(automatic, monkeypatch, failure):
    root, directory = automatic
    campaign.start(root, candidate("completed"), directory)
    def fail(*args, **kwargs):
        raise OSError("injected publication failure")
    write = store.write
    def fail_receipt(path, value):
        if Path(path) == directory / "publication.json":
            fail()
        return write(path, value)
    with monkeypatch.context() as patch:
        if failure == "render":
            patch.setattr(render, "write", fail)
        elif failure == "index":
            patch.setattr(index, "rebuild", fail)
        else:
            patch.setattr(store, "write", fail_receipt)
        with pytest.raises(OSError, match="injected publication"):
            finalize(directory, 1, "no_benefit")
    before = campaign.rebuild(directory)
    assert before["publication_required"] and before["current_stage"] == "publication"
    record = campaign._records(directory)[1][1]
    assert record["verdict"] == "no_benefit"
    with pytest.raises(RuntimeError, match="active"):
        campaign.start(root, candidate("next"), directory)
    with pytest.raises(ValueError, match="immutable"):
        finalize(directory, 1, "no_benefit")
    def no_experiment(*args, **kwargs):
        raise AssertionError("publication recovery must not run any experiment stage")
    monkeypatch.setattr(runner, "run", no_experiment)
    monkeypatch.setattr(runner, "run_stage", no_experiment)
    state = campaign.resume(directory)
    assert not state["publication_required"] and state["iterations"] == 2
    assert state["cost"] == before["cost"]
    assert len(campaign._records(directory)) == 2
    assert store.read(target_directory(root) / "summary.json")["latest_iteration"] == 1


def test_context_transition_publishes_new_segment_without_consuming_iteration(automatic):
    root, directory = automatic
    state = campaign.transition_context(root, directory, request(root, directory))
    assert not state["publication_required"]
    value = store.read(target_directory(root) / "trace.json")
    assert len(value["iterations"]) == 1
    assert len(value["segments"]) == 2
    assert store.read(directory / "publication.json") == {"iteration": 0, "segment": 1}


def test_cli_finalization_configures_automatic_publication(workspace, tmp_path):
    import json

    root, directory = workspace
    campaign.start(root, candidate("failure"), directory)
    outcome = tmp_path / "outcome.json"
    store.write(outcome, dict(verdict="correctness_failed", correctness={"status": "failed"},
                             measurement={"validity": "incomplete"}, qualification={"status": "not_run"}))
    result = subprocess.run([sys.executable, "-m", "lab.optimize", "campaign-finalize", str(directory),
                             "--root", str(root), "--iteration", "1", "--result", str(outcome)],
                            check=True, capture_output=True, text=True)
    assert not json.loads(result.stdout)["publication_required"]
    assert store.read(target_directory(root) / "trace.json")["iterations"][-1]["verdict"] == "correctness_failed"


def test_configured_repository_cannot_be_redirected(automatic, tmp_path):
    root, directory = automatic
    with pytest.raises(ValueError, match="publication repository"):
        campaign.configure_publication(directory, tmp_path / "different")
    from lab.results.publish import publish
    with pytest.raises(ValueError, match="publication repository"):
        publish(tmp_path / "different", directory)
    assert not campaign.rebuild(directory)["publication_required"]


@pytest.mark.parametrize("command", ["campaign-create", "campaign-open-or-create", "campaign-open-or-seed"])
def test_cli_creation_publishes_baseline_and_reopens_same_lineage(workspace, tmp_path, command):
    import json

    root, old = workspace
    baseline = campaign._records(old)[0][1]["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms",
                               protocol="latency-v2")
    baseline_file, key_file = tmp_path / "baseline.json", tmp_path / "key.json"
    store.write(baseline_file, baseline)
    store.write(key_file, key)
    args = [sys.executable, "-m", "lab.optimize", command]
    if command == "campaign-create":
        args += ["--objective", "e2e_chunk_latency_ms", "--protocol", "latency-v2",
                 "--fixture", baseline["measurement_context"]["fixture"]["id"]]
    else:
        args += [str(key_file)]
    result = subprocess.run(args + ["--root", str(root), "--baseline-evidence", str(baseline_file),
                                    "--source-input", "src/kernel.cu"],
                            check=True, capture_output=True, text=True)
    created = json.loads(result.stdout)
    assert not created["state"]["publication_required"]
    destination = target_directory(root)
    snapshot = store.read(destination / "resume.json")
    assert "publication_root" not in snapshot["history"]["metadata"]
    assert snapshot["next_iteration"] == 1
    result = subprocess.run([sys.executable, "-m", "lab.optimize", "campaign-open-or-create",
                             str(key_file), "--root", str(root), "--baseline-evidence", str(baseline_file)],
                            check=True, capture_output=True, text=True)
    reopened = json.loads(result.stdout)
    assert reopened["directory"] == created["directory"]
    assert reopened["state"]["iterations"] == 1


def test_dirty_source_cannot_become_an_unpublishable_accepted_verdict(automatic):
    root, directory = automatic
    (root / "src/kernel.cu").write_text("changed implementation")
    campaign.start(root, candidate("dirty"), directory)
    with pytest.raises(ValueError, match="differs from committed engine"):
        finalize(directory, 1, "accepted")
    assert campaign._records(directory)[1][1].get("verdict") is None
    assert campaign.rebuild(directory)["current_stage"] != "publication"
    campaign.finalize(directory, 1, "invalid", {"status": "not_run"},
                      {"validity": "incomplete"}, {"status": "not_run"})
    subprocess.run(["git", "-C", str(root), "add", "src/kernel.cu"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "-qm", "candidate"], check=True)
    campaign.start(root, candidate("committed"), directory)
    state = finalize(directory, 2, "accepted")
    assert not state["publication_required"]
    assert state["portable_incumbent"] == "iter-002"
    assert [row["verdict"] for row in store.read(target_directory(root) / "trace.json")["iterations"][1:]] == [
        "invalid", "accepted"]


def test_missing_source_is_rejected_before_binding_publication(workspace, tmp_path):
    root, original = workspace
    baseline = campaign._records(original)[0][1]["measurement"]["evidence"]
    directory = tmp_path / "without-source"
    campaign.create(directory, baseline, "e2e_chunk_latency_ms", "latency-v2", "pi05-perf-v1")
    with pytest.raises(ValueError, match="declared portable source inputs"):
        campaign.configure_publication(directory, root)
    assert "publication_root" not in store.read(directory / "campaign.json")
    assert campaign.rebuild(directory)["current_stage"] == "candidate_selection"


@pytest.mark.parametrize("command", ["campaign-create", "campaign-open-or-create", "campaign-open-or-seed"])
def test_cli_missing_baseline_source_leaves_no_reserved_campaign(workspace, tmp_path, command):
    root, directory = workspace
    baseline = campaign._records(directory)[0][1]["measurement"]["evidence"]
    baseline_file = tmp_path / "missing-source-baseline.json"
    store.write(baseline_file, baseline)
    args = [sys.executable, "-m", "lab.optimize", command]
    if command != "campaign-create":
        key_file = tmp_path / "key.json"
        store.write(key_file, campaign.campaign_key(
            baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2"))
        args += [str(key_file)]
    args += ["--root", str(root), "--baseline-evidence", str(baseline_file),
             "--objective", "e2e_chunk_latency_ms", "--protocol", "latency-v2",
             "--fixture", "pi05-perf-v1"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and "declared portable source inputs" in result.stderr
    assert not list((root / "artifacts/optimization/campaigns").glob("*/campaign.json"))
    subprocess.run(args + ["--source-input", "src/kernel.cu"],
                   check=True, capture_output=True, text=True)
    assert store.read(target_directory(root) / "summary.json")["latest_iteration"] == 0


def test_cli_finalization_in_execution_checkout_preserves_publication_root(automatic, tmp_path):
    import json

    root, directory = automatic
    checkout = tmp_path / "execution"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(checkout), "HEAD"],
                   check=True, capture_output=True)
    campaign.start(checkout, candidate("execution-failure"), directory)
    outcome = tmp_path / "execution-outcome.json"
    store.write(outcome, dict(verdict="correctness_failed", correctness={"status": "failed"},
                             measurement={"validity": "incomplete"}, qualification={"status": "not_run"}))
    result = subprocess.run([sys.executable, "-m", "lab.optimize", "campaign-finalize", str(directory),
                             "--root", str(checkout), "--iteration", "1", "--result", str(outcome)],
                            check=True, capture_output=True, text=True)
    assert not json.loads(result.stdout)["publication_required"]
    assert store.read(target_directory(root) / "summary.json")["latest_iteration"] == 1
    assert not (checkout / "results").exists()


def test_cli_seed_prefers_snapshot_over_unusable_new_baseline(portable, tmp_path):
    import json
    from lab.optimize.registry import CampaignRegistry
    from eval.tests.test_results_resume import clone

    root, directory, key = portable
    campaign.configure_publication(directory, root)
    campaign.resume(directory)
    baseline = campaign._records(directory)[0][1]["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms",
                               protocol="latency-v2")
    clone_root = tmp_path / "published-clone"
    clone(root, clone_root)
    key_file, baseline_file = tmp_path / "key.json", tmp_path / "unused-baseline.json"
    store.write(key_file, key)
    store.write(baseline_file, baseline)
    result = subprocess.run([sys.executable, "-m", "lab.optimize", "campaign-open-or-seed",
                             str(key_file), "--root", str(clone_root),
                             "--baseline-evidence", str(baseline_file)],
                            check=True, capture_output=True, text=True)
    seeded = json.loads(result.stdout)
    assert seeded["state"]["reanchor_required"]
    assert seeded["state"]["iterations"] == 4
    assert Path(seeded["directory"]) == CampaignRegistry(clone_root).find(key)
    assert "published_import" in store.read(Path(seeded["directory"]) / "campaign.json")
