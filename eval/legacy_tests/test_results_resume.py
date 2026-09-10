"""Fresh Git clones with compact receipts and CPU toy checkpoint transitions."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from eval.legacy_tests.test_weight_dependency import workspace, candidate, finalize
from eval.legacy_tests.test_context_transition import ADAPTER, request, next_spec
from eval.legacy_tests.test_results_publication import files
from lab.optimize import campaign, store
from lab.optimize.registry import CampaignRegistry
from lab.results import resume
from lab.results.publish import publish
from lab.results.rebuild import rebuild


def commit(root, message):
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", message], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def portable_command(mode):
    return dict(argv=["{python}", "adapter.py", mode, "{out}"], resource="cpu", timeout_s=10)


@pytest.fixture
def portable(workspace):
    root, old = workspace
    (root / "adapter.py").write_text(ADAPTER)
    (root / ".gitignore").write_text("artifacts/\nweights-*.json\njournal\nfail-*\n")
    revision = commit(root, "portable adapter")
    baseline = store.read(old / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    baseline["identity"]["engine_revision"] = revision
    baseline["correctness"]["identity"]["engine_revision"] = revision
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2")
    directory = CampaignRegistry(root).create(key, baseline=baseline, inputs=["src/kernel.cu", "adapter.py"])
    for iteration, dependency in enumerate(("rebuild", "invariant", "retune"), 1):
        (root / "src/kernel.cu").write_text(f"implementation {iteration}")
        commit(root, f"candidate {iteration}")
        spec = candidate(f"step{iteration}", dependency)
        spec["options"] = {"checkpoint": "weights-A.json", "openpi_config": "fixture-config"}
        spec["identity"]["plan"] = {"attention": f"route-{iteration}"}
        spec["inputs"].append("adapter.py")
        if dependency == "rebuild":
            spec["artifact_recipe"] = portable_command("rebuild")
        if dependency == "retune":
            spec["retune_recipe"] = portable_command("retune")
        campaign.start(root, spec, directory)
        finalize(directory, iteration, "no_benefit" if iteration == 2 else "accepted")
    store.write(directory / "hypotheses.json", {"unresolved": [{"mechanism": "next portable fusion"}]})
    return root, directory, key


def clone(root, destination, *, shallow=False):
    commit(root, "published continuation")
    source = root.as_uri() if shallow else str(root)
    command = ["git", "clone", "-q", "--no-local"]
    if shallow:
        command.append("--depth=1")
    subprocess.run([*command, source, str(destination)], check=True)
    assert not (destination / "artifacts").exists()
    return CampaignRegistry(destination)


@pytest.mark.parametrize("prior_transition", [False, True])
def test_fresh_clone_preserves_history_requires_new_anchor_and_continues(portable, tmp_path, prior_transition):
    root, directory, key = portable
    if prior_transition:
        campaign.transition_context(root, directory, request(root, directory))
    result = publish(root, directory)
    snapshot = store.read(Path(result["directory"]) / "resume.json")
    original_trace = store.read(Path(result["directory"]) / "trace.json")
    original_state = campaign.rebuild(directory)
    assert snapshot["next_iteration"] == 4 and snapshot["portable_incumbent"]["iteration"] == 3
    assert snapshot["portable_incumbent"]["plan"] == {"attention": "route-3"}
    assert "stages" not in json.dumps(snapshot["history"])
    clone_root = tmp_path / "new-layout"
    registry = clone(root, clone_root)
    shutil.rmtree(root)
    restored = registry.open_or_seed(key)
    state = campaign.rebuild(restored)
    assert state["reanchor_required"] and state["iterations"] == 4
    assert state["portable_incumbent"] == "iter-003"
    assert campaign.incumbent_record(restored, state)["spec"]["options"] == {
        "checkpoint": "weights-A.json", "openpi_config": "fixture-config"}
    assert campaign.implementation_identity(campaign.incumbent_record(restored, state))["plan"] == {"attention": "route-3"}
    assert state["failed_hypotheses"] == original_state["failed_hypotheses"]
    assert state["highest_value_unresolved_hypotheses"] == [{"mechanism": "next portable fusion"}]
    assert len(list((restored / "runs").glob("iter-*"))) == 4
    assert not (clone_root / "journal").exists()
    with pytest.raises(RuntimeError, match="re-anchor"):
        campaign.start(clone_root, next_spec(restored), restored)
    with pytest.raises(ValueError, match="re-anchor"):
        publish(clone_root, restored)
    incoming = request(clone_root, restored, latency=20.)
    incoming["measurement_context"]["weights"] = dict(checkpoint_id="new-task", checkpoint_digest="new-weights")
    incoming["require_recipes"] = True
    for mode in ("compatibility", "check", "measure"):
        incoming[mode] = portable_command(mode)
    state = campaign.transition_context(clone_root, restored, incoming)
    assert not state["reanchor_required"]
    assert state["current_measurement_segment"] == int(prior_transition) + 1
    assert (clone_root / "journal").read_text().splitlines() == ["compatibility", "rebuild", "retune", "check", "measure"]
    execution = Path(state["execution_repository"])
    spec = next_spec(restored)
    spec["inputs"].append("adapter.py")
    campaign.start(execution, spec, restored)
    finalize(restored, 4, "no_benefit", result=dict(validity="valid", candidate_ms=20., parent_incumbent_ms=20.))
    continued = publish(clone_root, restored)
    assert continued["summary"]["latest_iteration"] == 4
    value = store.read(Path(continued["directory"]) / "trace.json")
    assert value["iterations"][:4] == original_trace["iterations"]
    assert value["segments"][:len(original_trace["segments"])] == original_trace["segments"]
    assert rebuild(clone_root, check=True)["changed"] == []
    # A fresh process resumes the local ledger without rerunning any receipt.
    key_path = tmp_path / "key.json"
    store.write(key_path, key)
    receipt = subprocess.run([sys.executable, "-m", "lab.optimize", "campaign-open-or-seed", str(key_path),
                              "--root", str(clone_root)], check=True, capture_output=True, text=True)
    assert json.loads(receipt.stdout)["state"]["iterations"] == 5


def test_local_ledger_is_preferred_and_corruption_does_not_seed_over_it(portable):
    root, directory, key = portable
    result = publish(root, directory)
    (Path(result["directory"]) / "resume.json").write_text("bad published file")
    registry = CampaignRegistry(root)
    assert registry.open_or_seed(key) == directory
    (directory / "campaign.json").write_text("bad local file")
    with pytest.raises(json.JSONDecodeError):
        registry.open_or_seed(key)


def test_missing_engine_commit_does_not_create_or_silently_reset_lineage(portable, tmp_path):
    root, directory, key = portable
    publish(root, directory)
    clone_root = tmp_path / "shallow"
    registry = clone(root, clone_root, shallow=True)
    with pytest.raises(subprocess.CalledProcessError):
        registry.open_or_seed(key)
    assert not registry._path(key).exists()


def test_partial_seed_cannot_be_opened_as_a_complete_history(portable, tmp_path, monkeypatch):
    root, directory, key = portable
    publish(root, directory)
    registry = clone(root, tmp_path / "clone")
    write = store.write
    def interrupt(path, value):
        if Path(path) == registry._path(key) / "campaign.json":
            raise OSError("interrupted seed publication")
        return write(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(store, "write", interrupt)
        with pytest.raises(OSError, match="interrupted seed"):
            registry.open_or_seed(key)
    with pytest.raises(FileNotFoundError):
        registry.open_or_seed(key)
    assert not (registry._path(key) / "campaign.json").exists()


def test_snapshot_retains_actual_aba_and_rejects_corrupted_receipts(portable):
    root, directory, _ = portable
    result = publish(root, directory)
    path = Path(result["directory"])
    snapshot = store.read(path / "resume.json")
    original = campaign._records(directory)[1][1]["measurement"]["aba"]
    assert snapshot["history"]["records"][1]["measurement"]["aba"]["legs"] == original["legs"]
    snapshot["history"]["records"][1]["measurement"]["aba"]["legs"][0]["metrics"]["chunk_latency"]["min"] = 99.
    with pytest.raises(ValueError, match="spread|latency"):
        resume.check(snapshot, store.read(path / "trace.json"))


def test_dirty_portable_source_cannot_be_exported_as_committed(portable):
    root, directory, _ = portable
    (root / "src/kernel.cu").write_text("uncommitted fast change")
    campaign.start(root, candidate("dirty"), directory)
    finalize(directory, 4, "accepted")
    with pytest.raises(ValueError, match="commit and requalify"):
        publish(root, directory)


def test_missing_trace_and_stale_resume_state_are_rebuilt_from_retained_facts(portable):
    root, directory, _ = portable
    result = publish(root, directory)
    destination = Path(result["directory"])
    expected = files(root / "results")
    snapshot = store.read(destination / "resume.json")
    snapshot["next_iteration"] = 99
    store.write(destination / "resume.json", snapshot)
    (destination / "trace.json").unlink()
    with pytest.raises(ValueError, match="stale"):
        rebuild(root, check=True)
    rebuild(root)
    assert files(root / "results") == expected


def test_absent_local_and_published_campaign_can_create_validated_baseline(workspace):
    root, directory = workspace
    baseline = store.read(directory / "runs/iter-000-baseline/evidence.json")["measurement"]["evidence"]
    key = campaign.campaign_key(baseline["identity"], objective="e2e_chunk_latency_ms", protocol="latency-v2")
    path = CampaignRegistry(root).open_or_seed(key, baseline=baseline, inputs=["src/kernel.cu"])
    assert campaign.rebuild(path)["iterations"] == 1

def test_resume_without_index_or_trace_cannot_be_replaced_with_new_baseline(portable, tmp_path):
    root, directory, key = portable
    publish(root, directory)
    clone_root = tmp_path / "clone"
    registry = clone(root, clone_root)
    (clone_root / "results/index.json").unlink()
    next((clone_root / "results").rglob("trace.json")).unlink()
    baseline = campaign._records(directory)[0][1]["measurement"]["evidence"]
    with pytest.raises(ValueError, match="index is missing"):
        registry.open_or_seed(key, baseline=baseline, inputs=["src/kernel.cu"])
    assert not registry._path(key).exists()
    rebuild(clone_root)
    assert campaign.rebuild(registry.open_or_seed(key))["iterations"] == 4


def test_legacy_failed_hypotheses_survive_snapshot_and_seed(portable, tmp_path):
    root, directory, key = portable
    path, baseline = campaign._records(directory)[0]
    item = dict(source="old/candidates.jsonl", evidence_level="legacy_import", line=1,
                record=dict(id="old-rejected", thesis="fusion did not help", status="REJECTED"))
    baseline["measurement"]["evidence"]["legacy_history"] = {
        "accepted": [], "rejected": [item], "unclassified": [], "experiment_evidence": []}
    store.write(path, baseline)
    before = campaign.rebuild(directory)
    result = publish(root, directory)
    snapshot = store.read(Path(result["directory"]) / "resume.json")
    assert snapshot["failed_hypotheses"] == before["failed_hypotheses"]
    registry = clone(root, tmp_path / "clone")
    state = campaign.rebuild(registry.open_or_seed(key))
    assert state["failed_hypotheses"] == before["failed_hypotheses"]
    assert state["legacy_import"] == before["legacy_import"]


def test_checkpoint_specific_source_is_not_inherited_and_second_clone_still_reanchors(portable, tmp_path):
    root, directory, key = portable
    portable_source = (root / "src/kernel.cu").read_text()
    (root / "src/kernel.cu").write_text("checkpoint-specific source")
    commit(root, "context overlay")
    campaign.start(root, candidate("specific", "checkpoint_specific"), directory)
    finalize(directory, 4, "accepted")
    result = publish(root, directory)
    snapshot = store.read(Path(result["directory"]) / "resume.json")
    assert snapshot["portable_incumbent"]["iteration"] == 3
    clone_root = tmp_path / "clone"
    registry = clone(root, clone_root)
    restored = registry.open_or_seed(key)
    state = campaign.rebuild(restored)
    source = campaign.incumbent_record(restored, state)["source"]
    assert (Path(source["root"]) / "src/kernel.cu").read_text() == portable_source
    incoming = request(clone_root, restored)
    for mode in ("compatibility", "check", "measure"):
        incoming[mode] = portable_command(mode)
    state = campaign.transition_context(clone_root, restored, incoming)
    execution = Path(state["execution_repository"])
    assert (execution / "src/kernel.cu").read_text() == portable_source
    spec = next_spec(restored)
    spec["parent_plan"] = snapshot["portable_incumbent"]["plan"]
    spec["inputs"].append("adapter.py")
    campaign.start(execution, spec, restored)
    finalize(restored, 5, "no_benefit", result=dict(validity="valid", candidate_ms=20., parent_incumbent_ms=20.))
    publish(clone_root, restored)
    again = clone(clone_root, tmp_path / "second-clone")
    second = campaign.rebuild(again.open_or_seed(key))
    assert second["iterations"] == 6 and second["reanchor_required"]
    assert second["portable_incumbent"] == "iter-003"


def test_missing_resume_is_incomplete_publication(portable):
    from lab.results.rebuild import validate_results

    root, directory, key = portable
    result = publish(root, directory)
    (Path(result["directory"]) / "resume.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_results(root)


def test_corrupt_matching_key_index_location_does_not_start_new_lineage(portable, tmp_path):
    root, directory, key = portable
    publish(root, directory)
    clone_root = tmp_path / "clone"
    registry = clone(root, clone_root)
    index_path = clone_root / "results/index.json"
    index = store.read(index_path)
    index["campaigns"][0]["summary"] = "targets/arbitrary/summary.json"
    store.write(index_path, index)
    with pytest.raises(ValueError, match="noncanonical"):
        registry.open_or_seed(key)
    assert not registry._path(key).exists()
