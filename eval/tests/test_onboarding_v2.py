"""Onboarding semantics and actual Registry publication; CPU fixtures are not model acceptance."""
from copy import deepcopy
from pathlib import Path

import pytest

from eval.tests.test_target_onboarding import (
    spec as legacy_spec, official_reference, inventory, target_bring_up, ladder, floor_profile)
from eval.tests.test_weight_dependency import workspace
from flash_vla.runtime.identity import inference_signature
from lab import onboarding
from lab.optimize import campaign, store


CONTRACT = dict(architecture={"layers": 1}, parameter_shapes={"weight": [2, 2]},
                weight_layout="cpu-test", io_contract={"input": [1, 2], "output": [1, 2]},
                control_flow={"steps": 1})


@pytest.fixture
def initial(workspace, tmp_path):
    root, old = workspace
    baseline = deepcopy(campaign._records(old)[0][1]["measurement"]["evidence"])
    signature = inference_signature(**CONTRACT)
    baseline["identity"]["inference_signature"] = signature
    baseline["correctness"]["identity"]["inference_signature"] = signature
    reference = {"repository": "test-official-repository", "commit": "1" * 40}
    baseline["measurement_context"]["reference_provenance"] = reference
    baseline["correctness"]["measurement_context"]["reference_provenance"] = reference
    old_spec = legacy_spec()
    value = {key: old_spec[key] for key in (
        "correctness_requirements", "deployment_bound", "optimization_budget", "assumptions")}
    value["optimization_budget"] = store.read(old / "campaign.json")["budget"]
    value.update(version=2, target=campaign.target_key(baseline["identity"]),
                 initial_weights=baseline["measurement_context"]["weights"],
                 execution_variant=baseline["identity"]["execution_variant"], reference=reference,
                 deployment="CPU orchestration fixture; not a real H100 measurement",
                 performance_objective={"metric": "e2e_chunk_latency_ms", "direction": "minimize", "unit": "ms"},
                 benchmark_protocol={"id": "latency-v2"})
    return root, tmp_path / "onboarding", value, baseline


def prepare(directory, value):
    onboarding.create(directory, value)
    onboarding.record(directory, "reference_freeze", dict(status="passed", **value["reference"]))
    onboarding.record(directory, "model_contract", dict(status="passed", contract=deepcopy(CONTRACT)))
    onboarding.record(directory, "weights_compatibility", dict(
        status="passed", weights=value["initial_weights"], contract=deepcopy(CONTRACT)))
    onboarding.record(directory, "official_reference", official_reference())
    onboarding.write_compatibility_report(directory, inventory())
    onboarding.record(directory, "target_bring_up", target_bring_up())
    onboarding.record(directory, "correctness_ladder", ladder(onboarding.CORRECTNESS_LADDER))
    onboarding.record(directory, "baseline_ladder", ladder(onboarding.BASELINE_LADDER))
    onboarding.record(directory, "floor_profile", floor_profile())


def test_v2_separates_target_weights_and_reference(initial):
    _, directory, value, _ = initial
    state = onboarding.create(directory, value)
    assert state["current_stage"] == "reference_freeze"
    changed = deepcopy(value)
    changed["initial_weights"]["checkpoint_digest"] = "new-fine-tuned-values"
    changed["reference"]["commit"] = "2" * 40
    assert onboarding.validate_spec(changed)["target"] == value["target"]


def test_incompatible_weight_contract_blocks_old_target(initial):
    _, directory, value, _ = initial
    onboarding.create(directory, value)
    onboarding.record(directory, "reference_freeze", dict(status="passed", **value["reference"]))
    onboarding.record(directory, "model_contract", dict(status="passed", contract=CONTRACT))
    incompatible = deepcopy(CONTRACT)
    incompatible["parameter_shapes"]["weight"] = [3, 2]
    with pytest.raises(ValueError, match="inference signature"):
        onboarding.record(directory, "weights_compatibility", dict(
            status="passed", weights=value["initial_weights"], contract=incompatible))
    assert onboarding.rebuild(directory)["current_stage"] == "weights_compatibility"


def test_textual_campaign_reference_cannot_complete_handoff(initial):
    _, directory, value, _ = initial
    prepare(directory, value)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        onboarding.record(directory, "campaign_creation", dict(
            status="passed", directory="does-not-exist", objective="e2e_chunk_latency_ms",
            protocol="latency-v2", fixture="pi05-perf-v1", state="BASELINED"))
    assert onboarding.rebuild(directory)["current_stage"] == "campaign_creation"


def test_real_registry_and_publication_are_required_for_ready(initial):
    root, directory, value, baseline = initial
    prepare(directory, value)
    state = onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    assert state["status"] == "READY_FOR_OPTIMIZATION"
    creation = store.read(directory / "stages/campaign_creation/attempt-001.json")
    ledger = Path(creation["directory"])
    assert campaign.rebuild(ledger)["iterations"] == 1
    assert store.read(root / "results/index.json")["campaigns"][0]["lineage_id"] == ledger.name
    (directory / "state.json").unlink()
    assert onboarding.validate(directory)["status"] == "READY_FOR_OPTIMIZATION"
    publication = next((root / "results/targets").glob("*/summary.json"))
    publication.unlink()
    with pytest.raises(FileNotFoundError):
        onboarding.validate(directory)


@pytest.mark.parametrize("failure", ["render", "index"])
def test_handoff_failure_retains_campaign_and_retries_publication_only(initial, monkeypatch, failure):
    from lab.results import render, index
    from lab.optimize import runner

    root, directory, value, baseline = initial
    prepare(directory, value)
    def fail(*args, **kwargs):
        raise OSError("injected onboarding publication failure")
    with monkeypatch.context() as patch:
        patch.setattr(render if failure == "render" else index, "write" if failure == "render" else "rebuild", fail)
        with pytest.raises(OSError, match="injected"):
            onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    state = onboarding.rebuild(directory)
    assert state["current_stage"] == "publication" and state["status"] != "READY_FOR_OPTIMIZATION"
    receipt = store.read(directory / "stages/campaign_creation/attempt-001.json")
    def no_experiment(*args, **kwargs):
        raise AssertionError("handoff retry cannot rerun any experiment")
    monkeypatch.setattr(runner, "run", no_experiment)
    monkeypatch.setattr(runner, "run_stage", no_experiment)
    state = onboarding.handoff(directory, root, baseline)
    assert state["status"] == "READY_FOR_OPTIMIZATION"
    assert campaign.rebuild(Path(receipt["directory"]))["iterations"] == 1
    assert state["attempts"]["campaign_creation"] == 1


def test_repeated_onboarding_reuses_one_campaign_and_published_anchor(initial):
    root, directory, value, baseline = initial
    prepare(directory, value)
    onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    other = directory.parent / "second-onboarding"
    prepare(other, value)
    onboarding.handoff(other, root, baseline)
    first = store.read(directory / "stages/campaign_creation/attempt-001.json")
    second = store.read(other / "stages/campaign_creation/attempt-001.json")
    assert first["campaign_id"] == second["campaign_id"]
    assert campaign.rebuild(Path(first["directory"]))["iterations"] == 1


def test_baseline_mismatch_cannot_reserve_campaign(initial):
    root, directory, value, baseline = initial
    prepare(directory, value)
    baseline["measurement_context"]["weights"]["checkpoint_digest"] = "different"
    with pytest.raises(ValueError, match="checkpoint"):
        onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    assert not (root / "artifacts/optimization/campaigns").exists()


def test_legacy_spec_is_not_used_for_new_onboarding(tmp_path):
    with pytest.raises(ValueError, match="new onboarding requires v2"):
        onboarding.create(tmp_path / "new", legacy_spec())


@pytest.mark.parametrize("change", ["checkpoint", "reference", "fixture", "environment"])
def test_existing_campaign_transitions_without_new_lineage_or_iteration(initial, change):
    from eval.tests.test_context_transition import request, journal

    root, directory, value, baseline = initial
    prepare(directory, value)
    onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    receipt = store.read(directory / "stages/campaign_creation/attempt-001.json")
    ledger = Path(receipt["directory"])
    incoming = request(root, ledger)
    incoming["measurement_context"] = deepcopy(baseline["measurement_context"])
    if change == "checkpoint":
        incoming["measurement_context"]["weights"] = {"checkpoint_id": "task-b", "checkpoint_digest": "manifest-b"}
    elif change == "reference":
        incoming["measurement_context"]["reference_provenance"]["commit"] = "2" * 40
    elif change == "fixture":
        incoming["measurement_context"]["fixture"] = {"id": "fixture-b", "digest": "fixture-b"}
    else:
        incoming["measurement_context"]["environment"]["driver"] = "new-driver"
    updated = deepcopy(value)
    updated["initial_weights"] = incoming["measurement_context"]["weights"]
    updated["reference"] = incoming["measurement_context"]["reference_provenance"]
    new_baseline = deepcopy(baseline)
    new_baseline["measurement_context"] = incoming["measurement_context"]
    next_directory = directory.parent / "next-onboarding"
    prepare(next_directory, updated)
    with pytest.raises(ValueError, match="needs context validation"):
        onboarding.handoff(next_directory, root, new_baseline)
    assert campaign.rebuild(ledger)["current_measurement_segment"] == 0
    assert not (root / "journal").exists()
    state = onboarding.handoff(next_directory, root, new_baseline, transition_request=incoming)
    assert state["status"] == "READY_FOR_OPTIMIZATION"
    after = store.read(next_directory / "stages/campaign_creation/attempt-001.json")
    assert after["campaign_id"] == receipt["campaign_id"] and after["segment"] == 1
    assert campaign.rebuild(ledger)["iterations"] == 1
    assert journal(root) == ["compatibility", "check", "measure"]
    # Original onboarding retains its validated historical handoff segment.
    assert onboarding.validate(directory)["status"] == "READY_FOR_OPTIMIZATION"


def test_duplicate_cli_handoff_uses_one_campaign(initial, tmp_path):
    import json
    import subprocess
    import sys

    root, directory, value, baseline = initial
    prepare(directory, value)
    baseline_path = tmp_path / "baseline.json"
    store.write(baseline_path, baseline)
    command = [sys.executable, "-m", "lab.onboarding", "handoff", str(directory),
               str(baseline_path), "--root", str(root), "--source-input", "src/kernel.cu"]
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for _ in range(2)]
    outputs = [worker.communicate(timeout=45) for worker in workers]
    assert [worker.returncode for worker in workers] == [0, 0], outputs
    assert all(json.loads(stdout)["status"] == "READY_FOR_OPTIMIZATION" for stdout, _ in outputs)
    assert len(list((root / "artifacts/optimization/campaigns").glob("*/campaign.json"))) == 1
    assert onboarding.rebuild(directory)["attempts"]["publication"] == 1


@pytest.mark.parametrize("field", ["checkpoint_id", "checkpoint_digest"])
def test_weight_compatibility_receipt_binds_initial_asset(initial, field):
    _, directory, value, _ = initial
    onboarding.create(directory, value)
    onboarding.record(directory, "reference_freeze", dict(status="passed", **value["reference"]))
    onboarding.record(directory, "model_contract", dict(status="passed", contract=CONTRACT))
    weights = dict(value["initial_weights"], **{field: "wrong"})
    with pytest.raises(ValueError, match="different initial weights"):
        onboarding.record(directory, "weights_compatibility", dict(
            status="passed", contract=CONTRACT, weights=weights))


def test_unregistered_budget_cannot_silently_be_replaced(initial):
    root, directory, value, baseline = initial
    value["optimization_budget"]["candidates"] += 1
    prepare(directory, value)
    with pytest.raises(ValueError, match="registered Target budget"):
        onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    assert not (root / "artifacts/optimization/campaigns").exists()


@pytest.fixture
def published_clone(initial, tmp_path):
    from eval.tests.test_results_resume import clone
    from eval.tests.test_weight_dependency import candidate

    root, directory, value, baseline = initial
    prepare(directory, value)
    onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    receipt = store.read(directory / "stages/campaign_creation/attempt-001.json")
    ledger = Path(receipt["directory"])
    trial = candidate("retained-failure")
    trial["identity"] = baseline["identity"]
    trial["measurement_context"] = baseline["measurement_context"]
    campaign.start(root, trial, ledger)
    campaign.finalize(ledger, 1, "invalid", {"status": "not_run"},
                      {"validity": "incomplete"}, {"status": "not_run"})
    (root / ".gitignore").write_text("artifacts/\n")
    clone_root = tmp_path / "fresh-clone"
    clone(root, clone_root)
    new_directory = tmp_path / "fresh-onboarding"
    prepare(new_directory, value)
    return clone_root, new_directory, value, baseline


def test_fresh_clone_handoff_seeds_history_and_requires_a_new_anchor(published_clone):
    from eval.tests.test_context_transition import request, journal
    from lab.optimize.registry import CampaignRegistry

    root, directory, value, baseline = published_clone
    with pytest.raises(ValueError, match="needs context validation"):
        onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    ledger = CampaignRegistry(root).find(onboarding._campaign_key(value))
    current = campaign.rebuild(ledger)
    assert current["iterations"] == 2 and current["reanchor_required"]
    assert current["failed_hypotheses"][0]["candidate_id"] == "retained-failure"
    incoming = request(root, ledger)
    incoming["measurement_context"] = deepcopy(baseline["measurement_context"])
    result = onboarding.handoff(directory, root, baseline, transition_request=incoming)
    assert result["status"] == "READY_FOR_OPTIMIZATION"
    current = campaign.rebuild(ledger)
    assert current["iterations"] == 2 and current["current_measurement_segment"] == 1
    assert journal(root) == ["compatibility", "check", "measure"]


def test_manual_handoff_receipts_cannot_reuse_unvalidated_imported_segment(published_clone):
    from lab.optimize.registry import CampaignRegistry

    root, directory, value, _ = published_clone
    ledger = CampaignRegistry(root).open_or_seed(onboarding._campaign_key(value))
    receipt = dict(status="passed", repository=str(root), directory=str(ledger),
                   campaign_id=ledger.name, segment=0)
    with pytest.raises(ValueError, match="current validated anchor"):
        onboarding.record(directory, "campaign_creation", receipt)
    assert onboarding.rebuild(directory)["current_stage"] == "campaign_creation"
    assert campaign.rebuild(ledger)["reanchor_required"]


def test_new_publication_receipt_cannot_accept_stale_same_segment_history(initial, monkeypatch):
    from eval.tests.test_weight_dependency import candidate
    from lab.results import render

    root, directory, value, baseline = initial
    prepare(directory, value)
    onboarding.handoff(directory, root, baseline, inputs=["src/kernel.cu"])
    receipt = store.read(directory / "stages/campaign_creation/attempt-001.json")
    ledger = Path(receipt["directory"])
    trial = candidate("pending-publication")
    trial["identity"] = baseline["identity"]
    trial["measurement_context"] = baseline["measurement_context"]
    campaign.start(root, trial, ledger)
    def fail(*args, **kwargs):
        raise OSError("injected pending publication")
    with monkeypatch.context() as patch:
        patch.setattr(render, "write", fail)
        with pytest.raises(OSError, match="pending publication"):
            campaign.finalize(ledger, 1, "invalid", {"status": "not_run"},
                              {"validity": "incomplete"}, {"status": "not_run"})
    fresh = directory.parent / "fresh-handoff"
    prepare(fresh, value)
    onboarding.record(fresh, "campaign_creation", receipt)
    with pytest.raises(ValueError, match="publication is pending"):
        onboarding.record(fresh, "publication", receipt)
    assert onboarding.rebuild(fresh)["current_stage"] == "publication"
    # The previously completed handoff remains valid for its historical anchor.
    assert onboarding.validate(directory)["status"] == "READY_FOR_OPTIMIZATION"
    assert onboarding.handoff(fresh, root, baseline)["status"] == "READY_FOR_OPTIMIZATION"
