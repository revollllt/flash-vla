"""Portable optimization lineage; CPU tests do not establish checkpoint transfer."""
from pathlib import Path
import subprocess
import sys

import pytest

from lab.optimize import campaign, schema, store, trace, transition
from eval import acceptance
from eval.legacy_tests.test_campaign import experiment, PASS, VALID, QUALIFIED
from eval.tests.test_identity_v3 import payload, context as identity_context
from eval.legacy_tests.test_optimization_trace import SEGMENT0


def context():
    value = identity_context()
    value["fixture"]["id"] = "pi05-perf-v1"
    return value


def finalize(directory, iteration, verdict, correctness=PASS, result=VALID, qualification=QUALIFIED):
    record = store.read(next((directory / "runs").glob(f"iter-{iteration:03d}-*/evidence.json")))
    spec = record["spec"]
    identity = campaign.implementation_identity(record)
    checked = dict(correctness, identity=identity, measurement_context=spec["measurement_context"])
    measured = dict(identity=identity, measurement_context=spec["measurement_context"],
                    measurement_segment=dict(id=spec["measurement_segment"], benchmark_protocol="latency-v2",
                                             **spec["measurement_context"]["environment"]),
                    protocol="latency-v2", instrumented=False, candidate_ms=14.0,
                    parent_incumbent_ms=16.0)
    measured.update(result)
    parent = campaign.implementation_identity(
        campaign.incumbent_record(directory, campaign.rebuild(directory)))
    if measured.get("validity") == "valid" and "aba" not in measured:
        policy = acceptance.for_target(identity["target"])["latency"]
        legs = [dict(identity=who, measurement_context=spec["measurement_context"],
                     metrics={"chunk_latency": {"min": value}})
                for who, value in ((parent, measured["parent_incumbent_ms"]),
                                   (identity, measured["candidate_ms"]),
                                   (parent, measured["parent_incumbent_ms"]))]
        measured["aba"] = dict(legs=legs, config={key: policy[key] for key in ("reps", "warmup")})
    return campaign.finalize(directory, iteration, verdict, checked, measured, qualification)


def recipe():
    return dict(argv=[sys.executable, "-c", "print('rebuilt')"], resource="cpu", timeout_s=10)


def candidate(name="candidate", dependency="invariant"):
    spec = experiment(name, identity=payload())
    spec["applicability"] = {"weight_dependency": dependency}
    spec["measurement_context"] = context()
    spec["measurement_segment"] = 0
    if dependency == "rebuild":
        spec["artifact_recipe"] = recipe()
    elif dependency == "retune":
        spec["retune_recipe"] = recipe()
    return spec


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "--allow-empty",
                    "-qm", "baseline"], check=True)
    (root / "src").mkdir()
    (root / "src/kernel.cu").write_text("original")
    subprocess.run(["git", "-C", str(root), "add", "src/kernel.cu"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "-qm", "source"], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    identity = payload(engine_revision=revision)
    directory = tmp_path / "campaign"
    baseline = dict(identity=identity, measurement_context=context(), measurement_segment=SEGMENT0,
                    protocol="latency-v2", validity="valid", instrumented=False,
                    correctness=dict(status="pass", identity=identity, measurement_context=context()),
                    objective=dict(name="e2e_chunk_latency_ms", unit="ms", value=16.0))
    campaign.create(directory, baseline, "e2e_chunk_latency_ms",
                    "latency-v2", "pi05-perf-v1", root=root, inputs=["src/kernel.cu"])
    return root, directory


@pytest.mark.parametrize("dependency", schema.WEIGHT_DEPENDENCIES)
def test_declared_dependencies_validate(dependency):
    schema.validate(candidate(dependency=dependency))


@pytest.mark.parametrize("dependency", [None, "", "portable"])
def test_undeclared_or_unknown_dependency_cannot_allocate(workspace, dependency):
    root, directory = workspace
    spec = candidate(dependency=dependency)
    if dependency is None:
        del spec["applicability"]
    with pytest.raises(ValueError, match="applicability.weight_dependency"):
        campaign.start(root, spec, directory)
    assert len(list((directory / "runs").glob("iter-*"))) == 1


@pytest.mark.parametrize("dependency,field", [
    ("rebuild", "artifact_recipe"), ("retune", "retune_recipe"),
])
@pytest.mark.parametrize("bad_recipe", [None, "manual", {}, {"argv": "echo rebuilt"}])
def test_weight_dependent_candidates_need_executable_recipe(dependency, field, bad_recipe):
    spec = candidate(dependency=dependency)
    spec[field] = bad_recipe
    with pytest.raises(ValueError, match=field):
        schema.validate(spec)


def test_checkpoint_specific_result_is_retained_without_portable_promotion(workspace):
    root, directory = workspace
    campaign.start(root, candidate("fusion"), directory)
    finalize(directory, 1, "accepted", PASS, VALID, QUALIFIED)
    campaign.start(root, candidate("special_weights", "checkpoint_specific"), directory)
    state = finalize(directory, 2, "accepted", PASS, VALID, QUALIFIED)
    assert state["portable_incumbent"] == state["current_incumbent"] == "iter-001"
    result = store.read(directory / "runs/iter-002-special_weights/evidence.json")
    assert result["verdict"] == "accepted"
    assert result["spec"]["applicability"]["weight_dependency"] == "checkpoint_specific"
    receipt = campaign.materialize_incumbent(root, directory)
    next_spec = candidate("next")
    next_spec["parent_plan"] = receipt["identity"]["plan"]
    next_record = campaign.start(root, next_spec, directory)
    assert next_record["iteration"] == 3
    assert next_record["parent_incumbent"] == "iter-001"
    assert [r["iteration"] for r in campaign.portable_optimizations(directory)] == [1]
    (directory / "state.json").unlink()
    assert campaign.rebuild(directory)["portable_incumbent"] == "iter-001"


def test_transfer_preserves_recipes_of_all_accepted_ancestors(workspace):
    root, directory = workspace
    for iteration, dependency in enumerate(("invariant", "rebuild", "retune"), 1):
        campaign.start(root, candidate(f"step{iteration}", dependency), directory)
        finalize(directory, iteration, "accepted", PASS, VALID, QUALIFIED)
    campaign.start(root, candidate("rejected", "rebuild"), directory)
    finalize(directory, 4, "no_benefit", PASS, VALID, QUALIFIED)
    lineage = campaign.portable_optimizations(directory)
    assert [r["iteration"] for r in lineage] == [1, 2, 3]
    assert lineage[1]["spec"]["artifact_recipe"] == recipe()
    assert lineage[2]["spec"]["retune_recipe"] == recipe()
    assert campaign.rebuild(directory)["portable_incumbent"] == "iter-003"


def test_corrupted_dependency_is_rejected_during_recovery(workspace):
    root, directory = workspace
    campaign.start(root, candidate(), directory)
    path = directory / "runs/iter-001-candidate/evidence.json"
    record = store.read(path)
    del record["spec"]["applicability"]
    store.write(path, record)
    with pytest.raises(ValueError, match="applicability.weight_dependency"):
        campaign.rebuild(directory)


def test_failed_correctness_never_promotes_checkpoint_specific_candidate(workspace):
    root, directory = workspace
    campaign.start(root, candidate(dependency="checkpoint_specific"), directory)
    with pytest.raises(ValueError, match="accepted requires"):
        finalize(directory, 1, "accepted", {"status": "failed"}, VALID, QUALIFIED)
    assert campaign.rebuild(directory)["portable_incumbent"] == "iter-000"


def test_source_and_plan_are_restored_before_next_candidate(workspace):
    root, directory = workspace
    source = root / "src/kernel.cu"
    source.write_text("portable fusion")
    spec = candidate("fusion")
    spec["identity"]["plan"] = {"attention": "fusion"}
    campaign.start(root, spec, directory)
    finalize(directory, 1, "accepted", PASS, VALID, QUALIFIED)
    source.write_text("special checkpoint constants")
    special = candidate("special", "checkpoint_specific")
    special["identity"]["plan"] = {"attention": "special"}
    campaign.start(root, special, directory)
    finalize(directory, 2, "accepted", PASS, VALID, QUALIFIED)
    with pytest.raises(RuntimeError, match="materialize"):
        campaign.start(root, candidate("next"), directory)
    assert source.read_text() == "special checkpoint constants"
    receipt = campaign.materialize_incumbent(root, directory)
    assert source.read_text() == "portable fusion"
    assert receipt["identity"]["plan"] == {"attention": "fusion"}
    next_spec = candidate("next")
    next_spec["parent_plan"] = special["identity"]["plan"]
    with pytest.raises(ValueError, match="parent_plan"):
        campaign.start(root, next_spec, directory)
    source.write_text("portable fusion plus next change")
    next_spec["parent_plan"] = receipt["identity"]["plan"]
    next_record = campaign.start(root, next_spec, directory)
    assert (Path(next_record["source"]["root"]) / "src/kernel.cu").read_text() == source.read_text()
    assert next_record["parent_materialization"] == receipt


def test_materialization_preserves_unrecorded_edits(workspace):
    root, directory = workspace
    source = root / "src/kernel.cu"
    source.write_text("special checkpoint constants")
    campaign.start(root, candidate("special", "checkpoint_specific"), directory)
    finalize(directory, 1, "accepted", PASS, VALID, QUALIFIED)
    source.write_text("unrelated user edit")
    with pytest.raises(ValueError, match="unrecorded edit"):
        campaign.materialize_incumbent(root, directory)
    assert source.read_text() == "unrelated user edit"
    assert not list((directory / "materializations").glob("*.json"))


def test_trace_keeps_context_only_result_without_moving_portable_curve(workspace):
    root, directory = workspace
    for iteration, dependency, value in [(1, "invariant", 14.0),
                                          (2, "checkpoint_specific", 12.0)]:
        record = campaign.start(root, candidate(f"step{iteration}", dependency), directory)
        identity = dict(record["campaign_identity"],
                        engine_revision=record["change"]["engine_revision"])
        measurement = dict(validity="valid", identity=identity,
                           measurement_context=context(), measurement_segment=dict(id=0, benchmark_protocol="latency-v2",
                                                       **context()["environment"]),
                           candidate_ms=value, parent_incumbent_ms=16.0 if iteration == 1 else 14.0)
        finalize(directory, iteration, "accepted", PASS, measurement, QUALIFIED)
    entries = trace.normalize(directory)["iterations"]
    assert [r["current_incumbent_latency_ms"] for r in entries] == [16.0, 14.0, 14.0]
    assert entries[2]["candidate_latency_ms"] == 12.0
    assert entries[2]["promotion"] == "context_only"
    assert entries[2]["applicability"] == {"weight_dependency": "checkpoint_specific"}


def test_v3_refuses_legacy_reanchor_before_mutating_ledger(workspace):
    root, directory = workspace
    evidence = dict(identity=payload(), measurement_context=context(),
                    measurement_segment=SEGMENT0,
                    objective=dict(name="e2e_chunk_latency_ms", unit="ms", value=16.0))
    with pytest.raises(ValueError, match="requires context transition"):
        campaign.reanchor(directory, evidence)
    assert campaign.rebuild(directory)["iterations"] == 1
    assert campaign.portable_optimizations(directory) == []
