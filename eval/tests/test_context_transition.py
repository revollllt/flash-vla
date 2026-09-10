"""Actual CPU subprocess transfer tests; these do not certify real GPU checkpoints."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lab.optimize import campaign, store, transition
from eval.tests.test_weight_dependency import candidate, finalize, workspace, context

# Tiny on-disk weights make recipe execution and compatibility observable.
ADAPTER = """
import json, sys, subprocess
from pathlib import Path
mode, out = sys.argv[1], Path(sys.argv[2])
request = json.loads((out / 'request.json').read_text())
weights = json.loads(Path(request['weights_path']).read_text())
journal = Path(request['journal'])
with journal.open('a') as log:
    log.write(mode + '\\n')
observed_identity = dict(request['identity'])
observed_identity['engine_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True) or mode == 'compatibility'
base = dict(status='pass', identity=observed_identity,
            measurement_context=request['measurement_context'])
if mode == 'compatibility':
    assert weights['shape'] == [2], 'incompatible actual weight shape'
elif mode == 'rebuild':
    (out / 'packed.json').write_text(json.dumps(weights['values'][::-1]))
elif mode == 'retune':
    (out / 'calibration.json').write_text(json.dumps(sum(weights['values'])))
elif mode == 'check':
    if request.get('require_recipes'):
        assert json.loads((out / 'packed.json').read_text()) == weights['values'][::-1]
        assert json.loads((out / 'calibration.json').read_text()) == sum(weights['values'])
    if Path(request['failure_flag']).exists():
        base['status'] = 'failed'
elif mode == 'measure':
    if Path(request['measure_failure_flag']).exists():
        sys.exit(2)
    base.update(validity='valid', instrumented=False, protocol=request['protocol'],
                objective=dict(name=request['objective'], unit='ms', value=request['test_latency']))
print(json.dumps(base))
"""


def command(root, mode):
    return dict(argv=[sys.executable, str(root / "adapter.py"), mode, "{out}"],
                resource="cpu", timeout_s=10)


def request(root, directory, *, latency=20.0):
    (root / "adapter.py").write_text(ADAPTER)
    (root / "weights-b.json").write_text(json.dumps(dict(shape=[2], values=[3, 7])))
    state = campaign.rebuild(directory)
    incoming = copy.deepcopy(state["measurement_context"])
    incoming["weights"] = dict(checkpoint_id="task-b", checkpoint_digest="manifest-b")
    return dict(identity=campaign.implementation_identity(campaign.incumbent_record(directory, state)),
                protocol="latency-v2", objective="e2e_chunk_latency_ms",
                measurement_context=incoming,
                compatibility=command(root, "compatibility"),
                check=command(root, "check"), measure=command(root, "measure"),
                weights_path=str(root / "weights-b.json"), journal=str(root / "journal"),
                failure_flag=str(root / "fail-check"), measure_failure_flag=str(root / "fail-measure"),
                test_latency=latency)


def inherited_recipes(root, directory):
    (root / "adapter.py").write_text(ADAPTER)
    for iteration, dependency in enumerate(("rebuild", "invariant", "retune"), 1):
        spec = candidate(f"step{iteration}", dependency)
        if dependency == "rebuild":
            spec["artifact_recipe"] = command(root, "rebuild")
        if dependency == "retune":
            spec["retune_recipe"] = command(root, "retune")
        campaign.start(root, spec, directory)
        finalize(directory, iteration, "no_benefit" if iteration == 2 else "accepted")


def journal(root):
    return (root / "journal").read_text().splitlines()


def next_spec(directory, name="next"):
    state = campaign.rebuild(directory)
    spec = candidate(name)
    spec.update(measurement_context=state["measurement_context"],
                measurement_segment=state["current_measurement_segment"], fixture=state["fixture"])
    return spec


def test_transfer_executes_recipes_reanchors_and_continues_iter4(workspace):
    root, directory = workspace
    inherited_recipes(root, directory)
    incoming = request(root, directory)
    incoming["require_recipes"] = True
    before = campaign.rebuild(directory)
    state = campaign.transition_context(root, directory, incoming)
    assert journal(root) == ["compatibility", "rebuild", "retune", "check", "measure"]
    assert state["iterations"] == before["iterations"] == 4
    assert state["portable_incumbent"] == "iter-003"
    assert state["current_measurement_segment"] == 1
    assert state["measurement_context"]["weights"] == incoming["measurement_context"]["weights"]
    assert state["segment_anchor_latency_ms"] == state["current_incumbent_latency_ms"] == 20.0
    assert state["improvement_vs_context_anchor_pct"] == 0.0
    assert state["improvement_vs_baseline_pct"] is None
    assert state["budget"]["used"]["candidates"] == before["budget"]["used"]["candidates"]
    saved = transition.records(directory)[0][1]
    assert saved["inherited_iterations"] == [1, 3]
    assert saved["conclusion"] == "reanchor"
    assert saved["status"] == "activated"
    assert saved["cost"]["cpu_seconds"] > 0
    (directory / "state.json").unlink()
    assert campaign.rebuild(directory)["current_measurement_segment"] == 1
    record = campaign.start(root, next_spec(directory), directory)
    assert record["iteration"] == 4
    assert record["parent_incumbent"] == "iter-003"


@pytest.mark.parametrize("change", ["fixture", "driver", "power_policy", "checkpoint"])
def test_equal_latency_does_not_hide_segment_boundary(workspace, change):
    root, directory = workspace
    incoming = request(root, directory, latency=16.0)
    incoming["measurement_context"] = context()
    if change == "checkpoint":
        incoming["measurement_context"]["weights"]["checkpoint_digest"] = "changed"
    elif change == "fixture":
        incoming["measurement_context"]["fixture"] = {"id": "fixture-b", "digest": "fixture-b"}
    else:
        incoming["measurement_context"]["environment"][change] = "changed"
    state = campaign.transition_context(root, directory, incoming)
    assert state["current_measurement_segment"] == 1
    assert len(state["contexts"]) == (2 if change in ("checkpoint", "fixture") else 1)
    assert state["contexts"][state["active_context"]]["latest_segment"] == 1
    assert state["current_incumbent_latency_ms"] == 16.0
    assert state["iterations"] == 1
    assert campaign.start(root, next_spec(directory), directory)["iteration"] == 1


@pytest.mark.parametrize("field,value", [
    ("inference_signature", "incompatible"), ("shape", {"chunk": 32}),
    ("plan", {"attention": "wrong-plan"}), ("engine_revision", "wrong-engine"),
])
def test_identity_mismatch_refused_before_any_work(workspace, field, value):
    root, directory = workspace
    incoming = request(root, directory)
    incoming["identity"][field] = value
    with pytest.raises(ValueError, match="mismatch|implementation"):
        campaign.transition_context(root, directory, incoming)
    assert not (root / "journal").exists()
    assert transition.records(directory) == []


def test_actual_weight_shape_failure_keeps_old_context(workspace):
    root, directory = workspace
    incoming = request(root, directory)
    Path(incoming["weights_path"]).write_text(json.dumps(dict(shape=[3], values=[3, 7, 9])))
    with pytest.raises(RuntimeError, match="compatibility failed"):
        campaign.transition_context(root, directory, incoming)
    state = campaign.rebuild(directory)
    assert state["measurement_context"] == context()
    assert state["current_measurement_segment"] == 0
    assert journal(root) == ["compatibility"]
    assert "materialization" not in transition.records(directory)[0][1]


def test_failed_correctness_is_not_activation_and_retry_is_explicit(workspace):
    root, directory = workspace
    inherited_recipes(root, directory)
    incoming = request(root, directory)
    Path(incoming["failure_flag"]).touch()
    with pytest.raises(ValueError, match="correctness did not pass"):
        campaign.transition_context(root, directory, incoming)
    state = campaign.rebuild(directory)
    assert state["current_measurement_segment"] == 0
    assert state["measurement_context"] == context()
    with pytest.raises(RuntimeError, match="active candidate"):
        campaign.start(root, candidate("forbidden"), directory)
    with pytest.raises(RuntimeError, match="reconcile"):
        campaign.resume(directory)
    Path(incoming["failure_flag"]).unlink()
    state = campaign.resume(directory, recovered_seconds=0)
    assert state["current_measurement_segment"] == 1
    assert journal(root) == ["compatibility", "rebuild", "retune", "check", "check", "measure"]
    assert len(transition.records(directory)[0][1]["attempts"]) == 1


def test_failed_measurement_does_not_rerun_completed_recipes(workspace):
    root, directory = workspace
    inherited_recipes(root, directory)
    incoming = request(root, directory)
    Path(incoming["measure_failure_flag"]).touch()
    with pytest.raises(RuntimeError, match="measure failed"):
        campaign.transition_context(root, directory, incoming)
    Path(incoming["measure_failure_flag"]).unlink()
    state = campaign.resume(directory, recovered_seconds=0)
    assert state["iterations"] == 4
    assert journal(root) == ["compatibility", "rebuild", "retune", "check", "measure", "measure"]


def test_pending_transition_blocks_duplicate_and_activation_is_idempotent(workspace):
    root, directory = workspace
    incoming = request(root, directory)
    run_dir = transition.begin(root, directory, incoming)
    with pytest.raises(RuntimeError, match="already pending"):
        transition.begin(root, directory, incoming)
    transition.run(directory, run_dir)
    before = journal(root)
    state = transition.run(directory, run_dir)
    assert journal(root) == before
    assert state["iterations"] == 1


def test_abort_requires_new_anchor_before_candidates(workspace):
    root, directory = workspace
    incoming = request(root, directory)
    transition.begin(root, directory, incoming)
    state = transition.abort(directory)
    assert state["reanchor_required"]
    with pytest.raises(RuntimeError, match="requires an incumbent re-anchor"):
        campaign.start(root, candidate(), directory)
    state = campaign.transition_context(root, directory, incoming)
    assert not state["reanchor_required"]
    assert state["current_measurement_segment"] == 1


def test_missing_environment_and_protocol_changes_refused_before_work(workspace):
    root, directory = workspace
    incoming = request(root, directory)
    incoming["measurement_context"]["environment"]["power_policy"] = None
    with pytest.raises(ValueError, match="power_policy"):
        transition.begin(root, directory, incoming)
    incoming["measurement_context"] = context()
    incoming["protocol"] = "other-protocol"
    with pytest.raises(ValueError, match="new Campaign"):
        transition.begin(root, directory, incoming)
    assert not (root / "journal").exists()


def test_stale_context_candidate_cannot_allocate_or_finalize(workspace):
    root, directory = workspace
    campaign.transition_context(root, directory, request(root, directory))
    with pytest.raises(ValueError, match="context/segment"):
        campaign.start(root, candidate(), directory)
    spec = next_spec(directory)
    campaign.start(root, spec, directory)
    with pytest.raises(ValueError, match="context changed"):
        finalize(directory, 1, "accepted", result={"validity": "valid",
                                                 "measurement_context": context()})
    assert campaign.rebuild(directory)["portable_incumbent"] == "iter-000"


def test_successful_transition_clears_imported_reanchor_requirement(workspace):
    root, directory = workspace
    path = directory / "runs/iter-000-baseline/evidence.json"
    baseline = store.read(path)
    baseline["measurement"]["evidence"]["continuation"] = {"reanchor_required": True}
    store.write(path, baseline)
    assert campaign.rebuild(directory)["reanchor_required"]
    state = campaign.transition_context(root, directory, request(root, directory))
    assert not state["reanchor_required"]
    campaign.start(root, next_spec(directory), directory)


def test_old_environment_correctness_cannot_activate_new_environment(workspace):
    root, directory = workspace
    incoming = request(root, directory)
    incoming["measurement_context"]["environment"]["driver"] = "new-driver"
    source = ADAPTER.replace("elif mode == 'measure':",
        "    base['measurement_context']['environment']['driver'] = 'old-driver'\nelif mode == 'measure':")
    (root / "adapter.py").write_text(source)
    with pytest.raises(ValueError, match="correctness context/environment"):
        campaign.transition_context(root, directory, incoming)
    assert campaign.rebuild(directory)["current_measurement_segment"] == 0
    assert journal(root) == ["compatibility", "check"]


def test_committed_incumbent_runs_in_clean_checkout_at_its_actual_revision(workspace):
    from flash_vla.runtime.identity import git_revision

    root, directory = workspace
    source = root / "src/kernel.cu"
    source.write_text("portable implementation")
    subprocess.run(["git", "-C", str(root), "add", "src/kernel.cu"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "-qm", "portable"], check=True)
    campaign.start(root, candidate("portable"), directory)
    finalize(directory, 1, "accepted")
    portable_revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    source.write_text("checkpoint-specific implementation")
    subprocess.run(["git", "-C", str(root), "add", "src/kernel.cu"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.org", "commit", "-qm", "specific"], check=True)
    campaign.start(root, candidate("specific", "checkpoint_specific"), directory)
    finalize(directory, 2, "accepted")
    state = campaign.transition_context(root, directory, request(root, directory))
    execution_root = Path(state["execution_repository"])
    assert git_revision(execution_root / "src/kernel.cu") == portable_revision
    assert (execution_root / "src/kernel.cu").read_text() == "portable implementation"
    assert source.read_text() == "checkpoint-specific implementation"
    spec = next_spec(directory)
    spec["parent_plan"] = state["target"].get("plan", candidate()["identity"]["plan"])
    assert campaign.start(execution_root, spec, directory)["parent_incumbent"] == "iter-001"


def test_transition_anchor_is_not_an_iteration_in_trace_or_plot(workspace):
    from lab.optimize import trace, render

    root, directory = workspace
    inherited_recipes(root, directory)
    campaign.transition_context(root, directory, request(root, directory))
    normalized = trace.normalize(directory)
    assert [r["iteration"] for r in normalized["iterations"]] == [0, 1, 2, 3]
    assert normalized["segments"][1]["before_iteration"] == 4
    _, points = render.from_trace(normalized)
    assert points[-1].iteration is None and points[-1].reanchor
    assert points[-1].incumbent_ms == 20.0
    assert points[-1].delta_parent_pct is None
    campaign.start(root, next_spec(directory), directory)
    finalize(directory, 4, "accepted", result={"validity": "valid",
                                              "candidate_ms": 19.0, "parent_incumbent_ms": 20.0})
    normalized = trace.normalize(directory)
    assert normalized["iterations"][-1]["current_incumbent_latency_ms"] == 19.0
    assert normalized["iterations"][-1]["segment_baseline_latency_ms"] == 20.0
    assert normalized["iterations"][-1]["delta_vs_baseline_pct"] == pytest.approx(-5.0)


def test_parent_latency_requires_matching_context_control_legs(workspace):
    from eval import acceptance
    root, directory = workspace
    campaign.transition_context(root, directory, request(root, directory))
    record = campaign.start(root, next_spec(directory), directory)
    state = campaign.rebuild(directory)
    candidate_identity = campaign.implementation_identity(record)
    parent_identity = campaign.implementation_identity(campaign.incumbent_record(directory, state))
    policy = acceptance.for_target(candidate_identity["target"])["latency"]
    aba = dict(config={key: policy[key] for key in ("reps", "warmup")}, legs=[
        dict(identity=who, measurement_context=state["measurement_context"],
             metrics={"chunk_latency": {"min": value}})
        for who, value in ((parent_identity, 20.0), (candidate_identity, 19.0), (parent_identity, 20.0))])
    result = dict(validity="valid", candidate_ms=19.0, parent_incumbent_ms=30.0, aba=aba)
    with pytest.raises(ValueError, match="must come from"):
        finalize(directory, 1, "accepted", result=result)
    result["parent_incumbent_ms"] = 20.0
    result["aba"]["legs"][0]["measurement_context"] = context()
    with pytest.raises(ValueError, match="context changed"):
        finalize(directory, 1, "accepted", result=result)
    assert campaign.rebuild(directory)["portable_incumbent"] == "iter-000"


def test_reconcile_cannot_unpublish_activated_transition(workspace):
    from lab.optimize import runner

    root, directory = workspace
    campaign.transition_context(root, directory, request(root, directory))
    path = transition.records(directory)[0][0]
    with pytest.raises(RuntimeError, match="terminal"):
        runner.reconcile(path.parent, recovered_seconds=0)
    assert campaign.rebuild(directory)["current_measurement_segment"] == 1


def test_concurrent_transition_reservation_has_one_winner(workspace):
    from concurrent.futures import ThreadPoolExecutor

    root, directory = workspace
    incoming = request(root, directory)
    def allocate():
        try:
            return transition.begin(root, directory, incoming)
        except RuntimeError as error:
            assert "already pending" in str(error)
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: allocate(), range(2)))
    assert sum(outcome is not None for outcome in outcomes) == 1
    assert len(transition.records(directory)) == 1


@pytest.mark.parametrize("change", ["checkpoint", "fixture"])
def test_render_draws_separate_latency_lines_for_context_segments(workspace, tmp_path, change):
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("Matplotlib not installed")
    from unittest.mock import patch
    from matplotlib.axes import Axes
    from lab.optimize import trace, render

    root, directory = workspace
    inherited_recipes(root, directory)
    incoming = request(root, directory)
    if change == "fixture":
        incoming["measurement_context"]["weights"] = context()["weights"]
        incoming["measurement_context"]["fixture"] = {"id": "fixture-b", "digest": "fixture-b"}
    campaign.transition_context(root, directory, incoming)
    campaign.start(root, next_spec(directory), directory)
    finalize(directory, 4, "accepted", result={"validity": "valid",
                                              "candidate_ms": 19.0, "parent_incumbent_ms": 20.0})
    metadata, points = render.from_trace(trace.normalize(directory))
    lines = []
    original = Axes.plot
    def capture(axis, xs, ys, *args, **kwargs):
        if kwargs.get("color") == "C0":
            lines.append((list(xs), list(ys)))
        return original(axis, xs, ys, *args, **kwargs)
    with patch.object(Axes, "plot", capture):
        render.render_optimization_progress(metadata=metadata, points=points,
                                             output_svg=tmp_path / "contexts.svg",
                                             output_png=tmp_path / "contexts.png")
    assert lines == [([0, 1, 2, 3], [16.0, 14.0, 14.0, 14.0]),
                     ([3.5, 4], [20.0, 19.0])]
    svg = (tmp_path / "contexts.svg").read_text()
    assert any(("fixture fixture-b" if change == "fixture" else "checkpoint task-b")
               in point.summary for point in points if point.reanchor)
    assert "segment 1 re-anchor" in svg
    assert "iter None" not in svg
    render.render_optimization_progress(metadata=metadata, points=points,
                                         output_svg=tmp_path / "repeated.svg")
    assert (tmp_path / "repeated.svg").read_text() == svg


def test_same_assets_new_environment_still_materializes_required_artifacts(workspace):
    root, directory = workspace
    inherited_recipes(root, directory)
    incoming = request(root, directory)
    incoming["require_recipes"] = True
    campaign.transition_context(root, directory, incoming)
    incoming["measurement_context"]["environment"]["driver"] = "driver-updated"
    state = campaign.transition_context(root, directory, incoming)
    assert state["iterations"] == 4
    assert state["current_measurement_segment"] == 2
    assert journal(root) == ["compatibility", "rebuild", "retune", "check", "measure"] * 2


def test_context_registry_restores_history_and_reuses_returning_context(workspace):
    from flash_vla.runtime.identity import MeasurementContext

    root, directory = workspace
    before = campaign.rebuild(directory)
    original = copy.deepcopy(before["measurement_context"])
    context_a = MeasurementContext.from_dict(original).context_id
    assert before["active_context"] == context_a
    assert before["contexts"] == {context_a: dict(
        weights=original["weights"], fixture=original["fixture"],
        created=original["timestamp"], latest_segment=0)}

    incoming = request(root, directory)
    incoming["measurement_context"]["timestamp"] = original["timestamp"] + 10
    switched = campaign.transition_context(root, directory, incoming)
    context_b = MeasurementContext.from_dict(incoming["measurement_context"]).context_id
    assert context_b != context_a
    assert switched["active_context"] == context_b
    assert switched["contexts"][context_a] == before["contexts"][context_a]
    assert switched["contexts"][context_b]["latest_segment"] == 1

    returning = request(root, directory, latency=31.0)
    returning["measurement_context"] = copy.deepcopy(original)
    returning["measurement_context"]["timestamp"] += 20
    returning["measurement_context"]["environment"]["driver"] = "new driver"
    state = campaign.transition_context(root, directory, returning)
    assert state["active_context"] == context_a
    assert len(state["contexts"]) == 2
    assert state["contexts"][context_a] == dict(before["contexts"][context_a], latest_segment=2)
    assert state["contexts"][context_b] == switched["contexts"][context_b]
    assert state["current_incumbent_latency_ms"] == state["segment_anchor_latency_ms"] == 31.0
    assert state["improvement_vs_baseline_pct"] is None
    assert state["iterations"] == 1
    store.write(directory / "state.json", {"contexts": {}, "active_context": "stale"})
    rebuilt = campaign.rebuild(directory)
    assert rebuilt["contexts"] == state["contexts"]
    assert rebuilt["active_context"] == context_a


def test_pending_and_aborted_transition_do_not_register_context(workspace):
    root, directory = workspace
    before = campaign.rebuild(directory)
    incoming = request(root, directory)
    transition.begin(root, directory, incoming)
    pending = campaign.rebuild(directory)
    assert pending["pending_transition"] is not None
    assert pending["contexts"] == before["contexts"]
    assert pending["active_context"] == before["active_context"]
    aborted = transition.abort(directory)
    assert aborted["contexts"] == before["contexts"]
    assert aborted["active_context"] == before["active_context"]
