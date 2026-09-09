"""Production-format adapters exercised through real harness logic with CPU-only toy engines."""
from contextlib import nullcontext
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import latency
from eval import acceptance, correctness as in_engine, gate
from eval.tests.test_identity_v3 import payload
from flash_vla.runtime.identity import Identity
from lab.optimize import campaign, promotion, reports, store

ENV = dict(gpu="test GPU", driver="test driver", torch_cuda="13.0", torch="test torch",
           tilelang="test compiler", clock_policy="unlocked", clocks="unlocked",
           power_policy={"requested_limit_w": 700., "enforced_limit_w": 700.},
           node="test node", job="test job")
ASSETS = dict(weights={"checkpoint_id": "toy-a", "checkpoint_digest": "toy-manifest-a"},
              fixture={"id": "toy-inputs", "digest": "toy-inputs-v1"})


class Engine:
    program = ()
    stage_outputs = {}
    graph = SimpleNamespace(call_sites=("attention",))
    target = SimpleNamespace(
        select_plan=lambda name: {"attention": "reference"},
        registry=SimpleNamespace(resolve=lambda plan, sites: dict(plan)))
    device = torch.device("cpu")

    def __init__(self, plan, *, steps=None, layers=None, **kwargs):
        self.identity = Identity.from_dict(payload(
            plan={"attention": plan},
            shape={"chunk": 50, "steps": 10 if steps is None else steps,
                   "layers": 18 if layers is None else layers}))
        self.measurement_context = deepcopy(ASSETS)

    def sample_inputs(self, seed):
        return {}

    def stage(self, **inputs):
        pass

    def forward(self, **inputs):
        return torch.tensor([1., 2.])


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(in_engine, "build", lambda target, plan, **kw: Engine(plan, **kw))
    monkeypatch.setattr(in_engine, "_env", lambda *args: deepcopy(ENV))
    monkeypatch.setattr(latency, "build", lambda target, plan, **kw: Engine(plan, **kw))
    monkeypatch.setattr(latency, "_env", lambda *args: deepcopy(ENV))
    monkeypatch.setattr(gate, "declare", lambda target, plan, **kw: Engine(plan, **kw))
    monkeypatch.setattr(gate, "_registry_version", lambda: "fixture registry")

    def official(scripts, checks, run, python, expected, seed, expected_weights=None, options=None):
        assert run is True
        evidence = [dict(script=name, status="passed", returncode=0,
                         identities=[expected.as_dict()], weights=[deepcopy(expected_weights)])
                    for name in scripts]
        return [dict(check=c["check"], mode=c["mode"], status="passed", scripts=evidence)
                for c in checks if c.get("oracle") == "official_baseline"]

    monkeypatch.setattr(gate, "_run_baseline_checks", official)
    return monkeypatch


def measured(harness, *, plans=("shipped", "candidate", "shipped"), values=(16., 14., 16.01)):
    sequence = iter(values)

    def measure(engine, inputs, reps, *args, **kwargs):
        value = next(sequence)
        stats = dict(min=value, median=value + .01, p99=value + .02, n=reps)
        return dict(chunk_latency=stats, device_latency=stats,
                    host_time={}, segment_latency={}, overhead=stats)

    harness.setattr(latency, "measure", measure)
    return latency.run("h100/pi05", list(plans), attribution=False)


def checked(harness, tmp_path):
    return gate.run("h100/pi05", baseline=True, correctness_only=True,
                    out_dir=str(tmp_path / "correctness"))


def test_correctness_only_runs_existing_ladder_without_latency(harness, tmp_path):
    harness.setattr(latency, "run", lambda *a, **kw: pytest.fail("latency was executed"))
    raw = checked(harness, tmp_path)
    assert raw["latency"] is None
    assert raw["verdict"] == "correctness_pass"
    normalized = reports.correctness(raw)
    assert normalized["status"] == "pass"
    assert normalized["identity"]["shape"] == {"chunk": 50, "steps": 10, "layers": 18}
    assert normalized["measurement_context"]["weights"] == ASSETS["weights"]


def test_correctness_producer_rejects_environment_drift(harness):
    environments = iter([ENV, dict(ENV, driver="changed driver")])
    harness.setattr(in_engine, "_env", lambda *args: deepcopy(next(environments)))
    with pytest.raises(ValueError, match="context changed"):
        in_engine.run("h100/pi05")


def test_correctness_producer_rejects_fixture_mismatch(harness):
    def build(target, plan, **kwargs):
        engine = Engine(plan, **kwargs)
        if plan != "reference":
            engine.measurement_context["fixture"]["digest"] = "different"
        return engine

    harness.setattr(in_engine, "build", build)
    with pytest.raises(ValueError, match="fixture provenance"):
        in_engine.run("h100/pi05")


def test_real_latency_report_converts_without_losing_aba(harness):
    raw = measured(harness)
    result = reports.latency(raw, objective="e2e_chunk_latency_ms", segment=2)
    assert result["parent_incumbent_ms"] == 16.
    assert result["candidate_ms"] == 14.
    assert result["measurement_segment"]["id"] == 2
    assert result["aba"] == raw
    assert result["identity"] == raw["legs"][1]["identity"]


@pytest.mark.parametrize("field,value", [
    ("reps", 5), ("warmup", 0), ("soak_s", 0), ("p99_min_reps", 5),
])
def test_sampling_overrides_cannot_be_relabelled_as_fixed_protocol(harness, field, value):
    raw = measured(harness)
    raw["config"][field] = value
    with pytest.raises(ValueError, match="sampling"):
        reports.latency(raw, objective="e2e_chunk_latency_ms", segment=0)


@pytest.mark.parametrize("change", ["missing_clock", "checkpoint", "engine", "instrumented",
                                   "missing_protocol", "n", "p99", "middle_anchor_drift"])
def test_invalid_or_incomplete_latency_cannot_activate_anchor(harness, change):
    raw = measured(harness, plans=("shipped",) * 3, values=(16., 16., 16.01))
    if change == "missing_clock":
        for leg in raw["legs"]:
            leg["measurement_context"]["environment"]["clock_policy"] = None
    elif change == "checkpoint":
        raw["legs"][2]["measurement_context"]["weights"]["checkpoint_digest"] = "other"
    elif change == "engine":
        raw["legs"][2]["identity"]["engine_revision"] = "other"
    elif change == "instrumented":
        raw["instrumented"] = True
    elif change == "missing_protocol":
        del raw["protocol"]
    elif change in ("n", "p99"):
        raw["legs"][1]["metrics"]["chunk_latency"][change] = 1 if change == "n" else None
    else:
        raw["legs"][1]["metrics"]["chunk_latency"]["min"] = 15.
    with pytest.raises(ValueError):
        reports.latency(raw, objective="e2e_chunk_latency_ms", anchor=True)


@pytest.mark.parametrize("change", ["missing_gate", "report_only", "wrong_depth", "wrong_plan",
                                   "wrong_weights", "wrong_environment", "nonfinite",
                                   "tolerance", "official_missing", "official_weights"])
def test_correctness_adapter_does_not_upgrade_weak_evidence(harness, tmp_path, change):
    raw = checked(harness, tmp_path)
    checks = {c["check"]: c for c in raw["checks"]}
    shallow = checks["in_engine_shallow"]
    full = checks["in_engine_multistep"]
    if change == "missing_gate":
        raw["checks"].remove(shallow)
    elif change == "report_only":
        shallow["mode"] = "report"
    elif change == "wrong_depth":
        shallow["report"]["config"]["layers"] = 2
    elif change == "wrong_plan":
        full["report"]["identity"]["candidate"]["plan"] = {"attention": "other"}
    elif change == "wrong_weights":
        full["report"]["measurement_context"]["candidate"]["weights"]["checkpoint_digest"] = "other"
    elif change == "wrong_environment":
        full["report"]["measurement_context"]["candidate"]["environment"]["driver"] = "other"
    elif change == "nonfinite":
        full["report"]["finite"] = False
    elif change == "tolerance":
        shallow["report"]["min_cosine"] = 0.
    elif change == "official_missing":
        checks["baseline_layer0"]["scripts"] = []
    else:
        checks["baseline_layer0"]["scripts"][0]["weights"][0]["checkpoint_digest"] = "other"
    with pytest.raises(ValueError):
        reports.correctness(raw)


def test_cli_anchor_from_harness_reports_creates_campaign(harness, tmp_path):
    raw_check = checked(harness, tmp_path)
    checked_path = tmp_path / "checked.json"
    store.write(checked_path, reports.correctness(raw_check))
    raw_path = tmp_path / "latency.json"
    raw = measured(harness, plans=("shipped",) * 3, values=(16., 16.01, 16.02))
    store.write(raw_path, raw)
    output = subprocess.run(
        [sys.executable, "-m", "lab.optimize.reports", "anchor", str(raw_path),
         "--correctness", str(checked_path)],
        capture_output=True, text=True, check=True)
    baseline = json.loads(output.stdout)
    directory = tmp_path / "campaign"
    campaign.create(directory, baseline, "e2e_chunk_latency_ms", "latency-v2",
                    ASSETS["fixture"]["id"])
    assert campaign.rebuild(directory)["current_incumbent_latency_ms"] == 16.01
    assert baseline["aba"]["legs"] == raw["legs"]

def test_failed_correctness_stops_before_official_and_measurement(harness, tmp_path):
    def forward(self, **inputs):
        sign = 1. if self.identity.plan["attention"] == "reference" else -1.
        return torch.tensor([1., 2.]) * sign

    harness.setattr(Engine, "forward", forward)
    harness.setattr(gate, "_run_baseline_checks", lambda *a, **kw: pytest.fail("official work ran"))
    harness.setattr(latency, "run", lambda *a, **kw: pytest.fail("measurement ran"))
    raw = checked(harness, tmp_path)
    assert raw["verdict"] == "fail"
    assert raw["latency"] is None
    with pytest.raises(ValueError, match="did not pass"):
        reports.correctness(raw)


def test_formal_gate_requests_uninstrumented_latency(harness, tmp_path):
    raw = measured(harness, plans=("shipped",) * 3, values=(16., 16.01, 16.02))

    def invoke(*args, **kwargs):
        assert kwargs["attribution"] is False
        return raw

    harness.setattr(latency, "run", invoke)
    record = gate.run("h100/pi05", baseline=True, mode="no_regression",
                      out_dir=str(tmp_path / "gate"))
    assert record["verdict"] == "pass"
    assert record["latency"]["report"]["instrumented"] is False

@pytest.mark.parametrize("change", ["candidate_route", "wrong_revision", "oracle_revision",
                                   "wrong_script", "missing_script"])
def test_oracle_provenance_cannot_be_replaced_with_self_comparison(harness, tmp_path, change):
    raw = checked(harness, tmp_path)
    checks = {c["check"]: c for c in raw["checks"]}
    report = checks["in_engine_shallow"]["report"]
    if change == "candidate_route":
        report["identity"]["reference"]["plan"] = dict(report["identity"]["candidate"]["plan"])
    elif change == "wrong_revision":
        report["identity"]["reference"]["engine_revision"] = "other"
    elif change == "oracle_revision":
        report["numerical_oracle"]["identity"]["engine_revision"] = "other"
    elif change == "wrong_script":
        checks["baseline_layer0"]["scripts"][0]["script"] = "unregistered.adapter"
    else:
        del checks["baseline_layer0"]["scripts"][0]["script"]
    with pytest.raises(ValueError):
        reports.correctness(raw)


def test_leg_instrumentation_cannot_be_hidden_by_top_level_flags(harness):
    raw = measured(harness)
    raw["legs"][1]["attribution"] = {"loops": {"chunk_latency": "diagnostic samples"}}
    with pytest.raises(ValueError, match="attribution"):
        reports.latency(raw, objective="e2e_chunk_latency_ms", segment=0)

def test_correctness_only_cli_success_cannot_qualify_promotion(harness, tmp_path):
    harness.setattr(latency, "run", lambda *a, **kw: pytest.fail("measurement ran"))
    out = tmp_path / "cli"
    assert gate.main(["--target", "h100/pi05", "--baseline", "--correctness-only",
                      "--out-dir", str(out)]) == 0
    report_path, = out.rglob("*.json")
    record = store.read(report_path)
    # All other applicability conditions hold, so rejection is specifically
    # the difference between correctness success and full qualification.
    source = dict(root=str(tmp_path), inputs=[])
    evidence = dict(incumbent_source=source, candidate_source=source,
                    acceptance=record["acceptance"], qualified_targets=[record["target"]],
                    gate_verdict=record["verdict"])
    result = promotion.applicability(evidence, source, source, record["acceptance"],
                                     [record["target"]])
    assert result["applicable"] is False
    assert result["reasons"] == ["existing gate has not passed"]
