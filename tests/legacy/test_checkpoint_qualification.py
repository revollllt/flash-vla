"""Checkpoint forwarding through the optional historical qualification command."""
import json
import pytest
from flash_vla import inference as targets
from lab.optimize import gate
from tests.test_checkpoint_factory import checkpoint_options
from tests.test_openpi05_config import upstream


def test_gate_forwards_construction_options_to_every_consumer(checkpoint_options, monkeypatch):
    observed = {}
    def declaration(*args, **kwargs):
        observed["declare"] = kwargs
        return targets.declare("pi05", **checkpoint_options)
    monkeypatch.setattr(gate, "declare", declaration)
    monkeypatch.setattr(gate, "_registry_version", lambda: "test")
    def checks(*args, **kwargs):
        observed["check"] = kwargs
        return [dict(check="numeric", mode="gate", status="passed")]
    monkeypatch.setattr(gate, "_run_in_engine_checks", checks)
    def baseline(*args, **kwargs):
        observed["baseline"] = kwargs
        return []
    monkeypatch.setattr(gate, "_run_baseline_checks", baseline)
    def measure(*args, **kwargs):
        observed["latency"] = kwargs
        return {}
    monkeypatch.setattr(gate.latency, "run", measure)
    monkeypatch.setattr(gate, "_latency_verdict",
                        lambda *args: dict(run_valid=True, passed=True))
    monkeypatch.setattr(gate, "_deployment_verdict", lambda *args: dict(status="passed"))
    def floor(*args, **kwargs):
        observed["floor"] = kwargs
        return {}
    monkeypatch.setattr(gate.floor_model, "run", floor)
    monkeypatch.setattr(gate, "_finish", lambda record, out: record)
    result = gate.run("pi05", baseline=True, include_floor=True, **checkpoint_options)
    assert result["verdict"] == "pass"
    for name in ("declare", "check", "latency", "floor"):
        assert checkpoint_options.items() <= observed[name].items()
    assert observed["baseline"]["options"] == checkpoint_options
    assert observed["latency"]["attribution"] is False


def test_baseline_options_run_as_literal_subprocess_arguments(tmp_path, checkpoint_options, monkeypatch):
    import sys
    expected = targets.declare("pi05", **checkpoint_options)
    payload = dict(identity=expected.identity.as_dict(),
                   measurement_context={"weights": expected.measurement_context["weights"]})
    script = tmp_path / "fixture_adapter.py"
    options = {**checkpoint_options, "checkpoint": "/a path/with $(literal) chars"}
    script.write_text(
        "import argparse,json\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--seed')\n"
        "p.add_argument('--option',action='append',default=[])\n"
        "a=p.parse_args()\n"
        + "assert dict(x.split('=',1) for x in a.option) == " + repr(options) + "\n"
        + "print(" + repr(json.dumps(payload)) + ")\n")
    monkeypatch.setattr(gate, "REPO", tmp_path)
    result = gate._run_baseline_checks(
        ("fixture_adapter",), [dict(check="official", mode="gate", oracle="official_baseline")],
        True, sys.executable, expected.identity, 0,
        expected_weights=expected.measurement_context["weights"], options=options)
    assert result[0]["status"] == "passed"
    assert result[0]["scripts"][0]["weights"] == [expected.measurement_context["weights"]]


def test_gate_workload_depth_does_not_override_fixed_shallow_checks(monkeypatch):
    depths = []
    def check(*args, **kwargs):
        depths.append((kwargs["steps"], kwargs["layers"], kwargs["checkpoint"]))
        return dict(replay_identical=True, finite=True, within_tolerance=True,
                    min_cosine=1, max_rel_rms=0, tolerance={})
    monkeypatch.setattr(gate.in_engine, "run", check)
    checks = [
        dict(check="shallow", mode="gate", oracle="in_engine_reference", threshold="shallow",
             config={"steps": 1, "layers": 1}),
        dict(check="deep", mode="report", oracle="in_engine_reference",
             config={"steps": 1, "layers": "full"}),
        dict(check="full", mode="report", oracle="in_engine_reference",
             config={"steps": "full", "layers": "full"}),
    ]
    gate._run_in_engine_checks("target", "candidate", checks, 0, steps=5, layers=12,
                              checkpoint="weights")
    assert depths == [(1, 1, "weights"), (1, 12, "weights"), (5, 12, "weights")]
