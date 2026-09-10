"""Official adapter coverage must retain each stage's actual checked depth."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from flash_vla.inference import declare
from lab.optimize import policy as acceptance, gate

SCRIPT = "eval.pi05.reference"


def payloads():
    runner = declare("pi05")
    full = runner.identity.as_dict()
    expert = deepcopy(full)
    expert["shape"]["steps"] = 1
    context = runner.measurement_context
    return runner, [
        dict(stage="llm_backbone", identity=full, measurement_context=context, passed=True),
        dict(stage="action_expert", identity=expert, measurement_context=context, passed=True),
    ]


def test_registered_official_stages_accept_their_actual_depths(monkeypatch):
    runner, outputs = payloads()
    proc = subprocess.CompletedProcess([], 0, stdout="\n".join(map(json.dumps, outputs)), stderr="")
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: proc)
    checks = [dict(check="baseline", mode="gate", oracle="official_baseline")]
    result = gate._run_baseline_checks((SCRIPT,), checks, True, sys.executable,
                                      runner.identity, 0, expected_weights=runner.measurement_context["weights"])
    assert result[0]["status"] == "passed"
    record = result[0]["scripts"][0]
    assert record["stages"] == ["llm_backbone", "action_expert"]
    assert record["identities"][1]["shape"]["steps"] == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "depth", "chunk", "variant", "signature"])
def test_coverage_cannot_excuse_another_workload(change):
    runner, outputs = payloads()
    if change == "missing":
        outputs.pop()
    elif change == "duplicate":
        outputs[1]["stage"] = "llm_backbone"
    elif change == "unknown":
        outputs[1]["stage"] = "unknown"
    elif change in {"depth", "chunk"}:
        outputs[1]["identity"]["shape"]["steps" if change == "depth" else "chunk"] += 1
    elif change == "variant":
        outputs[1]["identity"]["execution_variant"]["quantization"]["mode"] = "fp8"
    else:
        outputs[1]["identity"]["inference_signature"] = "sha256:other"
    with pytest.raises(ValueError):
        acceptance.validate_baseline_workloads(
            SCRIPT, [p["identity"] for p in outputs], runner.identity,
            [p["stage"] for p in outputs])


def test_stage_order_is_not_semantic():
    runner, outputs = payloads()
    outputs.reverse()
    acceptance.validate_baseline_workloads(
        SCRIPT, [p["identity"] for p in outputs], runner.identity,
        [p["stage"] for p in outputs])
