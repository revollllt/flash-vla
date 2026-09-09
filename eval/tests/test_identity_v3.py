"""Identity-v3 acceptance: architecture, execution policy and measurement axes."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest

from flash_vla.runtime import identity as identity_module
from flash_vla.runtime.identity import Identity
from lab.optimize import campaign


def payload(**changes):
    value = {
        "schema_version": 3,
        "target": "hardware/nvidia/h100/pi05",
        "hardware": "h100-sxm5-80gb",
        "model": "pi05",
        "model_revision": "pi05-r1",
        "inference_signature": "sha256:pi05-architecture",
        "shape": {"chunk": 50, "steps": 10},
        "execution_variant": {"quantization": {"mode": "bf16"}, "cache": {"mode": "none"}},
        "plan": {"attention": "reference"},
        "engine_revision": "engine-a",
    }
    value.update(changes)
    return value


def context(**changes):
    value = {
        "weights": {"checkpoint_id": "task-a", "checkpoint_digest": "manifest-a"},
        "fixture": {"id": "fixture-a", "digest": "fixture-manifest-a"},
        "environment": {
            "gpu_sku": "h100-sxm5-80gb", "driver": "driver-a",
            "cuda_runtime": "13.1", "pytorch": "torch-a", "tilelang": "tilelang-a",
            "clock_policy": "unlocked", "power_policy": "default",
            "capture_regime": "cuda-graph",
        },
        "hostname": "node-a", "slurm_job_id": "1", "timestamp": 1,
        "reference_provenance": {"repository": "openpi", "commit": "upstream-a"},
    }
    value.update(changes)
    return value


def context_value(**changes):
    cls = getattr(identity_module, "MeasurementContext", None)
    assert cls is not None, "measurement provenance must have a separate MeasurementContext"
    return cls.from_dict(context(**changes))


def key(identity, *, objective="e2e_chunk_latency_ms", protocol="latency-v2"):
    fn = getattr(campaign, "campaign_key", None)
    assert fn is not None, "Campaign discovery needs a key independent of checkpoint and fixture"
    return fn(identity, objective=objective, protocol=protocol)


@pytest.mark.parametrize("change,same_target,same_campaign", [
    ({"engine_revision": "engine-b"}, True, True),
    ({"plan": {"attention": "cuda-fused"}}, True, True),
    ({"shape": {"chunk": 32, "steps": 10}}, False, False),
    ({"hardware": "h100-pcie-80gb"}, False, False),
    ({"model_revision": "pi05-r2"}, False, False),
    ({"inference_signature": "sha256:incompatible"}, False, False),
    ({"target": "hardware/nvidia/h100/pi0", "model": "pi0",
      "model_revision": "pi0-r1", "inference_signature": "sha256:pi0"}, False, False),
    ({"execution_variant": {"quantization": {"mode": "fp8"}, "cache": {"mode": "none"}}},
     True, False),
    ({"execution_variant": {"quantization": {"mode": "bf16"}, "cache": {"mode": "dit_cache"}}},
     True, False),
])
def test_identity_and_campaign_axes(change, same_target, same_campaign):
    first, other = (Identity.from_dict(value).as_dict() for value in (payload(), payload(**change)))
    assert (campaign.target_key(first) == campaign.target_key(other)) is same_target
    assert (key(first) == key(other)) is same_campaign


def test_protocol_and_objective_are_campaign_axes():
    value = Identity.from_dict(payload()).as_dict()
    assert key(value) != key(value, protocol="latency-v3")
    assert key(value) != key(value, objective="device_latency_ms")


def test_variant_round_trip_and_workload_comparison():
    first = Identity.from_dict(payload())
    other = Identity.from_dict(payload(execution_variant={
        "quantization": {"mode": "bf16"}, "cache": {"mode": "dit_cache"},
    }))
    assert first.as_dict()["execution_variant"] == payload()["execution_variant"]
    assert "precision" not in first.as_dict()
    assert not first.same_workload(other)
    assert first.same_workload(replace(first, plan={"attention": "candidate"}))


@pytest.mark.parametrize("field", ["weights", "fixture"])
def test_checkpoint_and_fixture_change_context_not_campaign(field):
    first = context_value()
    replacement = ({"checkpoint_id": "task-b", "checkpoint_digest": "manifest-b"}
                   if field == "weights" else {"id": "fixture-b", "digest": "fixture-manifest-b"})
    other = context_value(**{field: replacement})
    assert first.context_id != other.context_id
    assert first.segment_key != other.segment_key
    report_a = {"identity": payload(), "measurement_context": first.as_dict()}
    report_b = {"identity": payload(), "measurement_context": other.as_dict()}
    assert key(report_a["identity"]) == key(report_b["identity"])


@pytest.mark.parametrize("field", [
    "gpu_sku", "driver", "cuda_runtime", "pytorch", "tilelang",
    "clock_policy", "power_policy", "capture_regime",
])
def test_environment_changes_segment_but_not_context_id(field):
    first = context_value()
    environment = context()["environment"]
    environment[field] += "-changed"
    other = context_value(environment=environment)
    assert first.context_id == other.context_id
    assert first.segment_key != other.segment_key


def test_observation_and_oracle_provenance_do_not_split_compatible_context():
    first = context_value()
    other = context_value(
        hostname="node-b", slurm_job_id="2", timestamp=2,
        reference_provenance={"repository": "openpi", "commit": "upstream-b"},
    )
    assert first.context_id == other.context_id
    assert first.segment_key == other.segment_key
    assert other.as_dict()["reference_provenance"]["commit"] == "upstream-b"


def test_checkpoint_location_is_provenance_only():
    first = context_value(weights={**context()["weights"], "location": "/machine-a/model"})
    other = context_value(weights={**context()["weights"], "location": "/machine-b/model"})
    assert first.context_id == other.context_id
    assert first.segment_key == other.segment_key


@pytest.mark.parametrize("target", ["h100/pi0", "h100/pi05"])
def test_random_seed_keeps_target_and_architecture_revision(target):
    from benchmarks.targets import declare
    first = declare(target, seed=0)
    other = declare(target, seed=1)
    assert first.identity.model_revision == other.identity.model_revision
    assert first.identity.same_workload(other.identity)
    assert first.measurement_context["weights"] != other.measurement_context["weights"]
    first_context = context_value(weights=first.measurement_context["weights"])
    other_context = context_value(weights=other.measurement_context["weights"])
    assert first_context.context_id != other_context.context_id
    assert first_context.segment_key != other_context.segment_key
    assert first.identity.inference_signature == other.identity.inference_signature


@pytest.mark.parametrize("target,revision", [
    ("h100/pi0", "pi0-r1"), ("h100/pi05", "pi05-r1"),
    ("h100/lingbot_vla", "lingbot-vla-r1"),
])
def test_target_owns_architecture_metadata(target, revision):
    from benchmarks.targets import declare
    runner = declare(target)
    assert runner.identity.model_revision == revision
    assert runner.target.model_revision == revision
    assert runner.identity.inference_signature == runner.target.inference_signature


def test_signature_canonicalization_and_semantic_sensitivity():
    fn = getattr(identity_module, "inference_signature", None)
    assert fn is not None, "inference compatibility needs a machine-checkable ABI signature"
    contract = dict(
        architecture={"layers": 2, "heads": 2, "hidden_dim": 8},
        parameter_shapes={"q.weight": (8, 8), "v.weight": (4, 8)},
        weight_layout="out-in",
        io_contract={"input": ["batch", 8], "output": ["batch", 4]},
        control_flow={"state": "discrete-tokens", "denoising": "euler"},
    )
    first = fn(**contract)
    reordered = copy.deepcopy(contract)
    reordered["parameter_shapes"] = dict(reversed(list(contract["parameter_shapes"].items())))
    assert fn(**reordered) == first
    for name, value in (
        ("architecture", {**contract["architecture"], "heads": 4}),
        ("parameter_shapes", {"q.weight": (16, 8), "v.weight": (4, 8)}),
        ("weight_layout", "in-out"),
        ("io_contract", {"input": ["batch", 8], "output": ["batch", 8]}),
        ("control_flow", {"state": "continuous-projection", "denoising": "euler"}),
    ):
        assert fn(**{**contract, name: value}) != first


@pytest.mark.parametrize("target", ["h100/pi0", "h100/pi05"])
def test_signature_excludes_shape_profile_and_candidate_plan(target):
    from benchmarks.targets import declare
    first = declare(target, "reference", steps=10, chunk_size=50)
    other = declare(target, "shipped", steps=5, chunk_size=32)
    assert first.identity.inference_signature == other.identity.inference_signature
    assert not first.identity.same_workload(other.identity)


def test_legacy_report_remains_explicitly_unmigrated():
    value = payload(schema_version=2, model_revision="checkpoint-a", precision="bf16")
    value.pop("inference_signature")
    value.pop("execution_variant")
    legacy = Identity.from_dict(value)
    assert legacy.model_revision == "checkpoint-a"
    assert legacy.inference_signature is None
    assert legacy.as_dict()["schema_version"] == 2
    assert not legacy.same_workload(Identity.from_dict(payload()))


@pytest.mark.parametrize("target", ["h100/pi0", "h100/pi05", "h100/lingbot_vla"])
def test_runner_rejects_incompatible_weight_schema_before_allocation(target):
    import torch
    from benchmarks.targets import declare
    from flash_vla.runtime import ModelRunner

    declaration = declare(target)
    weights = {
        name: torch.empty(shape, device="meta")
        for name, shape in declaration.graph.weight_shapes.items()
    }
    name = next(iter(weights))
    weights[name] = torch.empty((1,), device="meta")
    config = {"prompt_len": 0} if target == "h100/pi0" else {}
    with patch.object(torch, "empty", side_effect=AssertionError("allocated before ABI check")):
        with pytest.raises(ValueError, match="inference signature mismatch"):
            ModelRunner(declaration.target, weights, checkpoint_id="incompatible",
                        device="cpu", capture=False, **config)


def test_runner_rejects_same_shapes_with_incompatible_semantic_signature():
    from benchmarks.targets import declare
    from flash_vla.runtime import ModelRunner
    target = declare("h100/pi05").target
    with pytest.raises(ValueError, match="inference signature mismatch"):
        ModelRunner(target, None, checkpoint_id="other-architecture",
                    checkpoint_signature="sha256:changed-attention-semantics",
                    device="cpu", capture=False)
