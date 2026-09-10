from dataclasses import replace
from unittest.mock import patch
import pytest
from flash_vla.runtime.identity import Identity
from lab.optimize import campaign
from eval.tests.test_identity_v3 import payload, context, context_value


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


@pytest.mark.parametrize("target,model,revision", [
    ("hardware/nvidia/h100/pi0", "pi0", "pi0-r1"),
    ("hardware/nvidia/h100/pi05", "pi05", "pi05-r1"),
    ("hardware/nvidia/h100/lingbot_vla", "lingbot-vla", "lingbot-vla-r1"),
])
def test_explicit_known_v2_migration_preserves_checkpoint_and_requires_revalidation(target, model, revision):
    from lab.optimize.migrate import identity_v3

    old = payload(schema_version=2, target=target, model=model,
                  model_revision="checkpoint-a", precision="bf16")
    old.pop("inference_signature")
    old.pop("execution_variant")
    report = {"identity": old, "objective": {"value": 12.0, "unit": "ms"},
              "measurement_context": {"hostname": "old-node"}}
    before = copy.deepcopy(report)
    migrated = identity_v3(report)
    assert report == before
    assert migrated["identity"]["model_revision"] == revision
    assert migrated["identity"]["inference_signature"]
    assert migrated["measurement_context"]["weights"]["checkpoint_id"] == "checkpoint-a"
    assert migrated["measurement_context"]["weights"]["checkpoint_digest"] is None
    assert migrated["measurement_context"]["hostname"] == "old-node"
    assert migrated["legacy_identity"] == old
    assert migrated["continuation"]["correctness_required"]
    assert migrated["continuation"]["reanchor_required"]
    assert migrated["objective"] == report["objective"]
    other = identity_v3({**report, "identity": {**old, "model_revision": "checkpoint-b"}})
    assert key(migrated["identity"]) == key(other["identity"])


def test_unknown_legacy_target_needs_explicit_mapping():
    from lab.optimize.migrate import identity_v3
    old = payload(schema_version=2, target="unknown-target",
                  model_revision="checkpoint-a", precision="bf16")
    with pytest.raises(ValueError, match="explicit architecture mapping"):
        identity_v3({"identity": old})
