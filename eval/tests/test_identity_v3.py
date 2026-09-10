"""Identity-v3 acceptance: architecture, execution policy and measurement axes."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest

from flash_vla.runtime import identity as identity_module
from flash_vla.runtime.identity import Identity


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


def test_variant_round_trip_and_workload_comparison():
    first = Identity.from_dict(payload())
    other = Identity.from_dict(payload(execution_variant={
        "quantization": {"mode": "bf16"}, "cache": {"mode": "dit_cache"},
    }))
    assert first.as_dict()["execution_variant"] == payload()["execution_variant"]
    assert "precision" not in first.as_dict()
    assert not first.same_workload(other)
    assert first.same_workload(replace(first, plan={"attention": "candidate"}))


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


def test_runner_legacy_revision_is_architecture_only():
    from benchmarks.targets import declare
    from flash_vla.runtime import ModelRunner
    target = declare("h100/pi05").target
    with pytest.warns(DeprecationWarning):
        runner = ModelRunner(target, None, model_revision=target.model_revision,
                             device="cpu", capture=False)
    assert runner.identity.model_revision == target.model_revision
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="checkpoint_id"):
            ModelRunner(target, None, model_revision="checkpoint-a",
                        device="cpu", capture=False)


def test_pi0_runtime_layout_changes_do_not_redefine_source_abi():
    from flash_vla.models.pi0 import spec
    before = identity_module.inference_signature(**spec.INFERENCE_CONTRACT)
    with patch.object(spec, "weight_shapes", return_value={"candidate-packed-table": (5, 7)}):
        assert spec.source_weight_shapes() == spec.INFERENCE_CONTRACT["parameter_shapes"]
        assert identity_module.inference_signature(**spec.INFERENCE_CONTRACT) == before
    assert not any("fused" in name or "language_embeds" in name
                   for name in spec.INFERENCE_CONTRACT["parameter_shapes"])


def test_pi0_reference_cli_passes_the_checkpoint_id(tmp_path):
    from eval.pi0 import reference
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    with patch.object(reference, "run", autospec=True, return_value={"passed": True}) as run:
        assert reference.main(["--checkpoint", str(checkpoint),
                               "--checkpoint-id", "manifest-a"]) == 0
    assert run.call_args.kwargs["checkpoint_id"] == "manifest-a"


@pytest.mark.parametrize("field", ["weights", "fixture"])
def test_profile_rejects_cross_context_delta(field):
    from types import SimpleNamespace
    from benchmarks import profile

    observed = []
    def build(target, plan, **options):
        provenance = context()
        if plan == "b":
            provenance[field]["checkpoint_digest" if field == "weights" else "digest"] = "changed"
        return SimpleNamespace(
            identity=Identity.from_dict(payload()), measurement_context=provenance,
            sample_inputs=lambda seed: {}, forward=lambda **kwargs: observed.append(plan),
            graph_contract={},
        )
    with patch.object(profile, "require_cuda"), \
         patch.object(profile.torch.cuda, "init"), \
         patch.object(profile.torch.cuda, "synchronize"), \
         patch.object(profile.torch.cuda, "empty_cache"), \
         patch.object(profile.torch.cuda, "get_device_properties",
                      return_value=SimpleNamespace(multi_processor_count=132)), \
         patch.object(profile, "resolve", side_effect=lambda name: name), \
         patch.object(profile, "build", side_effect=build), \
         patch.object(profile, "_env", return_value={}), \
         patch.object(profile, "segments", return_value=()), \
         patch.object(profile, "_deltas") as deltas:
        with pytest.raises(ValueError, match="measurement context changed"):
            profile.run("test", ["a", "b"])
    assert observed == ["a"]
    deltas.assert_not_called()


def test_new_identity_cannot_silently_emit_v2_without_signature():
    value = payload()
    value.pop("schema_version")
    value.pop("inference_signature")
    with pytest.raises(ValueError, match="requires architecture revision and inference signature"):
        Identity(**value)
