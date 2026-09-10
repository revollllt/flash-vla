"""Checkpoint construction and evaluator forwarding without GPU execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from flash_vla import inference as targets
from flash_vla.models.pi05 import openpi as openpi05
from tests.test_openpi05_config import upstream
from flash_vla.models.pi05 import weights


@pytest.fixture
def checkpoint_options(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"value": torch.tensor([3.0])}, str(checkpoint))
    monkeypatch.setattr(openpi05, "resolve_config",
                        lambda checkpoint, name: SimpleNamespace(action_horizon=50, max_token_len=200))
    return dict(checkpoint=str(checkpoint), checkpoint_id="trained-a",
                checkpoint_digest="publisher-revision-a", openpi_config="training-a")


def test_real_declaration_preserves_target_and_distinct_weight_provenance(checkpoint_options, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("declaration loaded weight values")
    monkeypatch.setattr(openpi05, "build_model", forbidden)
    a = targets.declare("pi05", **checkpoint_options)
    b = targets.declare("pi05", **{**checkpoint_options, "checkpoint_id": "trained-b",
                                  "checkpoint_digest": "publisher-revision-b"})
    assert a.identity.same_workload(b.identity)
    assert a.measurement_context["weights"] == {
        "checkpoint_id": "trained-a", "checkpoint_digest": "publisher-revision-a"}
    assert b.measurement_context["weights"]["checkpoint_digest"] == "publisher-revision-b"


@pytest.mark.parametrize("missing", ["checkpoint_id", "checkpoint_digest", "openpi_config"])
def test_real_factory_requires_asset_provenance_and_config(checkpoint_options, missing):
    del checkpoint_options[missing]
    with pytest.raises(ValueError):
        targets.declare("pi05", **checkpoint_options)


def test_random_weights_cannot_be_relabelled_as_a_real_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        targets.declare("pi05", checkpoint_id="trained", checkpoint_digest="manifest")


def test_checkpoint_horizon_defaults_from_source_config_and_conflicts_fail(checkpoint_options, monkeypatch):
    monkeypatch.setattr(openpi05, "resolve_config",
                        lambda checkpoint, name: SimpleNamespace(action_horizon=15, max_token_len=200))
    declared = targets.declare("pi05", **checkpoint_options)
    assert declared.shape["chunk"] == 15
    with pytest.raises(ValueError, match="chunk_size"):
        targets.declare("pi05", **checkpoint_options, chunk_size=50)


def test_real_values_are_converted_and_folded_for_each_construction(checkpoint_options, monkeypatch):
    calls = []
    monkeypatch.setattr(openpi05, "build_model",
                        lambda checkpoint, device, **kwargs: load_file(str(checkpoint)))
    monkeypatch.setattr(openpi05, "target_checkpoint", lambda model: model)
    monkeypatch.setattr(weights, "random_checkpoint",
                        lambda **kwargs: pytest.fail("real checkpoint fell back to random weights"))
    def fold(checkpoint, steps):
        calls.append((checkpoint["value"].item(), steps))
        return {"folded": checkpoint["value"] * steps}
    monkeypatch.setattr(weights, "fold", fold)
    from flash_vla.models.pi05 import tokenize
    monkeypatch.setattr(tokenize, "Pi05Tokenizer", lambda path: object())
    def runner(target, checkpoint, *, checkpoint_id, checkpoint_digest, **kwargs):
        return SimpleNamespace(checkpoint=checkpoint, measurement_context={
            "weights": dict(checkpoint_id=checkpoint_id, checkpoint_digest=checkpoint_digest)})
    monkeypatch.setattr(targets, "ModelRunner", runner)
    a = targets.build("pi05", **checkpoint_options, device="cpu", steps=2, seed=1)
    b = targets.build("pi05", **checkpoint_options, device="cpu", steps=3, seed=9)
    assert calls == [(3.0, 2), (3.0, 3)]
    assert a.checkpoint["folded"].item() == 6
    assert b.checkpoint["folded"].item() == 9
    assert a.measurement_context["weights"] == b.measurement_context["weights"]
    assert a.measurement_context["fixture"] != b.measurement_context["fixture"]


def test_conversion_metadata_cannot_be_overridden_by_a_different_config(tmp_path, upstream):
    from tests.test_openpi05_config import Config
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"not reached")
    (tmp_path / "config.json").write_text(json.dumps({"action_horizon": 10}))
    for load in (
        lambda: openpi05.resolve_config(checkpoint, "pi05_droid"),
        lambda: openpi05.build_model(checkpoint, device="cpu", config=Config()),
    ):
        with pytest.raises(ValueError, match="config.json contradicts"):
            load()


def test_official_reference_preserves_separate_digest():
    from eval.pi05.reference import _checkpoint_digest
    assert _checkpoint_digest("weights", "model-a", "revision-17") == "revision-17"
    with pytest.raises(ValueError, match="checkpoint_digest"):
        _checkpoint_digest("weights", "model-a", None)
    assert _checkpoint_digest(None, "random/seed-0", None) == "random/seed-0"


def test_reference_options_reach_both_stages_without_relabeling(monkeypatch):
    from eval.pi05 import reference
    received = []
    def run(*args, **kwargs):
        received.append((args, kwargs))
        return {"passed": True}
    monkeypatch.setattr(reference, "run_backbone", run)
    monkeypatch.setattr(reference, "run_expert", run)
    assert reference.main([
        "--option", "checkpoint=/weights/a with spaces",
        "--option", "checkpoint_id=trained-a",
        "--option", "checkpoint_digest=revision-a",
        "--option", "openpi_config=pi05_droid",
    ]) == 0
    for args, kwargs in received:
        assert args[1] == "/weights/a with spaces"
        assert kwargs["checkpoint_id"] == "trained-a"
        assert kwargs["checkpoint_digest"] == "revision-a"
        assert kwargs["openpi_config"] == "pi05_droid"


def test_correctness_cli_forwards_checkpoint_options(monkeypatch, capsys):
    from eval import correctness
    seen = {}
    def run(*args, **kwargs):
        seen.update(kwargs)
        return {"passed": True}
    monkeypatch.setattr(correctness, "run", run)
    assert correctness.main(["--target", "pi05", "--option", "checkpoint=real-weights",
                             "--option", "checkpoint_digest=revision-a"]) == 0
    assert seen["checkpoint"] == "real-weights"
    assert seen["checkpoint_digest"] == "revision-a"


@pytest.mark.parametrize("axis", ["steps", "layers"])
def test_correctness_cli_rejects_duplicate_depth_sources(monkeypatch, axis):
    from eval import correctness
    monkeypatch.setattr(correctness, "run", lambda *a, **kw: pytest.fail("ambiguous depth ran"))
    with pytest.raises(SystemExit) as error:
        correctness.main(["--target", "pi05", "--option", axis + "=5"])
    assert error.value.code == 2


def test_conversion_precision_alias_must_match_reference(tmp_path, upstream):
    (tmp_path / "config.json").write_text(json.dumps({"precision": "float32"}))
    with pytest.raises(ValueError, match="precision"):
        openpi05.resolve_config(tmp_path, "pi05_droid")
