"""Checkpoint configuration, loading and reference argument forwarding."""
from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file, save_model

from flash_vla import inference as targets
from flash_vla.models.pi05 import openpi as openpi05, weights
from eval.pi05 import official as official_pi05


@dataclass(frozen=True)
class Config:
    dtype: str = "bfloat16"
    pi05: bool = True
    discrete_state_input: bool = True
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pytorch_compile_mode: str | None = "max-autotune"


@pytest.fixture
def upstream(monkeypatch):
    configs = {
        "pi05_droid": Config(action_horizon=15),
        "pi05_libero": Config(discrete_state_input=False, action_horizon=10),
    }
    monkeypatch.setitem(sys.modules, "openpi.models.pi0_config",
                        SimpleNamespace(Pi0Config=Config))
    monkeypatch.setitem(sys.modules, "openpi.training.config",
                        SimpleNamespace(get_config=lambda name: SimpleNamespace(model=configs[name])))
    return configs


def test_real_checkpoint_requires_explicit_config_before_openpi_import():
    with pytest.raises(ValueError, match="explicit.*config"):
        openpi05.build_model("model.safetensors", device="cpu")


def test_registered_config_preserves_horizon_and_disables_compile(upstream):
    config = openpi05.resolve_config("model.safetensors", "pi05_droid")
    assert config.action_horizon == 15
    assert config.pytorch_compile_mode is None
    assert upstream["pi05_droid"].pytorch_compile_mode == "max-autotune"


def test_state_free_checkpoint_rejected_before_weight_loading(upstream):
    with pytest.raises(ValueError, match="discrete_state_input"):
        openpi05.resolve_config("model.safetensors", "pi05_libero")


@pytest.mark.parametrize("change", [
    {"pi05": False}, {"action_dim": 7}, {"max_token_len": 48},
    {"paligemma_variant": "gemma_2b_lora"},
    {"action_expert_variant": "gemma_300m_lora"},
])
def test_unsupported_adapter_contract_rejected_before_model_allocation(change):
    with pytest.raises(ValueError, match="incompatible"):
        openpi05.build_model("model.safetensors", device="cpu", config=Config(**change))


def test_real_safetensors_values_loaded_with_supplied_config(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Linear):
        def __init__(self, config):
            super().__init__(2, 2)
            self.config = config

    monkeypatch.setitem(sys.modules, "openpi.models_pytorch.pi0_pytorch",
                        SimpleNamespace(PI0Pytorch=TinyModel))
    config = Config(action_horizon=15)
    source = TinyModel(config)
    with torch.no_grad():
        source.weight.fill_(0.75)
        source.bias.fill_(-0.25)
    save_model(source, str(tmp_path / "model.safetensors"))
    loaded = openpi05.build_model(tmp_path, device="cpu", config=config, exact_rope=False)
    assert loaded.config.action_horizon == config.action_horizon
    assert loaded.config.pytorch_compile_mode is None
    assert config.pytorch_compile_mode == "max-autotune"
    assert not loaded.training
    torch.testing.assert_close(loaded.weight, source.weight)
    torch.testing.assert_close(loaded.bias, source.bias)


def test_random_model_config_keeps_existing_shape(upstream):
    config = openpi05.resolve_config(None, None)
    assert config.pi05 and config.discrete_state_input
    assert config.action_horizon == 50


def test_reference_provenance_tracks_loaded_module_and_dirty_state(tmp_path):
    import importlib.util
    import subprocess
    from dataclasses import asdict

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "upstream_probe.py"
    source.write_text("class Model: pass\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", source.name], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.org", "commit", "-qm", "upstream"], check=True)
    spec = importlib.util.spec_from_file_location("upstream_provenance_probe", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        observed = official_pi05.reference_provenance(module.Model(), Config(), exact_rope=True)
        head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                                       text=True).strip()
        assert observed["upstream"] == dict(commit=head, dirty=False, module=spec.name)
        assert observed["repository"] == "https://github.com/Physical-Intelligence/openpi.git"
        assert observed["commit"] == head
        assert observed["config"] == asdict(Config())
        assert observed["exact_rope"] is True
        source.write_text("class Model: pass\n# local implementation edit\n")
        changed = official_pi05.reference_provenance(module.Model(), Config())
        assert changed["upstream"]["commit"] == head
        assert changed["upstream"]["dirty"] is True
    finally:
        del sys.modules[spec.name]


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


@pytest.mark.parametrize("arguments", [
    ["--checkpoint", "/weights/a with spaces", "--checkpoint-id", "trained-a",
     "--checkpoint-digest", "revision-a", "--openpi-config", "pi05_droid"],
    ["--option", "checkpoint=/weights/a with spaces", "--option", "checkpoint_id=trained-a",
     "--option", "checkpoint_digest=revision-a", "--option", "openpi_config=pi05_droid"],
])
def test_reference_options_reach_both_stages_without_relabeling(monkeypatch, arguments):
    from eval.pi05 import reference
    received = []
    def run(*args, **kwargs):
        received.append((args, kwargs))
        return {"passed": True}
    monkeypatch.setattr(reference, "run_backbone", run)
    monkeypatch.setattr(reference, "run_expert", run)
    assert reference.main(arguments) == 0
    assert len(received) == 2
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
