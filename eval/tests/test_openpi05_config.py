"""CPU checks for explicit OpenPI checkpoint configuration, without model allocation."""
from dataclasses import dataclass
from types import SimpleNamespace
import sys

import pytest
import torch
from safetensors.torch import save_model

from eval.baselines import openpi05


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


def test_reference_cli_forwards_config_to_both_stages(monkeypatch):
    from eval.pi05 import reference
    received = []
    def run(*args, **kwargs):
        received.append(kwargs["openpi_config"])
        return {"passed": True}
    monkeypatch.setattr(reference, "run_backbone", run)
    monkeypatch.setattr(reference, "run_expert", run)
    assert reference.main(["--checkpoint", "weights", "--checkpoint-id", "asset-v1",
                           "--openpi-config", "pi05_droid"]) == 0
    assert received == ["pi05_droid", "pi05_droid"]


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
        observed = openpi05.reference_provenance(module.Model(), Config(), exact_rope=True)
        head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                                       text=True).strip()
        assert observed["upstream"] == dict(commit=head, dirty=False, module=spec.name)
        assert observed["config"] == asdict(Config())
        assert observed["exact_rope"] is True
        source.write_text("class Model: pass\n# local implementation edit\n")
        changed = openpi05.reference_provenance(module.Model(), Config())
        assert changed["upstream"]["commit"] == head
        assert changed["upstream"]["dirty"] is True
    finally:
        del sys.modules[spec.name]
