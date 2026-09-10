"""Header-based compatibility uses actual files; tiny models are software evidence only."""
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file, save_model

from eval.baselines import openpi05
from eval.tests.test_openpi05_config import Config
from flash_vla.models.pi05 import spec


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 2))
            self.alias = self.weight
            self.bias = torch.nn.Parameter(torch.empty(2))
    model = Tiny(Config())
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.fill_(0.5)
    path = tmp_path / "model.safetensors"
    save_model(model, str(path))
    monkeypatch.setitem(sys.modules, "openpi.models_pytorch.pi0_pytorch",
                        SimpleNamespace(PI0Pytorch=Tiny))
    def normalized(model):
        assert model.weight.device.type == "meta"
        return {"normalized": model.weight}
    monkeypatch.setattr(openpi05, "target_checkpoint", normalized)
    monkeypatch.setattr(spec, "weight_shapes", lambda: {"normalized": (2, 2)})
    contract = deepcopy(spec.INFERENCE_CONTRACT)
    contract["parameter_shapes"] = {"normalized": (2, 2)}
    monkeypatch.setattr(spec, "INFERENCE_CONTRACT", contract)
    return path


def test_actual_header_and_tied_parameter_alias_produce_contract(checkpoint):
    result = openpi05.checkpoint_contract(checkpoint, Config())
    assert result["contract"]["parameter_shapes"] == {"normalized": (2, 2)}
    assert result["checkpoint_schema"]["stored_tensors"] == 2
    assert len(result["checkpoint_schema"]["aliases"]) == 1


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "false_alias"])
def test_invalid_file_schema_is_rejected(checkpoint, change):
    tensors = load_file(str(checkpoint))
    with safe_open(str(checkpoint), framework="pt") as f:
        metadata = f.metadata()
    if change == "missing":
        tensors.pop("bias")
    elif change == "extra":
        tensors["unrecognized"] = torch.zeros(1)
    elif change == "shape":
        source = next(iter(metadata.values()))
        tensors[source] = torch.zeros(3, 2)
    else:
        alias = next(iter(metadata))
        metadata[alias] = "bias"
    save_file(tensors, str(checkpoint), metadata=metadata)
    with pytest.raises(ValueError):
        openpi05.checkpoint_contract(checkpoint, Config())


def test_unsupported_state_semantics_fail_before_header_check(checkpoint):
    with pytest.raises(ValueError, match="discrete_state_input"):
        openpi05.checkpoint_contract(checkpoint, Config(discrete_state_input=False))


def test_normalized_adapter_layout_must_match_target(checkpoint, monkeypatch):
    monkeypatch.setattr(openpi05, "target_checkpoint",
                        lambda model: {"normalized": torch.empty(3, 2, device="meta")})
    with pytest.raises(ValueError, match="weight ABI"):
        openpi05.checkpoint_contract(checkpoint, Config())


@pytest.fixture
def producer_report(monkeypatch, tmp_path):
    from eval.pi05 import compatibility
    monkeypatch.setattr(openpi05, "resolve_config", lambda *args: Config())
    monkeypatch.setattr(openpi05, "checkpoint_contract", lambda *args: {
        "contract": deepcopy(spec.INFERENCE_CONTRACT),
        "checkpoint_schema": {"stored_tensors": 2, "aliases": {}, "normalized_tensors": 1},
    })
    return compatibility.run(str(tmp_path / "weights"), checkpoint_id="fine-tuned-a",
                             checkpoint_digest="publisher-a", openpi_config="explicit-model-config")
