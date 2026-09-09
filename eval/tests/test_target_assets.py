"""CPU asset isolation and filesystem relocation without model/GPU loading."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from benchmarks import targets
from flash_vla.hardware.nvidia.h100.lingbot_vla import TARGET
from flash_vla.runtime import ModelRunner


def fixture(path, value):
    shape = TARGET.shape(TARGET.configure())
    tensors = {item.name: torch.full(item.dims(shape), value, dtype=item.dtype)
               for item in TARGET.INPUTS}
    save_file(tensors, str(path))
    return path


def test_interleaved_runners_keep_their_fixture_and_identity(tmp_path, monkeypatch):
    a = fixture(tmp_path / "a.safetensors", 1)
    b = fixture(tmp_path / "b.safetensors", 2)
    options = dict(capture=False, device="cpu")
    assets = {"fixture": a}
    first = ModelRunner(TARGET, assets=assets, **options)
    second = ModelRunner(TARGET, assets={"fixture": b}, **options)
    assets["fixture"] = b
    monkeypatch.setenv("LINGBOT_FIXTURE", str(b))
    assert first.identity.same_workload(second.identity)
    assert first.sample_inputs(42)["state"].eq(1).all()
    assert second.sample_inputs(42)["state"].eq(2).all()
    assert first.sample_inputs(42)["state"].eq(1).all()
    with pytest.raises(TypeError):
        first.assets["fixture"] = b


def test_factory_resolves_logical_assets_per_layout_without_mutating_environment(tmp_path, monkeypatch):
    from flash_vla.models import lingbot
    loaded = []
    def load(path):
        loaded.append(Path(path))
        return {}
    def declared(target, checkpoint, **kwargs):
        kwargs["capture"] = False
        return ModelRunner(target, None, **kwargs)
    monkeypatch.setattr(lingbot, "load_checkpoint", load)
    monkeypatch.setattr(targets, "ModelRunner", declared)
    monkeypatch.setenv("LINGBOT_CHECKPOINT", "unrelated-existing-checkpoint")
    monkeypatch.setenv("LINGBOT_FIXTURE", "unrelated-existing-fixture")
    runners = []
    for name in ("machine-a", "machine-b"):
        directory = tmp_path / name
        directory.mkdir()
        fixture(directory / "fixture.safetensors", 1)
        mapping = {
            TARGET.ASSETS["checkpoint"]: "weights",
            TARGET.ASSETS["fixture"]: "fixture.safetensors",
            TARGET.ASSETS["upstream"]: "upstream",
            TARGET.ASSETS["qwen"]: "qwen",
        }
        path = directory / "assets.json"
        path.write_text(json.dumps(mapping))
        runners.append(targets.build("lingbot_vla", device="cpu", asset_config=str(path)))
    assert loaded == [tmp_path / name / "weights" for name in ("machine-a", "machine-b")]
    assert runners[0].identity.same_workload(runners[1].identity)
    assert runners[0].measurement_context == runners[1].measurement_context
    torch.testing.assert_close(runners[0].sample_inputs(42)["state"], runners[1].sample_inputs(42)["state"])
    import os
    assert os.environ["LINGBOT_CHECKPOINT"] == "unrelated-existing-checkpoint"
    assert os.environ["LINGBOT_FIXTURE"] == "unrelated-existing-fixture"


def test_explicit_checkpoint_requires_provenance_before_loading():
    with pytest.raises(ValueError, match="checkpoint_id"):
        targets.declare("lingbot_vla", checkpoint="/different/weights")


def test_declaration_does_not_require_machine_asset_configuration(monkeypatch):
    monkeypatch.delenv("FLASH_VLA_ASSETS", raising=False)
    assert targets.declare("lingbot_vla").identity.model == "lingbot-vla"


def test_backend_delayed_load_uses_its_runner_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from flash_vla.runtime.runner import Scratch
    from flash_vla.hardware.nvidia.h100.lingbot_vla.backends import upstream

    class Visual:
        def __init__(self, value):
            self.value = value

        def preprcess_grid_thw(self, grid):
            return (torch.zeros(1),) * 4

        def __call__(self, pixels, **kwargs):
            return torch.full((3, 64, 2048), self.value)

    loaded = []
    def policy(weights, layers, cache_rope_frequency, assets):
        loaded.append(Path(assets["checkpoint"]))
        visual = Visual(int(Path(assets["checkpoint"]).read_text()))
        return SimpleNamespace(qwenvl_with_expert=SimpleNamespace(qwenvl=SimpleNamespace(visual=visual)))

    monkeypatch.setattr(upstream, "_build_policy", policy)
    paths = [tmp_path / "a", tmp_path / "b"]
    wrappers = []
    for index, path in enumerate(paths, 1):
        path.write_text(str(index))
        scratch = Scratch(torch.device("cpu"), assets={"checkpoint": path})
        wrappers.append(upstream.make_wrappers(scratch)["lingbot_vision"])
    monkeypatch.setenv("LINGBOT_CHECKPOINT", str(paths[1]))
    pixels, output = torch.zeros(1), torch.zeros(3, 64, 2048)
    for index in (0, 1, 0):
        wrappers[index](pixels, output, 36)
        assert output.eq(index + 1).all()
    assert loaded == paths


def test_official_parity_cli_forwards_machine_asset_config(monkeypatch, capsys):
    from eval.lingbot import parity
    received = []
    def run(*args, **kwargs):
        received.append(kwargs)
        return {"passed": True}
    monkeypatch.setattr(parity, "run", run)
    assert parity.main(["--option", "asset_config=/new layout/assets.json"]) == 0
    assert received == [{"asset_config": "/new layout/assets.json"}]


@pytest.mark.parametrize("name", ["pi0", "pi05"])
def test_random_input_targets_accept_the_shared_asset_argument(name):
    from dataclasses import asdict

    declared = targets.declare(name)
    with_assets = ModelRunner(declared.target, capture=False, device="cpu",
                              assets={"unused": "/not-a-model-input"},
                              **asdict(declared.config))
    left, right = declared.sample_inputs(3), with_assets.sample_inputs(3)
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key])
