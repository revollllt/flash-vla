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


@pytest.mark.parametrize("configured", [False, True])
def test_reference_locations_come_from_machine_environment(tmp_path, configured):
    import os
    import subprocess
    import sys

    selected = {
        "OPENPI_PYTHON": str(tmp_path / "openpi runtime/python"),
        "LINGBOT_PYTHON": str(tmp_path / "lingbot runtime/python"),
        "OPENPI_PI0_CHECKPOINT": str(tmp_path / "pi0 weights"),
        "OPENPI_PI0_MODEL_REVISION": "registered-weights/v1",
    }
    env = {key: value for key, value in os.environ.items() if key not in selected}
    if configured:
        env.update(selected)
    probe = """
import json
from eval import acceptance
print(json.dumps({
    "openpi": acceptance.for_target("hardware/nvidia/h100/pi05")["baseline_python"],
    "lingbot": acceptance.for_target("hardware/nvidia/h100/lingbot_vla")["baseline_python"],
    "checkpoint": acceptance.OPENPI_PI0_CHECKPOINT,
    "checkpoint_id": acceptance.OPENPI_PI0_MODEL_REVISION,
}))
"""
    process = subprocess.run([sys.executable, "-c", probe], env=env,
                             check=True, text=True, capture_output=True)
    actual = json.loads(process.stdout)
    assert actual == (dict(zip(("openpi", "lingbot", "checkpoint", "checkpoint_id"),
                               selected.values())) if configured else
                      dict.fromkeys(("openpi", "lingbot", "checkpoint", "checkpoint_id")))


def test_missing_pi0_asset_is_unavailable_before_model_loading(monkeypatch, capsys):
    from eval.pi0 import reference

    monkeypatch.setattr(reference, "OPENPI_PI0_CHECKPOINT", None)
    def unexpected_load(*args, **kwargs):
        pytest.fail("missing asset must not start model loading")
    monkeypatch.setattr(reference, "run", unexpected_load)
    assert reference.main([]) == reference.UNAVAILABLE
    assert "OPENPI_PI0_CHECKPOINT" in capsys.readouterr().err


def test_explicit_pi0_asset_keeps_checkpoint_id(tmp_path, monkeypatch):
    from eval.pi0 import reference

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(reference, "OPENPI_PI0_CHECKPOINT", None)
    monkeypatch.setattr(reference, "OPENPI_PI0_MODEL_REVISION", None)
    received = []
    def run(path, **kwargs):
        received.append((path, kwargs))
        return {"passed": True}
    monkeypatch.setattr(reference, "run", run)
    assert reference.main(["--checkpoint", str(checkpoint)]) == reference.UNAVAILABLE
    assert not received
    assert reference.main(["--checkpoint", str(checkpoint),
                           "--checkpoint-id", "checkpoint-a"]) == 0
    assert received == [(str(checkpoint), dict(checkpoint_id="checkpoint-a", seed=0,
                                              device="cuda", plan="reference"))]


def test_lingbot_parity_reports_the_cached_oracle_producer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eval.lingbot import parity
    provenance = {"upstream_commit": "oracle-source", "adapter_revision": "oracle-producer",
                  "checkpoint_revision": "oracle-weights"}
    (tmp_path / "official-eager.json").write_text(json.dumps({"identity": provenance}))
    names = ["vision_embeddings", "prefix_k", "prefix_v", "velocity_step_0", "actions", "physical_actions"]
    values = {name: torch.ones(1) for name in names}
    monkeypatch.setattr(parity, "load_file", lambda path: values)
    engine = SimpleNamespace(identity=SimpleNamespace(as_dict=lambda: {"engine_revision": "candidate-source"}),
                             buffers=values, program=(), measurement_context={"weights": {}}, assets={},
                             sample_inputs=lambda seed: {}, stage=lambda **kw: None,
                             forward=lambda **kw: torch.ones(1))
    monkeypatch.setattr(parity, "build", lambda *a, **kw: engine)
    monkeypatch.setattr(parity, "_physical_actions", lambda *a: torch.ones(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    report = parity.run(oracle=tmp_path)
    assert report["passed"]
    assert report["reference_provenance"] == dict(
        provenance, repository="https://github.com/Robbyant/lingbot-vla.git", commit="oracle-source")
    assert report["identity"]["engine_revision"] == "candidate-source"
