"""Coarse profiling keeps real forward order; detailed work stays in the selected segment."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.profiling import model as profile
from tests.test_latency import _Identity


@pytest.fixture
def engine(monkeypatch):
    calls = []
    engine = SimpleNamespace(identity=_Identity(), device="cuda:0",
                             measurement_context={"weights": {"checkpoint_id": "a"},
                                                  "fixture": {"id": "inputs"}}, graph_contract={},
                             sample_inputs=lambda seed: {},
                             replay=lambda name: calls.append(name),
                             host=lambda name, **kw: calls.append(name), calls=calls)
    def forward(**kw):
        engine.replay("vision")
        engine.host("prepare")
        engine.replay("expert")
    engine.forward = forward
    monkeypatch.setattr(profile, "require_cuda", lambda: None)
    monkeypatch.setattr(profile.torch.cuda, "init", lambda: None)
    monkeypatch.setattr(profile.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profile.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(profile.torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(multi_processor_count=132))
    monkeypatch.setattr(profile, "resolve", lambda target: target)
    monkeypatch.setattr(profile, "build", lambda *a, **kw: engine)
    monkeypatch.setattr(profile, "_env", lambda *a: {})
    monkeypatch.setattr(profile, "segments", lambda e: ["vision", "expert"])
    return engine


def test_detailed_profile_only_attributes_selected_segment(engine, monkeypatch):
    selected = []
    def attribute(e, segment, *args):
        selected.append(segment)
        return dict(kernel_names=[], valid=True)
    monkeypatch.setattr(profile, "attribute", attribute)
    report = profile.run("test", ["a"], segment="expert")
    assert selected == ["expert"]
    assert list(report["legs"][0]["segments"]) == ["expert"]
    assert report["legs"][0]["contract"]["passed"] is None


def test_unknown_segment_fails(engine):
    with pytest.raises(ValueError, match="unknown segment"):
        profile.run("test", ["a"], segment="missing")


def test_overview_profiles_forward_and_restores_methods(engine, monkeypatch, tmp_path):
    replay, host = engine.replay, engine.host
    monkeypatch.setattr(profile, "profile", lambda **kw: nullcontext())
    monkeypatch.setattr(profile, "_trace_events", lambda *a: [
        dict(cat="kernel", ts=10, dur=2), dict(cat="kernel", ts=15, dur=3)])
    report = profile.overview("test", "a", trace_dir=str(tmp_path))
    assert engine.calls == ["vision", "prepare", "expert"] * 6
    assert engine.replay is replay and engine.host is host
    assert report["diagnostic_only"] is True
    assert report["gpu_activity_us"] == 5
    assert report["gpu_span_us"] == 8
    assert report["gpu_gaps_us"] == 3
