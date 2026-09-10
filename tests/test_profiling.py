"""Coarse profiling keeps real forward order; detailed work stays in the selected segment."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import unittest
from tools.profiling.timeline import intervals, occurrences, region_occurrences
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



def event(start, end, site='a', inv=0, stream=7):
    return dict(start_us=start, end_us=end, call_site=site, invocation_id=inv, stream=stream)


class TimelineTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(intervals([event(0, 10), event(5, 15)]),
                         dict(sum_kernel_duration_us=20, interval_union_us=15, region_makespan_us=15))

    def test_serial_and_gap(self):
        stats = intervals([event(0, 10), event(20, 30)])
        self.assertEqual(stats['interval_union_us'], 20)
        self.assertEqual(stats['region_makespan_us'], 30)

    def test_empty_and_reversed(self):
        self.assertEqual(intervals([])['region_makespan_us'], 0)
        with self.assertRaises(ValueError):
            intervals([event(10, 0)])

    def test_repeated_groups_are_not_whole_graph_span(self):
        events = [event(0, 10, 'a', 0), event(5, 15, 'b', 1),
                  event(100, 110, 'a', 2), event(105, 115, 'b', 3)]
        result = region_occurrences(events, {'a', 'b'})
        self.assertEqual(result['sum_occurrence_makespan_us'], 30)
        self.assertEqual(len(occurrences(events)), 4)

    def test_multistream_unknown_group_but_valid_union(self):
        events = [event(0, 10, 'a', 0, 1), event(5, 15, 'b', 1, 2)]
        self.assertEqual(intervals(events)['interval_union_us'], 15)
        self.assertEqual(region_occurrences(events, {'a', 'b'})['status'], 'unsupported')

    def test_unattributed(self):
        events = [event(0, 10, None, None)]
        self.assertEqual(occurrences(events), [])
        self.assertEqual(region_occurrences(events, {'a'})['status'], 'unsupported')

    def test_incomplete_or_ambiguous_groups(self):
        for events in ([event(0, 10)], [event(0, 10, 'a', 0), event(10, 20, 'a', 1), event(20, 30, 'b', 2)]):
            self.assertEqual(region_occurrences(events, {'a', 'b'})['status'], 'unsupported')

    def test_multiple_kernels_one_invocation(self):
        events = [event(0, 3), event(5, 7), event(8, 10, 'b', 1)]
        self.assertEqual(len(occurrences(events)), 2)
        self.assertEqual(region_occurrences(events, {'a', 'b'})['sum_occurrence_makespan_us'], 10)
