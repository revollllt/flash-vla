from contextlib import nullcontext
import unittest
from unittest.mock import patch

from benchmarks import latency


class _Identity:
    def same_workload(self, other):
        return isinstance(other, _Identity)

    def as_dict(self):
        return {"schema_version": 2, "target": "test", "model": "test",
                "hardware": "test", "model_revision": "test-revision",
                "shape": {}, "plan": {}}


class _Engine:
    def __init__(self, plan):
        self.plan = plan
        self.device = "cuda:0"
        self.identity = _Identity()
        self.measurement_context = {
            "weights": {"checkpoint_id": "a", "checkpoint_digest": "a"},
            "fixture": {"id": "inputs", "digest": "inputs"},
        }

    def capture(self):
        raise AssertionError("measurement must use the initial capture")

    def sample_inputs(self, seed):
        return {"seed": seed}


class LatencyRunTests(unittest.TestCase):
    def setUp(self):
        worker = patch.object(latency, "_run_leg", side_effect=latency._measure_leg)
        worker.start()
        self.addCleanup(worker.stop)

    def test_builds_two_independent_versions_without_recapture(self):
        events = []

        def build(target, plan, **kwargs):
            events.append(("build", plan))
            engine = _Engine(plan)
            return engine

        def measure(engine, inputs, *args, **kwargs):
            self.assertEqual(kwargs["soak_s"], 0)
            events.append(("measure", engine.plan))
            stats = {"min": 1.0, "median": 1.0, "p99": 1.0}
            return {"chunk_latency": stats, "device_latency": stats,
                    "host_time": {}, "segment_latency": {}, "overhead": stats}

        with patch.object(latency.torch.cuda, "device", return_value=nullcontext()), \
             patch.object(latency, "require_cuda"), \
             patch.object(latency.torch.cuda, "init"), \
             patch.object(latency, "resolve", side_effect=lambda value: value), \
             patch.object(latency, "build", side_effect=build), \
             patch.object(latency, "measure", side_effect=measure), \
             patch.object(latency, "_env", side_effect=[
                 {"runtime_observation": {"clocks.sm": str(value)}}
                 for value in (1590, 1980, 1980, 1980)]):
            report = latency.run("test", ["a", "b"], reps=1, warmup=0)

        self.assertEqual(events, [("build", "a"), ("measure", "a"),
                                  ("build", "b"), ("measure", "b")])
        self.assertEqual(report["config"]["capture_policy"], "fresh-process-first-capture-per-leg")
        self.assertEqual([leg["plan"] for leg in report["legs"]], ["a", "b"])
        self.assertFalse(report["instrumented"])
        self.assertFalse(report["config"]["breakdown"])
        self.assertEqual(report["config"]["primary_statistic"], "median")
        self.assertTrue(report["deltas"]["valid"])
        self.assertIsNone(report["deltas"]["minimum_detectable_effect"])
        self.assertIsNone(report["deltas"]["control_spread_ms"])
        self.assertEqual(report["legs"][0]["runtime_observation"],
                         {"before": {"clocks.sm": "1590"}, "after": {"clocks.sm": "1980"}})


    def test_same_plan_source_variants_build_separately_and_keep_two_controls(self):
        built, measured = [], []
        def build(target, plan, **options):
            engine = _Engine(plan)
            assert "soak_s" not in options
            engine.source = options["source_checkout"]
            built.append(engine.source)
            return engine
        def measure(engine, *args, **kwargs):
            assert kwargs["soak_s"] == 10
            measured.append(engine.source)
            value = 1.0 if engine.source == "old" else 0.5
            stats = dict(min=value, median=value, p99=value)
            return dict(chunk_latency=stats)
        with patch.object(latency.torch.cuda, "device", return_value=nullcontext()), \
             patch.object(latency, "require_cuda"), \
             patch.object(latency.torch.cuda, "init"), \
             patch.object(latency, "resolve", side_effect=lambda value: value), \
             patch.object(latency, "build", side_effect=build), \
             patch.object(latency, "measure", side_effect=measure), \
             patch.object(latency, "_env", return_value={}):
            report = latency.run(
                "test", ["shipped"] * 3, reps=1, warmup=0, attribution=False,
                source_checkout="new", soak_s=10,
                leg_options=[{"source_checkout": "old"}, {}, {"source_checkout": "old"}],
            )
        self.assertEqual(report["config"]["soak_s"], 10)
        self.assertEqual(built, ["old", "new", "old"])
        self.assertEqual(measured, ["old", "new", "old"])
        self.assertEqual(report["deltas"]["control_legs"], 1)
        self.assertEqual(report["deltas"]["control_spread_ms"], 0)
        self.assertFalse(report["deltas"]["legs"][0]["same_as_reference"])

    def test_context_change_rejects_comparison(self):
        for changed in ("weights", "fixture", "environment"):
            with self.subTest(changed=changed):
                measured = []
                environment_calls = []

                def build(target, plan, **kwargs):
                    engine = _Engine(plan)
                    if plan == "b" and changed != "environment":
                        if changed == "weights":
                            engine.measurement_context["weights"]["checkpoint_digest"] = "b"
                        else:
                            engine.measurement_context["fixture"]["digest"] = "b"
                    return engine

                def environment(*args):
                    environment_calls.append(1)
                    return {"driver": "b" if changed == "environment" and len(environment_calls) > 1 else "a"}

                def measure(engine, *args, **kwargs):
                    measured.append(engine.plan)
                    stats = {"min": 1.0, "median": 1.0, "p99": 1.0}
                    return {"chunk_latency": stats, "device_latency": stats,
                            "host_time": {}, "segment_latency": {}, "overhead": stats}

                with patch.object(latency.torch.cuda, "device", return_value=nullcontext()), \
                     patch.object(latency, "require_cuda"), \
                     patch.object(latency.torch.cuda, "init"), \
                     patch.object(latency, "resolve", side_effect=lambda value: value), \
                     patch.object(latency, "build", side_effect=build), \
                     patch.object(latency, "measure", side_effect=measure), \
                     patch.object(latency, "_env", side_effect=environment):
                    with self.assertRaisesRegex(ValueError, "measurement context changed"):
                        latency.run("test", ["a", "b", "a"], reps=1, warmup=0,
                                    attribution=False)
                self.assertEqual(measured, ["a"] if changed == "environment" else ["a", "b"])


if __name__ == "__main__":
    unittest.main()


def test_raw_samples_preserve_time_order_without_affecting_statistics():
    samples = [90., 79., 85., 80.]
    result = latency._stats(samples, p99_min_reps=4)
    assert result["samples_ms"] == [90., 79., 85., 80.]
    assert (result["min"], result["median"], result["p99"]) == (79., 82.5, 90.)


def test_default_measure_times_only_complete_forward(monkeypatch):
    from types import SimpleNamespace
    calls = []
    engine = SimpleNamespace(forward=lambda **kw: calls.append(kw))
    monkeypatch.setattr(latency.torch.cuda, "synchronize", lambda: None)
    def wall(fn, reps, warmup, trace):
        fn()
        return [2., 1., 3.]
    monkeypatch.setattr(latency, "_time_wall", wall)
    result = latency.measure(engine, {"input": 1}, 3, 5, 100)
    assert list(result) == ["chunk_latency"]
    assert result["chunk_latency"]["samples_ms"] == [2., 1., 3.]
    assert calls == [{"input": 1}, {"input": 1}]


def test_breakdown_is_explicit_and_retains_legacy_metrics(monkeypatch):
    from types import SimpleNamespace
    calls = []
    engine = SimpleNamespace(forward=lambda **kw: None,
                             host=lambda name, **kw: calls.append(name),
                             replay=lambda name: calls.append(name))
    monkeypatch.setattr(latency.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(latency, "host_slots", lambda e: ["prepare"])
    monkeypatch.setattr(latency, "segments", lambda e: ["expert"])
    def timer(fn, reps, warmup, trace):
        fn()
        return [2.]
    monkeypatch.setattr(latency, "_time_wall", timer)
    monkeypatch.setattr(latency, "_time_event", timer)
    result = latency.measure(engine, {}, 1, 0, 100, breakdown=True)
    assert set(result) == {"chunk_latency", "device_latency", "host_time", "segment_latency", "overhead"}
    assert calls == ["prepare", "expert"]


def test_optional_repeated_control_still_reports_detected_drift():
    legs = [dict(plan=p, metrics={"chunk_latency": dict(min=v, median=v)})
            for p, v in [("a", 10.), ("b", 9.), ("a", 11.)]]
    result = latency._deltas(legs)
    assert result["valid"] is None
    assert result["control_spread_ms"] == 1.
    assert result["control_spread_max_ms"] is None
    assert latency._deltas(legs, control_spread_max_ms=0.1)["valid"] is False


def test_comparison_rejects_different_physical_gpus(monkeypatch):
    import pytest
    from flash_vla.environment import report_context
    engine = _Engine("a")
    context = report_context(engine, {})
    responses = iter([(dict(identity=engine.identity.as_dict(), measurement_context=context,
                            gpu_uuid=uuid), {}) for uuid in ("GPU-1", "GPU-2")])
    monkeypatch.setattr(latency, "resolve", lambda target: target)
    monkeypatch.setattr(latency, "_run_leg", lambda *a, **kw: next(responses))
    with pytest.raises(ValueError, match="same physical GPU"):
        latency.run("test", ["a", "b"])


def test_worker_failure_propagates(monkeypatch):
    import subprocess
    import pytest
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])
    monkeypatch.setattr(latency.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        latency._run_leg("test", "a")
    assert error.value.returncode == 7


def test_cli_default_and_diagnostics(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(latency, "run", lambda *a, **kw: calls.append(kw) or {})
    latency.main(["--target", "h100/pi05"])
    latency.main(["--target", "h100/pi05", "--breakdown", "--attribution"])
    assert [(c["breakdown"], c["attribution"]) for c in calls] == [(False, False), (True, True)]
