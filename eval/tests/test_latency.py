from contextlib import nullcontext
import unittest
from unittest.mock import patch

from benchmarks import latency


class _Identity:
    def same_workload(self, other):
        return isinstance(other, _Identity)

    def as_dict(self):
        return {"schema_version": 2, "target": "test"}


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
        pass

    def sample_inputs(self, seed):
        return {"seed": seed}


class LatencyRunTests(unittest.TestCase):
    def test_reuses_one_built_engine_for_repeated_control_leg(self):
        events = []

        def build(target, plan, **kwargs):
            events.append(("build", plan))
            engine = _Engine(plan)
            engine.capture = lambda: events.append(("capture", plan))
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
             patch.object(latency.torch.cuda, "empty_cache"), \
             patch.object(latency, "resolve", side_effect=lambda value: value), \
             patch.object(latency, "build", side_effect=build), \
             patch.object(latency, "measure", side_effect=measure), \
             patch.object(latency, "_env", side_effect=[
                 {"runtime_observation": {"clocks.sm": str(value)}}
                 for value in (1590, 1980, 1980, 1980, 1980, 1980)]):
            report = latency.run("test", ["a", "b", "a"], reps=1, warmup=0,
                                 attribution=False)

        self.assertEqual(events, [("build", "a"), ("build", "b"),
                                  ("capture", "a"), ("measure", "a"),
                                  ("capture", "b"), ("measure", "b"),
                                  ("capture", "a"), ("measure", "a")])
        self.assertEqual(report["config"]["capture_policy"], "fresh-graph-stream-per-leg")
        self.assertEqual([leg["plan"] for leg in report["legs"]], ["a", "b", "a"])
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
             patch.object(latency.torch.cuda, "empty_cache"), \
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
        self.assertEqual(built, ["old", "new"])
        self.assertEqual(measured, ["old", "new", "old"])
        self.assertEqual(report["deltas"]["control_legs"], 1)
        self.assertEqual(report["deltas"]["control_spread_ms"], 0)
        self.assertFalse(report["deltas"]["legs"][0]["same_as_reference"])

    def test_context_change_rejects_candidate_before_measurement(self):
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
                     patch.object(latency.torch.cuda, "empty_cache"), \
                     patch.object(latency, "resolve", side_effect=lambda value: value), \
                     patch.object(latency, "build", side_effect=build), \
                     patch.object(latency, "measure", side_effect=measure), \
                     patch.object(latency, "_env", side_effect=environment):
                    with self.assertRaisesRegex(ValueError, "measurement context changed"):
                        latency.run("test", ["a", "b", "a"], reps=1, warmup=0,
                                    attribution=False)
                self.assertEqual(measured, ["a"])


if __name__ == "__main__":
    unittest.main()


def test_raw_samples_preserve_time_order_without_affecting_statistics():
    samples = [90., 79., 85., 80.]
    result = latency._stats(samples, p99_min_reps=4)
    assert result["samples_ms"] == [90., 79., 85., 80.]
    assert (result["min"], result["median"], result["p99"]) == (79., 82.5, 90.)
