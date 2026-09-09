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
        self.identity = _Identity()

    def sample_inputs(self, seed):
        return {"seed": seed}


class LatencyRunTests(unittest.TestCase):
    def test_reuses_one_built_engine_for_repeated_control_leg(self):
        events = []

        def build(target, plan, **kwargs):
            events.append(("build", plan))
            return _Engine(plan)

        def measure(engine, inputs, *args, **kwargs):
            events.append(("measure", engine.plan))
            stats = {"min": 1.0, "median": 1.0, "p99": 1.0}
            return {"chunk_latency": stats, "device_latency": stats,
                    "host_time": {}, "segment_latency": {}, "overhead": stats}

        with patch.object(latency, "require_cuda"), \
             patch.object(latency.torch.cuda, "init"), \
             patch.object(latency.torch.cuda, "empty_cache"), \
             patch.object(latency, "resolve", side_effect=lambda value: value), \
             patch.object(latency, "build", side_effect=build), \
             patch.object(latency, "measure", side_effect=measure), \
             patch.object(latency, "_env", return_value={}):
            report = latency.run("test", ["a", "b", "a"], reps=1, warmup=0,
                                 attribution=False)

        self.assertEqual(events, [("build", "a"), ("build", "b"),
                                  ("measure", "a"), ("measure", "b"),
                                  ("measure", "a")])
        self.assertEqual([leg["plan"] for leg in report["legs"]], ["a", "b", "a"])


if __name__ == "__main__":
    unittest.main()
