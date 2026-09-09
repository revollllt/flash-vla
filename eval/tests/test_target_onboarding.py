import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lab import onboarding
from lab.optimize import store


def spec():
    return {
        "version": 1,
        "target": "hardware/nvidia/h100/unseen_vla",
        "upstream": {"repository": "https://example.invalid/unseen-vla.git",
                     "commit": "1" * 40},
        "checkpoint": {"id": "unseen-vla/checkpoint-v1",
                       "source": "/models/unseen-vla"},
        "model_revision": "unseen-vla/model-v1",
        "hardware": {"name": "H100 SXM5 80GB",
                     "deployment": "one GPU, clocks unlocked"},
        "precision": "bf16",
        "shape_profile": {"batch": 1, "views": 2, "steps": 10,
                          "image": [224, 224], "action_chunk": 32},
        "performance_objective": {"metric": "chunk_latency", "direction": "minimize",
                                  "unit": "ms"},
        "correctness_requirements": ["official output parity", "deterministic replay"],
        "benchmark_protocol": {"id": "latency-v2", "reps": 100, "warmup": 5},
        "deployment_bound": {"metric": "chunk_latency.p99_minus_min",
                             "comparator": "<=", "value": 0.5, "unit": "ms"},
        "optimization_budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "assumptions": ["views and image size come from the official deployment config"],
    }


def passed(evidence="artifact.json"):
    return {"status": "passed", "evidence": evidence}


def upstream_freeze(value):
    return {"status": "passed", "repository": value["upstream"]["repository"],
            "commit": value["upstream"]["commit"],
            "checkpoint": value["checkpoint"]["id"],
            "model_revision": value["model_revision"]}


def official_reference():
    checks = {name: passed(f"official/{name}.json")
              for name in onboarding.OFFICIAL_REFERENCE_CHECKS}
    checks["upstream_official_optimized"] = {
        "status": "unavailable", "reason": "upstream exposes eager only"
    }
    return {"status": "passed", "checks": checks}


def inventory():
    return [
        {"name": "vision attention", "category": "existing runtime op",
         "evidence": "upstream/model.py:10", "action": "bind existing attention",
         "risks": []},
        {"name": "action packing", "category": "Target-local composition",
         "evidence": "upstream/model.py:40", "action": "declare graph nodes",
         "risks": ["dynamic shape"]},
        {"name": "scheduler sync", "category": "capture blocker",
         "evidence": "upstream/scheduler.py:12", "action": "stage host value before capture",
         "risks": ["host/device sync", "graph capture blocker"]},
    ]


def target_bring_up(runtime_required=False):
    checks = {name: passed(f"source/{name}") for name in onboarding.TARGET_BRING_UP_CHECKS}
    runtime = {"required": runtime_required,
               "reason": ("legal graph cannot express an ordered host slot" if runtime_required
                          else "existing vocabulary covers the Target")}
    if runtime_required:
        runtime["unexpressible_invariant"] = "ordered immutable host slot"
        runtime["checks"] = {name: passed(f"tests/{name}")
                             for name in onboarding.RUNTIME_MODIFICATION_CHECKS}
    return {"status": "passed", "checks": checks, "runtime_modification": runtime}


def ladder(names, unavailable=()):
    rows = []
    for name in names:
        rows.append({"name": name, "status": "unavailable", "reason": "not offered upstream"}
                    if name in unavailable else
                    {"name": name, "status": "passed", "evidence": f"reports/{name}.json"})
    return {"status": "passed", "ladder": rows}


def floor_profile():
    return {"status": "passed", "profile": "reports/profile.json", "call_sites": [{
        "name": "vision attention",
        "geometry": "b1-h16-q196-k196-d128",
        "minimal_bytes": 3211264,
        "flops": 157351936,
        "measured_ceiling": {"geometry": "b1-h16-q196-k196-d128",
                             "metric": "attention_tflops", "value": 412.0,
                             "unit": "TFLOP/s", "evidence": "measured/vision-attn.json"},
        "recoverable_latency_ms": 0.18,
    }]}


class TargetOnboardingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.directory = self.base / "onboarding"
        self.spec = spec()
        store.write(self.directory / "onboarding.json",
                    {"version": 1, "created": 0, "spec": self.spec})
        onboarding.record(self.directory, "requirement_freeze", {"status": "passed", "spec": "onboarding.json"})

    def advance_to_compatibility(self):
        onboarding.record(self.directory, "upstream_freeze", upstream_freeze(self.spec))
        onboarding.record(self.directory, "official_reference", official_reference())

    def test_spec_contains_every_human_contract_field_and_assumptions(self):
        schema = {"required": list(self.spec)}
        self.assertEqual(set(schema["required"]), set(self.spec))
        self.assertEqual(onboarding.validate_spec(self.spec), self.spec)
        for key in schema["required"]:
            broken = dict(self.spec)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises((ValueError, AttributeError)):
                onboarding.validate_spec(broken)

    def test_requirement_and_upstream_identity_are_frozen(self):
        state = onboarding.rebuild(self.directory)
        self.assertEqual(state["completed_stages"], ["requirement_freeze"])
        self.assertEqual(state["current_stage"], "upstream_freeze")
        (self.directory / "state.json").unlink()
        evidence = upstream_freeze(self.spec)
        evidence["commit"] = "2" * 40
        with self.assertRaisesRegex(ValueError, "does not match"):
            onboarding.record(self.directory, "upstream_freeze", evidence)

    def test_official_oracle_requires_loaders_fixtures_seed_outputs_and_environment(self):
        onboarding.record(self.directory, "upstream_freeze", upstream_freeze(self.spec))
        evidence = official_reference()
        evidence["checks"].pop("reference_outputs")
        with self.assertRaisesRegex(ValueError, "reference_outputs"):
            onboarding.record(self.directory, "official_reference", evidence)
        evidence = official_reference()
        evidence["checks"]["tokenizer"] = {"status": "unavailable", "reason": "missing"}
        with self.assertRaisesRegex(ValueError, "tokenizer.status"):
            onboarding.record(self.directory, "official_reference", evidence)

    def test_compatibility_report_answers_all_nine_questions_in_json_and_markdown(self):
        self.advance_to_compatibility()
        report = onboarding.write_compatibility_report(self.directory, inventory())
        self.assertEqual(set(report["questions"]), {
            "existing_runtime_vocabulary", "reusable_components", "new_target_local_ops",
            "runtime_primitive_required", "dynamic_shape", "post_freeze_allocation",
            "host_device_sync", "graph_capture_blocker", "data_dependent_control_flow",
        })
        self.assertEqual(report["summary"]["by_category"]["capture blocker"], 1)
        self.assertEqual(report["questions"]["dynamic_shape"], ["action packing"])
        self.assertTrue((self.directory / "compatibility-report.json").is_file())
        markdown = (self.directory / "compatibility-report.md").read_text()
        self.assertIn("Existing runtime vocabulary coverage", markdown)
        self.assertIn("scheduler sync", markdown)

    def test_runtime_modification_requires_the_five_boundary_checks(self):
        self.advance_to_compatibility()
        onboarding.write_compatibility_report(self.directory, inventory())
        evidence = target_bring_up(runtime_required=True)
        evidence["runtime_modification"]["checks"].pop("pi05_smoke")
        with self.assertRaisesRegex(ValueError, "pi05_smoke"):
            onboarding.record(self.directory, "target_bring_up", evidence)
        onboarding.record(self.directory, "target_bring_up", target_bring_up(runtime_required=True))

    def test_correctness_failure_is_retained_and_blocks_performance_stages(self):
        self.advance_to_compatibility()
        onboarding.write_compatibility_report(self.directory, inventory())
        onboarding.record(self.directory, "target_bring_up", target_bring_up())
        state = onboarding.record(self.directory, "correctness_ladder",
                                  {"status": "failed", "reason": "stage parity mismatch"})
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(state["current_stage"], "correctness_ladder")
        with self.assertRaisesRegex(RuntimeError, "correctness_ladder"):
            onboarding.record(self.directory, "baseline_ladder",
                              ladder(onboarding.BASELINE_LADDER))
        state = onboarding.record(self.directory, "correctness_ladder",
                                  ladder(onboarding.CORRECTNESS_LADDER))
        self.assertEqual(state["attempts"]["correctness_ladder"], 2)
        first = store.read(self.directory / "stages/correctness_ladder/attempt-001.json")
        self.assertEqual(first["reason"], "stage parity mismatch")

    def test_ladders_have_fixed_order_and_all_four_baselines(self):
        self.advance_to_compatibility()
        onboarding.write_compatibility_report(self.directory, inventory())
        onboarding.record(self.directory, "target_bring_up", target_bring_up())
        wrong = ladder(onboarding.CORRECTNESS_LADDER)
        wrong["ladder"][0], wrong["ladder"][1] = wrong["ladder"][1], wrong["ladder"][0]
        with self.assertRaisesRegex(ValueError, "fixed order"):
            onboarding.record(self.directory, "correctness_ladder", wrong)
        onboarding.record(self.directory, "correctness_ladder",
                          ladder(onboarding.CORRECTNESS_LADDER))
        onboarding.record(self.directory, "baseline_ladder",
                          ladder(onboarding.BASELINE_LADDER,
                                 {"upstream official optimized/compile"}))

    def test_floor_ceiling_must_match_each_call_site_geometry(self):
        self._advance_to_floor()
        evidence = floor_profile()
        evidence["call_sites"][0]["measured_ceiling"]["geometry"] = "pi-gemma-geometry"
        with self.assertRaisesRegex(ValueError, "geometry must match"):
            onboarding.record(self.directory, "floor_profile", evidence)

    def _advance_to_floor(self):
        self.advance_to_compatibility()
        onboarding.write_compatibility_report(self.directory, inventory())
        onboarding.record(self.directory, "target_bring_up", target_bring_up())
        onboarding.record(self.directory, "correctness_ladder",
                          ladder(onboarding.CORRECTNESS_LADDER))
        onboarding.record(self.directory, "baseline_ladder",
                          ladder(onboarding.BASELINE_LADDER,
                                 {"upstream official optimized/compile"}))

    def test_unseen_model_reaches_campaign_handoff_and_state_rebuilds(self):
        self._advance_to_floor()
        onboarding.record(self.directory, "floor_profile", floor_profile())
        campaign = {
            "status": "passed", "directory": "artifacts/optimization/unseen-vla",
            "objective": "chunk_latency", "protocol": "latency-v2",
            "fixture": "unseen-vla-perf-v1", "state": "BASELINED",
        }
        wrong = dict(campaign, protocol="different-protocol")
        with self.assertRaisesRegex(ValueError, "does not match"):
            onboarding.record(self.directory, "campaign_creation", wrong)
        state = onboarding.record(self.directory, "campaign_creation", campaign)
        self.assertEqual(state["status"], "LEGACY_REVALIDATION_REQUIRED")
        (self.directory / "state.json").unlink()
        self.assertEqual(onboarding.validate(self.directory), state)
        output = subprocess.check_output(
            [sys.executable, "-m", "lab.onboarding", "status", str(self.directory)], text=True)
        self.assertEqual(json.loads(output)["next_action"],
                         "create an explicit v2 spec and revalidate legacy onboarding")


if __name__ == "__main__":
    unittest.main()
