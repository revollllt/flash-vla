import copy
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flash_vla.bench import KernelResult, write_csv
from flash_vla.runtime.identity import IDENTITY_SCHEMA_VERSION, Identity, git_revision


def identity(**overrides):
    values = {
        "target": "hardware/nvidia/h100/pi05",
        "hardware": "h100-sxm5-80gb",
        "model": "pi05",
        "model_revision": "openpi@immutable:pi05",
        "shape": {"chunk": 50, "steps": 10},
        "plan": {"site": "reference"},
        "precision": "bf16",
        "engine_revision": "aaaaaaa",
    }
    values.update(overrides)
    return Identity(**values)


class WorkloadIdentityTests(unittest.TestCase):
    def test_different_plan_is_the_same_workload(self):
        self.assertTrue(identity().same_workload(identity(plan={"site": "candidate"})))

    def test_different_engine_revision_is_the_same_workload(self):
        self.assertTrue(identity().same_workload(identity(engine_revision="bbbbbbb")))

    def test_different_model_revision_is_not_the_same_workload(self):
        self.assertFalse(identity().same_workload(identity(model_revision="openpi@other:pi05")))

    def test_different_hardware_is_not_the_same_workload(self):
        self.assertFalse(identity().same_workload(identity(hardware="h100-pcie-80gb")))

    def test_different_chunk_is_not_the_same_workload(self):
        self.assertFalse(identity().same_workload(
            identity(shape={"chunk": 32, "steps": 10})))

    def test_different_denoise_steps_is_not_the_same_workload(self):
        self.assertFalse(identity().same_workload(
            identity(shape={"chunk": 50, "steps": 5})))

    def test_different_precision_is_not_the_same_workload(self):
        other = copy.copy(identity())
        object.__setattr__(other, "precision", "fp16")
        self.assertFalse(identity().same_workload(other))


class IdentitySerializationTests(unittest.TestCase):
    def test_v2_uses_unambiguous_revision_fields(self):
        payload = identity().as_dict()
        self.assertEqual(payload["schema_version"], IDENTITY_SCHEMA_VERSION)
        self.assertEqual(payload["model_revision"], "openpi@immutable:pi05")
        self.assertEqual(payload["engine_revision"], "aaaaaaa")
        self.assertNotIn("revision", payload)
        self.assertEqual(Identity.from_dict(payload), identity())

    def test_v1_report_remains_readable_but_is_not_comparable(self):
        payload = identity().as_dict()
        payload.pop("schema_version")
        payload.pop("model_revision")
        payload["revision"] = payload.pop("engine_revision")
        legacy = Identity.from_dict(payload)
        self.assertIsNone(legacy.model_revision)
        self.assertEqual(legacy.engine_revision, "aaaaaaa")
        self.assertFalse(legacy.same_workload(legacy))

    def test_v1_ignores_fields_that_only_exist_in_v2(self):
        payload = identity().as_dict()
        payload.pop("schema_version")
        payload["revision"] = payload.pop("engine_revision")
        legacy = Identity.from_dict(payload)
        self.assertIsNone(legacy.model_revision)
        self.assertFalse(legacy.same_workload(legacy))

    def test_unknown_schema_version_is_rejected(self):
        payload = identity().as_dict()
        payload["schema_version"] = 3
        with self.assertRaises(ValueError):
            Identity.from_dict(payload)

    def test_mutable_model_revision_labels_are_rejected(self):
        for value in ("", " ", " latest ", "latest", "main", "current", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                identity(model_revision=value)


class ProducerIdentityTests(unittest.TestCase):
    def test_random_checkpoint_seed_changes_model_revision(self):
        from benchmarks.targets import declare
        from eval.pi05.reference import _checkpoint_revision

        for target in ("h100/pi0", "h100/pi05"):
            with self.subTest(target=target):
                first = declare(target, seed=0).identity
                repeat = declare(target, "reference", seed=0).identity
                other = declare(target, seed=1).identity
                self.assertTrue(first.same_workload(repeat))
                self.assertFalse(first.same_workload(other))
        self.assertNotEqual(declare("h100/pi05", seed=0).identity.model_revision,
                            _checkpoint_revision(None, None, 0))

    def test_kernel_csv_carries_complete_identity(self):
        expected = identity().as_dict()
        result = KernelResult("site", [1.0], identity=expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.csv"
            write_csv(str(path), [result])
            with path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["identity"]), expected)

    def test_kernel_csv_rejects_legacy_header_before_append(self):
        result = KernelResult("site", [1.0], identity=identity().as_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.csv"
            path.write_text("label,median_ms,min_ms,mean_ms,std_ms,p99_ms,"
                            "tflops,tb_per_sec,num_samples\n")
            with self.assertRaisesRegex(ValueError, "CSV schema mismatch"):
                write_csv(str(path), [result])

    def test_pi0_checkpoint_override_requires_its_own_revision(self):
        from eval.pi0 import reference

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.touch()
            with patch.object(reference, "run") as run:
                status = reference.main(["--checkpoint", str(checkpoint)])
        self.assertEqual(status, reference.UNAVAILABLE)
        run.assert_not_called()


class EngineRevisionTests(unittest.TestCase):
    def test_dirty_checkout_has_no_engine_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source.py"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            source.write_text("value = 1\n")
            subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
            subprocess.run([
                "git", "-C", str(repo), "-c", "user.name=Test",
                "-c", "user.email=test@example.com", "commit", "-qm", "baseline",
            ], check=True)
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(git_revision(source), head)
            source.write_text("value = 2\n")
            self.assertIsNone(git_revision(source))


if __name__ == "__main__":
    unittest.main()
