import copy
import unittest

from flash_vla.runtime.identity import IDENTITY_SCHEMA_VERSION, Identity


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

    def test_unknown_schema_version_is_rejected(self):
        payload = identity().as_dict()
        payload["schema_version"] = 3
        with self.assertRaises(ValueError):
            Identity.from_dict(payload)

    def test_mutable_model_revision_labels_are_rejected(self):
        for value in ("latest", "main", "current", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                identity(model_revision=value)


if __name__ == "__main__":
    unittest.main()
