"""Route typos must fail before any backend factory or GPU work."""
import importlib.util
from pathlib import Path
import sys
import unittest

# Load the hardware-free module without runtime.__init__ importing torch.
spec = importlib.util.spec_from_file_location(
    "binding_under_test", Path(__file__).resolve().parents[1] / "src/flash_vla/runtime/binding.py")
binding = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = binding
spec.loader.exec_module(binding)


class ResolveTests(unittest.TestCase):
    def test_typo_rejected(self):
        with self.assertRaisesRegex(KeyError, "qvk"):
            binding.resolve({"qvk": "cuda"}, "reference", ["qkv"])

    def test_default_and_override_with_generator(self):
        self.assertEqual(
            binding.resolve({"qkv": "cuda"}, "reference", iter(["norm", "qkv"])),
            {"norm": "reference", "qkv": "cuda"})

    def test_registered_but_pruned_site_is_not_a_typo(self):
        self.assertEqual(binding.resolve({"qkv": "cuda"}, "reference", [],
                                         known_call_sites=["qkv"]), {})
        with self.assertRaisesRegex(KeyError, "qvk"):
            binding.resolve({"qvk": "cuda"}, "reference", [], known_call_sites=["qkv"])

    def test_empty_plan(self):
        self.assertEqual(binding.resolve(None, "reference", ["qkv"]),
                         {"qkv": "reference"})

    def test_nonempty_plan_for_empty_graph_rejected(self):
        with self.assertRaisesRegex(KeyError, "qkv"):
            binding.resolve({"qkv": "cuda"}, "reference", [])


if __name__ == "__main__":
    unittest.main()
