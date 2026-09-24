"""The dependency rules of ARCHITECTURE.md, checked on the source without importing it.

Every rule here was broken once and fixed; the test keeps it from regressing.
A module's imports are read from its syntax tree, relative imports resolved
against its package, including imports inside functions.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"
DEVICES = SOURCE / "flash_vla/hardware/nvidia"
#: (device, package) of every Target: a device package holding a `target.py`.
TARGETS = tuple(sorted((path.parent.parent.name, path.parent.name)
                       for path in DEVICES.glob("*/*/target.py")))
#: (device, package) of every component: the device's other packages.
COMPONENTS = tuple(sorted((path.parent.name, path.name) for path in DEVICES.glob("*/*")
                          if path.is_dir() and path.name not in ("measured", "__pycache__")
                          and not (path / "target.py").exists()))
HARNESSES = ("benchmarks", "eval", "tools", "tests", "lab", "measurement")


def module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path: Path, root: Path) -> set[str]:
    """Absolute names of every module `path` imports."""
    name = module_name(path, root)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            anchor = package.split(".")[:len(package.split(".")) - node.level + 1]
            base = ".".join(anchor + ([base] if base else []))
        found.add(base)
        found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def modules(root: Path, prefix: str) -> dict[str, set[str]]:
    base = root / prefix.replace(".", "/")
    return {module_name(path, root): imports_of(path, root)
            for path in sorted(base.rglob("*.py")) if "__pycache__" not in path.parts}


def violations(importers: dict[str, set[str]], forbidden: tuple[str, ...]) -> dict[str, list[str]]:
    found = {name: sorted(target for target in imported
                          if any(target == rule or target.startswith(rule + ".") for rule in forbidden))
             for name, imported in importers.items()}
    return {name: targets for name, targets in found.items() if targets}


def test_runtime_knows_no_model_device_or_entry_point():
    assert violations(modules(SOURCE, "flash_vla.runtime"),
                      ("flash_vla.models", "flash_vla.hardware", "flash_vla.inference",
                       "flash_vla.source")) == {}


def test_models_are_hardware_free():
    assert violations(modules(SOURCE, "flash_vla.models"),
                      ("flash_vla.hardware", "flash_vla.inference", "flash_vla.source")) == {}


@pytest.mark.parametrize("leaf", ["provenance", "assets"])
def test_provenance_and_assets_are_leaves(leaf):
    assert violations({f"flash_vla.{leaf}": imports_of(SOURCE / f"flash_vla/{leaf}.py", SOURCE)},
                      ("flash_vla",)) == {}


def test_entry_point_names_targets_without_importing_them():
    """`inference` reaches a Target and its model's sources by module name, on first use."""
    assert violations({"flash_vla.inference": imports_of(SOURCE / "flash_vla/inference.py", SOURCE)},
                      ("flash_vla.models", "flash_vla.hardware", "flash_vla.source")) == {}


@pytest.mark.parametrize("device,other", [("h100", "rtx5090"), ("rtx5090", "h100")])
def test_a_device_imports_nothing_of_another_device(device, other):
    assert violations(modules(SOURCE, f"flash_vla.hardware.nvidia.{device}"),
                      (f"flash_vla.hardware.nvidia.{other}",)) == {}


def test_a_target_imports_no_other_target():
    found = {}
    for device, target in TARGETS:
        others = tuple(f"flash_vla.hardware.nvidia.{other_device}.{other}"
                       for other_device, other in TARGETS if (other_device, other) != (device, target))
        found.update(violations(modules(SOURCE, f"flash_vla.hardware.nvidia.{device}.{target}"),
                                others))
    assert found == {}


def test_components_read_only_model_constants_and_reference_math():
    """A component may read a model's spec and reference math, never its graph or definition."""
    found = {}
    for device, component in COMPONENTS:
        importers = modules(SOURCE, f"flash_vla.hardware.nvidia.{device}.{component}")
        for name, imported in importers.items():
            # flash_vla.models.<model>.<module>[.<name>]: only spec and reference modules.
            models = sorted(target for target in imported
                            if target.startswith("flash_vla.models.") and len(target.split(".")) > 3
                            and target.split(".")[3] not in ("spec", "reference"))
            if models:
                found[name] = models
    assert found == {}


def test_production_imports_no_harness():
    assert violations(modules(SOURCE, "flash_vla"), HARNESSES) == {}


def test_shared_siglip_geometry_matches_both_models():
    """One SigLIP component package serves Pi0 and Pi0.5, which holds only while
    both models' vision towers have the same shapes."""
    from flash_vla.hardware.nvidia.h100.siglip import geometry
    from flash_vla.models.pi0 import spec as pi0_spec

    # prompt_len sizes only Pi0's prompt embedding; every vision weight is independent of it.
    shapes = pi0_spec.weight_shapes(prompt_len=0)
    expected = {
        "vision_attn_qkv_w": (geometry.LAYERS, geometry.DIM, geometry.QKV_DIM),
        "vision_attn_o_w": (geometry.LAYERS, geometry.DIM, geometry.DIM),
        "vision_ffn_up_w": (geometry.LAYERS, geometry.DIM, geometry.FFN),
        "vision_ffn_down_w": (geometry.LAYERS, geometry.FFN, geometry.DIM),
        "vision_pre_attn_norm_w": (geometry.LAYERS, geometry.DIM),
        "vision_pre_ffn_norm_w": (geometry.LAYERS, geometry.DIM),
    }
    assert {name: shapes[name] for name in expected} == expected
    assert geometry.HEADS * geometry.HEAD_DIM == geometry.DIM
