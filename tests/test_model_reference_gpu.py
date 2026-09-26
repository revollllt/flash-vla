"""Every Target against its model's end-to-end reference, on a GPU.

The gate: `eval.model_reference` at one denoising step and one layer, on
seeded random official-layout weights (or the Target's own checkpoint, where
it has one configured), against the float32 reference -- every routed
kernel, the weight conversion and fold, and the graph, held to the model
itself within the shared shallow tolerances. The cases are every registered
Target, every workload it builds (`Target.workloads`), plan and quantization
recipe; a case runs on the device its kernels are built for, with the assets
it needs, and a plan that runs the reference itself in plain PyTorch
(`PORTABLE`) runs on any.

Bit for bit: GR00T's plans and LingBot's reference plan bind the runner's
weights to the reference's own modules and only stage the tables capture
needs, so the captured engine equals the eager bfloat16 reference exactly.
GR00T's workload has no shallow form, so this is its check here. Fidelity to
upstream is separate: the official oracles (`eval/<model>/`), and for
LingBot, which has no local checkpoint or upstream run, the reference's
line-by-line translation.
"""
import json
import os
from pathlib import Path

import pytest
import torch
from _pytest.mark.structures import ParameterSet

from flash_vla.inference import TARGETS, get_target
from flash_vla.runtime.vla import ConfigValue

DEVICE = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
#: The compute capability each device's kernels are built for (`sm_90a`, `sm_120a`).
CAPABILITIES = {"h100-sxm5-80gb": (9, 0), "rtx5090-32gb": (12, 0)}
#: Plans that run the model reference itself, in plain PyTorch on any CUDA
#: device, with the construction options that need no asset.
PORTABLE: dict[tuple[str, str], dict[str, ConfigValue]] = {
    ("hardware/nvidia/h100/lingbot_vla", "reference"): {"synthetic": True},
}
#: Targets the gate cannot take, and what checks them instead.
EXEMPT = {"hardware/nvidia/rtx5090/groot_n17": "a fixed-depth workload; its plans run the "
                                               "reference bit for bit (below)"}


def configured(asset: str) -> bool:
    """Whether the machine's asset map (`FLASH_VLA_ASSETS`) locates `asset`."""
    path = os.environ.get("FLASH_VLA_ASSETS")
    return path is not None and asset in json.loads(Path(path).read_text())


def gate_case(name: str, workload: str, plan: str, recipe: str | None) -> ParameterSet:
    """One gate case and what it needs: any CUDA device for a portable plan,
    else the Target's own device and, for a Target of frozen assets, its
    checkpoint in the asset map."""
    target = get_target(name)
    portable = (name, plan) in PORTABLE
    capability = CAPABILITIES[target.hardware]
    ready = DEVICE is not None and (portable or (
        DEVICE == capability and (not target.assets or configured(target.assets["checkpoint"]))))
    options = {**PORTABLE.get((name, plan), {}), "workload": workload,
               **({} if recipe is None else {"quantization": recipe})}
    need = "a CUDA device" if portable else f"sm_{capability[0]}{capability[1]} and the Target's assets"
    return pytest.param(name, plan, options,
                        id=f"{name}-{workload}-{plan}-{recipe or target.precision}",
                        marks=pytest.mark.skipif(not ready, reason=f"needs {need}"))


@pytest.mark.parametrize("target,plan,options", [
    gate_case(name, workload, plan, recipe)
    for name in sorted(TARGETS) if name not in EXEMPT
    for workload in get_target(name).workloads
    for plan in ("shipped", "reference")
    for recipe in (None, *get_target(name).quantization)])
def test_target_passes_its_model_reference_gate(target: str, plan: str,
                                                options: dict[str, ConfigValue]) -> None:
    from eval.model_reference import run

    report = run(target, plan, **options)
    assert report["mode"] == "gate" and report["passed"], report["stages"]


def test_exempt_targets_are_registered() -> None:
    assert set(EXEMPT) <= set(TARGETS)


@pytest.mark.skipif(DEVICE is None or not configured(get_target("groot-n17").assets["checkpoint"]),
                    reason="needs CUDA and the GR00T checkpoint in FLASH_VLA_ASSETS")
@pytest.mark.parametrize("plan", ["shipped", "reference"])
def test_groot_engine_runs_the_bfloat16_reference_bit_for_bit(plan: str) -> None:
    from eval.model_reference import run

    report = run("rtx5090/groot_n17", plan, steps=None, layers=None, reference_precision="bfloat16")
    assert report["finite"] and {stage: metrics["rel_rms"] for stage, metrics in report["stages"].items()} \
        == {stage: 0.0 for stage in report["stages"]}


@pytest.mark.skipif(DEVICE is None, reason="needs a CUDA device")
def test_lingbot_reference_plan_runs_the_bfloat16_reference_bit_for_bit() -> None:
    from eval.model_reference import run

    report = run("h100/lingbot_vla", "reference", steps=None, layers=None,
                 reference_precision="bfloat16", synthetic=True)
    assert report["finite"] and {stage: metrics["rel_rms"] for stage, metrics in report["stages"].items()} \
        == {stage: 0.0 for stage in report["stages"]}
