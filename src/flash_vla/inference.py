"""Target names, and how a harness constructs a Target's runner by name.

The one place a harness learns which Targets exist and how to construct one.
`TARGETS` names each Target's module and its model's `sources` module
(`models/<model>/sources.py`), which says where the weights and the input
fixture come from and what provenance each carries; `build_runner` constructs
every named Target's runner from that, so every construction records its
provenance and engine revision the same way. Everything after construction
goes through the engine protocol (`flash_vla.runtime.engine`), so the latency,
profile, kernel, floor and correctness runners contain no model names; this
registry is where they enter.

A plan is `"shipped"` (the Target's one deployed plan, the default),
`"reference"` (its correctness oracle route), a JSON object, or a path to a
JSON file (the candidate plans of the optimization workspace, `lab/plans/`).
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from flash_vla.provenance import ImplementationProvenance, git_revision
from flash_vla.runtime import ModelRunner
from flash_vla.runtime.vla import ConfigValue, PlanSpec, Target


@dataclass(frozen=True)
class TargetEntry:
    """Where one Target lives, both modules imported on first use:
    `target_module` holds its `TARGET`, and `sources_module` is its model's,
    whose `runner_source(target, device=, declare=, **options)` returns the
    `RunnerSource` a runner is built from. Every model's `runner_source`
    accepts `device`, `declare` and `seed`, even where one selects nothing,
    so a harness passes the same options to every Target."""
    target_module: str
    sources_module: str


#: Full Target name -> its entry. Short aliases resolve through `resolve`.
TARGETS: dict[str, TargetEntry] = {
    "hardware/nvidia/rtx5090/groot_n17": TargetEntry(
        "flash_vla.hardware.nvidia.rtx5090.groot_n17", "flash_vla.models.groot_n17.sources"),
    "hardware/nvidia/h100/lingbot_vla": TargetEntry(
        "flash_vla.hardware.nvidia.h100.lingbot_vla", "flash_vla.models.lingbot.sources"),
    "hardware/nvidia/h100/pi05": TargetEntry(
        "flash_vla.hardware.nvidia.h100.pi05", "flash_vla.models.pi05.sources"),
    "hardware/nvidia/h100/pi0": TargetEntry(
        "flash_vla.hardware.nvidia.h100.pi0", "flash_vla.models.pi0.sources"),
    "hardware/nvidia/rtx5090/pi0": TargetEntry(
        "flash_vla.hardware.nvidia.rtx5090.pi0", "flash_vla.models.pi0.sources"),
    "hardware/nvidia/rtx5090/pi05": TargetEntry(
        "flash_vla.hardware.nvidia.rtx5090.pi05", "flash_vla.models.pi05.sources"),
}
#: Short name -> full Target name.
ALIASES = {"rtx5090/groot_n17": "hardware/nvidia/rtx5090/groot_n17",
           "5090/groot_n17": "hardware/nvidia/rtx5090/groot_n17",
           "groot-n17": "hardware/nvidia/rtx5090/groot_n17",
           "rtx5090_groot_n17": "hardware/nvidia/rtx5090/groot_n17",
           "h100/pi05": "hardware/nvidia/h100/pi05", "pi05": "hardware/nvidia/h100/pi05",
           "h100/pi0": "hardware/nvidia/h100/pi0", "pi0": "hardware/nvidia/h100/pi0",
           "h100/lingbot_vla": "hardware/nvidia/h100/lingbot_vla",
           "lingbot_vla": "hardware/nvidia/h100/lingbot_vla",
           "rtx5090/pi0": "hardware/nvidia/rtx5090/pi0",
           "5090/pi0": "hardware/nvidia/rtx5090/pi0",
           "rtx5090/pi05": "hardware/nvidia/rtx5090/pi05",
           "5090/pi05": "hardware/nvidia/rtx5090/pi05",
           # The `lab/plans/<target>-<name>.json` prefix form, which cannot
           # carry a slash.
           "rtx5090_pi0": "hardware/nvidia/rtx5090/pi0",
           "rtx5090_pi05": "hardware/nvidia/rtx5090/pi05"}

#: The two plan names every Target understands.
PLAN_NAMES = ("shipped", "reference")


def resolve(name: str) -> str:
    """The full Target name for `name` or one of its aliases."""
    full = ALIASES.get(name, name)
    if full not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)} "
                       f"and aliases {sorted(ALIASES)}")
    return full


def get_target(name: str) -> Target:
    """The Target called `name` or one of its aliases."""
    return import_module(TARGETS[resolve(name)].target_module).TARGET


def build_runner(target: Target, plan: PlanSpec = "shipped", *, quantization: str | None = None,
                 device: str = "cuda", declare: bool = False, capture: bool = True,
                 implementation_source: ImplementationProvenance | None = None,
                 **options: ConfigValue) -> ModelRunner:
    """Construct a runner of `target` on `plan`.

    `target` is a registered Target or a variant of one under its name, such
    as one whose backends another checkout provides (`flash_vla.source`); that
    checkout is its `implementation_source`, and its revision the engine
    revision. `options` are the model's (its `runner_source`). With `declare`
    the runner stops at its graph: no weights, no assets, no capture.
    """
    source = import_module(TARGETS[target.name].sources_module).runner_source(
        target, device=device, declare=declare, **options)
    engine_revision = (git_revision() if implementation_source is None
                       else implementation_source.revision)
    return ModelRunner(target, source.checkpoint, weights_provenance=source.weights_provenance,
                       fixture_provenance=source.fixture_provenance,
                       implementation_source=implementation_source,
                       engine_revision=engine_revision, plan=plan, quantization=quantization,
                       device=device, capture=capture and not declare, assets=source.assets,
                       **source.config)


def build(name: str, plan: PlanSpec = "shipped", **options: ConfigValue) -> ModelRunner:
    """Construct the runner of Target `name` on `plan`."""
    return build_runner(get_target(name), plan, **options)


def declare(name: str, plan: PlanSpec = "shipped", **options: ConfigValue) -> ModelRunner:
    """Construct the runner of Target `name` up to its graph: no weights, no device."""
    return build_runner(get_target(name), plan, declare=True, device="cpu", **options)


def parse_options(items: list[str]) -> dict[str, ConfigValue]:
    """`key=value` strings to a dict; true/false and integers are converted."""
    out: dict[str, ConfigValue] = {}
    for item in items:
        key, _, value = item.partition("=")
        low = value.lower()
        out[key] = (True if low == "true" else False if low == "false"
                    else int(value) if value.lstrip("-").isdigit() else value)
    return out


__all__ = ["ALIASES", "PLAN_NAMES", "TARGETS", "TargetEntry", "build", "build_runner", "declare",
           "get_target", "parse_options", "resolve"]
