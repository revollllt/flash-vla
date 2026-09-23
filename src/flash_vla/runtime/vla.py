"""A Target: one model definition composed with one device's backends and plans.

    images -> vision_encoder -> [host: prompt] -> llm_backbone -> action_expert -> actions
                                                   (prefix KV)      (denoise over KV)

Two halves, kept apart so a model is written once and deployed on any device:

- `ModelDefinition` is the hardware-free model contract, one subclass per
  model under `flash_vla/models/<model>/definition.py`: identity, configuration,
  the shape numbers, the weight schema, the forward inputs and stage outputs,
  the extension ops its graph uses, the host slots, and `build`, which writes
  the computation graph against the op vocabulary with the graph API
  (`runtime/graph.py`). Hardware layout choices (row padding, which backbone
  call sites carry the prefix mask) are constructor arguments of the model
  object, never branches on a device name.
- `Target` composes a model object with one device's backend registry, its
  shipped and reference plans, its quantization recipes, its measured
  ceilings and the logical IDs of its assets. It holds no logic of its own
  beyond plan selection and recipe checks, so a new device is a new `Target`
  value, not a subclass of another device's Target.

Onboarding a model means rewriting its original forward pass as a
`ModelDefinition`: each op with explicit inputs, outputs and weights, the way
vLLM model files rewrite their transformers originals. The result must pass
the precision gate against the original implementation; it need not be
written the same way.

Graph rules a model must follow (the smoke check enforces what it can):

- every op writes its outputs in place through the parameters the spec names;
  the graph never uses an op's return value;
- node arguments are references (`g.buf`, `g.w`) or scalars; slicing and
  `view` act on references, never on tensors;
- buffers are declared with their padded allocation and the view the graph
  sees; a stage's contract region of a larger buffer is an alias view.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Generic, Mapping, TypeVar

import torch

from .cost import Ceiling, Pricing
from .graph import Graph
from .ops import OpSpec, Vocabulary
from .registry import Registry

#: Canonical stage names, in program order.
STAGES = ("vision_encoder", "llm_backbone", "action_expert")

#: Canonical stage contracts: what each stage produces for the next one and for
#: a correctness harness to compare or inject. The KV cache is layer-major, so
#: its layer axis is 0.
STAGE_OUTPUTS: Mapping[str, tuple[tuple[str, int | None], ...]] = {
    "vision_encoder": (("vision_encoder_x", None),),
    "llm_backbone": (("prefix_k", 0), ("prefix_v", 0)),
    "action_expert": (("actions", None), ("suffix_k", 0), ("suffix_v", 0)),
}

#: Weight and activation dtype of each precision policy.
DTYPES = {"bf16": torch.bfloat16}

#: A construction option a runner forwards to `ModelDefinition.configure`.
ConfigValue = int | str | bool | None
#: What names a plan: "shipped", "reference", a mapping, a JSON object or a path to one.
PlanSpec = str | Mapping[str, str] | None

ConfigT = TypeVar("ConfigT")
HostStateT = TypeVar("HostStateT")


@dataclass(frozen=True)
class Input:
    """One forward input: its shape as a function of the shape numbers, its dtype,
    and the buffer it is staged into (`None` when a host slot consumes it)."""
    name: str
    dims: Callable[[Mapping[str, int]], tuple[int, ...]]
    dtype: torch.dtype
    buffer: str | None


class CheckpointReader(ABC):
    """The weights a runner loads: their shapes, checked before any allocation,
    and one copy into the allocated runtime weights.

    File-backed checkpoints subclass this to read shapes from metadata and
    stream tensors on the copy; an in-memory mapping is a `TensorCheckpoint`.
    """
    #: Name -> shape of every tensor, known without loading a value.
    shapes: Mapping[str, tuple[int, ...]]

    @abstractmethod
    def copy_into(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy every runtime weight in `weights` from this checkpoint."""


class TensorCheckpoint(CheckpointReader):
    """An in-memory checkpoint: name -> tensor."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}

    def copy_into(self, weights: Mapping[str, torch.Tensor]) -> None:
        missing = sorted(set(weights) - set(self.tensors))
        if missing:
            raise KeyError(f"checkpoint is missing {missing}")
        for name, weight in weights.items():
            weight.copy_(self.tensors[name])


@dataclass(frozen=True)
class QuantizationRecipe:
    """A human-approved quantization of some call sites, selected at build time.

    `spec` fixes the math (formats, scale granularity, rounding points) and is
    recorded, with the call sites, as the Identity's
    `execution_variant.quantization`, so every recipe is its own workload;
    `spec["mode"]` names the tolerance tier its kernels meet against its
    reference. `plan` routes each quantized call site to its kernel backend over
    the Target's shipped plan; `reference_plan` routes it to the backend that
    defines the math (fake quantization) over the Target's reference plan.
    `backends` names every backend implementing the recipe: its call sites
    accept only these, and no other call site accepts any recipe's backends.
    `pricing` is how the floor model prices those call sites: the tensor-core
    format and the quantized operands' bytes (`runtime/cost.py`).
    Agents optimize the kernels; changing `spec` or the call sites needs approval.
    """
    spec: Mapping[str, str]
    plan: Mapping[str, str]
    reference_plan: Mapping[str, str]
    backends: frozenset[str]
    pricing: Mapping[str, Pricing] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.plan) != set(self.reference_plan):
            raise ValueError("a recipe's plan and reference plan must name the same call sites")
        if not set(self.pricing) <= set(self.plan):
            raise ValueError("a recipe prices only its own call sites")
        if not {*self.plan.values(), *self.reference_plan.values()} <= self.backends:
            raise ValueError(f"a recipe routes only to its own backends {sorted(self.backends)}")

    def identity(self) -> dict[str, str | list[str]]:
        """`Identity.execution_variant.quantization`: the spec, plus the call
        sites of a recipe that quantizes any."""
        if not self.plan:
            return dict(self.spec)
        return {**self.spec, "call_sites": sorted(self.plan)}


class ModelDefinition(ABC, Generic[ConfigT, HostStateT]):
    """The hardware-free contract of one model; subclasses declare, the runner runs.

    `ConfigT` is the model's frozen configuration and `HostStateT` what its host
    slots keep across calls (`None` for a model without host work).
    """

    #: Identity axes.
    name: str
    model_revision: str
    inference_signature: str
    #: Ordered graph-static axes returned by `shape`.
    shape_axes: tuple[str, ...]
    #: The forward inputs, in the order `sample_inputs` draws them.
    inputs: tuple[Input, ...] = ()
    #: The buffer `forward` returns.
    output: str = "actions"
    stages: tuple[str, ...] = STAGES
    stage_outputs: Mapping[str, tuple[tuple[str, int | None], ...]] = STAGE_OUTPUTS
    #: The op specs this model's graph uses beyond the standard vocabulary.
    ops: tuple[OpSpec, ...] = ()

    @abstractmethod
    def configure(self, **config: ConfigValue) -> ConfigT:
        """A frozen configuration; an unknown key raises `TypeError`."""

    @abstractmethod
    def shape(self, config: ConfigT, checkpoint: CheckpointReader | None) -> dict[str, int]:
        """The identity's shape numbers, in `shape_axes` order."""

    @abstractmethod
    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        """The runtime weight schema at `shape`."""

    @abstractmethod
    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        """Write the computation graph with the graph API."""

    def host_state(self, config: ConfigT, shape: Mapping[str, int],
                   assets: Mapping[str, Path]) -> HostStateT:
        """State the host slots keep across calls (a tokenizer's staging).

        A model without host work is a `ModelDefinition[..., None]` and keeps this default.
        """
        return None

    def host(self, slot: str, host_state: HostStateT, buffers: Mapping[str, torch.Tensor],
             inputs: Mapping[str, torch.Tensor]) -> None:
        """Run host slot `slot`, one of the graph's declared slots (the runner checks),
        between two stages."""
        raise KeyError(f"{self.name} declares no host slot {slot!r}")

    def sample_inputs(self, shape: Mapping[str, int], seed: int, device: torch.device,
                      assets: Mapping[str, Path]) -> dict[str, torch.Tensor]:
        """Seeded inputs at `shape`, drawn in `inputs` order from one generator.

        A model whose inputs come from a recorded fixture reads it from `assets`
        and may reject a seed its fixture does not provide.

        Inputs a host slot consumes (`Input.buffer is None`) are returned in
        pinned host memory, as deployment delivers them: a device-resident copy
        would force a device synchronization inside `forward`, and every host
        hiccup during that wait would read as chunk latency. Values are drawn
        on `device` first so a dump stays comparable across this choice.
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        drawn = {}
        for spec in self.inputs:
            tensor = torch.randn(spec.dims(shape), generator=generator, device=device,
                                 dtype=spec.dtype)
            host_consumed = spec.buffer is None and tensor.device.type == "cuda"
            drawn[spec.name] = tensor.cpu().pin_memory() if host_consumed else tensor
        return drawn

    def stage(self, buffers: Mapping[str, torch.Tensor], inputs: Mapping[str, torch.Tensor]) -> None:
        """Copy the device inputs into their buffers; host-consumed inputs are skipped."""
        for spec in self.inputs:
            if spec.buffer is None:
                continue
            buffers[spec.buffer].copy_(inputs[spec.name])


@dataclass(frozen=True)
class Target(Generic[ConfigT, HostStateT]):
    """One model on one device: the model object and everything the device decides.

    `assets` maps asset roles to the logical IDs a machine's asset map resolves
    to files (`FLASH_VLA_ASSETS`); `ceilings` are measured ceilings a hardware
    unit test established for a call site's own geometry on this device, which
    the floor model uses in place of the constants' rule. `call_site_aliases`
    maps call-site names a saved plan may still use to this Target's names (a
    device whose layout renames call sites keeps its old plans binding).
    Every mapping is frozen on construction.
    """
    name: str
    hardware: str
    model: ModelDefinition[ConfigT, HostStateT]
    registry: Registry
    plan: Mapping[str, str]
    reference_plan: Mapping[str, str] = field(default_factory=dict)
    precision: str = "bf16"
    quantization: Mapping[str, QuantizationRecipe] = field(default_factory=dict)
    ceilings: Mapping[str, Ceiling] = field(default_factory=dict)
    assets: Mapping[str, str] = field(default_factory=dict)
    call_site_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A frozen dataclass still hands out its dicts; freeze their contents too.
        object.__setattr__(self, "plan", MappingProxyType(dict(self.plan)))
        object.__setattr__(self, "reference_plan", MappingProxyType(dict(self.reference_plan)))
        object.__setattr__(self, "quantization", MappingProxyType(dict(self.quantization)))
        object.__setattr__(self, "ceilings", MappingProxyType(dict(self.ceilings)))
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))
        object.__setattr__(self, "call_site_aliases", MappingProxyType(dict(self.call_site_aliases)))

    def graph(self, shape: Mapping[str, int]) -> Graph:
        """The model's checked computation graph at `shape`."""
        model = self.model
        g = Graph(Vocabulary(model.ops), model.weight_shapes(shape), model.stages,
                  weight_dtype=DTYPES[self.precision])
        model.build(g, shape)
        g.check()
        undeclared = [(stage, buffer) for stage, outputs in model.stage_outputs.items()
                      for buffer, _axis in outputs if buffer not in g.buffers]
        if undeclared:
            raise KeyError(f"stage outputs {undeclared} are not declared")
        return g

    def quantization_recipe(self, quantization: str) -> QuantizationRecipe:
        """The recipe named `quantization`; the `precision` policy is the empty recipe."""
        if quantization == self.precision:
            return QuantizationRecipe({"mode": self.precision}, {}, {}, frozenset())
        if quantization not in self.quantization:
            raise KeyError(f"{self.name} has no quantization {quantization!r}; "
                           f"it supports {[self.precision, *self.quantization]}")
        return self.quantization[quantization]

    def check_quantization(self, routes: Mapping[str, str], quantization: str) -> None:
        """Raise `ValueError` unless `routes` runs exactly `quantization`: its
        call sites on its backends and no other call site on a recipe's backend."""
        recipe = self.quantization_recipe(quantization)
        quantized = frozenset().union(*(other.backends for other in self.quantization.values()))
        misrouted = {site: backend for site, backend in routes.items()
                     if (backend not in recipe.backends if site in recipe.plan
                         else backend in quantized)}
        if misrouted:
            raise ValueError(f"quantization {quantization!r} runs {sorted(recipe.plan)} on "
                             f"{sorted(recipe.backends)} and no other call site on a quantized "
                             f"backend; misrouted: {misrouted}")

    def select_plan(self, plan: PlanSpec, quantization: str) -> dict[str, str]:
        """The routes `plan` names, with this Target's call-site aliases applied.

        `"shipped"` (or `None`) and `"reference"` take `quantization`'s routes
        for its call sites; a mapping, a JSON object or the path of a JSON file
        is used as given. An aliased old name yields to the current name when
        a plan names both.
        """
        recipe = self.quantization_recipe(quantization)
        if plan is None or plan == "shipped":
            routes = {**self.plan, **recipe.plan}
        elif plan == "reference":
            routes = {**self.reference_plan, **recipe.reference_plan}
        elif isinstance(plan, Mapping):
            routes = dict(plan)
        else:
            is_file = not plan.lstrip().startswith("{") and Path(plan).is_file()
            text = Path(plan).read_text() if is_file else plan
            try:
                routes = json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"plan {plan!r} is neither 'shipped', 'reference', a JSON "
                                 "object nor a path to one") from None
            if not isinstance(routes, dict):
                raise ValueError(f"plan {plan!r} must be a JSON object")
        for old, new in self.call_site_aliases.items():
            if old not in routes:
                continue
            routes.setdefault(new, routes.pop(old))
        return routes


__all__ = ["CheckpointReader", "ConfigT", "ConfigValue", "HostStateT", "DTYPES", "Input", "ModelDefinition", "PlanSpec",
           "QuantizationRecipe", "STAGES", "STAGE_OUTPUTS", "Target", "TensorCheckpoint"]
