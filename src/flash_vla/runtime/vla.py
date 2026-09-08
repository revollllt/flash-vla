"""The VLA template: the three-stage inference graph every Target instantiates.

    images ─► vision_encoder ─► [host: prompt] ─► llm_backbone ─► action_expert ─► actions
                                                    (prefix KV)      (denoise over KV)

A Target subclasses `VLA` and supplies the model contract -- its configuration,
the shape numbers that make up its identity, the weight schema and loader, its
backend registry, its shipped and reference plans -- and one method, `build`,
that writes the model's computation graph against the op vocabulary with the
graph API (`runtime/graph.py`). Everything else (inputs, staging, the program,
stage outputs, costs, plan selection) has a default here that reads the graph.

Onboarding a model means rewriting its original forward pass in this form:
each op with explicit inputs, outputs and weights, the way vLLM model files
rewrite their transformers originals. The result must pass the precision gate
against the original implementation; it need not be written the same way.

Graph rules a subclass must follow (the smoke check enforces what it can):

- every op writes its outputs in place through the parameters the spec names;
  the graph never uses an op's return value;
- node arguments are references (`g.buf`, `g.w`) or scalars; slicing and
  `view` act on references, never on tensors;
- buffers are declared with their padded allocation and the view the graph
  sees; a stage's contract region of a larger buffer is an alias view.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .cost import Ceiling
from .graph import Graph
from .ops import Vocabulary
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


@dataclass(frozen=True)
class Input:
    """One forward input: its shape as a function of the shape numbers, its dtype,
    and the buffer it is staged into (`None` when a host slot consumes it)."""
    name: str
    dims: Callable[[Mapping[str, int]], tuple[int, ...]]
    dtype: torch.dtype
    buffer: str | None


class VLA:
    """Template base class of a Target. Subclasses declare; they do not run."""

    #: Identity axes.
    name: str
    hardware: str
    model: str
    precision: str = "bf16"
    #: Ordered graph-static axes returned by `shape`; each Target owns this schema.
    shape_axes: tuple[str, ...]
    #: The forward inputs, in the order `sample_inputs` draws them.
    INPUTS: tuple[Input, ...] = ()
    #: The buffer `forward` returns.
    OUTPUT: str = "actions"
    STAGES: tuple[str, ...] = STAGES
    STAGE_OUTPUTS: Mapping[str, tuple[tuple[str, int | None], ...]] = STAGE_OUTPUTS
    #: The one shipped plan and the reference (oracle) plan, call site -> backend.
    plan: Mapping[str, str] = {}
    reference_plan: Mapping[str, str] = {}
    #: Measured ceilings a hardware unit test established for a call site's own
    #: geometry on this hardware, call site -> `Ceiling`; the floor model's
    #: ceiling column uses them in place of the constants' rule.
    CEILINGS: Mapping[str, Ceiling] = {}
    #: The backends this Target routes to.
    registry: Registry

    # -- model contract (subclass) -----------------------------------------

    def configure(self, **config: Any) -> Any:
        """A frozen configuration; an unknown key must raise `TypeError`."""
        raise NotImplementedError

    def shape(self, config: Any, checkpoint: Mapping[str, Any] | None = None) -> dict[str, int]:
        """The identity's shape numbers, in a fixed order."""
        raise NotImplementedError

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        raise NotImplementedError

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        """Write the computation graph with the graph API."""
        raise NotImplementedError

    def load_weights(self, checkpoint: Mapping[str, torch.Tensor],
                     weights: Mapping[str, torch.Tensor]) -> None:
        """Copy the checkpoint into the allocated weights; every weight must be covered."""
        missing = sorted(set(weights) - set(checkpoint))
        if missing:
            raise KeyError(f"checkpoint is missing {missing}")
        for name, value in checkpoint.items():
            if name in weights:
                weights[name].copy_(value)

    def host_state(self, config: Any, shape: Mapping[str, int]) -> Any | None:
        """State a host slot keeps across calls (a tokenizer's staging); `None` when none."""
        return None

    def host(self, slot: str, *, host_state: Any, buffers: Mapping[str, torch.Tensor],
             **inputs: Any) -> None:
        """Run host slot `slot`: the host work between two stages."""
        raise KeyError(f"{self.name} declares no host slot {slot!r}")

    # -- defaults read from the declarations -------------------------------

    def graph(self, shape: Mapping[str, int]) -> Graph:
        """The checked computation graph at `shape`."""
        g = Graph(Vocabulary(self.registry.ops()), self.weight_shapes(shape), self.STAGES,
                  weight_dtype=DTYPES[self.precision])
        self.build(g, shape)
        g.check()
        for stage, outputs in self.STAGE_OUTPUTS.items():
            for buffer, _axis in outputs:
                if buffer not in g.buffers:
                    raise KeyError(f"stage output {buffer!r} of {stage!r} is not declared")
        return g

    def sample_inputs(self, shape: Mapping[str, int], seed: int, device) -> dict[str, torch.Tensor]:
        """Seeded random inputs at `shape`, drawn in `INPUTS` order from one generator.

        Inputs a host slot consumes (`Input.buffer is None`) are returned in
        pinned host memory, as deployment delivers them: a device-resident copy
        would force a device synchronization inside `forward`, and every host
        hiccup during that wait would read as chunk latency. Values are drawn
        on `device` first so a dump stays comparable across this choice.
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        out = {}
        for inp in self.INPUTS:
            tensor = torch.randn(inp.dims(shape), generator=generator, device=device,
                                 dtype=inp.dtype)
            if inp.buffer is None and tensor.device.type == "cuda":
                tensor = tensor.cpu().pin_memory()
            out[inp.name] = tensor
        return out

    def stage(self, buffers: Mapping[str, torch.Tensor], **inputs: Any) -> None:
        """Copy the device inputs into their buffers; host-consumed inputs are skipped."""
        for inp in self.INPUTS:
            if inp.buffer is not None:
                buffers[inp.buffer].copy_(inputs[inp.name])

    def select_plan(self, plan: Any) -> dict[str, str]:
        """`"shipped"`, `"reference"`, a mapping, a JSON object, or a path to one."""
        if plan is None or plan == "shipped":
            return dict(self.plan)
        if plan == "reference":
            return dict(self.reference_plan)
        if isinstance(plan, Mapping):
            return dict(plan)
        if isinstance(plan, str):
            text = plan
            if not plan.lstrip().startswith("{") and Path(plan).is_file():
                text = Path(plan).read_text()
            try:
                routes = json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"plan {plan!r} is neither 'shipped', 'reference', a JSON "
                                 "object nor a path to one") from None
            if not isinstance(routes, dict):
                raise ValueError(f"plan {plan!r} must be a JSON object")
            return routes
        raise TypeError(f"unsupported plan {plan!r}")


__all__ = ["DTYPES", "Input", "STAGES", "STAGE_OUTPUTS", "VLA"]
