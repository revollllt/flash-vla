"""`ModelRunner`: the one engine every Target runs on.

Given a Target (`runtime/vla.py`) and a checkpoint, the runner builds the
model's graph, binds the plan, builds the op table, allocates every buffer the
graph declares at fixed addresses, resolves every node's references to
tensors once, warms up, freezes the workspace allocator, and captures each
stage into its own CUDA graph. `forward` stages the inputs, walks the program
(stage replays and host slots in the declared order) and returns the output
buffer. Nothing here depends on the model: the graph says what to run, the
registry says who runs it.

The runner is also the engine protocol's implementation (`runtime/engine.py`):
harnesses see identity, buffers, program, stage outputs, costs, the graph,
and the instrumentation hooks, and never a model name.

Construction with `checkpoint=None` and `capture=False` stops after the graph
is built and checked: no device, no allocation. That is the declaration path
the CPU smoke check uses.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import partial
import gc
from pathlib import Path
from typing import Generic, Iterator, Mapping
import warnings

import torch

from .cost import SegmentCosts
from .cuda.arena import StaticArena
from .cuda.program import Program, Segment, Step
from .engine import WrapOp, wrap_ops
from .graph import BufRef, Graph, Node, WeightRef
from .identity import ExecutionVariant, Identity, validate_weight_schema
from .registry import GraphContract, Wrapper
from .vla import (
    DTYPES,
    CheckpointReader,
    ConfigT,
    ConfigValue,
    HostStateT,
    PlanSpec,
    Target,
    TensorCheckpoint,
)
from .workspace import Scratch

#: A node argument once its references are resolved to the allocated tensors.
BoundArgument = torch.Tensor | int | float | None


class ModelRunner(Generic[ConfigT, HostStateT]):
    """One constructed Target: graph built, plan bound, buffers allocated, stages captured.

    `checkpoint` is a `CheckpointReader` or an in-memory name -> tensor mapping;
    its shapes are checked against the model's weight schema before anything is
    allocated.

    `assets` is a construction-time role-to-local-path mapping, copied read-only
    for this runner's input sampler, host state and backend factories. It is
    not part of configuration, shape, graph arguments or identity.

    `engine_revision` is the source revision the identity records
    (`flash_vla.provenance.git_revision` at the entry point); the runner never
    inspects a checkout itself, so `None` means the caller named none.

    `quantization` names one of the Target's recipes (`Target.quantization`),
    or its precision policy. It selects the named plans' routes for the
    recipe's call sites, is checked against the resolved routes, and is
    recorded in the identity's execution variant.

    Python's cyclic collector is disabled while the stages are captured (a
    collection inside a capture frees device memory and invalidates the
    capture) and run once afterwards. That is preparation, not a deployment
    policy: the deployment path is graph replay, and the collector is the
    application's to manage. The runner never freezes the heap: it references
    itself through its captured segments, so a frozen runner could never be
    collected.
    """

    def __init__(self, target: Target[ConfigT, HostStateT],
                 checkpoint: CheckpointReader | Mapping[str, torch.Tensor] | None = None, *,
                 checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
                 checkpoint_signature: str | None = None, model_revision: str | None = None,
                 engine_revision: str | None = None, plan: PlanSpec = "shipped",
                 quantization: str | None = None, device: str = "cuda", capture: bool = True,
                 warmup: int = 3, assets: Mapping[str, Path] | None = None,
                 **config: ConfigValue) -> None:
        model = target.model
        if model_revision is not None:
            warnings.warn("model_revision is owned by the model; use checkpoint_id for weights",
                          DeprecationWarning, stacklevel=2)
            if model_revision != model.model_revision:
                raise ValueError("model_revision must equal the model's revision; "
                                 "checkpoint provenance belongs in checkpoint_id")
        if checkpoint_signature is not None and checkpoint_signature != model.inference_signature:
            raise ValueError("inference signature mismatch; resolve a compatible Target/model revision")
        reader = (checkpoint if checkpoint is None or isinstance(checkpoint, CheckpointReader)
                  else TensorCheckpoint(checkpoint))
        self.target = target
        self.measurement_context = {
            "weights": {"checkpoint_id": checkpoint_id, "checkpoint_digest": checkpoint_digest},
        }
        self.config = model.configure(**config)
        self.shape: dict[str, int] = dict(model.shape(self.config, reader))
        self.graph: Graph = target.graph(self.shape)
        if reader is not None:
            validate_weight_schema(reader.shapes, self.graph.weight_shapes)
        self.quantization: str = target.precision if quantization is None else quantization
        self.plan: dict[str, str] = target.select_plan(plan, self.quantization)
        routes = target.registry.resolve(self.plan, self.graph.call_sites)
        target.check_quantization(routes, self.quantization)
        variant = ExecutionVariant(
            quantization=target.quantization_recipe(self.quantization).identity())
        self.identity = Identity(target=target.name, hardware=target.hardware,
                                 model=model.name, model_revision=model.model_revision,
                                 inference_signature=model.inference_signature,
                                 shape=self.shape, plan=routes, engine_revision=engine_revision,
                                 precision=target.precision, execution_variant=variant)
        self.program: tuple[Step, ...] = tuple(self.graph.program)
        self.stage_outputs = {stage: tuple(outputs)
                              for stage, outputs in model.stage_outputs.items()}
        self.derived: dict[str, int] = dict(self.graph.derived)
        self.device = torch.device(device)
        self.scratch = Scratch(self.device, assets=assets)
        self.assets = self.scratch.assets
        #: The op table in force: the routed wrappers, wrapped while `instrument` is active.
        self.ops: Mapping[str, Wrapper] = target.registry.op_table(routes, self.scratch)

        self.weights: dict[str, torch.Tensor] = {}
        self.arena: StaticArena | None = None
        self.buffers: Mapping[str, torch.Tensor] = {}
        #: The model's host state; `None` until a checkpoint is loaded.
        self.host_state: HostStateT | None = None
        self.graphs: Program | None = None
        #: Per stage, every node with its references resolved once to tensors.
        self.bound: dict[str, list[tuple[Node, tuple[BoundArgument, ...]]]] = {}
        if reader is None:
            if capture:
                raise ValueError("capturing needs a checkpoint; pass capture=False to only "
                                 "declare the graph")
            return

        dtype = DTYPES[target.precision]
        self.weights = {name: torch.empty(shape, dtype=dtype, device=self.device)
                        for name, shape in self.graph.weight_shapes.items()}
        reader.copy_into(self.weights)
        self.arena = StaticArena(self.graph.buffers, self.device)
        self.buffers = self.arena.buffers
        self.host_state = model.host_state(self.config, self.shape, self.assets)

        def resolve(argument: BufRef | WeightRef | int | float | None) -> BoundArgument:
            if isinstance(argument, BufRef):
                return argument.resolve(self.buffers)
            if isinstance(argument, WeightRef):
                return argument.resolve(self.weights)
            return argument

        # The addresses never move after this, so every capture binds the same tensors.
        self.bound = {stage: [(node, tuple(resolve(argument) for argument in node.args))
                              for node in self.graph.nodes_of(stage)]
                      for stage in self.graph.segment_names}
        if capture:
            self.capture(warmup=warmup)

    def capture(self, *, warmup: int = 3) -> None:
        """Recapture fresh graph/stream pairs, retaining weights and static buffers."""
        segments = [Segment(name, partial(self.run_eager, name))
                    for name in self.graph.segment_names]
        collector_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                self.graphs = Program(segments, warmup=warmup, after_warmup=self.scratch.freeze)
        finally:
            if collector_was_enabled:
                gc.enable()
        gc.collect()

    # -- execution ------------------------------------------------------------

    def run_eager(self, segment: str) -> None:
        """Issue one stage's nodes outside its graph, on the current stream."""
        if segment not in self.bound:
            raise KeyError(f"no stage {segment!r}; stages: {self.graph.segment_names}")
        ops = self.ops
        scratch = self.scratch
        for node, args in self.bound[segment]:
            scratch.current = node.index
            if node.is_copy:
                args[0].copy_(args[1])
            else:
                ops[node.call_site](*args)
        scratch.current = None

    @contextmanager
    def instrument(self, wrap: WrapOp) -> Iterator[None]:
        """While active, every op-table entry `name` is replaced by `wrap(name, fn)`."""
        original = self.ops
        self.ops = wrap_ops(original, wrap)
        try:
            yield
        finally:
            self.ops = original

    def replay(self, segment: str) -> None:
        """Replay one stage on its capture stream, ordered with the caller."""
        if self.graphs is None:
            raise RuntimeError("this runner was built without capture")
        self.graphs.replay(segment)

    # -- the engine protocol --------------------------------------------------

    @property
    def costs(self) -> SegmentCosts:
        """Per stage, every call site's minimal bytes and FLOPs, derived from the graph
        and priced in the quantization recipe's formats, with the ceilings the Target
        declares (`Target.ceilings`) attached by call site."""
        ceilings = self.target.ceilings
        pricing = self.target.quantization_recipe(self.quantization).pricing
        return {stage: [replace(inv, ceiling=ceilings.get(inv.call_site)) for inv in rows]
                for stage, rows in self.graph.costs(pricing).items()}

    @property
    def graph_contract(self) -> GraphContract:
        return self.target.registry.graph_contract(dict(self.identity.plan))

    @property
    def atomic_groups(self) -> tuple[frozenset[str], ...]:
        return self.target.registry.atomic_groups(dict(self.identity.plan))

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""
        if self.arena is None:
            raise RuntimeError("this runner declared its graph without allocating")
        return self.arena.allocation(name)

    def sample_inputs(self, seed: int = 0) -> dict[str, torch.Tensor]:
        """Seeded inputs at this runner's shapes."""
        return self.target.model.sample_inputs(self.shape, seed, self.device, assets=self.assets)

    def stage(self, **inputs: torch.Tensor) -> None:
        """Copy the device inputs into their static addresses; run nothing."""
        self.target.model.stage(buffers=self.buffers, inputs=inputs)

    def host(self, slot: str, **inputs: torch.Tensor) -> None:
        """Run one declared host slot."""
        if slot not in self.graph.host_slots:
            raise KeyError(f"no host slot {slot!r}; this Target declares {self.graph.host_slots}")
        self.target.model.host(slot, host_state=self.host_state, buffers=self.buffers, inputs=inputs)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Stage the inputs, run the program in order, return the output view."""
        self.stage(**inputs)
        for step in self.program:
            if step.kind == "host":
                self.host(step.name, **inputs)
            else:
                self.replay(step.name)
        return self.buffers[self.target.model.output]


__all__ = ["ModelRunner", "Scratch"]
