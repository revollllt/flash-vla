"""`ModelRunner`: the one engine every Target runs on.

Given a Target (`runtime/vla.py`) and a checkpoint, the runner builds the
Target's graph, binds the plan, builds the op table, allocates every buffer
the graph declares at fixed addresses, resolves every node's references to
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

import gc
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

import torch

from .cost import SegmentCosts
from .cuda.arena import StaticArena
from .cuda.program import Program, Segment, Step
from .engine import wrap_ops
from .graph import BufRef, Graph, Node, WeightRef
from .identity import Identity
from .vla import DTYPES, VLA


class Scratch:
    """The workspace allocator the runner injects into every backend.

    Keyed by role, shape, dtype and device; each key is allocated once, on
    first request, and reused. The runner records which node asked. After
    warmup the allocator is frozen: a request warmup did not cover raises
    instead of allocating during graph capture.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._buffers: dict[tuple, torch.Tensor] = {}
        self.owners: dict[tuple, int | None] = {}
        self.current: int | None = None
        self.frozen = False

    def __call__(self, role: str, shape, dtype, device) -> torch.Tensor:
        key = (role, tuple(shape), dtype, str(device))
        buffer = self._buffers.get(key)
        if buffer is None:
            if self.frozen:
                raise RuntimeError(f"workspace is frozen but {key} was requested: warmup did "
                                   "not cover it, so it would allocate mid-capture")
            buffer = torch.zeros(shape, dtype=dtype, device=device)
            self._buffers[key] = buffer
            self.owners[key] = self.current
        return buffer

    def freeze(self) -> None:
        self.frozen = True

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._buffers.values())

    def __len__(self) -> int:
        return len(self._buffers)


class ModelRunner:
    """One constructed Target: graph built, plan bound, buffers allocated, stages captured.

    Python's cyclic collector is disabled while the stages are captured (a
    collection inside a capture frees device memory and invalidates the
    capture) and run once afterwards. That is preparation, not a deployment
    policy: the deployment path is graph replay, and the collector is the
    application's to manage. The runner never freezes the heap: it references
    itself through its captured segments, so a frozen runner could never be
    collected.
    """

    def __init__(self, target: VLA, checkpoint: Mapping[str, torch.Tensor] | None = None, *,
                 plan: Any = "shipped", device: str = "cuda", capture: bool = True,
                 warmup: int = 3, **config: Any) -> None:
        self.target = target
        self.config = target.configure(**config)
        self.shape: dict[str, int] = dict(target.shape(self.config, checkpoint))
        self.graph: Graph = target.graph(self.shape)
        self.plan: dict[str, str] = target.select_plan(plan)
        routes = target.registry.resolve(self.plan, self.graph.call_sites)
        self.identity = Identity(target=target.name, hardware=target.hardware,
                                 model=target.model, shape=self.shape, plan=routes,
                                 precision=target.precision)
        self.program: tuple[Step, ...] = tuple(self.graph.program)
        self.stage_outputs = {stage: tuple(outputs)
                              for stage, outputs in target.STAGE_OUTPUTS.items()}
        self.derived: dict[str, int] = dict(self.graph.derived)
        self.device = torch.device(device)
        self.scratch = Scratch(self.device)
        self._ops = target.registry.op_table(routes, self.scratch)

        self.weights: dict[str, torch.Tensor] = {}
        self.arena: StaticArena | None = None
        self.buffers: Mapping[str, torch.Tensor] = {}
        self.host_state: Any = None
        self.graphs: Program | None = None
        self._bound: dict[str, list[tuple[Node, tuple[Any, ...]]]] = {}
        if checkpoint is None:
            if capture:
                raise ValueError("capturing needs a checkpoint; pass capture=False to only "
                                 "declare the graph")
            return

        dtype = DTYPES[target.precision]
        self.weights = {name: torch.empty(shape, dtype=dtype, device=self.device)
                        for name, shape in self.graph.weight_shapes.items()}
        target.load_weights(checkpoint, self.weights)
        self.arena = StaticArena(self.graph.buffers, self.device)
        self.buffers = self.arena.buffers
        self.host_state = target.host_state(self.config, self.shape)
        self._bind()
        if capture:
            segments = [Segment(name, partial(self.run_eager, name))
                        for name in self.graph.segment_names]
            # A collection during capture invalidated it (CUDA error 901, job
            # 598959).
            collector_was_enabled = gc.isenabled()
            gc.disable()
            try:
                self.graphs = Program(segments, warmup=warmup, after_warmup=self.scratch.freeze)
            finally:
                if collector_was_enabled:
                    gc.enable()
            gc.collect()

    # -- binding and execution ---------------------------------------------

    def _resolve(self, arg: Any) -> Any:
        if isinstance(arg, BufRef):
            return arg.resolve(self.buffers)
        if isinstance(arg, WeightRef):
            return arg.resolve(self.weights)
        return arg

    def _bind(self) -> None:
        """Resolve every node's references once; the addresses never move."""
        for stage in self.graph.segment_names:
            self._bound[stage] = [(node, tuple(self._resolve(a) for a in node.args))
                                  for node in self.graph.nodes_of(stage)]

    @property
    def ops(self) -> SimpleNamespace:
        """The op table in force: the routed wrappers, wrapped while `instrument` is active."""
        return self._ops

    def run_eager(self, segment: str) -> None:
        """Issue one stage's nodes outside its graph, on the current stream."""
        if segment not in self._bound:
            raise KeyError(f"no stage {segment!r}; stages: {self.graph.segment_names}")
        ops = self._ops
        scratch = self.scratch
        for node, args in self._bound[segment]:
            scratch.current = node.index
            if node.is_copy:
                args[0].copy_(args[1])
            else:
                getattr(ops, node.call_site)(*args)
        scratch.current = None

    @contextmanager
    def instrument(self, wrap: Callable[[str, Callable], Callable]) -> Iterator[None]:
        """While active, every op-table entry `name` is replaced by `wrap(name, fn)`."""
        original = self._ops
        self._ops = wrap_ops(original, wrap)
        try:
            yield
        finally:
            self._ops = original

    def replay(self, segment: str) -> None:
        """Replay one captured stage on the current stream."""
        if self.graphs is None:
            raise RuntimeError("this runner was built without capture")
        self.graphs.replay(segment)

    # -- the engine protocol ------------------------------------------------

    @property
    def costs(self) -> SegmentCosts:
        """Per stage, every call site's minimal bytes and FLOPs, derived from the graph,
        with the ceilings the Target declares (`VLA.CEILINGS`) attached by call site."""
        ceilings = self.target.CEILINGS
        return {stage: [replace(inv, ceiling=ceilings.get(inv.call_site)) for inv in rows]
                for stage, rows in self.graph.costs().items()}

    @property
    def graph_contract(self) -> dict[str, list[str]]:
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
        return self.target.sample_inputs(self.shape, seed, self.device)

    def stage(self, **inputs: Any) -> None:
        """Copy the device inputs into their static addresses; run nothing."""
        self.target.stage(self.buffers, **inputs)

    def host(self, slot: str, **inputs: Any) -> None:
        """Run one declared host slot."""
        if slot not in self.graph.host_slots:
            raise KeyError(f"no host slot {slot!r}; this Target declares {self.graph.host_slots}")
        self.target.host(slot, host_state=self.host_state, buffers=self.buffers, **inputs)

    def forward(self, **inputs: Any) -> torch.Tensor:
        """Stage the inputs, run the program in order, return the output view."""
        self.stage(**inputs)
        for step in self.program:
            if step.kind == "host":
                self.host(step.name, **inputs)
            else:
                self.replay(step.name)
        return self.buffers[self.target.OUTPUT]


__all__ = ["ModelRunner", "Scratch"]
