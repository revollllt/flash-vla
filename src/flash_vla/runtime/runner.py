"""`ModelRunner`: the one engine every Target runs on.

Given a Target (`runtime/vla.py`) and a checkpoint, the runner builds the
model's graph, binds the plan, builds the op table, allocates every buffer the
graph declares at fixed addresses, resolves every node's references to
tensors once, warms up, freezes the workspace allocator, and captures each
stage into its own CUDA graph. `forward` stages the inputs, walks the program
(stage replays and host slots in the declared order) and returns the output
buffer. Nothing here depends on the model: the graph says what to run, the
registry says who runs it.

A model with a replay-time axis on a Target with a replay granularity
(`runtime/replay.py`) has one graph per bucket, all over the same buffers: the
stages the axis reaches are bound and captured once per bucket, the others
once. The host slot that decides the axis (or the staging, when no slot does)
selects the bucket each inference replays and runs the backends' replay hooks
with its exact rows (`select_replay`).

The runner is also the engine protocol's implementation (`runtime/engine.py`):
harnesses see identity, provenance, buffers, program, stage outputs,
the graph, and the instrumentation hooks, and never a model name.

Construction with `checkpoint=None` and `capture=False` stops after the graph
is built and checked: no device, no allocation. That is the declaration path
the CPU smoke check uses.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import gc
from pathlib import Path
from typing import Generic, Iterator, Mapping
import warnings

import torch

from flash_vla.provenance import FixtureProvenance, ImplementationProvenance, WeightsProvenance

from .binding import Candidates
from .cuda.arena import StaticArena
from .cuda.program import Program, Segment, Step, nothing_to_prepare
from .engine import StepScope, WrapOp, wrap_ops
from .graph import BufRef, Graph, Node, WeightRef
from .identity import ExecutionVariant, Identity, validate_weight_schema
from .registry import GraphContract, Wrapper
from .replay import ReplayAxis, replay_buckets
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


@dataclass(frozen=True)
class RunnerSource:
    """What a runner is built from beyond its Target and plan.

    A model's `sources` module resolves one from construction options: the
    weights (`None` declares the graph without them) and their provenance, the
    fixture a measurement feeds the runner, its local assets and the model
    configuration (`flash_vla.inference.build_runner`).
    """
    checkpoint: CheckpointReader | Mapping[str, torch.Tensor] | None
    weights_provenance: WeightsProvenance
    fixture_provenance: FixtureProvenance
    assets: Mapping[str, Path]
    config: Mapping[str, ConfigValue]


class ModelRunner(Generic[ConfigT, HostStateT]):
    """One constructed Target: graph built, plan bound, buffers allocated, stages captured.

    `checkpoint` is a `CheckpointReader` or an in-memory name -> tensor mapping;
    its shapes are checked against the model's weight schema before anything is
    allocated.

    `assets` is a construction-time role-to-local-path mapping, copied read-only
    for this runner's input sampler, host state and backend factories. It is
    not part of configuration, shape, graph arguments or identity.

    `workload` names the model workload the construction options came from
    (`Target.workloads`), `None` when a caller composed them itself; it labels
    reports and changes nothing the runner does.

    `weights_provenance`, `fixture_provenance` and `implementation_source`
    are the provenance the caller names (`flash_vla.provenance`), and
    `engine_revision` the source revision the identity records (`git_revision`
    at the entry point). The runner never inspects a checkout or a file
    itself, so `None` means the caller named none, and none of them changes
    after construction. `measurement_context` is their report form: the named
    weights and fixture, derived on each read.

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
                 workload: str | None = None,
                 weights_provenance: WeightsProvenance | None = None,
                 fixture_provenance: FixtureProvenance | None = None,
                 implementation_source: ImplementationProvenance | None = None,
                 checkpoint_signature: str | None = None, model_revision: str | None = None,
                 engine_revision: str | None = None, plan: PlanSpec = "shipped",
                 quantization: str | None = None, device: str = "cuda", capture: bool = True,
                 warmup: int = 3, assets: Mapping[str, Path] | None = None,
                 **config: ConfigValue) -> None:
        model = target.model
        if model_revision is not None:
            warnings.warn("model_revision is owned by the model; use weights_provenance for the checkpoint",
                          DeprecationWarning, stacklevel=2)
            if model_revision != model.model_revision:
                raise ValueError("model_revision must equal the model's revision; "
                                 "checkpoint provenance belongs in weights_provenance")
        if checkpoint_signature is not None and checkpoint_signature != model.inference_signature:
            raise ValueError("inference signature mismatch; resolve a compatible Target/model revision")
        reader = (checkpoint if checkpoint is None or isinstance(checkpoint, CheckpointReader)
                  else TensorCheckpoint(checkpoint))
        self.target = target
        self.workload = workload
        self.weights_provenance = weights_provenance
        self.fixture_provenance = fixture_provenance
        self.implementation_source = implementation_source
        self.config = model.configure(**config)
        self.shape: dict[str, int] = dict(model.shape(self.config, reader))
        axis = model.replay_axis
        self.replay_axis: ReplayAxis | None = axis
        self.replay_buckets: tuple[int, ...] = (() if axis is None else replay_buckets(
            axis, self.shape, None if workload is None else model.workload(workload).replay_range,
            target.replay_granularity))
        #: The stages the replay axis reaches when there is more than the full
        #: bucket: bound and captured once per bucket.
        self.replay_stages: frozenset[str] = frozenset(
            axis.stages if axis is not None and len(self.replay_buckets) > 1 else ())
        #: The bucket of the stages the axis does not reach: the full one.
        self.full_bucket: int | None = None if axis is None else self.replay_buckets[-1]
        #: The bucket the next replay of a stage the axis reaches runs, and this
        #: inference's value of the axis (`select_replay`); the full one until then.
        self.bucket: int | None = self.full_bucket
        self.extent: int | None = self.full_bucket
        #: The graph at the full bucket: the buffers, call sites and stage outputs.
        self.graph: Graph = target.graph(self.shape)
        #: Each bucket's graph, over the same buffers as `graph`.
        self.variants: dict[int | None, Graph] = {
            bucket: self.graph if bucket == self.full_bucket else target.graph(self.shape, bucket)
            for bucket in (self.replay_buckets or (None,))}
        layout = {name: (spec.shape, spec.dtype, spec.view, spec.alias)
                  for name, spec in self.graph.buffers.items()}
        mismatched = [bucket for bucket, variant in self.variants.items()
                      if {name: (spec.shape, spec.dtype, spec.view, spec.alias)
                          for name, spec in variant.buffers.items()} != layout
                      or variant.program != self.graph.program]
        if mismatched:
            raise ValueError(f"buckets {mismatched} declare other buffers or steps than the full "
                             "graph; every bucket must run over the same buffers")
        if reader is not None:
            validate_weight_schema(reader.shapes, self.graph.weight_shapes)
        self.quantization: str = target.precision if quantization is None else quantization
        self.plan: Candidates = target.select_plan(plan, self.quantization)
        # One op table serves every bucket, so every route must support every bucket's shape.
        routes_by_bucket = {
            bucket: target.registry.resolve(
                self.plan, variant.call_sites,
                self.shape if axis is None else axis.at(self.shape, bucket))
            for bucket, variant in self.variants.items()}
        routes = routes_by_bucket[self.full_bucket]
        partial_routes = {bucket: {site: backend for site, backend in bucket_routes.items()
                                   if routes[site] != backend}
                          for bucket, bucket_routes in routes_by_bucket.items()
                          if bucket_routes != routes}
        if partial_routes:
            raise ValueError(f"the plan routes differently at buckets {partial_routes}: a backend "
                             "supports only some of the buckets")
        target.check_quantization(routes, self.quantization)
        variant = ExecutionVariant(
            quantization=target.quantization_recipe(self.quantization).identity())
        self.identity = Identity(target=target.name, hardware=target.hardware,
                                 model=model.name, model_revision=model.model_revision,
                                 inference_signature=model.inference_signature,
                                 shape=self.shape, plan=routes, engine_revision=engine_revision,
                                 precision=target.precision, execution_variant=variant,
                                 # The full bucket alone is implied by the shape.
                                 replay_buckets=(self.replay_buckets
                                                 if len(self.replay_buckets) > 1 else ()))
        self.program: tuple[Step, ...] = tuple(self.graph.program)
        self.stage_outputs = {stage: tuple(outputs)
                              for stage, outputs in model.stage_outputs.items()}
        self.derived: dict[str, int] = dict(self.graph.derived)
        self.device = torch.device(device)
        self.scratch = Scratch(self.device, assets=assets)
        self.assets = self.scratch.assets
        #: The op table in force: the routed wrappers, wrapped while `instrument` is active.
        self.ops: Mapping[str, Wrapper] = target.registry.op_table(routes, self.scratch)
        #: Whether an inference reads the axis's value: only when there is a bucket to
        #: choose or a backend that runs the exact length (a read can synchronize).
        self.reads_extent: bool = axis is not None and (len(self.replay_buckets) > 1
                                                        or bool(self.scratch.replay_hooks))

        self.weights: dict[str, torch.Tensor] = {}
        self.arena: StaticArena | None = None
        self.buffers: Mapping[str, torch.Tensor] = {}
        #: The model's host state; `None` until a checkpoint is loaded.
        self.host_state: HostStateT | None = None
        self.graphs: Program | None = None
        #: Per (stage, bucket), every node with its references resolved once to tensors.
        self.bound: dict[tuple[str, int | None], list[tuple[Node, tuple[BoundArgument, ...]]]] = {}
        #: The scope each replay and host slot runs in while `observe` is active.
        self.step_scope: StepScope | None = None
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
        self.bound = {(stage, bucket): [(node, tuple(resolve(argument) for argument in node.args))
                                        for node in variant.nodes_of(stage)]
                      for bucket, variant in self.variants.items()
                      for stage in variant.segment_names
                      if bucket == self.full_bucket or stage in self.replay_stages}
        if capture:
            self.capture(warmup=warmup)

    def capture(self, *, warmup: int = 3) -> None:
        """Recapture fresh graph/stream pairs, retaining weights and static buffers.

        A stage captured once per bucket is one segment per bucket,
        `stage@bucket`. Every segment of a stage the replay axis reaches is
        prepared by selecting its bucket at the bucket's full rows, which runs
        the backends' replay hooks before it is warmed up and captured."""
        prepared = frozenset(self.replay_axis.stages if self.reads_extent else ())
        segments = [Segment(f"{stage}@{bucket}" if stage in self.replay_stages else stage,
                            partial(self.issue, stage, bucket),
                            prepare=(partial(self.select_replay, bucket) if stage in prepared
                                     else nothing_to_prepare))
                    for stage in self.graph.segment_names
                    for bucket in (self.replay_buckets if stage in self.replay_stages
                                   else (self.full_bucket,))]
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

    def select_replay(self, extent: int) -> None:
        """Replay the stages the replay axis reaches at the smallest bucket holding
        `extent`, this inference's value of the axis, and run the backends' replay
        hooks with its rows and the bucket's."""
        self.extent = extent
        self.bucket = next(bucket for bucket in self.replay_buckets if bucket >= extent)
        rows = self.replay_axis.rows(self.shape, extent)
        bucket_rows = self.replay_axis.rows(self.shape, self.bucket)
        for hook in self.scratch.replay_hooks:
            hook(rows, bucket_rows)

    def run_eager(self, segment: str) -> None:
        """Issue one stage's nodes outside its graph, on the current stream, at the
        selected bucket when the stage is captured per bucket."""
        if segment not in self.graph.segment_names:
            raise KeyError(f"no stage {segment!r}; stages: {self.graph.segment_names}")
        self.issue(segment, self.bucket if segment in self.replay_stages else self.full_bucket)

    def issue(self, stage: str, bucket: int | None) -> None:
        """Issue the nodes of `stage` at `bucket`, on the current stream."""
        ops = self.ops
        scratch = self.scratch
        for node, args in self.bound[(stage, bucket)]:
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

    @contextmanager
    def observe(self, scope: StepScope) -> Iterator[None]:
        """While active, every stage replay runs inside `scope("segment:<name>")`
        and every host slot inside `scope("host:<name>")`."""
        outer = self.step_scope
        self.step_scope = scope
        try:
            yield
        finally:
            self.step_scope = outer

    def replay(self, segment: str) -> None:
        """Replay one stage on its capture stream, ordered with the caller, at the
        selected bucket when the replay axis reaches it."""
        if self.graphs is None:
            raise RuntimeError("this runner was built without capture")
        captured = f"{segment}@{self.bucket}" if segment in self.replay_stages else segment
        if self.step_scope is None:
            self.graphs.replay(captured)
            return
        with self.step_scope(f"segment:{segment}"):
            self.graphs.replay(captured)

    # -- the engine protocol --------------------------------------------------

    @property
    def measurement_context(self) -> dict[str, dict[str, str]]:
        """The named weights and fixture in report form, a fresh copy on each read."""
        return {role: provenance.as_dict()
                for role, provenance in (("weights", self.weights_provenance),
                                         ("fixture", self.fixture_provenance))
                if provenance is not None}

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
        return self.target.model.sample_inputs(self.config, self.shape, seed, self.device,
                                               assets=self.assets)

    def stage(self, **inputs: torch.Tensor) -> None:
        """Copy the device inputs into their static addresses; run nothing. When
        the staged inputs decide the replay axis, select its bucket."""
        self.target.model.stage(buffers=self.buffers, inputs=inputs)
        if not self.reads_extent or self.replay_axis.slot is not None:
            return
        self.select_replay(self.replay_axis.extent(self.host_state, inputs))

    def host(self, slot: str, **inputs: torch.Tensor) -> None:
        """Run one declared host slot; the slot that decides the replay axis then
        selects its bucket."""
        if slot not in self.graph.host_slots:
            raise KeyError(f"no host slot {slot!r}; this Target declares {self.graph.host_slots}")
        if self.step_scope is None:
            self.run_host(slot, inputs)
            return
        # The replay hooks it runs are host work of the slot.
        with self.step_scope(f"host:{slot}"):
            self.run_host(slot, inputs)

    def run_host(self, slot: str, inputs: Mapping[str, torch.Tensor]) -> None:
        """Run host slot `slot`; the slot that decides the replay axis then selects its bucket."""
        self.target.model.host(slot, host_state=self.host_state, buffers=self.buffers,
                               inputs=inputs)
        if not self.reads_extent or slot != self.replay_axis.slot:
            return
        self.select_replay(self.replay_axis.extent(self.host_state, inputs))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Stage the inputs, run the program in order, return the output view."""
        self.stage(**inputs)
        for step in self.program:
            if step.kind == "host":
                self.host(step.name, **inputs)
            else:
                self.replay(step.name)
        return self.buffers[self.target.model.output]


__all__ = ["ModelRunner", "RunnerSource", "Scratch"]
