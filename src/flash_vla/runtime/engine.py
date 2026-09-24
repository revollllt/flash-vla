"""The engine protocol: what a runner exposes to generic harnesses.

The latency, profile, kernel and floor runners, the correctness runner and the
promotion gate are written against this surface and nothing else. Its one
implementation is `ModelRunner` (`runtime/runner.py`); the model-specific
facts a harness needs -- which buffers hold the KV cache, which rows are
valid -- reach it through the graph's named buffers, the Target's declared
stage outputs and the identity's shape numbers, never by importing a Target.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from flash_vla.provenance import ImplementationProvenance

from .cost import SegmentCosts
from .cuda.program import Step
from .graph import Graph
from .identity import Identity
from .registry import GraphContract, Wrapper

#: Replaces op-table entry `call_site` while `Engine.instrument` is active.
WrapOp = Callable[[str, Wrapper], Wrapper]
#: The scope `Engine.observe` runs each step in, given the step's label.
StepScope = Callable[[str], AbstractContextManager[object]]


@runtime_checkable
class Engine(Protocol):
    """One constructed Target: graph built, plan bound, buffers allocated, stages captured."""

    #: The four axes, the shape numbers and the resolved plan.
    identity: Identity
    #: The checkout the backends were loaded from, when it is not this one.
    implementation_source: ImplementationProvenance | None
    #: The device the buffers and weights live on.
    device: torch.device
    #: The quantization recipe in force (`Target.quantization`), or the precision policy.
    quantization: str
    #: The shape numbers of the identity, and derived numbers (a prefix length).
    shape: Mapping[str, int]
    derived: Mapping[str, int]
    #: The raw plan the runner was built with (before route resolution).
    plan: Mapping[str, str]
    #: The explicit computation graph: stages, nodes, buffer declarations.
    graph: Graph
    #: Named static buffers, the views the graph runs on.
    buffers: Mapping[str, torch.Tensor]
    #: The forward pass as an ordered list of stages and host slots.
    program: tuple[Step, ...]
    #: Per stage, the buffers it produces as its contract, with the axis that
    #: indexes layers where one exists. A correctness harness compares these
    #: and may inject an oracle's values into them.
    stage_outputs: Mapping[str, tuple[tuple[str, int | None], ...]]
    #: Per stage, every call site with its minimal bytes and FLOPs, from the graph.
    costs: SegmentCosts
    #: What the model's host slots keep across calls (`None` when it has none).
    host_state: object
    #: The op table in force: call site -> wrapper.
    ops: Mapping[str, Wrapper]
    #: Kernel-name patterns the captured program must and must not contain.
    graph_contract: GraphContract
    #: Call sites the resolved plan must invoke together.
    atomic_groups: Sequence[frozenset[str]]

    @property
    def measurement_context(self) -> Mapping[str, Mapping[str, str]]:
        """The weights and fixture the caller named, in report form
        (`WeightsProvenance.as_dict`, `FixtureProvenance.as_dict`)."""

    def sample_inputs(self, seed: int = 0) -> dict[str, torch.Tensor]:
        """Seeded inputs at this engine's shapes, for measurement and comparison."""

    def stage(self, **inputs: torch.Tensor) -> None:
        """Copy the device inputs into their static addresses; run nothing."""

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Stage the inputs, run every step in order, return the output view."""

    def capture(self, *, warmup: int = 3) -> None:
        """Capture fresh graph/stream pairs without reloading weights."""

    def replay(self, segment: str) -> None:
        """Replay one stage on its capture stream, ordered with the caller."""

    def host(self, slot: str, **inputs: torch.Tensor) -> None:
        """Run one host slot: the host work that sits between two stages."""

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""

    def run_eager(self, segment: str) -> None:
        """Issue one stage's nodes outside its graph, on the current stream."""

    def instrument(self, wrap: WrapOp) -> AbstractContextManager[None]:
        """While active, every op-table entry `name` is replaced by `wrap(name, fn)`."""

    def observe(self, scope: StepScope) -> AbstractContextManager[None]:
        """While active, every stage replay runs inside `scope("segment:<name>")`
        and every host slot inside `scope("host:<name>")`."""


def wrap_ops(ops: Mapping[str, Wrapper], wrap: WrapOp) -> Mapping[str, Wrapper]:
    """A copy of an op table with every wrapper passed through `wrap`."""
    return MappingProxyType({name: wrap(name, function) for name, function in ops.items()})


def segments(engine: Engine) -> tuple[str, ...]:
    """The stage names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "segment")


def host_slots(engine: Engine) -> tuple[str, ...]:
    """The host slot names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "host")
