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
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from .cost import SegmentCosts
from .cuda.program import Step
from .graph import Graph
from .identity import Identity


@runtime_checkable
class Engine(Protocol):
    """One constructed Target: graph built, plan bound, buffers allocated, stages captured."""

    #: The four axes, the shape numbers and the resolved plan.
    identity: Identity
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
    #: The op table in force: call site -> wrapper.
    ops: SimpleNamespace
    #: Kernel-name patterns the captured program must and must not contain.
    graph_contract: Mapping[str, Sequence[str]]
    #: Call sites the resolved plan must invoke together.
    atomic_groups: Sequence[frozenset[str]]

    def sample_inputs(self, seed: int = 0) -> dict[str, Any]:
        """Seeded inputs at this engine's shapes, for measurement and comparison."""

    def stage(self, **inputs: Any) -> None:
        """Copy the device inputs into their static addresses; run nothing."""

    def forward(self, **inputs: Any) -> torch.Tensor:
        """Stage the inputs, run every step in order, return the output view."""

    def replay(self, segment: str) -> None:
        """Replay one captured stage on the current stream."""

    def host(self, slot: str, **inputs: Any) -> None:
        """Run one host slot: the host work that sits between two stages."""

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""

    def run_eager(self, segment: str) -> None:
        """Issue one stage's nodes outside its graph, on the current stream."""

    def instrument(self, wrap: Callable[[str, Callable], Callable]) -> AbstractContextManager:
        """While active, every op-table entry `name` is replaced by `wrap(name, fn)`."""


def wrap_ops(ops: SimpleNamespace, wrap: Callable[[str, Callable], Callable]) -> SimpleNamespace:
    """A copy of an op table with every callable passed through `wrap`."""
    return SimpleNamespace(**{name: wrap(name, fn) for name, fn in vars(ops).items()})


def segments(engine: Engine) -> tuple[str, ...]:
    """The stage names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "segment")


def host_slots(engine: Engine) -> tuple[str, ...]:
    """The host slot names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "host")
