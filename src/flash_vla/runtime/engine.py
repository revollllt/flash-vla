"""The engine protocol: what every Target's engine exposes to generic harnesses.

A latency runner, a correctness runner and the promotion gate are written
against this surface and nothing else. A Target satisfies it by construction
through the runtime (`StaticArena`, `Program`); the model-specific facts a
harness needs -- which buffers hold the KV cache, which rows are valid -- are
exposed through named buffers and the identity's shape numbers, never by
importing the Target.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from .cost import SegmentCosts
from .cuda.program import Step
from .identity import Identity


@runtime_checkable
class Engine(Protocol):
    """One constructed Target: weights loaded, buffers materialized, segments captured."""

    #: The four axes, the shape numbers and the resolved plan.
    identity: Identity
    #: Named static buffers, the views the pipeline runs on.
    buffers: Mapping[str, torch.Tensor]
    #: The forward pass as an ordered list of graph segments and host slots.
    program: tuple[Step, ...]
    #: The buffers each segment produces as its contract, with the axis that
    #: indexes layers where one exists: segment -> ((buffer, layer_axis), ...).
    #: A correctness harness compares these and may inject an oracle's values
    #: into them; a padded region behind them must stay finite.
    stage_outputs: Mapping[str, tuple[tuple[str, int | None], ...]]
    #: Per segment, the call sites it invokes with their minimal bytes and
    #: FLOPs at this engine's shapes: the numerator of the floor model.
    costs: SegmentCosts
    #: What the captured program must and must not contain, as kernel-name
    #: patterns declared by the backends the plan routes to: `forbid` names
    #: that must not appear, `require_one` patterns exactly one distinct
    #: kernel name must match.
    graph_contract: Mapping[str, Sequence[str]]
    #: Call sites the resolved plan must invoke together, in pipeline order
    #: (they share implementation-owned state across the calls); a harness
    #: that times call sites one at a time times such a group as one case.
    atomic_groups: Sequence[tuple[str, ...]]

    def sample_inputs(self, seed: int = 0) -> dict[str, Any]:
        """Seeded inputs at this engine's shapes, for measurement and comparison."""

    def stage(self, **inputs: Any) -> None:
        """Copy the device inputs into their static addresses; run nothing."""

    def forward(self, **inputs: Any) -> torch.Tensor:
        """Stage the inputs, run every step in order, return the output view."""

    def replay(self, segment: str) -> None:
        """Replay one captured segment on the current stream."""

    def host(self, slot: str, **inputs: Any) -> None:
        """Run one host slot: the host work that sits between two segments."""

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""

    def run_eager(self, segment: str) -> None:
        """Issue one segment's kernels outside its graph, on the current stream."""

    def instrument(self, wrap: Callable[[str, Callable], Callable]) -> AbstractContextManager:
        """While active, every op-table entry `name` is replaced by `wrap(name, fn)`."""

    def scratch_scope(self) -> AbstractContextManager:
        """Route scratch to this engine's pool, for calling an op outside a segment."""


def wrap_ops(ops: SimpleNamespace, wrap: Callable[[str, Callable], Callable]) -> SimpleNamespace:
    """A copy of an op table with every callable passed through `wrap`."""
    return SimpleNamespace(**{name: wrap(name, fn) for name, fn in vars(ops).items()})


def segments(engine: Engine) -> tuple[str, ...]:
    """The segment names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "segment")


def host_slots(engine: Engine) -> tuple[str, ...]:
    """The host slot names of `engine`, in order."""
    return tuple(step.name for step in engine.program if step.kind == "host")
