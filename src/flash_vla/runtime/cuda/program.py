"""Graph segments over static buffers: warm up, freeze, capture, replay.

A Target declares its forward pass as an ordered list of `Segment`s -- each a
callable that issues its kernels in place on the static buffers -- and the
host slots between them. `Program` runs the one lifecycle every Target shares:

    warmup -> freeze the scratch pool -> capture each segment once -> replay

Warmup runs every segment enough times to compile every kernel and fill the
pool with every scratch key; from `freeze` onward nothing may allocate, so a
missed pre-allocation raises instead of allocating mid-capture. Each segment
is captured into its own graph and is replayable alone, which is what makes
a per-segment latency split and stage-level oracle injection possible. A
Target with no host work and no measurement split declares one segment.

The runtime knows nothing about what a segment runs: a segment's callable
closes over the Target's pipeline, weights, buffers and scratch-pool scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from .arena import ScratchPool


@dataclass(frozen=True)
class Segment:
    """One captured graph: a name and the callable that issues its kernels."""
    name: str
    run: Callable[[], None]


@dataclass(frozen=True)
class Step:
    """One position in a Target's forward: a graph segment or a host slot."""
    kind: Literal["segment", "host"]
    name: str


class Program:
    """The captured segments of one engine, replayable by name or in order."""

    def __init__(self, segments: Sequence[Segment], pool: ScratchPool,
                 warmup: int = 3) -> None:
        names = [s.name for s in segments]
        if len(set(names)) != len(names):
            raise ValueError(f"segment names must be unique, got {names}")
        self.order: tuple[str, ...] = tuple(names)
        self._segments = {s.name: s for s in segments}
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}

        for _ in range(warmup):
            for segment in segments:
                segment.run()
        torch.cuda.synchronize()
        pool.freeze()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for segment in segments:
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin()
                segment.run()
                graph.capture_end()
                self.graphs[segment.name] = graph
        torch.cuda.synchronize()

    def replay(self, name: str) -> None:
        """Replay one segment on the current stream."""
        self.graphs[name].replay()

    def replay_all(self) -> None:
        """Replay every segment in declared order, with no host work between."""
        for name in self.order:
            self.graphs[name].replay()

    def run_eager(self, name: str) -> None:
        """Issue one segment's kernels outside its graph, on the current stream.

        The same callable capture recorded, so a profiler sees each launch with
        its CPU-side correlation; nothing allocates because the pool is frozen.
        """
        self._segments[name].run()

    def __len__(self) -> int:
        return len(self.order)
