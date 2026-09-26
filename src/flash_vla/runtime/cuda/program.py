"""Graph segments over static buffers: warm up, freeze, capture, replay.

A Target declares its forward pass as an ordered list of `Segment`s -- each a
callable that issues its kernels in place on the static buffers -- and the
host slots between them. `Program` runs the one lifecycle every Target shares:

    warmup -> after_warmup (the runner freezes its workspace) -> capture -> replay

Warmup runs every segment enough times to compile every kernel and let every
backend request its workspace; the runner then freezes the allocator, so a
missed request raises instead of allocating mid-capture. Each segment
is captured into its own graph and is replayable alone, which is what makes
a per-segment latency split and stage-level oracle injection possible. A
Target with no host work and no measurement split declares one segment.

The runtime knows nothing about what a segment runs: a segment's callable
closes over the runner's bound graph nodes for that stage. A stage captured
once per replay-time bucket is one segment per bucket (`runtime/replay.py`);
each has host preparation (`Segment.prepare`) that runs before it is warmed up
and before it is captured, outside the capture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from .graph import StreamGraph


def nothing_to_prepare() -> None:
    """The preparation of a segment that needs none."""


@dataclass(frozen=True)
class Segment:
    """One captured graph: a name, the callable that issues its kernels, and the
    host work that must precede them (a kernel's plan for a bucket's rows)."""
    name: str
    run: Callable[[], None]
    prepare: Callable[[], None] = nothing_to_prepare


@dataclass(frozen=True)
class Step:
    """One position in a Target's forward: a graph segment or a host slot."""
    kind: Literal["segment", "host"]
    name: str


class Program:
    """The captured segments of one engine, replayable by name or in order."""

    def __init__(self, segments: Sequence[Segment], warmup: int = 3,
                 after_warmup: Callable[[], None] | None = None) -> None:
        names = [s.name for s in segments]
        if len(set(names)) != len(names):
            raise ValueError(f"segment names must be unique, got {names}")
        self.order: tuple[str, ...] = tuple(names)
        self._segments = {s.name: s for s in segments}
        self.graphs: dict[str, StreamGraph] = {}

        for _ in range(warmup):
            for segment in segments:
                segment.prepare()
                segment.run()
        torch.cuda.synchronize()
        if after_warmup is not None:
            after_warmup()

        for segment in segments:
            segment.prepare()
            graph = StreamGraph()
            with graph.capture():
                segment.run()
            self.graphs[segment.name] = graph
        torch.cuda.synchronize()

    def replay(self, name: str) -> None:
        """Replay one segment on its capture stream, ordered with the caller."""
        self.graphs[name].replay()

    def replay_all(self) -> None:
        """Replay every segment in declared order, with no host work between:
        every bucket of a stage captured per bucket, in turn."""
        for name in self.order:
            self.graphs[name].replay()

    def run_eager(self, name: str) -> None:
        """Issue one segment's kernels outside its graph, on the current stream.

        The same callable capture recorded, so a profiler sees each launch with
        its CPU-side correlation; nothing allocates because the workspace is frozen.
        """
        self._segments[name].run()

    def __len__(self) -> int:
        return len(self.order)
