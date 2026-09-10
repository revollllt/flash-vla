"""A CUDA graph captured and replayed on one dedicated stream."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class StreamGraph:
    """Own one graph/stream pair; order callers' input and output work around it."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        self._graph = torch.cuda.CUDAGraph()
        self._ready = torch.cuda.Event()
        self._done = torch.cuda.Event()

    @contextmanager
    def capture(self) -> Iterator[None]:
        with torch.cuda.graph(self._graph, stream=self.stream):
            yield

    def replay(self) -> None:
        caller = torch.cuda.current_stream(self.stream.device)
        if caller == self.stream:
            self._graph.replay()
            return
        self._ready.record(caller)
        self.stream.wait_event(self._ready)
        with torch.cuda.stream(self.stream):
            self._graph.replay()
            self._done.record(self.stream)
        caller.wait_event(self._done)
