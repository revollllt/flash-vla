"""CUDA graph-safe runtime utilities."""

from .arena import ScratchPool
from .timing import capture, graph_time_cold

__all__ = ["ScratchPool", "capture", "graph_time_cold"]
