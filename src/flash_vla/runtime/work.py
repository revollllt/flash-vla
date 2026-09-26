"""What a model declares for the work of its forward (`measurement.work` measures it).

Each model's `models/<model>/work.py` declares `RULES`, which attribute the
ops of its reference's forward to runtime call sites, and `reference_run`,
which builds that reference on the meta device at a workload's shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class CallSiteRule:
    """Ops run inside a module whose path matches `module` (a regex searched in the
    dotted path) and, when `op` is given, whose aten name fully matches it, belong
    to `call_site`. An op no rule matches belongs to the previous counted op's
    call site: the residual add after a projection, the RoPE after Q/K/V."""
    module: str
    op: str | None
    call_site: str


@dataclass(frozen=True)
class ReferenceRun:
    """A model's reference ready to trace at one shape: `forward` runs it on the
    meta device and returns the tensors the forward exists to produce, `inputs`
    are the inference inputs it reads."""
    model: nn.Module
    forward: Callable[[], tuple[torch.Tensor, ...]]
    inputs: tuple[torch.Tensor, ...]


__all__ = ["CallSiteRule", "ReferenceRun"]
