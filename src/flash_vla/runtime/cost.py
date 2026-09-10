"""Cost declarations: what a call site must move and compute, as data.

A Target declares, per segment, the call sites it runs, how many times, and
the minimal work of each: bytes read once, bytes written once, floating-point
operations. The declaration is a property of the call site's contract at the
Target's shapes and precision, not of any backend: it is the numerator the
floor model divides by measured hardware constants
(`tools/profiling/floor.py`). No torch dependency, so a
floor can be computed on a login node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

#: Bytes per element of the `bf16` precision policy.
BF16 = 2


@dataclass(frozen=True)
class Cost:
    """Minimal traffic and math of one invocation of a call site."""
    bytes_read: int
    bytes_written: int
    flops: int

    @property
    def bytes(self) -> int:
        return self.bytes_read + self.bytes_written


@dataclass(frozen=True)
class Ceiling:
    """What this machine has delivered for one call of a call site's geometry.

    Declared by a Target when a hardware unit test measured the call site's
    own geometry (bytes, CTAs, box) cold, so the floor model's ceiling column
    uses the observed number rather than the constants' rule. `tag` names the
    measured constant it was read against and `job` the Slurm job that
    produced it; both are copied into every floor report that uses it.
    """
    us: float
    tag: str
    job: int


@dataclass(frozen=True)
class Invocation:
    """A call site in a segment: its cost per call, how often it is called, and
    optionally the measured ceiling of one call."""
    call_site: str
    cost: Cost
    count: int
    ceiling: Ceiling | None = None

    @property
    def bytes(self) -> int:
        return self.cost.bytes * self.count

    @property
    def flops(self) -> int:
        return self.cost.flops * self.count


#: segment -> the invocations it runs, in pipeline order.
SegmentCosts = Mapping[str, Sequence[Invocation]]


def gemm(m: int, k: int, n: int, *, residual: bool = False, extra_read: int = 0,
         elem: int = BF16) -> Cost:
    """M x K @ K x N: activations once, weight once, output once (+ a residual read)."""
    read = (m * k + k * n) * elem + extra_read
    if residual:
        read += m * n * elem
    return Cost(bytes_read=read, bytes_written=m * n * elem, flops=2 * m * k * n)


def dual_gemm(m: int, k: int, n: int, *, elem: int = BF16) -> Cost:
    """Two M x K @ K x N GEMMs sharing the activation (a gated FFN's gate and up)."""
    return Cost(bytes_read=(m * k + 2 * k * n) * elem, bytes_written=m * n * elem,
                flops=2 * (2 * m * k * n))


def attention(queries: int, keys: int, head_dim: int, q_heads: int, kv_heads: int,
              *, elem: int = BF16) -> Cost:
    """QK^T and PV over `keys` for `queries` rows; Q, K, V read once, O written once."""
    read = (queries * q_heads * head_dim + 2 * keys * kv_heads * head_dim) * elem
    return Cost(bytes_read=read, bytes_written=queries * q_heads * head_dim * elem,
                flops=4 * queries * keys * head_dim * q_heads)


def total(costs: SegmentCosts) -> dict[str, dict[str, int]]:
    """Per-segment sums of bytes and flops."""
    return {segment: {"bytes": sum(i.bytes for i in invocations),
                      "flops": sum(i.flops for i in invocations),
                      "invocations": sum(i.count for i in invocations)}
            for segment, invocations in costs.items()}


__all__ = ["BF16", "Ceiling", "Cost", "Invocation", "SegmentCosts", "attention", "dual_gemm",
           "gemm", "total"]
