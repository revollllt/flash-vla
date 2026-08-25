"""Per-kernel throughput metrics and result formatting.

Ported from FlashInfer's attention metric helpers: given per-iteration times
and the shape configuration, report the same headline numbers their
benchmark does --

    [PERF] <backend> :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec; achieved tb_per_sec 1.87 TB/sec

The metric functions here are intentionally backend-agnostic. A caller that
benchmarks a fused kernel with a custom FLOP count supplies ``flops`` and
``bytes`` directly; attention-shaped workloads can use the ``attention_flops`` /
``attention_tb_per_sec`` helpers.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch


# ---------------------------------------------------------------------------
# FLOPs / bandwidth accounting
# ---------------------------------------------------------------------------


def attention_flops(
    batch_size: int,
    qo_seqlen: int,
    kv_seqlen: int,
    head_dim_qk: int,
    head_dim_vo: int,
    num_qo_heads: int,
    causal: bool,
) -> int:
    """FLOPs for one attention layer (bmm1 + bmm2)."""
    if causal and qo_seqlen > kv_seqlen:
        raise ValueError(
            "qo_seqlen must be <= kv_seqlen for causal attention"
        )
    if causal:
        eff = 2 * kv_seqlen - qo_seqlen
        bmm1 = batch_size * eff * qo_seqlen * num_qo_heads * head_dim_qk
        bmm2 = batch_size * eff * qo_seqlen * num_qo_heads * head_dim_vo
    else:
        bmm1 = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_qk
        bmm2 = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_vo
    return bmm1 + bmm2


def tflops_per_sec(flops: int, time_ms: float) -> float:
    """TFLOPS from a FLOP count and a per-iteration time in ms."""
    if time_ms is None or math.isnan(time_ms) or time_ms <= 0:
        return 0.0
    return flops / time_ms / 1e9


def tb_per_sec(total_bytes: int, time_ms: float) -> float:
    """TB/s from a byte count and a per-iteration time in ms (decimal TB)."""
    if time_ms is None or math.isnan(time_ms) or time_ms <= 0:
        return 0.0
    time_in_sec = time_ms / 1e3
    return (total_bytes / 1e12) / time_in_sec


def attention_tb_per_sec(
    batch_size: int,
    qo_seqlen: int,
    kv_seqlen: int,
    head_dim_qk: int,
    head_dim_vo: int,
    num_qo_heads: int,
    num_kv_heads: int,
    time_ms: float,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
    o_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Achieved memory bandwidth for one attention layer, in TB/s."""
    q_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_qk * q_dtype.itemsize
    k_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_qk * kv_dtype.itemsize
    v_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_vo * kv_dtype.itemsize
    o_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_vo * o_dtype.itemsize
    return tb_per_sec(q_bytes + k_bytes + v_bytes + o_bytes, time_ms)


# ---------------------------------------------------------------------------
# Result container / formatting
# ---------------------------------------------------------------------------


@dataclass
class KernelResult:
    """Timing + derived metrics for one benchmarked configuration.

    ``samples`` is the raw per-iteration ms list. All headline fields are
    derived from it at construction so a row can be printed the way
    FlashInfer's ``[PERF]`` line is.
    """

    label: str
    samples: list[float]
    flops: Optional[int] = None
    bytes: Optional[int] = None
    median_ms: float = field(init=False)
    min_ms: float = field(init=False)
    mean_ms: float = field(init=False)
    std_ms: float = field(init=False)
    p99_ms: float = field(init=False)
    tflops: float = field(init=False)
    tb_per_sec: float = field(init=False)

    def __post_init__(self):
        s = sorted(self.samples)
        n = len(s)
        self.median_ms = statistics.median(s)
        self.min_ms = s[0]
        self.mean_ms = statistics.fmean(s)
        self.std_ms = statistics.stdev(s) if n > 1 else 0.0
        self.p99_ms = s[min(n - 1, int(round((n - 1) * 0.99)))]
        self.tflops = tflops_per_sec(self.flops or 0, self.median_ms)
        self.tb_per_sec = tb_per_sec(self.bytes or 0, self.median_ms)

    def perf_line(self) -> str:
        """The FlashInfer-style [PERF] line."""
        parts = [f"median time {self.median_ms:.3f} ms", f"std {self.std_ms:.3f} ms"]
        if self.flops:
            parts.append(f"achieved tflops {self.tflops:.1f} TFLOPs/sec")
        if self.bytes:
            parts.append(f"achieved tb_per_sec {self.tb_per_sec:.2f} TB/sec")
        return f"{self.label:24} :: " + "; ".join(parts)


def _num(x: float) -> str:
    return f"{x:.3f}"


def render_table(results: Iterable[KernelResult]) -> str:
    """Render a comparison table, one row per KernelResult."""
    rows = list(results)
    if not rows:
        return "(no results)"
    header = f"{'label':24} {'median ms':>10} {'std ms':>8} {'min ms':>8} {'tflops':>9} {'TB/s':>9}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.label:24} {_num(r.median_ms):>10} {_num(r.std_ms):>8} "
            f"{_num(r.min_ms):>8} {(f'{r.tflops:.1f}' if r.flops else '-'):>9} "
            f"{(f'{r.tb_per_sec:.2f}' if r.bytes else '-'):>9}"
        )
    return "\n".join(lines)


def write_csv(path: str, results: Iterable[KernelResult]) -> None:
    """Append-or-create a CSV with one row per result."""
    rows = list(results)
    if not rows:
        return
    new_file = not __import__("os").path.exists(path)
    with open(path, "w" if new_file else "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["label", "median_ms", "min_ms", "mean_ms", "std_ms", "p99_ms",
                        "tflops", "tb_per_sec", "num_samples"],
        )
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({
                "label": r.label,
                "median_ms": r.median_ms,
                "min_ms": r.min_ms,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p99_ms": r.p99_ms,
                "tflops": r.tflops,
                "tb_per_sec": r.tb_per_sec,
                "num_samples": len(r.samples),
            })


__all__ = [
    "KernelResult",
    "attention_flops",
    "attention_tb_per_sec",
    "render_table",
    "tb_per_sec",
    "tflops_per_sec",
    "write_csv",
]
