"""Benchmarks for the TileLang Pi0 port.

    python -m tilelang_infer.bench e2e      full-pipeline wall clock, all backends
    python -m tilelang_infer.bench profile  per-kernel GPU time inside the graph replay
    python -m tilelang_infer.bench parity   numerical parity vs the Triton reference

All three need a GPU, which on this cluster means submitting through
`sbatch/run_gpu.sbatch`. `autotune` is a library, not a command: it sweeps one
kernel's tiling x warp-specialization at its real shape.
"""
from __future__ import annotations

from . import autotune, e2e, metrics, parity, profile, synthetic

__all__ = ["autotune", "e2e", "metrics", "parity", "profile", "synthetic"]
