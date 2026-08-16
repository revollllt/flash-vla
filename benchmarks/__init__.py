"""Benchmarks for the TileLang Pi0 port.

    python -m benchmarks e2e      full-pipeline wall clock
    python -m benchmarks profile  per-kernel GPU time inside the graph replay

Both commands need a GPU. `autotune` is a library, not a command: it sweeps one
kernel's tiling x warp-specialization at its real shape.
"""
