"""Benchmarks for the TileLang Pi0 port.

    python -m benchmarks e2e      full-pipeline wall clock
    python -m benchmarks profile  per-kernel GPU time inside the graph replay

Both commands need a GPU.

Per-kernel config sweeps are not here: a tuner has to drive a backend's private
compilation machinery, which this package deliberately cannot reach. The sweep
loop is in `flash_vla.tuning`, and the TileLang half of it -- device axes and
kernel rewrapping -- sits with the backend that owns the configs, at
`flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.autotune`.
"""
