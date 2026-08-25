"""Benchmarks for the TileLang Pi0 port.

    python -m benchmarks e2e      full-pipeline wall clock
    python -m benchmarks profile  per-kernel GPU time inside the graph replay
    python -m benchmarks kernels  one kernel launch at a time, in isolation

All three commands need a GPU.

`profile` and `kernels` measure different things: `profile` attributes time
inside the captured graph, so it reports what a kernel costs in the pipeline it
actually runs in; `kernels` launches one kernel against cold L2 outside any
graph, so it reports what that kernel costs on its own. The timing methodology
`kernels` uses is generic and lives in `flash_vla.bench`, which is why a kernel
that is not part of flash-vla can be benchmarked the same way.

Per-kernel config sweeps are not here: a tuner has to drive a backend's private
compilation machinery, which this package deliberately cannot reach. The sweep
loop is in `flash_vla.tuning`, and the TileLang half of it -- device axes and
kernel rewrapping -- sits with the backend that owns the configs, at
`flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.autotune`.
"""
