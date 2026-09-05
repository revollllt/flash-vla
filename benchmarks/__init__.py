"""Benchmarks for any Target, through the engine protocol.

    python -m benchmarks latency  --target h100/pi05 --plan a --plan b --plan a
    python -m benchmarks profile  --target h100/pi05 --plan a
    python -m benchmarks kernels  --target h100/pi05 --plan a
    python -m benchmarks floor    --target h100/pi05 --plan a

The model is an input, as a kernel is to the kernel benchmark: every command
builds the Target's engine through `benchmarks.targets`, the one place Targets
are named, and reads segments, host slots, call sites, costs and contracts
from the engine. None of them contains a model, stage or kernel name.

`latency` is the only latency baseline: chunk, device, host, segment and
overhead latency with min/median/p99, deltas only between legs of one
process against the control-leg spread. `profile` attributes time inside the
captured graphs to call sites, diagnostically. `kernels` launches one call
site at a time outside any graph, on the engine's own buffers and recorded
arguments, with the methodology of `flash_vla.bench`. `floor` derives the
latency objective from the Target's cost declarations and measured hardware
constants. All four need a GPU.

Per-kernel config sweeps are not here: a tuner has to drive a backend's
private compilation machinery, which this package deliberately cannot reach.
The sweep loop is in `flash_vla.tuning`, and the TileLang half of it sits with
the backend that owns the configs (`backends/tilelang/autotune`).
"""
