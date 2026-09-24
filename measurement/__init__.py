"""The measurement layer: how every harness times, attributes and records.

Harnesses (`benchmarks/`, `eval/`, `tools/`, `lab/`) share their measuring
code through this package; `benchmarks` and `tools` never import each other:

    timing            timed loops (`wall_samples`, `event_samples`) and `summarize`
    kernel_bench      one kernel's GPU time, ported from FlashInfer's `bench_gpu_time`
    attribution       per-leg evidence of why a forward was late
    environment       the device and toolchain an observation ran on
    provenance        `MeasurementContext`: the weights, fixture and environment
    source_checkout   a LingBot Target whose backends come from another checkout
    results           saved-result pages and figures (`python -m measurement.results`)
    cli               the `--option key=value` convention

It depends on `flash_vla` and on nothing of the harnesses; `flash_vla` never
imports it (`tests/test_layering.py`).
"""
