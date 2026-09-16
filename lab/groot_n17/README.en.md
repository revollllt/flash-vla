# GR00T N1.7 / RTX 5090 roofline estimate

[中文](README.md) | [English](README.en.md)

2026-09-17; reference implementation `cf2d196`. This directory is versioned in the main project and supplied to all four runs through their common starting commit, using the project's existing `Cost`, `gemm`, `attention` and `tools.profiling.floor.site_row`. Inference code was not changed.

**The conditional estimate for the current dense computation is 8.261 ms. Reaching 90% of that estimated performance means full-model latency ≤9.179 ms.** This is an optimization objective with unproven attainability; the run still stops when its budget is exhausted.

| Stage | Dense FLOPs | Streamed weights | Ideal latency estimate |
|---|---:|---:|---:|
| Vision | 349.503 GFLOPs | 0.808 GB | 1.667 ms |
| Language backbone | 254.445 GFLOPs | 1.611 GB | 1.214 ms |
| Action head | 469.315 GFLOPs | 9.203 GB | 5.380 ms |
| Total | 1.073 TFLOPs | 11.622 GB | **8.261 ms** |

Each operation costs `max(FLOPs / 209.6 TFLOP/s, weight bytes / 1.792 TB/s)`, summed over actual invocation counts. These are the RTX 5090 BF16-input/FP32-accumulation dense peaks from project `rtx5090/spec.py`, without FP8, sparsity or FP16-accumulation throughput.

Workload: `libero_10`, BF16, batch 1, two 256×256 views, 156 tokens, 16 backbone layers, 32 DiT blocks and 4 denoising steps. Internal output is `[1,40,132]`. Each observation includes fresh vision, language and complete action computation.

Key assumptions:

- Ideal activation reuse in cache/on chip; each selected weight matrix streams once per invocation without persistent reuse across invocations. Repeated reads across four denoising steps are counted. Only the active embodiment is read, not all 32 banks.
- Embedding lookup does not stream the entire vocabulary. Lookup, bias, normalization and other small operations, scheduling, synchronization and input staging are omitted. Image preprocessing, CPU-to-GPU input delivery and action decoding remain outside experimental timing.
- Actual weight residency in the 96 MiB L2 has not been measured. Constant folding, cross-step KV reuse, fusion or reordered computation requires revisiting these counts and traffic assumptions. 8.261 ms is not a strict physical lower bound over all equivalent implementations.

Pricing every intermediate activation read/write at DRAM gives 8.458 ms as a traffic-sensitivity comparison. Applying existing measured primitive constants gives 12.736 ms, including per-invocation cold-read fixed costs. Its compute throughput was measured at different clocks, and it is not an observed attainable model latency. Neither replaces the explicitly chosen objective above.

The unmodified generic `tools.profiling.floor` returned 5.154 ms with `valid=false`: three coarse stage declarations miss loop traffic, and the tool compares the specification's 209.6 TFLOP/s with 253 TFLOP/s measured under different clocks as though conditions matched. **That invalid report was not used for the objective**, and no tolerance was changed to make it pass.

Expanded matrix/attention FLOPs agree with all three stage declarations and were checked against one real eager forward using `FlopCounterMode`. The language stage additionally executes 59,904 FP32 FLOPs for the RoPE frequency outer product, recorded separately. Final `[1,40,132]` outputs were finite. Memory traffic remains an assumption, not DRAM bytes measured by the FLOP check.

Recompute from any Flash-VLA checkout root using the GR00T environment; no GPU or weights are needed:

```bash
PYTHONPATH=src:. python -m lab.groot_n17.roofline --out artifacts/groot-n17/roofline.json
```

The common numerical record is [roofline.json](../../results/groot-n17-rtx5090/reference/roofline.json), and the real-forward check is [roofline-execution-check.json](../../results/groot-n17-rtx5090/reference/roofline-execution-check.json). The script is [roofline.py](roofline.py). All four runs inherit these files from the same common commit. Launch prompts reference this page, with 8.261 ms as the original comparison denominator and ≤9.179 ms as the objective. Record evidence for any modeling correction and apply it consistently during analysis; do not change only one run's comparison denominator.
