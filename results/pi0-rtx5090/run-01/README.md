# Pi0 · RTX 5090 · run-01

The first end-to-end measurement of `rtx5090/pi0`. It is an **anchor, not a
result**: every call site runs in plain torch, nothing is fused and nothing is
tuned. The point is to have a number the optimization workflow can start from,
on this machine, under this repository's protocol.

## Workload and measurement conditions

| | |
|---|---|
| Target | `hardware/nvidia/rtx5090/pi0`, bf16, 18 layers, 10 denoise steps |
| Checkpoint | seeded synthetic weights, `seed=0` — Pi0 needs no asset |
| Fixture | `flash-vla/pi0-inputs-v1/seed-0` |
| Shape | 3 views × 224², 768 visual + 200 prompt tokens, chunk 50 |
| Environment | RTX 5090, driver 580.142, torch 2.13.0+cu130, CUDA 13.1 |
| Protocol | `latency-v2`: fresh process, first capture, warmup 5, 100 reps, median |

```bash
python -m benchmarks latency --target rtx5090/pi0 --plan shipped --seed 0 \
  --warmup 5 --reps 100 --out results/pi0-rtx5090/run-01/measurements/000.json
```

## Result

| # | change | median | delta |
|---|---|---:|---:|
| 0 | every call site in torch | **46.794 ms** | |

`min` 46.731 ms over 100 reps, so the run itself is quiet — well inside the 6%
noise floor the [measured table](../../../src/flash_vla/hardware/nvidia/rtx5090/measured/README.md)
records. The GPU sat at P1, 55 °C, 459 W during the run.

**Do not compare this with any H100 figure.** A different GPU is a different
comparison context; the two machines differ by 1.8x on streaming bandwidth, 3.4x
on tensor-core throughput and 2.29x on shared memory per block. This run starts
its own curve.

## What the correctness check does and does not say

`eval.correctness` compares a Target's **reference plan against its candidate
plan**. Both are the torch route here, so it reports cosine 1.0 and zero error
by construction:

```bash
python -m eval.correctness --target rtx5090/pi0 --plan reference --steps 1 --layers 1
# passed: true, min_cosine 1.0, replay_identical: true
```

That is worth exactly three things, and they are not nothing: the model **runs
end to end** on sm_120, its CUDA graph **replays deterministically**, and every
output is **finite**. It is not evidence about the arithmetic.

Pi0 has no official oracle on this machine — `eval.pi0.reference` needs
`OPENPI_PI0_CHECKPOINT` and the OpenPI interpreter, neither configured here. The
arithmetic is instead checked per operator against the TileLang wrappers the
torch backend was written from, wherever their tile configuration fits this
part's shared memory: `lab/sm120/pi0_torch_parity.py`. See
`correctness/000-op-parity.txt`.

## Where the time goes, and what to do about it

Not yet profiled. The CUPTI timeline is available on this host even without
counter permission
([ncu-metrics.md](../../../src/flash_vla/hardware/nvidia/rtx5090/measured/ncu-metrics.md)),
so the next step is the top-down profile from
[the optimization workflow](../../../docs/optimization.md) step 4:

```bash
python -m tools.profiling.model --target rtx5090/pi0 --plan shipped --seed 0 \
  --overview --trace-dir artifacts/profile/overview
```

What is already known about the machine, and will shape every kernel written
against it:

* **99 KB of shared memory per block**, not 227 [smem.bytes.cta.max]. Ten of
  H100/Pi0's nineteen TileLang tile shapes do not fit, which is why none of them
  were inherited.
* **`mma.sync` is the only tensor-core path** and it already runs at 100% of the
  hardware's 512 FLOP/cycle/SM for bf16 with fp32 accumulate [mma.rate.sm.bf16].
  A tensor-bound kernel cannot be tuned out of it — but fp16 accumulation and
  fp8 input are 2x levers that H100 does not have [mma.ratio.sm.acc].
* **The floor model is `3.35 + MB/1.524` µs per launch** [ld.bw.dev.dram], and a
  launch inside a CUDA graph costs 0.45 µs [launch.lat.dev.ramp].
