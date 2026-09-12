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

## Where the time goes

Top-down timeline, CUPTI activity tracing, unprivileged
(`artifacts/profile/overview/`). GPU activity 46.879 ms against the benchmark's
46.794, so the profile is not distorting much.

```bash
python -m tools.profiling.model --target rtx5090/pi0 --plan shipped --seed 0 \
  --overview --trace-dir artifacts/profile/overview
```

| segment | GPU ms | % | launches | launch cost at 0.45 µs |
|---|---:|---:|---:|---:|
| `action_expert` | 22.701 | 48.4% | **11193** | **5.04 ms — 22% of the segment** |
| `llm_backbone` | 18.145 | 38.7% | 832 | 0.37 ms |
| `vision_encoder` | 6.050 | 12.9% | 789 | 0.36 ms |

**The two big segments are bound by opposite things**, which is the whole result.

`action_expert` holds **87% of all launches for 48% of the time**. Its GEMMs
average 7.88 µs and its element-wise kernels 1.20 µs — 18 layers × 10 denoise
steps is 180 layer-steps, so it runs ~62 launches per layer-step on tiny
matrices. At `[launch.lat.dev.ramp]`'s 0.45 µs, a fifth of the segment is launch
overhead before any work happens. This is the deep-and-narrow shape run-02
diagnosed on LingBot's H100 expert, on a different model and a different machine.

| `action_expert` | ms | % | launches | µs each |
|---|---:|---:|---:|---:|
| cuBLAS GEMM | 10.168 | 44.8% | 1290 | 7.88 |
| element-wise (torch, unfused) | 9.001 | 39.7% | **7511** | 1.20 |
| cuBLAS splitKreduce | 1.219 | 5.4% | 730 | 1.67 |
| reduce (torch) | 0.916 | 4.0% | 370 | 2.48 |
| memcpy | 0.513 | 2.3% | 571 | 0.90 |

`llm_backbone` is the opposite: **121 GEMMs at 121.93 µs each carry 81% of it**,
in 832 launches total. Big matrices, launch cost irrelevant.

| `llm_backbone` | ms | % | launches | µs each |
|---|---:|---:|---:|---:|
| cuBLAS GEMM | 14.754 | 81.3% | 121 | 121.93 |
| element-wise (torch, unfused) | 2.708 | 14.9% | 569 | 4.76 |

`vision_encoder` sits between them: 109 GEMMs at 35.07 µs for 63%, plus 0.367 ms
of `flash_fwd_kernel` — the one call site that was already torch on H100 and is
already a fused SDPA here.

## What to do about it

Two different problems, so two different levers, and they should not be
confused:

1. **`action_expert`: fuse, to delete launches.** 7511 element-wise launches for
   9.0 ms is 1.20 µs apiece against a 0.45 µs floor — these are not doing work,
   they are paying tolls. The norms, activations, residual adds and RoPE that
   torch spells as separate kernels are the obvious first fusion, and this is
   where a hand-written kernel earns the most.
2. **`llm_backbone`: check the GEMM against the tensor core before touching it.**
   14.754 ms of cuBLAS at 122 µs per launch may already be at the hardware
   ceiling — `[mma.rate.sm.bf16]` says `mma.sync` reaches 100% of 512
   FLOP/cycle/SM and cuBLAS is not obviously leaving that on the table. If it is
   at the roofline, no kernel written here can beat it and the segment is done.

The floor model settles (2) and is the next thing to run:

```bash
python -m tools.profiling.floor --target rtx5090/pi0 --plan shipped --seed 0
```

Machine facts that will shape whatever gets written:

* **99 KB of shared memory per block**, not 227 [smem.bytes.cta.max].
* **`mma.sync` is the only tensor-core path** and already runs at 100% of
  512 FLOP/cycle/SM for bf16 with fp32 accumulate [mma.rate.sm.bf16]. fp16
  accumulation and fp8 input are 2× levers H100 does not have
  [mma.ratio.sm.acc].
* **A cold read costs `3.35 + MB/1.524` µs** [ld.bw.dev.dram]; an in-graph
  launch costs 0.45 µs [launch.lat.dev.ramp].
* NCU counters need `sudo` here, but no reboot — see
  [ncu-metrics.md](../../../src/flash_vla/hardware/nvidia/rtx5090/measured/ncu-metrics.md).
