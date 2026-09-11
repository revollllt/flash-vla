# LingBot-VLA-4B · H100 · run-01

Deployed end-to-end latency of `h100/lingbot_vla` on its real post-training
checkpoint and frozen seed-42 fixture, starting from the current `shipped`
plan. The objective is the deployed `chunk_latency` median.

![Optimization progress](progress.svg)

## Workload and measurement conditions

| | |
|---|---|
| Target | `hardware/nvidia/h100/lingbot_vla`, bf16, 36 layers, 10 denoise steps |
| Checkpoint | `lingbot-vla-4b-posttrain-robotwin@fb71a2c…+qwen2.5-vl-3b@66285546…` |
| Fixture | `lingbot-robotwin-canonical-v1/seed-42` (prompt "adjust bottle") |
| Shape | 3 views × 224², 192 visual + 72 language = 264 prefix tokens, 51 suffix tokens |
| Environment | Python 3.12.12, torch 2.9.1+cu126, H100 80GB HBM3, `acd_u` |
| Protocol | `latency-v2`: fresh process and first capture per leg, warmup 5, 100 reps, no soak, median |

Commands run from the project root with `FLASH_VLA_ASSETS` pointing at the
machine-local asset map and `LINGBOT_PYTHON` selecting the upstream environment:

```bash
python -m benchmarks latency --target h100/lingbot_vla --plan shipped --seed 42 \
  --warmup 5 --reps 100 --out results/lingbot-h100/run-01/measurements/000.json
python -m eval.lingbot.parity --plan shipped --seed 42 --layers 36 --steps 10 \
  --out results/lingbot-h100/run-01/correctness/000-shipped-parity.json
```

Each model trial is measured against the retained version in the **same job on
the same physical GPU**, because the partition mixes two driver generations
(570.86.10 and 610.43.02) and a leg's measurement context records the driver.
`latency_ms` in `iterations.csv` is that trial's own median.

## Iteration 0 — start point

`shipped` = `fused-norm` on all three call sites (`ef798e0`).

Three legs of the start version, one job (614199, ACD1-6, driver 570.86.10):

| leg | min | median | p99 |
|---|---:|---:|---:|
| 0 | 61.208 | **61.253** | 61.474 |
| 1 | 61.605 | 62.415 | 62.630 |
| 2 | 61.209 | 61.254 | 61.530 |

Repeat-leg spread: 0.397 ms on `min`, **1.162 ms on `median`**. A single
unpaired median difference below ~1.2 ms is therefore not distinguishable from
run-to-run drift; paired same-job A/B is used for every decision.

Correctness (614205, ACD1-21, driver 610.43.02): full-depth 10-step parity
against the frozen upstream eager oracle **passes**; `vision_embeddings`,
`prefix_k` and `prefix_v` are bit-identical, and the denoised outputs are within
the bf16 `deepest` tolerance (`actions` cos 0.9999986, `physical_actions`
cos 0.9999752). Replay is deterministic.

Overview profile (same job): GPU activity 57.26 ms inside a 64.30 ms span, so
7.04 ms sits in host/launch gaps under the profiler.

## Where the time was

`python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42`
on the start version (job 614222, torch-profiler replay, diagnostic only —
profiled totals run ~15% above the uninstrumented benchmark):

| segment | in-graph | launches | under one wave |
|---|---:|---:|---:|
| `vision_encoder` | 10.40 ms | 1272 | 498 |
| `llm_backbone` | 10.46 ms | 2188 | 559 |
| `action_expert` | **50.47 ms** | **17104** | 10944 |

The expert issues 47 kernels per layer-step across 36 layers and 10 denoise
steps, and 14.62 ms of its 50.47 ms is copy kernels. Against the machine's
measured `launch.lat.dev.ramp` of 1.24 µs per launch, the launch count alone
accounts for most of the segment, so every accepted change below removes
launches at unchanged arithmetic rather than making a kernel faster.

The vision encoder's three feed-forward projections ran
`cutlass_80_tensorop_bf16_s16816gemm_bf16_256x128_64x3_tn_align2` at 57.4 µs
each (5.51 ms of 10.40 ms) because its 3420-wide FFN is not 8-element aligned.

## Accepted iterations

Each row is a paired A/B/A in one job on one GPU: the retained plan, the
candidate, then the retained plan again. `delta` is the candidate's median
against leg 0; `spread` is leg 2 against leg 0.

| # | change | base median | candidate median | delta | spread | job |
|---|---|---:|---:|---:|---:|---|
| 1 | packed q/k/v and gate/up GEMMs, grouped attention | 62.162 | 55.494 | **−6.668** | 0.069 | 614224 |
| 2 | Target-owned denoising loop, resident KV cache | 51.204 | 48.969 | **−2.235** | 0.008 | 614238 |
| 3 | fused CUDA projection epilogue (widen + RoPE + cache write) | 52.578 | 42.267 | **−10.313** | 0.002 | 614268 |
| 4 | head-major cache, fused softmax and attention epilogue | 39.754 | 37.300 | **−2.454** | 0.018 | 614286 |
| 5 | Target-owned backbone prefix pass on the same kernels | 37.308 | 34.983 | **−2.324** | 0.008 | 614301 |

Iterations 1 and 2 were measured on driver 570.86.10 and 3-5 on 610.43.02; the
two generations differ by ~8% on this workload, which is why every decision is
a paired same-job comparison and why the curve is re-measured in one final
ladder job rather than stitched from these legs.

### Correctness

Full-depth ten-step parity against the frozen upstream eager oracle
(`eval.lingbot.parity`) passes at every iteration, with `replay_identical`.
Through iteration 5 the vision embeddings stay bit-identical to the oracle and
the action drift is unchanged from the start version (`actions` cos 0.9999972,
`physical_actions` cos 0.9999541; threshold cos 0.9943, rel_rms 0.34).

The hand-written kernels are checked against their torch expressions at the
Target's real shapes by `python -m lab.lingbot_kernel_check` (job 614334): the
projection epilogue, the attention epilogue, the RMSNorm, the AdaRMS-with-
residual and the gated activation are **bit-identical**; the masked softmax
differs by 2.8e-9 (a division against torch's warp reduction).
