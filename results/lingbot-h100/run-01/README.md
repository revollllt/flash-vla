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
