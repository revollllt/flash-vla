# LingBot-VLA-4B · H100 · run-02

Continues [run-01](../run-01/README.md) from its final version, with an
explicit target: reach Pi0.5's 15.864 ms, which a pure roofline comparison says
LingBot should be capable of.

![Optimization progress](progress.svg)

Workload, checkpoint, fixture, environment and protocol are run-01's; only the
starting version differs. Every number here is driver **570.86.10**, which runs
this workload ~4% slower than the 610 the run-01 curve used — run-01's final
version measures 25.501 ms on 610 and 26.601 ms on 570, and this run uses the
570 figure as its start so the two points either side of each change share a
driver.

## Result so far

| # | change | median | delta |
|---|---|---:|---:|
| 0 | run-01 final (`split-attention`) | 26.601 | |
| 1 | paired gated activation, shuffle-reduced norms | 25.369 | **−1.232** |
| 2 | one-launch vision q/k/v rope | 24.396 | **−0.973** |

Iteration 2 is paired in one job (615599). Iteration 1 is not plan-selectable —
it changes kernels the deployed route already uses — so it is measured as the
same route before and after on the same driver, and attributed per kernel by
profile:

| kernel | segment | before | after |
|---|---|---:|---:|
| gated activation | backbone | 14.30 µs | 8.35 |
| gated activation | vision | 11.37 | 7.35 |
| gated activation | expert | 2.12 | 1.81 |
| `ada_rms_add` | expert | 3.51 | 2.66 |
| `rms_norm_add` | backbone | 7.19 | 3.21 |
| `rms_norm` | vision | 4.73 | 3.52 |
| masked softmax | backbone | 8.82 | 8.41 |

In-graph total across the run: 26.847 → 23.579 ms (jobs 615421, 615538, 615555,
615567).

## Where the remaining time is

Profiled in-graph after iteration 2 (job 615567; profiled totals run a few
percent above the uninstrumented benchmark):

| segment | in-graph | launches |
|---|---:|---:|
| `vision_encoder` | 3.068 ms | 369 |
| `llm_backbone` | 4.784 ms | 443 |
| `action_expert` | **15.727 ms** | 3814 |

The expert, per layer-step (360 of them):

| stage | each | total | streaming floor |
|---|---:|---:|---:|
| `o_proj` + `down_proj` (skinny GEMM) | 2 × 5.69 µs | 4.10 ms | 2.99 / 3.38 µs |
| split-key attention (slices + combine) | 9.34 + 3.24 | 4.53 ms | 2.40 |
| `gate_up` (cuBLAS) | 5.86 | 2.11 ms | 4.90 |
| `qkv` (cuBLAS) | 5.01 | 1.80 ms | 3.27 |
| `ada_rms_add` ×2 | 2 × 2.66 | 1.92 ms | ~2.0 |
| RoPE + gated activation | 2.02 + 1.81 | 1.38 ms | at floor |
