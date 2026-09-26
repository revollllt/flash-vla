# Pi0.5 / RTX 5090: workload transfer, before any workload change

The shipped plan, with `bf16` and with the `mxfp8-llm-ffn` recipe, on the two
Pi0.5 workloads of [docs/workloads.md](../../../docs/workloads.md):

- `robodojo`: 3 cameras, chunk 50, a 200-slot prompt carrying 14 state values.
- `libero`: 2 cameras, chunk 10, a task-only prompt.

Revision: `workload/generalization` at W0; weights are the seeded official-layout
random ones (latency does not depend on weight values).

| Fixture | Physical prefix | Valid prefix | Backbone bucket |
|---|---:|---:|---|
| `robodojo`, seeds 0 and 42 | 968 | 838 (70 prompt tokens) | short (M896) |
| `libero`, seeds 0 and 42 | 712 | 524 (12 prompt tokens) | none: 712 rows run the full-row plan |

## Latency

Each cell is two fresh-process runs of 100 chunks, ordered robodojo, libero,
libero, robodojo (`measurements/latency-*.json`). Median chunk latency:

| Recipe | robodojo | libero | libero / robodojo |
|---|---:|---:|---:|
| bf16 | 30.11, 30.14 ms | 26.07, 26.14 ms | 0.87 |
| mxfp8-llm-ffn | 21.74, 21.74 ms | 19.13, 19.15 ms | 0.88 |

libero has 2/3 of the vision work, 74% of the physical prefix rows and a
fifth of the chunk, but runs only 12-13% faster.

## Where it does not transfer

Kernel time per call site from `tools.profiling.model` (`measurements/profile/`),
bf16. The last column is libero's time divided by robodojo's:

| Stage | Call site | robodojo us | libero us | ratio |
|---|---|---:|---:|---:|
| vision | total | 4048 | 3139 | 0.78 |
| vision | `vision_encoder_attention` | 405 | 404 | **1.00** |
| vision | `vision_encoder_out_proj_residual` | 415 | 347 | 0.84 |
| backbone | total | 16529 | 14018 | 0.85 |
| backbone | `llm_backbone_norm_gated_ffn_masked` | 9367 | 8017 | 0.86 |
| backbone | `llm_backbone_ffn_down_residual_masked` | 4303 | 3721 | 0.86 |
| backbone | `llm_backbone_norm_qkv_rope` | 1019 | 990 | **0.97** |
| backbone | `llm_backbone_attention` | 1121 | 668 | 0.60 |
| expert | total | 9395 | 8962 | 0.95 |
| expert | `action_expert_ffn_down_residual` | 1528 | 1632 | **1.07** |
| expert | `action_expert_action_out_proj` | 50 | 151 | **3.01** |
| expert | `action_expert_norm_qkv_rope` | 1791 | 1761 | 0.98 |

With mxfp8-llm-ffn, `llm_backbone_ffn_down_residual_masked` scales by 0.94
(1797 against 1688 us), and the rest matches bf16 within 1%.

## Routes named against paths run

The identity names the same routes on both workloads. What runs differs inside
two backends:

- `bucketed-backbone` plans its M896/M968 buckets only when the prefix has 968
  rows (`bucketed_backbone.py`: `rows == 968`). At 712 rows it silently runs
  `cutlass_backbone`'s full-row plan, and the identity still says
  `bucketed-backbone`.
- `mxfp8-backbone` buckets only a 968-row prefix (`BUCKETED_PREFIX`). At 712
  rows it runs one plan with split-K 3 on the down GEMM (`DOWN_CONFIG_SPLIT`,
  96 tiles on 170 SMs).

## Candidates this matrix points at

For W0 these are observations only; floor shares come with the F phase.

1. Vision attention does not shrink with a third fewer views.
2. Backbone `norm_qkv_rope` does not shrink with 26% fewer rows.
3. The backbone's 712-row prefix runs unbucketed full-row GEMMs, and 25% of
   its rows are padding (valid 518-534).
4. At chunk 10 the expert's `action_out_proj` is three times slower and its
   `ffn_down_residual` 7% slower than at chunk 50. The expert kernels were
   tuned only at M=50.
