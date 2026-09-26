# Pi0.5 / RTX 5090: workload generalization

`robodojo` is the workload the Target is optimized for; `libero` is the
workload this round generalizes to ([docs/workloads.md](../../../docs/workloads.md)).
The diagnosis that chose the work is G0 in
[../workload-transfer/README.md](../workload-transfer/README.md). Weights are
the seeded official-layout random ones: latency does not depend on their
values, and correctness is judged against the model's reference.

## Result

A/B in fresh processes, ordered A B B A, medians of 100 chunks each, the
change's A the base named in `iterations.csv`. From R (main `c8026ea`) to the
end of G (`abba/g-total-*`):

| Recipe | robodojo | libero |
|---|---:|---:|
| bf16 | 30.15 -> 30.18 ms (+0.03, A runs spread 0.20) | 26.18 -> 24.41 ms (**-1.77, -6.8%**) |
| mxfp8-llm-ffn | 21.83 -> 21.83 ms (+0.00) | 19.14 -> 18.05 ms (**-1.10, -5.7%**) |

robodojo runs the same kernels as before, at the same configurations.

The transfer matrices at the end of R and of G (`measurements/matrix-{R,G}-*.json`,
`python -m benchmarks.latency --workloads robodojo,libero`) put libero's share
of its physical-layout floor at:

| Recipe | R | G |
|---|---:|---:|
| bf16 | 0.597 (backbone 0.75) | 0.638 (backbone 0.85) |
| mxfp8-llm-ffn | 0.441 (backbone 0.49) | 0.468 (backbone 0.57) |

The two matrices were measured hours apart and robodojo's controls drifted by
0.2 ms between them, so the gains above come from the A/B runs, not from the
difference of the matrices.

## What was kept

`iterations.csv` has one row per change, workload and recipe: the stage the
change touches and the total, each the mean of the two A and the two B
medians, and the spread of the two A runs.

| # | Step | Change | libero, stage it touches |
|---|---|---|---:|
| 1 | G1 | Vision attention output projection at M=512 on family config 8 instead of 10 | vision -0.05 ms |
| 2 | G2 | Backbone row buckets derived from the shape: the short bucket ends at the first 128-row tile boundary past the image tokens (M640/M712 for two views, M896/M968 for three as before), in the BF16 and MXFP8 kernels; MXFP8 down GEMM at M640 on stream-K (config 3) | backbone -1.60 ms bf16, -0.83 ms mxfp8 |
| 3 | G1 | Backbone QKV GEMM at M=712 on CUTLASS config 0 instead of cuBLAS | backbone -0.12 ms both |
| 4 | G1 | Expert `action_out_proj` GEMM at M=10 on family config 9 instead of cuBLAS | expert -0.09 ms |

Each G1 change is a table keyed by the GEMM's (M, K, N) in its backend,
consulted at plan time; the geometries robodojo runs are not in any table.

## Evidence for each choice

- `g1/geometry-screen.json` (`python -m lab.pi05.geometry_screen`): every GEMM
  call site G0 named, at both workloads' geometry, over cuBLAS and the 12 tiles
  of the stream-K family, each timed in one graph over the distinct weights the
  forward reads (one per layer, or the action output's one weight once per
  step); and the vision attention over SDPA's backends. A
  winner is kept only above 5%:
  - vision out_proj, M=512: config 8 10.06 us against 12.76 (config 10);
  - backbone QKV, M=712: CUTLASS 40.3-41.1 us against cuBLAS 49.9; at M=968
    they tie (50.5), so robodojo keeps cuBLAS;
  - `action_out_proj`, M=10: config 9 5.0 us against cuBLAS 14.0, which runs
    one CTA along K; at M=50 cuBLAS wins (4.0 us) and stays.
- `g2/mxfp8-gemm-screen-libero.json` (`lab/pi05/mxfp8_gemm_screen.py --workload
  libero`): at M640 the down GEMM takes 78 us on stream-K against 95 us on the
  three-way split robodojo's M896 uses; at M712 the split stays best (100 us).
- Correctness: `tests/test_pi05_bucketed_backbone.py` and
  `tests/test_pi05_mxfp8_backbone.py` replay both buckets of both prefixes from
  one graph against full-row CUTLASS and the fake-quant reference;
  `eval.model_reference` passes both workloads at full depth, and the GPU
  reference gate passes every workload, plan and recipe.

## What was not changed, and why

- Vision attention (libero at 0.18 of its floor, robodojo at 0.26): SDPA's
  flash kernel runs split-KV and takes 16.2 us per layer at two views and 16.3
  at three. No other backend is faster (cuDNN 17.9, memory-efficient 20.5,
  math 63), nor a plain batched matmul-softmax-matmul (32). The kernel is
  latency-bound at any view count, so it belongs to the robodojo main line.
- Expert `ffn_down` and `out_proj` at M=10: their deployed tile is also the
  family's best plain GEMM. cuBLAS is 6-15% faster as a bare GEMM but would
  need a separate gated-residual pass. The 7% G0 measured in the engine is 1%
  in isolation.
- Backbone attention (0.23 against robodojo's 0.35): a compiled
  matmul-softmax-matmul over every prefix row, padding included. Skipping the
  padding needs a masked attention kernel, which would also serve robodojo:
  main line.
- The expert's attention and QKV preparation: the same time at chunk 10 and
  50, launch-bound (G0). Main line.

Robodojo main-line notes from the screen, not acted on here: at M=768 the vision
QKV GEMM runs 4-6% faster on config 5 than on the deployed config 10 (two runs
of the screen), and at M=50 cuBLAS beats config 9 by 9% on the expert
`out_proj` GEMM alone.
