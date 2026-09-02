# Agent Note: Prefix GEMM epilogues -- cuBLAS for the plain residual sites, staged store for the gated FFN

Status: implemented

Date: 2026-09-03

## Problem

Three encoder GEMM bodies still wrote their result straight from the wgmma
accumulator fragment and read the residual per element: the attention
output projection (M=968, N=2048, K=2048; 21.7 us/layer, 374 TFLOP/s), the
FFN down projection (K=16384; 98.6 us, 659 TFLOP/s) and the gated FFN
(N=16384; 219 us, 594 TFLOP/s, a 32 MB output per call). The vision stage
had just shown that this epilogue traffic, not the tiling, is what holds
short-K TileLang GEMMs well under the tensor-core ceiling
(`2026-09-02-vision-gemm-epilogue`).

## Decision

- The two plain residual sites run cuBLAS `addmm_` in place. Its epilogue
  is exactly the contract (`out += x @ W`, fp32 accumulate, bf16 out), and
  in the same cold-weight timer it beat every TileLang body measured,
  including the smem-staged one: 13.9 vs 21.4 us at o_proj, 86.4 vs 97.3 us
  at ffn:down. The in-place form is itself load-bearing -- with a separate
  output tensor cuBLAS measures 17.4 / 89.9 us.
- The gated FFN keeps its TileLang body with the result staged through a
  bf16 shared tile before the global store (production tile config
  unchanged): 219 -> 208 us/layer. Two cuBLAS GEMMs plus a gelu*up pass
  measure 234 us, so the fused dual-GEMM stays hand-written.
- The plain residual GEMM body (`_matmul_res`, both warp-specialization
  variants) is removed from `kernels/base.py`; the bias+residual bodies stay
  because cuBLAS has no bias+residual epilogue.

## Alternatives considered

- Keep TileLang at the residual sites with the staged epilogue: measured
  slower than cuBLAS at both (best 16.2 us at o_proj after a 20-config
  re-sweep; at ffn:down the staged body is slower than the original at
  every config -- the K=16384 mainloop hides the epilogue and the extra
  32 KB tile costs occupancy).
- A tile re-sweep of the staged gate body: 128x128x64 with 3 stages ties
  the production 128x128x128 with 2 stages (209 vs 208 us); no change.

## Consequences

- Encoder per-layer launch sequence is unchanged in count; the two residual
  sites now appear as cuBLAS `nvjet` kernels in traces, which
  `benchmarks/layer_breakdown.py` expects positionally.
- cuBLAS chooses its own reduction order; whole-prefix parity against OpenPI
  moves only at bf16-rounding level (see Verification). The goal's numerics
  invariant (bf16 with reduction-order tolerance) is met.
- A fixed-shape sm90 GEMM from the tile library may reclaim either cuBLAS
  site when it beats these numbers in the same timer; the sweep harness in
  the task workspace already carries that comparison.

## Verification

- Sweep, cold weights, `graph_time_cold`, ACD1-8, job 588745 (array tasks
  588746 o_proj / 588747 ffn_down / 588745 gate_up), 0 parity failures over
  50 measured configs; workspace `artifacts/ktasks/prefix-gemm-epilogue/`.
- Whole-prefix parity (openpi env) vs baseline 585323: job 588758 passed;
  layer-0 K/V cosine unchanged (0.999938 / 0.999937), every deeper layer
  within 1e-4 of the baseline cosine (reduction order only).
- Decoder plan parity `attn-ffn-cuda-fused-producer-pdl --steps 1 --layers 18`:
  job 588759 passed.
- E2E `profile_pi05.sh`, pdl plan, ACD1-33, job 588760: prefix stage min
  6.919 ms (585554) -> 6.306 ms; forward wall min 16.838 -> 16.254 ms;
  vision and decoder unchanged (2.012 / 7.442). Per layer in the graph
  (`layer_breakdown.py`, 159 launches asserted): gate_up 219 -> 204 us,
  ffn:down 98.6 -> 84.2 us, o_proj 21.7 -> 15.1 us.
