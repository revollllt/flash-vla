# Agent Note: Vision GEMM epilogues -- smem staging and cuBLASLt fused epilogues

Status: implemented

Date: 2026-09-02

## Problem

The SigLIP vision stage (27 layers, M=768) spent 75 of its 90 us per layer
in four TileLang GEMMs running at 25-35% of datasheet MFU, against 60-67%
for the encoder GEMMs of the same kernel family. The stage sat 3.8x above
its compute floor and had not been touched since the Pi0 port.

The obvious hypothesis -- wave quantization (two sites launch 108 CTAs on
132 SMs) and stale tile configs -- was falsified by measurement: a
132-config sweep per site over tile shape, pipeline depth, thread count and
warp specialization could not beat the shipped configs, and every config
plateaued near 300 TFLOP/s while cuBLAS reached 450-500 in the same timer.

## Decision

The binding cost is the epilogue, not the mainloop. Every vision GEMM has
K=1152 or a 1152-wide output, so per-element traffic in the wgmma fragment
layout (4-byte strided stores; 2-byte residual loads) is a large fraction
of each kernel.

Two changes, one per epilogue kind:

- `_matmul_bias` and `tl_matmul_bias_res` in
  `backends/tilelang/kernels/base.py` stage their result through a bf16
  shared-memory tile and load the residual tile with one shared-memory copy.
  Same names, signatures and tile configs; the residual body's in-place
  aliasing stays safe because the whole R tile is read before C is written.
  This also changes the encoder projector and the decoder action
  in-projection, which share `tl_matmul_bias`.
- The two sites whose epilogue cuBLASLt fuses natively run cuBLASLt from the
  wrapper: vision QKV via `torch.addmm` (bias) and the FFN up-projection via
  `torch._addmm_activation(use_gelu=True)`, whose GELU is the tanh
  approximation upstream uses. Both are allocation-free with `out=` and
  graph-capturable. The two residual sites (o_proj, FFN down) stay on
  TileLang; cuBLAS has no bias+residual epilogue and its residual form
  (`addmm` with a matrix input) measured slower than the staged TileLang body.

Tile configs `_VIS_OUT_PROJ` / `_VIS_FFN_DOWN` are unchanged; `_VIS_QKV` and
`_VIS_FFN_UP` are retired with their call sites.

## Alternatives considered

- Tile / stage / warp-specialization re-tune: rejected on evidence -- four
  sites, 132 feasible configs each, the shipped config is within noise of
  the best in every sweep (jobs 585341-585343, 585316).
- Split-K for the two 108-CTA sites: not pursued; the sweep showed the
  per-CTA body, not SM occupancy, was binding (64x64 tiles with 216 CTAs
  ranked below 108-CTA configs).
- cuBLAS for all four sites with a separate residual pass: rejected -- the
  staged TileLang residual body beats cuBLAS's matrix-input `addmm` at both
  residual sites, and a second pass would add a launch per site.

## Consequences

- Vision runs two cuBLASLt kernels per layer inside the captured graph; the
  per-kernel breakdown (`benchmarks/profile_pi05.py`) now shows cuBLAS kernel
  names (`nvjet`/`cublasLt`) for those sites.
- Numerics: the TileLang bodies keep fp32 accumulation and the same bf16
  rounding point; cuBLASLt accumulates in fp32 with its own reduction order.
  Whole-prefix parity against OpenPI is unchanged layer by layer (see below).
- `tl_matmul_bias_nows` and `tl_matmul_bias_gelu` lost their only call sites
  (vision QKV and FFN-up) to cuBLASLt and were removed; the sweep results
  for both bodies stay in the task workspace ledger.

## Verification

- Sweeps (cold weights, `graph_time_cold`, MAIN
  `artifacts/ktasks/vision-gemm-retune/`): tile sweeps 585341/585342/585343/
  585316; staged-epilogue sweeps 585400/585401/585402/585399; per-site
  outcomes in `candidates.jsonl`.
- Prefix parity vs OpenPI (`eval/correctness/pi05/prefix_parity.py`, openpi
  env): baseline job 585323 (HEAD 7505c35) vs job 585475 (this change):
  `passed` both; per-layer K/V cosine equal within 1e-5 at every layer
  (layer 0: 0.999937 vs 0.999938).
- E2E (`sbatch/profile_pi05.sh`, plan `attn-ffn-cuda-fused-producer-pdl`,
  `CAPTURE_TRACES=1`): see the table below.

| measurement | before (585139/585140, ACD1-33) | after (585509, ACD1-28) |
|---|---:|---:|
| vision stage, e2e min | 2.460 ms | 2.021 ms |
| prefix stage, e2e min | 7.400 ms | 7.416 ms |
| qkv, trace us/layer | 19.65 | 13.95 |
| o_proj, trace us/layer | 8.29 | 7.30 |
| fc1, trace us/layer | 25.41 | 16.48 |
| fc2, trace us/layer | 22.11 | 24.80 |

Open observation: in the captured graph the staged fc2 body runs 20.5 us at
layer 0 and 24-26 us at every later layer, while in isolation it beats the
old body in every measured context (separate or aliased residual, hot A,
directly behind the cuBLASLt fc1: 20.0-20.5 vs 21.6-22.2 us, probe job
585532). The cause is not established; the leading hypothesis is a lower
sustained clock under the denser stage (cuBLASLt kernels at 450-500
TFLOP/s), which the longest wgmma kernel shows first. It is bounded at
~0.07 ms and does not change the decision; a same-node A/B on a 610-driver
node is the next check.
