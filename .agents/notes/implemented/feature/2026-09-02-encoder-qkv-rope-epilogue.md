# Encoder QKV projection with RoPE and Q/K/V scatter in the GEMM epilogue

Status: implemented (2026-09-02). Owner of the Pi0.5 encoder QKV call site
(`encoder_norm_qkv_rope`, `tl_matmul_rope_scatter`).

## Problem

The encoder's QKV call site ran three kernels per layer: RMSNorm, a
warp-specialized GEMM into a packed (seq, 2560) bf16 buffer, and a
rope-and-scatter pass that re-read that buffer to rotate the Q/K pairs and
split it into the Q buffer and the layer's K/V cache rows. The scatter pass
moved ~10 MB per layer at ~11% of HBM bandwidth and cost as much as the GEMM
(27 us vs 29 us per layer in trace 585140; 0.49 ms over 18 layers -- the
largest non-GEMM item in the prefix stage).

## Decision

One TileLang kernel, `tl_matmul_rope_scatter`, projects and scatters. The
fp32 accumulator is staged through a bf16 shared-memory tile -- the same
rounding the packed buffer applied -- the rotation runs in fp32 on those
rounded values inside the epilogue, and each N tile is written straight into
Q, K or V because BLOCK_N divides HEAD_DIM. The encoder keeps its
normalize-before-GEMM order; only the packed intermediate (and its scratch)
disappears. The output is bit-identical to the retired pair, so no downstream
parity threshold moves.

Tile config `_ENC_QKV_ROPE` = 128x64x64, 3 stages, 128 threads, warp
specialization OFF. At the retired GEMM's config (128x128, WS on) the fused
kernel is 42 us; the no-WS 128x64 config is what makes the epilogue free and
is load-bearing.

## Alternatives considered

- Fold RMSNorm as a per-row factor, as the decoder's `tl_qkv_gemm_rope`
  does: rejected -- the encoder rounds the normalized activations to bf16
  before the projection, and the two orders are not numerically
  interchangeable against upstream.
- Speed up the standalone rope-scatter pass (wider loads): rejected without
  measurement -- it stays a second pass over 10 MB behind a launch, against
  an epilogue measured to be free.
- Library attention/projection kernels do not apply: the scatter target is
  the layer-indexed KV cache with interleaved-pair RoPE.

## Consequences

- Prefix launches per layer drop from 10 to 9 (177 -> 159 per inference);
  `benchmarks/layer_breakdown.py` `PREFIX_LAYER` carries the new sequence
  (`qkv:gemm_rope`).
- `tl_rope_scatter_bf16` and the `encoder_qkv` scratch are removed from the
  Pi0.5 backend (the Pi0 backend keeps its own copy).
- A re-tune of `_ENC_QKV_ROPE` must re-check the no-WS choice and BLOCK_N |
  HEAD_DIM; the tuning harness lives in the task workspace, not the repo.

## Verification

- Kernel gate (workspace harness `artifacts/ktasks/encoder-qkv-rope/bench_parity.py`,
  job 585370, ACD1-8): fused output bit-exact vs the production pair for Q,
  K and V at both the retired and the promoted config; 86 configs measured,
  0 parity failures. Same-timer cold-weight latency: production pair
  53.6 us/layer (GEMM 30.1 + scatter 23.6) -> fused 29.2 us/layer.
- Route gate: `eval/correctness/pi05/prefix_parity.py` (openpi env), job
  585323 (HEAD 7505c35, before) vs job 585470 (after): PASSED, and the
  per-layer K/V metrics are identical to the baseline at every layer
  (maximum cosine delta 0.0), which is what a bit-exact kernel predicts.
  Both result files are retained in the task workspace.
- Decoder gate: `plan_parity --plan attn-ffn-cuda-fused-producer-pdl
  --steps 1 --layers 18`, job 585511: PASSED.
- E2E: `sbatch/profile_pi05.sh` (plan attn-ffn-cuda-fused-producer-pdl),
  job 585510: prefix stage min 7.400 -> 6.952 ms, forward wall min
  17.866 -> 17.351 ms; the untouched stages moved 2.460 -> 2.480 (vision)
  and 7.429 -> 7.434 (decoder), which bounds node and run drift for this
  comparison. Clocks are unpinned on this partition; deltas under 6% are
  noise, and this one is 6.1%, so the kernel gate above -- same-timer,
  same-job, 53.6 -> 29.2 us/layer over 18 layers = 0.44 ms predicted
  against 0.448 measured -- is the load-bearing evidence rather than the
  stage delta alone.

The first attempt at these three gates (585471, 585472, 585507) failed on a
worktree environment fault, not on the change; 585510 and 585511 are the
runs that count. Their job logs have since been lost with the worktree that
produced them (see the note's Consequences), so the numbers above are
carried by this note and the commit message rather than by an artifact.
