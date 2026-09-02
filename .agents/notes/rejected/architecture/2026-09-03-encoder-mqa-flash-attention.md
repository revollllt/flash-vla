# Agent Note: Fused encoder MQA attention in TileLang

Status: rejected — the single-pass TileLang kernel saves 5.2 us/layer (0.089 ms over 17 layers), under the 10 us/layer / 0.10 ms promotion bar; the design that would clear it is a two-warpgroup CUDA kernel, not another TileLang variant

Date: 2026-09-03

## Problem

The Pi0.5 encoder attention runs as a torch.compile chain of three kernels
(cuBLAS QK^T, a Triton softmax, cuBLAS PV) that materializes a 7744 x 968
score matrix: 30.1-30.9 us per layer, 0.51 ms over 17 layers. The op is
multi-query with one K/V head, so it is single-head attention over 7744
flattened query rows (token * 8 + head) against 968 keys at head_dim 256
with an additive per-key mask. Library SDPA backends were rejected earlier
(cudnn 60 us, efficient 82 us, flash refuses the float mask).

## Proposal (measured, all on the same cold-rotating graph timer as the chain)

1. One flash-style pass -- the decoder's masked split kernel with a single
   split and the normalization in the epilogue, one CTA per 64 query rows
   (121 CTAs). Best config 64 x 128-key tiles, 1 stage, 128 threads, warp
   specialization on: **25.7 us (300 TFLOP/s) vs 30.9 us**. 128-row tiles
   with two warpgroups (61 CTAs): 49 us. Warp specialization off: 39-51 us.
   Key tiles of 192/256 do not fit shared memory.
2. Key-split + combine (the production decoder pair at this shape), 2-8
   splits: 34.8-68 us, monotonically worse with more splits.
3. Mask tile through one fragment copy per key block: identical to (1).

Parity of every measured candidate vs fp32 torch over valid rows: cosine
0.999997, max relative error 3.7e-3 (bf16 output rounding).

## Alternatives considered

- Promote (1) anyway: 17% faster per launch and two fewer launches per
  layer, but 0.089 ms e2e is below the goal's promotion bar and inside the
  6% stage noise floor, so it could not be confirmed end to end. Kept in
  the task workspace as a ready candidate that becomes promotable in
  composition with another prefix change.
- More TileLang tiling: exhausted -- the 256-thread 64-row forms fail to
  lower (fragment layout conflict between the score tile and its bf16 cast),
  and wider key tiles exceed shared memory.

## Why it stalls (mechanism)

Per 128-key block a CTA issues two wgmma groups (~1.3 us of tensor-core
time at the per-SM ceiling) and then runs the online softmax, the
accumulator rescale and the bf16 cast on the same warpgroup, so the tensor
core idles for roughly a third of each block. The warp-specialized kernel's
register and shared-memory footprint holds one CTA per SM, so a second CTA
cannot fill the gap -- which is exactly what the key-split experiment shows:
doubling the CTA count added Q re-reads and partial round trips and
overlapped nothing.

## What would clear the bar

A CUDA kernel on `hardware/nvidia/cuda/tile/sm90` with one TMA producer
warp and two math warpgroups on the same 64-row tile, each owning half of
the keys (intra-CTA combine through shared memory), so one warpgroup's
softmax overlaps the other's wgmma (the FlashAttention-3 ping-pong
structure). Expected 14-17 us per layer from the per-block arithmetic
above (-0.23 to -0.28 ms over 17 layers). The decoder attention task loop
(`backends/cuda/kernels/attn_taskloop.cu`, kAttention path) already has the
TMA ring, the online-softmax register code and the fast exp2 for a 64-row
tile; the encoder version drops the task table and adds the second
warpgroup. Budget it as a full kernel-design task (contract, parity
harness, 4-6 candidates), not as a variant of this one.

## Verification

Jobs 588808 (candidate sweep, 12 configs), 588814 (key-split), 588832
(variants), node mix ACD1-2/-22; harness and kernel source in the task
workspace `artifacts/ktasks/prefix-mqa-attn/` (bench_attn*.py,
attn_kernel.py, candidates.jsonl, runs/*.json). No production file changed.
