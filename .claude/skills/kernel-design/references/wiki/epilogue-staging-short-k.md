---
id: epilogue-staging-short-k
type: pattern
arch: sm90
tags: [gemm, epilogue, tilelang, short-k, cublas]
confidence: measured
---

# A short-K GEMM that plateaus under a tile sweep is epilogue-bound

## Context

A GEMM with K around 1K and a full-width output (M in the hundreds, N in
the thousands) sits at ~30% of the tensor-core ceiling, a tile x stages x
threads x warp-specialization sweep ranks the shipped config at or near the
top, and a library GEMM (cuBLAS) at the same shape is 30-40% faster in the
same timer. Wave quantization looks like the cause and is not: the ranking
is flat across CTA counts from half a wave to several.

## Move

Look at the epilogue before the mainloop. Two costs dominate when K is
short, because output and residual bytes per FLOP are 10x what they are at
the long-K sites the same body was tuned on:

- storing the accumulator fragment straight to global leaves each thread
  writing 4-byte pieces in the wgmma layout -- stage the tile through
  shared memory and let one wide copy (TMA under warp specialization) do
  the store;
- a residual added per element from global reads bf16 scalars in the same
  scattered pattern -- bring the residual tile in with one shared-memory
  copy, then add from smem.

Expect 8-20% at unchanged tile configs. Then check whether the library
already fuses the epilogue you need: cuBLASLt carries bias and tanh-GELU
epilogues (`torch.addmm`, `torch._addmm_activation`), and at these shapes it
beat the staged TileLang body by a further ~15%. It has no bias+residual
epilogue, so residual sites stay hand-written.

## Why it works

The mainloop is fine -- the same body reaches 60-70% of the ceiling when K
is 16K. What changes with K is the ratio of epilogue traffic to math, and
the fragment-layout store is half-efficient per sector while the scalar
residual load is worse. Both are hidden at long K and binding at short K.

## Caveats

The staged store costs a BLOCK_M x BLOCK_N bf16 tile of shared memory;
re-run the smem arithmetic if the ring is near the limit. In-place residual
(R aliases C) stays safe only if the whole R tile is read into smem before
any C element is written. A library epilogue changes the reduction order;
gate it with whole-stage parity, not just a per-kernel cosine.
