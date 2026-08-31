---
id: wgmma-tile-n-floor
type: pattern
arch: sm90
tags: [wgmma, tile-shape, smem-bandwidth, attention]
confidence: measured
---

# A wgmma below N=64 is shared-memory-bound

## Context

A wgmma-fed stage runs far over its instruction-count estimate and the
ablation points at one gemm whose output tile N is 32 — `S = Q K^T` in an
attention kernel is the classic case, because the key-tile width sets N
there.

## Move

Keep **>= 64 columns per wgmma**. At N=32 the `m64n32k16` instruction
re-reads the full A-tile per half-sized B-tile and runs at ~3x
`[wgmma.issue.wg.ss]`, bound on shared-memory bandwidth against the landing
TMA frames rather than on the tensor core. If the *output* tile is
genuinely below N=32, change instruction rather than tile: `mma.sync`
crosses over there (`[mma.xover.n.wgmma]`).

## Why it works

Halving N halves the tensor-core work per instruction but not the A-operand
traffic, so per-useful-flop shared-memory reads double — and the mainloop
becomes a smem bandwidth contest between the math warps and the producer's
landing frames.

## Caveats

Widening N grows the per-stage frame and ring footprint; re-run the
shared-memory and ring-depth arithmetic when N moves.
