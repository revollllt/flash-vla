---
id: scale-on-register-fragment
type: technique
arch: sm90
tags: [wgmma, rs-operands, ldmatrix, elementwise-fusion]
confidence: measured
---

# Apply per-K factors to the register fragment, not in smem

## Context

An elementwise per-K factor (a norm scale, a dequant scale) must be applied
to the A operand before the gemm. Applying it as a read-modify-write over
the landed shared-memory frame needs a proxy fence plus a warpgroup barrier
every stage, and the SS wgmma that follows still re-reads the frame it just
rewrote.

## Move

`ldmatrix` the A operand into registers, apply the factor to the register
fragment, and run **RS wgmma**. The frame is never rewritten in shared
memory.

## Why it works

The per-stage fence and barrier disappear with the RMW, and the wgmma's
shared-memory operand reads halve because A now comes from registers. The
in-smem variant is not a small tax — the RMW plus its synchronization can
rival the gemm itself on a short stage.

## Caveats

RS operands are exactly what expose C7518 — pair this with the
two-fragment stage-parity pattern in
[c7518-wgmma-serialization](c7518-wgmma-serialization.md). And do not
"optimize" side transactions that are already overlapped: making a small
per-stage factor slice resident in smem to save its copies buys nothing
when those copies were never on the critical path — check the timeline
before retiring traffic.
