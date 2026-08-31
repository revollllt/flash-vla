---
id: tma-3d-box-row-major
type: technique
arch: sm90
tags: [tma, box-geometry, sw128, bk256]
confidence: measured
---

# One 3-D TMA box loads a row-major deep-K tile

## Context

Row-major bf16 caps a 2-D TMA box row at 64 elements (128 B under SW128,
`[tma.bytes.txn.max]`), so a deep-K stage (BK = 128/256) appears to need
several boxes per stage — or an M-major transpose of the activation, which
is a whole extra pass.

## Move

Use a 3-D box `{64, rows, k/64}` with the 64-element chunk as the outer
dimension. It lands in shared memory as `[chunk][row][64]` — exactly the
CuTe SW128 K-major image — so **one TMA copies a row-major (rows x K)
tile**: no transpose, no extra descriptors, one transaction where a 2-D
tiling needs K/64 of them.

## Why it works

The copy engine charges per transaction (`[tma.issue.warp]`), so collapsing
four boxes into one removes issue cost directly, and the K-major landing
image is already what the wgmma smem descriptors want. The win scales with
the transaction count removed.

## Caveats

Box extents are bytes, not elements — size the contiguous dimension in
bytes `[tma.bytes.txn.dtype]`; the 32 KB descriptor cap
`[tma.bytes.txn.max]` still bounds the product of the three dims.
