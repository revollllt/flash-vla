---
id: technique-tma-3d-box-row-major
title: "One 3-D TMA box loads a row-major deep-K tile"
type: technique
architectures: [sm90]
tags: [tma, tma-box-geometry, swizzling, wgmma]
confidence: measured
reproducibility: snippet
prerequisites: [hw-tma]
related: [technique-swizzling, technique-pipeline-stages, hw-wgmma]
sources: [doc-hardware-unit-test, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# One 3-D TMA box loads a row-major deep-K tile

Row-major bf16 caps a 2-D TMA box row at 64 elements (128 B under SW128,
[tma.bytes.txn.max]), so a deep-K stage (BK = 128/256) appears to need
several boxes per stage, or an M-major transpose of the activation, which is
a whole extra pass. Use a 3-D box `{64, rows, k/64}` with the 64-element
chunk as the outer dimension. It lands in shared memory as
`[chunk][row][64]`, exactly the CuTe SW128 K-major image, so one TMA copies a
row-major (rows x K) tile: no transpose, no extra descriptors, one
transaction where a 2-D tiling needs K/64 of them.

The copy engine charges per transaction [tma.issue.warp], so collapsing four
boxes into one removes issue cost directly, and the K-major landing image is
already what the wgmma shared-memory descriptors want. The win scales with
the transaction count removed.

## Source-backed fragment

The 3-D form, from `sm90_common.cuh` in `doc-kernel-design-templates`:

```cpp
__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map, void* dst,
                                            int32_t c0, int32_t c1, int32_t c2,
                                            uint64_t* full) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
        "r"(smem_u32(full))
      : "memory");
}
```

## Caveats

Box extents are bytes, not elements: size the contiguous dimension in bytes
[tma.bytes.txn.dtype]; the descriptor cap [tma.bytes.txn.max] still bounds
the product of the three dims.
