---
id: technique-scale-on-register-fragment
title: "Apply per-K factors to the register fragment, not in shared memory"
type: technique
architectures: [sm90]
tags: [wgmma, ldmatrix, register-fragments, per-k-specialization, fine-grained-quantization]
confidence: measured
reproducibility: snippet
prerequisites: [hw-wgmma]
related: [technique-wgmma-rs-fragment-parity, pattern-serialized-wgmma, technique-fine-grained-quantization, kernel-deepgemm]
sources: [doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Apply per-K factors to the register fragment, not in shared memory

An elementwise per-K factor (a norm scale, a dequant scale) must be applied
to the A operand before the GEMM. Applying it as a read-modify-write over the
landed shared-memory frame needs a proxy fence plus a warpgroup barrier every
stage, and the SS wgmma that follows still re-reads the frame it just
rewrote. Instead, `ldmatrix` the A operand into registers, apply the factor
to the register fragment, and run RS wgmma. The frame is never rewritten in
shared memory: the per-stage fence and barrier disappear with the RMW, and the
wgmma's shared-memory operand reads halve because A now comes from registers.
The in-smem variant is not a small tax; the RMW plus its synchronization can
rival the GEMM itself on a short stage.

## Source-backed fragment

The register path, from `sm90_common.cuh` in `doc-kernel-design-templates`:

```cpp
// One ldmatrix moves four 8x8 tiles into the register layout mma.sync expects,
// transposing lanes for free.  The address is per-lane: lane l supplies the
// row it wants, which is why callers compute a swizzled per-lane offset.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&out)[4], const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
               : "r"(smem_u32(smem)));
}
```

## Caveats

RS operands are exactly what expose ptxas C7518; pair this with the
two-fragment stage-parity pattern in `technique-wgmma-rs-fragment-parity`.
And do not "optimize" side transactions that are already overlapped: making
a small per-stage factor slice resident in shared memory to save its copies
buys nothing when those copies were never on the critical path. Check the
timeline before retiring traffic.
