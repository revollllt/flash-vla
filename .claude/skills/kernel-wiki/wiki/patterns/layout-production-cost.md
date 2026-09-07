---
id: pattern-layout-production-cost
title: "A layout's production cost must stay under the kernel-side gain"
type: pattern
architectures: [sm90]
tags: [layout-production, kernel-fusion, proxy-fence, swizzling]
symptoms: [layout-production-cost, fusion-regression]
candidate_techniques: [technique-producer-fusion-pdl, technique-tma-3d-box-row-major]
related: [pattern-fusion-latency-chain, technique-bulk-store-publish, technique-swizzling]
sources: [doc-ptx-isa-sm90, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# A layout's production cost must stay under the kernel-side gain

## Symptom

A pre-arranged input layout (K-major, pre-scaled) makes a kernel measurably
faster, and the temptation is to add a producer for it: a standalone kernel,
an in-kernel prep phase, a readiness pipeline.

## Likely Causes

The gain is bounded by the layout delta inside one kernel; the production is
real data movement plus synchronization on the critical path. Hiding it in a
different launch or phase changes who pays, not the price: a standalone
launch pays its ramp and bandwidth; an in-kernel prep phase pays the same
data movement plus a grid-wide sync; a readiness pipeline pays the counters.
At decode shapes these production costs routinely exceed a modest
kernel-side gain.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Producer fusion](../techniques/producer-fusion-pdl.md) | The only shape that fits the budget: a producer that replaces existing work, fusing the layout production into an op that already touches the data |
| [3-D TMA box](../techniques/tma-3d-box-row-major.md) | Often removes the need for the layout at all: a row-major deep-K tile lands K-major in one transaction |

## Diagnosis Checklist

```
1. Establish the upper bound: measure the kernel with the layout produced
   outside the timed path.
2. Budget every producer honestly: launch ramp + bandwidth, or data
   movement + grid sync, or counters.
3. Reject any new pass; accept only a producer folded into an existing op.
```

## Caveats

Two correctness traps from this class of work:

- **Cross-proxy ordering.** Release/acquire on a readiness counter orders
  generic global memory but is not a fence for a following TMA read of the
  same data; place `fence.proxy.async.global` between generic stores and any
  TMA consumption of them. The failure is silent and data-dependent.
- **CuTe swizzle offsets.** A composed `Layout_MN_SW128_Atom` carrying
  `smem_ptr_flag` must not be called directly and treated as a bf16 physical
  offset for generic shared-memory access. The MN-major SW128 mapping is
  `physical = k * M_PAD + (((m >> 3) ^ (k & 7)) << 3) + (m & 7)`; getting
  this wrong corrupts a few rows, not the whole tile, and looks like a math
  bug rather than a layout bug.
