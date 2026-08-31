---
id: layout-production-budget
type: case
arch: sm90
tags: [layout, upper-bound, budget, proxy-fence, swizzle]
confidence: measured
---

# A layout's production cost must stay under the kernel-side gain

## Context

A pre-arranged input layout (K-major, pre-scaled) makes a kernel measurably
faster, and the temptation is to add a producer for it — a standalone
kernel, an in-kernel prep phase, a readiness pipeline.

## Move

First establish the **upper bound**: measure the kernel with the layout
produced *outside* the timed path. Then budget honestly: every way of
producing the layout costs what it costs wherever you hide it — a
standalone launch pays its ramp and bandwidth; an in-kernel prep phase
pays the same data movement plus a grid-wide sync; a readiness pipeline
pays the counters. Experience: at decode shapes these production costs
routinely exceed a modest kernel-side gain, and the only shape that fits
the budget is a producer that **replaces existing work** — fuse the layout
production into an op that already touches the data (see
[producer-fusion-pdl](producer-fusion-pdl.md)) rather than adding any new
pass.

## Why it works

The gain is bounded by the layout delta inside one kernel; the production
is real data movement plus synchronization on the critical path. Hiding it
in a different launch or phase changes who pays, not the price.

## Caveats — two correctness traps from this class of work

- **Cross-proxy ordering.** Release/acquire on a readiness counter orders
  generic global memory but is NOT a fence for a *following TMA read* of
  the same data; place `fence.proxy.async.global` between generic stores
  and any TMA consumption of them. The failure is silent and
  data-dependent.
- **CuTe swizzle offsets.** A composed `Layout_MN_SW128_Atom` carrying
  `smem_ptr_flag` must not be called directly and treated as a bf16
  physical offset for generic shared-memory access. The MN-major SW128
  mapping is
  `physical = k * M_PAD + (((m >> 3) ^ (k & 7)) << 3) + (m & 7)` —
  getting this wrong corrupts a few rows, not the whole tile, and looks
  like a math bug rather than a layout bug.
