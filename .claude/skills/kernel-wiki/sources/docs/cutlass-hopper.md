---
id: doc-cutlass-hopper
title: "CUTLASS 4.7.1 Hopper headers, as pinned in third_party/cutlass"
url: https://github.com/NVIDIA/cutlass/tree/v4.7.1
source_category: official-doc
architectures: [sm90, sm90a]
tags: [wgmma, tma, mbarrier, cute-dsl, cuda-cpp]
retrieved_at: 2026-09-07
---

# CUTLASS Hopper headers at the pinned commit

This repository pins NVIDIA/cutlass as the `third_party/cutlass` submodule at
v4.7.1 (`cb4247394dd82148787aed73e5dc7cef33cbf862`). The templates in
`doc-kernel-design-templates` build against its CuTe headers, and two facts
the wiki states are read directly from them:

- `include/cute/arch/mma_sm90_gmma.hpp` enumerates the wgmma atoms for sm90:
  inputs are bf16, f16, tf32, fp8 (e4m3 and e5m2) and int8, and no e2m1 or
  other sub-8-bit atom exists. Block-scaled and sub-byte MMA atoms live in
  `mma_sm100_umma.hpp` and `mma_sm120.hpp`.
- `include/cutlass/gemm/collective/sm90_mma_*_warpspecialized*.hpp` and the
  `sm90_visitor_*` epilogue headers are the Hopper warp-specialized
  collectives the archetype templates distil; their PDL call sites carry the
  comment that trigger timing affects performance, not correctness.

The Python CuTe DSL documentation is a separate source (`doc-cutlass-cute-dsl`).
