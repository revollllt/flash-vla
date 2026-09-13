---
id: blog-humming
title: "humming: sub-byte weight quantization, and a persistent stream-K scheduler carried across architectures"
author: inclusionAI
url: https://github.com/inclusionAI/humming
source_category: community-note
architectures: [sm90, sm120]
tags: [gemm, quantization, fp8, fp4, mma-sync, stream-k, tile-scheduling, persistent-kernel]
retrieved_at: 2026-09-07
---

# humming

humming (Apache-2.0) carries the e2m1 to e4m3 conversion the sm90 bundle's
W4A8 template uses directly: e2m1's eight magnitudes are one `prmt` source
pair, so the conversion is a register-resident lookup, and the table is built
per group by an integer multiply-add that folds the MXFP4 block scale into it.

It is also a readable stream-K implementation. `humming/include/humming/
scheduler.cuh` is a persistent scheduler that runs the bulk of the tiles
data-parallel and puts only the leftover partial wave through a K split;
`epilogue/pipeline.cuh` and `utils/ptx/barrier.cuh` hold the fixup and its
lock. One scheduler serves every target it supports — `tune/sm90.py` and
`tune/sm120.py` differ in the policy that selects and shapes it, not in the
mechanism — which is the evidence that this kind of scheduling ports where an
ISA feature does not.

## What the wiki takes from it

The fused LUT and the A8 rule that the MMA stays in the quantized domain
while the scales meet once on the accumulator. Template `22` of the sm90
bundle uses the method; `quant_sm90.cuh` names it.

The scheduler is read in [technique-stream-k](../../wiki/techniques/stream-k.md):
the remainder-only split, the slice-index reversal that decides which CTA holds
which fixup role, and the self-resetting counting lock.
