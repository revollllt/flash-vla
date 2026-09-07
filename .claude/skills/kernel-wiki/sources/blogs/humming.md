---
id: blog-humming
title: "humming: sub-byte weight quantization kernels with a fused prmt lookup"
author: inclusionAI
url: https://github.com/inclusionAI/humming
source_category: community-note
architectures: [sm90]
tags: [gemm, quantization, fp8, fp4, mma-sync]
retrieved_at: 2026-09-07
---

# humming

humming (Apache-2.0) carries the e2m1 to e4m3 conversion the sm90 bundle's
W4A8 template uses directly: e2m1's eight magnitudes are one `prmt` source
pair, so the conversion is a register-resident lookup, and the table is built
per group by an integer multiply-add that folds the MXFP4 block scale into it.

## What the wiki takes from it

The fused LUT and the A8 rule that the MMA stays in the quantized domain
while the scales meet once on the accumulator. Template `22` of the sm90
bundle uses the method; `quant_sm90.cuh` names it.
