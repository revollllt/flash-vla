---
id: blog-flashinfer-glue-kernels
title: "FlashInfer norm, activation and positional-encoding kernels"
author: flashinfer-ai
url: https://github.com/flashinfer-ai/flashinfer
source_category: community-note
architectures: [sm80, sm90, sm100]
tags: [pdl, vectorized-loads, kernel-fusion, cuda-cpp]
retrieved_at: 2026-09-07
---

# FlashInfer glue kernels

`include/flashinfer/norm.cuh`, `activation.cuh` and `pos_enc.cuh`
(Apache-2.0) are the memory-bound kernels between the GEMMs: RMSNorm with a
fused residual, activation with fused quantization, rope over two layouts.
Every one of them brackets its body with `griddepcontrol` rather than only
the interesting ones, which is the source-reported rationale the wiki's
row-traversal page carries; SGLang's per-token quantization kernels follow
the same shape.

## What the wiki takes from it

Fp32 accumulation over bf16 rows, 16-byte vector units, fusion to remove a
traversal, and PDL on every glue kernel. Templates `30`-`33` and
`elementwise_sm90.cuh` of the sm90 bundle distil them.
