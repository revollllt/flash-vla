---
id: blog-marlin
title: "Marlin: mixed-precision INT4xFP16 GEMM kernels for LLM inference"
author: IST-DASLab
url: https://github.com/IST-DASLab/marlin
source_category: community-note
architectures: [sm80, sm90]
tags: [gemm, quantization, fine-grained-quantization, mma-sync]
retrieved_at: 2026-09-07
---

# Marlin

Marlin (Apache-2.0) is the INT4-weight, FP16-activation GEMM whose method
vLLM and SGLang carry: weights permuted offline so each thread's 16 bytes are
its MMA operand, `cp.async` into shared memory rather than TMA (a permuted
blob has no rectangular box), `mma.sync` at small batch, and a register-side
dequant written as bit arithmetic (`lop3` and the fixed-exponent mantissa
trick, after FasterTransformer's interleaved conversion), with per-group
scales applied to the weight before the MMA.

## What the wiki takes from it

The pack order and its inverse, the dequant vocabulary, and the rule that a
quantized kernel and its packer are one artifact. Templates `20` and `23`
of the sm90 bundle distil it; the repository's own kernels are the
production version.
