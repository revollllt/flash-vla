---
id: doc-flash-attention-3
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
url: https://arxiv.org/abs/2407.08608
source_category: paper
architectures: [sm90]
tags: [attention, flash-attention, warp-specialization, ping-pong-scheduling, fp8, wgmma, tma]
retrieved_at: 2026-09-07
author: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
---

# FlashAttention-3

The paper (arXiv 2407.08608, 2024) describes the Hopper attention kernel
that ships as the `hopper/` directory of Dao-AILab/flash-attention. Three
techniques are named: producer/consumer warp specialization over TMA and
wgmma, overlap of block-wise GEMM and softmax through pingpong scheduling
between two math warpgroups and intra-warpgroup pipelining, and FP8 with
block quantization plus incoherent processing to bound the error.

## Source-reported results

The abstract reports a 1.5-2.0x speedup over FlashAttention-2 on H100, FP16
forward throughput of up to 740 TFLOPS (75% utilization), and FP8 close to
1.2 PFLOPS. The maximizing head dimension and sequence length are stated in
the paper's benchmark sections, not in the abstract; a wiki page that quotes
the 740 TFLOPS figure must keep it as a sweep maximum.

Its benchmark shapes are long-sequence prefill and training. Decode shapes
have different split and combine economics; the wiki's kernel page says which
mechanisms transfer.
