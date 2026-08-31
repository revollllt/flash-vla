---
id: ext-flashmla-sm90
type: external
arch: sm90
tags: [attention, decode, kv-split, cute]
confidence: source-reported
---

# FlashMLA — sm90 decode-attention structure reference

## What to read it for

DeepSeek's MLA decode kernels: a reference for warp-role layout in a
decode-shaped attention kernel, seqlen-split scheduling with a separate
combine pass, and CuTe wgmma choreography. Its compact CuTe `sm90::gemm`
wrapper is worth reusing as-is for WGMMA choreography rather than
re-deriving the descriptor and commit/wait dance in raw PTX — rewriting it
duplicates a proven contract and makes review harder.

## What usually does not transfer

Its dynamic tile scheduler and its head/paging geometry — a fixed-shape
target compiles its geometry in. Split + combine economics also depend on
the query-row count and cache length; price them per shape (see
[fusion-economics](fusion-economics.md)).

## Source

- https://github.com/deepseek-ai/FlashMLA

Its performance claims are Hopper, but on MLA serving shapes — transfer
structure, not numbers.
