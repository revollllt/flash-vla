---
id: ext-deepgemm-sm90
type: external
arch: sm90
tags: [gemm, persistent-kernel, tma, jit-specialization]
confidence: source-reported
---

# DeepGEMM — sm90 GEMM choreography reference

## What to read it for

DeepSeek's JIT-specialized GEMM library; its sm90 path is a clean, compact
reference for: persistent tile scheduling over a fixed SM count, TMA
multicast and descriptor choreography, and block-size selection that keeps
small or odd shapes on full SM utilization — directly relevant to small-M
decode GEMMs. Its fp8 fine-grained-scaling path also shows where per-K
scaling sits in a wgmma mainloop (compare
[scale-on-register-fragment](scale-on-register-fragment.md)).

## What usually does not transfer

Its runtime tile scheduler exists for dynamic serving shapes; a workload
with a fixed offline task table does not need it, and importing a dynamic
scheduler into a static task graph adds machinery without a payer.

## Source

- https://github.com/deepseek-ai/DeepGEMM

Numbers in upstream docs are their machines and shapes — re-measure before
costing anything against them.
