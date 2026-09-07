---
id: doc-ptx-isa-sm90
title: "PTX ISA: the Hopper (sm_90a) mechanisms this wiki cites"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm90, sm90a]
tags: [ptx, wgmma, tma, mbarrier, cluster, dsmem, pdl, setmaxnreg, ldmatrix, cache-hint]
retrieved_at: 2026-09-07
---

# PTX ISA source map for sm90

This page maps the wiki's Hopper claims to NVIDIA's rolling PTX ISA, accessed
on 2026-09-07. Section numbers shift between ISA versions, so locators below
are section titles; the exact qualifier sets and target notes must be read
from the current document, and every instruction form the templates emit is
asserted against `ptxas` output by `check_templates.py` rather than quoted
from memory.

## Asynchronous warpgroup-level matrix multiply (wgmma)

`wgmma.mma_async` is issued collectively by a four-warp warpgroup. Operands A
and B come from shared memory through matrix descriptors (SS form), or A from
registers (RS form); D accumulates in registers. A batch is bracketed by
`wgmma.fence`, the instructions, `wgmma.commit_group`, and
`wgmma.wait_group N`, which retires all but the newest N groups. Shape
`m64nNk16` with N a multiple of 8 up to 256 for bf16/f16 inputs.

Locators: "Asynchronous Warpgroup Level Matrix Multiply-Accumulate
Instructions" and its subsections "Matrix Descriptor Format", "wgmma.fence",
"wgmma.commit_group", "wgmma.wait_group".

## Tensor memory accelerator and bulk copies

`cp.async.bulk.tensor` copies a descriptor-defined box; the global-to-shared
form completes on an mbarrier through `mbarrier::complete_tx::bytes`, with an
optional `.multicast::cluster` mask. `cp.async.bulk` moves a contiguous
16-byte-aligned span without a tensor map. Shared-to-global forms complete
through bulk groups (`cp.async.bulk.commit_group`,
`cp.async.bulk.wait_group[.read] N`). `cp.async.bulk.prefetch.L2` warms L2
without landing data.

Locators: "Data Movement and Conversion Instructions: cp.async.bulk",
"cp.async.bulk.tensor", "cp.async.bulk.prefetch", "cp.async.bulk.wait_group".

## mbarrier

`mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.arrive`,
`mbarrier.try_wait.parity`, and `fence.mbarrier_init.release.cluster`.
Arrival count and transaction count are two accounting dimensions; a phase
completes when both are satisfied. `.shared::cluster` forms address a peer
CTA's barrier.

Locators: "Parallel Synchronization and Communication Instructions:
mbarrier", "mbarrier.expect_tx", "mbarrier.try_wait".

## Proxies and fences

Generic-proxy stores are not ordered against async-proxy reads of the same
bytes by release/acquire alone. `fence.proxy.async.shared::cta` orders shared
writes before a TMA store or a wgmma descriptor read; `fence.proxy.async.global`
orders global writes before a later TMA read.

Locators: "Memory Consistency Model: Proxies", "Membar/Fence Instructions:
fence.proxy".

## Clusters and distributed shared memory

`mapa.shared::cluster` maps a CTA-local shared address into a peer CTA;
`barrier.cluster.arrive` / `barrier.cluster.wait` synchronize the cluster;
`%cluster_ctarank` names the issuing CTA. `mbarrier.arrive` defaults to
`.release.cta` scope; ordering DSMEM stores for a peer needs `.cluster`
scope on both sides.

Locators: "Cluster-related special registers", "mapa", "barrier.cluster",
"Memory Consistency Model: scopes".

## Programmatic dependent launch

`griddepcontrol.wait` blocks until all prerequisite grids have completed and
their memory is visible; `griddepcontrol.launch_dependents` only permits the
dependent grid to be scheduled. Available from `sm_90`.

Locators: "Miscellaneous Instructions: griddepcontrol".

## Register redistribution, election, matrices, cache hints

`setmaxnreg.inc` / `setmaxnreg.dec` are `.sync.aligned` over a warpgroup;
`elect.sync` picks one lane; `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
moves four 8x8 tiles into mma fragment layout; `createpolicy.fractional` and
the `L2::cache_hint` / `L1::no_allocate` qualifiers on `ld.global` express
cache policy. `ld.global.L2::evict_first` is not a legal spelling on sm_90a;
the policy goes through `createpolicy` plus `L2::cache_hint`.

Locators: "setmaxnreg", "elect.sync", "ldmatrix", "createpolicy",
"Cache Operators", "ld".
