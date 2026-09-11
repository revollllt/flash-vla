---
id: blog-hazyresearch-megakernels
title: "HazyResearch Megakernels: no bubbles, a low-latency Llama decode step as one kernel"
author: HazyResearch (Stanford)
url: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
source_category: community-note
architectures: [sm90]
tags: [megakernel, persistent-kernel, static-scheduling, decode, fused-kernel, tma, mbarrier]
retrieved_at: 2026-09-07
---

# HazyResearch Megakernels

The blog and the repository https://github.com/HazyResearch/Megakernels (MIT)
describe a whole Llama decode step as one persistent kernel that interprets
per-SM instruction streams emitted by a host planner. The machine as upstream
builds it: sixteen consumer warps plus loader, storer, launcher and controller
warps; a two-deep instruction ring; pages of shared memory released in an
op-declared order across instruction boundaries; one global counter per
(layer, op, slice); weights prefetched under the dependency wait.

## Source-reported results

The blog reports under 1 ms per Llama-1B decode step on H100 at about 78% of
memory bandwidth. The wiki's port keeps that as the upstream number beside its
own STATUS measurement rather than as a comparable row.
