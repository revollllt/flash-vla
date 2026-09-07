---
id: technique-external-source-map-research
title: External Source-Map Research For Kernel Edits
type: technique
architectures:
- sm100
- sm90
tags:
- cuda-cpp
- cute-dsl
- tma
- wgmma
- swizzling
- vectorized-loads
- persistent-kernel
confidence: source-reported
reproducibility: snippet
prerequisites:
- technique-vectorized-loads
- technique-swizzling
- technique-persistent-kernels
related:
- technique-pipeline-stages
- technique-tile-scheduling
- lang-cute-dsl
sources:
- blog-nvidia-code-samples
- blog-colfax-article-source-kernels
- blog-colfax-cutlass-kernels
- blog-simveit-effective-transpose
- blog-simveit-load-and-store
blackwell_relevance: These source-map repositories complement PR pages with compact, searchable examples for TMA, swizzling, Stream-K, persistent scheduling, and load/store mechanics on Hopper and Blackwell-era kernels.
---

## Use

Use external source-map research after a profile or benchmark identifies an edit
family but the local PR pages do not expose a small enough implementation
example. The route is code-first: clone a source-map repository, grep for the
measured mechanism, and cite exact files before adapting an idea.

```bash
git clone https://github.com/ColfaxResearch/cfx-article-src external/colfax-cfx
git clone https://github.com/simveit/load_and_store external/simveit-load-store
rg -n "tma|mbarrier|swizzle|ldmatrix|stream" external/colfax-cfx external/simveit-load-store
```

## When It Helps

- Long-scoreboard or poor sector utilization: search load/store and transpose
  examples before changing vector width or memory layout.
- Barrier or TMA wait stalls: search pipelined GEMM examples before changing
  stage count or producer/consumer split.
- Tail waves: search persistent and Stream-K examples before adding a
  shape-specific dispatcher.

## Provenance Rule

Do not cite this page as implementation evidence by itself. Cite one of its
source pages plus the concrete upstream file path, commit, or URL that shaped the
candidate edit.
