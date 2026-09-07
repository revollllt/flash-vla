---
id: blog-simveit-load-and-store
title: simveit load_and_store
author: Simon Veitner
url: https://github.com/simveit/load_and_store
source_category: community-note
architectures:
- sm90
- sm100
tags:
- cute-dsl
- gemm
- tma
- wgmma
- vectorized-loads
- shared-memory-optimization
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for CuTe load/store and shared-memory movement examples.
---

The load_and_store repository is a source-map route for CuTe load/store
mechanics. It is most useful when converting a profiler symptom into a concrete
data-movement edit: vector width, layout, shared staging, or TMA-friendly access
structure.
