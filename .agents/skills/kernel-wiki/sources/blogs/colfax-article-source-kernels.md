---
id: blog-colfax-article-source-kernels
title: Colfax Article Source Kernels
author: Colfax Research
url: https://github.com/ColfaxResearch/cfx-article-src
source_category: community-note
architectures:
- sm90
- sm100
tags:
- cuda-cpp
- cute-dsl
- gemm
- tma
- wgmma
- pipeline-stages
- persistent-kernel
- swizzling
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for TMA, pipelined GEMM, Stream-K, and CuTe transpose examples.
---

This repository collects source files for Colfax Research articles. The useful
KernelWiki path is code-first: inspect the TMA, pipeline GEMM, Stream-K,
transpose-cute, and CUTLASS GEMM folders when a profile points to pipeline
bubbles, tail effects, or memory-layout pressure.
