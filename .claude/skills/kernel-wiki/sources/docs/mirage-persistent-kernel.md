---
id: doc-mirage-mpk
title: "Mirage Persistent Kernel: compiling LLM inference into a task-graph megakernel"
url: https://arxiv.org/abs/2512.22219
source_category: paper
architectures: [sm90, sm100]
tags: [megakernel, persistent-kernel, static-scheduling, decode, fused-kernel]
retrieved_at: 2026-09-07
author: Mirage project (mirage-project/mirage)
---

# Mirage Persistent Kernel (MPK)

MPK compiles a model's forward pass into tasks and events; one persistent
kernel runs worker CTAs that consume per-worker task queues and scheduler
warps that consume event queues, with cumulative counters carrying the kernel
across decode steps. Code: https://github.com/mirage-project/mirage
(Apache-2.0).

## What the wiki takes from it

The task/event runtime, event granularity as a compiler decision, and the
choice between pre-enqueueing the whole graph (the current runtime) and
launching dependents when an event fires. The sm90 template that reproduces
the runtime and the measurements taken on it are recorded under
`doc-kernel-design-templates`; upstream's published speedups are serving
workloads on its own machines and are not restated here.
