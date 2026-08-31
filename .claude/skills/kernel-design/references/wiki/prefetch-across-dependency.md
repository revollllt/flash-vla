---
id: prefetch-across-dependency
type: technique
arch: sm90
tags: [task-loop, prefetch, frame-level-dependencies, scheduling]
confidence: measured
---

# Prefetch across the dependency, not across the slot

## Context

In a task loop, some CTAs sit idle in an early slot while another kind
runs, and the waiting kind's inputs are only *partly* dependent — weights
and a KV-cache prefix have no producer; only a few rows and the query do.

## Move

Deal the dependent kind onto the CTAs idle in the earlier slot, and make
dependencies **frame-level, not task-level**: issue every input frame with
no producer immediately at task start; wait only where the true dependency
lands (the produced rows' frame on its counter, the query on its
producer's counter). Do **not** free CTAs for prefetch by weakening the
producer — dropping its split parallelism exposes ring refills that
typically cost more than the prefetch buys.

## Why it works

The wait shrinks to the true dependency's tail — the slowest producer
tile — while every independent byte overlaps the producer's execution.
Task-level dependencies force the whole input set to wait on the newest
piece of it.
