---
id: technique-prefetch-across-dependency
title: "Prefetch across the dependency, not across the slot"
type: technique
architectures: [sm90]
tags: [task-loop, frame-level-dependency, data-reuse, tma, mbarrier]
confidence: measured
reproducibility: snippet
prerequisites: [hw-tma, hw-mbarrier]
related: [pattern-fusion-latency-chain, pattern-cold-burst-ceiling, kernel-megakernel-forms, technique-pipeline-stages]
sources: [doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Prefetch across the dependency, not across the slot

In a task loop, some CTAs sit idle in an early slot while another kind runs,
and the waiting kind's inputs are only partly dependent: weights and a
KV-cache prefix have no producer; only a few rows and the query do. Deal the
dependent kind onto the CTAs idle in the earlier slot, and make dependencies
frame-level, not task-level: issue every input frame with no producer
immediately at task start; wait only where the true dependency lands (the
produced rows' frame on its counter, the query on its producer's counter).

The wait shrinks to the true dependency's tail, the slowest producer tile,
while every independent byte overlaps the producer's execution. Task-level
dependencies force the whole input set to wait on the newest piece of it.
The same rule is what separates a megakernel that wins from one that loses
(`kernel-megakernel-forms`): the loader never waits on a global counter; only
the consumer's read of the activation does.

## Source-backed fragment

The placement of a wait at the data dependency rather than at task entry, in
the combine kernel of `13_mla_decode_split_kv.cu` from
`doc-kernel-design-templates`:

```cpp
  // Prologue first: reading the task metadata does not depend on the partials,
  // so it belongs before the wait.
  const int32_t begin = split_begin[request];
  const int32_t count = split_count[request];

  // PDL-WAIT: before the first read of o_accum / lse_accum.
  // DERIVED from the data dependency, never swept.
  tmpl::pdl_wait();
```

## Design checks

- Do not free CTAs for prefetch by weakening the producer: dropping its split
  parallelism exposes ring refills that typically cost more than the prefetch
  buys.
- A frame that has no producer is issued at task start whatever slot the
  task runs in; the ring depth then has to cover the dependent frames only.
- Account for the counter hop [atom.lat.dev.hop] once, at the dependent
  frame, not once per task.
