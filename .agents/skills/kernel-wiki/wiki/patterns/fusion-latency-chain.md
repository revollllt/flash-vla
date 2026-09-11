---
id: pattern-fusion-latency-chain
title: "A fused persistent kernel that loses to the launches it replaced"
type: pattern
architectures: [sm90]
tags: [kernel-fusion, fusion-pricing, persistent-kernel, task-loop, megakernel]
symptoms: [no-eligible-warp, latency-bound, fusion-regression]
candidate_techniques: [technique-producer-fusion-pdl, technique-reduction-own-task-kind, technique-prefetch-across-dependency, hw-pdl-gdc]
related: [kernel-megakernel-forms, pattern-pdl-primary-is-a-resource, pattern-layout-production-cost, pattern-low-sm-utilization]
sources: [doc-hardware-unit-test, doc-ncu-report, note-2026-09-06-optimization-campaign-plan, note-2026-09-07-expert-megakernel-pricing]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-07-expert-megakernel-pricing
---

# A fused persistent kernel that loses to the launches it replaced

## Symptom

Deciding whether to fuse ops into one persistent launch, or explaining why a
correct fused kernel does not beat the composition it replaced. The Nsight
Compute signature: most scheduler cycles show no eligible warp while DRAM and
SM utilization sit in single digits. The kernel is a latency chain, not a
throughput problem (playbook Pattern P in `doc-ncu-report`).

## Likely Causes

On short-mainloop (decode-shaped) work the synchronization shape, not the
tiles, dominates: the mainloops may account for a minority of the total while
the serial chain of hops, publishes and joins accounts for the rest. No
amount of tile tuning moves that chain; only removing links does.

## Price the graph before writing code

- Removing a kernel boundary saves one launch ramp [launch.lat.dev.ramp].
- Each in-kernel dependency costs a counter round trip [atom.lat.dev.hop]
  plus the first TMA frame at task start (typically 1-2 us per task kind),
  and each partial publish and join adds more, all in series on the critical
  path.
- A kernel boundary also gives a free grid-wide barrier and a free
  re-partitioning of work across CTAs; in-kernel you pay for both.
- What fusion uniquely buys: prefetching the next op's dependency-free inputs
  (weights, a KV prefix) across the boundary.

If the hops and joins outnumber the boundaries removed, the composition
wins.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Producer fusion with PDL](../techniques/producer-fusion-pdl.md) | Removes launches without adding hops; the consumer's preamble overlaps the producer's tail |
| [Reduction as its own task kind](../techniques/reduction-own-task-kind.md) | Turns a serial fold into a wide stage; does not remove the hop in front of it |
| [Prefetch across the dependency](../techniques/prefetch-across-dependency.md) | Puts the hop under the weight stream instead of in front of it |
| [PDL](../hardware/pdl-gdc.md) | The boundary keeps its free barrier and re-partitioning |

## Diagnosis Checklist

```
1. Count boundaries removed x [launch.lat.dev.ramp] against hops added x
   [atom.lat.dev.hop] plus publishes and joins, in series.
2. Confirm the ncu signature: no-eligible-warp dominant, DRAM and SM in
   single digits.
3. Re-check with a per-task timeline once the kernel runs.
4. Harvest the op-level improvements found while fusing into the unfused
   kernels, whatever the fusion verdict.
```

## Caveats

The balance shifts when mainloops are long enough to cover the hops (prefill
and training shapes). A combine that avoids global memory entirely is the
escape hatch; DSMEM within a cluster is that route, but check cluster
co-residency first [cluster.count.max]: a persistent counter protocol
deadlocks on a non-resident cluster.
