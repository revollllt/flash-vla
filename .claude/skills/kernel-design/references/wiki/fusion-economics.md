---
id: fusion-economics
type: pattern
arch: sm90
tags: [fusion, persistent-kernel, latency-chain, task-graph, pricing]
confidence: measured
---

# Price a fusion before building it: boundaries vs hops

## Context

Deciding whether to fuse ops into one persistent launch — or explaining why
a correct fused kernel does not beat the composition it replaced. The NCU
signature of the failure mode: most scheduler cycles show **no eligible
warp** while DRAM and SM utilization sit in single digits — the kernel is a
latency chain, not a throughput problem.

## Move

Price the graph before writing code:

- Removing a kernel boundary saves one launch ramp
  `[launch.lat.dev.ramp]`.
- Each in-kernel dependency costs a counter round trip
  `[atom.lat.dev.hop]` **plus** the first TMA frame at task start
  (typically 1–2 us per task kind), and each partial publish/join adds
  more — all in **series** on the critical path.
- A kernel boundary also gives a free grid-wide barrier and a free
  re-partitioning of work across CTAs; in-kernel you pay for both.
- What fusion uniquely buys: prefetching the next op's dependency-free
  inputs (weights, a KV prefix) across the boundary.

If the hops and joins outnumber the boundaries removed, the composition
wins. Re-check with a per-task timeline once the kernel runs, and expect
the op-level improvements found while fusing to transfer to unfused
kernels — harvest them whatever the fusion verdict.

## Why it works

On short-mainloop (decode-shaped) work the synchronization shape, not the
tiles, can dominate: the mainloops may account for a minority of the total
while the serial chain of hops, publishes and joins accounts for the rest.
No amount of tile tuning moves that chain — only removing links does.

## Caveats

The balance shifts when mainloops are long enough to cover the hops
(prefill/training shapes). A combine that avoids global memory entirely is
the escape hatch; DSMEM within a cluster is that route, but check cluster
co-residency first (`[cluster.count.max]`) — a persistent counter protocol
deadlocks on a non-resident cluster.
