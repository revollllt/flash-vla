---
id: technique-cluster-barrier-placement
title: "Cluster barriers: place them, count them, scope them"
type: technique
architectures: [sm90]
tags: [cluster, dsmem, mbarrier, barrier-placement, tma-multicast]
confidence: measured
reproducibility: snippet
prerequisites: [hw-cluster-dsmem, hw-mbarrier]
related: [hw-cluster-dsmem, technique-tile-scheduling, pattern-fusion-latency-chain]
sources: [doc-hardware-unit-test, doc-kernel-design-templates, doc-ptx-isa-sm90, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Cluster barriers: place them, count them, scope them

Declaring a cluster is free; the barriers and their placement are the entire
cost. Design the number of barriers, not the cluster size
[cluster.lat.sync].

## Rules

- **A barrier at the end of a kernel is nearly free; hoisting it earlier adds
  fill skew to the critical path.** The kernel's duration is already the max
  over its CTAs, so a trailing barrier only absorbs skew that was being paid
  anyway; an early one makes every CTA wait for the slowest peer while all
  still have their mainloops ahead. Do not hoist barriers on the theory that
  load latency will cover them; measure first.
- **A DSMEM reduction should push, not pull**: each owner pushes its rows,
  then one barrier, versus a sync, remote reads, and a second barrier so
  nobody retires while a peer reads. Same sum, same rank order, one barrier
  fewer. The receive buffer must not alias retired ring frames: a peer writes
  it while you may still be in your mainloop.
- **A trailing cluster_sync behind a TMA multicast prologue is removable**:
  the transaction barrier cannot fire until every peer's bytes have landed.
  Plain `st.shared::cluster` stores carry no such counter and do need the
  barrier.
- **Scope trap**: `mbarrier.arrive` defaults to `.release.cta`, and a
  cluster-addressed arrive helper can emit exactly that, a CTA-scope release
  that orders nothing for a peer reading your DSMEM stores. It fails silently
  and data-dependently. Use `cute::cluster_sync()` or explicit
  `.release.cluster` / `.acquire.cluster` (or `fence.acq_rel.cluster`) on
  both sides.
- **Placement limit**: a full-machine persistent grid with a large cluster
  size may not be co-resident [cluster.count.max], and a persistent counter
  protocol deadlocks on a non-resident cluster. The lever is occupancy, the
  placer works from blocks per SM, so shrink shared memory until two CTAs fit
  per SM, not grid size.

## Source-backed fragment

This contiguous excerpt is from `sm90_common.cuh` in
`doc-kernel-design-templates`: the peer arrive a consumer must perform on
every receiving CTA of a multicast frame, and the two-instruction cluster
barrier.

```cpp
__device__ __forceinline__ void mbarrier_arrive_cluster(const void* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               ::"r"(map_shared_rank(bar, rank)) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
```

## Caveats

Barrier cost is position-dependent, the same barrier can cost roughly double
at the start of a kernel what it costs at the end, and, like every such
slope, it does not transfer between kernel bodies. Re-measure in the body that
will carry it.
