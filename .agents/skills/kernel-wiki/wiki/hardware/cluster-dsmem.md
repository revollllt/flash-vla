---
id: hw-cluster-dsmem
title: "Thread-block clusters and distributed shared memory (DSMEM)"
type: hardware
architectures: [sm90, sm90a, sm100]
tags: [cluster, dsmem, mbarrier, tma-multicast]
confidence: measured
reproducibility: snippet
related: [hw-tma, hw-mbarrier, technique-cluster-barrier-placement, technique-tile-scheduling, hw-2sm-cooperative]
sources: [doc-ptx-isa-sm90, doc-hardware-unit-test, doc-kernel-design-templates]
evidence_basis:
  - evidence_type: official-doc
    source_id: doc-ptx-isa-sm90
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: upstream-code
    source_id: doc-kernel-design-templates
aliases: [cluster, DSMEM, "distributed shared memory", "thread block cluster", mapa]
---

# Thread-block clusters and distributed shared memory (DSMEM)

A cluster is a group of co-scheduled CTAs that can address each other's
shared memory (`mapa.shared::cluster`), arrive on each other's mbarriers
(`mbarrier.arrive.shared::cluster`), receive one TMA box by multicast
(`.multicast::cluster` with a CTA mask), and synchronize with
`barrier.cluster.arrive` / `barrier.cluster.wait`. Declaring a cluster is
free; the barriers and their placement are the entire cost
(`technique-cluster-barrier-placement`).

## Machine facts, by tag

- A cluster barrier costs [cluster.lat.sync], and about double at the start
  of a kernel what it costs at the end.
- Co-residency of a full-machine persistent grid is bounded by
  [cluster.count.max]; a persistent counter protocol deadlocks on a
  non-resident cluster. The lever is occupancy (blocks per SM), not grid
  size.

## Source-backed fragment

From `sm90_common.cuh` in `doc-kernel-design-templates`: the rank register,
the address map into a peer, and the peer arrive a consumer must perform on
every receiving CTA of a multicast frame.

```cpp
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

// Maps a CTA-local shared address into another CTA of the cluster (DSMEM).
__device__ __forceinline__ uint32_t map_shared_rank(const void* p, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(out)
               : "r"(smem_u32(p)), "r"(rank));
  return out;
}
```

## Design checks

- Only one CTA issues a multicast; each receiving CTA arms its own barrier
  for the bytes it will receive.
- `mbarrier.arrive` defaults to `.release.cta`; ordering DSMEM stores for a
  peer needs `.cluster` scope on both sides or `cute::cluster_sync()`.
- A DSMEM barrier whose remote arrivals are still in flight cannot be safely
  deconstructed when the kernel exits: the cluster teardown wait is not
  optional.
- Grid size must be a whole number of clusters, and a 128-CTA grid at one CTA
  per SM cannot use cluster 4 or 8 without deadlock risk.
