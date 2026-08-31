---
id: cluster-barrier-placement
type: technique
arch: sm90
tags: [cluster, dsmem, mbarrier, barrier-placement, occupancy]
confidence: measured
---

# Cluster barriers: place them, count them, scope them

## Context

Any cluster/DSMEM design. Declaring a cluster is free; the barriers and
their placement are the entire cost.

## Move

- **Design the number of barriers, not the cluster size**
  (`[cluster.lat.sync]`).
- **A barrier at the end of a kernel is nearly free; hoisting it earlier
  adds fill skew to the critical path.** The kernel's duration is already
  the max over its CTAs, so a trailing barrier only absorbs skew that was
  being paid anyway; an early one makes every CTA wait for the slowest
  peer while all still have their mainloops ahead. Do not hoist barriers
  on the theory that load latency will cover them — measure first.
- **A DSMEM reduction should PUSH, not pull**: each owner pushes its rows,
  then one barrier — versus a sync, remote reads, and a second barrier so
  nobody retires while a peer reads. Same sum, same rank order, one
  barrier fewer. The receive buffer must not alias retired ring frames: a
  peer writes it while you may still be in your mainloop.
- **A trailing cluster_sync behind a TMA multicast prologue is
  removable**: the transaction barrier cannot fire until every peer's
  bytes have landed. Plain `st.shared::cluster` stores carry no such
  counter and do need the barrier.
- **Scope trap**: `mbarrier.arrive` defaults to `.release.cta`, and a
  cluster-addressed arrive helper can emit exactly that — a CTA-scope
  release that orders *nothing* for a peer reading your DSMEM stores. It
  fails silently and data-dependently. Use `cute::cluster_sync()` or
  explicit `.release.cluster` / `.acquire.cluster` (or
  `fence.acq_rel.cluster`) on both sides.
- **Placement limit**: a full-machine persistent grid with a large cluster
  size may not be co-resident (`[cluster.count.max]`), and a persistent
  counter protocol deadlocks on a non-resident cluster. The lever is
  occupancy — the placer works from blocks-per-SM, so shrink shared memory
  until two CTAs fit per SM — not grid size.

## Caveats

Barrier cost is position-dependent — the same barrier can cost roughly
double at the start of a kernel what it costs at the end — and, like every
such slope, it does not transfer between kernel bodies. Re-measure in the
body that will carry it.
