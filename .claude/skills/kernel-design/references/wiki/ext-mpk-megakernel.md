---
id: ext-mpk-megakernel
type: external
arch: sm90
tags: [megakernel, task-graph, persistent-kernel, planner]
confidence: source-reported
---

# Megakernel task-graph interpreters (MPK / HazyResearch)

## What to read it for

The megakernel idiom: an offline CPU planner emits a static task table
plus dependency counters ("schedule as data"), and one persistent
interpreter kernel executes it ("dispatch as code") — replacing per-op
launches whose grid ramps sit inside their self-time.

## Portable design rules for the idiom

- Runtime ordering via **gmem counters** (`[atom.lat.dev.hop]`; observers
  are free, so one counter can gate many waiters) — never a cycle-count
  schedule, and no cooperative launch in the interpreter.
- The planner must emit **truncated task tables from day one**:
  persistent-kernel bugs hang rather than fail, and bisection/parity needs
  runnable table prefixes per task kind.
- Register budget = the max over op branches; the interpreter ABI splits
  into an arch-invariant core (rings, counters, descriptor format) and
  per-arch geometry (warp roles, MMA unit, copy mechanism).
- The cost side of the ledger is real: dependency hops and joins are
  serial latency. Budget
  `[launch.lat.dev.ramp]` x launches removed minus
  `[atom.lat.dev.hop]` x hops added before believing any projected win —
  see [fusion-economics](fusion-economics.md) for how that ledger can
  come out negative at one-op scope.

## Source

- Mirage Persistent Kernel: https://github.com/mirage-project/mirage
  (paper: https://arxiv.org/abs/2506.10202)
- HazyResearch megakernel writeup:
  https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

Their speedups are serving workloads on their machines; transfer the
idiom and the failure modes, not the numbers.
