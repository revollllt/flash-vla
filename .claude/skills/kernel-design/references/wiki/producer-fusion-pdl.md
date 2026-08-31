---
id: producer-fusion-pdl
type: technique
arch: sm90
tags: [pdl, cooperative-launch, grid-sync, launch-count]
confidence: measured
---

# Fuse the producer chain; overlap the consumer with PDL

## Context

A chain of small kernels sits between two heavy stages (an epilogue, a
norm, a layout producer, a counter reset, then a persistent consumer).
Every boundary pays a launch ramp `[launch.lat.dev.ramp]`; a grid barrier
costs `[coop.lat.dev.sync]`; and a relaunch costs ~1.3x a grid_sync
(`[coop.ratio.dev.relaunch]`) — so neither "more kernels" nor "one
cooperative kernel" wins by itself unless the total count drops.

## Move

Fuse the producer chain into **one cooperative kernel** with the grid_sync
inside; fold the consumer's counter/state reset into the producer (the
graph must then not carry a standalone reset kernel); trigger PDL after
the sync; let the persistent consumer `wait` at kernel entry. An **early
trigger is memory-safe**: PTX `griddepcontrol.wait` guarantees the
prerequisite grid completed and its memory is visible, while
`.launch_dependents` only controls when the dependent may be scheduled.

## Why it works

The overlap won is the consumer's dependency-free preamble — its weight
prefetch, which needs nothing from the producer — running under the
producer's tail. That overlap only exists if the producer triggers early;
correctness is carried entirely by the consumer's wait.

## Caveats

Trigger and wait deploy as a chain, never per-kernel in isolation. A
cooperative launch needs a fail-fast residency check. Version-sensitive
(TileLang 0.1.x): the PDL launch attribute is driven by the presence of
`pdl_sync` in the kernel body, `__ldg` is rejected in such kernels, and
`__restrict__` is silently dropped from their parameters — measure that
before assuming it free. PDL survives CUDA-graph capture as a programmatic
dependency edge.
