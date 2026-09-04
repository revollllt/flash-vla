---
id: pdl-placement
type: technique
arch: sm90
tags: [pdl, launch-latency, kernel-chain, ablation]
confidence: source-reported
---

# The PDL wait is derived; the trigger has to be swept

## Context

A chain of small kernels carries `griddepcontrol` and did not get faster, or PDL
was added mechanically — wait on the first line, trigger on the last — and the
measurement is a null. Or a kernel is being tuned to the end and the trigger
position has never been varied.

## Move

The two halves look symmetric and are not. They differ in whether a wrong
choice is a bug or a slowdown, and that decides how each one is found.

**The wait is DERIVED. Never sweep it.** It has to sit immediately before the
first read of producer data. Later is a race, not a slower kernel. Everything
producer-independent — index arithmetic, model parameters, scheduling metadata,
tensor-map and descriptor prefetch — belongs above it, and hoisting a parameter
row up there costs registers, which is the one real trade on this side.

**The trigger is SWEPT. It cannot be derived.** It publishes nothing: the
dependent's wait is what orders memory, so every position in the kernel is
functionally correct and only measurement separates them. CUTLASS says this at
its own call site — "the timing of calling this function only influences
performance, not functional correctness" — and triggers on the last mainloop
tile, long before its epilogue writes anything.

Which position wins depends on **this kernel's structure**, not on any rule
about PDL: how much tail remains after the trigger, how much producer-independent
prologue the dependent has to fill it with, and how early the dependent may
start contending for SMs before it steals more than it saves. None of those are
visible without running the chain. In practice moving the trigger earlier often
just helps, and how much earlier it can usefully go is exactly the thing nobody
knows in advance.

Two consequences worth stating plainly:

- A trigger on the **last line is close to a no-op** — the kickoff already fires
  automatically once every CTA exits, so it lands where the automatic path would
  have landed anyway.
- **No release fence belongs before the trigger.** (The header's "provides no
  memory visibility guarantee itself" describes a dependent that reads producer
  data *before* its own wait, which is the thing not to do.)

Use `cudaGridDependencySynchronize()` and
`cudaTriggerProgrammaticLaunchCompletion()` rather than hand-rolled asm. They are
exactly these instructions, but their compiler barriers differ on purpose: the
wait is an acquire and carries a `"memory"` clobber, the trigger carries none so
independent work can still move across it. CUTLASS's helpers spell them the
same way.

## The sweep

This is a sweep a human will not run and an agent should. It is mechanical, it
is safe because no position can break the kernel, and the answer is worth a
real fraction of a latency-bound chain — which together make it a place where
batch search beats intuition.

**Make the position a compile-time knob**, so the sweep is a recompile rather
than an edit. Enumerate the candidate points in the kernel and select one:

```cuda
#ifndef PDL_TRIGGER_POINT
#define PDL_TRIGGER_POINT 2
#endif
__device__ __forceinline__ void pdl_trigger_at(int point) {
  if (point == PDL_TRIGGER_POINT) { cudaTriggerProgrammaticLaunchCompletion(); }
}
```

Candidate points are the kernel's own phase boundaries — entry, just after the
wait, after each pass stops reading producer data, after a reduction, after the
stores. A kernel with two passes has about five; that is a small enough space to
enumerate exhaustively rather than search.

**Measure the chain, not the kernel.** PDL does not exist in a single-kernel
timing ([measure-in-the-graph](measure-in-the-graph.md)): use at least three
kernels in production order, CUDA-graph captured, timed end to end, reading
`min` under unpinned clocks. Include a PDL-off cell and a launch-attribute-off
cell as baselines.

Report per cell the chain latency **and the dependent's register count and
occupancy** — a cell can win on latency and lose in the graph.

**Record the result where the next reader will look.** A placement nobody can
audit gets re-guessed by the next agent. Every PDL site should declare itself:

```
// PDL-WAIT: before the first read of `input` -- DERIVED, never swept
// PDL-TRIGGER: point 2, inputs consumed -- SWEPT: best of 5, 1.9x over point 4
```

so that a later reader can tell a measured position from an untested guess at a
glance. `scripts/check_templates.py` fails a template that uses PDL without
both declarations.

## Caveats

- The kickoff waits for **every** CTA to trigger or exit, so one straggler sets
  the time; an early trigger in almost every CTA buys nothing.
- The device code does nothing without
  `cudaLaunchAttributeProgrammaticStreamSerialization`; a chain that shows no
  change may simply not be launching that way.
- The sweep is only valid for the pair it was run on. Change either kernel's
  shape or phase structure and the winning point moves — the annotation records
  a measurement, not a constant.
- A tiny launch-bound kernel on a PDL chain is a resource, and fusing it away
  can regress the chain —
  [pdl-primary-is-a-resource](pdl-primary-is-a-resource.md).
