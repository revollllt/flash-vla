---
id: technique-pdl-placement
title: "The PDL wait is derived; the trigger has to be swept"
type: technique
architectures: [sm90]
tags: [pdl, griddepcontrol, pdl-placement, ablation, measurement]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-pdl-gdc]
related: [hw-pdl-gdc, technique-producer-fusion-pdl, pattern-pdl-primary-is-a-resource, pattern-isolated-timer-overstates-fusion]
sources: [doc-cuda-programming-guide-pdl, doc-ptx-isa-sm90, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
---

# The PDL wait is derived; the trigger has to be swept

A chain of small kernels carries `griddepcontrol` and did not get faster, or
PDL was added mechanically, wait on the first line and trigger on the last,
and the measurement is a null. The two halves look symmetric and are not.
They differ in whether a wrong choice is a bug or a slowdown, and that
decides how each one is found.

**The wait is derived. Never sweep it.** It has to sit immediately before the
first read of producer data. Later is a race, not a slower kernel. Everything
producer-independent, index arithmetic, model parameters, scheduling
metadata, tensor-map and descriptor prefetch, belongs above it; hoisting a
parameter row up there costs registers, which is the one real trade on this
side.

**The trigger is swept. It cannot be derived.** It publishes nothing: the
dependent's wait is what orders memory, so every position in the kernel is
functionally correct and only measurement separates them. CUTLASS says this
at its own call site, "the timing of calling this function only influences
performance, not functional correctness", and triggers on the last mainloop
tile, long before its epilogue writes anything. Which position wins depends
on this kernel's structure: how much tail remains after the trigger, how much
producer-independent prologue the dependent can fill it with, and how early
the dependent may start contending for SMs before it steals more than it
saves.

Two consequences: a trigger on the last line is close to a no-op, because
the kickoff already fires once every CTA exits; and no release fence belongs
before the trigger. Use `cudaGridDependencySynchronize()` and
`cudaTriggerProgrammaticLaunchCompletion()` rather than hand-rolled asm; they
are exactly these instructions, but the wait carries a `"memory"` clobber and
the trigger carries none, so independent work can still move across it.

## The sweep

Make the position a compile-time knob so the sweep is a recompile rather
than an edit; candidate points are the kernel's own phase boundaries, about
five in a two-pass kernel, few enough to enumerate:

```cuda
#ifndef PDL_TRIGGER_POINT
#define PDL_TRIGGER_POINT 2
#endif
__device__ __forceinline__ void pdl_trigger_at(int point) {
  if (point == PDL_TRIGGER_POINT) { cudaTriggerProgrammaticLaunchCompletion(); }
}
```

Measure the chain, not the kernel: at least three kernels in production
order, CUDA-graph captured, timed end to end, reading `min` under unpinned
clocks, with a PDL-off cell and a launch-attribute-off cell as baselines.
Report per cell the chain latency and the dependent's register count and
occupancy. Record the result where the next reader will look; every PDL site
declares itself, and `check_templates.py` fails a template that uses PDL
without both lines:

```
// PDL-WAIT: before the first read of `input` -- DERIVED, never swept
// PDL-TRIGGER: point 2, inputs consumed -- SWEPT: best of 5, 1.9x over point 4
```

## Caveats

- The kickoff waits for every CTA to trigger or exit, so one straggler sets
  the time; an early trigger in almost every CTA buys nothing.
- The device code does nothing without
  `cudaLaunchAttributeProgrammaticStreamSerialization`; a chain that shows no
  change may simply not be launching that way.
- The sweep is only valid for the pair it was run on. Change either kernel's
  shape or phase structure and the winning point moves; the annotation
  records a measurement, not a constant.
- A tiny launch-bound kernel on a PDL chain is a resource, and fusing it away
  can regress the chain: `pattern-pdl-primary-is-a-resource`.
