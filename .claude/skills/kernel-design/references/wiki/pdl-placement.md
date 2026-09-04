---
id: pdl-placement
type: technique
arch: sm90
tags: [pdl, launch-latency, kernel-chain, ablation]
confidence: source-reported
---

# Where the PDL wait and trigger go, and how to find out

## Context

A chain of small kernels carries `griddepcontrol` and the chain did not get
faster, or PDL was added mechanically — wait on the first line, trigger on the
last — and the measurement is a null. The placement is usually the reason, and
two facts from NVIDIA's own device-runtime header decide it.

## Move

**The trigger publishes nothing.** It only lets the dependent be *scheduled*
earlier, so the dependent's pre-wait prologue can run while this kernel is still
going. The ordering comes from the other side: the wait "blocks the thread until
all direct grid dependencies have completed", which is the same guarantee
ordinary stream serialization gives. CUTLASS states the consequence at its own
call site — "the timing of calling this function only influences performance,
not functional correctness" — and places the trigger on the last mainloop tile,
long before its epilogue writes anything.

Two things follow, and they surprise people in opposite directions:

- **No release fence belongs before the trigger.** The dependent's wait does
  that job. (The header's "provides no memory visibility guarantee itself"
  describes a dependent that reads producer data *before* its own wait — which
  is the thing not to do, not a reason to fence.)
- **A trigger on the last line is close to a no-op**, because the kickoff
  happens automatically once every CTA has exited. It fires when the automatic
  path would have fired anyway.

So the two sides get opposite rules:

- **Producer** — trigger EARLY. There is no data-readiness point to respect, so
  the only question is how soon the dependent may start competing for SMs. The
  natural place is as soon as this kernel stops reading its inputs; a
  single-pass kernel with no tail can trigger immediately after its wait.
- **Dependent** — wait at the *latest* point before the first read of producer
  data. Index arithmetic, model parameters, scheduling metadata and tensor-map
  or descriptor prefetch all belong above it. Hoisting a parameter row above the
  wait costs registers, which is a real trade against occupancy, not a free win.

Use `cudaGridDependencySynchronize()` and
`cudaTriggerProgrammaticLaunchCompletion()` rather than hand-rolled asm. They
are exactly these instructions, but their compiler barriers differ on purpose:
the wait is an acquire and carries a `"memory"` clobber, the trigger carries
none so the compiler stays free to move independent work across it. Hand-written
versions tend to get one of the two wrong in the direction that hurts.

## The ablation

PDL exists only in a chain, so a per-kernel timing cannot see it
([measure-in-the-graph](measure-in-the-graph.md)). The harness is a chain of at
least three kernels in production order, CUDA-graph captured, measured
end-to-end; read `min` under unpinned clocks.

Vary one placement at a time, and in **both** directions — move it earlier and
later ([price-the-direction](price-the-direction.md)). A lever that only ever
helps is usually mis-measured.

| axis | cells |
|---|---|
| dependent's wait | kernel entry / after parameter prefetch / immediately before the first producer read |
| producer's trigger | immediately after the wait / when inputs stop being read / last line / omitted (automatic kickoff) |
| baseline | PDL off entirely, and the launch attribute removed |

Record per cell: chain latency, and the **dependent's register count and
occupancy**. Prefetching above the wait trades registers for latency hiding, so
a cell can win on latency and lose in the graph.

One null result is common and it is a finding, not a failure: a dependent with
no producer-independent prologue has nothing to overlap, so moving its wait
cannot help and triggering earlier only lets it contend for SMs sooner. That
contention is the real cost of an early trigger, and it is why "as early as
possible" is a starting point for the sweep rather than the answer.

## Caveats

- The kickoff waits for **every** CTA to trigger or exit, so one straggler sets
  the time; an early trigger in almost every CTA buys nothing.
- The device code does nothing without the launch attribute
  (`cudaLaunchAttributeProgrammaticStreamSerialization`); a chain that shows no
  change may simply not be launching that way.
- A tiny launch-bound kernel sitting on a PDL chain is a resource, and fusing it
  away can regress the chain —
  [pdl-primary-is-a-resource](pdl-primary-is-a-resource.md).
