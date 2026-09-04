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

**The trigger carries no memory ordering.** It "only enables scheduling of the
secondary kernel. It provides no memory visibility guarantee itself." If the
dependent reads what this kernel wrote, a release fence of the right scope must
precede the trigger; the trigger is not one.

**The kickoff is automatic once every CTA exits.** A trigger on the last line of
a kernel therefore fires at the moment the automatic path would have fired
anyway — it is a no-op wearing an optimization's clothes. PDL pays only when
the trigger is genuinely earlier than the kernel's end, which means the producer
must have tail work left to overlap.

So the two sides get different rules, and they are not symmetric:

- **Producer** — trigger at the earliest point after which everything the
  dependent reads is written *and fenced*. When that point is the last store,
  the kernel is a poor PDL producer; record that and take the win from the other
  side rather than leaving a trigger that does nothing.
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
| producer's trigger | earliest legal point / last line / omitted (automatic kickoff) |
| baseline | PDL off entirely, and the launch attribute removed |

Record per cell: chain latency, and the **dependent's register count and
occupancy**. Prefetching above the wait trades registers for latency hiding, so
a cell can win on latency and lose in the graph.

Two null results are common and both are findings, not failures: a dependent
with no producer-independent prologue cannot benefit from moving its wait, and a
producer whose last store is what the dependent reads cannot benefit from moving
its trigger. Either one means the pair is the wrong place for PDL.

## Caveats

- The kickoff waits for **every** CTA to trigger or exit, so one straggler sets
  the time; an early trigger in almost every CTA buys nothing.
- The device code does nothing without the launch attribute
  (`cudaLaunchAttributeProgrammaticStreamSerialization`); a chain that shows no
  change may simply not be launching that way.
- A tiny launch-bound kernel sitting on a PDL chain is a resource, and fusing it
  away can regress the chain —
  [pdl-primary-is-a-resource](pdl-primary-is-a-resource.md).
