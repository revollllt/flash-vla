---
id: c7518-wgmma-serialization
type: pattern
arch: sm90
tags: [wgmma, rs-operands, ptxas, register-fragments]
confidence: measured
---

# ptxas C7518: silently serialized wgmma

## Context

A wgmma mainloop measures ~3x its issue floor `[wgmma.issue.wg.ss]` with no
memory-side explanation: per-instruction cost sits near 75–110 cycles and
stays there however the surrounding code changes. The build looks clean —
because the compile log was filtered to errors, register counts and spills,
and this diagnostic is a warning. It can hide across many revisions while
every ablation chases the wrong term.

## Move

Read the **full** ptxas output on every compile; grep for `C7518`
(`wgmma.mma_async instructions are serialized`) and `C7515`. The classic
trigger: RS-operand wgmma whose register A fragments are refilled across a
runtime loop that also contains a divergent exit (a watchdog break) —
ptxas inserts a `WG.DP` wait before every wgmma. Fix: **two A fragments
alternated by stage parity plus a fully unrolled stage loop**, with
`wait_group<1>` so stage g's retirement frees the fragment stage g+1
overwrites:

```cpp
Tensor tCrA0 = thr.partition_fragment_A(sA);   // stage parity 0
Tensor tCrA1 = thr.partition_fragment_A(sA);   // stage parity 1
// unrolled: (g & 1) ? stage(g, tCrA1) : stage(g, tCrA0)
```

## Why it works

wgmma reads its register operands asynchronously; refilling the same
registers before the group retires forces ptxas to prove safety with a full
wait on every instruction. Alternating fragments gives the in-flight group
exclusive registers, and unrolling removes the divergent back edge that
blocks the proof. Expect a serialized mainloop to shed a factor of ~2–3
when fixed.

## Caveats

SS-operand wgmma (operands in shared memory) is not exposed to this trap.
Ship the pair — fragment alternation and the unroll — rather than either
half alone. The root failure is the filtered build log: keep warnings
visible in any build wrapper.
