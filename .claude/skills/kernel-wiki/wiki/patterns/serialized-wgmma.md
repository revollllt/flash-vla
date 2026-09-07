---
id: pattern-serialized-wgmma
title: "Serialized wgmma (ptxas C7518)"
type: pattern
architectures: [sm90]
tags: [wgmma, register-fragments]
symptoms: [serialized-wgmma, stall-gmma, low-tensor-core-utilization]
candidate_techniques: [technique-wgmma-rs-fragment-parity, technique-scale-on-register-fragment]
related: [pattern-compute-bound, hw-wgmma]
sources: [doc-hardware-unit-test, doc-ncu-report, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
version_sensitive:
  id: vs-cuda-13-ptxas-c7518
---

# Serialized wgmma (ptxas C7518)

## Symptom

A wgmma mainloop measures about 3x its issue floor [wgmma.issue.wg.ss] with
no memory-side explanation: per-instruction cost sits near 75-110 cycles and
stays there however the surrounding code changes. Nsight Compute shows the
`gmma` stall reason (aggregate) or `warpgroup_arrive` (per-PC sampler)
dominating while the tensor pipe is underfed, playbook Pattern Q in
`doc-ncu-report`. The build looks clean, because the compile log was filtered
to errors and register counts, and this diagnostic is a warning.

## Likely Causes

1. **RS-operand wgmma with refilled register fragments** across a runtime
   loop that also contains a divergent exit (a watchdog break): ptxas inserts
   a `WG.DP` wait before every wgmma and prints `C7518`
   (`wgmma.mma_async instructions are serialized`).
2. **A filtered build log** that hid the warning across many revisions while
   every ablation chased the wrong term.
3. Less often, `C7515` from an accumulator that is read between fence and
   wait.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Two A fragments by stage parity, unrolled](../techniques/wgmma-rs-fragment-parity.md) | Gives the in-flight group exclusive registers and removes the divergent back edge; a serialized mainloop sheds a factor of about 2-3 |
| [Scale on the register fragment](../techniques/scale-on-register-fragment.md) | The RS form that exposes the trap is worth keeping for per-K factors; pair it with the parity fix |

## Diagnosis Checklist

```
1. Read the full ptxas output for the kernel; grep C7518 and C7515.
2. Confirm the operand form: SS wgmma is not exposed; RS is.
3. Measure the mainloop against [wgmma.issue.wg.ss]; ~3x with no memory
   cause is the signature.
4. Apply fragment alternation AND the unroll, never one half alone.
5. Re-check the warning is gone and the stall reason moved off gmma.
```

## Caveats

Keep warnings visible in every build wrapper; the root failure is the
filtered log, not the instruction.
