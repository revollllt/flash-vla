---
id: pattern-stacked-floors
title: "Two floors close together make every single-lever experiment read as null"
type: pattern
architectures: [sm90]
tags: [measurement, ablation, weight-streaming, pipeline-stages]
symptoms: [null-ablation, stacked-floors]
candidate_techniques: [technique-same-process-aba]
related: [pattern-one-sided-gradient, pattern-cold-burst-ceiling]
sources: [doc-hardware-unit-test, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Two floors close together make every single-lever experiment read as null

## Symptom

A phase resists every candidate. Bandwidth is freed by construction: small
gain. Transactions are halved: no gain, or a loss. The ring is deepened:
nothing. Tiles are widened: worse. Each experiment is sound and each result
is a null, and the temptation is to conclude the phase is irreducible or to
promote the last measured gradient to a "pole".

## Likely Causes

Two floors within a small factor of each other. Removing either alone
changes nothing because the other one binds, which is exactly the null
pattern. A null is evidence about the binding constraint, not about the
lever. Comparing the phase to itself under one perturbation cannot separate
stacked constraints; comparing it to the machine at the same geometry can,
because the machine pays one of the two costs and not the other.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Same-process A/B/A](../techniques/same-process-aba.md) | Take the phase's own rate and the bare machine's rate for the same geometry in one job, in two regimes: cold, and with the data made free by construction |

## Diagnosis Checklist

```
1. A phase whose cold rate matches the machine's cold rate is at that floor.
2. If its warm rate is still a large multiple of the machine's warm rate, a
   second floor (protocol, handshake, dependency) sits underneath.
3. Only a change that lowers the binding floor can pay; the lower floor's
   slack says how much room a future change would have.
4. Watch for the one-sided gradient: when taking a resource away hurts but
   adding it does nothing, the resource is not the pole, it is what keeps
   the lower floor below the upper one.
```

## Caveats

The two rates must share job, node, clock and geometry; this comparison
turns on a few percent. The bare probe's consumer differs from the kernel's,
so the protocol gap it exposes is an upper bound.
