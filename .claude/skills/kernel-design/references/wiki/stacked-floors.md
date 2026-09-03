---
id: stacked-floors
type: pattern
arch: sm90
tags: [methodology, ablation, pipeline, dram, ring]
confidence: measured
---

# Two floors close together make every single-lever experiment read as null

## Context

A phase resists every candidate. Bandwidth is freed by construction: small
gain. Transactions are halved: no gain, or a loss. The ring is deepened:
nothing. Tiles are widened: worse. Each experiment is sound and each result
is a null, and the temptation is to conclude the phase is irreducible or, worse,
to promote the last measured gradient to a "pole".

## Move

Suspect **two floors within a small factor of each other**, and measure them
separately against the machine rather than against each other. Take the phase's
own rate and the bare machine's rate for the same geometry **in one job**, in
two regimes — cold, and with the data made free by construction. A phase whose
cold rate matches the machine's cold rate is at that floor; if its warm rate is
still a large multiple of the machine's warm rate, a second floor — protocol,
handshake, dependency — sits underneath. Removing either alone changes nothing
because the other one binds, which is exactly the null pattern.

Once the two floors are known, the decision is arithmetic: only a change that
lowers the *binding* floor can pay, and the lower floor's slack tells you how
much room a future change would have if the binding one ever moved.

## Why it works

A null is evidence about the binding constraint, not about the lever. Comparing
the phase to itself under one perturbation cannot separate stacked constraints;
comparing it to the machine at the same geometry can, because the machine pays
one of the two costs and not the other.

## Caveats

The two rates must share job, node, clock and geometry — this comparison turns
on a few percent. The bare probe's consumer differs from the kernel's, so the
protocol gap it exposes is an upper bound. And a one-sided gradient is the
signature to watch for: when taking a resource away hurts but adding it does
nothing, the resource is not the pole, it is the thing that keeps the lower
floor below the upper one (`price-the-direction`).
