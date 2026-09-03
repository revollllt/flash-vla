---
id: pdl-primary-is-a-resource
type: pattern
arch: sm90
tags: [pdl, launch, fusion, persistent, scheduling]
confidence: measured
---

# On a PDL chain, the kernel in front of you is a resource — fusing it away costs more than it saves

## Context

A chain of dependent launches runs under programmatic dependent launch. One
link is a tiny launch-bound kernel (a norm factor, a counter reset, a
combine) and folding it into its consumer looks like free money: one less
launch, one less grid ramp.

## Move

Before folding, ask what the small kernel is doing for its successor *as a
primary*. Under PDL the successor's grid ramps and prefetches its weight
ring under the primary's tail, so a 1-3 us kernel can be buying more overlap
than its own cost. Price the fold as
`saved launch - lost overlap`, and measure the fold e2e on the whole chain,
never per kernel.

Two corollaries that follow from the mechanism, not from any one kernel:

- **A persistent kernel is a bad primary.** If the predecessor is a
  persistent task loop whose CTAs retire at different times, triggering it
  at entry makes the whole successor chain resident during the loop's last,
  most latency-sensitive phase, where the extra grids prefetch into its
  dependency chain. Triggering at completion gives the successor no ramp at
  all. Neither is free; keep a cheap non-persistent kernel between them.
- **There is no middle trigger.** A programmatic launch waits for *every*
  CTA of the primary to trigger, so any placement later than entry
  degenerates to the implicit completion trigger.

## Why it works

PDL's value is ramp/prefetch overlap, which is a property of the *pair*, not
of either kernel. Removing a launch changes who the primary is, and the new
pairing can be strictly worse — a small kernel with a wide, early-triggering
grid is the ideal primary, and a persistent loop is the worst one.

## Caveats

Fold anyway when the small kernel's output is on the consumer's critical
path in a way that forces a serialization the fold removes; then the win is
the dependency, not the launch. Always keep an upper bound measurement (skip
the kernel outright, accepting wrong numerics) so a fold is never built
against a prize smaller than its own risk.
