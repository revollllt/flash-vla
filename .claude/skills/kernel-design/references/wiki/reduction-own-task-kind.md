---
id: reduction-own-task-kind
type: technique
arch: sm90
tags: [task-loop, split-k, reduction, combine]
confidence: measured
---

# A reduction is its own task kind, never "split 0 folds"

## Context

A split produces partials that must be folded — a split-K join, an
attention combine. The obvious design has the first split's CTA (or the
last arriver) reduce after its siblings publish.

## Move

Make the reduction **its own parallel stage**: a separate task kind dealt
across CTAs idle in that slot, or a separate reduce kernel — each unit
folding a small fragment. Fix the fold order explicitly: a fold whose order
depends on arrival order breaks replay bit-identity.

## Why it works

The single-CTA fold serializes the entire reduction behind the slowest
sibling *and* pushes all partial traffic through one CTA's delivery
ceiling; parallel fragments turn that into a wide, short stage. Experience:
a split kernel plus a parallel reduce beats an in-kernel last-arriver fold
once the fold reaches ~128 KB — below that, the extra launch or hop can
still win, so price both.

## Caveats

Parallelizing the fold does not remove the dependency hop in front of it —
see [fusion-economics](fusion-economics.md).
