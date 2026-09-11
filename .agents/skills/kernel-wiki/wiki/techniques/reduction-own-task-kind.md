---
id: technique-reduction-own-task-kind
title: "A reduction is its own task kind, never \"split 0 folds\""
type: technique
architectures: [sm90]
tags: [task-loop, split-k, reduction, pdl]
confidence: measured
reproducibility: snippet
prerequisites: [hw-pdl-gdc]
related: [pattern-serial-epilogue-owner, pattern-fusion-latency-chain, technique-prefetch-across-dependency, kernel-flashmla]
sources: [doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan, note-2026-09-03-ffn-dr-wide-tile]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
  - evidence_type: benchmark
    source_id: note-2026-09-03-ffn-dr-wide-tile
---

# A reduction is its own task kind, never "split 0 folds"

A split produces partials that must be folded: a split-K join, an attention
combine. The obvious design has the first split's CTA (or the last arriver)
reduce after its siblings publish. Make the reduction its own parallel stage
instead: a separate task kind dealt across CTAs idle in that slot, or a
separate reduce kernel, each unit folding a small fragment. Fix the fold order
explicitly: a fold whose order depends on arrival order breaks replay
bit-identity.

The single-CTA fold serializes the entire reduction behind the slowest
sibling and pushes all partial traffic through one CTA's delivery ceiling;
parallel fragments turn that into a wide, short stage. Experience: a split
kernel plus a parallel reduce beats an in-kernel last-arriver fold once the
fold reaches about 128 KB; below that the extra launch or hop can still win,
so price both.

## Source-backed fragment

The combine as a separate, trivially parallel kernel on a PDL chain, from
`13_mla_decode_split_kv.cu` in `doc-kernel-design-templates`:

```cpp
// Merges the splits of one request.  Trivially parallel and bandwidth-bound;
// its whole cost should hide under the split kernel's tail.
__global__ __launch_bounds__(128) void mla_decode_combine_kernel(
    const float* __restrict__ o_accum, const float* __restrict__ lse_accum,
    const int32_t* __restrict__ split_begin, const int32_t* __restrict__ split_count,
    float* __restrict__ o_final) {
  const int32_t request = blockIdx.x;
  const int32_t row = threadIdx.x;
```

## Caveats

Parallelizing the fold does not remove the dependency hop in front of it;
see `pattern-fusion-latency-chain`. When one worker owns the epilogue,
widening the tile makes the serial chain longer, not shorter
(`pattern-serial-epilogue-owner`).
