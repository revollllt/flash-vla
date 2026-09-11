---
id: technique-release-on-retirement
title: "Release ring frames on wgmma retirement, before the epilogue"
type: technique
architectures: [sm90]
tags: [wgmma, mbarrier, pipeline-stages, double-buffering, tma]
confidence: measured
reproducibility: snippet
prerequisites: [hw-wgmma, hw-tma, hw-mbarrier]
related: [technique-pipeline-stages, technique-double-buffering, technique-bulk-store-publish, pattern-serialized-wgmma]
sources: [doc-hardware-unit-test, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan, note-2026-09-03-ffn-gu-ring-depth]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-03-ffn-gu-ring-depth
---

# Release ring frames on wgmma retirement, before the epilogue

A TMA ring stalls its producers even at the measured stage knee when frames
are freed too late: after the epilogue runs, or only when the whole ring
drains. Arrive the frame's `empty` barrier on wgmma group retirement
(`wait_group` at the measured knee [wgmma.stages.wg.knee]), before any
epilogue work, so the producer and copy column never wait on the epilogue.
Group retirement is the earliest point the frame is provably consumed, the
operands were read by `ldmatrix` or by the wgmma itself, so anything later,
epilogue math, stores, joins, is pure added producer latency.

When one BK stage already commits enough wgmma instructions to fill the
pipeline, keep one group per stage rather than tying retirement to the full
ring depth. Derive barrier phases arithmetically; runtime-indexed local phase
arrays put local-memory loads and stores on the hot path.

## Source-backed fragment

This contiguous excerpt is from `03_wgmma_mainloop.cu` in
`doc-kernel-design-templates`: one batch in flight, and the previous stage's
frame released only after the wait that retires the batch reading it.

```cpp
    warpgroup_commit_batch();
    // One batch stays in flight: the previous stage's math overlaps this
    // stage's copies.  warpgroup_wait<0> here would serialize the mainloop.
    warpgroup_wait<1>();
    warpgroup_fence_operand(acc);

    // Released only after the wait above retires the batch that reads it --
    // releasing on issue would let the producer overwrite a live operand.
    if (stage > 0) {
      const int32_t prev = (stage - 1) % kDepth;
      tmpl::mbarrier_arrive(&empty[prev]);
    }
```

## Caveats

A fused TMA+wgmma consumer often needs a deeper ring than the isolated
copy-engine knee suggests; one stage more than [tma.stages.warp.knee] is
common. And a knob's slope does not transfer between kernel bodies: the same
transformation can win on one body and regress its sibling. Re-measure per
body.
