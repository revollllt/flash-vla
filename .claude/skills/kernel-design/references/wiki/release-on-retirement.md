---
id: release-on-retirement
type: technique
arch: sm90
tags: [wgmma, ring-buffer, barriers, pipeline]
confidence: measured
---

# Release ring frames on wgmma retirement, before the epilogue

## Context

A TMA ring stalls its producers even at the measured stage knee; frames
are freed too late — after the epilogue runs, or only when the whole ring
drains.

## Move

Arrive the frame's `empty` barrier on **wgmma group retirement**
(`wait_group` at the measured knee, `[wgmma.stages.wg.knee]`), before any
epilogue work, so the producer/copy column never waits on the epilogue.
When one BK stage already commits enough wgmma instructions to fill the
pipeline, keep **one** group per stage rather than tying retirement to the
full ring depth. Derive barrier phases arithmetically — runtime-indexed
local phase arrays put local-memory loads and stores on the hot path.

## Why it works

Group retirement is the earliest point the frame is provably consumed
(the operands were read by `ldmatrix` or by the wgmma itself), so anything
later — epilogue math, stores, joins — is pure added producer latency.

## Caveats

A fused TMA+wgmma consumer often needs a deeper ring than the isolated
copy-engine knee suggests — one stage more than `[tma.stages.warp.knee]`
is common. And a knob's slope does not transfer between kernel bodies: the
same transformation can win on one body and regress its sibling.
Re-measure per body.
