---
id: ext-fa3-pingpong
type: external
arch: sm90
tags: [attention, warp-specialization, pingpong, softmax-overlap]
confidence: source-reported
---

# FlashAttention-3 — pingpong scheduling, and when to reach for it

## What to read it for

FA3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024) is the canonical
sm90 warp-specialized attention: TMA producer / wgmma consumer split, and
**pingpong scheduling** — two math warpgroups alternating so one runs GEMM
while the other runs softmax/epilogue, hiding the non-GEMM latency.

## How to think about it

- A second math warpgroup buys **no** tensor-core throughput
  (`[wgmma.ratio.sm.wg2]`); pingpong is a latency-hiding device, not a
  FLOPs device. Reach for it only when a compute-bound body measurably
  stalls its math column on softmax/epilogue work.
- A dedicated epilogue warp is the weaker cousin: the accumulator lives in
  the math group's register file, so handing it off costs smem staging
  plus a sync that can exceed the epilogue itself at small tiles. The
  decision rule: *does the epilogue gate the copy column or a counter
  chain, and is the handoff cheaper than the stall it removes?* If not,
  release frames early instead (see
  [release-on-retirement](release-on-retirement.md)) and let the epilogue
  run in the math group.

## Source

- Paper: https://arxiv.org/abs/2407.08608
- Code: https://github.com/Dao-AILab/flash-attention (`hopper/`)

Its benchmark shapes are long-sequence prefill/training; decode shapes
have different split/combine economics — see
[fusion-economics](fusion-economics.md).
