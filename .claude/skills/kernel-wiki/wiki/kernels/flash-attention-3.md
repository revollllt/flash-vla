---
id: kernel-flash-attention-3
title: "FlashAttention-3 (Hopper): warp specialization and pingpong, and when to reach for it"
type: kernel
architectures: [sm90]
tags: [attention, flash-attention, warp-specialization, ping-pong-scheduling, online-softmax, wgmma, tma, fp8]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, flash-attention, prefill]
languages: [cuda-cpp, cute-dsl]
related: [kernel-flash-attention-4, technique-warp-specialization, technique-ping-pong-scheduling, hw-wgmma, technique-release-on-retirement, pattern-fusion-latency-chain]
sources: [doc-flash-attention-3, doc-hardware-unit-test, doc-kernel-design-templates]
performance_claims:
  - gpu: H100
    dtype: fp16
    shape: "paper-reported forward sweep; exact maximizing head dimension and sequence length not stated in the abstract"
    metric: TFLOPS
    value: 740
    utilization: "75%"
    source_id: doc-flash-attention-3
    source_locator: "arXiv 2407.08608 abstract"
  - gpu: H100
    dtype: fp8
    shape: "paper-reported forward sweep; shape not stated in the abstract"
    metric: TFLOPS
    value: 1200
    source_id: doc-flash-attention-3
    source_locator: "arXiv 2407.08608 abstract, stated as close to 1.2 PFLOPS"
---

# FlashAttention-3 (Hopper)

FA3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024) is the canonical
sm90 warp-specialized attention: a TMA producer warpgroup and wgmma consumer
warpgroups, and pingpong scheduling, two math warpgroups alternating so one
runs GEMM while the other runs softmax and epilogue, hiding the non-GEMM
latency. Code: the `hopper/` directory of Dao-AILab/flash-attention.

## How to think about it

- A second math warpgroup buys no tensor-core throughput
  [wgmma.ratio.sm.wg2]; pingpong is a latency-hiding device, not a FLOPs
  device. Reach for it only when a compute-bound body measurably stalls its
  math column on softmax or epilogue work.
- A dedicated epilogue warp is the weaker cousin: the accumulator lives in
  the math group's register file, so handing it off costs shared-memory
  staging plus a sync that can exceed the epilogue itself at small tiles.
  The decision rule: does the epilogue gate the copy column or a counter
  chain, and is the handoff cheaper than the stall it removes? If not,
  release frames early instead (`technique-release-on-retirement`) and let
  the epilogue run in the math group.
- Its benchmark shapes are long-sequence prefill and training; decode
  shapes have different split and combine economics
  (`pattern-fusion-latency-chain`, `kernel-flashmla`).

## Source-backed fragment

The online-softmax rescale as the sm90 template `12_attention_online_softmax.cu`
in `doc-kernel-design-templates` spells it: `exp2f` with log2(e) folded into
the scale, and a correction for everything accumulated under the old maximum.

```cpp
    const float new_m_0 = fmaxf(m_0, t_0 * softmax_scale);
    const float new_m_1 = fmaxf(m_1, t_1 * softmax_scale);
    // Correction for everything already accumulated under the old maximum.
    const float corr_0 = exp2f((m_0 - new_m_0) * 1.4426950408889634f);
    const float corr_1 = exp2f((m_1 - new_m_1) * 1.4426950408889634f);
    m_0 = new_m_0;
    m_1 = new_m_1;
```

## Performance boundary

The abstract's 740 TFLOPS FP16 (75% utilization) and about 1.2 PFLOPS FP8
are sweep maxima on H100; the maximizing shape is in the paper's benchmark
section, not restated here.
