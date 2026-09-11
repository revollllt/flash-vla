---
id: pattern-wgmma-tile-n-floor
title: "A wgmma below N=64 is shared-memory-bound"
type: pattern
architectures: [sm90]
tags: [wgmma, mma-sync, attention, shared-memory-optimization]
symptoms: [smem-bandwidth-bound, stall-mio-throttle, low-tensor-core-utilization]
candidate_techniques: [hw-wgmma, technique-pipeline-stages]
related: [pattern-compute-bound, pattern-serial-epilogue-owner, kernel-flash-attention-3]
sources: [doc-hardware-unit-test, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
---

# A wgmma below N=64 is shared-memory-bound

## Symptom

A wgmma-fed stage runs far over its instruction-count estimate and the
ablation points at one GEMM whose output tile N is 32; `S = Q K^T` in an
attention kernel is the classic case, because the key-tile width sets N
there. Nsight Compute shows `mio_throttle` and shared-memory bandwidth
pressure rather than a tensor-pipe limit.

## Likely Causes

Halving N halves the tensor-core work per instruction but not the A-operand
traffic, so per-useful-flop shared-memory reads double, and the mainloop
becomes a shared-memory bandwidth contest between the math warps and the
producer's landing frames. At N=32 the `m64n32k16` instruction re-reads the
full A tile per half-sized B tile and runs at about 3x
[wgmma.issue.wg.ss].

## Candidate Techniques

| Technique | Effect |
|---|---|
| [wgmma](../hardware/wgmma.md) | Keep at least 64 columns per wgmma; if the output tile is genuinely below N=32, change instruction rather than tile: `mma.sync` crosses over there [mma.xover.n.wgmma] |
| [Pipeline stages](../techniques/pipeline-stages.md) | Widening N grows the per-stage frame and ring footprint; re-run the shared-memory and ring-depth arithmetic when N moves |

## Diagnosis Checklist

```
1. Find the GEMM with N < 64 in the ablation.
2. Compare its rate with [wgmma.issue.wg.ss]; ~3x is the signature.
3. Widen N, or switch to mma.sync below the crossover.
4. Re-run the smem and ring-depth arithmetic for the new N.
```
