---
id: pattern-epilogue-bound-short-k
title: "A short-K GEMM that plateaus under a tile sweep is epilogue-bound"
type: pattern
architectures: [sm90]
tags: [gemm, epilogue-staging, epilogue-fusion, tilelang]
symptoms: [epilogue-bound, tile-sweep-flat, low-tensor-core-utilization]
candidate_techniques: [technique-bulk-store-publish, technique-epilogue-fusion]
related: [pattern-compute-bound, pattern-serial-epilogue-owner, technique-swizzling]
sources: [note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# A short-K GEMM that plateaus under a tile sweep is epilogue-bound

## Symptom

A GEMM with K around 1K and a full-width output (M in the hundreds, N in the
thousands) sits at about 30% of the tensor-core ceiling; a tile x stages x
threads x warp-specialization sweep ranks the shipped config at or near the
top; and a library GEMM (cuBLAS) at the same shape is 30-40% faster in the
same timer. Wave quantization looks like the cause and is not: the ranking is
flat across CTA counts from half a wave to several.

## Likely Causes

1. **Fragment-layout stores.** Storing the accumulator straight to global
   leaves each thread writing 4-byte pieces in the wgmma layout, half
   efficient per sector.
2. **Scalar residual loads** in the same scattered pattern, worse still.
3. Both are hidden at long K and binding at short K: output and residual
   bytes per FLOP are 10x what they are at the long-K sites the same body
   was tuned on. The mainloop is fine; the same body reaches 60-70% of the
   ceiling when K is 16K.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Bulk store from a staged image](../techniques/bulk-store-publish.md) | Stage the tile through shared memory and let one wide copy (TMA under warp specialization) do the store |
| Residual through shared memory | Bring the residual tile in with one shared-memory copy, then add from smem |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Check whether the library already fuses the epilogue you need: cuBLASLt carries bias and tanh-GELU epilogues (`torch.addmm`, `torch._addmm_activation`), and at these shapes it beat the staged TileLang body by a further ~15%. It has no bias+residual epilogue, so residual sites stay hand-written |

Expect 8-20% at unchanged tile configs from the two staging moves.

## Diagnosis Checklist

```
1. Compare the shipped config against the library kernel in the same timer.
2. Check the sweep is flat across CTA counts; if it is, stop tuning tiles.
3. Compute output+residual bytes per FLOP at this K against the long-K site.
4. Stage the store, then the residual; measure each alone.
5. Try the library epilogue; gate it with whole-stage parity, not a
   per-kernel cosine, because it changes the reduction order.
```

## Caveats

The staged store costs a BLOCK_M x BLOCK_N bf16 tile of shared memory;
re-run the smem arithmetic if the ring is near the limit. In-place residual
(R aliases C) stays safe only if the whole R tile is read into smem before
any C element is written.
