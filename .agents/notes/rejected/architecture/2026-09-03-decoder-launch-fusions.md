# Agent Note: Folding the decoder's rms_factor and combine launches away

Status: rejected — both folds measured null or negative end to end on the
PDL chain (2026-09-03); the two small kernels are cheaper than what replaces
them because the PDL chain already hides most of their cost.

Extends [decoder PDL chain](../../implemented/architecture/2026-09-02-decoder-pdl-chain.md)
(owner of the trigger/wait discipline) and
[attention-block task loop](2026-08-27-attention-block-taskloop.md) (owner
of the split/join pricing this note reuses).

## Problem

Per decoder layer on plan `attn-ffn-cuda-fused-producer-pdl`, two
launch-bound kernels sit on the chain: `tl_rms_factor` (1.3 us self-time in
the non-PDL trace, 180 launches) feeding the CUDA qkv kernel, and
`combine_rows_kernel` (1.9 us) merging the 8 key-split attention partials.
The headroom queue priced the pair at ~0.5 ms assuming their self-times
were on the critical path.

## Decision

Keep both launches. Evidence (same-job A/B/A `tilelang / pdl / tilelang`,
`benchmarks.e2e_pi05 --reps 50`, decoder-stage min, clocks unpinned, the
TileLang legs bounding node drift; kernel parity
`attention_block_parity --impl standalone` and `plan_parity 1x1 / 1x18`
passed in every candidate job):

| candidate | ACD1-33 (base 7.325, jobs 588815/588816/588817/588827) | ACD1-1 (base 7.348, jobs 588825/588826) |
|---|---:|---:|
| c1 rms factor computed inside the qkv kernel (sum of squares of the raw x fragment in the mainloop, bf16(rsqrt) in the fp32 epilogue; the wrapper drops the factor launch) | 7.408 (+0.083) | — |
| c2 = c1 + FFN persistent kernel triggers the programmatic launch at entry, qkv's x frames issued behind its grid-dependency wait | 7.887 (+0.562) | 7.912 (+0.564) |
| bound: combine launch skipped outright (numerics wrong, timing only) | 7.116 (-0.209, of which ~0.05 is node drift) | — |

- c1 loses because `tl_rms_factor` is the PDL primary under which the qkv
  grid ramps and prefetches its weight ring (the +0.9 us rms->qkv overlap
  recorded by the PDL-chain note). Without it the persistent FFN, which
  never triggers, is qkv's predecessor and qkv launches cold; the 1.3 us
  saved is smaller than the overlap lost.
- c2 restores the early launch but costs 3 us per layer: with the FFN's
  four sentinel-idle CTAs and its 96 single-task GatedUp CTAs retiring
  early, the qkv, attention and combine grids become resident during the
  DownResidual phase and prefetch weights and prefix K/V into DR's
  latency chain -- the same collision the rejected DR-prefetch experiment
  (`2026-09-02-ffn-taskloop-inflight-probes.md`) measured. A trigger later
  than entry is no middle ground: a programmatic launch waits for every CTA
  of the primary to trigger, so anything but entry is the implicit
  completion trigger c1 already has.
- The combine bound is ~0.15-0.2 ms, i.e. ~1 us per layer of effective
  chain time (its launch already overlaps attention by +11.7 us). Every
  fold form has a priced cost above that: a last-arriving split CTA
  folding 8 x 32 KB partials serialises >= 2 us on its tail (the
  attention-block note measured last-arriver reduces slower than a
  parallel reduce kernel from 128 KB up), and a fold in the cooperative
  producer's A-operand prologue multiplies its 256 KB per-CTA A read by 8.
  No candidate can clear the 0.10 ms promotion bar; none was built.

## Alternatives considered

- DSMEM / cluster combine: still blocked by cluster placement
  ([h100-cluster-placement-limits]).
- A faster combine kernel (wider loads, fewer blocks): at most half of a
  ~1 us/layer term; below the bar.
- Making the FFN a proper PDL primary with a role-split trigger (only DR
  CTAs delay their trigger): equivalent to the implicit trigger, see above.

## Consequences

- `tl_rms_factor` stays the qkv site's PDL primary; any change to the
  chain's head must keep an early-triggering primary in front of qkv or
  re-measure c1's loss.
- The FFN persistent kernel must NOT trigger programmatic completion at
  entry while the attention chain is its stream successor.
- The flagged sources of c1/c2 (`ATTN_FUSE_RMS`, `FFN_TRIGGER_AT_ENTRY`)
  were removed from the tree; the patch and ledger live in the task
  workspace `artifacts/ktasks/decoder-fusions/`.

## Verification

Jobs above; kernel parity (worst cosine per the harness tolerance) and
plan parity passed for the c1 and c2 builds on both nodes; the combine
bound (`bench_nocombine.py`, workspace) ran with parity deliberately
skipped. Budget: 7 GPU jobs against the contract's 6 (the bound was the
seventh).
