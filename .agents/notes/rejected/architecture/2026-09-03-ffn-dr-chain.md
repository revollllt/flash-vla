# Agent Note: DownResidual's cost is diffuse — no single link pays, and neither bytes nor transactions are the pole

Status: rejected — measured (jobs 589228 / 589273 / 589331). Every link of the
DownResidual dependency chain was priced by removing it; the three candidates
the pricing justified all landed under the promotion bar, and two competing
hypotheses about the phase (weight bytes, TMA transaction count) are refuted
by their own bounds. The one structural change that attacks several links at
once — a 64-wide DownResidual output tile — is named here as a separate lane.

Date: 2026-09-03

## Problem

`2026-09-03-ffn-stream-continuity.md` closed the bandwidth family for the
persistent FFN task loop and left DownResidual as the target: 10.5 us/layer
for 0.54 GFLOP (compute floor 0.64 us) with only 0.25 us of exposed weight
stream, i.e. ~1.89 ms of the decoder's 7.44 ms sitting in a serial chain —
readiness poll, first activation load, wgmma stages, split-K join, fold,
residual read-modify-write.

## Decision

Reject all three candidates and record the decomposition, because the
decomposition is the durable result.

### The chain, priced link by link (job 589228, one job, one node)

Each build deletes exactly one link and keeps every TMA, barrier and wgmma;
numerics are invalid by construction, timing is valid. `txns` is the TMA
transactions a DownResidual CTA issues per task, unchanged by every row.

| link removed | txns/CTA | fused | Δfused | dr-only | Δdr-only |
|---|---:|---:|---:|---:|---:|
| baseline | 25 | 19.57 | — | 10.31 | — |
| GatedUp readiness poll | 25 | 19.03 | −0.54 | 9.26 | −1.05 |
| split-K join wait | 25 | 18.61 | −0.96 | 10.37 | +0.06 |
| 3-partial fold reads | 25 | 18.47 | −1.10 | 8.94 | −1.37 |
| residual read | 25 | 18.74 | −0.83 | 9.62 | −0.69 |

The four links sum to 3.43 us/layer of fused time, so the pricing said a
chain rewrite was worth up to ~0.6 ms over 180 launches.

Transaction accounting for the table above, for reuse: a DownResidual CTA
issues 8 weight boxes (8 KiB each), 16 activation boxes (2 x 8 KiB per stage,
the SW128 row limit caps the hidden descriptor's innermost box at
[K64, M64]), and 1 L2 prefetch, over 8 stages — 25 transactions. GatedUp
issues 4 + 4 over 4 stages at 32 KiB per box.

### The candidates the pricing justified (job 589273, gates passed, worst cosine 0.9999999)

| candidate | fused | Δ | dr-only | Δ |
|---|---:|---:|---:|---:|
| baseline | 19.64 | — | 10.65 | — |
| fold owner = last-arriving split | 19.64 | 0.00 | 10.45 | −0.20 |
| readiness poll issued one stage ahead | 19.53 | −0.11 | 10.54 | −0.11 |
| a tile's partials contiguous | 19.41 | −0.23 | 10.19 | −0.46 |
| all three | 19.41 | −0.23 | 10.36 | −0.29 |

Best is −0.23 us/layer = 0.041 ms over 180 launches, under the 0.10 ms bar,
and the three do not compose. **A link's cost is the work it does, not the
order it does it in.** Deleting the join wait bought 0.96 us; moving the wait
to the split that arrives last bought nothing, so it is not a straggler wait
but the counter observe itself (`[atom.lat.dev.hop]` ~650 ns, which every
owner pays). Deleting the fold reads bought 1.10 us; making them contiguous
recovered a fifth of that, the locality share, leaving the reads themselves.
Deleting the readiness poll bought 0.54 us; hiding it behind an in-flight TMA
recovered a fifth, because a gpu-scope acquire load is longer than one TMA.

### Neither bytes nor transactions are the pole (job 589331)

A cross-lane hypothesis held that DownResidual's time is its TMA issue column:
25 transactions at `[tma.issue.warp]` 248 ns is 6.2 us, most of the 10.5, and
the sm90 short-K GEMM lane had just measured that this cost applies per SM and
does not parallelize across producer warps.

A bound issuing one of the two activation boxes per stage — 8 of 25
transactions and half the activation bytes removed — measured **−0.20 us
fused, −0.45 us dr-only** against a predicted 1.98 us if the column were
exposed. So ~56 ns per transaction is exposed here, not 248: two producer
warps issuing into a ring deep enough to cover the math hide most of it. The
hypothesis is refuted for this kernel, alongside the byte hypothesis the
preceding note refuted (a whole 8.4 MB stream freed = 1.02 us).

## Alternatives considered

- **Split-K 4 -> 2**, the candidate the 2026-09-02 probes note named first.
  Not built: with 32 output tiles and 128 workers, S=4 is what places one
  subtask per CTA. S=2 halves the joins and the partial traffic (~1.4 us of
  the priced links) but doubles each CTA's serial K span from 8 stages to 16,
  against a mainloop that the bounds leave at ~7 us. The trade is
  unfavourable by inspection and no job was spent on it.
- **Atomic (`red.global.add.f32`) partial accumulation**, now that the goal
  permits reduction-order differences. Rejected on the same evidence: it
  attacks the fold reads, whose whole removal is 1.37 us and whose
  recoverable share measured 0.23.
- **Coarser readiness granularity** (COUNTER_K 128 -> 512, 2 polls per split
  instead of 8): attacks a 0.54 us link and delays DownResidual's start.

## Consequences

- DownResidual's 10.5 us is diffuse: the four largest links are 0.5-1.4 us
  each, they do not share a mechanism, and only their removal — not their
  reordering — pays. Incremental chain surgery on this kernel is closed.
- **The named successor is a 64-wide DownResidual output tile.** It is the
  only change that attacks several links at once: 32 output tiles become 16,
  which halves the joins and the fold traffic; and the wgmma moves off the
  N=32 atom, which `[wgmma.issue.wg.ss]` measures at 24.7 cycles against an
  architectural 15.3 (62% of peak) where m64n64k16 runs at 93% — the same
  1.49x GatedUp already took. It is a task-table and split-K redesign, so it
  needs its own lane and its own contract; the compounding is what makes it
  worth one, since no single component of it clears the bar alone.
- The bound technique generalizes and is cheap: one job, one node, one build
  per link, numerics invalid and stated as such. It has now retired the byte
  hypothesis, the transaction hypothesis and four chain links for the price
  of three jobs. Bound a link before building against it.

## Verification

Jobs 589228 (chain decomposition, 5 builds), 589273 (3 candidates + the
combination, each with `ffn_taskloop_parity --modes gu,dr,full`, all PASS at
worst cosine 0.9999999), 589331 (transaction bound). All on ACD1-33, driver
610.43.02, CUDA 13.1, torch 2.13.0+cu130, clocks not lockable so the 6% noise
floor applies; every table is one job on one node, so no cross-job
normalization is involved. No route or e2e job was run: the best candidate is
0.041 ms against a 0.10 ms bar, and e2e time would not change that. The
candidate flags are removed from the source per the rejected-flag precedent;
the patch and the run artifacts stay in the task workspace.
