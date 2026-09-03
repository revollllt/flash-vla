# Agent Note: the GatedUp phase is bound by weight-ring depth, not by math, epilogue or bandwidth

Status: rejected — the one candidate built is under the promotion bar; the
accounting and the named successor are the durable result.

Date: 2026-09-03


> **Superseded in part** (job 591174, `2026-09-03-ffn-gu-copy-column.md`):
> this note's headline -- that in-flight weight frames are the pole -- reads a
> one-sided gradient as a pole. The phase has two floors stacked ~20% apart, a
> 9.41 us cold-DRAM delivery floor and a ~7.7 us ring-protocol floor, and the
> production column sits within 1.6% of the former. Ring depth 3->2 hurts
> because it lifts the protocol floor above the DRAM floor; deeper buys nothing
> because it is already below. The decomposition table below stands.

## Problem

The decoder FFN task loop's GatedUp phase measured 11.89 us/layer while its
math is ~1 us (0.84 GFLOP at `[wgmma.clock.sm]`) and freeing its entire
16.8 MB weight stream by construction bought only 1.37 us. Roughly 7-9
us/layer — about 1.3-1.6 ms over the decoder's 180 launches — had no
explanation, the largest unexplained pool left in the model.

## Decision

Reject the candidate, record the accounting, and name the successor.

### The phase, priced link by link

Each build deletes exactly one link and keeps every other TMA, barrier and
wgmma; numerics are invalid by construction, timing is valid. Deltas are
always within one job, because the per-job baseline drifts (11.56-12.45
across five jobs and two nodes). `txns` is TMA transactions per GatedUp CTA:
4 weight boxes at 32 KiB, 4 activation boxes at 32 KiB, 1 L2 prefetch.

| link removed | txns | gu-only | Δ | job |
|---|---:|---:|---:|---|
| baseline | 9 | 11.86 | — | 589369 |
| epilogue stores only | 9 | 11.85 | −0.01 | 589369 |
| whole gelu epilogue | 9 | 9.82 | **−2.04** | 589369 |
| release fence | 9 | 11.56 | −0.30 | 589369 |
| wgmma chain | 9 | 11.39 | −0.47 | 589369 |
| bias gmem loads (baseline 11.65) | 9 | 11.46 | −0.19 | 589430 |
| activation wait (baseline 11.65) | 9 | 11.39 | −0.26 | 589430 |
| math + epilogue, copies kept (baseline 11.56 / 11.59) | 9 | 9.60 / 9.68 | −1.96 / −1.91 | 589633 / 589900 |
| activation copy entirely (baseline 11.59) | 5 | 12.15 | **+0.56** | 589900 |
| weight ring depth 3 → 2 (baseline 12.45) | 9 | 13.97 | **+1.52** | 589934 |

### What the table says

**The copy pipeline and its waits are 9.6 of the 11.6 us — 83% of the
phase.** Everything the kernel computes, stores and publishes is 1.96 us,
reproduced in two jobs.

**That copy column is not bandwidth and not transactions.** Freeing the
weight DRAM bought 1.37 us (job 589178, the stream-continuity lane), and
removing four of the nine transactions along with the entire activation
broadcast made the phase 0.56 us *slower*. Neither bytes nor transaction
count is the pole.

**It is how many weight frames can be in flight.** The ring is 3 deep
against 4 trips, so the phase never reaches steady state: three frames are
issued at once, and the fourth cannot be issued until the math retires the
first. Removing one frame of depth costs 1.52 us, which prices the gradient
directly — the phase is paying roughly one extra frame latency it cannot
overlap.

**The epilogue's 2.04 us is arithmetic, not operands.** Its stores are free
(0.01) and its bias loads nearly so (0.19), leaving the gelu/product itself.

## Alternatives considered

- **Hoisting the bias loads** (the candidate that was built): the shipped
  epilogue re-reads the same 64 bias values from gmem once per output
  element, 32 scalar loads per thread, all after the last wgmma. Hoisting
  them to the top of the task as aligned 4-byte pair loads is numerically
  identical (parity PASS, worst cosine 0.9999999) and measured −0.37
  us/layer = 0.067 ms over 180 launches, under the 0.10 ms bar. Rejected on
  size, not on correctness; the paired bound showed the loads were never
  worth more than 0.19 us.
- **Whole-phase brackets** (delete the math warpgroup, or delete the
  producers): not constructible. Removing the frame waits leaves TMAs in
  flight when the slot seam reinitializes the barrier pool, which faults
  (jobs 589569, 589594). The same number is reachable safely by keeping
  every wait and deleting only the wgmma and the epilogue.
- **Widening the wgmma atom**, as the DownResidual chain note proposes for
  its own phase: does not apply here. GatedUp already issues
  `TiledMmaWide` = m64n64k16 and has taken that 1.49x.

## Consequences

- The bandwidth-side family stays closed for this phase, now for a second,
  independent reason: depth, not delivery, is what the copy column costs.
- `FFN_GU_*` flags exist only in the workspace patch, not in the shipped
  source. The kernel is unchanged by this lane.

## The successor, not built

Trade one stationary activation frame for one weight frame: activation ring
3 deep and rotating, weight ring 4 deep. Shared memory is exactly neutral
(4x32 + 3x32 KiB today, 3x32 + 4x32 KiB after), which matters because the
data plane is at 229376 B of a 232448 B cap with no headroom for a frame.
With all four weight frames in flight the phase should shed about the
1.52 us the depth gradient prices, ~0.27 ms over 180 launches, above the
bar. The cost is that the activation loader gains the empty/reuse protocol
it does not have today, mirroring the weight loader's, plus three barrier
words. It needs its own contract: a parity gate, then a same-job A/B/A on
two nodes.

## Verification

Seven GPU jobs (one over the lane's budget, spent on the depth gradient
because it named the pole): 589369, 589430, 589569 (fault), 589594 (fault),
589633, 589900, 589934. Every job's baseline row runs
`eval/correctness/pi05/ffn_taskloop_parity.py --modes gu,dr,full` and passed
at worst cosine 0.9999999, which is also the proof that the flags are
behavior-neutral when off; the two numerically valid variants (bias hoist,
depth 2) ran the same gate and passed. Workspace, ledger and the flag patch:
`artifacts/ktasks/gu-decomposition/`.
