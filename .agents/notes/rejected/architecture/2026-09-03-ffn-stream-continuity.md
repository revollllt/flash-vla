# Agent Note: The FFN task loop is dependency-latency bound, not weight-stream bound; continuity is rejected

Status: rejected — measured (job 589178): freeing an entire weight DRAM stream
is worth 1.0-1.4 us per layer, so the continuity lever it was supposed to fund
cannot reach the promotion bar. The finding generalizes: this kernel's weight
streams are ~80% hidden, which closes the warm/prefetch/retile/continuity
family for it.

Date: 2026-09-03

## Problem

`2026-09-02-ffn-gu-dram-ceiling.md` left one untested lever: fold the
DownResidual weight stream onto the tail of the GatedUp stream so a layer
issues one continuous 25.2 MB stream instead of a 16.8 MB burst, a drain and
an 8.4 MB burst. Priced from the cold-burst curve
(`[tma.bw.dev.burst]`), two bursts cost 9.3 + 5.7 us against ~12.9 us for one
stream, i.e. ~2 us per layer, ~0.36 ms over the decoder's 180 launches.

That estimate assumes the weight streams are on the kernel's critical path.
This note tests that assumption instead of the implementation.

## Decision

Reject the continuity lever, and with it the premise shared by the two
preceding FFN notes.

Two compile-time bounds pinned every weight stage to its task's first K tile,
so all but the first read hit L2 while transaction count, ring handshake,
barrier accounting and the WGMMA dependency chain stayed identical. Numerics
are invalid by construction; the bounds price "what if this stream were
free". Measured with the acceptance bench (CUDA graph, three 25 MB cold
weight sets, CUDA events, 50 reps), all rows in one job on one node:

| variant | fused | gu-only | dr-only | fused delta |
|---|---:|---:|---:|---:|
| baseline (gate passed, worst cosine 0.9999999) | 19.65 | 11.89 | 10.50 | — |
| DownResidual weight DRAM stream free | 18.63 | 11.80 | 10.25 | −1.02 |
| GatedUp weight DRAM stream free | 18.28 | 10.40 | 10.35 | −1.37 |

- Removing 7/8 of the 8.4 MB DownResidual stream buys **1.02 us per layer**
  (0.18 ms over 180 launches). That is the ceiling for *any* change to that
  stream. Continuity removes only its burst ramp, a fraction of the whole,
  so it cannot reach the 0.10 ms bar.
- Removing 12.6 MB of the 16.8 MB GatedUp stream buys **1.37 us per layer**,
  where the burst curve prices those bytes at ~7 us. Roughly 80% of the
  weight traffic is already hidden behind the kernel's math and dependency
  structure.
- Both streams free is 2.39 us per layer, 0.43 ms — the ceiling on every
  bandwidth-side idea for this kernel combined.

The kernel is therefore bound by its dependency chain and phase structure,
not by weight delivery. This is the same conclusion as hypothesis (b) in the
GatedUp warmth note's verdict, now measured rather than conjectured, and it
explains that null: warming a stream that was never exposed cannot pay.

## Alternatives considered

- **Build continuity anyway and measure e2e.** Rejected: it is not
  constructible in the current kernel without a layout change (below), and
  the bound already caps its prize below the bar.
- **De-alias the two data planes so the DownResidual weight ring can fill
  during GatedUp.** `SharedStorage.mainloop` is a union of the GatedUp data
  plane (229376 B) and the DownResidual one (98304 B); GatedUp alone leaves
  2936 B of the 232448 B opt-in budget, less than one 8192 B DownResidual
  weight frame, and a de-aliased layout needs 327816 B. One layout does fit —
  `GATED_UP_WEIGHT_DEPTH` 3 to 2 frees exactly the 32768 B a 4-deep
  DownResidual weight ring needs, landing at today's footprint, and the burst
  probe records ring depth 2-6 as a bandwidth null — but it was left unbuilt
  once the bounds priced the prize.
- **Cross-launch continuity across layers.** Same ceiling: it addresses the
  same weight streams, which the bounds show are ~80% hidden.

## Consequences

- The weight-stream family is closed for this kernel: warming
  (`2026-09-02-ffn-gu-dram-ceiling`), prefetch
  (`2026-09-02-ffn-taskloop-inflight-probes`), retiling and continuity all
  divide the same ~2.4 us per layer, most of which is already overlapped.
- Remaining decoder FFN headroom is in the dependency chain and phase
  structure — the DownResidual counter round trip, split-K join, fold and
  residual read-modify-write — not in delivery. `dr-only` is 10.50 us for
  0.54 GFLOP and 8.4 MB, and only 0.25 us of that is its weight stream.
- `[tma.bw.dev.burst]`'s rule applies to a phase whose copy is *exposed*.
  Before dividing a phase's bytes by that curve, bound the stream's exposed
  share; a kernel that overlaps its copy is priced by its dependency chain
  instead.
- The bound flags are removed from the source per the rejected-flag
  precedent; the patch and the run artifacts stay in the task workspace.

## Verification

Job 589178 (ACD1-11, driver 610.43.02, CUDA 13.1, torch 2.13.0+cu130, clocks
not lockable so the 6% noise floor applies). One job carries all three rows,
so the comparison needs no cross-job normalization. The flagless baseline in
the same job passed the shipped kernel gate
(`ffn_taskloop_parity.py --modes gu,dr,full`, worst cosine 0.9999999,
replay-stable), which also shows the source edit was behavior-neutral with
the flags off. No e2e or route job was run: the bound closed the question
below the bar, and profiler or e2e time would not change it.
