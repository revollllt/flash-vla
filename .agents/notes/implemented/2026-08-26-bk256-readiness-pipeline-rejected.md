# BK256 per-slice readiness pipeline rejected

Date: 2026-08-26  
Branch: `perf-bk256-readiness`  
Base: `origin/main@017877c`

## Question

PR #6 showed that producing the complete 128 KiB scaled activation and then
executing a grid-wide synchronization costs too much. This experiment removes
that barrier and pipelines four statically indexed BK256 slices.

CTA 0..63 directly own one 32x32 `(M,K)` producer tile each. For every BK256
slice, eight K tiles times two M halves give exactly sixteen producers. Each
producer publishes one arrival after its generic global stores. Every
GatedProjection activation producer waits only for the next slice's fixed
sixteen arrivals before issuing its TMA load. The counters are dependency
signals, not a task queue or runtime scheduler.

The launch remains one cooperative 132-CTA persistent kernel. The host checks
that all CTAs can be resident together, so waiting consumers cannot prevent
the fixed producer CTAs from running. Scaled-input readiness uses counters
`[0,4)` and the existing DownResidual split join uses `[4,36)`.

## SM90 proxy ordering

Release/acquire on a readiness counter orders ordinary generic global memory,
but it is not by itself a cross-proxy fence for a following TMA read. The
experiment therefore places `fence.proxy.async.global` in the elected TMA lane
after its acquire wait and before the TMA issue. The same rule also applies to
the existing hidden generic-store to DownResidual TMA path; the experimental
source added the fence at both boundaries.

These fences are required for the experiment to be meaningful, but the
experimental source remains uncommitted because the performance result loses.

## Measurement

Job 552228 ran on ACD1-9 with CUDA 13.1, torch 2.11, CUDA graphs, three
cold-weight sets, and 30 samples. GPU clock locking failed, so the same-job
TileLang composition is retained as a control.

- GatedProjection parity cosine: 1.0; maximum absolute error: 6.10e-5
- Fused median: 27.88 us; minimum: 27.38 us
- GatedProjection median: 18.31 us
- DownResidual diagnostic: 14.18 us
- TileLang composition median: 23.73 us
- Watchdog/deadlock: none
- Kernel resources: 109 registers/thread, zero stack/local memory, 1 KiB
  static plus 196736 bytes dynamic shared memory

The production BK64/depth4 point is 25.25 us fused and 15.64 us
GatedProjection. The readiness candidate regresses by 2.63 us fused and
2.67 us GatedProjection. Relative to the rejected full-grid-sync candidate
(28.67 us fused, 18.58 us GatedProjection), per-slice readiness recovers only
0.79 us fused and 0.27 us GatedProjection. Its same-job
`fused - composition` gap is 4.15 us, versus about 2.41 us for production.

The job ran the requested GatedProjection parity gate only; DownResidual and
full-chain parity/replay were not used as acceptance evidence because the
candidate already failed the performance gate.

## Decision

Reject the BK256 global-scratch path, including both global synchronization
and per-slice readiness variants. Retain the production BK64/depth4/N64/wait1
kernel. Do not merge the experimental cooperative launch, workspace cache,
readiness counters, or BK256/readiness-specific proxy-fence edits through this
performance PR.

The existing production hidden generic-store to DownResidual TMA boundary is
an independent SM90 memory-model correctness issue. Its
`fence.proxy.async.global` fix must land separately from this rejected
performance candidate, with focused GatedProjection, DownResidual, full-chain,
replay, and performance evidence. Rejecting BK256 readiness does not waive
that required production fix.

Skip `column_cohort=2` for this dataflow: it depends on a useful BK256 prep
path and would additionally reduce GatedProjection compute CTAs from 128 to 64
and pipeline depth from 3 to 2. Resume optimization from the production BK64
path and use the offline planner to select the next independently attributable
transport or scheduling experiment.
