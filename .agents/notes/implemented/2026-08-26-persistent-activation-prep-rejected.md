# Persistent BK256 activation preparation: global-sync path rejected

Date: 2026-08-26  
Branch: `perf-persistent-activation-prep`  
Base: `origin/main@2f2880f`

## Question

Can the pre-scaled `[K, M_PAD]` BK256 upper bound be made end-to-end by
producing its 128 KiB activation buffer inside the same 132-CTA persistent
kernel, without a second launch, runtime scheduler, or atomic work queue?

The experiment retains one N=64 WGMMA atom and the BK256/depth3 dataflow from
PR #5. It uses a cooperative 132-CTA launch so every H100 SM owns exactly one
CTA. CTA 0..63 statically cover the complete 2x32 grid of 32x32 `(M,K)` prep
tiles; all 132 CTAs reach one grid synchronization before CTA 0..127 consume
the result. Sentinel CTA 128..131 return only after that synchronization.

The workspace is explicitly partitioned into a 128 KiB BF16 scaled-input
region followed by the existing 768 KiB FP32 DownResidual partial region.
Their overlapping lifetimes never alias.

## Three-part measurement

All jobs ran on ACD1-9 with CUDA graphs, three cold-weight sets, 30 samples,
and the existing minimum parity gate.

| Mode | Job | GU (us) | DR (us) | Fused (us) | TileLang composition (us) | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6A: cooperative launch, no grid sync, input pre-scaled before capture | 552178 | 15.49 | 12.95 | 25.32 | 23.74 | 1.0 |
| 6B: cooperative launch plus no-op grid sync | 552179 | 14.90 | 13.36 | 24.87 | 22.87 | 1.0 |
| 6C: real 64-CTA prep plus grid sync | 552180 | 18.58 | 14.86 | 28.67 | 23.66 | 1.0 |

The production BK64/depth4 result is 15.64 us GU and 25.25 us fused. The
pre-scaled BK256/depth3 upper bound from job 552091 is 14.28 us GU and
23.80 us fused. The real in-kernel producer adds 3.80 us fused and 3.68 us GU
relative to the empty-sync mode, far beyond the complete 1.45 us budget
between the upper bound and production.

GPU clock locking failed in all three jobs and the TileLang control drifted by
0.87 us, so the cross-job raw delta is conservative rather than causal. Using
the within-job `fused - composition` gap, the empty grid sync adds about
0.42 us over cooperative launch and real prep adds about 3.01 us over the
empty-sync mode. This agrees with the separately measured 3.13 us standalone
producer cost and still exceeds the budget by roughly 2x.

All three persistent variants compile to 107 registers/thread with zero stack
and zero local memory. The binary reports 1 KiB static shared memory and the
launch requests 196736 bytes dynamic shared memory, preserving one CTA per SM.
The performance loss is therefore the phase-0 producer and global
synchronization/data-visibility critical path, not spilling or a residency
change. The mode-2 DR-only timing includes unnecessary activation prep and is
diagnostic rather than the latency of the DownResidual body itself.

## Decision

Reject full-buffer preparation followed by a grid-wide barrier. Do not commit
the experimental kernel or its setup-only pre-scale cache. Retain the PR #3
BK64/depth4 kernel as production.

The next admissible experiment is a statically indexed per-BK readiness
pipeline: prep CTAs publish individual BK256 slices and their fixed consumer
CTAs wait only for those slices, allowing preparation to overlap GatedUp. It
must remain a single cooperative persistent launch with direct `blockIdx.x`
ownership; readiness flags are dependency signals, not a task scheduler or
work queue. This is a separate PR so the effect is attributable.

`column_cohort=2` remains conditional. It should not be combined with the
readiness pipeline because it reduces GatedUp compute CTAs from 128 to 64 and
forces BK256 pipeline depth from 3 to 2.
