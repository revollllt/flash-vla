# Standalone RMS-XFS plus BK256 is not a reproducible FFN winner

Date: 2026-08-26

Branch: `perf-rms-xfs-kmajor`

Base: `origin/main@cda270b`

## Question

The existing decoder path launches a factor-only RMS kernel, then lets the
BK64 persistent FFN apply the factor and per-channel scale. The BK256 upper
bound showed that a pre-arranged K-major `[1024,64]` activation can reduce the
persistent FFN from 25.25 us to about 23.6--23.8 us. This experiment asked
whether an upstream TileLang producer can materialize exact BF16 XFS cheaply
enough to realize that gain:

```text
xfs_kmajor[k,m] = bf16(bf16(X[m,k] * rstd(X[m,:])) * scale[k])
```

## Producer diagnosis

The first implementation wrote K-major output directly from an M-major
fragment. Generated CUDA confirmed scalar BF16 stores whose neighboring warp
lanes were 64 elements apart. The best direct version was 4.060 us (job
553602).

Staging `[OUTPUT_K, BLOCK_M]` through shared memory and using a vectorized
global copy fixed the store mapping. A 256-CTA implementation with
`BLOCK_M=8`, `OUTPUT_K=32`, `BLOCK_K=256`, and 128 threads measured 2.416 us
versus a same-job factor-only control of 1.900 us (job 553627). It matched the
exact-rounding reference with cosine 1.0 and maximum absolute error 0.

That implementation recomputes row factors per output-K tile and expands X
reads to roughly 4 MiB. A cold-traffic alternative computed each row factor
once and exposed 16 CTAs. Its best standalone point was 3.080 us at
`BLOCK_M=4`, `OUTPUT_K=128`, and 256 threads (job 553748). An explicit shared
pair swizzle was rejected at 2.624 us because it regressed and TileLang
reported a possible data race (job 553732).

## End-to-end measurements

The consumer used one N64 WGMMA atom, BK256/depth3, one 32 KiB activation TMA,
and one 32 KiB packed-weight TMA per stage. DownResidual remained BK64.

An early dedicated A/B harness produced one anomalously low 25.685 us result
(job 553749). It did not reproduce in either complete harness run and is not
accepted as evidence.

| Candidate | Job | Full | TileLang control | Result |
| --- | ---: | ---: | ---: | --- |
| 256-CTA recompute producer + BK256 | 553647 | 27.243 us | 23.627 us | reject |
| Same path with proxy fence/residency guard | 553879 | 27.259 us | 23.547 us | reject |
| Single-factor cold-traffic producer + BK256 | 553883 | 28.117 us | 23.627 us | reject |

The previous complete path is approximately 27.15 us: factor-only RMS around
1.90 us plus the production BK64 persistent FFN at 25.25 us. The best
reproducible RMS-XFS + BK256 point is therefore about 0.1 us slower, not a
winner. GPU clocks could not be locked, so a sub-microsecond cross-job claim
would be unsafe in either direction; acceptance requires a clear margin.

## Correctness and review

Independent PR review found two required production guards: a
`fence.proxy.async.global` between generic hidden stores and DownResidual TMA,
and a fail-fast 132-CTA residency check. Both were implemented on the measured
head. Job 553879 passed GU, DR, full-chain, and two replay iterations with
worst cosine 0.9999999. These fixes do not remain in this rejected performance
PR; the proxy fence remains on its separate correctness branch pending its own
PR.

## Decision

Reject the standalone RMS-XFS producer and BK256 consumer integration. Revert
all implementation and harness changes from this branch and merge only this
decision record.

The useful result is narrower: shared staging eliminates the original scalar,
warp-strided K-major stores, but the separate producer launch and RMS/data
movement consume the BK256 kernel-side gain. The next experiment should fuse
exact XFS production into the upstream residual-producing operation, so it
replaces existing work instead of adding a launch. Do not put RMS reduction
inside the persistent FFN and do not revive grid synchronization or a runtime
readiness scheduler.
