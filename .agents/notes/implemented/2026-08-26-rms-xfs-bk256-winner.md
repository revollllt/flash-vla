# RMS-XFS producer unlocks the BK256 GatedProjection path

Date: 2026-08-26

Branch: `perf-rms-xfs-kmajor`

Base: `origin/main@cda270b`

## Contract

The producer replaces the decoder's factor-only RMS launch and writes the
exact BF16 tensor consumed by GatedProjection:

```text
xfs_kmajor[k,m] = bf16(bf16(X[m,k] * rstd(X[m,:])) * scale[k])
shape = [K=1024, M_PAD=64]
```

The K-major output makes M the contiguous 128-byte TMA row. A BK256
GatedProjection stage therefore loads its complete activation tile with one
32 KiB TMA instead of four row-major boxes.

## Store-layout diagnosis

The first TileLang implementation computed the right values but wrote them
directly from an M-major fragment. Generated CUDA confirmed scalar BF16 stores
whose neighboring warp lanes were 64 elements apart. The best direct version
was 4.060 us (job 553602).

The accepted implementation stages `[OUTPUT_K, BLOCK_M]` through shared
memory and uses TileLang's vectorized copy for the final K-major write. Its
generated CUDA emits coalesced vector stores. `BLOCK_M=8`, `OUTPUT_K=32`,
`BLOCK_K=256`, and 128 threads measured 2.416 us versus a same-job factor-only
control of 1.900 us (job 553627). Output matched the exact-rounding reference
with cosine 1.0 and maximum absolute error 0.

The 2D grid intentionally recomputes eight-row RMS factors for every output-K
tile. That expands the X reads to roughly 4 MiB but exposes 256 independent
CTAs. A single-factor-per-row alternative reduced X traffic but exposed only
16 useful CTAs at its best configuration; it measured 3.080 us at
`BLOCK_M=4`, `OUTPUT_K=128`, and 256 threads (job 553748). The full-grid
version remains the measured end-to-end winner.

An explicit shared-memory pair swizzle was rejected: it regressed from 2.430
to 2.624 us and TileLang's checker reported a possible data race (job 553732).

## End-to-end result

The persistent consumer keeps one N64 WGMMA atom and uses BK256/depth3 with
one 32 KiB activation TMA plus one 32 KiB packed-weight TMA per stage. The
DownResidual BK64 path and its scheduling remain unchanged.

Job 553749 measured the producer and consumer in one CUDA graph with three
cold-weight sets:

| Path | Full | GatedProjection only |
| --- | ---: | ---: |
| Pre-scaled BK256 upper bound | 23.637 us | 14.155 us |
| RMS-XFS producer + BK256 | 25.685 us | 17.355 us |

The new complete path replaces both the existing factor-only launch
(approximately 1.90 us) and the production BK64 persistent FFN (25.25 us).
Its 25.685 us result is approximately 1.47 us faster than that 27.15 us chain.
GatedProjection parity passed with cosine 1.0 in job 553647.

## Decision

Accept the shared-transpose RMS-XFS producer and the K-major BK256
GatedProjection consumer. Keep the producer as the upstream normalization
boundary rather than adding RMS reduction work to the persistent FFN. This
preserves direct `blockIdx` ownership in the 132-CTA persistent kernel and
removes the rejected in-kernel readiness scheduler and grid synchronization.

The producer is still a separate upstream launch. A later fusion with the
preceding residual-producing operation may remove its launch floor, but is not
required for this acceptance point and must be measured as a separate PR.
