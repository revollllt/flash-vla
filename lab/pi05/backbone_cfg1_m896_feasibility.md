# Backbone cfg1 at M896: stop after CPU screening

Do not schedule a new cfg1 experiment on the present evidence. The proposed
CTA128x64x64 / warp64x32x64 / four-stage kernel is the existing cfg1; it lost to
cfg0 at M968. **M896 cfg1 has not been measured.** The smaller M does change
StreamK partitioning, but the source arithmetic below supplies no new
load-balancing advantage sufficient to prioritize this retry. This is a
bounded screening decision, not proof that cfg1 cannot win at M896.

## Existing experiment and provenance

- Probe: `lab/pi05/cutlass_gemm_screen.py`, original commit `9b6449e`.
  Raw: `results/pi05-rtx5090/gpt6-run-01/measurements/cutlass-screen.json`,
  committed in `8ea1cc2`; original remote log:
  `/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-cutlass-gemm-screen.log`.
- These are **Pi0.5 actual inputs/weights**, not Pi0 fixtures: the probe builds
  `rtx5090/pi05` with the reference plan and `sample_inputs(42)` (lines 186–188).
  Its actual backbone traversal records the first two FFN layers' normalized
  inputs and gate/up weights (lines 56–87), giving four distinct 64 MiB weights.
  The raw options explicitly name
  `/home/ubuntu/models/pi05_belt_cup_pytorch` and checkpoint ID/digest
  `kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha`.
  These match the identifiers in current `measurements/024.json`; the old
  record does not identify a different checkpoint.
- Only the lab native implementation/cache came from Pi0:
  `src/flash_vla/hardware/nvidia/rtx5090/pi0/backends/cuda/kernels/cutlass_gemm.cu`,
  cfg0/cfg1 at lines 86–87; its source revision in the probe ancestry is
  `76da6cc`. The raw library path is
  `/home/ubuntu/flash-vla/.cache/cuda_ext/rtx5090_pi0_cutlass_3b9ae13bbaa360d9/libcutlass_gemm.so`,
  arch `sm_120a`, PDL disabled. The JSON did **not** save the actual loaded
  checkout revision; the commits above identify retained source/evidence,
  not a reconstructed process revision.
- Shape is M968 K2048 N16384, BF16 A/B/output, FP32 accumulation, alpha=1,
  beta=0, linear epilogue. Each CUDA graph rotates all four real weights
  (256 MiB versus 96 MiB L2); 15 samples/config, reported per GEMM.
  Numeric checks cover cycle entries 0 and 3, not all layers; both configs
  pass the existing shallow criteria (worst relative RMS: cfg0 2.64e-5,
  cfg1 2.15e-5).

Within that same run, cfg0 median was **0.297296 ms**, cfg1 **0.311480 ms**:
cfg1 was **14.184 us / 4.77% slower**. This was a sequential screen, not a
cfg0/cfg1 ABBA. Use this relative result only; do not compare these old times
to current deployment timing or infer a current M896 regression from them.

## M896 source arithmetic, not measured dispatch

CUTLASS vendor revision: `cb4247394dd82148787aed73e5dc7cef33cbf862`.
Source: `third_party/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h`.
Lines 434–449 compute tiles, 303–395 choose DP/SK populations, 244–298 score
SK work, 487–507 partition K iterations, 509–595 handle reduction/cohorts/
semi-persistence, and 606–622 count blocks.

With 170 SMs, K/64=32 iterations and one CTA/SM (both require 96 KiB shared):

| CPU-derived M896 quantity | cfg0 | cfg1 |
| --- | ---: | ---: |
| Output tiles | 7 × 128 = 896 | 7 × 256 = 1792 |
| DP tiles / blocks | 680 | 1530 |
| SK tiles / blocks | 216 / 170 | 262 / 170 |
| SK iterations, normal / big block | 40 / 41 | 49 / 50 |
| Big SK blocks | 112 | 54 |
| Total launch blocks | 850 | 1700 |

Both have one SK wave, no separate reduction wave, no cohort raster
(tiled M=7 versus cohort M=8, lines 101 and 556–570), and no remapping.
Normalizing cfg1's half-width MMA work, the simple critical-work count is
`4*32+41=169` for cfg0 and `(9*32+50)/2=169` for cfg1.
This ignores memory, scheduling and fixup costs; it does not predict latency.

Halving N halves each CTA's MMA count and logical FP32 accumulator storage
(128 to 64 values/thread), but doubles output CTAs; total useful MMA work is
unchanged. Actual cfg1 total registers/spills are not established here.
Lower accumulator demand does not remove the shared-memory occupancy limit.

Per-stage A+B shared payload falls from 32 to 24 KiB, while stages rise from
3 to 4: both still allocate 96 KiB. The multistage prologue loads stages−1
(`gemm/threadblock/mma_multistage.h:360–370`), so nominal prologue payload
rises from 64 to 72 KiB. Each stage has half the N compute; a fourth stage
is not free latency hiding. Ignoring StreamK boundary duplication, nominal
CTA-request A traffic doubles **448 to 896 MiB**; B remains 448 MiB
(total 896 to 1344 MiB). Unique A is only 3.5 MiB and can benefit from L2:
these are request-volume counts, **not measured DRAM traffic**.

Current gate/up work is 60.1295 GFLOP. Using the parent-reported approximate
264 us and 2.86 GHz gives about 227.76 TF/s versus the
`512*170*2.86 GHz = 248.93 TF/s` primitive estimate, roughly 91.5%.
The frequency is not a complete per-kernel frequency record; this ratio is
only a prioritization guide, not the true attainable limit or a recoverable
time budget. No M896 cfg1 sample, resource compilation, GPU work, production
change, or new benchmark script was added for this decision.
