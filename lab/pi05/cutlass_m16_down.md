# Fixed CUTLASS M16 expert FFN down screen

Prepared on base `80818f9`, including the vision-022 revert, with the accepted
021 deployment. Only FFN down is under test. The existing cfg9 type, output
projection and shipped routes remain unchanged. The candidate type and lab ABI
share the same `libcutlass_backbone.so` as the control; no second CUTLASS library
or vendor changes are introduced.

## Single-variable hypothesis

Change CTA 32x64x32 / warp 32x32x32 to CTA 16x64x32 / warp 16x32x32, retaining
eight stages, BF16 operands, the same RoundedGatedResidual operator, and
Stream-K scheduling. The candidate uses the same Target-local destination
iterator advance that fixes the vendor broadcast reduction. The M16 tile still
has two accumulator fragments, so that correction remains required.

This exact geometry is absent from the prior twelve-config expert screen.
The prior cfg7 was 16x128x64 / warp16x64x64 / four stages, a different candidate.
The rejected Triton down used a different pipeline and no split-K; it does not
measure this CUTLASS geometry.

Both CTA shapes pad M50 to 64, so no M-tail work disappears. Source derivation
predicts 64 threads, 48 to 40 KiB mainloop shared memory, and accumulator storage
32 to 16 FP32 values per thread. This does not establish a final register count.
Both shared-memory footprints still limit residency to two CTAs per SM.

Conditioned on actual occupancy remaining two on 170 SMs, the vendor scheduler
cost formula predicts output tiles 32 to 64, SK producer blocks 160 to 170, and
accumulator fragments 4 to 2. Both have 128 reduction blocks, one SK wave, zero
DP blocks and a 298-block launch grid. Workspace capacity is predicted to fall
from 2,622,080 to 1,393,408 bytes; capacity is not measured traffic or latency saved.
These derived values need checking against the compiled and launched kernel.

The smaller accumulator/partial payload may help. Conversely each producer's
K32 iteration count rises from about 25/26 to 48/49 while its MMA M halves.
B size per iteration is unchanged, and four logical M groups request each
weight instead of two. The 18-layer 144 MiB weight set exceeds 96 MiB L2.
There is no measured attribution of Stream-K internal overhead to establish
how much time is recoverable. The current 019 down duration is 8.481528 us/call
across 180 calls; the new shape requires a local comparison, not an estimate
from workspace capacity.

## Validation sequence

Compile the one existing native library with its unchanged flags. Inspect
ptxas resources and SASS to ensure BF16 roundtrip plus independent RN FP32
multiply/add remain. Both types reuse the same output operator.

Run `--phase coverage` before model loading. It checks all 50 rows for the
same three constants used to diagnose the original reduction issue: residual
only, all-ones product, and varying channel gate. Their GEMM results are
analytically 0 or 4096. Every per-row mismatch count and numerical error is saved;
a failure is raised immediately.

After constants pass, one warmed launch per type is profiled solely to obtain
actual grid, registers and shared memory. The trace durations are not timing
evidence. The JSON also records the native workspace queries. No NCU sweep or
extra measurement series is required for this metadata check.

Only then run `--phase actual` with the converted belt-cup checkpoint, seed 42.
The probe records 180 calls at the invocation boundary, cloning inputs, gates,
old residual and actual output while retaining 18 distinct weights. Both routes
use those same addresses and one shared C=D output. It saves every control and
candidate comparison against the actual output, using existing shallow
tolerances. Calls 0/90/179 also extract the candidate BF16 projection using
gate=1/C=0; the existing native separate residual operation must reproduce the
fused result exactly. A numerical failure stops before timing.

The only timing is one reset-inclusive ABBA, 15 raw samples per leg over all
180 calls. Every timed call copies the saved residual to the shared output
before its GEMM, identically for A and B. All 60 samples, mean gain, both drifts
and conservative separation are retained. Gain at or below observed drift,
or a negative gain, ends this fixed-tile experiment. No extra tile or
output-projection experiment follows.

## Commands after an exclusive GPU/JIT slot is granted

The loader compiles the updated native source into its existing cache path.
Capture its compiler output for ptxas evidence, then inspect the same library's
SASS. For coverage and the small resource trace:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass PYTHONPATH=$PWD/src:$PWD \
  /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_m16_down \
  --phase coverage --output /tmp/cutlass-m16-down-coverage.json
```

After coverage/resources are checked:

```sh
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass PYTHONPATH=$PWD/src:$PWD \
  /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_m16_down \
  --phase actual --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /tmp/cutlass-m16-down-local.json
```

## Measured result: rejected

Both native types compiled successfully using the unchanged flags. The M16
candidate uses 80 registers, zero stack/spills and 40 KiB shared memory; control
uses 120 registers and 48 KiB. The metadata trace confirms both grids are 298x1x1,
both blocks have 64 threads, and both remain limited by shared memory to two
resident CTAs/SM. Queried workspace is 2,622,080 bytes for control and 1,393,408 for
candidate, matching the earlier conditional calculation.

SASS in the candidate includes BF16 packs at 3b10/3b20, FP32 bit expansions,
FMUL from 3b80, independent FADD from 3c10 and final BF16 packs at 3d50-3d80.
[ptxas/SASS evidence](../../results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-cutlass-m16-resources.txt)
preserves the relevant excerpt. No new arithmetic expression or epilogue
rounding change was needed.

All three constant cases match exactly on every one of 50 rows.
[Coverage and actual launch metadata](../../results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-cutlass-m16-coverage.json)
retain per-row mismatch counts, native workspace sizes and both kernel records.

All 180 actual calls pass the existing shallow thresholds. Control matches
every captured output exactly. Candidate matches 10 calls exactly; worst
max_abs=0.25, rel_rms=0.0001540602504, and minimum cosine=0.999999988138.
All three same-mainloop decomposition checks (calls 0/90/179) are exact.

| Leg | Median reset-inclusive us/call |
| --- | ---: |
| A1: existing cfg9 | 10.569600 |
| B1: fixed M16 | 13.077156 |
| B2: fixed M16 | 13.084445 |
| A2: existing cfg9 | 10.581867 |

The candidate is slower by 2.505067 us/call using mean leg medians.
A/B drift is 0.012267/0.007289 us, much smaller than this separation.
The [raw actual results](../../results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-cutlass-m16-rejected.json)
contain 180 comparisons, three decomposition checks and all 60 samples, including
the first sample of each leg.

This fixed M16 Stream-K candidate is rejected. The measurements establish that
lower registers/shared/workspace did not improve this local sequence; they do
not isolate the cause of the slowdown. No further tile, profiling, or deployed
measurement followed. The GPU/NVCC/JIT/model window was released after process
exit.

The 85-line native delta was reverted and its experiment binary removed.
[cutlass_m16_down.patch](cutlass_m16_down.patch) retains the exact experiment
against the recorded base. Apply it in an isolated checkout before running
either probe phase, so control and candidate remain in the one updated
Target library. Production native code, routes and out-projection remain
unchanged. Full compiler/SASS, coverage trace and local log artifacts remain
under the main checkout's ignored artifacts directory with prefix
`gpt6-m16-down-`.
