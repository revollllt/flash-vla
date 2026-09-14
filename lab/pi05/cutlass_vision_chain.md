# Pi0.5 vision cfg10 full-chain check

The candidate adds one 64x128x32 Stream-K tile (warp32x64x32, 5 stages) to the
existing Pi0.5 CUTLASS library. Both FFN projections use an FP32 accumulator
plus broadcast BF16 bias, then store BF16. Up reuses `fused_vision` norm and
GELU. Down retains a BF16 projection buffer and the separate residual add.
The production delta is commit `13338d3`, based on `df996b8`.

The probe records all 27 actual eager layer calls from the deployed vision
path, preserving each normalized-chain input and initial residual. It checks
the entire replacement callsite output against the captured current output.
Both sites must pass all per-layer existing shallow numerical checks and
bit-identical repeated CUDA Graph replay before either is timed. Scratch is
warmed and frozen before A/B/B/A timing. Every leg retains 15 CUDA-event
samples. Down's identical residual restore is outside both timed paths.

The run used source revision `2499f52`, the belt-cup
`orbax-39999+openpi-convert-pi05_aloha` checkpoint and seed 42. All 27 layers
passed at each site. Worst rel_rms/minimum cosine were 7.28506e-5/0.99999999735
for up and 0.00306710/0.99999529643 for down. Both replay checks were identical.

| Site | A1 total ms | B1 total ms | B2 total ms | A2 total ms |
|---|---:|---:|---:|---:|
| Up: norm + GEMM + GELU | 1.398624 | 1.298080 | 1.297056 | 1.425248 |
| Down: GEMM + residual | 1.394720 | 1.116160 | 1.112064 | 1.384448 |

Each total covers 27 calls. Up control drift is 0.026624 ms and down drift
0.010272 ms. Comparing the faster A with the slower B at each site gives
0.100544 + 0.268288 = 0.368832 ms of local savings. This supports deployed
validation; it is not a measured E2E reduction.

The shared library compiled successfully: cfg10 uses 160 registers with no
spills. CPU declaration checks passed 8/8 for up only, down only and both
routes, without CUDA initialization or a configured compiler at declaration.
No production routing was changed in this worker branch.

Raw evidence is retained in the deployment checkout's ignored
`artifacts/rtx5090-pi05/` directory:

- `gpt6-vision-cfg10-chain.json`: actual base plan, every layer's metrics and raw ABBA samples.
- `gpt6-vision-cfg10-chain.log`: run output.
- `gpt6-vision-cfg10-build.log`: native compiler resources and CPU-only library load.

The probe depends on `lab/pi05/cutlass_gemm_screen.py` from commit `9b6449e`
(or its cherry-pick `76a5f2a`). That one-file dependency supplies the existing
`samples_ms` graph timer. Commit `2499f52` adds the full-chain probe. Use
`python -m lab.pi05.cutlass_vision_chain --help` for checkpoint and output
arguments. Execute from a checkout containing the candidate; the probe does
not redirect imports to another deployment tree.
