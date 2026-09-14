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

## Reusing cfg10 for attention out-projection

The out-projection candidate starts from `7b2d9d0` and changes only the Python
wrapper: the existing residual closure derives K from the weight and serves
both FFN down and attention out-projection. The latter has contiguous BF16
x(3,256,1152), weight(1152,1152), bias(1152), and res=out(3,256,1152).
It preserves BF16(accumulator + bias), followed by the existing BF16 residual
add. No native kernel, tile, ABI, or deployment route changes in this candidate.
Both residual sites can share the existing 1,769,472-byte projection scratch.

In the 010 overview trace, 27 unchanged out-projection GEMMs total 476.274 us;
their residual adds total 40.443 us. The complete site is 516.717 us and two
kernels per call. The candidate retains the add, so its local improvement is
a GEMM implementation result, not a removed launch.

The `--site out` probe records only the 27 actual out-projection layers and
preserves res=out when cloning inputs. The run used the same belt-cup
checkpoint and seed 42. Every layer passes existing shallow tolerances:
worst relative RMS 0.001054180, minimum cosine 0.999999444, maximum absolute
error 1.0. Outputs are not bitwise equal to cuBLAS. Repeated candidate graph
replays from the same saved inputs are identical.

| Route | Total ms / 27 calls | Median us / call |
|---|---:|---:|
| A1 | 0.587776 | 21.76948 |
| B1 | 0.454656 | 16.83911 |
| B2 | 0.454656 | 16.83911 |
| A2 | 0.587776 | 21.76948 |

Each leg retains 15 samples. The same residual reset precedes each timed graph
on its capture stream and is excluded from both routes. Control median drift
is zero; control median absolute deviations are 0.07467 and 0.15170 us/call.
The 4.93037 us/call difference (0.133120 ms over 27 calls) exceeds those noise
scales and supports an end-to-end candidate. The 27 weights total 68.34375 MiB,
below the GPU's 96 MiB L2: isolated reuse can differ from the interleaved
deployed model, so this is not a measured deployment gain.

`gpt6-vision-outproj-local.json` in the artifact directory above records the
source revision, plan, all 27 layer errors and 60 raw timing samples;
`gpt6-vision-outproj-local.log` preserves stdout. Syntax and diff checks pass.
The current native ABI includes expert K parameterization; the run used the
matching source and native library in the candidate worktree.
