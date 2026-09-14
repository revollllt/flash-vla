# Pi0.5 expert CUTLASS screen

The 2026-09-15 run used deployment revision `7085a4b`, the belt-cup
`orbax-39999+openpi-convert-pi05_aloha` checkpoint, and fixture seed 42.
The probe records the actual imported plan and each layer's first deployed
`torch.mm` inputs. Both projections use contiguous BF16 and alpha=1, beta=0;
the expert down's gated residual update follows its BF16 GEMM.

Each CUDA graph cycles 18 distinct real weights, 288 MiB for packed FFN and
144 MiB for down, exceeding the 96 MiB L2. Times are CUDA events on the
capture stream, 15 samples per configuration, with one workspace per plan.
All 12 existing linear configurations passed the existing shallow numerical
requirements on the concatenated outputs of all 18 weights.

| Site / M,K,N | Torch before / after (us) | Config 9 (us) | Relative RMS / cosine |
|---|---:|---:|---:|
| Packed FFN / 50,1024,8192 | 13.7742 / 13.7867 | 13.3920 | 3.68e-5 / 0.9999999993 |
| Down / 50,4096,1024 | 10.1404 / 10.1369 | 9.5609 | 0.001676 / 0.999998596 |

Config 9 is tile 32x64x32, warp 32x32x32, 8 stages. Control median drift was
0.0124 us and 0.0036 us; the first sample of a leg was often 0.7-1.1 us higher.
The median differences imply approximately 0.17 ms across 180 calls of each
site. Keep config 9 as a small-gain candidate; complete-chain and deployed
measurements have not been run for it.

The initial full-runner probe failed on config 0 with status 719. The same
50x1024x8192 GEMM passed standalone. A two-library reproducer showed that
Pi0's existing library and Pi0.5's backbone library share the GNU-unique TLS
`GemmUniversalBase::device_ordinal_` address. Initializing the first library
makes the second skip `cudaFuncSetAttribute`, leaving its own kernel entry
without the required dynamic shared-memory opt-in. A diagnostic-only reset of
that TLS value to -1 before planning made the second library run successfully.

The probe therefore uses the deployed Pi0.5 native library for config 0;
configs 1-11 use the existing Pi0 lab interface. No production code changes
or runtime TLS resets were added. The native source was built against the
main vendored CUTLASS revision `cb4247394dd82148787aed73e5dc7cef33cbf862`.

Raw evidence remains in the deployment checkout's ignored
`artifacts/rtx5090-pi05/` directory:

- `gpt6-cutlass-expert-screen.json`: all samples, configurations, metrics, and imported plan.
- `gpt6-cutlass-expert-screen-first-failed.log`: the original failure.
- `gpt6-expert-cfg0-standalone.log`: the successful standalone shape.
- `gpt6-expert-dual-library.log` and `gpt6-expert-dual-static-reset.log`: the coexistence reproducer and causal check.

Run `python -m lab.pi05.cutlass_expert_screen --help` from the probe checkout
for the source-checkout, checkpoint options, seed, and output arguments.
