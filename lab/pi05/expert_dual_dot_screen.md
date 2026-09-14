# Pi0.5 expert dual-dot suffix screen

Hypothesis: two FP32 dot accumulators in one small Triton kernel can replace
the packed BF16 GEMM and gated-activation kernel, eliminating the intermediate
projection write/read and one launch. Prepare and its BF16 normalized input
remain the deployed implementation. No production route is changed.

## Current cost and scope

The focused 010 expert trace attributes the 180 FFN calls as follows:

| Stage | Total ms | Mean us/call |
|---|---:|---:|
| Prepare / AdaRMS | 0.260006 | 1.44448 |
| Packed GEMM | 2.162485 | 12.01381 |
| Bias + GELU + product | 0.417994 | 2.32219 |

The last two stages total 14.33599 us/call in that instrumented trace. The
GEMM launch has 128 CTAs (grid8x16), 128 threads, 96 registers/thread and
48 KiB shared memory; 128 CTAs are fewer than the GPU's 170 SMs. The source
trace is `results/pi05-rtx5090/gpt6-run-01/profiles/010-expert/0_shipped_action_expert.json`.

Each call reads 16 MiB of packed weights. The 50x8192 BF16 intermediate costs
819,200 bytes written and then read; removing both saves 1,638,400 logical
bytes/call, or 281.25 MiB across 180 calls. The existing GEMM's weight-only
rate is already about 1.396 TB/s. The activation's 0.418 ms includes math and
final stores that still must execute, so it is not wholly recoverable. These
observations suggest a gain measured in tenths of a millisecond; they do not
establish an achievable speedup. Timings must compare the same local paths.

## Why use the short Triton path

Vendor `examples/45_dual_gemm` supplies a dual mainloop and optional omission
of D0/D1 stores. Its `DualEpilogue::apply_output_operator_`, however, converts
each branch into the BF16 OutputAccessType before the two-argument combined
operator. Its ordinary bias path computes bf16(acc+bias), whereas this expert
requires float(bf16(acc))+bias, with both FP32 bias results retained through
GELU/product. Preserving that order needs an internal epilogue-interface
change, not just replacing the example's SiLU functor. The example also has
no existing Stream-K path. This is a larger first probe than a short dual-dot
kernel; it is not evidence that DualGemm cannot ultimately serve this case.

The Triton kernel uses the unchanged packed Kx8192 layout, loads A once per
K tile for the two dot operations, explicitly converts each FP32 accumulator
to BF16 round-to-nearest-even and back to FP32, then adds BF16 bias in FP32.
It follows the existing cube/tanh GELU and product expression before the
final BF16 store. `libdevice.tanh` maps to `__nv_tanhf`; `enable_fp_fusion=False`
preserves separate multiply/add steps. `enable_reflect_ftz=False` follows the
native build's default non-FTZ mode. No fast-tanh or reduced-precision dot is
introduced. Generated PTX and compiler resource usage are retained for review.

Only three fixed tiles are used, all BK32, 4 warps and 3 stages:

| BM x BN | CTAs | Question |
|---|---:|---|
| 32 x 32 | 256 | Can fewer M partitions limit repeated weight loads while filling the GPU? |
| 16 x 64 | 256 | At the same CTA count and tile area, is wider N reuse better? |
| 16 x 32 | 512 | Does more parallelism offset the smaller tile and repeated loads? |

The probe records the first actual prepared activation and output for each
of the 18 packed weight pairs, preserving real BF16 bias. The 288 MiB weight
cycle exceeds L2. It shares one intermediate and one output buffer like the
existing suffix. Each candidate checks all 18 real outputs using the existing
shallow metrics before A/B/B/A graphs of 15 raw samples per leg. Numerical
failure raises before that candidate is timed. Both timed paths exclude
prepare; end-to-end validation remains a later step if a candidate wins.

The run used deployment revision `7b2d9d0` and all 18 real weight pairs.
All three tiles produced outputs exactly equal to the captured packed path.

| Tile | A1 / A2 us | B1 / B2 us | Registers / shared bytes |
|---|---:|---:|---:|
| 32x32x32 | 16.0853 / 16.1476 | 18.1902 / 18.1920 | 64 / 12,288 |
| 16x64x32 | 16.0462 / 16.1280 | 14.3022 / 14.3253 | 64 / 18,432 |
| 16x32x32 | 16.0871 / 16.0960 | 15.9964 / 16.0000 | 40 / 10,240 |

All compiled without spills. The 16x64x32 winner saves approximately
1.72-1.83 us/call against its bracketing controls, implying 0.31-0.33 ms over
180 calls before full-model validation. Its control drift was 0.0818 us;
first samples were typically 1-1.5 us high and remain in the raw results.

The winner's PTX uses `mma.sync.aligned.m16n8k16` with BF16 inputs and FP32
accumulation. Both accumulator sets explicitly pass through `cvt.rn.bf16.f32`
and `cvt.f32.bf16` before bias additions. Surrounding FP32 arithmetic retains
separate `add.rn` and `mul.rn` operations. FMA instructions occur inside the
inlined libdevice `__nv_tanhf` implementation, not through expression fusion.
Raw JSON, log and all three PTX files are in the deployment checkout under
`artifacts/rtx5090-pi05/gpt6-expert-dual-dot-screen*`.

AST source checks and the bounded GPU screen passed. The script depends
on `lab/pi05/cutlass_gemm_screen.py` from `9b6449e` for the existing graph timer.
Use `--source-checkout` to select the deployed model/cache and the same
checkpoint options and seed used by the other Pi0.5 probes.
