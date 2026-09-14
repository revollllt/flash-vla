# Pi0.5 expert down with a rounded gated epilogue

The 2026-09-15 candidate replaces only expert FFN down at M50 K4096 N1024.
It uses the previously screened cfg9 (CTA 32x64x32, warp 32x32x32, 8 stages)
and folds the gated residual into its fully reduced FP32 accumulator epilogue:
`BF16(FP32(BF16(accumulator)) * FP32(gate) + FP32(old_out))`.
Explicit `__fmul_rn` / `__fadd_rn` retain separate FP32 operations. The BF16
conversion remains visible before those instructions in the compiled SASS.

The new type and ABI share the existing Target's CUTLASS native library.
`cutlass_expert_residual.make_wrappers` exports only
`action_expert_ffn_down_residual`; no deployment routing is changed here.
Each factory owns its plans, input/output references, and one scratch role.
The plans share 2,622,080 workspace bytes because execution is serial on the
capture stream; the output aliases the residual source. Factory construction
is lazy and succeeds without CUDA_HOME or a CUDA context.

The first actual-call correctness attempt failed catastrophically before any
timing. Three constants then showed correct first-row values but unwritten
later row fragments. CUTLASS's `epilogue_with_broadcast.h::reduce` advances
the source iterator by `reduce_fragment_idx` but omits the destination advance;
ordinary `epilogue.h::reduce` advances both. A small Target-local derived
epilogue advances only the destination before invoking the upstream method.
The vendored headers and cfg9 mainloop are unchanged. After that fix, residual
only, all-ones product, and varying-column-gate probes are exact on all 50 rows.
The corrected kernel compiles for sm_120a with 120 registers and no spills.

The actual workload probe starts from shipped revision `63fa41a`, belt-cup
`orbax-39999+openpi-convert-pi05_aloha`, seed 42. It clones x and the old residual
at each of 180 calls; weights and precomputed timestep gates retain their
original references. All 180 output comparisons pass the existing shallow
requirements: maximum relative RMS 0.001951039, minimum cosine 0.999998142,
maximum absolute error 4.0. This is not exact cuBLAS parity because GEMM
reduction order differs. For calls 0/90/179, gate=1 and residual=0 extract the
candidate's BF16 projection; applying the existing native gated residual to
that projection exactly reproduces the fused result. That isolates the
epilogue's intermediate rounding from the GEMM reduction difference.

Each timing graph visits all 180 recorded calls and 18 distinct layer weights.
Both routes copy the saved residual to out before each call, avoiding repeated
residual accumulation. Each ABBA leg contains 30 raw samples; medians below
include the same reset copy in both routes.

| Route | Median us/call |
|---|---:|
| Current A1 | 12.72080 |
| Candidate B1 | 10.67627 |
| Candidate B2 | 10.68071 |
| Current A2 | 12.71182 |

The local median difference is 2.03782 us/call, or 0.36681 ms over 180 calls.
Control drift is 0.00898 us; candidate drift is 0.00444 us. This includes a
GEMM implementation change and must not be attributed solely to removing the
post kernel. The deployed profile's original post kernel totals only 0.185 ms;
official model correctness and end-to-end latency remain to be measured.

Raw evidence in `/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/`:

- `gpt6-expert-epilogue-failure.json`: original actual-call failure.
- `gpt6-epilogue-sanity.jsonl` and `gpt6-expert-epilogue-failed.so`: first coverage evidence and failing binary.
- `gpt6-expert-epilogue-fixed-build.log`: corrected build/resource usage.
- `gpt6-epilogue-sanity-fixed.jsonl`: corrected constant coverage, per-row mismatch counts.
- `gpt6-expert-epilogue-fixed-local.json`: all 180 errors, three decomposition checks, and 120 timing samples.

Reproduce with `python -m lab.pi05.rtx5090_expert_epilogue --seed 42` plus the
converted checkpoint, checkpoint identity options, and `--output` path.
