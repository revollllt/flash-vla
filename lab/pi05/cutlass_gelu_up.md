# Rejected fixed-cfg0 backbone up/GELU epilogue

The fixed cfg0 fused-up candidate is slower in this local screen and was not
routed into the model. No other tile or end-to-end experiment was run.

## Hypothesis and exact scope

At source revision `76086d9564d4083860cf3335843742e3ad9f5b87`, the 17 backbone
FFNs use separate cfg0 gate and up GEMMs, then `backbone_gelu_mul`. Actual up
shape is M=968, K=2048, N=16384. The 010 trace attributed 5013.357 us to the
17 up GEMMs and 443.883 us to the 17 standalone GELU/product kernels.
Thus 0.443883 ms is an optimistic ceiling if the up GEMM did not slow down;
an 8.85% up slowdown consumes that entire ceiling.

The experiment keeps RMSNorm, gate GEMM, cfg0 tile (128x128x64, four warps,
three stages), and source weights unchanged. Ordinary matrix C supplies the
materialized BF16 gate; D fully overwrites the up output. The new output op
first rounds the up FP32 accumulator to BF16 and converts it back to FP32.
It then evaluates precisely the deployed tanhf expression for GELU(gate)
and multiplies by the rounded up, finally rounding the product to BF16.
The gate buffer is read-only; neither path reads the previous D.

The operator uses the ordinary Stream-K epilogue, whose reduction completes
before applying the output op. It does not use the previously repaired
broadcast epilogue. The nonlinear `kIsHeavy=true` hint follows upstream
LinearCombinationGELU. This is one fixed implementation, not a hint or tile sweep.

## Compilation and numerical evidence

The control and candidate were compiled into the same existing
`libcutlass_backbone.so`; there was no second CUTLASS library. The separate
already deployed pointwise library contains only the control RMS/GELU kernels.
CUDA 13.1 flags were unchanged: `-O3 -std=c++17 --shared -Xcompiler -fPIC
--expt-relaxed-constexpr -Xptxas=-v -gencode arch=compute_120a,code=sm_120a`.
Neither fast-math nor `--fmad=false` was added.

Both the original cfg0 and RoundedGeluMul compiled at 254 registers with zero
stack frame, spill stores, or spill loads. The unchanged tile still has the
large shared-memory footprint that previously limited it to one CTA per SM;
zero extra registers alone does not imply a free epilogue.

Generated SASS retains the BF16 roundtrip. For example, candidate offsets
c6b0/c6c0 pack FP32 accumulators into BF16, c6d0 and following SHF/LOP3 instructions
expand those rounded bits to FP32, then FMUL consumes them before the final BF16
packs at c7d0-c820. A corresponding sequence appears at 8960 and following.
The compact ptxas and SASS excerpt is in
[backbone-gelu-up-cfg0-resources.txt](../../results/pi05-rtx5090/gpt6-run-01/measurements/backbone-gelu-up-cfg0-resources.txt).

The probe clones actual normalized inputs and completed FFN outputs at each
of the 17 belt-cup seed-42 invocations, retaining every layer's source weights.
It recomputes the gate with the original cfg0, then checks both separate
up+GELU and fused-up against the captured actual output. All 17 layers have
max_abs=0 and rel_rms=0 for both paths. The original shallow tolerance checks
were reused; no new correctness gate was introduced.

## Fixed ABBA result

Each leg is one CUDA graph over all 17 distinct up weights, with 15 event
samples on the capture stream. Both variants overwrite the same output.
RMSNorm and gate GEMM are precomputed outside both timed regions; neither route
needs a residual reset. Workspaces are equal at 22,283,008 bytes.

| Leg | Median total, 17 calls (ms) |
| --- | ---: |
| A1: separate up + GELU | 5.528512 |
| B1: fused up | 5.716608 |
| B2: fused up | 5.778464 |
| A2: separate up + GELU | 5.623680 |

The fused path is slower by 0.171440 ms (3.07%) using the mean of leg medians.
Control drift is 0.095168 ms and candidate drift is 0.061856 ms. Even the
faster B leg exceeds the slower A leg by 0.092928 ms. This screen gives no
reason to advance this fixed cfg0 candidate.

The timing region pre-materializes all 17 gate matrices (539,230,208 bytes),
rather than immediately consuming the shared gate produced in the deployment.
Distinct up weights total 1,140,850,688 bytes. This changes cache context.
Consequently the measurement does not establish deployment latency or isolate
whether nonlinear computation, memory behavior, or limited CTA concurrency
caused the slowdown. It rejects this local fixed cfg0 implementation only.

[Raw results](../../results/pi05-rtx5090/gpt6-run-01/measurements/backbone-gelu-up-cfg0-rejected.json)
preserve all 17 per-layer errors, plan identity, workspace sizes, and 60 samples
(the raw samples are milliseconds per call; multiply by 17 for total time).

## Reproduction

The production CUDA/ABI delta was reverted after the screen and its experiment
binary removed. [cutlass_gelu_up.patch](cutlass_gelu_up.patch) preserves exactly
the two-file native/ABI delta. Apply it to the recorded source base in an
isolated checkout; use the one updated native library for both variants.
Do not load a second library instantiating the original cfg0 template because
its GNU-unique CUTLASS TLS state can collide.

With the patch applied, run:

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass PYTHONPATH=$PWD/src:$PWD \
  /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.cutlass_gelu_up \
  --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /tmp/backbone-gelu-up-local.json
```

Full compiler log and SASS remain in the ignored main artifacts directory as
`gpt6-gelu-up-build.log`, `gpt6-gelu-up.sass`, and
`gpt6-gelu-up-candidate.sass`. No production route or vendor file changed.
