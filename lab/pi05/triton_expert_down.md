# Fixed-tile expert FFN down screen

Prepared from `76828b7b3cc962497cd5f52b510b0998b225f0fa`, which includes 021.
This is a lab-only candidate for `action_expert_ffn_down_residual`; no production
route, native ABI, other expert site, or tile search changes.

## Hypothesis and limits

The 019 profile attributes 1.526675 ms to 180 current cfg9 residual GEMMs
(8.481528 us/call). This is already one kernel per call, including the rounded
gated residual. There is no remaining post launch or projection buffer to remove.

Try exactly the previously validated QKV tile, 16x32x32, four warps and three
stages, now at M50 K4096 N1024 with no split-K. It launches 4x32=128 CTAs.
Its simple mainloop might reduce Stream-K coordination and partial-accumulator
traffic. The current 2,622,080-byte workspace is capacity, not measured traffic
that the candidate necessarily saves.

The current trace shows 298 CTAs, 64 threads/CTA, 120 registers and 49,152 bytes
shared memory. Candidate 128 CTAs cannot occupy all 170 SMs simultaneously;
it has 512 total warps against the current 596. K is four times QKV's K and N
is smaller, so the prior QKV 320-CTA result does not establish a benefit here.
Each logical M16 group requests B, giving four row groups versus cfg9's two
logical M32 groups; actual Stream-K requests and L2 reuse are not measured.

Each weight is 8 MiB and the 18 distinct weights total 144 MiB, exceeding the
96 MiB L2. A repeated isolated down graph still has different cache conditions
from interleaved model execution. A 0.5-1 us local improvement would correspond
to only 0.09-0.18 ms by multiplying the call count; this is a screening scale,
not a performance prediction or deployment result.

## Arithmetic and alias

Inputs are contiguous BF16 x(50,4096), weight(4096,1024), gate(1024), and
out(50,1024), with C=D. The fixed dot uses FP32 accumulation, then explicitly
rounds to BF16 RN and converts back to FP32. It separately multiplies by the
FP32-converted channel gate, adds FP32 old out, and stores BF16 RN.
`enable_fp_fusion=False` and `enable_reflect_ftz=False` match the validated
QKV epilogue's settings. Each CTA owns a disjoint output tile, so the residual
can be read and overwritten in place. GEMM reduction order may differ from
cfg9; BF16 roundtrip preservation does not promise bitwise parity.

## Bounded protocol

First run `--compile-only`. Triton dtype warmup compiles the one tile without
launching it or loading the model; loading its function exposes register/spill
counts. Save PTX and resources, then inspect the BF16 conversion/expansion and
independent FP32 multiply/add before proceeding. No second CUTLASS library is
built by this experiment.

The full run records all 180 real belt-cup seed-42 calls, cloning x, gate, old
residual, and actual output at the invocation boundary while retaining weights.
Both routes use the same snapshots and a single shared output. Control uses
the current deployed cfg9 wrapper; no replacement control implementation is
introduced. Per-call errors and exact-equality diagnostics are written as the
checks proceed. Existing shallow tolerances apply; any failure is saved and
raised before timing.

Calls 0/90/179 additionally use the same candidate with gate=1 and residual=0
to extract its BF16 projection, then apply the existing native separate gated
residual. Exact agreement with the fused output isolates epilogue rounding from
GEMM reduction-order differences.

After numerical checks pass, run one ABBA with 15 raw samples per leg over all
180 calls. A copy of the saved residual precedes every kernel inside both timed
graphs. This prevents accumulation and preserves the same C=D alias. Report
mean leg gain, both leg drifts, conservative separation and all 60 raw samples.
If gain does not exceed the observed drift, or is negative, stop; no tile
expansion or deployment timing follows from this screen.

## Commands after an exclusive GPU/JIT slot is granted

```sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass PYTHONPATH=$PWD/src:$PWD \
  /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.triton_expert_down \
  --compile-only --output /tmp/triton-expert-down-compile.json
```

After inspecting the resulting PTX/resources:

```sh
CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass PYTHONPATH=$PWD/src:$PWD \
  /home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.triton_expert_down \
  --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /tmp/triton-expert-down-local.json
```

## Measured result: rejected

The sole candidate completed on the recorded source base with Triton 3.7.1.
Compilation reports 38 registers, zero spills and 6,144 bytes shared memory.
PTX lines 167/170 use BF16-input FP32-accumulator mma.sync; lines 207-210
round to BF16 before a shared-memory layout conversion, and 255/261 convert
back to FP32. Lines 296-297 multiply and 299-300 add using separate RN f32x2
instructions, followed by final BF16 RN conversion at 303/305. There is no fma.
The [resource/PTX excerpt](../../results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-triton-fixed-resources.txt)
preserves this evidence.

All 180 actual calls pass the existing shallow checks. The cfg9 control is
exactly equal to every captured output. The candidate is not bitwise equal:
maximum absolute error is 0.5, worst relative RMS is 0.000269683128, and minimum
cosine similarity is 0.999999963692. All three same-mainloop projection
decompositions (0/90/179) match exactly.

| Leg | Median reset-inclusive us/call |
| --- | ---: |
| A1: deployed cfg9 | 10.567289 |
| B1: Triton fixed tile | 29.861511 |
| B2: Triton fixed tile | 29.867199 |
| A2: deployed cfg9 | 10.577244 |

The candidate is slower by 19.292088 us/call using mean leg medians.
Control/candidate drift is 0.009955/0.005688 us, far smaller than the separation.
All 60 raw samples are retained, including the first sample of each leg.
The result rejects this specific no-split-K 16x32x32 implementation. The
128-CTA/long-K/cache concerns remain possible explanations, not measured
causal attributions; low register count alone did not predict performance.
No additional tile, profiling, or deployment timing was run.

[Raw results](../../results/pi05-rtx5090/gpt6-run-01/measurements/expert-down-triton-fixed-rejected.json)
include all 180 per-call comparisons, three exact decomposition checks,
actual layouts, 18 retained weights (150,994,944 bytes), deployed plan, and
the fixed ABBA samples. Full compile and local PTX/log artifacts remain as
`artifacts/rtx5090-pi05/gpt6-triton-down-{compile,local}.{json,ptx,log}`
in the main checkout. The GPU/JIT window was released immediately after process
exit. The candidate remains lab-only; production code and routes are unchanged.
