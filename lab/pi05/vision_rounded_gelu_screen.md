# Vision FFN-up rounded GELU: fixed cfg10 experiment

Prepared on a fresh branch from main `4de57c9`. This revision adds one optional
native GEMM type plus lab code; it does not change a production route.
The fixed experiment has completed in its exclusive NVCC/GPU window; measured
evidence is recorded below.

## Evidence and scope

The separate vision GELU takes 27 calls and 90.916 us in the validated
`profiles/021-overview/overview.json`, close to 90.366 us in 015.
In 015 the immediately preceding cfg10 FFN-up GEMMs total 1110.793 us,
or 41.140 us/layer, and use 160 registers/thread, 60 KiB dynamic shared memory,
128 threads and 340 CTAs. These are profile observations, not new measurements.

The only candidate retains cfg10 tile 64x128x32, warp 32x64x32, 5 stages,
BF16 operands, FP32 accumulation, alpha=beta=1 and bias through C with ldc=0.
A custom output operator first calls the existing LinearCombination operator,
obtaining its BF16-rounded biased result. GELU then runs on that value in FP32,
followed by the final BF16 RN conversion. No GEMM input, normalization, bias
folding, or adjacent vision route changes.

The deployed pointwise source is `fused_vision.cu::pi05_vision_gelu`, compiled
by `fused_vision.py` with `--fmad=false`. The shared CUTLASS library omits that
flag. The candidate therefore uses explicit RN intrinsics locally, without
changing the flags of any existing kernel:

```
cubic = mul_rn(mul_rn(mul_rn(0.044715f, x), x), x)
inner = mul_rn(0.7978845608028654f, add_rn(x, cubic))
gelu = mul_rn(mul_rn(0.5f, x), add_rn(1.0f, tanhf(inner)))
```

Here x is the BF16 biased projection converted back to FP32. The same tanhf
routine is retained. CUTLASS GELU_taylor is not used: it changes the expression,
contraction and tanh implementation. The candidate sets kIsHeavy=true, following
the vendor's heavy-activation convention to limit epilogue-loop expansion.

The ordinary Stream-K epilogue applies this output operator after complete
accumulator reduction, in both the cooperative and separate reduction paths.
No broadcast epilogue or reduction override is introduced.

## Resource risk

The removed intermediate write/read is 13,221,888 B/layer, 340.45 MiB/model.
The separate pass already operates substantially from cache; these bytes cannot
all be treated as DRAM traffic or added to its measured time as extra savings.
Its arithmetic moves into GEMM, where fewer resident warps may hide tanhf
latency less effectively. About an 8% GEMM slowdown could erase the standalone
pass's current cost. Register growth, local-memory traffic, instruction count,
and longer output processing remain unresolved until compilation and timing.

Old and new cfg10 types compile in the SAME updated `libcutlass_backbone.so`
inside this worktree's cache. There is no second GEMM .so or duplicate cfg10
control template in another library. The unchanged pointwise library continues
to provide LayerNorm and the control's separate GELU.

The run preserves existing nvcc -Xptxas=-v output in its log and saves
cuobjdump --dump-resource-usage for the same library. A narrow query returns
the candidate's compiled sizeof(GemmKernel::SharedStorage), its launch's dynamic
shared-memory allocation. No resource value is guessed from the old kernel.

## Verification and one timing experiment

Use the existing recorder to capture real inputs and expected complete FFN-up
outputs for all 27 vision layers of the selected shipped engine. Control calls
the existing cutlass-vision wrapper using the updated shared library; candidate
uses the same native LayerNorm and its new rounded-GELU GEMM. Both use identical
input, normalized activation and final output addresses. Workspaces and plans
are owned by the probe/Scratch and warmed before capture.

Verify the unchanged control by elementwise equality. Check the one candidate
on every layer with existing error_metrics and shallow tolerances; retain all
diagnostics. Any numerical or execution failure is saved and re-raised before
timing. Then freeze scratch and run exactly one complete-chain ABBA, 15 raw
samples/leg. Report all medians and both directions of drift. The 27 real
weights total 255.34 MiB; the local chain still is not the full deployment
sequence or E2E evidence. No extra tile, flag or repeated sweep is scheduled.

The script dependencies `lab/pi05/cutlass_vision_chain.py::record_calls` and
`lab/pi05/cutlass_gemm_screen.py::samples_ms` are already present in the base.
The main checkout's existing CUTLASS vendor must be selected explicitly below.

## Command after the window is granted

```bash
cd /home/ubuntu/flash-vla-gpt6-backbone
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
export CUTLASS_DIR=/home/ubuntu/flash-vla/third_party/cutlass
set -o pipefail
/home/ubuntu/flash-vla/.venv/bin/python -m lab.pi05.vision_rounded_gelu_screen \
  --seed 42 \
  --option converted_checkpoint=/home/ubuntu/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --option checkpoint_digest=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha \
  --output /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-screen.json \
  2>&1 | tee /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-screen.log
```

## Measured result

The single experiment completed at source `f46185a` with old and new cfg10
types in the same updated worktree library. The initial compile used the
existing optimization/architecture flags plus --keep/--keep-dir to retain PTX;
no arithmetic code-generation flag changed. The prepared probe then reused
that library rather than rebuilding it.

All 27 unchanged-control outputs matched the recorded model call elementwise.
The candidate also matched every element: max absolute error and relative RMS
error were zero. The actual shape was M768 K1152 N4304, and the 27 weights
occupied 267,743,232 bytes.

One ABBA, reported as each leg's median over 15 raw samples:

| A1 (ms/27 layers) | B1 | B2 | A2 |
|---:|---:|---:|---:|
| 1.293344 | 1.252384 | 1.262688 | 1.296096 |

Mean A-B is 0.037184 ms; conservative min(A)-max(B) is 0.030656 ms.
Control drift is 0.002752 ms and candidate drift is 0.010304 ms.
The second B leg has a visible step within its raw samples; all samples were
retained, and the cause was not diagnosed. The conservative gain exceeds the
observed local leg drift, supporting one deployment experiment, but this small
local difference is not established E2E savings. No repeat, tile, or hint sweep
was run. The exclusive window was returned immediately on successful exit.

Resource evidence from the same build: candidate 156 registers/thread versus
control 160; both have zero stack bytes and zero spill stores/loads. The candidate
query reports 61,440 B of compiled dynamic shared storage. The register reduction
does not by itself explain the timing result.

The extracted candidate PTX confirms the intended boundary and sequence.
Lines 726-733 apply the existing alpha/beta accumulator-plus-bias operation;
734-746 explicitly convert it to BF16 RN. Lines 749-756 widen the BF16 bits back
to FP32. Lines 758-765 show the three left-associated RN products, RN addition,
inner multiplier and half multiplier; 784-785 show the final RN addition and
product after tanhf. FMA and approximate instructions inside the lowered tanhf,
and FMA in the preexisting alpha/beta operation, must not be mistaken for
contraction of the explicitly separated GELU arithmetic.

Raw artifacts in the main checkout:

- `artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-build.log`
- `artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-kernel.ptx` (candidate entry only)
- `artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-screen.json`
- `artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-screen.log`
- `artifacts/rtx5090-pi05/gpt6-vision-rounded-gelu-screen-resources.txt`

The complete kept PTX remains under this worktree's ignored
`.cache/cuda_ext/rtx5090_pi05_cutlass_backbone/vision_rounded_gelu_keep/`.
Generated files are not included in the source commit.
