# Pi0.5 on RTX 5090 — MXFP8 backbone FFN (`mxfp8-llm-ffn`)

Optimization run of the quantized workload `mxfp8-llm-ffn` on branch
`exp/pi05-5090/mxfp8-llm-ffn`, starting at `29a29ea`. The recipe puts the three
GEMMs of all 18 LLM-backbone FFN layers (gate, up, down) in MXFP8: E4M3 values
with one UE8M0 scale per 32 elements along K, weights quantized once from BF16,
activations quantized on the device. Every other call site keeps its BF16 route.
The recipe was approved on its quality on 408 LIBERO observations
([`quant-pi05-ffn-libero`](../../quant-pi05-ffn-libero/compare.md)); its math is
fixed in `rtx5090/pi05/target.py` (`QUANTIZATION`) and recorded in every
identity's `execution_variant.quantization`.

A quantized workload is its own comparison context. This curve starts at the
recipe's fake-quant reference, and BF16 → MXFP8 is a latency/quality trade-off,
not an optimization gain. For scale, the BF16 shipped plan on the same revision
measures **30.07 ms** ([`measurements/bf16-shipped.json`](measurements/bf16-shipped.json)).

![Optimization progress](progress.svg)

## Workload and environment

Converted kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha weights;
batch 1; 3 × 224 × 224 images; 200 prompt slots; chunk 50; 18 layers; 10
denoising steps; timing input seed 42. RTX 5090 (170 SMs, 96 MiB L2), driver
580.142, torch 2.13.0+cu130, CUDA 13.1 for native builds, CUTLASS `cb424739`
(the repository's submodule). Clocks unlocked; the 600 W limit is enforced and
the MXFP8 version runs at it (SW power cap reason active, 593 W).

Measurements follow the workflow's conditions: fresh process per version, 5
warmup and 100 measured iterations, median, no profiler, serial on this GPU
under `/tmp/flash-vla-rtx5090.lock`. The checkpoint is passed through the
ignored symlink `artifacts/models/pi05_belt_cup_pytorch`, so no measurement
records a machine path.

```sh
. artifacts/rtx5090-pi05/gpt6-env.sh
ID=kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha
python -m benchmarks latency --target rtx5090/pi05 --plan shipped --seed 42 \
  --option quantization=mxfp8-llm-ffn --option converted_checkpoint=artifacts/models/pi05_belt_cup_pytorch \
  --option checkpoint_id=$ID --option checkpoint_digest=$ID --out results/pi05-rtx5090/mxfp8-llm-ffn/measurements/NNN.json
python -m eval.pi05.parity compare --oracle artifacts/rtx5090-pi05/oracle-belt-cup \
  --checkpoint artifacts/models/pi05_belt_cup_pytorch --checkpoint-id $ID --quantization mxfp8-llm-ffn
python -m eval.correctness --target rtx5090/pi05 --steps 1 --layers 2 --isolate --option quantization=mxfp8-llm-ffn ...
python lab/pi05/mxfp8_gemm_screen.py --out results/pi05-rtx5090/mxfp8-llm-ffn/gemm-screen.json
```

## Correctness

The recipe's kernels are held to its fake-quant reference
(`backends/fake_quant_ffn.py`: the operands are quantized by the same quant_ops
kernels and multiplied exactly, with the BF16 route's rounding points), with the
BF16 tolerances unchanged (`eval/tolerances.py`, tier `mxfp8`).

| check | result |
|---|---|
| Backend at real shapes, 2 layers, graph replay (`tests/test_pi05_mxfp8_backbone.py`) | FFN contribution rel RMS 1.7e-4, over 99% of outputs bit-identical |
| In-engine, 2 layers, isolated, belt-cup weights ([`recipe-layers2-isolated.json`](correctness/recipe-layers2-isolated.json)) | prefix K/V rel RMS 1.8e-4 / 2.7e-4 against the reference plan; within tolerance |
| In-engine, full depth, isolated ([`recipe-full-isolated.json`](correctness/recipe-full-isolated.json)) | drift report: deepest prefix V cosine 0.99853; actions rel RMS 0.0021 |
| Same, BF16 workload for scale ([`bf16-full-isolated.json`](correctness/bf16-full-isolated.json)) | deepest prefix V cosine 0.99972; actions rel RMS 0.0020 |
| Official OpenPI (BF16 oracle), MXFP8 ([`official-mxfp8-llm-ffn.json`](correctness/official-mxfp8-llm-ffn.json)) | passed; actions rel RMS 0.0090, deepest cosine 0.99830 |
| Official OpenPI, BF16 shipped ([`official-bf16.json`](correctness/official-bf16.json)) | passed; actions rel RMS 0.0042, deepest cosine 0.99931 |

With identical inputs the kernels reproduce the reference to 1e-4. Over 18
layers the quantized comparison drifts about twice as far as the BF16 one,
because a BF16-level difference upstream can move an E4M3 code. The action
chunk is unaffected. Against the official model, MXFP8 doubles BF16's action
error, the same ratio as on the LIBERO observations. LIBERO closed-loop
success has not been run for this recipe.

## 000 — Fake-quant reference, start

The shipped plan with both FFN call sites on `fake-quant-mxfp8`
([`plans/000-fake-quant-ffn.json`](plans/000-fake-quant-ffn.json)): **45.43 ms**.
The per-call PyTorch quantize, dequantize and FP32-output GEMMs make it the
slow end of the workload; it is its definition, not a candidate.

## 001 — MXFP8 kernels, retained

Per layer: `quant_ops.rms_norm` writes the MXFP8 activation; one CUTLASS SM120
block-scaled GEMM (`mx_float8_t`, 128×128×128 tile, persistent cooperative
schedule, PDL) multiplies it with gate|up packed as [32768, 2048] into BF16;
`quant_ops.gated_act` writes the MXFP8 hidden; the down GEMM adds its product to
the residual in the epilogue (β = 1). The hidden stays in backend scratch, so
both call sites route together (`RouteConstraint.atomic`). Quantized weights
add 1.74 GiB of scratch (99 MiB per layer); the BF16 copies stay allocated but
are not read.

Tile screen ([`gemm-screen.json`](gemm-screen.json); 18 layers of distinct cold
weights in one graph, median per GEMM):

| GEMM (M = 968) | 128×128 persistent | 128×64 | 128×32 | 128×128 Stream-K | 128×64 Stream-K |
|---|---:|---:|---:|---:|---:|
| gate/up N = 32768, K = 2048 | **228.2 µs** (569 TFLOP/s) | 276.5 | 437.3 | 257.0 | 297.0 |
| down N = 2048, K = 16384 | **124.2 µs** (523 TFLOP/s) | 156.8 | 229.9 | 126.4 | 157.6 |

For reference, the survey measured FlashInfer's tuned CUTLASS MXFP8 GEMM at
240.95 / 121.48 µs and cuDNN at 238.9 / 118.9 µs on these shapes
([`quant-kernel-survey-rtx5090`](../../quant-kernel-survey-rtx5090/README.md)).

Deployed median 45.43 → **22.48 ms**, revision `5961be0`. Against BF16 shipped
on the same revision (30.07 ms), the recipe saves 7.59 ms (25.2%).
