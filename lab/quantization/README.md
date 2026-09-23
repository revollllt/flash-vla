# Quantized GEMM survey on RTX 5090

2026-09-23. How close do existing MXFP8 and NVFP4 GEMM and quantize kernels get
to the roofline at Flash-VLA's real shapes, and where do they fall short?
Results: [`results/quant-kernel-survey-rtx5090/`](../../results/quant-kernel-survey-rtx5090/summary.md).

## Method

- **Shapes** (`vla_shapes.py`): the 27 main linear call sites of GR00T N1.7
  (from `lab/groot_n17/roofline.py`, checked against a real forward) and Pi0.5
  (from `models/pi05/spec.py`), packed as shipped, with calls per observation.
  Pi0.5's SigLIP FFN width 4304 is padded to 4352.
- **Formats**: MXFP8 (E4M3, UE8M0 per [1,32] on both operands), NVFP4 (E2M1,
  UE4M3 per 16, FP32 global), and 1D2D block FP8 ([1,128] × [128,128], FP32
  scales) for comparison. Random BF16 inputs; performance only, not quality.
- **Libraries**, each in its own venv under torch 2.13.0+cu130: FlashInfer
  0.7.0 (CUTLASS, cuDNN and b12x backends, default and `flashinfer.autotune`),
  vLLM 0.30.0 (CUTLASS NVFP4 and blockwise FP8), SGLang 0.5.20's KDA NVFP4
  kernel (standalone CuTe DSL file from the vendored copy), cuBLAS BF16 via
  torch as the baseline. CUTLASS itself is measured through the FlashInfer and
  vLLM instantiations.
- **Timing** (`bench_common.py`): a CUDA graph of ~256 calls cycling through
  enough weight copies to exceed 2× the 96 MiB L2, so weights are cold as in the
  model and the activation is warm; median of 5 replays; every backend is first
  checked against the BF16 matmul by cosine (> 0.99 MXFP8/1D2D, > 0.97 NVFP4).
- **Roofline** (`vla_shapes.floor_us`): max(2MNK / measured tensor peak,
  weight-and-scale MiB / 1.524 MiB/µs). Tensor peaks come from
  `mma_blockscale_clock.cu` on this card. The project's per-launch 3.35 µs
  cold-read spin-up is not charged: a back-to-back graph chain overlaps it and a
  smoke run beat it.

Clocks unlocked (observed 2.8–2.9 GHz), one idle GPU under the project lock.

## Tensor-core peaks measured (FP32 accumulate)

| MMA | FLOP/cycle/SM | TFLOP/s at ~2.89 GHz |
|---|---:|---:|
| BF16 `m16n8k16` | 511.7 | 253 |
| FP8 legacy `m16n8k32.e4m3` | 1023.1 | 506 |
| FP8 `kind::f8f6f4` (unscaled; CUTLASS SM120 FP8 and blockwise mainloops) | 1023.1 | 504 |
| MXFP8 `kind::mxf8f6f4.block_scale.scale_vec::1X` | 2016.3 | 990 |
| NVFP4 `kind::mxf4nvf4.block_scale.scale_vec::4X`, ue4m3 | 4032.6 | 1926 |

The block-scaled kinds run at twice the unscaled FP8 rate with FP32
accumulation, so software-scaled 1D2D FP8 has half MXFP8's compute ceiling.

## Headline

Sum of calls × time over the surveyed GEMMs, per observation (ms):

| model | format | roofline | best library | ×roofline | + unfused quantize | BF16 cuBLAS | best vs BF16 |
|---|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | MXFP8 | 3.17 | 5.47 | 1.73 | 0.84 | 11.35 | 2.07× |
| GR00T N1.7 | NVFP4 | 1.72 | 3.64 | 2.12 | 0.68 | 11.35 | 3.12× |
| GR00T N1.7 | 1D2D FP8 | 3.39 | 10.16 | 3.00 | 1.17 | 11.35 | 1.12× |
| Pi0.5 | MXFP8 | 6.53 | 12.42 | 1.90 | 1.25 | 30.09 | 2.42× |
| Pi0.5 | NVFP4 | 3.42 | 7.19 | 2.10 | 1.00 | 30.09 | 4.18× |
| Pi0.5 | 1D2D FP8 | 10.82 | 20.17 | 1.86 | 1.97 | 30.09 | 1.49× |

Count-weighted best / roofline by shape class: small M with wide N 1.29 (MXFP8)
/ 1.63 (NVFP4); down projections (K > N) 1.90 / 2.27; small weights 2.83 / 3.24;
mid M (156–768) 1.88 / 2.21; large M (Pi0.5 backbone) 1.83 / 1.84.

## Findings

- FlashInfer's **b12x** (SM12x CuTe DSL, `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`,
  BSD-3-Clause) is the best kernel at small and mid M: swap-AB, 16/32-row
  tiles. Its default planner and autotuner never use split-K.
- Forcing b12x split-K (`bench_b12x_splitk.py`) is correct only while M fits one
  tile: at M ≤ 64 S=4 cuts `groot/dit_down` 12.6 → 9.5 µs (MXFP8) and
  9.9 → 6.8 µs (NVFP4); at M ≥ 156 every S > 1 returns wrong output.
- FlashInfer's CUTLASS SM120 path tiles M at 128 or 256 without swap-AB, so at
  M = 41 it computes mostly padding; `autotune` rescues it by picking stream-K,
  which makes it the best choice on a few mid- and large-M shapes.
- cuDNN is best on several large-M MXFP8 shapes and reaches about 55% of the
  block-scaled peak there; no library exceeds that on the Pi0.5 backbone.
- SGLang's KDA NVFP4 kernel is decode-specialised (16×64×512 tile): slower than
  b12x wherever it is correct, and wrong on every shape here with N < K, which
  its public dispatcher would not admit.
- vLLM has no MXFP8 GEMM of its own (its MXFP8 path uses FlashInfer); its NVFP4
  CUTLASS kernel trails b12x by 1.5–2× below large M; its 1D2D FP8 is the only
  1D2D measured and is 1.1–1.5× faster than BF16 overall.
- FlashInfer's `cute-dsl` GEMM backends reject SM120.
- Standalone quantize kernels cost 0.8–2.2 µs per activation (up to 8 µs on
  Pi0.5's K = 16384 input), 15–19% on top of the GEMMs if not fused.

## Fused producer-quantize

The ops live in [`hardware/nvidia/quant_ops`](../../src/flash_vla/hardware/nvidia/quant_ops/README.md);
the results are in [`results/quant-fused-rtx5090/`](../../results/quant-fused-rtx5090/README.md).

- `check_quant_ops.py`: our quantize against FlashInfer's, byte for byte, at
  every surveyed activation shape. The inputs include zero, subnormal and
  outlier blocks. The script also checks the torch reference and feeds the
  output to b12x.
- `bench_fused_quant.py`: producer → quantize → GEMM against fused producer →
  GEMM at every GR00T and Pi0.5 A operand. `--no-pdl` launches our kernels
  without PDL. Fused with PDL, against unfused without, the chains take
  1.36–1.47 ms less per GR00T observation and 1.64–2.10 ms less per Pi0.5 one.
  Attention outputs keep a standalone quantize (0.13–0.22 ms with PDL).
- `compare_flashinfer.py`: our quantize and RMSNorm→NVFP4 against FlashInfer's,
  in kernel time (CUPTI, PDL off on both sides) and in graph-node time with PDL
  on and off. Kernel times are 0.80× (MXFP8) and 0.87× (NVFP4) of
  FlashInfer's, RMSNorm→NVFP4 1.11×. Graph nodes with PDL are 0.82×, 0.93× and
  0.87×.
- The PDL trigger point was swept with `FLASH_VLA_QUANT_PDL_TRIGGER` and
  `bench_fused_quant.py --only`
  ([`pdl_sweep`](../../results/quant-fused-rtx5090/pdl_sweep/README.md)).

## Reproduce on yx5090

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a mma_blockscale_clock.cu -o mma_blockscale_clock
```

The environments live outside the repository (`/home/ubuntu/quant-survey/venv-*`,
created with `uv` from the default index so the project's cached torch is reused;
FlashInfer's JIT needs `ninja` in the venv and `CUDA_HOME` set):

```bash
python bench_flashinfer.py [--autotune --skip-quant] --out <dir>/flashinfer[_tuned].json
python bench_vllm.py --out <dir>/vllm.json
python bench_sglang_kda.py --out <dir>/sglang_kda.json
python bench_b12x_splitk.py --fmt {mxfp8,nvfp4} --split {1,2,4,8} --out <dir>/b12x_splitk_<fmt>_s<S>.json
python bench_bf16.py --out <dir>/bf16_cublas.json
python summarize.py <dir>
```

The fused-quant scripts import the repository, so run them from its root with
`PYTHONPATH=src`, the FlashInfer venv and
`CUDA_HOME=$HOME/cuda-13.1`:

```bash
python lab/quantization/check_quant_ops.py
python lab/quantization/bench_fused_quant.py --out results/quant-fused-rtx5090
python lab/quantization/bench_fused_quant.py --no-pdl --out results/quant-fused-rtx5090/no_pdl
python lab/quantization/compare_flashinfer.py --out results/quant-fused-rtx5090
```
