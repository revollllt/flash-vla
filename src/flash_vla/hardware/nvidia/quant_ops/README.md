# quant_ops: producers that emit MXFP8 / NVFP4

The A operand of a quantized GEMM comes out of a norm or an activation. Running
that producer in BF16 and then a separate quantize kernel costs one extra
graph node and one extra round trip of the activation per call site. These ops
quantize inside the producer's epilogue. They write E4M3 or E2M1 values plus the
128x4 swizzled scales that Blackwell's block-scaled GEMMs (CUTLASS, cuDNN,
FlashInfer `mm_mxfp8` / `mm_fp4`) read. They are a starting point for Targets
and agents: correct, graph-safe and measured, not tuned to the roofline.

**Scope: every Blackwell GPU (sm_100 or later).** The kernels are plain CUDA,
with no tcgen05, TMA or cluster features. The one architectural requirement is
the E2M1 conversion `cvt.e2m1x2.f32`, which Hopper and earlier lack. They also
lack the block-scaled MMA that consumes this output. The library builds for the
current GPU's family: `sm_100f` (B200/GB200, B300), `sm_110f` (Thor) or `sm_120f`
(RTX 50, DGX Spark). All five arch-specific targets from sm_100a to sm_121a also
compile. It is **tested on the RTX 5090 (sm_120) only**; the measurements below
are that card's. The device-side quantization lives in
[`cuda/quant/block_quant.cuh`](../cuda/quant/block_quant.cuh), so other kernels
(attention or GEMM epilogues) can include it directly.

| op | computes | used by |
|---|---|---|
| `rms_norm` | RMSNorm, weight `none` / `mul` (bf16(bf16(x·r)·w)) / `one_plus` (Gemma); optional residual add, bf16-rounded factor and `factor_out` (Pi0.5 AdaRMS) | Qwen3 LLM, Pi0.5 Gemma and action expert |
| `layer_norm` | LayerNorm in fp32 with one rounding; optional affine; optional AdaLN `y·(1+scale)+shift` rounded per op, per-row-group and row-strided modulation | SigLIP / Qwen3-VL vision, GR00T refiner and DiT |
| `gelu_tanh` | PyTorch's `gelu(approximate="tanh")` | vision MLPs, GR00T FFNs |
| `gated_act` | `act(gate)·up`, GELU-tanh or SiLU, one rounding (Pi0.5 GeGLU) or HF's bf16 `silu(g)` first (SwiGLU); gate and up may be the two halves of one GEMM output | Pi0.5, Qwen3 |
| `quantize` | the input as is | A operands no producer here covers (attention output) |

```python
from flash_vla.hardware.nvidia.quant_ops import ops

a = ops.empty(m, k, "mxfp8")                         # once, before capture
ops.rms_norm(x, a, weight=w)                         # a.values: float8_e4m3fn [m, k]; PDL on
flashinfer.mm_mxfp8(a.values, w_q.t(), a.scale, w_sf, out_dtype=torch.bfloat16)

s = torch.tensor([448 * 6 / act_amax], device="cuda")   # calibrated NVFP4 encode scale
b = ops.empty(m, k, "nvfp4", global_scale=s)            # b.values: uint8 [m, k/2]
```

Pass `allocate=lambda n, shape, dtype: scratch(role + n, shape, dtype, device)` to
`empty` to allocate outputs from a runner's `Scratch`.

## Contracts

- **Quantization** is FlashInfer 0.7.0's default fast path (`block_quant.cuh`
  cites the source; FlashInfer runs the same code on sm_100 and sm_120), including its `-use_fast_math` flush of subnormals.
  Values and whole scale buffers are byte-identical to
  `flashinfer.mxfp8_quantize(x, True)` and `fp4_quantize(x, S, 16, False, True)`,
  checked at every surveyed shape by
  [`lab/quantization/check_quant_ops.py`](../../../../../lab/quantization/check_quant_ops.py).
  An all-zero block gets scale code 0 and zero values.
- **Fusion**: `fmt="bf16"` runs the same kernel stopped at its rounding point.
  `quantize(op(x) → bf16)` equals `op(x) → mxfp8/nvfp4` byte for byte
  (`tests/test_quant_ops.py`).
- **Producers** round where the model references round
  (`flash_vla.quantization.reference`). They agree with it up to reduction
  order: a 1-ulp difference on under 1% of values.
- **Layout**: scales use the flat 128x4 layout (`quantization/formats.py`), with
  rows padded to 128 and scale columns to 4. Kernels never write the padding.
  `empty` zeroes it once, and it must stay zero because the GEMM reads it
  (UE8M0 0xFF is NaN).
- **NVFP4 global scale** is a device tensor, the encode scale `448·6/amax`. It
  is read at run time, so a calibrated value can change without recapture.
- Limits: bf16 inputs, 16-byte-aligned rows, K divisible by 32 (MXFP8) or 16
  (NVFP4). Row ops need K ≤ 8192 because they keep the row in registers, one CTA
  per row with one thread per 8 elements. On first use, `nvcc` builds the
  library into `.cache/cuda_ext/nvidia_quant_ops/sm_<family>/`. It uses
  `--fmad=false`, like the repository's other torch-matching kernels.

## Programmatic dependent launch

Every op launches with PDL unless called with `pdl=False`, so in a CUDA graph
it starts while the previous kernel drains and lets the next one start early.

- **Wait, derived:** `cudaGridDependencySynchronize()` sits before the first
  read of anything an earlier kernel writes: the activation, the residual, the
  AdaLN modulation (a per-step projection) and the NVFP4 global scale. Only the
  norm weight and bias load above it. No kernel of a captured chain may write
  those two tensors.
- **Trigger, swept:** dependents are released right after the wait (point 0 of
  three; [`pdl_sweep`](../../../../../results/quant-fused-rtx5090/pdl_sweep/README.md)).
  The trigger executes only when the grid fits in one wave. Each triggering CTA
  costs time, with or without PDL (15,488 CTAs: 8.3 → 15.0 µs), and in a
  multi-wave grid the trigger cannot fire early anyway.
- **Per call site:** the best setting belongs to the pair of our kernel and the
  GEMM after it. Pi0.5's `exp geglu -> down` in NVFP4 is 1.3 µs slower per call
  with PDL than without (0.23 ms per observation); pass `pdl=False` there.
- `tests/test_quant_ops.py` replays a captured chain of these ops, each reading
  the previous one's output, against the same chain run without PDL. With the
  elementwise wait moved below its loads, that test fails.

## Measured (RTX 5090, 2026-09-23)

[`results/quant-fused-rtx5090/`](../../../../../results/quant-fused-rtx5090/README.md),
from [`lab/quantization/bench_fused_quant.py`](../../../../../lab/quantization/bench_fused_quant.py).
Each A operand is timed as producer [→ quantize] → FlashInfer b12x GEMM(s) on
cold weights, with graphs replayed interleaved for 21 rounds. The sum over all
producer sites, in ms per observation:

| model | format | unfused, PDL off | fused, PDL off | unfused, PDL | fused, PDL | fused + PDL saves |
|---|---|---:|---:|---:|---:|---:|
| GR00T N1.7 | MXFP8 | 7.47 | 6.83 | 6.56 | **6.11** | 1.36 ms (18%) |
| GR00T N1.7 | NVFP4 | 6.00 | 5.34 | 5.08 | **4.53** | 1.47 ms (24%) |
| Pi0.5 | MXFP8 | 17.33 | 16.15 | 16.20 | **15.23** | 2.10 ms (12%) |
| Pi0.5 | NVFP4 | 11.11 | 9.67 | 9.92 | **9.06** | 2.05 ms (18%) |

These are sums over the listed sites, not end-to-end latency. Each site's chain
is replayed on its own with random data and cold weights. The sums leave out
attention, the output projections and their GEMMs, embeddings, action
projections and host work. For scale: Flash-VLA's shipped BF16 Pi0.5 replays
end to end in 32.2 ms, and the nine Pi0.5 GEMM sites summed here cost
27.5 ms in BF16 cuBLAS under the same conditions. An end-to-end number needs
a quantized Target measured with the latency protocol.

"PDL off" means our kernels are launched without PDL; FlashInfer's GEMMs keep
their default either way. Pi0.5 includes one site (`llm geglu -> down`) where
the unfused chain's 32 MiB BF16 intermediate evicts the GEMM operands from L2.
Counting only the kernels fusion removes there, Pi0.5 saves 1.99 ms (MXFP8)
and 1.64 ms (NVFP4). Fusion and PDL add up. Fusion removes a quantize node
per site, and PDL overlaps the launches that remain; a fused link gains a median
1.5 µs from PDL. Attention outputs keep a standalone quantize, which costs
0.13–0.22 ms per observation with PDL (0.18–0.27 ms without).

### Against FlashInfer's own kernels

[`compare_flashinfer.md`](../../../../../results/quant-fused-rtx5090/compare_flashinfer.md),
from [`lab/quantization/compare_flashinfer.py`](../../../../../lab/quantization/compare_flashinfer.py).
Ratios are ours / FlashInfer: the geometric mean over the survey's shapes, with
the range in parentheses. Below 1 means ours is faster. Kernel time comes from
CUPTI with PDL off on both sides, because a PDL-launched kernel counts its wait
for the predecessor as its own time. Node time is per CUDA-graph node.

| op | kernel | graph node, both with PDL | graph node, both without |
|---|---:|---:|---:|
| `quantize` → MXFP8, 14 shapes | 0.80 (0.55–0.97) | 0.82 (0.54–1.02) | 0.77 (0.57–0.93) |
| `quantize` → NVFP4, 14 shapes | 0.87 (0.70–1.10) | 0.93 (0.77–1.23) | 0.84 (0.72–1.08) |
| `rms_norm` → NVFP4 vs `rmsnorm_fp4quant`, 5 shapes | 1.11 (0.96–1.33) | 0.87 (0.66–1.19) | 1.08 (0.96–1.27) |

PDL makes our nodes 1.14× (quantize MXFP8), 1.16× (NVFP4) and 1.37×
(RMSNorm→NVFP4) faster. That is more than FlashInfer gains from its own PDL,
because our norms load their weights above the wait. Graph nodes where
FlashInfer still wins: NVFP4 quantize at M ≥ 512 (up to 1.23×) and
RMSNorm→NVFP4 at M = 968 (1.19×).

## Known headroom

- Large M: at M ≥ 512, FlashInfer's NVFP4 quantize and its CuTe DSL
  `rmsnorm_fp4quant` (vendored under
  `third_party/quant-references/flashinfer/flashinfer/cute_dsl/`) win per graph
  node. Their CTAs are fewer and larger than our 128-thread tiles and
  one-row CTAs. `rmsnorm_fp4quant` rounds once instead of HF Qwen3's twice, so
  about 2% of bytes differ. Its `global_scale` is the encode scale, although the
  docstring says the output is divided by it.
- A per-call-site trigger point: for the pair above, point 1 beats both point
  0 and no PDL.
- Each broadcast parameter vector (weight, bias, AdaLN scale/shift) adds about
  0.2 µs to a 41-row norm. The cause was not profiled (ncu needs admin here).
- Attention outputs still take a standalone quantize. The next saving is an
  attention epilogue that writes MXFP8 / NVFP4.
