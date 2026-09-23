# Source map — quantized kernels for SM120 (RTX 5090)

Where the open-source quantized GEMM, quantize and scale-layout code for
consumer Blackwell lives, at pinned revisions, and what does not carry over to
SM120. This is a locator, not a worked example: it names files and the facts
that decide whether a technique transfers. Every path and architecture claim
below was read from the pinned source.

**Scope: FP8 and NVFP4 on SM120's native low-bit tensor cores.** Weight-only
kernels (Marlin and similar dequantize-in-mainloop designs) are out of scope,
and W4A8 or other mixed precision waits until the FP8 and NVFP4 routes are
stable and faster; their sources are not fetched. FlashRT is this project's
comparison baseline and is not a reference ([AGENTS.md](../../../../AGENTS.md)).

| Source | Pin | Where it is | License |
|---|---|---|---|
| CUTLASS | `v4.7.1` | `third_party/cutlass/` (submodule) | BSD-3-Clause |
| vLLM | `v0.30.0` `ced6857a` | `third_party/quant-references/vllm/` | Apache-2.0 |
| SGLang | `v0.5.20` `94602c9c` | `third_party/quant-references/sglang/` | Apache-2.0; `kda_kernels/` files BSD-3-Clause (NVIDIA) |
| FlashInfer | `v0.7.0` `4d75a33f` | `third_party/quant-references/flashinfer/` | Apache-2.0 |

The three libraries are committed under `third_party/quant-references/` (about
12 MB; `update.sh` there moves a pin); CUTLASS needs
`git submodule update --init third_party/cutlass`.
Below, `vllm:`, `sgl:`, `fi:` and `cutlass:` abbreviate those roots;
`vllm:Q/` is `vllm:csrc/libtorch_stable/quantization/` and `sgl:K/` is
`sgl:python/sglang/kernels/`.

## SM120 facts that decide what transfers

- **Tensor cores are warp-level `mma.sync`, block-scaled included.**
  `cutlass:include/cute/arch/mma_sm120.hpp` issues
  `kind::mxf8f6f4.block_scale.scale_vec::1` (MXFP8/6/4, one UE8M0 scale per 32),
  `kind::mxf4nvf4.block_scale.scale_vec::2` (MXFP4) and `scale_vec::4` (NVFP4,
  one UE4M3 scale per 16). There is no `tcgen05`/TMEM/UMMA, so SM100 kernels do
  not run here, and no WGMMA, so SM90 kernels do not either — including the
  SM90/SM100 DeepGEMM paths in `third_party/deepgemm`.
- **TMA without multicast; cluster shape 1×1×1.** Stated in the header of
  `cutlass:examples/79_blackwell_geforce_gemm/79a_*.cu`, which also uses the
  persistent warp-specialized schedule and CLC dynamic scheduler.
- **Scale-factor layout is the SM100 one.** SM120 examples include
  `cutlass:include/cutlass/detail/sm100_blockscaled_layout.hpp`; the builder is
  `cutlass:include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`.
  Quantize kernel, GEMM and checkpoint repacking must agree on this layout.
- **Measured tensor rates** (FP32 accumulate, `lab/quantization/mma_blockscale_clock.cu`,
  2026-09-23): BF16 511.7, unscaled FP8 (legacy and `kind::f8f6f4`) 1023.1,
  MXFP8 block-scaled 2016.3, NVFP4 block-scaled 4032.6 FLOP/cycle/SM — about
  253 / 504 / 990 / 1926 TFLOP/s at 2.89 GHz. The block-scaled kinds run at
  twice the unscaled FP8 rate, so CUTLASS's SM120 FP8 and blockwise mainloops,
  which issue unscaled `kind::f8f6f4`, have half MXFP8's compute ceiling.

## Where each scheme lives

### FP8 with per-tensor or per-token × per-channel scales

- vLLM: `vllm:Q/w8a8/cutlass/scaled_mm_c3x_sm120.cu` →
  `c3x/scaled_mm_sm120_fp8.cu`, `c3x/scaled_mm_sm120_fp8_dispatch.cuh` (tile
  choice by M). Scale broadcast epilogues:
  `vllm:csrc/cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp`;
  epilogue algebra in `vllm:csrc/quantization/w8a8/cutlass/Epilogues.md`.
  `scaled_mm_entry.cu` routes every `version_num >= 120` device here.
- CUTLASS: builder over `cutlass:include/cutlass/gemm/collective/sm120_mma_tma.hpp`;
  `examples/58_ada_fp8_gemm` is the SM89 `mma.sync` FP8 pattern.
- Not SM120 at these pins: FlashInfer `fi:csrc/fp8_gemm_cutlass.cu` (SM100
  template); SGLang `sgl:K/aot/csrc/gemm/fp8_gemm_kernel.cu` (SM89 and SM90
  dispatch only). FlashInfer `fi:csrc/bmm_fp8.cu` is a cuBLASLt call.

### FP8 with block-128 / groupwise scales

- CUTLASS: `examples/87_blackwell_geforce_gemm_blockwise/` — 87b groupwise is the
  [1,128] activation × [128,128] weight configuration,
  `Sm120BlockwiseScaleConfig<1, 128, 128>`; 87a blockwise uses one scale per MMA
  tile; 87c grouped. Collective `sm120_mma_tma_blockwise_scaling.hpp`, which
  asserts that the scale's K granularity equals the tile's K: a finer K group
  means a shallower K tile and a promotion per group. vLLM, SGLang and FlashInfer
  below all wrap this collective.
  `examples/94_ada_fp8_blockwise` for SM89.
- vLLM: `vllm:Q/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8{.cu,_dispatch.cuh}`.
- SGLang: `sgl:K/jit/csrc/gemm/fp8_blockwise/fp8_blockwise_scaled_mm_sm120.cuh`.
- FlashInfer: `fi:csrc/gemm_groupwise_sm120.cu` over
  `fi:include/flashinfer/gemm/gemm_groupwise_sm120.cuh`; a hand-written CuTe
  version in `fi:csrc/cute_sm12x_gemm/sm120_blockscaling/` with
  `sf_fp8_tma_load.cuh` and shared `sm120_common/` (TMA loads, epilogue,
  scheduler).

### NVFP4 (E2M1 values, per-16 UE4M3 scales, FP32 global scale)

- CUTLASS: `examples/79_blackwell_geforce_gemm/` — 79a NVFP4→BF16, 79b
  NVFP4→NVFP4 with output scale factors generated in the epilogue, 79d grouped;
  collective `sm120_blockscaled_mma_tma.hpp`. `examples/91_fp4_gemv` for one row.
- vLLM: `vllm:Q/fp4/nvfp4_scaled_mm_sm120_kernels.cu` (entry
  `nvfp4_scaled_mm_entry.cu`, helpers `nvfp4_utils.cuh`).
- FlashInfer: b12x (above), the best measured NVFP4 kernel below large M;
  `fi:csrc/fp4_gemm_cutlass_sm120.cu` over
  `fi:include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`, whose
  `tilesSm120` list is the CTA-shape search space it autotunes over (cluster
  fixed 1×1×1); per-block and global scales are separate arguments.
- SGLang: `sgl:K/kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, a CuTe DSL port of the
  CUTLASS block-scaled example specialised to Qwen3 decode shapes. Its header
  records it as optimised by Kernel Design Agents (`mit-han-lab/kernel-design-agents`).

### What the libraries reach at VLA shapes

`lab/quantization/README.md` times every library below at the 27 GR00T N1.7
and Pi0.5 linear shapes with cold weights; raw results are in
`results/quant-kernel-survey-rtx5090/`. What decides a choice:

- **b12x is the reference for small and mid M.** FlashInfer's
  `fi:flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` (CuTe DSL,
  BSD-3-Clause, NVIDIA) handles MXFP8 and NVFP4 with 16/32-row tiles and
  swap-AB, reaching 1.27–1.33× the streaming roofline for MXFP8 at M = 41–50
  with wide N. Its planner and autotuner never enable its `split_k_slices`;
  forced on, split-K helps at M ≤ 64 and returns wrong output once M spans
  more than one tile.
- **FlashInfer's CUTLASS SM120 path tiles M at 128 or 256 without swap-AB**
  (`tilesSm120`), so small M is mostly padding; under `flashinfer.autotune` it
  picks stream-K, which wins a few mid- and large-M shapes.
- **Large M sits near 55% of the block-scaled peak** for the best of cuDNN,
  CUTLASS and b12x on the Pi0.5 backbone.
- **Down projections, small weights and mid M stay 1.9–3.2× off** (count-weighted,
  MXFP8 to NVFP4): too few output tiles, a fixed per-kernel cold-read cost, and
  128-row M tiles at M = 156–768.
- `sgl:K/kda_kernels/qwen3x_nvfp4_gemm_sm120.py` is decode-only and wrong outside
  its dispatcher's shapes (every N < K shape here); FlashInfer's `cute-dsl`
  backends reject SM120; vLLM has no MXFP8 GEMM of its own.

### FP8 with per-32 UE8M0 block scales (MXFP8)

The block-scaled tensor-core path for FP8, an alternative to software-applied
block-128 scales: `mx_float8_t<float_e4m3_t>` operands with
`OpClassBlockScaledTensorOp`, issuing
`kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` with UE8M0 scales. One scale
per 32 K elements for both operands — SFA is [M, K/32], SFB is [N, K/32] — applied
by the MMA itself. A [1,32] group through the software blockwise path above would
need a 32-deep K tile and FP32 scales (12.5% extra weight bytes), so 32-element
groups belong to this path.


- CUTLASS: 79c (MXFP8 × MXFP6 → BF16); `examples/80_blackwell_geforce_sparse_gemm/80a`.
- FlashInfer: b12x (above), the best measured MXFP8 kernel below large M;
  `fi:csrc/mxfp8_gemm_cutlass_sm120.cu` over
  `mxfp8_gemm_cutlass_template_sm120.h`; hand-written CuTe in
  `fi:csrc/cute_sm12x_gemm/sm120_blockscaled/` (the blockscaled path there is
  MXFP8, with `sf_mxfp8_tma_load.cuh`), op `cute_sm12x_mxfp8_op.cu`.

### Activation quantize and fusion

These decide whether a quantized GEMM pays off end to end: an unfused quantize
pass rereads the activation the GEMM just saved.

- vLLM: `vllm:Q/fp4/nvfp4_quant_kernels.cu` (`scaled_fp4_quant_sm1xxa`: BF16 →
  NVFP4 values plus swizzled scale factors);
  `fp4/activation_nvfp4_quant_fusion_kernels.cu` (`silu_and_mul_nvfp4_quant_sm1xxa`);
  `fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` (RMSNorm, optional
  residual, dynamic per-token FP8/INT8); `fused_kernels/fused_silu_mul_block_quant.cu`;
  `activation_kernels.cu` (`act_and_mul_quant_kernel`); `w8a8/fp8/common.cu`
  (static/dynamic per-tensor and per-token FP8); `w8a8/fp8/per_token_group_quant.cu`.
- SGLang: `sgl:K/aot/csrc/gemm/per_token_quant_fp8.cu`,
  `per_token_group_quant_8bit{,_v2}.cu`; JIT versions in `sgl:K/jit/csrc/gemm/`
  (`per_tensor_quant_fp8.cuh`, `per_token_quant_fp8.cuh`,
  `per_token_group_quant.cuh`).
- FlashInfer: `fi:csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh` with
  `fi:csrc/nv_internal/cpp/kernels/quantization.cu` (FP4 and MXFP8 quantize in the
  swizzled layout); Python in `fi:flashinfer/quantization/`. The library is built
  with `-use_fast_math`, so subnormals flush and an all-zero MXFP8 block
  quantizes to zeros. Its quantize kernels launch with PDL by default
  (`enable_pdl=None` means on where supported), which is worth ~0.35 µs per
  CUDA-graph node; compare kernels in kernel time, not only node time. `fi:flashinfer/cute_dsl/{,add_}rmsnorm_fp4quant.py` are
  CuTe DSL RMSNorm(+residual) → NVFP4/MXFP4 kernels that run on SM120. Their
  `global_scale` is the encode scale 448·6/amax, as for `fp4_quantize`,
  although the docstring says the output is divided by it.
- This repository (any sm_100+, tested on sm_120): `hardware/nvidia/quant_ops`
  implements RMSNorm,
  LayerNorm/AdaLN, GELU-tanh, gated activations and plain quantize, writing
  MXFP8 or NVFP4 with swizzled scales. Its output is byte-identical to
  FlashInfer's quantize of the BF16 result, and each op follows a model's
  rounding contract. Savings per observation are in
  `results/quant-fused-rtx5090/`. The device functions are in
  `hardware/nvidia/cuda/quant/block_quant.cuh`, for epilogues to include.
  Its kernels use PDL: the norm weight and bias load above the wait, and the
  trigger comes right after the wait (swept). Measured on sm_120,
  `griddepcontrol.launch_dependents` costs time per CTA that executes it,
  whether or not the grid was launched with PDL: an elementwise kernel with
  15,488 CTAs went from 8.3 to 15.0 µs. So trigger only in grids that fit in
  one wave, where an early trigger can fire early at all. No library above emits MXFP8 from a producer,
  or block-scaled output from LayerNorm, AdaLN, GELU or GeGLU.
- DiT-side fusion pattern: `sgl:K/ops/diffusion/sites/nvfp4_bias_gelu_site.py`
  (bias+GELU after an NVFP4 linear, stated bit-exact to the eager chain and
  mounted only under an explicit `quality="high"` contract) and
  `flux2_nvfp4_swiglu_quant_site.py`.

### Small M

VLA action heads run M = 41 (GR00T N1.7) or 51 (Pi0.5 expert); GR00T's
backbone runs M = 156. The upstream skinny kernels cover narrower shapes:
`sgl:K/kda_kernels/csrc/gemm/sm120_fp8_skinny_gemm.cuh` (16-row token tile,
M ≤ 9, KDA-generated) and `sgl:K/jit/csrc/gemm/sm120_fp8_gemv.cuh` (M = 1,
per-tensor scale). They show how the SM120 tile is narrowed, not a tile to
reuse at M = 41–51. FlashInfer's `tilesSm120` and vLLM's M-bucketed dispatch in
`scaled_mm_sm120_fp8_dispatch.cuh` show which shapes their authors considered.

### Quantized attention

`fi:csrc/nvfp4_attention_sm120/` with headers in
`fi:include/flashinfer/attention/sm120/nvfp4_attention_sm120/`: NVFP4 Q/K/V;
the binding instantiates head dim 64 (GR00T's ViT and VL self-attention are 64,
its DiT is 48).

### Checkpoint formats, scale layouts and numerical references

- vLLM Python: `vllm:vllm/model_executor/layers/quantization/modelopt.py`
  (ModelOpt NVFP4 checkpoints: per-group weight scales, a global
  `weight_scale_2`, `input_scale`), `fp8.py`, and `utils/` — `nvfp4_utils.py`,
  `fp8_utils.py`, `mxfp8_utils.py`, and `nvfp4_emulation_utils.py`, a PyTorch
  emulation usable as the numerical reference for an NVFP4 kernel.
- SGLang Python: `sgl:python/sglang/srt/layers/quantization/`.
- FlashInfer Python: `fi:flashinfer/gemm/gemm_base.py` chooses the SM120 FP4
  backend (CUTLASS, cuDNN — which needs cuDNN ≥ 9.14 for MXFP4 on SM120 — or
  CuTe DSL).

## Before transferring anything

1. Check the architecture guard first; SM90 WGMMA and SM100 tcgen05 code cover
   most of what does not run here.
2. Upstream tile tables target LLM decode (M ≤ 16) or large prefill. VLA shapes
   fall between; take local timing at the real M, N, K and cache state with
   [benchmark-kernel](../../benchmark-kernel/SKILL.md).
3. Scale granularity, layout and rounding must agree across the quantize
   kernel, the GEMM and the checkpoint repack; test them together, not one by one.
4. A quantized route changes the numerical contract, which
   [model-optimization](../../model-optimization/SKILL.md) reserves for approval.
   Compare against the BF16 reference and, where it approximates, a task-level
   evaluation.
5. Keep upstream attribution and license on anything ported; the `kda_kernels`
   files are BSD-3-Clause, not Apache-2.0.
6. Do not open FlashRT's source for any of this, including a checkout that
   happens to be on the machine.
