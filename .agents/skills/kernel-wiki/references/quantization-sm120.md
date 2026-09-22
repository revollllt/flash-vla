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

`bash third_party/quant-references/fetch.sh` creates the three sparse checkouts
(about 23 MB); CUTLASS needs `git submodule update --init third_party/cutlass`.
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
- **Throughput ratios are source-reported, not measured here.** 79a states NVFP4
  MMA at 2× MXFP8 and 4× Ada FP8. Measure with
  [hardware-unit-test](../../hardware-unit-test/SKILL.md) before sizing a gain.

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

- CUTLASS: `examples/87_blackwell_geforce_gemm_blockwise/` — 87a blockwise, 87b
  groupwise, 87c grouped; collective `sm120_mma_tma_blockwise_scaling.hpp`.
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
- FlashInfer: `fi:csrc/fp4_gemm_cutlass_sm120.cu` over
  `fi:include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`, whose
  `tilesSm120` list is the CTA-shape search space it autotunes over (cluster
  fixed 1×1×1); per-block and global scales are separate arguments.
- SGLang: `sgl:K/kda_kernels/qwen3x_nvfp4_gemm_sm120.py`, a CuTe DSL port of the
  CUTLASS block-scaled example specialised to Qwen3 decode shapes. Its header
  records it as optimised by Kernel Design Agents (`mit-han-lab/kernel-design-agents`).

### FP8 with per-32 UE8M0 block scales (MXFP8)

The block-scaled tensor-core path for FP8 (`kind::mxf8f6f4.block_scale`), an
alternative to software-applied block-128 scales.


- CUTLASS: 79c (MXFP8 × MXFP6 → BF16); `examples/80_blackwell_geforce_sparse_gemm/80a`.
- FlashInfer: `fi:csrc/mxfp8_gemm_cutlass_sm120.cu` over
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
  swizzled layout); Python in `fi:flashinfer/quantization/`.
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
