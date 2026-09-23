# Vendored CUDA dependencies and references

Submodule checkouts, plus vendored copies of pinned sparse checkouts under
`quant-references/`. The top-level `.gitmodules` records the official URLs;
`git clone --recurse-submodules` (or `git submodule update --init`) restores
the pinned revisions and their upstream LICENSE files.

CUTLASS is the one build dependency: the CUDA backends and the SM90 tile
primitives (`src/flash_vla/hardware/nvidia/cuda/tile/`) include CuTe from
`third_party/cutlass/include`. The build wrappers default `CUTLASS_DIR` to
this checkout; set the variable to point at another tree. FlashMLA and
DeepGEMM are read-only reference sources and are not on any include path.

agent-gpu-skills is neither: it is an agent skill bundle, surfaced through four
relative symlinks in `.agents/skills/` (`cuda-skill`, `cutlass-skill`,
`tilelang-skill`, `triton-skill`) so both skill entries expose them without a
second copy of their text. Three of the four index upstream source trees that
its own `.gitignore` excludes, so after `submodule update` run its
`update-repos.sh`, which makes sparse shallow checkouts; point `CUTLASS_REPO` at
this directory's `cutlass` to avoid fetching a third copy.

| component | upstream | pinned revision | upstream submodules | license |
| --- | --- | --- | --- | --- |
| CUTLASS | https://github.com/NVIDIA/cutlass | `v4.7.1` (latest stable release at pin time) | none required for the header-only CuTe build | `cutlass/LICENSE.txt` |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | `15f13e5030374295491c5ce31b02d7e63a7772c6` | CUTLASS `147f5673d0c1c3dcf66f78d677fd647e4a020219` | `flashmla/LICENSE` |
| DeepGEMM | https://github.com/deepseek-ai/DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` | `deepgemm/LICENSE` |
| agent-gpu-skills | https://github.com/slowlyC/agent-gpu-skills | `ae02d076fd424f3c134a5738a0e5cc5f28e747c3` | none; its own `third_party/` is gitignored and rebuilt by `update-repos.sh` | `agent-gpu-skills/LICENSE` (MIT) |

## Quantization references

`quant-references/` holds read-only, committed copies of parts of vLLM,
SGLang and FlashInfer. Scope is FP8 and NVFP4 on SM120's native low-bit tensor
cores: only their FP8/NVFP4 GEMM, quantize, scale-layout and quantization-config
paths are kept, plus each LICENSE — 893 text files, about 12 MB (2.2 MB
compressed). Weight-only kernels (Marlin, Machete, GPTQ, AWQ, AllSpark) and
SM90-only W4A8 are left out; mixed precision is added when FP8 and NVFP4 are
stable and faster. They are copies rather than submodules because the full
repositories are 94–380 MB with history, of which this is the relevant part.
`quant-references/update.sh` records each pin and regenerates the copies from a
sparse checkout at that commit; run it only to move a pin, and it refuses a tag
that has moved. Like FlashMLA and DeepGEMM, these are reading material and are on no
include path. The SM120 quantization entries in CUTLASS (examples 79, 80, 87, 91;
the `sm120_*` collectives) come from the existing submodule.
[kernel-wiki's SM120 source map](../.agents/skills/kernel-wiki/references/quantization-sm120.md)
says where each scheme lives and what does not run on SM120.

| component | upstream | pinned revision | fetched paths | license |
| --- | --- | --- | --- | --- |
| vLLM | https://github.com/vllm-project/vllm | `v0.30.0`, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `csrc/libtorch_stable/quantization` without its weight-only and W4A8 kernels, `csrc/quantization`, `csrc/cutlass_extensions`, `csrc/core`, `vllm/model_executor/layers/quantization` | Apache-2.0 |
| SGLang | https://github.com/sgl-project/sglang | `v0.5.20`, `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` | under `python/sglang/kernels/`: `jit/csrc/gemm` without Marlin and AWQ, `jit/include`, `kda_kernels`, `aot/csrc/gemm`, `aot/csrc/cutlass_extensions`, `ops/gemm`, `ops/diffusion`; and `python/sglang/srt/layers/quantization` | Apache-2.0; `kda_kernels/` files carry BSD-3-Clause headers |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | `v0.7.0`, `4d75a33f19aaf48b44d5b1c5dbca33bc1eca5c58` | `csrc/cute_sm12x_gemm`, `csrc/nv_internal`, `csrc/nvfp4_attention_sm120`, the quantized GEMM and quantize entry files directly in `csrc/`, `include/flashinfer/gemm`, `include/flashinfer/attention/sm120`, `flashinfer/gemm`, `flashinfer/quantization`, `flashinfer/jit/gemm`, `flashinfer/cute_dsl/{,add_}rmsnorm_fp4quant.py` | Apache-2.0 |

## Reuse map for the SM90 CuTe kernels

* FlashMLA `csrc/sm90/helpers.h` is the compact reference for a CuTe GEMM
  wrapper: fence register operands, warpgroup_arrive, issue an explicitly
  unrolled cute::gemm, commit, wait, and fence the accumulator. Its gemm_ss
  and gemm_rs variants show the shared/shared and register/shared contracts.
  TMA copies are centralized in launch_tma_copy, with the descriptor slice
  always taken from cute::_0{}.
* FlashMLA SM90 `decode/*/traits.h` and `components/helpers.h` show the naming
  and ownership convention for TMABarrier, aligned shared arrays, per-stage
  barriers, and producer-only TMA issue. The dense decode path is the closest
  persistent warp-specialized attention example.
* DeepGEMM `deep_gemm/include/deep_gemm/common/tma_copy.cuh` is the reusable
  TMA wrapper. It makes the inner atom and swizzle compile-time parameters,
  supports 2-D/3-D and SM90 multicast, and keeps barrier arrival beside the
  copy. Use this naming (BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, barrier_ptr,
  smem_ptr) for new helpers.
* DeepGEMM `deep_gemm/include/deep_gemm/mma/sm90.cuh` centralizes WGMMA
  selection (BF16MMASelector, FP8MMASelector), descriptor construction, and
  K/M/N constants. This is a better home for FFN GEMM geometry than open-coded
  instruction aliases in a kernel body.
* DeepGEMM `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` is the
  end-to-end SM90 pattern: one TMA warp-group, one or more math warp-groups,
  explicit register reconfiguration, full/empty ClusterTransactionBarrier
  rings, persistent Scheduler::get_next_block, and a TMA-store epilogue.
  `sm90_fp8_gemm_1d2d.cuh` is the matching scaled GEMM variant.
* DeepGEMM `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` provides the
  persistent tile naming (m_block_idx, n_block_idx, current_iter) and L2-aware
  swizzle. Its (++current_iter) * kNumSMs + blockIdx.x mapping is safe to
  borrow only when the runtime scheduler is intentional; the FFN offline
  planner should replace it with a precomputed per-CTA descriptor list while
  keeping the same coordinate names.
* DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp` is the reference for a
  machine-profile-driven choice of block sizes, swizzle modes, stage count,
  shared-memory budget, warp counts, and wave-efficiency scoring.

The references are not build dependencies. The reusable contracts they
demonstrate (the WGMMA fence/arrive/commit choreography, the TMA box
splitting by swizzle span, the instruction selector tables) live in
`src/flash_vla/hardware/nvidia/cuda/tile/` with upstream attribution; new
code composes that library rather than copying from the vendored trees, and
keeps repository-specific task descriptors and dependency protocols outside
them.
