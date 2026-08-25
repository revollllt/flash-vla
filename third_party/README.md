# Vendored CUDA kernel references

These checkouts are read-only reference sources for the SM90 action-expert
kernel work. They are shallow clones of the official repositories; each
checkout retains its upstream LICENSE and .git metadata so the exact revision
can be audited locally.

| component | upstream | pinned revision | upstream submodules | license |
| --- | --- | --- | --- | --- |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | `15f13e5030374295491c5ce31b02d7e63a7772c6` | CUTLASS `147f5673d0c1c3dcf66f78d677fd647e4a020219` | `flashmla/LICENSE` |
| DeepGEMM | https://github.com/deepseek-ai/DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` | `deepgemm/LICENSE` |

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

The references are not included in the FFN build yet. New code should copy
only the small helper or contract needed, preserve upstream attribution, and
keep repository-specific task descriptors and dependency protocols outside
the vendored trees.
