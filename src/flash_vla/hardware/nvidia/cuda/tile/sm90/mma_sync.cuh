#pragma once

// Warp-level mma.sync atoms with f32 accumulation, both operands in
// registers, K-major (TN).  16-bit inputs: m16n8k16; fp8 inputs: m16n8k32.
// On sm_90a ptxas lowers the fp8 mma.sync to F2FP unpack plus f16 HMMA
// (observed in SASS): numerically exact, but the fp8 tensor-core rate on
// Hopper is only reachable through wgmma.
// The selector in gemm.cuh picks these for warp-shaped tiles or when B is
// staged in registers; they are also the right tool when a tile is too
// small to keep a warpgroup MMA busy [hardware-unit-test wgmma.issue.wg.ss].

#include <cute/tensor.hpp>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

template <class TA, class TB>
struct MmaSyncAtom {
  static_assert(dependent_false_v<TA>,
                "no mma.sync atom for this (A, B) element pair");
};

#define FLASH_VLA_MMA_SYNC_ATOM(TA_, TB_, ATOM_, K_)          \
  template <>                                                 \
  struct MmaSyncAtom<TA_, TB_> {                              \
    using type = cute::ATOM_;                                 \
    static constexpr int kM = 16;                             \
    static constexpr int kN = 8;                              \
    static constexpr int kK = K_;                             \
  };

FLASH_VLA_MMA_SYNC_ATOM(BF16, BF16, SM80_16x8x16_F32BF16BF16F32_TN, 16)
FLASH_VLA_MMA_SYNC_ATOM(F16, F16, SM80_16x8x16_F32F16F16F32_TN, 16)
FLASH_VLA_MMA_SYNC_ATOM(E4M3, E4M3, SM89_16x8x32_F32E4M3E4M3F32_TN, 32)
FLASH_VLA_MMA_SYNC_ATOM(E4M3, E5M2, SM89_16x8x32_F32E4M3E5M2F32_TN, 32)
FLASH_VLA_MMA_SYNC_ATOM(E5M2, E4M3, SM89_16x8x32_F32E5M2E4M3F32_TN, 32)
FLASH_VLA_MMA_SYNC_ATOM(E5M2, E5M2, SM89_16x8x32_F32E5M2E5M2F32_TN, 32)

#undef FLASH_VLA_MMA_SYNC_ATOM

// WarpsM x WarpsN warps tile the CTA tile; each warp owns a 16 x 8 atom
// repeated over its share.
template <class TA, class TB, int WarpsM = 1, int WarpsN = 1>
using MmaSyncTiledMma = decltype(cute::make_tiled_mma(
    typename MmaSyncAtom<TA, TB>::type{},
    cute::Layout<cute::Shape<cute::Int<WarpsM>, cute::Int<WarpsN>, cute::_1>>{}));

}  // namespace flash_vla::sm90
