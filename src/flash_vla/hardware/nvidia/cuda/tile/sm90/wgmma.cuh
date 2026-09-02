#pragma once

// The complete Hopper warpgroup MMA table with f32 accumulation, keyed by
// (A type, B type, N, where A lives, A major, B major):
//   bf16/f16   64 x N x 16, N = 8..256 step 8, SS and RS, any major pair
//   fp8        64 x N x 32, N = 8..256 step 8, SS and RS, K-major only (TN),
//              every e4m3/e5m2 pairing
// B is always read from shared memory.  Every entry names a CuTe atom, so
// MMA_Traits, partitioning and descriptors come from CuTe unchanged.
// Modelled on DeepGEMM mma/sm90.cuh (BF16MMASelector / FP8MMASelector).

#include <cute/tensor.hpp>
// The N not in {8,16,32,64,96,128,192,256} live in the extended tables.
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/atom/mma_traits_sm90_gmma_ext.hpp>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

template <class TA, class TB, int N, Operand LocA, Major MajA, Major MajB>
struct WgmmaAtom {
  static_assert(dependent_false_v<TA>,
                "no wgmma atom for this (A, B, N, A location, majors) tuple: "
                "N must be a multiple of 8 in [8, 256]; fp8 needs K-major A and B");
};

#define FLASH_VLA_WGMMA_N_LIST(X)                                          \
  X(8) X(16) X(24) X(32) X(40) X(48) X(56) X(64) X(72) X(80) X(88) X(96)   \
  X(104) X(112) X(120) X(128) X(136) X(144) X(152) X(160) X(168) X(176)    \
  X(184) X(192) X(200) X(208) X(216) X(224) X(232) X(240) X(248) X(256)

// 16-bit inputs: SS takes both majors; RS reads A from registers K-major.
#define FLASH_VLA_WGMMA_16BIT(N_, T_, TAG_)                                 \
  template <Major MajA, Major MajB>                                         \
  struct WgmmaAtom<T_, T_, N_, Operand::kSmem, MajA, MajB> {                \
    using type = cute::SM90::GMMA::MMA_64x##N_##x16_F32##TAG_##TAG_##_SS<   \
        MajA, MajB>;                                                        \
    static constexpr int kM = 64, kN = N_, kK = 16;                         \
  };                                                                        \
  template <Major MajA, Major MajB>                                         \
  struct WgmmaAtom<T_, T_, N_, Operand::kReg, MajA, MajB> {                 \
    static_assert(MajA == Major::K,                                         \
                  "a register A fragment is K-major by construction");      \
    using type = cute::SM90::GMMA::MMA_64x##N_##x16_F32##TAG_##TAG_##_RS<   \
        MajA, MajB>;                                                        \
    static constexpr int kM = 64, kN = N_, kK = 16;                         \
  };

#define FLASH_VLA_WGMMA_BF16(N_) FLASH_VLA_WGMMA_16BIT(N_, BF16, BF16)
#define FLASH_VLA_WGMMA_F16(N_) FLASH_VLA_WGMMA_16BIT(N_, F16, F16)

FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_BF16)
FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_F16)

// fp8 inputs: hardware supports K-major operands only.
#define FLASH_VLA_WGMMA_FP8(N_, TA_, TB_, TAGA_, TAGB_)                      \
  template <>                                                                \
  struct WgmmaAtom<TA_, TB_, N_, Operand::kSmem, Major::K, Major::K> {       \
    using type =                                                             \
        cute::SM90::GMMA::MMA_64x##N_##x32_F32##TAGA_##TAGB_##_SS_TN<>;      \
    static constexpr int kM = 64, kN = N_, kK = 32;                          \
  };                                                                         \
  template <>                                                                \
  struct WgmmaAtom<TA_, TB_, N_, Operand::kReg, Major::K, Major::K> {        \
    using type =                                                             \
        cute::SM90::GMMA::MMA_64x##N_##x32_F32##TAGA_##TAGB_##_RS_TN<>;      \
    static constexpr int kM = 64, kN = N_, kK = 32;                          \
  };

#define FLASH_VLA_WGMMA_E4M3_E4M3(N_) FLASH_VLA_WGMMA_FP8(N_, E4M3, E4M3, E4M3, E4M3)
#define FLASH_VLA_WGMMA_E4M3_E5M2(N_) FLASH_VLA_WGMMA_FP8(N_, E4M3, E5M2, E4M3, E5M2)
#define FLASH_VLA_WGMMA_E5M2_E4M3(N_) FLASH_VLA_WGMMA_FP8(N_, E5M2, E4M3, E5M2, E4M3)
#define FLASH_VLA_WGMMA_E5M2_E5M2(N_) FLASH_VLA_WGMMA_FP8(N_, E5M2, E5M2, E5M2, E5M2)

FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_E4M3_E4M3)
FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_E4M3_E5M2)
FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_E5M2_E4M3)
FLASH_VLA_WGMMA_N_LIST(FLASH_VLA_WGMMA_E5M2_E5M2)

#undef FLASH_VLA_WGMMA_E4M3_E4M3
#undef FLASH_VLA_WGMMA_E4M3_E5M2
#undef FLASH_VLA_WGMMA_E5M2_E4M3
#undef FLASH_VLA_WGMMA_E5M2_E5M2
#undef FLASH_VLA_WGMMA_FP8
#undef FLASH_VLA_WGMMA_BF16
#undef FLASH_VLA_WGMMA_F16
#undef FLASH_VLA_WGMMA_16BIT
#undef FLASH_VLA_WGMMA_N_LIST

// The instruction N for a tile N: the tile itself up to 256, else the
// widest table entry that divides it (the tiled MMA repeats the atom).
// The widest legal N is also the fastest per FLOP [wgmma.issue.wg.ss].
template <int TileN>
constexpr int wgmma_atom_n() {
  static_assert(TileN % 8 == 0 && TileN > 0, "wgmma N is a multiple of 8");
  if constexpr (TileN <= 256) {
    return TileN;
  } else {
    int n = 256;
    while (TileN % n != 0) n -= 8;
    return n;
  }
}

// One warpgroup, one atom repeated over the tile by CuTe's partitioning.
template <class TA, class TB, int TileN, Operand LocA = Operand::kSmem,
          Major MajA = Major::K, Major MajB = Major::K>
using WgmmaTiledMma = decltype(cute::make_tiled_mma(
    typename WgmmaAtom<TA, TB, wgmma_atom_n<TileN>(), LocA, MajA, MajB>::type{}));

}  // namespace flash_vla::sm90
