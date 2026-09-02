#pragma once

// Shared-memory tile layouts that a TMA box, an ldmatrix/stmatrix, and a
// wgmma descriptor all agree on.  The atom is CuTe's GMMA swizzle atom; the
// tile is the atom repeated over (MN, K).  A TMA with the matching
// CU_TENSOR_MAP_SWIZZLE_* lands its box in exactly this image, so no
// shared-memory rewrite sits between the copy and the MMA.

#include <cute/tensor.hpp>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

// Swizzle width in bytes for a contiguous run of InnerElems elements: the
// widest of 128/64/32 that the run fills exactly, else 0 (no swizzle).
template <class T, int InnerElems>
constexpr int swizzle_bytes_for() {
  constexpr int bytes = InnerElems * static_cast<int>(sizeof(T));
  return bytes % 128 == 0 ? 128 : bytes % 64 == 0 ? 64 : bytes % 32 == 0 ? 32 : 0;
}

template <class T, Major major, int SwizzleBytes>
struct SwizzleAtom {
  static_assert(dependent_false_v<T>,
                "SwizzleBytes must be 0, 32, 64 or 128");
};

template <class T> struct SwizzleAtom<T, Major::K, 0> {
  using type = cute::GMMA::Layout_K_INTER_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::K, 32> {
  using type = cute::GMMA::Layout_K_SW32_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::K, 64> {
  using type = cute::GMMA::Layout_K_SW64_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::K, 128> {
  using type = cute::GMMA::Layout_K_SW128_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::MN, 0> {
  using type = cute::GMMA::Layout_MN_INTER_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::MN, 32> {
  using type = cute::GMMA::Layout_MN_SW32_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::MN, 64> {
  using type = cute::GMMA::Layout_MN_SW64_Atom<T>;
};
template <class T> struct SwizzleAtom<T, Major::MN, 128> {
  using type = cute::GMMA::Layout_MN_SW128_Atom<T>;
};

// A (Rows = MN, Cols = K) operand tile.  Order decides which mode the atoms
// walk first: Step<_1,_2> (default) fills all rows of one K-atom before the
// next K-atom, which is how consecutive TMA boxes along K land; Step<_2,_1>
// walks K first and matches a producer that places boxes along MN.
template <class T, int Rows, int Cols, Major major = Major::K,
          int SwizzleBytes = 128,
          class Order = cute::Step<cute::_1, cute::_2>>
struct SmemTileLayout {
  using Element = T;
  using Atom = typename SwizzleAtom<T, major, SwizzleBytes>::type;
  using type = decltype(cute::tile_to_shape(
      Atom{}, cute::Shape<cute::Int<Rows>, cute::Int<Cols>>{}, Order{}));

  static constexpr int kRows = Rows;
  static constexpr int kCols = Cols;
  static constexpr Major kMajor = major;
  static constexpr int kSwizzleBytes = SwizzleBytes;
  // Elements covered by one swizzle span along the contiguous mode; a TMA
  // box may not exceed it, so a wider tile is several boxes.
  static constexpr int kInnerElems =
      SwizzleBytes == 0 ? (major == Major::K ? Cols : Rows)
                        : SwizzleBytes / static_cast<int>(sizeof(T));
  static constexpr int kBytes = Rows * Cols * static_cast<int>(sizeof(T));

  static_assert((major == Major::K ? Cols : Rows) % kInnerElems == 0,
                "the contiguous extent must be a whole number of swizzle spans");
};

}  // namespace flash_vla::sm90
