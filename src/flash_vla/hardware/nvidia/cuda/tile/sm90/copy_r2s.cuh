#pragma once

// Register accumulator -> shared tile.  Two paths:
//   stmatrix  for 16-bit outputs: the copy is derived from the MMA so the
//             accumulator fragment lands row-major in the (swizzled) smem
//             tile without a shuffle; the tile can then go out by TMA store.
//   scalar    element-wise stores for any type (f32 partials).
// convert_fragment() turns the f32 accumulator into the output type in the
// same fragment layout, so both paths take the converted fragment.

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

// f32 accumulator -> TOut fragment, layout preserved, round-to-nearest.
// Adapted from FlashAttention-3 hopper/utils.h (convert_type_safe).
template <class TOut, class Acc>
__device__ __forceinline__ auto convert_fragment(Acc const& acc) {
  using TIn = typename Acc::value_type;
  constexpr int kN = decltype(cute::size(acc))::value;
  cutlass::NumericArrayConverter<TOut, TIn, kN> convert;
  const auto& in =
      *reinterpret_cast<const cutlass::Array<TIn, kN>*>(acc.data());
  auto out = convert(in);
  return cute::make_tensor(
      cute::make_rmem_ptr<TOut>(reinterpret_cast<TOut*>(&out)),
      acc.layout());
}

// A wgmma accumulator covers 64 x N and takes the x4 atom (16 x 16 per
// atom); an mma.sync accumulator is 16 x 8 per atom and takes x2.
template <class TOut, class TiledMma>
using StsmCopyAtom = cute::Copy_Atom<
    cute::conditional_t<is_wgmma_v<TiledMma>, cute::SM90_U32x4_STSM_N,
                        cute::SM90_U32x2_STSM_N>,
    TOut>;

template <class TOut, class TiledMma>
__device__ __forceinline__ auto make_r2s_copy_C(TiledMma const& mma) {
  static_assert(sizeof(TOut) == 2, "stmatrix is a 16-bit operation");
  return cute::make_tiled_copy_C(StsmCopyAtom<TOut, TiledMma>{}, mma);
}

template <class TOut, class TiledMma>
__device__ __forceinline__ auto make_r2s_copy_C_scalar(TiledMma const& mma) {
  return cute::make_tiled_copy_C(
      cute::Copy_Atom<cute::UniversalCopy<TOut>, TOut>{}, mma);
}

// rFrag is the (converted) accumulator fragment of this thread; sTile the
// whole CTA tile.  Every thread of the MMA calls it.
template <class TiledCopy, class RTensor, class STensor>
__device__ __forceinline__ void copy_r2s(TiledCopy const& tiled_copy, int tid,
                                         RTensor const& rFrag, STensor& sTile) {
  auto thr = tiled_copy.get_thread_slice(tid);
  auto src = thr.retile_S(rFrag);
  auto dst = thr.partition_D(sTile);
  cute::copy(tiled_copy, src, dst);
}

}  // namespace flash_vla::sm90
