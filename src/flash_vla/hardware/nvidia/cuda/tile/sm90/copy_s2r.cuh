#pragma once

// Shared -> register fragments through ldmatrix.  The tiled copy is derived
// from the MMA so the register image is exactly the fragment the MMA reads:
// the A operand of an RS wgmma, or either operand of mma.sync.  The smem
// tile keeps its swizzle; ldmatrix addresses through it and no rewrite
// happens.  ldmatrix moves 16-bit rows, so 8-bit operands work as packed
// pairs in K-major only.

#include <cute/tensor.hpp>

#include <type_traits>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

// K-major operands load with the non-transposed x4 form; MN-major 16-bit
// operands use the transposing form so the fragment is still K-major.
template <class T, Major major = Major::K>
using LdsmCopyAtom = cute::Copy_Atom<
    cute::conditional_t<major == Major::K, cute::SM75_U32x4_LDSM_N,
                        cute::SM75_U16x8_LDSM_T>,
    T>;

template <class T, Major major = Major::K, class TiledMma>
__device__ __forceinline__ auto make_s2r_copy_A(TiledMma const& mma) {
  static_assert(major == Major::K || sizeof(T) == 2,
                "ldmatrix.trans is a 16-bit operation");
  return cute::make_tiled_copy_A(LdsmCopyAtom<T, major>{}, mma);
}

template <class T, Major major = Major::K, class TiledMma>
__device__ __forceinline__ auto make_s2r_copy_B(TiledMma const& mma) {
  static_assert(major == Major::K || sizeof(T) == 2,
                "ldmatrix.trans is a 16-bit operation");
  return cute::make_tiled_copy_B(LdsmCopyAtom<T, major>{}, mma);
}

// rFrag is the MMA's register fragment (thr_mma.partition_fragment_A/B of the
// smem tensor); the copy retiles it to ldmatrix granularity.  Pass the
// (_,_,k) slices of both tensors to load one K block.
template <class TiledCopy, class ThrCopy, class STensor, class RTensor,
          class = std::enable_if_t<!std::is_integral_v<ThrCopy>>>
__device__ __forceinline__ void copy_s2r(TiledCopy const& tiled_copy,
                                         ThrCopy const& thr_copy,
                                         STensor const& sTile, RTensor& rFrag) {
  auto src = thr_copy.partition_S(sTile);
  auto dst = thr_copy.retile_D(rFrag);
  cute::copy(tiled_copy, src, dst);
}

template <class TiledCopy, class STensor, class RTensor>
__device__ __forceinline__ void copy_s2r(TiledCopy const& tiled_copy, int tid,
                                         STensor const& sTile, RTensor& rFrag) {
  copy_s2r(tiled_copy, tiled_copy.get_thread_slice(tid), sTile, rFrag);
}

}  // namespace flash_vla::sm90
