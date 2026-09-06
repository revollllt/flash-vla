#pragma once

#include <cstdint>

namespace flash_vla::pi05::sm90::ffn {

// Typed views over the shared barrier pool.  The pool is deliberately aliased
// between task kinds; naming the regions here prevents the old `bars + 8`
// arithmetic from leaking into every task body.
template <typename FullBarrier, typename EmptyBarrier,
          int GatedUpDepth, int DownResidualWeightDepth,
          int DownResidualActivationDepth>
struct BarrierViews {
  static_assert(sizeof(FullBarrier) == sizeof(uint64_t));
  static_assert(sizeof(EmptyBarrier) == sizeof(uint64_t));

  __device__ static FullBarrier* gated_up_full(uint64_t* base) {
    return reinterpret_cast<FullBarrier*>(base);
  }

  __device__ static EmptyBarrier* gated_up_empty(uint64_t* base) {
    return reinterpret_cast<EmptyBarrier*>(base + GatedUpDepth);
  }

  __device__ static FullBarrier* down_residual_weight_full(uint64_t* base) {
    return reinterpret_cast<FullBarrier*>(base);
  }

  __device__ static EmptyBarrier* down_residual_weight_empty(uint64_t* base) {
    return reinterpret_cast<EmptyBarrier*>(base + DownResidualWeightDepth);
  }

  __device__ static FullBarrier* down_residual_activation_full(uint64_t* base) {
    return reinterpret_cast<FullBarrier*>(base + 2 * DownResidualWeightDepth);
  }

  __device__ static EmptyBarrier* down_residual_activation_empty(uint64_t* base) {
    return reinterpret_cast<EmptyBarrier*>(
        base + 2 * DownResidualWeightDepth + DownResidualActivationDepth);
  }
};

}  // namespace flash_vla::pi05::sm90::ffn
