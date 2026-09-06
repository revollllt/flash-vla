#pragma once

namespace flash_vla::pi05::sm90::ffn {

// This profile uses one 128-thread math warpgroup and two single-warp TMA
// producers.  The final warp is intentionally reserved so the role contract
// remains explicit when the task body grows.
struct WarpRoles {
  static constexpr int kMathBegin = 0;
  static constexpr int kMathThreads = 128;
  static constexpr int kWeightProducerBegin = 128;
  static constexpr int kActivationProducerBegin = 160;
  static constexpr int kReservedBegin = 192;
  static constexpr int kThreads = 224;

  __device__ static constexpr bool is_math(int tid) {
    return tid < kMathThreads;
  }

  __device__ static constexpr bool is_weight_producer(int tid) {
    return tid >= kWeightProducerBegin && tid < kActivationProducerBegin;
  }

  __device__ static constexpr bool is_activation_producer(int tid) {
    return tid >= kActivationProducerBegin && tid < kReservedBegin;
  }

  __device__ static constexpr bool is_reserved(int tid) {
    return tid >= kReservedBegin && tid < kThreads;
  }

  __device__ static constexpr int warp_id(int tid) { return tid >> 5; }
};

static_assert(WarpRoles::kThreads == 7 * 32);

}  // namespace flash_vla::pi05::sm90::ffn
