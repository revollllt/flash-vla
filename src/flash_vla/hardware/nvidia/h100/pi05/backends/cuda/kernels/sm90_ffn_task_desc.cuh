#pragma once

#include <cstdint>
#include <type_traits>

namespace flash_vla::pi05::sm90::ffn {

// The host planner stores these fields as four int32 values.  Keep the binary
// ABI stable while making task meaning explicit in device code.
enum class TaskKind : int32_t {
  kGatedUp = 0,
  kDownResidual = 1,
  kSentinel = -1,
};

struct TaskDescriptor {
  TaskKind kind;
  int32_t column;
  int32_t dependency;
  int32_t split;
};

static_assert(std::is_same_v<std::underlying_type_t<TaskKind>, int32_t>);
static_assert(sizeof(TaskDescriptor) == 4 * sizeof(int32_t));
static_assert(alignof(TaskDescriptor) == alignof(int32_t));

__host__ __device__ constexpr bool is_sentinel(TaskKind kind) {
  return kind == TaskKind::kSentinel;
}

__host__ __device__ constexpr bool is_gated_up(TaskKind kind) {
  return kind == TaskKind::kGatedUp;
}

__host__ __device__ constexpr bool is_down_residual(TaskKind kind) {
  return kind == TaskKind::kDownResidual;
}

}  // namespace flash_vla::pi05::sm90::ffn
