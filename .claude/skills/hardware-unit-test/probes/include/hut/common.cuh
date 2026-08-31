// hut/common.cuh -- primitives every unit needs, and nothing a unit owns.
//
// Names follow references/vocabulary.md: anything naming a machine quantity
// carries the driver API's, PTX's or CUTLASS's word for it. Anything invented
// here is listed in that file's "ours" table.

#pragma once

#include <cuda.h>
#include <cstdint>

namespace hut {

// Generic shared-memory address for an inline-PTX operand. PTX addresses shared
// memory in a separate 32-bit window; this is the conversion, not a cast.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int32_t warp_id() {
  return static_cast<int32_t>(threadIdx.x >> 5);
}

__device__ __forceinline__ bool is_lane_zero() {
  return (threadIdx.x & 31u) == 0u;
}

// The SM this CTA landed on. Occupancy CAPACITY is not placement -- the
// scheduler spreads a grid before it stacks two CTAs on one SM -- so a per-SM
// claim has to READ this rather than infer it. [sched.ctas.sm.knee]
__device__ __forceinline__ int32_t sm_id() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return static_cast<int32_t>(s);
}

}  // namespace hut
