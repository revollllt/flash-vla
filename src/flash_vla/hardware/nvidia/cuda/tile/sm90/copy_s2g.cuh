#pragma once

// Shared -> global through the async proxy, issued by one elected lane.
// Contract before the issue: every writer of the smem source has executed
// fence_proxy_async_shared() and the writers are synchronized with the
// issuing lane.  Completion is by bulk group: store_commit_group() after
// the issues, then store_wait_group<N>() (global visible) or
// store_wait_group_read<N>() (only the smem source is reusable).

#include <cuda.h>
#include <cstdint>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map,
                                             const void* smem_src, int32_t c0,
                                             int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      ::"l"(map), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map,
                                             const void* smem_src, int32_t c0,
                                             int32_t c1, int32_t c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      ::"l"(map), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}

// Contiguous span; src, dst and bytes must be multiples of 16.
__device__ __forceinline__ void bulk_store_1d(void* gdst, const void* smem_src,
                                              uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(gdst), "r"(smem_u32(smem_src)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Global writes of all but the newest Pending groups are complete.
template <int Pending = 0>
__device__ __forceinline__ void store_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(Pending) : "memory");
}

// Only the smem source of all but the newest Pending groups has been read;
// enough when nothing in this kernel consumes the destination.
template <int Pending = 0>
__device__ __forceinline__ void store_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(Pending) : "memory");
}

}  // namespace flash_vla::sm90
