#pragma once

// Shared vocabulary for the SM90 tile primitives: element types, operand
// location and major-ness tags, and the thread-role and fence helpers that
// every copy and MMA header builds on.  Kernels include the primitive they
// need (copy_g2s.cuh, gemm.cuh, ...) or the tile/sm90/sm90.cuh umbrella.

#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

namespace flash_vla::sm90 {

using BF16 = cutlass::bfloat16_t;
using F16 = cutlass::half_t;
using E4M3 = cutlass::float_e4m3_t;
using E5M2 = cutlass::float_e5m2_t;

// Where an MMA operand lives when the instruction issues.  wgmma reads B from
// shared memory always and A from either place; mma.sync reads both from
// registers.  The selector in gemm.cuh turns these tags into an atom.
enum class Operand : int { kSmem = 0, kReg = 1 };

// Major-ness of a shared-memory operand as the tensor core sees it.  K-major
// means the contraction index is contiguous; MN-major means the row (A) or
// column (B) index is contiguous.  Aliased to CuTe's enum so atoms take it
// directly.
using Major = cute::GMMA::Major;

inline constexpr int kWarpThreads = 32;
inline constexpr int kWarpgroupThreads = 128;

template <class T>
inline constexpr bool dependent_false_v = false;

// True when a TiledMma is a Hopper warpgroup MMA: its B fragment is a smem
// descriptor iterator rather than a register array.  gemm() and the
// stmatrix atom choice in copy_r2s.cuh branch on it.
template <class TiledMma>
inline constexpr bool is_wgmma_v =
    cute::is_base_of<cute::GMMA::DescriptorIterator,
                     typename TiledMma::FrgTypeB>::value;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int lane_id() {
  return static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
}

__device__ __forceinline__ int warp_id() {
  return static_cast<int>(threadIdx.x) / kWarpThreads;
}

__device__ __forceinline__ int warpgroup_id() {
  return static_cast<int>(threadIdx.x) / kWarpgroupThreads;
}

// One lane of the calling warp; the elected lane issues TMA and barrier
// operations so the async proxy sees one request per warp.
__device__ __forceinline__ bool elect_one() {
  return cute::elect_one_sync() != 0;
}

// Orders this thread's generic-proxy global accesses before its later
// async-proxy (TMA) accesses.  Needed after an acquire of data another CTA
// published with ordinary stores and before the first TMA that reads it.
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// Orders this thread's generic-proxy shared-memory writes before a later
// async-proxy read of the same bytes (a TMA or bulk store, or a wgmma
// descriptor read of a tile written with st.shared).
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Makes mbarrier.init visible to the async proxy before any TMA targets it.
__device__ __forceinline__ void fence_barrier_init() {
  cutlass::arch::fence_barrier_init();
}

// Named barrier over a subset of the CTA (id 0 is __syncthreads; use 1..15).
__device__ __forceinline__ void named_barrier_sync(uint32_t id,
                                                   uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void named_barrier_arrive(uint32_t id,
                                                     uint32_t threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

}  // namespace flash_vla::sm90
