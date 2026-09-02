#pragma once

// Global -> shared.  Three mechanisms, chosen by what the caller has:
//   tma_load_*    a CUtensorMap and a full barrier; one elected lane issues,
//                 completion arrives on the barrier as complete_tx bytes.
//   bulk_load_1d  a contiguous 16-byte-aligned span, same completion path.
//   cp_async_16   one 16-byte chunk per calling thread, completion by
//                 cp_async_commit_group / cp_async_wait_group<N> in the
//                 issuing thread.
// Coordinates are element indices, innermost first ({c0, c1[, c2]}).

#include <cuda.h>
#include <cstdint>

#include <cute/arch/copy_sm90_desc.hpp>

#include "tile/sm90/barrier.cuh"
#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

// L2 residency hint on a TMA load.  kNone emits the plain instruction (the
// hardware default), the others the .L2::cache_hint form.  A one-shot
// stream (weights read once) takes kEvictFirst so it does not displace a
// reused working set; kEvictLast pins a tile that later tasks re-read.
enum class L2Hint : int { kNone = 0, kEvictNormal, kEvictFirst, kEvictLast };

constexpr uint64_t l2_hint_bits(L2Hint hint) {
  using cute::TMA::CacheHintSm90;
  return hint == L2Hint::kEvictFirst
             ? static_cast<uint64_t>(CacheHintSm90::EVICT_FIRST)
         : hint == L2Hint::kEvictLast
             ? static_cast<uint64_t>(CacheHintSm90::EVICT_LAST)
             : static_cast<uint64_t>(CacheHintSm90::EVICT_NORMAL);
}

template <L2Hint Hint = L2Hint::kNone>
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map,
                                            void* smem_dst, int32_t c0,
                                            int32_t c1, uint64_t* full_bar) {
  const uint32_t d = smem_u32(smem_dst), b = smem_u32(full_bar);
  if constexpr (Hint == L2Hint::kNone) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(b) : "memory");
  } else {
    constexpr uint64_t hint = l2_hint_bits(Hint);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
        ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(b), "l"(hint) : "memory");
  }
}

template <L2Hint Hint = L2Hint::kNone>
__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map,
                                            void* smem_dst, int32_t c0,
                                            int32_t c1, int32_t c2,
                                            uint64_t* full_bar) {
  const uint32_t d = smem_u32(smem_dst), b = smem_u32(full_bar);
  if constexpr (Hint == L2Hint::kNone) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
  } else {
    constexpr uint64_t hint = l2_hint_bits(Hint);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
        ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b), "l"(hint)
        : "memory");
  }
}

// Cluster multicast: every CTA whose bit is set in ctamask receives the box
// at the same shared offset and its own barrier at the same offset gets the
// complete_tx.  Issued by one CTA of the cluster.
__device__ __forceinline__ void tma_load_2d_multicast(const CUtensorMap* map,
                                                      void* smem_dst,
                                                      int32_t c0, int32_t c1,
                                                      uint64_t* full_bar,
                                                      uint16_t ctamask) {
  const uint32_t d = smem_u32(smem_dst), b = smem_u32(full_bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(b), "h"(ctamask) : "memory");
}

__device__ __forceinline__ void tma_load_3d_multicast(const CUtensorMap* map,
                                                      void* smem_dst,
                                                      int32_t c0, int32_t c1,
                                                      int32_t c2,
                                                      uint64_t* full_bar,
                                                      uint16_t ctamask) {
  const uint32_t d = smem_u32(smem_dst), b = smem_u32(full_bar);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
      ::"r"(d), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b), "h"(ctamask)
      : "memory");
}

// Pulls a box toward L2 with no shared-memory destination and no barrier.
// The later ordinary load still owns the transaction bytes.
__device__ __forceinline__ void tma_prefetch_2d_l2(const CUtensorMap* map,
                                                   int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];"
               ::"l"(map), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_prefetch_3d_l2(const CUtensorMap* map,
                                                   int32_t c0, int32_t c1,
                                                   int32_t c2) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.3d.L2.global [%0, {%1, %2, %3}];"
      ::"l"(map), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

// Makes the tensor map itself resident before the first load that uses it.
__device__ __forceinline__ void tma_prefetch_descriptor(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// Contiguous span; src, dst and bytes must be multiples of 16.
__device__ __forceinline__ void bulk_load_1d(void* smem_dst, const void* gsrc,
                                             uint32_t bytes,
                                             uint64_t* full_bar) {
  const uint32_t d = smem_u32(smem_dst), b = smem_u32(full_bar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      ::"r"(d), "l"(gsrc), "r"(bytes), "r"(b) : "memory");
}

// Per-thread 16-byte async copy that bypasses L1 (cg).  Completion is by
// group: commit after issuing a stage, wait<N> leaves N groups in flight.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               ::"r"(smem_u32(smem_dst)), "l"(gsrc) : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int Pending = 0>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Pending) : "memory");
}

// cp.async tile copy for a row-major (Rows x Cols, Cols contiguous) global
// tile into a swizzled smem tile: Threads threads each move one 16-byte
// chunk per step, laid out Threads / chunks_per_row rows by chunks_per_row.
// The destination swizzle is applied by CuTe from the smem tensor's layout.
template <class T, int Threads, int Cols>
__device__ __forceinline__ auto make_g2s_cp_async_copy() {
  constexpr int kChunkElems = 16 / static_cast<int>(sizeof(T));
  constexpr int kChunksPerRow = Cols / kChunkElems;
  static_assert(Cols % kChunkElems == 0, "a row is a whole number of 16-byte chunks");
  static_assert(Threads % kChunksPerRow == 0, "Threads must tile whole rows");
  using ThrLayout =
      cute::Layout<cute::Shape<cute::Int<Threads / kChunksPerRow>, cute::Int<kChunksPerRow>>,
                   cute::Stride<cute::Int<kChunksPerRow>, cute::_1>>;
  using ValLayout = cute::Layout<cute::Shape<cute::_1, cute::Int<kChunkElems>>>;
  return cute::make_tiled_copy(
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
      ThrLayout{}, ValLayout{});
}

// Every thread of the copy calls it; commit and wait with the group helpers.
template <class TiledCopy, class GTensor, class STensor>
__device__ __forceinline__ void copy_g2s(TiledCopy const& tiled_copy, int tid,
                                         GTensor const& gTile, STensor& sTile) {
  auto thr = tiled_copy.get_thread_slice(tid);
  auto src = thr.partition_S(gTile);
  auto dst = thr.partition_D(sTile);
  cute::copy(tiled_copy, src, dst);
}

// A whole operand tile as TMA boxes.  The tile is Inner x Outer elements
// with Inner contiguous in global memory; one box covers one swizzle span
// (SwizzleBytes) along Inner, and box i lands at smem + i * Outer * span so
// the image equals SmemTileLayout<T, Outer, Inner, K, SwizzleBytes> for a
// K-major operand (or the MN-major layout with Inner = MN).
template <class T, int Inner, int Outer, int SwizzleBytes = 128>
struct TmaTile2D {
  static constexpr int kInnerAtom =
      SwizzleBytes == 0 ? Inner : SwizzleBytes / static_cast<int>(sizeof(T));
  static constexpr int kBoxes = Inner / kInnerAtom;
  static constexpr uint32_t kBytes =
      static_cast<uint32_t>(Inner) * Outer * sizeof(T);
  static_assert(Inner % kInnerAtom == 0,
                "Inner must be a whole number of swizzle spans");
  static_assert(kInnerAtom * sizeof(T) <= 128,
                "a swizzled box row is at most 128 bytes");

  // The map's box must be {kInnerAtom, Outer}; see tma_host.cuh.
  template <L2Hint Hint = L2Hint::kNone>
  __device__ static __forceinline__ void load(const CUtensorMap* map, T* smem,
                                              int32_t inner_idx,
                                              int32_t outer_idx,
                                              uint64_t* full_bar) {
    CUTE_UNROLL
    for (int i = 0; i < kBoxes; ++i) {
      tma_load_2d<Hint>(map, smem + i * Outer * kInnerAtom,
                        inner_idx + i * kInnerAtom, outer_idx, full_bar);
    }
  }

  // Arms the full barrier for this tile's bytes; call once per tile before
  // (or in the same thread as) the loads.
  __device__ static __forceinline__ void expect(FullBarrier* full_bar) {
    full_bar->arrive_and_expect_tx(kBytes);
  }
};

}  // namespace flash_vla::sm90
