// Raw sm90 primitives shared by the templates in this directory.
//
// Toolkit-only on purpose: the choreography stays visible instead of hiding
// behind a library.  Production kernels in this repo call
// tile/sm90/{common,barrier,copy_g2s,copy_s2g}.cuh instead, which own the same
// instructions with fuller coverage; this header exists so a template compiles
// standalone and so the PTX is readable next to the rule it demonstrates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tmpl {

inline constexpr int kWarpThreads = 32;
inline constexpr int kWarpgroupThreads = 128;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- barriers

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               ::"r"(smem_u32(bar)), "r"(arrivals));
}

// Makes the inits visible to the async proxy, which is a different proxy from
// the generic stores that wrote them.  Without it a TMA can complete against a
// barrier the copy engine does not yet see.
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Arms a slot with exactly the bytes the copies it covers will deliver.  A
// wrong count hangs the consumer: the phase never completes.
__device__ __forceinline__ void arrive_and_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               ::"r"(smem_u32(bar)) : "memory");
}

// Returned as the selp word rather than bool so ptxas keeps the spin's loop
// shape instead of rotating it and re-issuing try_wait per site.
__device__ __forceinline__ uint32_t try_wait_parity(const uint64_t* bar, uint32_t phase) {
  uint32_t done = 0;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(phase));
  return done;
}

__device__ __forceinline__ void wait_parity(const uint64_t* bar, uint32_t phase) {
  while (try_wait_parity(bar, phase) == 0) {}
}

// Named barrier over a subset of the CTA.  Id 0 is __syncthreads; a kernel
// whose producer warps exit early must rendezvous its remaining roles on
// 1..15 with an explicit thread count, never on __syncthreads.
__device__ __forceinline__ void named_barrier_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void named_barrier_arrive(uint32_t id, uint32_t threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ------------------------------------------------------------------ fences

// Orders this thread's generic-proxy shared writes before a later async-proxy
// read of the same bytes (a TMA store, or a wgmma descriptor read of a tile
// written with st.shared).
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ------------------------------------------------------------------- roles

// One lane per warp, chosen by hardware.  The elected lane issues TMAs and
// barrier arrivals so the async proxy sees one request per warp.
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n .reg .b32 rx;\n .reg .pred px;\n"
      " elect.sync rx|px, %1;\n"
      " selp.u32 %0, 1, 0, px;\n}"
      : "=r"(pred)
      : "r"(0xffffffffu));
  return pred != 0;
}

// Register redistribution between warpgroups.  Both forms are `.aligned`:
// every thread of the warpgroup must execute them, so a lone producer *warp*
// cannot use them -- make the producer a whole warpgroup and let its unused
// warps release and exit.
template <int RegCount>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(RegCount));
}

template <int RegCount>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(RegCount));
}

// --------------------------------------------------------------------- pdl
//
// Use the CUDA intrinsics, not hand-rolled asm.  They ARE these instructions --
// cuda_device_runtime_api.h defines them as exactly this PTX -- but they carry
// the compiler barriers NVIDIA intends, and the two differ ASYMMETRICALLY:
//
//   cudaGridDependencySynchronize()           -> griddepcontrol.wait  ::: "memory"
//   cudaTriggerProgrammaticLaunchCompletion() -> griddepcontrol.launch_dependents ::: (none)
//
// The wait is an acquire and must stop the compiler hoisting a producer-data
// load above it.  The trigger deliberately is NOT a barrier: it must not stop
// the compiler moving independent work across it, because overlapping that work
// is the entire point.  CUTLASS's own helpers spell them the same way.
//
// WHAT EACH ONE ACTUALLY PROMISES
//
// The wait "blocks the thread until all direct grid dependencies have
// completed" -- that is where the memory ordering comes from, and it is the
// same ordering ordinary stream serialization would give.
//
// The trigger publishes nothing.  It only lets the dependent be SCHEDULED
// earlier, so its pre-wait prologue can run while this kernel is still going.
// CUTLASS states the consequence outright at its own call site: "the timing of
// calling this function only influences performance, not functional
// correctness".  So:
//
//   * No release fence belongs before it.  The dependent's wait does that job.
//     (The header's "provides no memory visibility guarantee itself" is about
//     a dependent that reads producer data BEFORE its own wait -- do not.)
//   * Trigger EARLY.  There is no data-readiness point to respect, so the only
//     question is how early the dependent may start competing for SMs.
//     CUTLASS triggers on the last mainloop tile, well before its epilogue
//     writes anything.
//   * A trigger on the LAST LINE is close to a no-op: the kickoff already
//     happens automatically once every CTA has exited, so it fires when the
//     automatic path would have anyway.
//
// The wait is the opposite: put it as LATE as possible, immediately before the
// first read of producer data.  Index arithmetic, model parameters, scheduling
// metadata and descriptor prefetch all belong above it.
//
// How early to trigger and how late to wait are tuning, not derivation --
// wiki/pdl-placement.md carries the ablation.

__device__ __forceinline__ void pdl_wait() { cudaGridDependencySynchronize(); }

__device__ __forceinline__ void pdl_trigger() {
  cudaTriggerProgrammaticLaunchCompletion();
}

// ----------------------------------------------------------------- cluster

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

// Maps a CTA-local shared address into another CTA of the cluster (DSMEM).
__device__ __forceinline__ uint32_t map_shared_rank(const void* p, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(out)
               : "r"(smem_u32(p)), "r"(rank));
  return out;
}

// Arrives on a barrier owned by another CTA of the cluster.  A consumer
// releasing a multicast frame must arrive on EVERY receiving CTA's barrier,
// not only its own.
__device__ __forceinline__ void mbarrier_arrive_cluster(const void* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               ::"r"(map_shared_rank(bar, rank)) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// -------------------------------------------------------------------- copy

// Coordinates are element indices, innermost dimension first.
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map, void* dst,
                                            int32_t c0, int32_t c1, uint64_t* full) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(full))
      : "memory");
}

// A 3-D map turns a per-expert / per-batch weight stack into one descriptor:
// the outermost coordinate selects the slice, so a grouped kernel needs one
// tensor map rather than one per group.
__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map, void* dst,
                                            int32_t c0, int32_t c1, int32_t c2,
                                            uint64_t* full) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
        "r"(smem_u32(full))
      : "memory");
}

// Every CTA whose bit is set in ctamask receives the box at the same shared
// offset, and each one's barrier at the same offset takes the complete_tx.
// One CTA of the cluster issues for all of them, so the bytes cross the
// interconnect once instead of once per CTA.
__device__ __forceinline__ void tma_load_2d_multicast(const CUtensorMap* map, void* dst,
                                                      int32_t c0, int32_t c1,
                                                      uint64_t* full, uint16_t ctamask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(full)),
        "h"(ctamask)
      : "memory");
}

// cp.async: the pre-Hopper async copy, still the right tool when the source
// layout is a pre-permuted blob rather than a tensor tile.  A TMA needs a
// tensor map and a rectangular box; a weight matrix permuted offline so each
// thread's 16 bytes ARE its MMA operand has neither.  16 bytes per thread is
// the widest form and the only one that bypasses L1 with .cg.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               ::"r"(smem_u32(smem_dst)), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until at most N groups remain in flight.  Unlike a TMA's mbarrier this
// is per-thread state, so every thread that issued must also wait.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// One ldmatrix moves four 8x8 tiles into the register layout mma.sync expects,
// transposing lanes for free.  The address is per-lane: lane l supplies the
// row it wants, which is why callers compute a swizzled per-lane offset.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&out)[4], const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
               : "r"(smem_u32(smem)));
}

// Shared -> global through the async proxy.  Completion is a bulk group, not
// an mbarrier: the issuing thread commits and waits.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, const void* src,
                                             int32_t c0, int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
      ::"l"(map), "r"(c0), "r"(c1), "r"(smem_u32(src))
      : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Waits until at most N bulk groups remain in flight.  Read-completion only:
// the shared frame is reusable, the global write may still be landing.
template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// ------------------------------------------------------------- host helper

// Dims and box are innermost first; strides are bytes and exclude the
// contiguous dimension.  The swizzle must equal the one the consumer reads
// through.  Not CUDA-Graph-capture safe: build maps before capture.
template <class T>
inline CUresult encode_tile_map_2d(CUtensorMap* map, const T* base, uint64_t inner,
                                   uint64_t outer, uint32_t box_inner,
                                   uint32_t box_outer, CUtensorMapDataType dtype,
                                   CUtensorMapSwizzle swizzle) {
  uint64_t dims[2] = {inner, outer};
  uint64_t strides[1] = {inner * sizeof(T)};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t elem_strides[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      map, dtype, 2, const_cast<T*>(base), dims, strides, box, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace tmpl
