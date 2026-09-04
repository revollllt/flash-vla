// Template 01 -- TMA into an mbarrier ring buffer (sm90).
//
// The base of every sm90 async pipeline: a tensor map built on the host, a
// ring of shared frames, and two barrier families per slot -- full (the
// producer arms it with a byte count, the copy engine completes it) and empty
// (consumers release the frame for reuse).  Templates 02-04 sit on this.
//
// The idea to copy is the parity arithmetic: both barriers of a slot flip once
// per use, so the phase to wait on is derived from the loop index and never
// stored per slot.
//
// Structural only.  The PTX assertions prove the instructions survive codegen,
// not that the values are right; numerical authority is a parity harness.
//
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: mbarrier\.init\.shared::cta\.b64
// CHECK-PTX: mbarrier\.arrive\.expect_tx\.shared::cta\.b64
// CHECK-PTX: mbarrier\.try_wait\.parity\.shared::cta\.b64
// CHECK-PTX: fence\.mbarrier_init\.release\.cluster

#include <cuda_bf16.h>

#include "sm90_common.cuh"

namespace {

// One box row is 64 bf16 = 128 B = exactly one SW128 swizzle span, the widest
// a swizzled box row may be [tma.bytes.txn.dtype].  A wider K tile is more
// boxes, not a wider box.
constexpr int kTileRows = 64;
constexpr int kTileK = 64;
constexpr int kFrameBytes =
    kTileRows * kTileK * static_cast<int>(sizeof(__nv_bfloat16));

// Four stages covers the DRAM latency of a cold stream at this box size
// [tma.stages.warp.knee]; two is enough from L2.
constexpr int kDepth = 4;
constexpr int kThreads = 128;

}  // namespace

// Streams k_tiles tiles of one row block through the ring.  The reduction is a
// placeholder consumer -- template 03 puts a wgmma here.
__global__ __launch_bounds__(kThreads) void tma_ring_kernel(
    const __grid_constant__ CUtensorMap tmap, int32_t k_tiles,
    float* __restrict__ out) {
  __shared__ alignas(1024) __nv_bfloat16 frames[kDepth][kTileRows * kTileK];
  __shared__ alignas(8) uint64_t full[kDepth];
  __shared__ alignas(8) uint64_t empty[kDepth];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  if (tid == 0) {
    for (int32_t s = 0; s < kDepth; ++s) {
      tmpl::mbarrier_init(&full[s], 1);           // the one lane issuing the TMA
      tmpl::mbarrier_init(&empty[s], kThreads);   // every consumer releases
    }
  }
  tmpl::fence_barrier_init();
  __syncthreads();

  float acc = 0.f;
  for (int32_t stage = 0; stage < k_tiles; ++stage) {
    const int32_t slot = stage % kDepth;
    const uint32_t use = static_cast<uint32_t>(stage / kDepth);

    if (tid == 0) {
      // The first kDepth uses find the frame free; later ones wait for the
      // release of the previous use of this slot.
      if (use > 0) { tmpl::wait_parity(&empty[slot], (use - 1) & 1u); }
      tmpl::arrive_and_expect_tx(&full[slot], kFrameBytes);
      tmpl::tma_load_2d(&tmap, frames[slot], stage * kTileK,
                        static_cast<int32_t>(blockIdx.x) * kTileRows, &full[slot]);
    }

    tmpl::wait_parity(&full[slot], use & 1u);
    for (int32_t i = tid; i < kTileRows * kTileK; i += kThreads) {
      acc += __bfloat162float(frames[slot][i]);
    }
    // Per-thread release; the empty barrier IS the release.  A __syncthreads
    // here would collapse the ring to one stage in flight.
    tmpl::mbarrier_arrive(&empty[slot]);
  }

  out[blockIdx.x * kThreads + tid] = acc;
}

CUresult make_tile_map(CUtensorMap* map, const __nv_bfloat16* base, uint64_t k,
                       uint64_t rows) {
  return tmpl::encode_tile_map_2d(map, base, k, rows, kTileK, kTileRows,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  CU_TENSOR_MAP_SWIZZLE_128B);
}
