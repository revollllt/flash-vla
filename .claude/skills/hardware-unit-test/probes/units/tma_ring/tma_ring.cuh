// tma_ring.cuh -- isolate TMA delivery rate from everything else.
//
// The FFN task-loop confounds six things at once (CTA count, stage count,
// bytes per box, box geometry, wgmma-retirement coupling, counter polling).
// This unit strips all of them but the copy engine: N producer warps per CTA,
// each owning a PRIVATE `stages`-deep ring of box_bytes-sized boxes. No wgmma,
// no counters, no consumer warp. The only thing measured is how fast the
// machine delivers TMA boxes.
//
// mode 0 = rate, mode 1 = the correctness gate (see hut_check).

#pragma once

#include <cstdint>

#include "hut/barrier.cuh"
#include "hut/common.cuh"
#include "hut/tma.cuh"
#include "hut/unit.hpp"
#include "hut/watchdog.cuh"

namespace hut {
namespace tma_ring {

constexpr int32_t kMaxStages = 16;   // PhaseRing holds one bit per stage
constexpr int32_t kMaxWarps = 8;

extern __shared__ uint8_t smem_pool[];

// Coordinate walk, per issue:
//   coord_0 = (idx & mask0) * step0             along the fastest dim
//   coord_1 = ((idx >> shift0) & mask1) * step1 along the strided dim
// The host guarantees mask0+1 and mask1+1 are powers of two and
// shift0 == log2(mask0+1), so the whole walk is shifts and masks. A runtime
// integer divide on the issue path would put the divider in the measurement
// instead of the copy engine. A contiguous descriptor passes mask0 = 0.
__device__ __forceinline__ void walk(int32_t idx, const HutParams& p,
                                     int32_t& coord_0, int32_t& coord_1) {
  coord_0 = (idx & p.mask0) * p.step0;
  coord_1 = ((idx >> p.shift0) & p.mask1) * p.step1;
}

__global__ __launch_bounds__(kMaxWarps * 32, 1)
void rate_kernel(const __grid_constant__ CUtensorMap tensor_map,
                 HutParams p, long long* __restrict__ dbg,
                 int32_t* __restrict__ sm_id_out) {
  const int32_t warp = warp_id();
  // Read back from the launch geometry rather than taken from p, so a
  // host/device disagreement about the producer count shows up here.
  const int32_t num_producers = static_cast<int32_t>(blockDim.x) >> 5;

  if (threadIdx.x == 0 && sm_id_out != nullptr) sm_id_out[blockIdx.x] = sm_id();

  auto* bars = reinterpret_cast<TransactionBarrier*>(
      smem_pool + static_cast<size_t>(num_producers) * p.stages * p.box_bytes);
  init_barriers(bars, num_producers * p.stages);

  uint8_t* my_pool =
      smem_pool + static_cast<size_t>(warp) * p.stages * p.box_bytes;
  TransactionBarrier* my_bars = bars + warp * p.stages;
  PhaseRing ring;

  // Each (CTA, warp) walks a disjoint stream, so the sweep measures delivery
  // rather than L2 broadcast of one hot address.
  const int32_t base = static_cast<int32_t>(blockIdx.x) * num_producers + warp;
  const bool lane_zero = is_lane_zero();      // hoisted out of the issue loop

  // Every parameter the issue loop reads is copied into a local FIRST.
  //
  // HutParams arrives in param space, and `walk(idx, p, ...)` taking it by
  // const reference left ptxas re-loading mask0 from the constant bank inside
  // the loop -- `ULDC UR7, c[0x0][0x32c]` sitting directly on the issue path.
  // That cost 13% of the issue interval (249 -> 282 ns/txn) and is invisible in
  // the instruction COUNT, which is 328 either way. The uniform ABI is worth
  // keeping; paying for it per iteration is not. [protocol.md rule 12]
  const int32_t k_tile_count = p.k_tile_count;
  const int32_t stages = p.stages;
  const int32_t box_bytes = p.box_bytes;
  const int32_t mask0 = p.mask0, shift0 = p.shift0, step0 = p.step0;
  const int32_t mask1 = p.mask1, step1 = p.step1;

  for (int32_t k = 0; k < k_tile_count; ++k) {
    const int32_t stage = k % stages;
    if (k >= stages) {
      wait(&my_bars[stage], ring.phase_of(stage), dbg);
      ring.flip(stage);
    }
    __syncwarp();
    const int32_t idx = base * k_tile_count + k;
    const int32_t coord_0 = (idx & mask0) * step0;
    const int32_t coord_1 = ((idx >> shift0) & mask1) * step1;
    if (lane_zero) {
      arrive_and_expect_tx(&my_bars[stage], box_bytes);
      cp_async_bulk_tensor_2d(&tensor_map,
                              my_pool + static_cast<size_t>(stage) * box_bytes,
                              coord_0, coord_1, &my_bars[stage]);
    }
  }

  // Drain: an outstanding TMA writing shared memory the CTA has retired is UB.
  const int32_t tail = k_tile_count < stages ? k_tile_count : stages;
  for (int32_t j = 0; j < tail; ++j) {
    const int32_t stage = (k_tile_count - tail + j) % stages;
    wait(&my_bars[stage], ring.phase_of(stage), dbg, kSiteBarrierDrain);
    ring.flip(stage);
  }
  __syncthreads();
}

// A rate measured on boxes nobody verified is a measurement of an unknown: a
// wrong descriptor, box geometry or coordinate walk delivers the WRONG bytes at
// the RIGHT speed, every row still looks reasonable, and the constants are
// folklore. [protocol.md rule 11]
//
// This replays rate_kernel's OWN walk at one stage -- one CTA, one warp, so
// `base` is 0 and `idx` is `k` -- and copies each delivered box out for the
// host to compare against the source. It also reports the coordinates it used,
// so a host/device disagreement about the walk fails here rather than becoming
// a hardware claim.
__global__ __launch_bounds__(32, 1)
void check_kernel(const __grid_constant__ CUtensorMap tensor_map, HutParams p,
                  uint4* __restrict__ out, int32_t* __restrict__ coords,
                  long long* __restrict__ dbg) {
  // Same base as rate_kernel, so the destination carries rate_kernel's
  // alignment: were smem_pool under-aligned for TMA, both would fail, not one.
  auto* bar = reinterpret_cast<TransactionBarrier*>(smem_pool + p.box_bytes);
  init_barriers(bar, 1);

  const int32_t n_vec = p.box_bytes / 16;
  const auto* src = reinterpret_cast<const uint4*>(smem_pool);
  PhaseRing ring;

  for (int32_t k = 0; k < p.k_tile_count; ++k) {
    int32_t coord_0, coord_1;
    walk(k, p, coord_0, coord_1);
    if (is_lane_zero()) {
      coords[k * 2] = coord_0;
      coords[k * 2 + 1] = coord_1;
      arrive_and_expect_tx(bar, p.box_bytes);
      cp_async_bulk_tensor_2d(&tensor_map, smem_pool, coord_0, coord_1, bar);
    }
    __syncwarp();
    wait(bar, ring.phase_of(0), dbg);
    ring.flip(0);
    // Drain the box before the next issue overwrites it. The trailing
    // __syncwarp is what makes that ordering hold, not the loop structure.
    for (int32_t i = threadIdx.x; i < n_vec; i += 32)
      out[static_cast<size_t>(k) * n_vec + i] = src[i];
    __syncwarp();
  }
}

}  // namespace tma_ring
}  // namespace hut
