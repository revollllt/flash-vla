// coop_launch.cuh -- what a device-wide barrier costs, against the relaunch it
// replaces.
//
// `launch.lat.dev.ramp` says every kernel starts ~1.24 us in debt, and the
// decision to reject cooperative launch for this repo's decoder rested on the
// ASSUMPTION that a grid barrier costs about the same. That assumption is
// marked `[I, UNMEASURED]` and is load-bearing: if a grid_sync is much cheaper
// than a relaunch, a persistent kernel with N barriers beats N launches.
//
// Everything here is the vendor's API, not a hand-rolled equivalent:
// `cooperative_groups::this_grid()`, `grid_group::sync()`, and
// `cudaLaunchCooperativeKernel` -- which is the ONLY legal launch for a kernel
// that calls grid_sync, and which refuses a grid too large to be co-resident
// instead of deadlocking. That refusal is a result to record, not an error to
// hide. [references/vocabulary.md, "Cooperative launch"]

#pragma once

#include <cooperative_groups.h>

#include "hut/common.cuh"
#include "hut/unit.hpp"

namespace cg = cooperative_groups;

namespace hut {
namespace coop_launch {

// mode selects what the timed loop contains. The two share one kernel, one
// launch path and one grid, so the barrier's cost is the DIFFERENCE between
// them and nothing else moved. [protocol.md rule 3]
enum : int32_t {
  MODE_GRID_SYNC = 0,   // n_iters x grid_sync()
  MODE_EMPTY     = 1,   // n_iters x the same loop, no barrier
};

// A dependent chain so the loop cannot be optimised away and so MODE_EMPTY
// measures loop overhead rather than nothing. One FMA per iteration is far
// below the barrier cost and is subtracted out by the pair anyway.
__device__ __forceinline__ float spin_work(float x) { return fmaf(x, 1.000001f, 1e-7f); }

__global__ void grid_sync_kernel(int32_t n_iters, int32_t mode,
                                 long long* __restrict__ cycles,
                                 int32_t* __restrict__ sm_id_out,
                                 float* __restrict__ sink) {
  cg::grid_group grid = cg::this_grid();

  if (threadIdx.x == 0 && sm_id_out != nullptr) sm_id_out[blockIdx.x] = sm_id();

  // Warm the barrier once before timing: the first grid_sync on a launch pays
  // arrival of the last block, which is grid ramp, not barrier cost.
  grid.sync();

  float x = 1.0f;
  const long long t0 = clock64();
  for (int32_t i = 0; i < n_iters; ++i) {
    x = spin_work(x);
    if (mode == MODE_GRID_SYNC) grid.sync();
  }
  const long long t1 = clock64();

  // Block 0 lane 0 owns the span: every block is inside the same barrier, so
  // one clock is the grid's clock. Per-block spans are still written so the
  // host can check they agree.
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (x == 1234.5678f) sink[blockIdx.x] = x;   // never true; keeps the chain live

  // A kernel that calls grid_sync must leave every block having called it the
  // same number of times, which the uniform loop bound guarantees.
}

// The relaunch baseline: the SAME loop body, no barrier, launched once per
// "iteration" by the host. What a grid_sync competes against is not zero -- it
// is this. Timed by the host across n_iters launches.
__global__ void relaunch_kernel(int32_t n_iters, float* __restrict__ sink) {
  float x = 1.0f;
  for (int32_t i = 0; i < n_iters; ++i) x = spin_work(x);
  if (x == 1234.5678f) sink[blockIdx.x] = x;
}

}  // namespace coop_launch
}  // namespace hut
