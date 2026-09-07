// Template 02 -- warp-specialized producer/consumer with register handoff (sm90).
//
// Splits the CTA into a copy role and a math role so the TMA column runs ahead
// of the math column instead of alternating with it.  Adds three rules to the
// ring of template 01:
//
//   1. The producer is a whole warpgroup, not a warp.  setmaxnreg is
//      `.aligned` -- every thread of the warpgroup must execute it -- so a lone
//      producer warp cannot hand its registers over at all.  Only warp 0 of the
//      group issues TMAs; the other three release and exit.
//   2. Roles that exit make __syncthreads unusable.  Surviving roles rendezvous
//      on a named barrier with an explicit thread count.
//   3. A second math warpgroup buys no tensor-core throughput
//      [wgmma.ratio.sm.wg2] -- add one to hide softmax/epilogue latency, never
//      to add FLOPs.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-GRADE: structural
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: elect\.sync
// CHECK-PTX: bar(?:rier)?\.sync
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global

#include <cuda_bf16.h>

#include "sm90_common.cuh"

namespace {

constexpr int kTileRows = 64;
constexpr int kTileK = 64;
constexpr int kFrameBytes =
    kTileRows * kTileK * static_cast<int>(sizeof(__nv_bfloat16));
constexpr int kDepth = 4;  // [tma.stages.warp.knee]

constexpr int kThreads = 2 * tmpl::kWarpgroupThreads;  // one math WG + one producer WG

// The producer only issues copies, so it keeps the minimum; the math group
// takes what it releases.  Two math warpgroups would take 232 each instead.
constexpr int kProducerRegs = 32;
constexpr int kMathRegs = 240;

// Id 0 is __syncthreads, which the exited producer warps can no longer join.
constexpr uint32_t kBarMath = 1;

}  // namespace

__global__ __launch_bounds__(kThreads) void warp_specialized_kernel(
    const __grid_constant__ CUtensorMap tmap, int32_t k_tiles,
    float* __restrict__ out) {
  __shared__ alignas(1024) __nv_bfloat16 frames[kDepth][kTileRows * kTileK];
  __shared__ alignas(8) uint64_t full[kDepth];
  __shared__ alignas(8) uint64_t empty[kDepth];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const bool is_math = tid < tmpl::kWarpgroupThreads;

  if (tid == 0) {
    for (int32_t s = 0; s < kDepth; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      // Only the math warpgroup consumes, so only it releases frames.
      tmpl::mbarrier_init(&empty[s], tmpl::kWarpgroupThreads);
    }
  }
  tmpl::fence_barrier_init();
  __syncthreads();  // the last point every role is still resident

  if (!is_math) {
    // Executed by all 128 producer threads before any of them leaves.
    tmpl::setmaxnreg_dec<kProducerRegs>();
    const int32_t warp_in_group =
        (tid - tmpl::kWarpgroupThreads) / tmpl::kWarpThreads;
    if (warp_in_group != 0) { return; }  // registers already released

    for (int32_t stage = 0; stage < k_tiles; ++stage) {
      const int32_t slot = stage % kDepth;
      const uint32_t use = static_cast<uint32_t>(stage / kDepth);
      if (use > 0) { tmpl::wait_parity(&empty[slot], (use - 1) & 1u); }
      if (tmpl::elect_one()) {
        tmpl::arrive_and_expect_tx(&full[slot], kFrameBytes);
        tmpl::tma_load_2d(&tmap, frames[slot], stage * kTileK,
                          static_cast<int32_t>(blockIdx.x) * kTileRows,
                          &full[slot]);
      }
    }
    return;
  }

  tmpl::setmaxnreg_inc<kMathRegs>();

  float acc = 0.f;
  for (int32_t stage = 0; stage < k_tiles; ++stage) {
    const int32_t slot = stage % kDepth;
    const uint32_t use = static_cast<uint32_t>(stage / kDepth);
    tmpl::wait_parity(&full[slot], use & 1u);
    for (int32_t i = tid; i < kTileRows * kTileK; i += tmpl::kWarpgroupThreads) {
      acc += __bfloat162float(frames[slot][i]);
    }
    tmpl::mbarrier_arrive(&empty[slot]);
  }

  // Math-only rendezvous: the producer warps are gone, so this must name its
  // own barrier and its own thread count rather than call __syncthreads.
  tmpl::named_barrier_sync(kBarMath, tmpl::kWarpgroupThreads);
  out[blockIdx.x * tmpl::kWarpgroupThreads + tid] = acc;
}
