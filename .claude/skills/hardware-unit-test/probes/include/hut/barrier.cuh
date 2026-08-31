// hut/barrier.cuh -- mbarrier operations, every wait bounded.
//
// Wraps CUTLASS's ClusterTransactionBarrier so the transaction accounting is
// the library's rather than ours, and routes every wait through
// watchdog.cuh::spin_until. There is deliberately no unbounded `wait` exported.

#pragma once

#include <cutlass/arch/barrier.h>

#include "hut/common.cuh"
#include "hut/watchdog.cuh"

namespace hut {

// CUTLASS's name for a barrier that counts BOTH arrivals and transaction bytes,
// which is what a TMA completion needs. [cutlass/arch/barrier.h]
using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;

__device__ __forceinline__ void init_barriers(TransactionBarrier* bar,
                                              int32_t count,
                                              int32_t arrive_count = 1) {
  if (threadIdx.x == 0) {
    for (int32_t i = 0; i < count; ++i) bar[i].init(arrive_count);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
}

// Poll a barrier whose shared-memory ADDRESS is already known.
//
// The address is a parameter rather than computed here so the cvta.to.shared
// stays out of the spin. Measured neutral on tma_ring sweep G -- it was tried
// as the suspect for a post-migration regression and did not move it, so it is
// kept on the merits (one fewer instruction in a poll loop), not as a fix.
// Do not cite it as one.
__device__ __forceinline__ bool try_wait_at(uint32_t bar_addr, uint32_t phase) {
  uint32_t done = 0;
  asm volatile("{\n .reg .pred p;\n"
               " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               " selp.u32 %0, 1, 0, p;\n}"
               : "=r"(done) : "r"(bar_addr), "r"(phase));
  return done != 0;
}

__device__ __forceinline__ bool try_wait(TransactionBarrier* bar,
                                         uint32_t phase) {
  return try_wait_at(smem_addr(bar), phase);
}

// Wait for `bar` to leave `phase`. Bounded; traps reporting `site` on timeout.
//
// The predicate sits in the LOOP CONDITION. That is a measured choice, not a
// style one: ptxas schedules the two spellings differently, and neither
// reproduces both units' pre-migration numbers.
//
//   tma_ring ns/txn (absolute constant tma.issue.warp)
//       pre-migration 249.2 | predicate-in-condition 257.8 (+3.5%)
//                           | variable-tested        279.4 (+12.1%)
//   overlap tma_iso (only a TERM of the ratio overlap.eff.sm)
//       pre-migration 123439 | predicate-in-condition -5.0% (an extra YIELD per
//                              wait site, which deprioritises the spinning warp)
//                            | variable-tested        -0.0%
//
// The tie-break is what each unit's CONSTANT is. tma_ring's is an absolute, so
// it has to land inside the 6% floor -- +3.5% does, +12.1% does not. overlap's
// is a ratio and read 1.248 under BOTH shapes, because its two terms move
// together; only its absolute moved. So the shape is chosen for the unit that
// cannot absorb it, and overlap is documented as indifferent.
//
// If a future unit reports an absolute that this shape penalises, that is a
// decision to re-open, not a constant to quietly re-record.
// [protocol.md rules 2 and 12]
__device__ __forceinline__ void wait(TransactionBarrier* bar, uint32_t phase,
                                     long long* dbg,
                                     int32_t site = kSiteBarrierWait) {
  const uint32_t addr = smem_addr(bar);
  const long long t0 = clock64();
  while (!try_wait_at(addr, phase)) {
    if (clock64() - t0 > kWatchdogCycles) trap_at(dbg, site);
  }
}

// Arrive and declare the bytes this barrier is waiting for, in one operation --
// PTX `mbarrier.arrive.expect_tx`. Must precede the transfer that completes it.
__device__ __forceinline__ void arrive_and_expect_tx(TransactionBarrier* bar,
                                                     int32_t transaction_bytes) {
  bar->arrive_and_expect_tx(static_cast<uint32_t>(transaction_bytes));
}

// A ring of `stages` barriers whose phase bits live in ONE register.
//
// The obvious alternative, `uint32_t phase[kMaxStages]` indexed by the runtime
// stage, lands in LOCAL memory -- a non-zero stack frame with zero spills --
// and every wait in the timed loop then pays a round trip through L1TEX. That
// cost 8% of tma.issue.warp before it was found by reading -Xptxas -v output
// rather than the probe's own numbers. [protocol.md rule 12]
struct PhaseRing {
  uint32_t bits = 0;
  __device__ __forceinline__ uint32_t phase_of(int32_t stage) const {
    return (bits >> stage) & 1u;
  }
  __device__ __forceinline__ void flip(int32_t stage) {
    bits ^= 1u << stage;
  }
};

}  // namespace hut
