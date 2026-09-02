#pragma once

// mbarrier vocabulary for producer/consumer rings.  A full barrier is armed by
// the producer with the byte count of the TMA it issues and completes when the
// bytes land; an empty barrier is arrived by consumers when a frame may be
// overwritten.  Both are one 64-bit word in shared memory.

#include <cstdint>

#include <cutlass/arch/barrier.h>

#include "tile/sm90/common.cuh"

namespace flash_vla::sm90 {

// Producer: arrive_and_expect_tx(bytes); TMA: complete_tx; consumer: wait(phase).
using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
// Consumer: arrive(); producer: wait(phase).
using EmptyBarrier = cutlass::arch::ClusterBarrier;

static_assert(sizeof(FullBarrier) == sizeof(uint64_t));
static_assert(sizeof(EmptyBarrier) == sizeof(uint64_t));

// Non-blocking parity test, 1 when the phase has completed, else 0; the
// caller owns the spin (and any watchdog).  Returned as the selp word rather
// than bool so a `while (!done)` spin keeps its loop shape (measured: a bool
// return made ptxas rotate the loop and add a try_wait per site).
__device__ __forceinline__ uint32_t mbarrier_try_wait_parity(const uint64_t* bar,
                                                             uint32_t phase) {
  uint32_t done = 0;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(phase));
  return done;
}

// Per-slot parity for a ring consumed in order.  take(slot) returns the
// parity to wait on and flips it, so the same call site serves every
// reuse of the slot.  One instance per waiting role, in registers.
template <int Depth>
struct PhaseRing {
  uint32_t phase[Depth] = {};

  __device__ __forceinline__ uint32_t take(int slot) {
    const uint32_t p = phase[slot];
    phase[slot] ^= 1u;
    return p;
  }
};

}  // namespace flash_vla::sm90
