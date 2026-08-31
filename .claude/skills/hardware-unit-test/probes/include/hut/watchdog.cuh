// hut/watchdog.cuh -- the ONE bounded wait in this library.
//
// protocol.md rule 9: a persistent probe hangs rather than fails unless every
// wait carries a deadline. That rule was written after a wrong ring index cost
// an hour of Slurm time, and was then forgotten anyway -- pipeline_ws.cu
// shipped with no watchdog at all and burned two allocations for zero output,
// because its waits were inside CUTLASS and writing a bounded one looked like
// extra work.
//
// So this header exports `spin_until` and nothing else. Every wait in
// barrier.cuh, tma.cuh and pipeline.cuh is built on it, and a unit that wants
// to wait has no unbounded primitive available to reach for.

#pragma once

#include <cstdint>

namespace hut {

// ~1.2 s at 1.8 GHz -- long enough that a slow configuration is never mistaken
// for a hang, short enough that a hang does not consume the allocation.
constexpr long long kWatchdogCycles = 1ll << 31;

// Trap sites. The library owns [1, kSiteUnitBase); a unit numbers its own from
// kSiteUnitBase so a library wait reports the same site whichever unit called
// it, and the host can name it without knowing which unit is running.
enum : int32_t {
  kSiteBarrierWait  = 1,   // waiting for a transaction barrier to flip
  kSiteBarrierDrain = 2,   // draining outstanding transfers before exit
  kSiteCounterPoll  = 3,   // polling a global counter or flag
  kSiteProducerAcq  = 4,   // pipeline producer_acquire (empty barrier)
  kSiteConsumerWait = 5,   // pipeline consumer_wait (full barrier)
  kSiteProducerTail = 6,   // pipeline producer_tail
  kSiteUnitBase     = 16,
};

__device__ __forceinline__ void trap_at(long long* dbg, int32_t site) {
  if (dbg != nullptr) {
    dbg[blockIdx.x * 2] = site;
    dbg[blockIdx.x * 2 + 1] = static_cast<long long>(threadIdx.x);
    __threadfence_system();
  }
  __trap();
}

// Spin until `done()` is true, or trap reporting `site`. `done` is a callable
// returning bool; it is re-evaluated every iteration.
template <class Done>
__device__ __forceinline__ void spin_until(Done done, long long* dbg,
                                           int32_t site) {
  // The predicate sits in the LOOP CONDITION, and that is measured, not
  // stylistic. Spelling it as `ready = done()` tested by a variable makes ptxas
  // emit one fewer YIELD and a slower loop, in BOTH units that poll:
  //
  //   gmem_atomic hop  661.7 ns predicate-in-condition | 680.1 variable-tested
  //   tma_ring ns/txn  257.8    predicate-in-condition | 279.4 variable-tested
  //
  // Both were measured against the shipped build in one job, on every row, in
  // one direction. A ceiling probe wants the faster loop: a poll that costs
  // more reports the thing it is waiting for as slower than it is. [rule 2]
  const long long t0 = clock64();
  while (!done()) {
    if (clock64() - t0 > kWatchdogCycles) trap_at(dbg, site);
  }
}

}  // namespace hut
