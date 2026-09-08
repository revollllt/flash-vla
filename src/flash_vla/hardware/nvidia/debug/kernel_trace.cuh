#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Diagnostic software timestamps, NOT a hardware-engine activity counter.
// Header-only prototype. No implicit block/warp barrier or async wait is added.
namespace flash_trace {
struct Stamp {
  std::uint64_t ns;
  std::uint32_t sm;
};

__device__ __forceinline__ Stamp read_stamp() {
  Stamp s;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s.ns) :: "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s.sm));
  return s;
}
// These reads are adjacent observations, not an atomic (time, SM) snapshot.
// 'memory' constrains compiler memory motion; it is NOT a GPU memory fence,
// an all-instruction scheduling fence, or a TMA/WGMMA completion operation.

struct alignas(8) RangeRecord {
  std::uint64_t begin_ns, end_ns;
  std::uint32_t begin_sm, end_sm;
  std::uint32_t cta, warp, stage, valid;
};
static_assert(sizeof(RangeRecord) == 40, "Record layout changed");

// Exactly ONE predetermined writer owns each slot. The host only consumes
// records after the containing kernel has completed. This is not a concurrent
// host-reader publication protocol. Slot ownership must NOT be based on SM ID.
__device__ __forceinline__ void store_range(
    RangeRecord* slot, Stamp begin, Stamp end,
    std::uint32_t cta, std::uint32_t warp, std::uint32_t stage) {
  *slot = RangeRecord{begin.ns, end.ns, begin.sm, end.sm, cta, warp, stage, 1};
}
}  // namespace flash_trace
