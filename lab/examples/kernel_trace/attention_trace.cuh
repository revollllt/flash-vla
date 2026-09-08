#pragma once
#include "flash_vla/hardware/nvidia/debug/kernel_trace.cuh"

namespace attention_trace {
constexpr int kWarps = 12, kSlots = 256;
__device__ flash_trace::RangeRecord* records;
__device__ int detail;

__device__ __forceinline__ bool selected(bool observer, int stage) {
  return observer && records && ((detail == 1 && stage == 7) ||
                                 (detail == 2 && blockIdx.x == 0));
}
__device__ __forceinline__ flash_trace::Stamp begin(bool observer, int stage) {
  return selected(observer, stage) ? flash_trace::read_stamp() : flash_trace::Stamp{};
}
__device__ __forceinline__ void end(flash_trace::Stamp stamp, bool observer,
                                   int iteration, int stage) {
  if (selected(observer, stage)) {
    const unsigned warp = threadIdx.x / 32;
    const unsigned slot = stage == 7 ? kSlots - 1 : iteration * 8 + stage;
    flash_trace::store_range(records + (blockIdx.x * kWarps + warp) * kSlots + slot,
                             stamp, flash_trace::read_stamp(), blockIdx.x, warp, stage);
  }
}
}
#define TRACE_BEGIN(name, observer, stage) auto name = attention_trace::begin(observer, stage)
#define TRACE_END(name, observer, iteration, stage) attention_trace::end(name, observer, iteration, stage)

extern "C" int enc_attn_trace_config(void* records, int detail) {
  cudaError_t status = cudaMemcpyToSymbol(attention_trace::records, &records, sizeof(records));
  if (status != cudaSuccess) return static_cast<int>(status);
  return static_cast<int>(cudaMemcpyToSymbol(attention_trace::detail, &detail, sizeof(detail)));
}
