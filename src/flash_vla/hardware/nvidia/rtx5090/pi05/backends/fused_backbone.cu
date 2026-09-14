// Pi0.5 backbone: contiguous BF16 rows of width 2048, FP32 nonlinear stages.
// The gate/up GEMMs round to BF16 before GELU; only the final product rounds
// again. out is the up input and output of backbone_gelu_mul.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int32_t kWidth = 2048;
constexpr int32_t kThreads = kWidth / 8;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void rms_norm(const __nv_bfloat16* __restrict__ input,
                         __nv_bfloat16* __restrict__ output) {
  const int64_t vector = int64_t(blockIdx.x) * kThreads + threadIdx.x;
  int4 raw = reinterpret_cast<const int4*>(input)[vector];
  __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(&raw);
  float squared = 0.f;
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const float x = __bfloat162float(values[i]);
    squared += x * x;
  }
  squared = warp_sum(squared);
  __shared__ float warps[kThreads / 32];
  __shared__ float inverse_rms;
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = squared;
  // Warp 0 consumes one partial sum written by each warp.
  __syncthreads();
  if (threadIdx.x < 32) {
    squared = threadIdx.x < kThreads / 32 ? warps[threadIdx.x] : 0.f;
    squared = warp_sum(squared);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(squared / kWidth + 1e-6f);
  }
  // Every warp consumes the row factor produced by lane 0.
  __syncthreads();
#pragma unroll
  for (int32_t i = 0; i < 8; ++i)
    values[i] = __float2bfloat16_rn(__bfloat162float(values[i]) * inverse_rms);
  reinterpret_cast<int4*>(output)[vector] = raw;
}

template <bool Masked>
__global__ void gelu_mul(const __nv_bfloat16* __restrict__ gate,
                         __nv_bfloat16* __restrict__ up_output,
                         int64_t vectors, const __nv_bfloat16* mask) {
  int64_t active_vectors = vectors;
  if constexpr (Masked)
    if (__bfloat162float(mask[896]) < 0.f)
      active_vectors = int64_t(896) * 16384 / 8;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors; index += int64_t(gridDim.x) * blockDim.x) {
    if constexpr (Masked) {
      // A long replay can leave finite nonzero tail data. Reapplying GELU/product
      // on stale padding can overflow, so short replays overwrite it without reads.
      if (index >= active_vectors) {
        reinterpret_cast<int4*>(up_output)[index] = make_int4(0, 0, 0, 0);
        continue;
      }
    }
    int4 gate_raw = reinterpret_cast<const int4*>(gate)[index];
    const int4 up_raw = reinterpret_cast<const int4*>(up_output)[index];
    __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(&gate_raw);
    const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&up_raw);
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
      const float x = __bfloat162float(g[i]);
      const float activation =
          0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
      g[i] = __float2bfloat16_rn(activation * __bfloat162float(u[i]));
    }
    reinterpret_cast<int4*>(up_output)[index] = gate_raw;
  }
}
}  // namespace

extern "C" int32_t backbone_rms_norm(const void* input, void* output,
                                    int32_t rows, void* stream) {
  rms_norm<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output));
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t backbone_gelu_mul(const void* gate, void* up_output,
                                    int64_t elements, void* stream) {
  constexpr int32_t threads = 256;
  // Reuse the 680-CTA pointwise configuration measured by this device's Pi0
  // implementation; local FFN timing decides whether it transfers to Pi0.5.
  constexpr int32_t max_blocks = 680;
  const int64_t vectors = elements / 8;
  const int32_t required = static_cast<int32_t>((vectors + threads - 1) / threads);
  const int32_t blocks = required < max_blocks ? required : max_blocks;
  gelu_mul<false><<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(up_output), vectors, nullptr);
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t backbone_masked_gelu_mul(
    const void* gate, void* up_output, int64_t elements, const void* mask, void* stream) {
  constexpr int32_t threads = 256;
  constexpr int32_t max_blocks = 680;
  const int64_t vectors = elements / 8;
  const int32_t required = static_cast<int32_t>((vectors + threads - 1) / threads);
  const int32_t blocks = required < max_blocks ? required : max_blocks;
  gelu_mul<true><<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(up_output), vectors,
      static_cast<const __nv_bfloat16*>(mask));
  return static_cast<int32_t>(cudaGetLastError());
}
