#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

// One CTA holds the complete <=1024-key row in registers between reductions.
template <int Threads>
__global__ void pi05_attention_softmax(
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ probabilities,
    int32_t keys, float scale) {
  const int32_t row = blockIdx.x;
  const int32_t tid = threadIdx.x;
  const int32_t lane = tid & 31;
  const int32_t warp = tid >> 5;
  constexpr int Items = 1024 / Threads;
  constexpr int Warps = Threads / 32;
  float held[Items];
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int32_t i = 0; i < Items; ++i) {
    const int32_t key = tid + Threads * i;
    held[i] = key < keys
        ? logits[int64_t(row) * keys + key] * scale + __bfloat162float(mask[key])
        : -CUDART_INF_F;
    maximum = fmaxf(maximum, held[i]);
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  __shared__ float warp_maxima[Warps];
  if (lane == 0) warp_maxima[warp] = maximum;
  // Each warp consumes all CTA maxima only after their writers finish.
  __syncthreads();
  maximum = lane < Warps ? warp_maxima[lane] : -CUDART_INF_F;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  float sum = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < Items; ++i) {
    held[i] = expf(held[i] - maximum);
    sum += held[i];
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  // Separate storage permits sum producers to overlap maximum consumers.
  __shared__ float warp_sums[Warps];
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  sum = lane < Warps ? warp_sums[lane] : 0.0f;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  const float reciprocal = 1.0f / sum;
#pragma unroll
  for (int32_t i = 0; i < Items; ++i) {
    const int32_t key = tid + Threads * i;
    if (key < keys)
      probabilities[int64_t(row) * keys + key] =
          __float2bfloat16_rn(held[i] * reciprocal);
  }
}

extern "C" int32_t pi05_attention_softmax_launch(
    const void* logits, const void* mask, void* probabilities,
    int32_t queries, int32_t keys, float scale, cudaStream_t stream) {
  pi05_attention_softmax<256><<<queries, 256, 0, stream>>>(
      static_cast<const float*>(logits), static_cast<const __nv_bfloat16*>(mask),
      static_cast<__nv_bfloat16*>(probabilities), keys, scale);
  return static_cast<int32_t>(cudaGetLastError());
}

// Lab-only fixed candidate; the production launch above remains 256 threads.
extern "C" int32_t pi05_attention_softmax_128_launch(
    const void* logits, const void* mask, void* probabilities,
    int32_t queries, int32_t keys, float scale, cudaStream_t stream) {
  pi05_attention_softmax<128><<<queries, 128, 0, stream>>>(
      static_cast<const float*>(logits), static_cast<const __nv_bfloat16*>(mask),
      static_cast<__nv_bfloat16*>(probabilities), keys, scale);
  return static_cast<int32_t>(cudaGetLastError());
}
