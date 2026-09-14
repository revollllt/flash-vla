#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// One CTA owns a 1152-value row. Retaining values avoids rereads for variance
// and the affine transform; centered variance avoids E[x*x] - E[x]^2 cancellation.
__global__ void pi05_vision_layer_norm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out) {
  const int32_t row = blockIdx.x;
  const int32_t tid = threadIdx.x;
  const int32_t lane = tid & 31;
  const int32_t warp = tid >> 5;
  float held[5];
  float sum = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < 5; ++i) {
    const int32_t col = tid + i * 256;
    held[i] = col < 1152 ? __bfloat162float(x[int64_t(row) * 1152 + col]) : 0.0f;
    sum += held[i];
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  __shared__ float warp_sums[8];
  if (lane == 0) warp_sums[warp] = sum;
  // All row-sum producers finish before any warp consumes their partials.
  __syncthreads();
  sum = lane < 8 ? warp_sums[lane] : 0.0f;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  const float mean = sum / 1152.0f;
  float variance = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < 5; ++i) {
    const int32_t col = tid + i * 256;
    held[i] -= mean;
    if (col < 1152) variance += held[i] * held[i];
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    variance += __shfl_xor_sync(0xffffffffu, variance, offset);
  // Separate storage lets variance writers overlap earlier sum readers.
  __shared__ float warp_variances[8];
  if (lane == 0) warp_variances[warp] = variance;
  __syncthreads();
  variance = lane < 8 ? warp_variances[lane] : 0.0f;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    variance += __shfl_xor_sync(0xffffffffu, variance, offset);
  const float inverse = rsqrtf(variance / 1152.0f + 1e-5f);
#pragma unroll
  for (int32_t i = 0; i < 5; ++i) {
    const int32_t col = tid + i * 256;
    if (col < 1152)
      out[int64_t(row) * 1152 + col] = __float2bfloat16_rn(
          held[i] * inverse * __bfloat162float(weight[col]) + __bfloat162float(bias[col]));
  }
}

// Input is the BF16-rounded addmm result. Keep the tanh GELU in FP32 until store.
__global__ void pi05_vision_gelu(__nv_bfloat16* values, int64_t vectors) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    int4 packed = reinterpret_cast<int4*>(values)[index];
    auto* elements = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
      const float x = __bfloat162float(elements[i]);
      const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
      elements[i] = __float2bfloat16_rn(0.5f * x * (1.0f + tanhf(inner)));
    }
    reinterpret_cast<int4*>(values)[index] = packed;
  }
}

extern "C" int32_t pi05_vision_layer_norm_launch(
    const void* x, const void* weight, const void* bias, void* out,
    int32_t rows, cudaStream_t stream) {
  pi05_vision_layer_norm<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(out));
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t pi05_vision_gelu_launch(
    void* values, int64_t elements, cudaStream_t stream) {
  const int64_t vectors = elements / 8;
  const int32_t blocks = static_cast<int32_t>((vectors + 255) / 256);
  pi05_vision_gelu<<<blocks, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(values), vectors);
  return static_cast<int32_t>(cudaGetLastError());
}
