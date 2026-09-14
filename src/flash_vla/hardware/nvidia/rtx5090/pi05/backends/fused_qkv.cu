#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// One CTA per 1024-wide row; scale and RMS read the same input once.
__global__ void pi05_qkv_scale_factor(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ scaled,
    __nv_bfloat16* __restrict__ factor) {
  const int32_t row = blockIdx.x;
  const int32_t tid = threadIdx.x;
  float sum = 0.0f;
#pragma unroll
  for (int32_t j = 0; j < 4; ++j) {
    const int32_t col = tid + j * 256;
    const float value = __bfloat162float(x[row * 1024 + col]);
    sum += value * value;
    scaled[row * 1024 + col] =
        __float2bfloat16_rn(value * __bfloat162float(scale[col]));
  }
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  __shared__ float warp_sums[8];
  if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
  // All warps publish their partial sum before warp zero consumes them.
  __syncthreads();
  if (tid < 32) {
    sum = tid < 8 ? warp_sums[tid] : 0.0f;
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (tid == 0)
      factor[row] = __float2bfloat16_rn(rsqrtf(sum * (1.0f / 1024) + 1e-6f));
  }
}

// Five CTAs per row cover the 1280 packed pairs; Q and K rotate, V only shifts.
__global__ void pi05_qkv_factor_bias_rope_scatter(
    const __nv_bfloat162* __restrict__ projected,
    const __nv_bfloat16* __restrict__ factor,
    const __nv_bfloat162* __restrict__ bias,
    const __nv_bfloat162* __restrict__ rope,
    __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k,
    __nv_bfloat162* __restrict__ v) {
  const int32_t row = blockIdx.x / 5;
  const int32_t pair = (blockIdx.x % 5) * 256 + threadIdx.x;
  const float f = __bfloat162float(factor[row]);
  const float2 p = __bfloat1622float2(projected[row * 1280 + pair]);
  const float2 b = __bfloat1622float2(bias[pair]);
  // Compile with --fmad=false: torch rounds multiply before bias/rotation sums.
  const float a0 = p.x * f + b.x;
  const float a1 = p.y * f + b.y;
  if (pair < 1152) {
    const float2 r = __bfloat1622float2(rope[row * 128 + pair % 128]);
    const __nv_bfloat162 rotated = __floats2bfloat162_rn(
        a0 * r.x - a1 * r.y, a1 * r.x + a0 * r.y);
    if (pair < 1024)
      q[row * 1024 + pair] = rotated;
    else
      k[row * 128 + pair - 1024] = rotated;
  } else {
    v[row * 128 + pair - 1152] = __floats2bfloat162_rn(a0, a1);
  }
}

extern "C" int32_t pi05_qkv_prepare(
    const void* x, const void* scale, void* scaled, void* factor,
    int32_t rows, cudaStream_t stream) {
  pi05_qkv_scale_factor<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<__nv_bfloat16*>(scaled), static_cast<__nv_bfloat16*>(factor));
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t pi05_qkv_finish(
    const void* projected, const void* factor, const void* bias,
    const void* rope, void* q, void* k, void* v,
    int32_t rows, cudaStream_t stream) {
  pi05_qkv_factor_bias_rope_scatter<<<rows * 5, 256, 0, stream>>>(
      static_cast<const __nv_bfloat162*>(projected),
      static_cast<const __nv_bfloat16*>(factor),
      static_cast<const __nv_bfloat162*>(bias),
      static_cast<const __nv_bfloat162*>(rope),
      static_cast<__nv_bfloat162*>(q), static_cast<__nv_bfloat162*>(k),
      static_cast<__nv_bfloat162*>(v));
  return static_cast<int32_t>(cudaGetLastError());
}
