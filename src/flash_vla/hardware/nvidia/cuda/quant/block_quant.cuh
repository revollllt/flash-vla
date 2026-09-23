// MXFP8 and NVFP4 block quantization for one thread's eight values, with the
// scale factor written in the 128x4 swizzled layout Blackwell's block-scaled
// GEMMs read (CUTLASS Sm1xxBlockScaledConfig, cuDNN, FlashInfer; sm_100 tcgen05
// and sm_120 mma.sync alike). Needs a Blackwell target (sm_100f, sm_110f,
// sm_120f or an arch-specific sm_1xxa): Hopper has no cvt.e2m1x2.
//
// The arithmetic is FlashInfer 0.7.0's default fast path, so values and scale
// bytes are bit-identical to flashinfer.mxfp8_quantize / fp4_quantize:
//   MXFP8  cvt_warp_fp16_to_mxfp8        (quantization_utils.cuh)
//   NVFP4  cvt_warp_fp16_to_fp4, default (not DISABLE_FP4_QUANT_FAST_MATH)
//   layout get_sf_out_offset_128x4
// Copyright (c) NVIDIA CORPORATION & AFFILIATES / FlashInfer team, Apache-2.0
// (third_party/quant-references/flashinfer). Adapted to take float values that
// are already BF16-representable, so a producer can quantize without first
// storing BF16.
//
// FlashInfer compiles with -use_fast_math, so its multiplies and zero tests
// flush subnormals. These functions flush explicitly (mul_ftz, is_zero_ftz)
// instead, which leaves the producers that include them IEEE. One visible
// consequence: a block whose max is below 2^-126 * 448 (MXFP8), including an
// all-zero block, gets scale code 0 and zero values.
//
// Contract: a thread owns elements [8j, 8j+8) of a row; the 4 (MXFP8) or 2
// (NVFP4) threads that share a scale block are consecutive, aligned lanes, and
// every lane of the warp calls the function (the block max is a shuffle). One
// lane per block passes the scale's address, the others nullptr.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_vla_quant {

constexpr int32_t kEltsPerThread = 8;
constexpr int32_t kMxfp8Block = 32;
constexpr int32_t kNvfp4Block = 16;

__device__ __forceinline__ float rcp_approx_ftz(float value) {
  float reciprocal;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(reciprocal) : "f"(value));
  return reciprocal;
}

__device__ __forceinline__ float mul_ftz(float lhs, float rhs) {
  float product;
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(product) : "f"(lhs), "f"(rhs));
  return product;
}

// value == 0 under flush-to-zero: true for zeros and subnormals.
__device__ __forceinline__ bool is_zero_ftz(float value) {
  return fabsf(value) < 1.17549435e-38f;
}

// Offset of scale (row, scale_col) in [row_tiles, col_tiles, 32 (row), 4 (row), 4 (col)].
__device__ __forceinline__ int64_t sf_offset_128x4(int32_t row, int32_t scale_col,
                                                   int32_t scale_cols) {
  const int64_t col_tiles = (scale_cols + 3) / 4;
  return int64_t(row / 128) * col_tiles * 512 + int64_t(scale_col / 4) * 512 +
         int64_t(row % 32) * 16 + int64_t((row % 128) / 32) * 4 + (scale_col % 4);
}

// Max over the kLanes consecutive lanes that share one scale block.
template <int32_t kLanes>
__device__ __forceinline__ float group_max(float lane_max) {
  if constexpr (kLanes >= 2) lane_max = fmaxf(lane_max, __shfl_xor_sync(0xffffffffu, lane_max, 1));
  if constexpr (kLanes >= 4) lane_max = fmaxf(lane_max, __shfl_xor_sync(0xffffffffu, lane_max, 2));
  return lane_max;
}

// Eight E4M3 bytes, element 0 in the lowest byte.
__device__ __forceinline__ uint2 mxfp8_quantize8(const float (&values)[8], uint8_t *scale_out) {
  // Upstream takes the max in BF16, which is exact, so no flush happens here.
  float lane_max = fabsf(values[0]);
#pragma unroll
  for (int32_t i = 1; i < 8; ++i) lane_max = fmaxf(lane_max, fabsf(values[i]));
  const float block_max = group_max<kMxfp8Block / kEltsPerThread>(lane_max);
  __nv_fp8_e8m0 scale_ue8m0;
  scale_ue8m0.__x = __nv_cvt_float_to_e8m0(mul_ftz(block_max, rcp_approx_ftz(448.0f)),
                                           __NV_SATFINITE, cudaRoundPosInf);
  const float block_scale = static_cast<float>(scale_ue8m0);
  // Code 0 is 2^-127, a subnormal, so the fast-math test treats it as zero.
  const float inverse_scale = is_zero_ftz(block_scale) ? 0.f : rcp_approx_ftz(block_scale);
  if (scale_out != nullptr) *scale_out = scale_ue8m0.__x;
  uint32_t words[2];
#pragma unroll
  for (int32_t word = 0; word < 2; ++word) {
    uint32_t packed = 0;
#pragma unroll
    for (int32_t half = 0; half < 2; ++half) {
      const int32_t i = 4 * word + 2 * half;
      float2 scaled = make_float2(mul_ftz(values[i], inverse_scale),
                                  mul_ftz(values[i + 1], inverse_scale));
      scaled.x = fmaxf(fminf(scaled.x, 448.0f), -448.0f);
      scaled.y = fmaxf(fminf(scaled.y, 448.0f), -448.0f);
      const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
      packed |= uint32_t(pair) << (16 * half);
    }
    words[word] = packed;
  }
  return make_uint2(words[0], words[1]);
}

// Eight E2M1 nibbles, element 0 in the lowest nibble. global_scale is the
// NVFP4 per-tensor encode scale (FlashInfer's SFScaleVal).
__device__ __forceinline__ uint32_t nvfp4_quantize8(const float (&values)[8], float global_scale,
                                                    uint8_t *scale_out) {
  float lane_max = fabsf(values[0]);
#pragma unroll
  for (int32_t i = 1; i < 8; ++i) lane_max = fmaxf(lane_max, fabsf(values[i]));
  const float block_max = group_max<kNvfp4Block / kEltsPerThread>(lane_max);
  const __nv_fp8_e4m3 scale_ue4m3(
      mul_ftz(global_scale, mul_ftz(block_max, rcp_approx_ftz(6.0f))));
  // A nonzero block whose scale underflows to code 0 gets an infinite
  // multiplier, as upstream: its values saturate (zeros become NaN, then 6),
  // and code 0 dequantizes all of them to zero.
  const float inverse_scale =
      is_zero_ftz(block_max)
          ? 0.f
          : rcp_approx_ftz(mul_ftz(static_cast<float>(scale_ue4m3), rcp_approx_ftz(global_scale)));
  if (scale_out != nullptr) *scale_out = scale_ue4m3.__x;
  float scaled[8];
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) scaled[i] = mul_ftz(values[i], inverse_scale);
  uint32_t packed;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(packed)
      : "f"(scaled[0]), "f"(scaled[1]), "f"(scaled[2]), "f"(scaled[3]), "f"(scaled[4]),
        "f"(scaled[5]), "f"(scaled[6]), "f"(scaled[7]));
  return packed;
}

}  // namespace flash_vla_quant
