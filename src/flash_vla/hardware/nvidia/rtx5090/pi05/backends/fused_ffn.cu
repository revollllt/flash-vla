// Pi0.5 expert FFN, M x 1024 -> M x 4096, contiguous bf16 tensors.
// AdaRMS uses one 256-thread CTA per row, keeping its four values in registers.
// The epilogue has no reduction; 256 threads process four columns apiece.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__global__ void ada_rms_kernel(const bf16 *__restrict__ x,
                               const bf16 *__restrict__ scale,
                               bf16 *__restrict__ a,
                               bf16 *__restrict__ factor) {
  const int32_t row = blockIdx.x;
  const int32_t lane = threadIdx.x & 31;
  const int32_t warp = threadIdx.x >> 5;
  float values[4];
  float sum = 0.f;
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    values[i] = __bfloat162float(x[row * 1024 + threadIdx.x + i * 256]);
    sum += values[i] * values[i];
  }
  sum = warp_sum(sum);
  __shared__ float partial[8];
  __shared__ float rounded_factor;
  if (lane == 0) partial[warp] = sum;
  // All eight warp sums must be visible before the first warp reduces them.
  __syncthreads();
  if (warp == 0) {
    sum = warp_sum(lane < 8 ? partial[lane] : 0.f);
    if (lane == 0) {
      const bf16 f = __float2bfloat16_rn(rsqrtf(sum / 1024.f + 1e-6f));
      factor[row] = f;
      rounded_factor = __bfloat162float(f);
    }
  }
  // Broadcast the rounded factor before consuming it in each activation.
  __syncthreads();
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    const int32_t col = threadIdx.x + i * 256;
    const bf16 normalized = __float2bfloat16_rn(values[i] * rounded_factor);
    a[row * 1024 + col] = __float2bfloat16_rn(
        __bfloat162float(normalized) * __bfloat162float(scale[col]));
  }
}

template <bool Packed>
__global__ void gated_activation_kernel(const bf16 *__restrict__ gate,
                                        const bf16 *__restrict__ up,
                                        const bf16 *__restrict__ gate_bias,
                                        const bf16 *__restrict__ up_bias,
                                        bf16 *__restrict__ out, int32_t elements) {
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    const int32_t index = blockIdx.x * 1024 + threadIdx.x + i * 256;
    if (index < elements) {
      const int32_t col = index % 4096;
      const int32_t source = Packed ? (index / 4096) * 8192 + col : index;
      const float g = __bfloat162float(gate[source]) + __bfloat162float(gate_bias[col]);
      const float u = __bfloat162float(up[source]) + __bfloat162float(up_bias[col]);
      // Match torch's fp32 approximate="tanh" expression before bf16 output.
      const float cube = g * g * g;
      const float gelu = 0.5f * g * (1.f + tanhf(0.7978845608028654f *
                                               (g + 0.044715f * cube)));
      out[index] = __float2bfloat16_rn(gelu * u);
    }
  }
}
}  // namespace

extern "C" int32_t ada_rms_launch(const void *x, const void *scale, void *a,
                                   void *factor, int32_t rows, void *stream) {
  ada_rms_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16 *>(x), static_cast<const bf16 *>(scale),
      static_cast<bf16 *>(a), static_cast<bf16 *>(factor));
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t gated_activation_launch(const void *gate, const void *up,
                                            const void *gate_bias, const void *up_bias,
                                            void *out, int32_t rows, void *stream) {
  gated_activation_kernel<false><<<rows * 4, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16 *>(gate), static_cast<const bf16 *>(up),
      static_cast<const bf16 *>(gate_bias), static_cast<const bf16 *>(up_bias),
      static_cast<bf16 *>(out), rows * 4096);
  return static_cast<int32_t>(cudaGetLastError());
}

extern "C" int32_t packed_gated_activation_launch(const void *gate, const void *up,
                                                   const void *gate_bias,
                                                   const void *up_bias,
                                                   void *out, int32_t rows,
                                                   void *stream) {
  gated_activation_kernel<true><<<rows * 4, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16 *>(gate), static_cast<const bf16 *>(up),
      static_cast<const bf16 *>(gate_bias), static_cast<const bf16 *>(up_bias),
      static_cast<bf16 *>(out), rows * 4096);
  return static_cast<int32_t>(cudaGetLastError());
}
