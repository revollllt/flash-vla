// BF16 pairs from Mx2560 QKV -> Q(Mx2048), K/V(Mx256), with adjacent-pair RoPE.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__global__ void rope_scatter(const __nv_bfloat162 *__restrict__ projected,
                             const __nv_bfloat162 *__restrict__ rope,
                             __nv_bfloat162 *__restrict__ query,
                             __nv_bfloat162 *__restrict__ key,
                             __nv_bfloat162 *__restrict__ value, int32_t pairs) {
  for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < pairs;
       index += gridDim.x * blockDim.x) {
    const int32_t row = index / 1280;
    const int32_t col = index % 1280;
    const __nv_bfloat162 input = projected[index];
    if (col < 1152) {
      const __nv_bfloat162 cs = rope[row * 128 + col % 128];
      const float x = __bfloat162float(input.x), y = __bfloat162float(input.y);
      const float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
      const __nv_bfloat162 rotated = __floats2bfloat162_rn(x * c - y * s, y * c + x * s);
      if (col < 1024)
        query[row * 1024 + col] = rotated;
      else
        key[row * 128 + col - 1024] = rotated;
    } else {
      value[row * 128 + col - 1152] = input;
    }
  }
}
}  // namespace

extern "C" int32_t prefix_rope_scatter(const void *projected, const void *rope,
                                        void *query, void *key, void *value,
                                        int32_t rows, void *stream) {
  constexpr int32_t threads = 256;
  // Flat pair traversal, capped like this Target's streaming pointwise stages.
  const int32_t needed = (rows * 1280 + threads - 1) / threads;
  const int32_t blocks = needed < 680 ? needed : 680;
  rope_scatter<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat162 *>(projected),
      static_cast<const __nv_bfloat162 *>(rope),
      static_cast<__nv_bfloat162 *>(query), static_cast<__nv_bfloat162 *>(key),
      static_cast<__nv_bfloat162 *>(value), rows * 1280);
  return static_cast<int32_t>(cudaGetLastError());
}
