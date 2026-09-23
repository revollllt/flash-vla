// Producers that write MXFP8 or NVFP4 (values plus 128x4 swizzled scales)
// directly, or BF16, from one kernel. The BF16 output is the same computation
// stopped at its rounding point, so quantize(producer_bf16(x)) == producer_q(x)
// bit for bit; that is the fusion contract the tests check.
//
// Row kernels (RMSNorm, LayerNorm, optional residual add and AdaLN modulation)
// run one CTA per row, one thread per 8 elements (K <= 8192), so the row stays
// in registers and a small-M row, alone on its SM, still has K/256 warps to hide
// latency with. Elementwise kernels (copy, GELU-tanh, gated activations) run a
// 128-thread CTA per 1024 columns of a row. Thread t owns chunk j of the row,
// elements [8j, 8j + 8), so the threads sharing a scale block are consecutive
// lanes.
//
// Every kernel supports programmatic dependent launch (PDL): launched with
// programmatic stream serialization, it may start while the previous kernel of
// the stream drains. Nothing an earlier kernel writes may be read, and nothing
// may be written, before cudaGridDependencySynchronize(), which returns once
// every earlier kernel has completed and its memory is visible. Only the norm
// weight and bias, which no kernel of a captured chain writes, load above it.
// Without the launch attribute the wait returns at once.
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "quant/block_quant.cuh"

namespace {
using bf16 = __nv_bfloat16;
using namespace flash_vla_quant;

constexpr int32_t kElementwiseThreads = 128;
constexpr int32_t kMaxRowThreads = 1024;
enum Out : int32_t { kOutBf16 = 0, kOutMxfp8 = 1, kOutNvfp4 = 2 };

// Where every kernel releases its PDL dependents: 0 right after the dependency
// wait, 1 once the inputs are consumed (before the stores), 2 after the last
// store (the implicit trigger). Any point is correct, since the dependent's own
// wait orders memory; only speed differs.
// PDL-TRIGGER: point 0 -- SWEPT (results/quant-fused-rtx5090/pdl_sweep): fused
//   producer -> b12x GEMM chains at 10 GR00T/Pi0.5 sites x 2 formats total
//   13.88 ms per observation at point 0, 14.03 at 1, 14.30 at 2, 14.32 without
//   PDL. Two pairs prefer a late trigger; see the README.
// The trigger executes only in a grid that fits in one wave (early_trigger).
// It costs per triggering CTA, launched with PDL or not (quantize 968x16384,
// 15488 CTAs: 8.3 -> 15.0 us), and in a multi-wave grid it cannot fire early
// anyway: the last wave starts only as earlier CTAs exit. Such grids leave the
// release to the implicit trigger at exit.
#ifndef FVQ_PDL_TRIGGER
#define FVQ_PDL_TRIGGER 0
#endif

template <int32_t kPoint>
__device__ __forceinline__ void pdl_trigger_at(bool early_trigger) {
  if constexpr (kPoint == FVQ_PDL_TRIGGER) {
    if (early_trigger) cudaTriggerProgrammaticLaunchCompletion();
  }
}

__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint4 load_16_bytes(const bf16 *src) {
  return *reinterpret_cast<const uint4 *>(src);
}

__device__ __forceinline__ void unpack_bf16x8(const uint4 &packed, float (&values)[8]) {
  const __nv_bfloat162 *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    const float2 pair = __bfloat1622float2(pairs[i]);
    values[2 * i] = pair.x;
    values[2 * i + 1] = pair.y;
  }
}

__device__ __forceinline__ void store_bf16x8(bf16 *dst, const float (&values)[8]) {
  uint4 packed;
  __nv_bfloat162 *pairs = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) pairs[i] = __floats2bfloat162_rn(values[2 * i], values[2 * i + 1]);
  *reinterpret_cast<uint4 *>(dst) = packed;
}

// Write chunk `chunk` of `row` in the output format. Every lane calls this
// (the scale block's max is a shuffle); only valid chunks store. The values
// must already be BF16-representable.
template <int32_t kOut>
__device__ __forceinline__ void write_chunk(const float (&values)[8], bool valid, int32_t row,
                                            int32_t chunk, int32_t cols, void *out,
                                            uint8_t *scale_out, float global_scale) {
  if constexpr (kOut == kOutBf16) {
    if (valid) store_bf16x8(static_cast<bf16 *>(out) + int64_t(row) * cols + 8 * chunk, values);
  } else if constexpr (kOut == kOutMxfp8) {
    uint8_t *block_scale_out =
        (valid && chunk % 4 == 0)
            ? scale_out + sf_offset_128x4(row, 8 * chunk / kMxfp8Block, cols / kMxfp8Block)
            : nullptr;
    const uint2 packed = mxfp8_quantize8(values, block_scale_out);
    if (valid)
      *reinterpret_cast<uint2 *>(static_cast<uint8_t *>(out) + int64_t(row) * cols + 8 * chunk) =
          packed;
  } else {
    uint8_t *block_scale_out =
        (valid && chunk % 2 == 0)
            ? scale_out + sf_offset_128x4(row, 8 * chunk / kNvfp4Block, cols / kNvfp4Block)
            : nullptr;
    const uint32_t packed = nvfp4_quantize8(values, global_scale, block_scale_out);
    if (valid)
      *reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(out) + int64_t(row) * (cols / 2) +
                                    4 * chunk) = packed;
  }
}

__device__ __forceinline__ float block_sum(float value, float *warp_partials) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  const int32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) warp_partials[warp] = value;
  // Every warp's partial must be visible before any thread sums them.
  __syncthreads();
  float total = 0.f;
  for (int32_t w = 0; w < int32_t(blockDim.x >> 5); ++w) total += warp_partials[w];
  // The partials are reused by the next reduction; nobody may overwrite them
  // until every thread has read them.
  __syncthreads();
  return total;
}

// weight_mode: RMSNorm 0 = bf16(x*r), 1 = bf16(bf16(x*r) * w) (Llama, Qwen,
// Pi0.5 AdaRMS scale), 2 = bf16(x*r*(1+w)) in fp32 (Gemma); LayerNorm
// 0 = no affine, 1 = bf16((x-mean)*r*w + b) in fp32.
// round_factor: r = bf16(rsqrt(...)) before use (Pi0.5 decoder), and
// factor_out, when set, receives that per-row factor.
// mod_scale/mod_shift: y = bf16(bf16(y * bf16(1 + scale)) + shift) (AdaLN),
// one [cols] vector per mod_group_rows rows (0: one vector for all rows), the
// vectors mod_row_stride elements apart. Every vector is 16-byte aligned.
struct RowArgs {
  const bf16 *x, *residual, *weight, *bias, *mod_scale, *mod_shift;
  bf16 *residual_out, *factor_out;
  void *out;
  uint8_t *scale_out;
  const float *global_scale;
  int64_t mod_row_stride;
  int32_t rows, cols, weight_mode, round_factor, mod_group_rows;
  float eps;
  bool early_trigger;   // set by launch()
};

template <bool kLayer, int32_t kOut>
__global__ void __launch_bounds__(kMaxRowThreads) row_norm_kernel(RowArgs args) {
  __shared__ float warp_partials[kMaxRowThreads / 32];
  const int32_t row = blockIdx.x, chunk = threadIdx.x;
  const bool valid = 8 * chunk < args.cols;
  // Model parameters: no kernel of the chain writes them, so they load before the wait.
  uint4 packed_weight = {}, packed_bias = {}, packed_mod_scale = {}, packed_mod_shift = {};
  if (valid && args.weight != nullptr) packed_weight = load_16_bytes(args.weight + 8 * chunk);
  if (valid && kLayer && args.weight_mode == 1) packed_bias = load_16_bytes(args.bias + 8 * chunk);
  // PDL-WAIT: before the first read of x, residual, modulation or global_scale -- DERIVED
  cudaGridDependencySynchronize();
  pdl_trigger_at<0>(args.early_trigger);
  const float global_scale = (kOut == kOutNvfp4) ? *args.global_scale : 0.f;
  float x[8] = {};
  if (valid) {
    const int64_t offset = int64_t(row) * args.cols + 8 * chunk;
    unpack_bf16x8(load_16_bytes(args.x + offset), x);
    if (args.residual != nullptr) {
      float residual[8];
      unpack_bf16x8(load_16_bytes(args.residual + offset), residual);
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) x[i] = round_bf16(x[i] + residual[i]);
      if (args.residual_out != nullptr) store_bf16x8(args.residual_out + offset, x);
    }
  }
  // AdaLN modulation comes from an earlier kernel (the timestep projection);
  // issued before the reductions so its latency overlaps them.
  const int64_t mod_offset =
      (args.mod_group_rows > 0 ? int64_t(row / args.mod_group_rows) * args.mod_row_stride : 0) +
      8 * chunk;
  if (valid && args.mod_scale != nullptr) {
    packed_mod_scale = load_16_bytes(args.mod_scale + mod_offset);
    packed_mod_shift = load_16_bytes(args.mod_shift + mod_offset);
  }
  float lane_sum = 0.f;
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) lane_sum += kLayer ? x[i] : x[i] * x[i];
  float mean = 0.f, rstd;
  if constexpr (kLayer) {
    mean = block_sum(lane_sum, warp_partials) / args.cols;
    float lane_sum_sq_dev = 0.f;
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) lane_sum_sq_dev += valid ? (x[i] - mean) * (x[i] - mean) : 0.f;
    rstd = rsqrtf(block_sum(lane_sum_sq_dev, warp_partials) / args.cols + args.eps);
  } else {
    rstd = rsqrtf(block_sum(lane_sum, warp_partials) / args.cols + args.eps);
  }
  pdl_trigger_at<1>(args.early_trigger);
  rstd = args.round_factor ? round_bf16(rstd) : rstd;
  if (args.factor_out != nullptr && threadIdx.x == 0)
    args.factor_out[row] = __float2bfloat16_rn(rstd);
  float weight[8], bias[8], mod_scale[8], mod_shift[8];
  unpack_bf16x8(packed_weight, weight);
  unpack_bf16x8(packed_bias, bias);
  unpack_bf16x8(packed_mod_scale, mod_scale);
  unpack_bf16x8(packed_mod_shift, mod_shift);
  float normalized[8];
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    float value;
    if constexpr (kLayer) {
      const float centered = (x[i] - mean) * rstd;
      value = round_bf16(args.weight_mode == 1 ? centered * weight[i] + bias[i] : centered);
    } else if (args.weight_mode == 1) {
      value = round_bf16(round_bf16(x[i] * rstd) * weight[i]);
    } else if (args.weight_mode == 2) {
      value = round_bf16(x[i] * rstd * (1.f + weight[i]));
    } else {
      value = round_bf16(x[i] * rstd);
    }
    const float modulated =
        args.mod_scale == nullptr
            ? value
            : round_bf16(round_bf16(value * round_bf16(1.f + mod_scale[i])) + mod_shift[i]);
    normalized[i] = valid ? modulated : 0.f;
  }
  write_chunk<kOut>(normalized, valid, row, chunk, args.cols, args.out, args.scale_out,
                    global_scale);
  pdl_trigger_at<2>(args.early_trigger);
}

// PyTorch's CUDA gelu(approximate="tanh") and silu, in the same fp32 order.
__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float kKappa = 0.044715f;
  const float x_cube = x * x * x;
  const float inner = kBeta * (x + kKappa * x_cube);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// kind: 0 copy, 1 gelu_tanh(x), 2 act(x) * up. act: 0 gelu_tanh, 1 silu.
// round_act: 0 = bf16(act(g) * u) in fp32 (one rounding), 1 = bf16(bf16(act(g)) * u).
// Row strides are in elements.
struct ElementwiseArgs {
  const bf16 *x, *up;
  int64_t x_row_stride, up_row_stride;
  void *out;
  uint8_t *scale_out;
  const float *global_scale;
  int32_t rows, cols, kind, act, round_act;
  bool early_trigger;   // set by launch()
};

template <int32_t kOut>
__global__ void __launch_bounds__(kElementwiseThreads) elementwise_kernel(ElementwiseArgs args) {
  const int32_t row = blockIdx.y;
  const int32_t chunk = blockIdx.x * kElementwiseThreads + threadIdx.x;
  const bool valid = 8 * chunk < args.cols;
  // PDL-WAIT: before the first read of x, up or global_scale -- DERIVED (no parameters to hoist)
  cudaGridDependencySynchronize();
  pdl_trigger_at<0>(args.early_trigger);
  const float global_scale = (kOut == kOutNvfp4) ? *args.global_scale : 0.f;
  float result[8] = {};
  if (valid) {
    float x[8];
    unpack_bf16x8(load_16_bytes(args.x + int64_t(row) * args.x_row_stride + 8 * chunk), x);
    if (args.kind == 0) {
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) result[i] = x[i];
    } else if (args.kind == 1) {
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) result[i] = round_bf16(gelu_tanh(x[i]));
    } else {
      float up[8];
      unpack_bf16x8(load_16_bytes(args.up + int64_t(row) * args.up_row_stride + 8 * chunk), up);
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        const float activated = args.act == 0 ? gelu_tanh(x[i]) : silu(x[i]);
        result[i] = round_bf16((args.round_act ? round_bf16(activated) : activated) * up[i]);
      }
    }
  }
  pdl_trigger_at<1>(args.early_trigger);
  write_chunk<kOut>(result, valid, row, chunk, args.cols, args.out, args.scale_out, global_scale);
  pdl_trigger_at<2>(args.early_trigger);
}

// Launch with programmatic stream serialization (PDL) when pdl is set, and let
// the kernel trigger its dependents early only when its grid fits in one wave.
template <typename Args>
cudaError_t launch(void (*kernel)(Args), dim3 grid, dim3 block, bool pdl, cudaStream_t stream,
                   Args args) {
  int device = 0, sms = 0, ctas_per_sm = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, int(block.x), 0);
  args.early_trigger = pdl && int64_t(grid.x) * grid.y <= int64_t(sms) * ctas_per_sm;
  cudaLaunchAttribute serialization;
  serialization.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  serialization.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.stream = stream;
  config.attrs = &serialization;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, args);
}

template <bool kLayer, int32_t kOut>
cudaError_t launch_row(const RowArgs &args, bool pdl, cudaStream_t stream) {
  const int32_t threads = (args.cols / 8 + 31) / 32 * 32;
  if (threads > kMaxRowThreads) return cudaErrorInvalidValue;
  return launch(row_norm_kernel<kLayer, kOut>, dim3(args.rows), dim3(threads), pdl, stream, args);
}

}  // namespace

extern "C" {

// layer: 0 RMSNorm, 1 LayerNorm. out_kind: 0 bf16, 1 mxfp8, 2 nvfp4.
int32_t fvq_row_norm(const void *x, const void *residual, void *residual_out, const void *weight,
                     const void *bias, const void *mod_scale, const void *mod_shift,
                     void *factor_out, void *out, void *scale_out, const float *global_scale,
                     int64_t mod_row_stride, int32_t rows, int32_t cols, int32_t weight_mode,
                     int32_t round_factor, int32_t mod_group_rows, float eps, int32_t layer,
                     int32_t out_kind, int32_t pdl, void *stream) {
  const RowArgs args{static_cast<const bf16 *>(x),         static_cast<const bf16 *>(residual),
                     static_cast<const bf16 *>(weight),    static_cast<const bf16 *>(bias),
                     static_cast<const bf16 *>(mod_scale), static_cast<const bf16 *>(mod_shift),
                     static_cast<bf16 *>(residual_out),    static_cast<bf16 *>(factor_out),
                     out,                                  static_cast<uint8_t *>(scale_out),
                     global_scale,                         mod_row_stride,
                     rows, cols, weight_mode, round_factor, mod_group_rows, eps};
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (layer) {
    if (out_kind == kOutBf16) return launch_row<true, kOutBf16>(args, pdl, cuda_stream);
    if (out_kind == kOutMxfp8) return launch_row<true, kOutMxfp8>(args, pdl, cuda_stream);
    return launch_row<true, kOutNvfp4>(args, pdl, cuda_stream);
  }
  if (out_kind == kOutBf16) return launch_row<false, kOutBf16>(args, pdl, cuda_stream);
  if (out_kind == kOutMxfp8) return launch_row<false, kOutMxfp8>(args, pdl, cuda_stream);
  return launch_row<false, kOutNvfp4>(args, pdl, cuda_stream);
}

// kind: 0 copy, 1 gelu_tanh, 2 gated (x is the gate).
int32_t fvq_elementwise(const void *x, int64_t x_row_stride, const void *up,
                        int64_t up_row_stride, void *out, void *scale_out,
                        const float *global_scale, int32_t rows, int32_t cols, int32_t kind,
                        int32_t act, int32_t round_act, int32_t out_kind, int32_t pdl,
                        void *stream) {
  const ElementwiseArgs args{static_cast<const bf16 *>(x), static_cast<const bf16 *>(up),
                             x_row_stride, up_row_stride, out, static_cast<uint8_t *>(scale_out),
                             global_scale, rows, cols, kind, act, round_act};
  const dim3 grid((cols / 8 + kElementwiseThreads - 1) / kElementwiseThreads, rows);
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (out_kind == kOutBf16)
    return launch(elementwise_kernel<kOutBf16>, grid, dim3(kElementwiseThreads), pdl, cuda_stream,
                  args);
  if (out_kind == kOutMxfp8)
    return launch(elementwise_kernel<kOutMxfp8>, grid, dim3(kElementwiseThreads), pdl, cuda_stream,
                  args);
  return launch(elementwise_kernel<kOutNvfp4>, grid, dim3(kElementwiseThreads), pdl, cuda_stream,
                args);
}

}  // extern "C"
