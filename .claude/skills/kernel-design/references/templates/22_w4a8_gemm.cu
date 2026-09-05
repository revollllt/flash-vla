// Template 22 -- INT4A8 and MXFP4A8: quantized activations (sm90).
//
// Dropping activations to 8 bits changes what the kernel is, not just how big
// it is.  The one idea to take away:
//
//   IN A16, THE SCALE MULTIPLIES THE WEIGHT BEFORE THE MMA.
//   IN A8, THE MMA STAYS IN THE QUANTIZED DOMAIN AND THE SCALES MULTIPLY THE
//   ACCUMULATOR ONCE.
//
// That is where the speed comes from.  Template 20 spends a bf16 multiply per
// weight to apply a group scale; here the tensor core consumes the quantized
// values directly -- int8 or fp8, both twice the rate of bf16 on sm90 -- and
// the per-token activation scale and per-channel weight scale meet once, on
// the fp32/int32 accumulator, in the epilogue.  Scales that live on different
// axes (one per row, one per column) can only be combined there anyway.
//
// Two paths, because the two formats reach the tensor core differently:
//
//   * INT4A8 (GPTQ/QQQ shape).  Weights unpack to int8 with a per-byte
//     subtract -- no exponent trick, since the destination is an integer.
//     s8 x s8 -> s32 accumulates exactly, so there is no precision argument
//     for breaking the K loop; one epilogue multiply finishes it.
//   * MXFP4A8.  Weights are e2m1, so they must become e4m3 to ride the fp8
//     tensor core.  e2m1 has eight magnitudes, which is exactly one prmt
//     source pair, so the conversion is a register-resident LUT -- and the
//     table is BUILT per group by an integer multiply-add, which folds the
//     MXFP4 block scale into it for two integer ops instead of a multiply per
//     weight (quant_sm90.cuh, method from humming).  The promotion that
//     follows is then a plain add: the scale is already in the operand.
//     fp8 accumulation on Hopper is still reduced-precision, so the K loop
//     breaks per scale block anyway -- template 11's two-level accumulation,
//     with the unpack in front.
//
// On sm120 the MXFP4A8 path collapses into a native block-scaled MMA and this
// unpack disappears; the INT4A8 path does not, because int4 is not an MX
// format.  See template 21's header for that fork.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-GRADE: structural
// CHECK-PTX: mma\.sync\.aligned\.m16n8k32\.row\.col\.s32\.s8\.s8\.s32
// CHECK-PTX: mma\.sync\.aligned\.m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32
// CHECK-PTX: prmt\.b32
// CHECK-PTX: lop3\.b32
// CHECK-PTX: cp\.async\.cg\.shared\.global
// CHECK-PTX: cp\.async\.wait_group

#include <cuda_fp8.h>

#include "sm90_common.cuh"
#include "quant_sm90.cuh"

namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kTileK = 128;  // k32 instructions, so a tile is 4 MMA steps
constexpr int kStages = 4;
constexpr int kThreads = 128;
constexpr int kPackFactor = 8;

// MXFP4's block: 32 elements share one ue8m0 scale.
constexpr int kMxBlock = 32;

constexpr int kAElems = kTileM * kTileK;              // int8 or e4m3, 1 byte each
constexpr int kBWords = kTileN * kTileK / kPackFactor;

constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kAElems;
constexpr int kSmemBytes = kOffB + kStages * kBWords * static_cast<int>(sizeof(int32_t));

// s8 x s8 -> s32.  A is 4 words (16 int8), B is 2 words (8 int8).
__device__ __forceinline__ void mma_s8_m16n8k32(const uint32_t (&a)[4],
                                                const uint32_t (&b)[2],
                                                int32_t (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// e4m3 x e4m3 -> f32, the same shape as the int8 form.
__device__ __forceinline__ void mma_e4m3_m16n8k32(const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2],
                                                  float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Eight packed uint4 -> eight int8 in two words.  The subtract must be
// PER BYTE: a nibble below 8 goes negative, and a plain 32-bit subtract would
// borrow into the neighbouring byte and corrupt it.
//
// __vsub4 is the portable spelling of that, but do not assume it is one
// instruction: the SIMD video ops are emulated on current toolkits, and here
// CUDA 13 lowers it to prmt + and + sub.  Correct, and still cheaper than
// converting through float, but count it as a few ops when budgeting the
// unpack against the weight stream.
__device__ __forceinline__ void unpack_u4_to_s8(int32_t q, uint32_t (&out)[2]) {
  constexpr uint32_t kNibble = 0x0f0f0f0f;
  constexpr uint32_t kBias = 0x08080808;
  out[0] = __vsub4(static_cast<uint32_t>(q) & kNibble, kBias);
  out[1] = __vsub4((static_cast<uint32_t>(q) >> 4) & kNibble, kBias);
}

// exp_offset 6 is the unscaled e2m1 table; the block's residual exponent is
// added to it, so the LUT itself carries the scale.
constexpr uint32_t kE2m1IdentityOffset = 6;

}  // namespace

// ---------------------------------------------------------------- INT4 x A8

__global__ __launch_bounds__(kThreads, 1) void w4a8_int8_kernel(
    const int8_t* __restrict__ act,          // per-token quantized activations
    const int32_t* __restrict__ qweight,     // eight uint4 per word
    const float* __restrict__ act_scale,     // one per token (row)
    const float* __restrict__ w_scale,       // one per output channel (column)
    int32_t k_tiles, float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<int8_t(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<int32_t(*)[kBWords]>(smem + kOffB);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % tmpl::kWarpThreads;
  const int32_t warp = tid / tmpl::kWarpThreads;

  // int32 throughout: s8 x s8 -> s32 is exact, so nothing is lost by carrying
  // the whole K reduction in the integer domain.
  int32_t acc[4] = {0, 0, 0, 0};
  uint32_t frag_a[4];
  uint32_t frag_b[2];

  auto fetch = [&](int32_t t) {
    for (int32_t i = tid * 16; i < kAElems; i += kThreads * 16) {
      tmpl::cp_async_16(sa[t % kStages] + i,
                        act + static_cast<int64_t>(t) * kAElems + i);
    }
    for (int32_t i = tid * 4; i < kBWords; i += kThreads * 4) {
      tmpl::cp_async_16(sb[t % kStages] + i,
                        qweight + static_cast<int64_t>(t) * kBWords + i);
    }
  };

  for (int32_t s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) { fetch(s); }
    tmpl::cp_async_commit();
  }

  for (int32_t t = 0; t < k_tiles; ++t) {
    const int32_t stage = t % kStages;
    tmpl::cp_async_wait<kStages - 2>();
    __syncthreads();

    #pragma unroll
    for (int32_t kb = 0; kb < kTileK / 32; ++kb) {
      const int32_t base = (warp * 16 + lane % 16) * kTileK + kb * 32;
      *reinterpret_cast<int4*>(frag_a) =
          *reinterpret_cast<const int4*>(&sa[stage][base]);
      unpack_u4_to_s8(sb[stage][kb * tmpl::kWarpThreads + lane], frag_b);
      mma_s8_m16n8k32(frag_a, frag_b, acc);
    }

    __syncthreads();
    if (t + kStages - 1 < k_tiles) { fetch(t + kStages - 1); }
    tmpl::cp_async_commit();
  }

  // The whole dequantization, once: a per-row scale times a per-column scale.
  // They live on different axes, so this epilogue is the only place they can
  // meet -- which is exactly why the inner loop never saw either of them.
  const int32_t row = warp * 16 + lane / 4;
  const int32_t col = (lane % 4) * 2;
  const float s_row = act_scale[row];
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[row * kTileN + col + i] =
        static_cast<float>(acc[i]) * s_row * w_scale[col + i];
  }
}

// --------------------------------------------------------------- MXFP4 x A8

__global__ __launch_bounds__(kThreads, 1) void mxfp4_a8_kernel(
    const __nv_fp8_storage_t* __restrict__ act,  // e4m3 activations
    const int32_t* __restrict__ qweight,         // e2m1, pre-permuted
    const uint8_t* __restrict__ w_exp_delta,     // RESIDUAL exponent per block
    const float* __restrict__ act_scale,         // one per token
    float group_base_scale,                      // the factored-out group base
    int32_t k_tiles, float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<__nv_fp8_storage_t(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<int32_t(*)[kBWords]>(smem + kOffB);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % tmpl::kWarpThreads;
  const int32_t warp = tid / tmpl::kWarpThreads;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float final_acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t frag_a[4];
  uint32_t frag_b[2];

  auto fetch = [&](int32_t t) {
    for (int32_t i = tid * 16; i < kAElems; i += kThreads * 16) {
      tmpl::cp_async_16(sa[t % kStages] + i,
                        act + static_cast<int64_t>(t) * kAElems + i);
    }
    for (int32_t i = tid * 4; i < kBWords; i += kThreads * 4) {
      tmpl::cp_async_16(sb[t % kStages] + i,
                        qweight + static_cast<int64_t>(t) * kBWords + i);
    }
  };

  for (int32_t s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) { fetch(s); }
    tmpl::cp_async_commit();
  }

  for (int32_t t = 0; t < k_tiles; ++t) {
    const int32_t stage = t % kStages;
    tmpl::cp_async_wait<kStages - 2>();
    __syncthreads();

    #pragma unroll
    for (int32_t kb = 0; kb < kTileK / kMxBlock; ++kb) {
      // One MMA step per scale block: the block boundary and the promotion
      // boundary are the same line, so neither costs the other anything.
      acc[0] = acc[1] = acc[2] = acc[3] = 0.f;

      const int32_t base = (warp * 16 + lane % 16) * kTileK + kb * kMxBlock;
      *reinterpret_cast<int4*>(frag_a) =
          *reinterpret_cast<const int4*>(&sa[stage][base]);

      // The block scale rides INTO the table, so the dequantized operand is
      // already scaled.  It must be a residual: a full ue8m0 exponent folded
      // into an e4m3 weight overflows the format.
      const uint32_t delta =
          w_exp_delta[static_cast<int64_t>(t) * (kTileK / kMxBlock) + kb];
      tmpl::unpack_e2m1_to_e4m3x8(
          static_cast<uint32_t>(sb[stage][kb * tmpl::kWarpThreads + lane]),
          kE2m1IdentityOffset + delta, frag_b);
      mma_e4m3_m16n8k32(frag_a, frag_b, acc);

      // A plain add: unlike template 11, there is no scale left to apply here.
      // The promotion exists only to escape the fp8 accumulator's precision.
      #pragma unroll
      for (int32_t i = 0; i < 4; ++i) { final_acc[i] += acc[i]; }
    }

    __syncthreads();
    if (t + kStages - 1 < k_tiles) { fetch(t + kStages - 1); }
    tmpl::cp_async_commit();
  }

  const int32_t row = warp * 16 + lane / 4;
  const int32_t col = (lane % 4) * 2;
  // The group base exponent, factored out so the per-block deltas could stay
  // inside e4m3's range, is re-applied once here alongside the token scale.
  const float s_row = act_scale[row] * group_base_scale;
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[row * kTileN + col + i] = final_acc[i] * s_row;
  }
}

cudaError_t configure_w4a8() {
  auto rc = cudaFuncSetAttribute(w4a8_int8_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSmemBytes);
  if (rc != cudaSuccess) { return rc; }
  return cudaFuncSetAttribute(mxfp4_a8_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
