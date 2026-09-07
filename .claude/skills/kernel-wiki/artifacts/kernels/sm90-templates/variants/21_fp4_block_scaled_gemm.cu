// Template 21 -- NVFP4 / MXFP4 weights with bf16 activations (sm90).
//
// Both formats are e2m1 weights plus a scale per small block of K.  They differ
// in exactly two places, and those two differences drive everything else:
//
//                    block   scale type   scale is exact?   extra
//   NVFP4              16      ue4m3            no          per-tensor fp32
//   MXFP4              32      ue8m0            yes         none
//
//   * MXFP4's ue8m0 scale is an exponent BYTE -- no sign, no mantissa.  It IS
//     a bf16 exponent field, so converting it is a shift and multiplying by it
//     is exact.  That is the entire reason the MX spec chose e8m0.
//   * NVFP4's ue4m3 scale has a mantissa, so it needs a real multiply and a
//     second, per-tensor fp32 scale to recentre the whole range.  It buys
//     finer granularity (16 vs 32) with that arithmetic.
//
// The sm90 fact that shapes this template: THERE IS NO FP4 TENSOR CORE HERE.
// `mma_sm90_gmma.hpp` contains zero e2m1 atoms; native block-scaled MMA is
// sm100 (`mma_sm100_umma.hpp`) and sm120 (`mma_sm120.hpp`) only.  So on H100
// the block scale is not an instruction operand -- it is a multiply the kernel
// performs, and the unpack-plus-scale is real work on the critical path.  Fold
// it into as few instructions as possible: the e2m1 exponent-bias correction
// and the block scale are one multiply, not two.
//
// Porting this to sm120 is a DELETION, not a translation: the unpack, the
// scale multiply and this whole file go away and the scales become operands of
// the instruction.  Do not carry the sm90 shape forward as if it were the
// portable one.
//
// Both formats are instantiated below so both dequant paths are in the PTX.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-GRADE: structural
// CHECK-PTX: mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32
// CHECK-PTX: cp\.async\.cg\.shared\.global
// CHECK-PTX: ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16
// CHECK-PTX-COUNT: 2 cp\.async\.wait_group

#include <cuda_bf16.h>

#include "sm90_common.cuh"
#include "quant_sm90.cuh"

namespace {

enum class Fp4Format : int { kNvfp4 = 0, kMxfp4 = 1 };

__host__ __device__ constexpr int block_size(Fp4Format f) {
  return f == Fp4Format::kNvfp4 ? 16 : 32;
}

constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kStages = 4;
constexpr int kThreads = 128;
constexpr int kPackFactor = 8;  // eight e2m1 per 32-bit word

constexpr int kAElems = kTileM * kTileK;
constexpr int kBWords = kTileN * kTileK / kPackFactor;
// One scale byte per block of K per output column.
__host__ __device__ constexpr int kSfBytesFor(Fp4Format f) {
  return kTileN * kTileK / block_size(f);
}

constexpr int kABytes = kAElems * static_cast<int>(sizeof(__nv_bfloat16));
constexpr int kBBytes = kBWords * static_cast<int>(sizeof(int32_t));
constexpr int kSfMax = kTileN * kTileK / 16;  // the finer of the two grids

constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kABytes;
constexpr int kOffSf = kOffB + kStages * kBBytes;
constexpr int kSmemBytes = kOffSf + kStages * kSfMax;

__device__ __forceinline__ void mma_m16n8k16(const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The exponent-bias correction e2m1 -> bf16 is 2^(127 - 1).  It is a constant
// per format, so it multiplies into the block scale ONCE per block instead of
// once per weight -- the difference between one multiply per value and two.
__device__ __forceinline__ __nv_bfloat162 block_multiplier(
    Fp4Format format, int32_t packed_scale, float global_scale) {
  __nv_bfloat162 sf[2];
  if (format == Fp4Format::kMxfp4) {
    // Exact: a power of two, converted by a shift.  No global scale exists.
    tmpl::dequant_ue8m0_to_bf16x2(packed_scale, sf);
    return sf[0];
  }
  tmpl::dequant_e4m3_to_bf16x2(packed_scale, sf);
  // NVFP4's second level: one fp32 per tensor, folded in here so the inner
  // loop never sees it.
  return __hmul2(sf[0], __float2bfloat162_rn(global_scale));
}

}  // namespace

template <Fp4Format kFormat>
__global__ __launch_bounds__(kThreads, 1) void fp4_block_scaled_gemm_kernel(
    const __nv_bfloat16* __restrict__ act,
    const int32_t* __restrict__ qweight,   // e2m1, eight per word
    const uint8_t* __restrict__ block_sf,  // ue4m3 (NVFP4) or ue8m0 (MXFP4)
    float global_scale,                    // NVFP4 only; 1.0f for MXFP4
    int32_t k_tiles, float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<__nv_bfloat16(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<int32_t(*)[kBWords]>(smem + kOffB);
  auto* const sf = reinterpret_cast<uint8_t(*)[kSfMax]>(smem + kOffSf);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % tmpl::kWarpThreads;
  const int32_t warp = tid / tmpl::kWarpThreads;

  constexpr int kSfBytes = kSfBytesFor(kFormat);
  // How many MMA k-steps share one scale block: this is the only place the
  // block size enters the loop structure.
  constexpr int kStepsPerBlock = block_size(kFormat) / 16;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t frag_a[4];
  uint32_t frag_b[2];

  auto fetch = [&](int32_t t) {
    constexpr int kHalfPerThread = 8;
    for (int32_t i = tid * kHalfPerThread; i < kAElems;
         i += kThreads * kHalfPerThread) {
      tmpl::cp_async_16(sa[t % kStages] + i,
                        act + static_cast<int64_t>(t) * kAElems + i);
    }
    for (int32_t i = tid * 4; i < kBWords; i += kThreads * 4) {
      tmpl::cp_async_16(sb[t % kStages] + i,
                        qweight + static_cast<int64_t>(t) * kBWords + i);
    }
    for (int32_t i = tid * 16; i < kSfBytes; i += kThreads * 16) {
      tmpl::cp_async_16(sf[t % kStages] + i,
                        block_sf + static_cast<int64_t>(t) * kSfBytes + i);
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

    tmpl::ldmatrix_x4(frag_a, &sa[stage][(warp * 16 + lane % 16) * kTileK +
                                         (lane / 16) * 8]);

    #pragma unroll
    for (int32_t kb = 0; kb < kTileK / 16; ++kb) {
      // One scale fetch per BLOCK, not per k-step: at MXFP4's 32-element block
      // that halves the scale traffic relative to NVFP4's 16.
      const int32_t sf_word =
          *reinterpret_cast<const int32_t*>(&sf[stage][(kb / kStepsPerBlock) * 4]);
      const __nv_bfloat162 mult =
          block_multiplier(kFormat, sf_word, global_scale);

      const int32_t q = sb[stage][kb * tmpl::kWarpThreads + lane];
      __nv_bfloat162 w[2];
      tmpl::dequant_e2m1_to_bf16x2(q, w);
      w[0] = __hmul2(w[0], mult);
      w[1] = __hmul2(w[1], mult);

      frag_b[0] = *reinterpret_cast<const uint32_t*>(&w[0]);
      frag_b[1] = *reinterpret_cast<const uint32_t*>(&w[1]);
      mma_m16n8k16(frag_a, frag_b, acc);
    }

    __syncthreads();
    if (t + kStages - 1 < k_tiles) { fetch(t + kStages - 1); }
    tmpl::cp_async_commit();
  }

  const int32_t col = warp * 8 + lane % 4;
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    out[col * 4 + i] = acc[i];
  }
}

// Both instantiated so both scale paths reach the PTX.
template __global__ void fp4_block_scaled_gemm_kernel<Fp4Format::kNvfp4>(
    const __nv_bfloat16*, const int32_t*, const uint8_t*, float, int32_t, float*);
template __global__ void fp4_block_scaled_gemm_kernel<Fp4Format::kMxfp4>(
    const __nv_bfloat16*, const int32_t*, const uint8_t*, float, int32_t, float*);

cudaError_t configure_fp4_block_scaled() {
  auto rc = cudaFuncSetAttribute(fp4_block_scaled_gemm_kernel<Fp4Format::kNvfp4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSmemBytes);
  if (rc != cudaSuccess) { return rc; }
  return cudaFuncSetAttribute(fp4_block_scaled_gemm_kernel<Fp4Format::kMxfp4>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
