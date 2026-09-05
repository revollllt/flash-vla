// Template 20 -- Marlin-style W4A16 mixed-precision GEMM (sm90).
//
// The archetype for quantized decode: bf16 activations, INT4 weights, batch in
// the single digits to low tens.  At that shape the GEMM is not compute-bound
// and not activation-bound -- it is bound by streaming the weight matrix once.
// Every design decision follows from that:
//
//   1. mma.sync, not wgmma.  A warpgroup instruction wants a wide N and a
//      shared-memory operand; here N is the batch, often below 32, where
//      mma.sync is already the better instruction [mma.xover.n.wgmma], and the
//      weights must pass through registers to be dequantized anyway.  Marlin's
//      whole inner loop is m16n8k16.
//   2. cp.async, not TMA.  Weights are PERMUTED OFFLINE so that the 16 bytes a
//      lane copies are exactly the operand its mma wants -- no ldmatrix, no
//      shuffle, no swizzle agreement.  That layout is not a rectangular tile,
//      so there is no tensor map to describe it.  Activations, which are a
//      normal tile, still use ldmatrix.
//   3. Dequantization must be free.  It sits between the shared tile and the
//      MMA on the critical path, so it is bit arithmetic (quant_sm90.cuh), not
//      conversion instructions, and its cost hides under the weight stream.
//   4. Scales are per group along K, so they are loaded once per group and
//      reused across the whole M tile -- a scale load per element would cost
//      more than the weights.
//
// The measure of success is not TFLOP/s: it is achieving DRAM bandwidth on the
// weight stream [ld.bw.dev.dram]. A W4A16 kernel at 40% of memory bandwidth is
// broken no matter what its FLOP number says.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// Numerical equivalence to a dequantize-then-GEMM reference needs a parity
// harness -- these bit tricks are exactly where sign and bias errors hide.
//
// CHECK-GRADE: structural
// CHECK-PTX: mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32
// CHECK-PTX: lop3\.b32
// CHECK-PTX: cp\.async\.cg\.shared\.global
// CHECK-PTX: ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16
// CHECK-PTX: cp\.async\.wait_group

#include <cuda_bf16.h>

#include "sm90_common.cuh"
#include "quant_sm90.cuh"

namespace {

constexpr int kTileM = 64;    // activation rows (the batch dimension)
constexpr int kTileN = 128;   // output columns owned by this CTA
constexpr int kTileK = 64;
constexpr int kStages = 4;    // deep enough to cover a cold weight stream

// Scales are shared along K within a group; 128 is the common GPTQ/AWQ choice.
constexpr int kGroupK = 128;

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;

// Eight uint4 per 32-bit word is what makes the weight stream worth the
// trouble: one int32 carries eight weights.
constexpr int kPackFactor = 8;

constexpr int kAElems = kTileM * kTileK;
constexpr int kBWords = kTileN * kTileK / kPackFactor;
constexpr int kABytes = kAElems * static_cast<int>(sizeof(__nv_bfloat16));
constexpr int kBBytes = kBWords * static_cast<int>(sizeof(int32_t));
constexpr int kSBytes = kTileN * static_cast<int>(sizeof(__nv_bfloat16));

constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kABytes;
constexpr int kOffS = kOffB + kStages * kBBytes;
constexpr int kSmemBytes = kOffS + kStages * kSBytes;

// m16n8k16 bf16: A is 4 words (8 values), B is 2 words (4 values), C is 4 floats.
__device__ __forceinline__ void mma_m16n8k16(const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One stage's fetch.  Each thread moves 16 bytes at a time, which is the widest
// cp.async and the only form that bypasses L1.  The three streams have very
// different sizes, so each gets its own strided loop rather than one fused one.
__device__ __forceinline__ void cp_async_stage(
    __nv_bfloat16* sa_stage, int32_t* sb_stage, __nv_bfloat16* ss_stage,
    const __nv_bfloat16* __restrict__ act, const int32_t* __restrict__ qweight,
    const __nv_bfloat16* __restrict__ scales, int32_t t, int32_t tid,
    int32_t n_size) {
  constexpr int kPerThread = 16 / static_cast<int>(sizeof(__nv_bfloat16));  // 8
  #pragma unroll
  for (int32_t i = tid * kPerThread; i < kAElems; i += kThreads * kPerThread) {
    tmpl::cp_async_16(sa_stage + i, act + static_cast<int64_t>(t) * kAElems + i);
  }
  constexpr int kWordsPerThread = 16 / static_cast<int>(sizeof(int32_t));  // 4
  #pragma unroll
  for (int32_t i = tid * kWordsPerThread; i < kBWords;
       i += kThreads * kWordsPerThread) {
    tmpl::cp_async_16(sb_stage + i, qweight + static_cast<int64_t>(t) * kBWords + i);
  }
  // Scales advance only once per group, so most stages re-copy the same row.
  // Cheap enough at kTileN entries that a predicate would cost more than it
  // saves.
  const int32_t group = (t * kTileK) / kGroupK;
  for (int32_t i = tid * kPerThread; i < kTileN; i += kThreads * kPerThread) {
    tmpl::cp_async_16(ss_stage + i,
                      scales + static_cast<int64_t>(group) * n_size + i);
  }
}

}  // namespace

__global__ __launch_bounds__(kThreads, 1) void marlin_w4a16_kernel(
    const __nv_bfloat16* __restrict__ act,   // (M, K) bf16
    const int32_t* __restrict__ qweight,     // permuted, 8 uint4 per word
    const __nv_bfloat16* __restrict__ scales,// (K / kGroupK, N)
    int32_t k_tiles, int32_t n_size, float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<__nv_bfloat16(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<int32_t(*)[kBWords]>(smem + kOffB);
  auto* const ss = reinterpret_cast<__nv_bfloat16(*)[kTileN]>(smem + kOffS);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid % tmpl::kWarpThreads;
  const int32_t warp = tid / tmpl::kWarpThreads;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t frag_a[4];
  uint32_t frag_b[2];

  // Prologue: fill the pipeline before the first math, so the mainloop never
  // sees an empty stage.
  for (int32_t s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) {
      // 16 bytes per thread is the widest cp.async and the only .cg form.
      cp_async_stage(sa[s], sb[s], ss[s], act, qweight, scales, s, tid, n_size);
    }
    tmpl::cp_async_commit();
  }

  for (int32_t t = 0; t < k_tiles; ++t) {
    const int32_t stage = t % kStages;

    // Keep kStages-2 groups in flight: waiting for zero would serialize the
    // weight stream against the math it is meant to hide under.
    tmpl::cp_async_wait<kStages - 2>();
    __syncthreads();

    // Activations are an ordinary tile, so ldmatrix does the lane transpose.
    tmpl::ldmatrix_x4(frag_a, &sa[stage][(warp * 16 + lane % 16) * kTileK +
                                         (lane / 16) * 8]);

    // One scale per group, reused across the whole M tile.
    const __nv_bfloat162 group_scale =
        __bfloat162bfloat162(ss[stage][(lane % 4) * 2]);

    #pragma unroll
    for (int32_t kb = 0; kb < kTileK / 16; ++kb) {
      // The permuted layout means this word IS this lane's operand: no
      // ldmatrix and no shuffle between the load and the MMA.
      const int32_t q = sb[stage][(kb * tmpl::kWarpThreads + lane)];

      __nv_bfloat162 w[2];
      tmpl::dequant_u4_to_bf16x2(q, w);
      // Scale after the bias subtract, never before: the -8 is in the
      // quantized domain and the scale is not.
      w[0] = __hmul2(w[0], group_scale);
      w[1] = __hmul2(w[1], group_scale);

      frag_b[0] = *reinterpret_cast<const uint32_t*>(&w[0]);
      frag_b[1] = *reinterpret_cast<const uint32_t*>(&w[1]);
      mma_m16n8k16(frag_a, frag_b, acc);
    }

    __syncthreads();
    // Refill the stage this iteration just drained.
    const int32_t fetch = t + kStages - 1;
    if (fetch < k_tiles) {
      cp_async_stage(sa[fetch % kStages], sb[fetch % kStages], ss[fetch % kStages],
                     act, qweight, scales, fetch, tid, n_size);
    }
    tmpl::cp_async_commit();
  }

  const int32_t col = warp * 8 + lane % 4;
  #pragma unroll
    
  for (int32_t i = 0; i < 4; ++i) {
    out[col * 4 + i] = acc[i];
  }
}

cudaError_t configure_marlin_w4a16() {
  return cudaFuncSetAttribute(marlin_w4a16_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
