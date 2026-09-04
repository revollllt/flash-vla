// Template 31 -- SwiGLU fused with per-token fp8 quantization (sm90).
//
// The activation between an FFN's two GEMMs, and the quantization the second
// GEMM wants its input in.  Kept apart they cost four traversals of the row:
// SwiGLU reads 2d and writes d, then the quantizer reads d for the amax, reads
// d again, and writes d/2.  Fused they cost one read of 2d and one write of
// d/2 -- and one launch instead of two [launch.lat.dev.ramp].
//
//   1. The gate and up halves are CONCATENATED, not interleaved: one token's
//      row is [gate(d) | up(d)].  So the two loads are `row + i` and
//      `row + d + i`, a fixed stride apart, and both are fully coalesced.  A
//      layout that interleaved them would halve the useful bytes per
//      transaction.
//   2. Quantization needs the whole row before it can scale anything, which is
//      the real obstacle to fusing.  The answer is to keep the SwiGLU result in
//      REGISTERS across the amax reduction.  That works while the row fits the
//      CTA's register budget; past that the fused kernel must fall back to a
//      second traversal, and the second traversal is at least L2-resident.
//   3. scale = amax / 448, and 448 is e4m3's finite max, not its infinity.  A
//      row of exact zeros must not produce a zero divisor -- that path yields
//      NaN in the GEMM that consumes it, far from here.
//   4. SiLU is computed in fp32.  x * sigmoid(x) in bf16 loses the small-x
//      behaviour that makes SwiGLU a gate at all.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: shfl\.sync\.bfly
// CHECK-PTX: ld\.global(\.nc)?\.v4
// CHECK-PTX: cvt\.rn\.satfinite.*e4m3

#include <cuda_fp8.h>

#include "elementwise_sm90.cuh"

namespace {

using Element = __nv_bfloat16;
constexpr int kVec = tmpl::vec_size_of<Element>;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;
// Vectors each thread holds across the reduction.  This IS the fusion budget:
// d = kVecsPerThread * kThreads * kVec.  Raising it past the register file
// forces the fallback traversal.
constexpr int kVecsPerThread = 2;
constexpr int kRowElems = kVecsPerThread * kThreads * kVec;  // 4096

__device__ __forceinline__ float silu(float x) {
  return x / (1.f + __expf(-x));
}

}  // namespace

// One CTA per token.  Output is fp8 plus one fp32 scale per token, which is
// exactly what template 22's A8 GEMM expects on its activation side.
__global__ __launch_bounds__(kThreads) void swiglu_fp8_quant_kernel(
    const Element* __restrict__ input,       // (tokens, 2 * d): [gate | up]
    __nv_fp8_storage_t* __restrict__ output, // (tokens, d)
    float* __restrict__ out_scale,           // (tokens,)
    int32_t d) {
  __shared__ float smem[kWarps];

  const int64_t row = static_cast<int64_t>(blockIdx.x) * 2 * d;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  tmpl::pdl_wait();

  // Held across the reduction; this is what buys the fusion.
  tmpl::FloatVec<Element, kVec> acc[kVecsPerThread];
  float amax = 0.f;

  #pragma unroll
  for (int32_t v = 0; v < kVecsPerThread; ++v) {
    const int32_t i = (v * kThreads + tid) * kVec;
    tmpl::FloatVec<Element, kVec> gate, up;
    gate.cast_load(input + row + i);
    // The up half is exactly d elements away -- one add, still coalesced.
    up.cast_load(input + row + d + i);
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      acc[v][j] = silu(gate[j]) * up[j];
      amax = fmaxf(amax, fabsf(acc[v][j]));
    }
  }

  // The input row is fully consumed; the reduction and the fp8 stores below are
  // tail the dependent can overlap.
  tmpl::pdl_trigger();

  amax = tmpl::block_reduce<true>(amax, smem, kWarps);

  // A zero row would otherwise divide by zero and poison the consuming GEMM.
  const float scale = amax / tmpl::kFp8E4M3Max;
  const float inv = (scale > 0.f) ? 1.f / scale : 0.f;
  if (tid == 0) { out_scale[blockIdx.x] = scale; }

  const int64_t out_row = static_cast<int64_t>(blockIdx.x) * d;
  #pragma unroll
  for (int32_t v = 0; v < kVecsPerThread; ++v) {
    const int32_t i = (v * kThreads + tid) * kVec;
    #pragma unroll
    for (int32_t j = 0; j < kVec; ++j) {
      // satfinite, not the default: a value above 448 must clamp, not become
      // inf, or the GEMM downstream sees an operand it cannot accumulate.
      output[out_row + i + j] =
          __nv_cvt_float_to_fp8(acc[v][j] * inv, __NV_SATFINITE, __NV_E4M3);
    }
  }

}

static_assert(kRowElems > 0, "row must be a whole number of per-thread vectors");
