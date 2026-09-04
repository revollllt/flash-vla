// Vocabulary for memory-bound elementwise and row-reduction kernels on sm90.
//
// These ops -- norm, activation, rope, quantization, softmax -- are the glue
// between the GEMMs, and they are bound by two things that have nothing to do
// with arithmetic:
//
//   1. BYTES.  The roofline is the row read plus the row written, so the only
//      levers are vector width, how many times the row is traversed, and how
//      much of the pipeline is fused into one traversal.  A separate kernel per
//      op re-reads the row every time.
//   2. LAUNCHES.  At decode shapes a row is a few KB, so the kernel runs for
//      microseconds and a launch costs 1.24 us [launch.lat.dev.ramp].  This is
//      why FlashInfer puts griddepcontrol in EVERY one of these kernels rather
//      than in the interesting ones: on a chain of small ops the launch ramp is
//      a first-class term, and PDL is what removes it.
//
// Method follows FlashInfer (Apache-2.0) `include/flashinfer/{norm,activation,
// pos_enc}.cuh` and SGLang's per-token quantization kernels.

#pragma once

#include <cuda_bf16.h>
#include <cstdint>

#include "sm90_common.cuh"

namespace tmpl {

// 16 bytes is the widest load a thread can issue and the width that saturates
// the memory pipe; every one of these kernels is written in units of it.
template <class T>
inline constexpr int vec_size_of = 16 / static_cast<int>(sizeof(T));

// Loads in the storage dtype, computes in fp32.  The upcast is free next to the
// load, and doing the math in the storage dtype is how norms lose accuracy that
// no epsilon recovers.
template <class T, int N>
struct FloatVec {
  float data[N];

  __device__ __forceinline__ void cast_load(const T* p) {
    // One 16-byte transaction, then widen in registers.
    int4 raw = *reinterpret_cast<const int4*>(p);
    const T* v = reinterpret_cast<const T*>(&raw);
    #pragma unroll
    for (int i = 0; i < N; ++i) { data[i] = static_cast<float>(v[i]); }
  }

  __device__ __forceinline__ void cast_store(T* p) const {
    int4 raw;
    T* v = reinterpret_cast<T*>(&raw);
    #pragma unroll
    for (int i = 0; i < N; ++i) { v[i] = static_cast<T>(data[i]); }
    *reinterpret_cast<int4*>(p) = raw;
  }

  __device__ __forceinline__ void fill(float x) {
    #pragma unroll
    for (int i = 0; i < N; ++i) { data[i] = x; }
  }

  __device__ __forceinline__ float& operator[](int i) { return data[i]; }
  __device__ __forceinline__ float operator[](int i) const { return data[i]; }
};

// ------------------------------------------------------------- reductions

// Butterfly: every lane ends with the result, so no broadcast follows.  A
// tree reduction would leave it in lane 0 and cost a shuffle to spread.
__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

// Two-level: warps reduce internally, one slot each in shared, then warp 0
// reduces those.  `smem` needs one float per warp.  Correct for up to 32 warps,
// which is the whole CTA range.
//
// CALLING IT TWICE ON THE SAME `smem` NEEDS A __syncthreads() BETWEEN.  This
// returns after a barrier but the read of smem[0] is not itself fenced, so a
// fast warp can reach the next call and overwrite smem[0] while a slow warp is
// still reading it.  A two-stage reduction -- a max then a sum, as softmax
// needs -- hits this every time.
template <bool kIsMax>
__device__ __forceinline__ float block_reduce(float v, float* smem, int num_warps) {
  const int lane = static_cast<int>(threadIdx.x) % kWarpThreads;
  const int warp = static_cast<int>(threadIdx.x) / kWarpThreads;
  v = kIsMax ? warp_reduce_max(v) : warp_reduce_sum(v);
  if (lane == 0) { smem[warp] = v; }
  __syncthreads();
  if (warp == 0) {
    float x = (lane < num_warps) ? smem[lane] : (kIsMax ? -INFINITY : 0.f);
    x = kIsMax ? warp_reduce_max(x) : warp_reduce_sum(x);
    if (lane == 0) { smem[0] = x; }
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------- shaping

// One warp per row, or one CTA per row?  The whole question is whether a row
// has enough 16-byte vectors to keep a CTA busy.  A warp-per-row kernel needs
// no shared memory and no __syncthreads, so it wins on short rows and on rows
// that fit a single traversal; a CTA-per-row kernel is the only option once a
// row exceeds what 32 lanes can hold in registers.
//
// The crossover is a measurement, not a constant -- but the shape of the rule
// is: warp-per-row while row_bytes / 32 lanes stays within a few vectors.
__device__ __forceinline__ constexpr bool prefer_warp_per_row(int row_elems,
                                                              int vec_size) {
  return row_elems <= 8 * kWarpThreads * vec_size;
}

// fp8 e4m3's finite maximum.  A per-token scale is amax / this, so a row of
// zeros must not divide.
inline constexpr float kFp8E4M3Max = 448.f;

}  // namespace tmpl
