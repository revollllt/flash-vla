// Pi0's action-expert QKV projection as one kernel: the RMS scale, the
// projection, the RoPE rotation and the Q/K/V scatter.
//
// The shipped route spells this as three launches -- `rms_norm`, cuBLAS, then
// `rope_scatter` -- and the call site measures 12.21 us against a 7.05 us
// ceiling built from [ld.bw.dev.dram] over its 5.37 MB. Timed with the
// launches amortized the three cost 4.11 + 8.89 + 4.64 us, and the two
// pointwise kernels move only 0.73 MB between them: what they are paying is
// the fixed cost of touching DRAM at all, three times over rather than once.
// Only fusing removes it. cuBLAS itself is at 76% of the 6.79 us its weight
// read alone costs, so the GEMM is not where the gap is.
//
// The norm is what makes the fusion possible. Pi0's expert RMSNorm has no
// learnable gain, so it is a per-row scalar and commutes with the projection:
//
//     rms(x) @ W  ==  diag(s) @ (x @ W),   s_m = rsqrt(mean_k(x[m,k]^2) + eps)
//
// so this kernel never materializes the normalized activations at all. It
// accumulates each row's sum of squares while streaming x for the GEMM it was
// going to do anyway, and applies s in the epilogue. That also drops one bf16
// rounding the shipped route pays, since the shipped `normed` buffer is bf16.
//
// Tiling, and why. A CTA owns all rows and 32 output columns, which is four
// mma N tiles, one per warp. 32 keeps the per-CTA weight slice at 1024 x 32
// and gives 2560 / 32 = 80 CTAs -- enough concurrent warps to keep the weight
// stream in flight, which is what this kernel is actually bound by, without
// re-reading x more times than necessary (each CTA reads all of x, so CTA
// count is also x's L2 amplification factor). The K dimension moves in chunks
// of 64 so that x and the transposed weight tile together take 14 KB of
// shared memory.
//
// The two neighbouring tilings were measured and are both worse, against the
// three-launch chain's 13.8 us at this shape:
//
//     kTileN  kChunkK  threads  CTAs  warps/CTA   fused
//         16       32      256   160          8   12.40 us
//         32       64      512    80         16    8.29 us
//         64      128     1024    40         32   12.40 us
//
// Warps per scheduler is what moves this, not the number of SMs given work:
// the 160-CTA form spreads over twice the part and still loses. The sweep was
// re-run after the weight staging changed and the middle row is still best.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mma_bf16.cuh"

namespace {

using flash_vla::rtx5090::ldmatrix_a;
using flash_vla::rtx5090::ldmatrix_b_trans;
using flash_vla::rtx5090::mma_m16n8k16;

//: Inside the mean, matching `tl_rms_factor` and the `rms_norm` kernel.
constexpr float kRmsEps = 1e-6f;

constexpr int32_t kTileM = 64;   // four mma M tiles; Pi0's expert sends 51 rows
constexpr int32_t kTileN = 32;   // four mma N tiles, one per warp
constexpr int32_t kChunkK = 64;  // four mma K steps per shared-memory stage
constexpr int32_t kMTiles = kTileM / 16;
constexpr int32_t kNTiles = kTileN / 8;
// One mma tile per warp rather than a column of four. Four warps per CTA put a
// single warp on each scheduler, and with nothing else to issue every shared
// load and every mma stood at its full latency -- that form measured 16.49 us
// against the three-launch chain's 13.97. Sixteen warps cost 1.6x more
// `ldmatrix` across the CTA, since a fragment is now fetched by each warp that
// needs it rather than reused down a column, and buy 4x the warps per
// scheduler to cover it.
constexpr int32_t kWarps = kMTiles * kNTiles;
constexpr int32_t kThreads = kWarps * 32;
constexpr int32_t kVec = 8;  // bf16 per 16-byte access

// Row strides padded so a fragment load spreads over all 32 banks: the stride
// in banks is ld/2, and 72/2 = 36 == 4 (mod 32), so the eight row groups of an
// A or B fragment land 4 banks apart and cover all of them.
constexpr int32_t kLdX = kChunkK + 8;
//: The weight tile is staged in its natural [k][n] order and transposed by
//: `ldmatrix.trans` on the way into the fragment, so this stride is over n.
//: 40 puts the eight k rows an `ldmatrix` gathers on eight different bank
//: groups, covering all 32.
constexpr int32_t kLdW = kTileN + 8;

//: x vectors per shared stage -- exactly one per thread at this tiling.
constexpr int32_t kXVecs = kTileM * kChunkK / kVec;
static_assert(kXVecs == kThreads, "x staging assumes one vector per thread");
//: weight vectors per shared stage; fewer than there are threads, so this one
//: is a guarded stride loop rather than a fixed count.
constexpr int32_t kWVecs = kChunkK * kTileN / kVec;
//: Threads sharing one x row while staging, an aligned octet of one warp.
constexpr int32_t kRowThreads = kChunkK / kVec;

__device__ __forceinline__ float sumsq(const int4 &v) {
  const __nv_bfloat16 *e = reinterpret_cast<const __nv_bfloat16 *>(&v);
  float s = 0.f;
#pragma unroll
  for (int32_t j = 0; j < kVec; ++j) {
    const float f = __bfloat162float(e[j]);
    s += f * f;
  }
  return s;
}

// Q/K/V = scatter(rope(diag(s) * (x @ w))), all in one launch.
//
// `x` is (m, k) bf16 and `w` is (k, n) bf16 with n = q_dim + 2 * head_dim.
// `rope` is (m, head_dim) holding interleaved [cos, sin] pairs; the rotation
// pairs ADJACENT columns, which is exactly the pair an mma D fragment lane
// already holds, so the epilogue rotates without leaving registers. `q` is
// (m, q_dim), `k_out` and `v_out` are (m, head_dim), all written in place.
__global__ __launch_bounds__(kThreads) void expert_qkv_kernel(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w,
    const __nv_bfloat16 *__restrict__ rope, __nv_bfloat16 *__restrict__ q,
    __nv_bfloat16 *__restrict__ k_out, __nv_bfloat16 *__restrict__ v_out,
    int32_t m, int32_t kdim, int32_t n, int32_t q_dim, int32_t head_dim) {
  __shared__ __nv_bfloat16 xs[kTileM * kLdX];
  __shared__ __nv_bfloat16 ws[kChunkK * kLdW];
  __shared__ float row_sq[kTileM], row_scale[kTileM];

  const int32_t n0 = blockIdx.x * kTileN;
  const int32_t m0 = blockIdx.y * kTileM;
  const int32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  if (threadIdx.x < kTileM) row_sq[threadIdx.x] = 0.f;

  //: This warp's single output tile: rows [mt*16, +16), columns [nt*8, +8).
  const int32_t mt = warp / kNTiles, nt = warp % kNTiles;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  __syncthreads();

  // Both tiles are staged global -> register -> shared one K chunk ahead. The
  // unpipelined form issued a chunk's loads, waited the full DRAM latency and
  // only then had 12 instructions of mma to run, sixteen times over; this puts
  // the next chunk's loads in flight before the current chunk's compute.
  //: x: exactly one 16-byte vector per thread. w: fewer vectors than threads.
  const int32_t xr = threadIdx.x / kRowThreads;
  const int32_t xd = (threadIdx.x % kRowThreads) * kVec;
  const int32_t wk = threadIdx.x / (kTileN / kVec);
  const int32_t wn = (threadIdx.x % (kTileN / kVec)) * kVec;
  int4 xreg, wreg = make_int4(0, 0, 0, 0);

  auto stage = [&](int32_t k0) {
    xreg = make_int4(0, 0, 0, 0);
    if (m0 + xr < m)
      xreg = *(const int4 *)(x + int64_t(m0 + xr) * kdim + k0 + xd);
    if (threadIdx.x < kWVecs)
      wreg = *(const int4 *)(w + int64_t(k0 + wk) * n + n0 + wn);
  };
  stage(0);

  for (int32_t k0 = 0; k0 < kdim; k0 += kChunkK) {
    // Publish x. The sum of squares this row needs for the norm is accumulated
    // here rather than in a separate pass: the bytes are already in registers,
    // and the eight threads holding one row are an aligned lane octet, so the
    // row reduction is three shuffles and one thread's update per row.
    *(int4 *)(&xs[xr * kLdX + xd]) = xreg;
    {
      float sq = sumsq(xreg);
#pragma unroll
      for (int32_t o = 1; o < kRowThreads; o <<= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
      if ((threadIdx.x % kRowThreads) == 0) row_sq[xr] += sq;
    }

    // Publish the weight in its natural [k][n] order: one 16-byte store, and
    // `ldmatrix.trans` does the transpose the `.col` B operand needs.
    if (threadIdx.x < kWVecs) *(int4 *)(&ws[wk * kLdW + wn]) = wreg;
    __syncthreads();
    if (k0 + kChunkK < kdim) stage(k0 + kChunkK);

#pragma unroll
    for (int32_t kk = 0; kk < kChunkK; kk += 16) {
      uint32_t a[4], b[2];
      ldmatrix_a(a, xs + mt * 16 * kLdX, kLdX, kk, lane);
      ldmatrix_b_trans(b, ws, kLdW, nt * 8, kk, lane);
      mma_m16n8k16(acc, a, b);
    }
    __syncthreads();
  }

  if (threadIdx.x < kTileM)
    row_scale[threadIdx.x] =
        rsqrtf(row_sq[threadIdx.x] / float(kdim) + kRmsEps);
  __syncthreads();

  // Epilogue: scale by the row's norm, rotate the pair this lane holds, and
  // send it to whichever of Q, K or V owns the column.
  const int32_t half_hd = head_dim >> 1;
  const int32_t col = n0 + nt * 8 + (lane & 3) * 2;
#pragma unroll
  for (int32_t h = 0; h < 2; ++h) {
    {
      const int32_t r = mt * 16 + (lane >> 2) + h * 8;
      const int32_t row = m0 + r;
      if (row >= m) continue;
      const float sc = row_scale[r];
      float v0 = acc[h * 2] * sc, v1 = acc[h * 2 + 1] * sc;
      if (col < q_dim + head_dim) {
        // The cos/sin table repeats every head_dim columns.
        const __nv_bfloat162 cs = ((const __nv_bfloat162 *)(
            rope + int64_t(row) * head_dim))[(col >> 1) % half_hd];
        const float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
        const float r0 = v0 * c - v1 * s;
        v1 = v1 * c + v0 * s;
        v0 = r0;
      }
      const __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
      if (col < q_dim)
        ((__nv_bfloat162 *)(q + int64_t(row) * q_dim))[col >> 1] = o;
      else if (col < q_dim + head_dim)
        ((__nv_bfloat162 *)(k_out + int64_t(row) * head_dim))
            [(col - q_dim) >> 1] = o;
      else
        ((__nv_bfloat162 *)(v_out + int64_t(row) * head_dim))
            [(col - q_dim - head_dim) >> 1] = o;
    }
  }
}

}  // namespace

extern "C" int expert_qkv_launch(const void *x, const void *w, const void *rope,
                                 void *q, void *k, void *v, int m, int kdim,
                                 int n, int q_dim, int head_dim, void *stream) {
  // The tiling assumes the shapes Pi0's expert actually sends. Anything else is
  // a routing error rather than something to handle slowly.
  if (n % kTileN || kdim % kChunkK || head_dim % 16 || q_dim % kTileN)
    return cudaErrorInvalidValue;
  const dim3 grid(n / kTileN, (m + kTileM - 1) / kTileM);
  expert_qkv_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)w,
      (const __nv_bfloat16 *)rope, (__nv_bfloat16 *)q, (__nv_bfloat16 *)k,
      (__nv_bfloat16 *)v, m, kdim, n, q_dim, head_dim);
  return (int)cudaGetLastError();
}
