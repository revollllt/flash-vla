// A bf16 GEMM for the projection call sites cuBLAS cannot fill the part with.
//
// `lab/sm120/pi0_gemm_wave_probe.py` establishes the opening: at M=768 the
// shapes Pi0 uses take the SAME time as the same shapes at M=1024 --
// 768 x 2048 x 2048 measures 49.42 us and 1024 x 2048 x 2048 measures 49.50 --
// because both are one wave of 128 x 128 tiles and the M=768 wave is 96 CTAs
// on a 170-SM part, 56% occupied. The idle 44% is what nine of the twelve
// remaining call-site gaps are made of.
//
// Two things that do not recover it, both measured: a batched K split is
// slower in every case before its partial sum is even added
// (`pi0_gemm_splitk_probe.py`), and storing the weight K-contiguous so cuBLAS
// picks a TN kernel buys 1.21x out of bf16 split-K accumulation rather than
// out of the layout, which the model's correctness gate rejects.
//
// So the tile shape has to come from a kernel that chooses it. This is that
// kernel, built from the same three things that took the fused QKV projection
// from 20.49 to 8.29 us: `ldmatrix` for the fragments, one job per warp so a
// scheduler has more than one warp to issue from, and each K chunk staged in
// registers a full iteration ahead of the shared memory that consumes it.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mma_bf16.cuh"

namespace {

using flash_vla::rtx5090::ldmatrix_a;
using flash_vla::rtx5090::ldmatrix_b_trans;
using flash_vla::rtx5090::mma_m16n8k16;

// Tile, swept against cuBLAS at the deployed shapes. 64 x 64 is the only one
// that wins anything: it puts 384 CTAs on the 768 x 2048 output where 128 x 128
// puts 96, and several CTAs per SM is what covers the latency a single warp
// per scheduler cannot.
//
//   tile      backbone out_proj   vision out_proj   backbone gate/up
//   128x128   0.86x               0.48x             0.96x
//    64x128   0.89x               0.89x             0.95x
//   128x64    0.89x               0.80x             0.93x
//    64x64    1.14x               0.89x             0.92x
//
// The kernel is routed only where it wins, which is two backbone shapes.
constexpr int32_t kBM = 64, kBN = 64, kBK = 32;
constexpr int32_t kWarpsM = 2, kWarpsN = 4;
constexpr int32_t kWarps = kWarpsM * kWarpsN;
constexpr int32_t kThreads = kWarps * 32;
//: Each warp owns 64 x 32 of the tile: four mma M tiles by four N tiles, so
//: eight `ldmatrix` feed sixteen `mma` per K step.
constexpr int32_t kWM = kBM / kWarpsM, kWN = kBN / kWarpsN;
constexpr int32_t kMT = kWM / 16, kNT = kWN / 8;
constexpr int32_t kVec = 8;

// Padded so the eight rows an `ldmatrix` gathers start in eight different bank
// groups, and so every row start stays 16-byte aligned as `ldmatrix` requires.
// A is staged [m][k] and read directly; B is staged [k][n], its natural order,
// and transposed into the `.col` operand by `ldmatrix.trans`.
constexpr int32_t kLdA = kBK + 8;
constexpr int32_t kLdB = kBN + 8;

constexpr int32_t kAVecs = kBM * kBK / kVec;
constexpr int32_t kBVecs = kBK * kBN / kVec;
constexpr int32_t kAPer = kAVecs / kThreads;
constexpr int32_t kBPer = kBVecs / kThreads;
static_assert(kAPer * kThreads == kAVecs && kBPer * kThreads == kBVecs,
              "staging assumes a whole number of vectors per thread");
//: Rows (A) and k lines (B) one pass of the staging loop covers.
constexpr int32_t kAStep = kBM / kAPer;
constexpr int32_t kBStep = kBK / kBPer;

// c[m, n] = res[m, n] + sum_k a[m, k] * b[k, n], all bf16, accumulated in fp32.
//
// `a` is (m, k) row-major and `b` is (k, n) row-major, both contiguous. `c` is
// (m, n), written in place. `res` is optional and MAY ALIAS `c`: each element
// is read by the one thread that then writes it, which is what the residual
// call sites need. M, N and K must be multiples of the tile.
__global__ __launch_bounds__(kThreads) void gemm_kernel(
    const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *res, __nv_bfloat16 *c, int32_t m, int32_t k,
    int32_t n) {
  __shared__ __nv_bfloat16 as[2][kBM * kLdA];
  __shared__ __nv_bfloat16 bs[2][kBK * kLdB];

  const int32_t n0 = blockIdx.x * kBN, m0 = blockIdx.y * kBM;
  const int32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int32_t wm = warp / kWarpsN, wn = warp % kWarpsN;

  float acc[kMT][kNT][4] = {};

  //: A: lanes of a quad cover one row's 32 columns. B: sixteen lanes cover one
  //: k row's 128 columns, which is a full 256-byte transaction.
  constexpr int32_t kALanes = kBK / kVec, kBLanes = kBN / kVec;
  const int32_t ar = threadIdx.x / kALanes, ad = (threadIdx.x % kALanes) * kVec;
  const int32_t bk = threadIdx.x / kBLanes, bc = (threadIdx.x % kBLanes) * kVec;
  int4 areg[kAPer], breg[kBPer];

  auto stage = [&](int32_t k0) {
#pragma unroll
    for (int32_t t = 0; t < kAPer; ++t)
      areg[t] = *(const int4 *)(a + int64_t(m0 + ar + t * kAStep) * k + k0 + ad);
#pragma unroll
    for (int32_t t = 0; t < kBPer; ++t)
      breg[t] = *(const int4 *)(b + int64_t(k0 + bk + t * kBStep) * n + n0 + bc);
  };
  auto publish = [&](int32_t buf) {
#pragma unroll
    for (int32_t t = 0; t < kAPer; ++t)
      *(int4 *)(&as[buf][(ar + t * kAStep) * kLdA + ad]) = areg[t];
#pragma unroll
    for (int32_t t = 0; t < kBPer; ++t)
      *(int4 *)(&bs[buf][(bk + t * kBStep) * kLdB + bc]) = breg[t];
  };

  // Three chunks in flight: one being read from shared memory, one in the
  // staging registers, one on its way from DRAM.
  const int32_t chunks = k / kBK;
  stage(0);
  publish(0);
  if (chunks > 1) stage(kBK);

  for (int32_t ch = 0; ch < chunks; ++ch) {
    __syncthreads();
    const __nv_bfloat16 *ab = as[ch & 1], *bb = bs[ch & 1];
#pragma unroll
    for (int32_t kk = 0; kk < kBK; kk += 16) {
      uint32_t af[kMT][4], bf[kNT][2];
#pragma unroll
      for (int32_t i = 0; i < kMT; ++i)
        ldmatrix_a(af[i], ab + (wm * kWM + i * 16) * kLdA, kLdA, kk, lane);
#pragma unroll
      for (int32_t j = 0; j < kNT; ++j)
        ldmatrix_b_trans(bf[j], bb, kLdB, wn * kWN + j * 8, kk, lane);
#pragma unroll
      for (int32_t i = 0; i < kMT; ++i)
#pragma unroll
        for (int32_t j = 0; j < kNT; ++j)
          mma_m16n8k16(acc[i][j], af[i], bf[j]);
    }
    if (ch + 1 < chunks) {
      publish((ch + 1) & 1);
      if (ch + 2 < chunks) stage((ch + 2) * kBK);
    }
  }

  // D layout: lane L holds rows L/4 and L/4+8 at columns (L%4)*2 and +1, so
  // each lane writes two adjacent columns as one 32-bit store.
  const int32_t r0 = lane >> 2, c0 = (lane & 3) * 2;
#pragma unroll
  for (int32_t i = 0; i < kMT; ++i) {
#pragma unroll
    for (int32_t j = 0; j < kNT; ++j) {
#pragma unroll
      for (int32_t h = 0; h < 2; ++h) {
        const int32_t row = m0 + wm * kWM + i * 16 + r0 + h * 8;
        const int32_t col = n0 + wn * kWN + j * 8 + c0;
        float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
        if (res != nullptr) {
          const __nv_bfloat162 r =
              ((const __nv_bfloat162 *)(res + int64_t(row) * n))[col >> 1];
          v0 += __bfloat162float(r.x);
          v1 += __bfloat162float(r.y);
        }
        ((__nv_bfloat162 *)(c + int64_t(row) * n))[col >> 1] =
            __floats2bfloat162_rn(v0, v1);
      }
    }
  }
}

}  // namespace

extern "C" int tiled_gemm_launch(const void *a, const void *b, const void *res,
                                 void *c, int m, int k, int n, void *stream) {
  // Tail handling is deliberately absent: these are fixed model shapes, and a
  // shape that does not tile is a routing error rather than something to
  // handle slowly.
  if (m % kBM || n % kBN || k % kBK) return cudaErrorInvalidValue;
  const dim3 grid(n / kBN, m / kBM);
  gemm_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b,
      (const __nv_bfloat16 *)res, (__nv_bfloat16 *)c, m, k, n);
  return (int)cudaGetLastError();
}
