// Tensor-core ceilings of SM120's block-scaled MMA, measured the way
// lab/sm120/mma_clock.cu measures bf16 and fp8: clock-free FLOP per cycle per
// SM from clock64() deltas, plus the achieved TFLOP/s and clock.
//
// spec.py records bf16 (511.5) and legacy fp8 (1023.0) with fp32 accumulate but
// leaves SM120's kind::f8f6f4 forms and the block-scaled kinds out. Those are the instructions an MXFP8 or
// NVFP4 GEMM issues, so their rates are the compute denominators of the
// quantized-kernel survey. PTX strings are CUTLASS v4.7.1 cute/arch/mma_sm120.hpp.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a mma_blockscale_clock.cu -o mma_blockscale_clock
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at line %d: %s\n", #x, __LINE__,                   \
             cudaGetErrorString(e_));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum Kind { BF16 = 0, FP8 = 1, MXFP8 = 2, NVFP4 = 3, F8F6F4 = 4 };

template <int K>
static __device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                           const uint32_t (&b)[2], uint32_t sfa,
                                           uint32_t sfb) {
  const uint16_t z = 0;
  if constexpr (K == BF16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (K == FP8) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (K == F8F6F4) {
    // The unscaled SM120 form CUTLASS's sm120 FP8 and blockwise-scaled mainloops issue.
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (K == MXFP8) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col."
        "f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
  } else {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
        "f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
  }
}

template <int K, int NACC, int ITERS>
__global__ void clocked(float *__restrict__ sink, uint64_t *__restrict__ t_start,
                        uint64_t *__restrict__ t_end, uint32_t *__restrict__ smid_out) {
  uint32_t a[4], b[2];
  const uint32_t lane = threadIdx.x & 31u;
  // Finite, varied operands: 0x38 is 1.0 in e4m3, 0x2 is 1.0 in e2m1.
  for (int32_t i = 0; i < 4; ++i) a[i] = (K == NVFP4 ? 0x22222222u : 0x38383838u) ^ ((lane + i) & 0x01010101u);
  for (int32_t i = 0; i < 2; ++i) b[i] = (K == NVFP4 ? 0x22222222u : 0x38383838u) ^ ((lane * 3 + i) & 0x01010101u);
  // ue8m0 127 = 2^0; four ue4m3 1.0 (0x38) for NVFP4's four 16-element blocks.
  const uint32_t sf = (K == NVFP4) ? 0x38383838u : 0x7fu;

  float acc[NACC][4];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) acc[k][j] = 0.f;

  __syncthreads();
  const uint64_t c0 = clock64();
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int32_t k = 0; k < NACC; ++k) mma<K>(acc[k], a, b, sf, sf);
  }
  const uint64_t c1 = clock64();

  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  if (total == 1.0e37f) sink[0] = total;

  if (lane == 0) {
    const uint32_t w = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x / 32u);
    t_start[w] = c0;
    t_end[w] = c1;
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smid_out[w] = sm;
  }
}

struct Result { double ms, fpc, tflops, ghz; };

template <int K, int NACC, int ITERS>
static Result run(int32_t sms, int32_t warps, int32_t reps, int32_t flop_per_mma) {
  const int32_t ctas = sms, n_warps = ctas * warps;
  float *sink; uint64_t *ts, *te; uint32_t *smid;
  CHECK(cudaMalloc(&sink, sizeof(float)));
  CHECK(cudaMalloc(&ts, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&te, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&smid, n_warps * sizeof(uint32_t)));
  auto launch = [&] { clocked<K, NACC, ITERS><<<ctas, warps * 32>>>(sink, ts, te, smid); };
  launch();
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int32_t i = 0; i < reps; ++i) {
    CHECK(cudaEventRecord(a)); launch(); CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  const double ms = v[v.size() / 2];
  std::vector<uint64_t> hs(n_warps), he(n_warps); std::vector<uint32_t> hsm(n_warps);
  CHECK(cudaMemcpy(hs.data(), ts, n_warps * 8, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(he.data(), te, n_warps * 8, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hsm.data(), smid, n_warps * 4, cudaMemcpyDeviceToHost));
  std::vector<uint64_t> lo(sms, ~0ull), hi(sms, 0); std::vector<int32_t> nw(sms, 0);
  for (int32_t w = 0; w < n_warps; ++w) {
    const uint32_t s = hsm[w]; if (s >= (uint32_t)sms) continue;
    lo[s] = std::min(lo[s], hs[w]); hi[s] = std::max(hi[s], he[w]); nw[s]++;
  }
  double sum = 0.0; int32_t counted = 0;
  for (int32_t s = 0; s < sms; ++s) {
    if (nw[s] == 0 || hi[s] <= lo[s]) continue;
    sum += double(nw[s]) * ITERS * NACC * flop_per_mma / double(hi[s] - lo[s]); counted++;
  }
  const double fpc = counted ? sum / counted : 0.0;
  const double tflops = double(n_warps) * ITERS * NACC * flop_per_mma / (ms * 1e-3) / 1e12;
  CHECK(cudaFree(sink)); CHECK(cudaFree(ts)); CHECK(cudaFree(te)); CHECK(cudaFree(smid));
  CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return {ms, fpc, tflops, tflops * 1e12 / (fpc * sms) / 1e9};
}

template <int K>
static void report(const char *name, int32_t sms, int32_t reps, int32_t flop_per_mma) {
  for (int32_t w : {4, 8, 16}) {
    Result r = run<K, 8, 4096>(sms, w, reps, flop_per_mma);
    printf("%-7s %6d %7d %10.3f %14.1f %11.1f %10.3f\n", name, sms, w, r.ms, r.fpc, r.tflops, r.ghz);
  }
}

int main(int argc, char **argv) {
  const int32_t reps = (argc > 1) ? atoi(argv[1]) : 20;
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int32_t sms = prop.multiProcessorCount;
  printf("# %s, %d SMs; FLOP/cycle/SM is clock-free, TFLOP/s and GHz are as achieved\n",
         prop.name, sms);
  printf("%-7s %6s %7s %10s %14s %11s %10s\n", "kind", "CTAs", "warps", "ms",
         "FLOP/cyc/SM", "TFLOP/s", "GHz");
  report<BF16>("bf16", sms, reps, 2 * 16 * 8 * 16);
  report<FP8>("fp8", sms, reps, 2 * 16 * 8 * 32);
  report<F8F6F4>("f8f6f4", sms, reps, 2 * 16 * 8 * 32);
  report<MXFP8>("mxfp8", sms, reps, 2 * 16 * 8 * 32);
  report<NVFP4>("nvfp4", sms, reps, 2 * 16 * 8 * 64);
  return 0;
}
