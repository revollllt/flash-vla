// Settle the tensor-core ceiling WITHOUT assuming a clock.
//
// mma_unit.cu reported 253 TFLOP/s bf16 and called it 60% of a 419.4 TFLOP/s
// peak that spec.py DERIVES from the FP32 peak. Both halves of that are
// suspect, and they are suspect in opposite directions:
//
//   * The derived peak assumes consumer Blackwell runs bf16-with-fp32-accumulate
//     at 4x the FP32 lane rate. Ada does not -- GeForce halves fp32-accumulate
//     against fp16-accumulate -- and the figure in circulation for this part is
//     209.5 TFLOP/s, half of 419.4.
//   * The measured TFLOP/s came from wall time, and the peak it was compared
//     against came from an assumed 2.407 GHz boost clock. Mixing a measured
//     numerator with an assumed denominator is exactly how a 60% appears.
//
// So measure the clock-free quantity instead: FLOP per cycle per SM. That is a
// property of the hardware, it needs no datasheet, and every TFLOP/s figure --
// measured or published -- is it multiplied by a clock. The kernel reports
// clock64() deltas alongside host wall time, so the achieved clock falls out
// too and can be checked against what the part is supposed to run at.
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

static __device__ __forceinline__ void mma_bf16(float (&d)[4],
                                                const uint32_t (&a)[4],
                                                const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void mma_fp8(float (&d)[4],
                                               const uint32_t (&a)[4],
                                               const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The fp16-accumulate form. NVIDIA's published 419 TF dense figure for this
// part is FP16 with FP16 accumulate; the fp32-accumulate rate should be half
// of it. Measuring this settles whether the halving is real and whether an
// fp16-accumulate path is worth anything to a kernel that currently
// accumulates in fp32. D and C are 2 packed b32 registers here, not 4 f32.
static __device__ __forceinline__ void mma_f16acc(uint32_t (&d)[2],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int NACC, int ITERS>
__global__ void mma_f16acc_clocked(float *__restrict__ sink,
                                   uint64_t *__restrict__ t_start,
                                   uint64_t *__restrict__ t_end,
                                   uint32_t *__restrict__ smid_out) {
  uint32_t a[4], b[2];
  const uint32_t lane = threadIdx.x & 31u;
  for (int32_t i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (lane * 2654435761u + i);
  for (int32_t i = 0; i < 2; ++i) b[i] = 0x3c003c00u ^ (lane * 40503u + i);
  uint32_t acc[NACC][2];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 2; ++j) acc[k][j] = 0u;
  __syncthreads();
  const uint64_t c0 = clock64();
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int32_t k = 0; k < NACC; ++k) mma_f16acc(acc[k], a, b);
  }
  const uint64_t c1 = clock64();
  uint32_t total = 0u;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 2; ++j) total ^= acc[k][j];
  if (total == 0xdeadbeefu) sink[0] = 1.f;
  if ((threadIdx.x & 31u) == 0) {
    const uint32_t w = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x / 32u);
    t_start[w] = c0; t_end[w] = c1;
    uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smid_out[w] = sm;
  }
}

// Every warp records its own start and end cycle. The SM-wide span is taken as
// max(end) - min(start) over the warps resident on that SM, so the number
// counts the SM's elapsed cycles rather than one warp's.
template <bool FP8, int NACC, int ITERS>
__global__ void mma_clocked(float *__restrict__ sink,
                            uint64_t *__restrict__ t_start,
                            uint64_t *__restrict__ t_end,
                            uint32_t *__restrict__ smid_out) {
  uint32_t a[4], b[2];
  const uint32_t lane = threadIdx.x & 31u;
  for (int32_t i = 0; i < 4; ++i) a[i] = 0x3f803f80u ^ (lane * 2654435761u + i);
  for (int32_t i = 0; i < 2; ++i) b[i] = 0x3f003f00u ^ (lane * 40503u + i);

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
    for (int32_t k = 0; k < NACC; ++k) {
      if constexpr (FP8) mma_fp8(acc[k], a, b);
      else mma_bf16(acc[k], a, b);
    }
  }
  const uint64_t c1 = clock64();

  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  if (total == 1.0e37f) sink[0] = total;

  if ((threadIdx.x & 31u) == 0) {
    const uint32_t w = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x / 32u);
    t_start[w] = c0;
    t_end[w] = c1;
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smid_out[w] = sm;
  }
}

struct Result {
  double ms, flop_per_cycle_per_sm, tflops, clock_ghz, warps_per_sm;
};

template <bool FP8, int NACC, int ITERS>
static Result run(int32_t sms, int32_t ctas, int32_t warps, int32_t reps,
                  int32_t flop_per_mma) {
  const int32_t n_warps = ctas * warps;
  float *sink;
  uint64_t *ts, *te;
  uint32_t *smid;
  CHECK(cudaMalloc(&sink, sizeof(float)));
  CHECK(cudaMalloc(&ts, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&te, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&smid, n_warps * sizeof(uint32_t)));

  auto launch = [&] {
    mma_clocked<FP8, NACC, ITERS><<<ctas, warps * 32>>>(sink, ts, te, smid);
  };
  launch();
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int32_t i = 0; i < reps; ++i) {
    CHECK(cudaEventRecord(a));
    launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  const double ms = v[v.size() / 2];

  std::vector<uint64_t> hs(n_warps), he(n_warps);
  std::vector<uint32_t> hsm(n_warps);
  CHECK(cudaMemcpy(hs.data(), ts, n_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(he.data(), te, n_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hsm.data(), smid, n_warps * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Per-SM elapsed cycles and per-SM mma count, then FLOP per cycle per SM.
  std::vector<uint64_t> lo(sms, ~0ull), hi(sms, 0);
  std::vector<int32_t> nw(sms, 0);
  for (int32_t w = 0; w < n_warps; ++w) {
    const uint32_t s = hsm[w];
    if (s >= (uint32_t)sms) continue;
    lo[s] = std::min(lo[s], hs[w]);
    hi[s] = std::max(hi[s], he[w]);
    nw[s]++;
  }
  double sum_fpc = 0.0, sum_warps = 0.0;
  int32_t counted = 0;
  for (int32_t s = 0; s < sms; ++s) {
    if (nw[s] == 0 || hi[s] <= lo[s]) continue;
    const double cycles = double(hi[s] - lo[s]);
    const double mmas = double(nw[s]) * ITERS * NACC;
    sum_fpc += mmas * flop_per_mma / cycles;
    sum_warps += nw[s];
    counted++;
  }
  const double fpc = counted ? sum_fpc / counted : 0.0;
  const double total_flop = double(n_warps) * ITERS * NACC * flop_per_mma;
  const double tflops = total_flop / (ms * 1e-3) / 1e12;
  // Achieved clock: device FLOP/s divided by (FLOP/cycle/SM x SMs).
  const double ghz = tflops * 1e12 / (fpc * sms) / 1e9;

  CHECK(cudaFree(sink)); CHECK(cudaFree(ts)); CHECK(cudaFree(te)); CHECK(cudaFree(smid));
  CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return {ms, fpc, tflops, ghz, counted ? sum_warps / counted : 0.0};
}

template <int NACC, int ITERS>
static Result run_f16acc(int32_t sms, int32_t ctas, int32_t warps, int32_t reps,
                         int32_t flop_per_mma) {
  const int32_t n_warps = ctas * warps;
  float *sink; uint64_t *ts, *te; uint32_t *smid;
  CHECK(cudaMalloc(&sink, sizeof(float)));
  CHECK(cudaMalloc(&ts, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&te, n_warps * sizeof(uint64_t)));
  CHECK(cudaMalloc(&smid, n_warps * sizeof(uint32_t)));
  auto launch = [&] {
    mma_f16acc_clocked<NACC, ITERS><<<ctas, warps * 32>>>(sink, ts, te, smid);
  };
  launch(); CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int32_t i = 0; i < reps; ++i) {
    CHECK(cudaEventRecord(a)); launch(); CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  const double ms = v[v.size() / 2];
  std::vector<uint64_t> hs(n_warps), he(n_warps);
  std::vector<uint32_t> hsm(n_warps);
  CHECK(cudaMemcpy(hs.data(), ts, n_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(he.data(), te, n_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hsm.data(), smid, n_warps * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::vector<uint64_t> lo(sms, ~0ull), hi(sms, 0);
  std::vector<int32_t> nw(sms, 0);
  for (int32_t w = 0; w < n_warps; ++w) {
    const uint32_t s = hsm[w];
    if (s >= (uint32_t)sms) continue;
    lo[s] = std::min(lo[s], hs[w]); hi[s] = std::max(hi[s], he[w]); nw[s]++;
  }
  double sum_fpc = 0.0; int32_t counted = 0;
  for (int32_t s = 0; s < sms; ++s) {
    if (nw[s] == 0 || hi[s] <= lo[s]) continue;
    sum_fpc += double(nw[s]) * ITERS * NACC * flop_per_mma / double(hi[s] - lo[s]);
    counted++;
  }
  const double fpc = counted ? sum_fpc / counted : 0.0;
  const double tflops = double(n_warps) * ITERS * NACC * flop_per_mma / (ms * 1e-3) / 1e12;
  CHECK(cudaFree(sink)); CHECK(cudaFree(ts)); CHECK(cudaFree(te)); CHECK(cudaFree(smid));
  CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return {ms, fpc, tflops, tflops * 1e12 / (fpc * sms) / 1e9, 0.0};
}

int main(int argc, char **argv) {
  const int32_t reps = (argc > 1) ? atoi(argv[1]) : 20;
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int32_t SMS = prop.multiProcessorCount;
  int clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  printf("# %s, %d SMs, driver-reported max clock %.3f GHz\n", prop.name, SMS,
         clock_khz / 1e6);
  printf("# FLOP/cycle/SM is the clock-free hardware constant; TFLOP/s is it\n"
         "# times the achieved clock, which is derived here rather than assumed.\n\n");

  printf("%-6s %6s %7s %10s %14s %11s %10s\n", "dtype", "CTAs", "warps", "ms",
         "FLOP/cyc/SM", "TFLOP/s", "GHz");
  for (int32_t w : {4, 8, 12}) {
    Result r = run<false, 8, 4096>(SMS, SMS, w, reps, 2 * 16 * 8 * 16);
    printf("%-6s %6d %7d %10.3f %14.1f %11.1f %10.3f\n", "bf16", SMS, w, r.ms,
           r.flop_per_cycle_per_sm, r.tflops, r.clock_ghz);
  }
  for (int32_t w : {4, 8}) {
    Result r = run<true, 8, 4096>(SMS, SMS, w, reps, 2 * 16 * 8 * 32);
    printf("%-6s %6d %7d %10.3f %14.1f %11.1f %10.3f\n", "fp8", SMS, w, r.ms,
           r.flop_per_cycle_per_sm, r.tflops, r.clock_ghz);
  }
  for (int32_t w : {4, 8}) {
    Result r = run_f16acc<8, 4096>(SMS, SMS, w, reps, 2 * 16 * 8 * 16);
    printf("%-6s %6d %7d %10.3f %14.1f %11.1f %10.3f\n", "f16acc", SMS, w,
           r.ms, r.flop_per_cycle_per_sm, r.tflops, r.clock_ghz);
  }

  printf("\n# Reference points for FLOP/cycle/SM, bf16 with fp32 accumulate:\n"
         "#   512  -> the 209.5 TFLOP/s figure in circulation, at 2.41 GHz\n"
         "#  1024  -> the 419.4 TFLOP/s spec.py currently derives, at 2.41 GHz\n");
  return 0;
}
