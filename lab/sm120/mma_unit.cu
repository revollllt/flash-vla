// mma unit -- what warp-level mma.sync can do on this device.
//
// On sm90 this unit has two halves, and the wgmma half does not exist here:
// ptxas refuses every wgmma form on sm_120a [isa.wgmma.absent], and tcgen05 is
// datacenter Blackwell only. So there is no second tensor-core instruction to
// cross over to, and sm90's [mma.xover.n.wgmma] -- "below tile N = 32 use
// mma.sync" -- has no meaning here. The question changes from "which
// instruction" to "how close to the ceiling can the only instruction get".
//
// Three questions, each with the hypothesis it kills:
//
//   M1  How many cycles does one m16n8k16 cost a warp, against the number of
//       independent accumulator sets it keeps in flight?
//       Kills "one accumulator is enough". A single chain measures the
//       instruction's LATENCY because each mma waits for the previous one's
//       accumulator; independent chains let the pipeline fill and measure its
//       ISSUE rate. The knee is where the two stop differing.
//
//   M2  What sustained bf16 throughput does the whole device reach?
//       Kills a FLOP/s target derived from the datasheet. spec.py derives
//       419.4 TFLOP/s dense from the FP32 peak; this measures what is actually
//       reachable with everything resident and nothing else in flight.
//
//   M3  What does feeding the operands from shared memory through ldmatrix
//       cost, against register-resident operands?
//       Kills "the feed is free once the tile is in shared memory". Reported as
//       a multiplier on M1's cycles-per-mma at several reuse factors.
//
// Isolation: operands are register-resident in M1 and M2, so no memory system
// is in the number. Accumulators are written to global at the end under a
// condition that is false, which stops ptxas eliminating the chain without
// adding store traffic to the measurement. Every rate is checked against a
// torch reference elsewhere (lab/sm120/mma_check.py) before it is reported.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      printf("CUDA error %s at line %d: %s\n", #x, __LINE__,                   \
             cudaGetErrorString(e_));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// One m16n8k16 bf16 MMA with f32 accumulate: A is 4 b32 registers, B is 2,
// C/D are 4 f32. FLOPs per instruction = 2 * 16 * 8 * 16 = 4096.
static __device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int32_t kFlopsPerMma = 2 * 16 * 8 * 16;

// The fp8 form, used only to cross-check spec.py's derived peak ladder. If the
// tensor core runs e4m3 at about twice the bf16 rate then the ladder (each
// halving of element width doubling throughput) holds, and the bf16 shortfall
// below is an efficiency of the instruction rather than an error in the peak.
// A is 4 b32 registers, B is 2, C/D are 4 f32. 2 * 16 * 8 * 32 = 8192 FLOP.
static __device__ __forceinline__ void mma_m16n8k32_fp8(float (&d)[4],
                                                        const uint32_t (&a)[4],
                                                        const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int32_t kFlopsPerMmaFp8 = 2 * 16 * 8 * 32;

template <int NACC, int ITERS>
__global__ void mma_sustained_fp8(float *__restrict__ sink) {
  uint32_t a[4], b[2];
  const uint32_t lane = threadIdx.x & 31u;
  for (int32_t i = 0; i < 4; ++i) a[i] = 0x38383838u ^ (lane * 2654435761u + i);
  for (int32_t i = 0; i < 2; ++i) b[i] = 0x38383838u ^ (lane * 40503u + i);
  float acc[NACC][4];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) acc[k][j] = 0.f;
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int32_t k = 0; k < NACC; ++k) mma_m16n8k32_fp8(acc[k], a, b);
  }
  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  if (total == 1.0e37f) sink[0] = total;
}

// M1: cycles per mma against NACC independent accumulator sets. Each set is a
// serial dependence chain; NACC of them run independently, so the ratio between
// NACC=1 and large NACC is latency over issue interval.
template <int NACC, int ITERS>
__global__ void mma_issue(float *__restrict__ sink, uint64_t *__restrict__ cycles) {
  uint32_t a[4], b[2];
  // Operand bits are arbitrary but must not be zero: a denormal/zero operand
  // could in principle take a different path, and this probe is about issue
  // rate, not about data.
  const uint32_t lane = threadIdx.x & 31u;
  for (int32_t i = 0; i < 4; ++i) a[i] = 0x3f803f80u ^ (lane * 2654435761u + i);
  for (int32_t i = 0; i < 2; ++i) b[i] = 0x3f003f00u ^ (lane * 40503u + i);

  float acc[NACC][4];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) acc[k][j] = 0.f;

  __syncthreads();
  const uint64_t t0 = clock64();
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int32_t k = 0; k < NACC; ++k) mma_m16n8k16(acc[k], a, b);
  }
  const uint64_t t1 = clock64();

  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  // Never true for finite accumulations; keeps the chain alive without stores.
  if (total == 1.0e37f) sink[0] = total;
  if (threadIdx.x == 0 && blockIdx.x == 0) *cycles = t1 - t0;
}

// M2: the same loop sized to fill the device, timed on the host so the answer
// is a device throughput rather than a per-warp cycle count.
template <int NACC, int ITERS>
__global__ void mma_sustained(float *__restrict__ sink) {
  uint32_t a[4], b[2];
  const uint32_t lane = threadIdx.x & 31u;
  for (int32_t i = 0; i < 4; ++i) a[i] = 0x3f803f80u ^ (lane * 2654435761u + i);
  for (int32_t i = 0; i < 2; ++i) b[i] = 0x3f003f00u ^ (lane * 40503u + i);
  float acc[NACC][4];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) acc[k][j] = 0.f;
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int32_t k = 0; k < NACC; ++k) mma_m16n8k16(acc[k], a, b);
  }
  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  if (total == 1.0e37f) sink[0] = total;
}

// M3: operands re-loaded from shared memory through ldmatrix every REUSE mmas.
// REUSE = 1 reloads for every instruction; larger values amortise the load the
// way a real mainloop does.
template <int NACC, int ITERS, int REUSE>
__global__ void mma_ldmatrix(float *__restrict__ sink, uint64_t *__restrict__ cycles) {
  // 16 x 16 bf16 tile per matrix; ldmatrix x4 reads four 8x8 b16 matrices.
  __shared__ uint32_t tile[32 * 32];
  for (int32_t i = threadIdx.x; i < 32 * 32; i += blockDim.x)
    tile[i] = 0x3f803f80u ^ uint32_t(i * 2654435761u);
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31u;
  // Each lane points at its own row of the tile; the exact swizzle does not
  // matter here because the probe measures issue cost, not correctness.
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&tile[(lane % 16) * 16]));

  uint32_t a[4], b[2];
  float acc[NACC][4];
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) acc[k][j] = 0.f;

  __syncthreads();
  const uint64_t t0 = clock64();
#pragma unroll 1
  for (int32_t it = 0; it < ITERS; ++it) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(addr));
#pragma unroll
    for (int32_t r = 0; r < REUSE; ++r)
#pragma unroll
      for (int32_t k = 0; k < NACC; ++k) mma_m16n8k16(acc[k], a, b);
  }
  const uint64_t t1 = clock64();

  float total = 0.f;
#pragma unroll
  for (int32_t k = 0; k < NACC; ++k)
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) total += acc[k][j];
  if (total == 1.0e37f) sink[0] = total;
  if (threadIdx.x == 0 && blockIdx.x == 0) *cycles = t1 - t0;
}

static float median_ms(void (*launch)(void *), void *arg, int32_t reps) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int32_t i = 0; i < reps; ++i) {
    CHECK(cudaEventRecord(a));
    launch(arg);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    v.push_back(ms);
  }
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

constexpr int32_t kIters = 4096;

template <int NACC>
static void run_issue(float *sink, uint64_t *d_cycles, double &out_cycles) {
  mma_issue<NACC, kIters><<<1, 32>>>(sink, d_cycles);
  CHECK(cudaDeviceSynchronize());
  uint64_t c = 0;
  CHECK(cudaMemcpy(&c, d_cycles, sizeof(c), cudaMemcpyDeviceToHost));
  out_cycles = double(c) / (double(kIters) * NACC);
}

template <int NACC, int REUSE>
static void run_ldm(float *sink, uint64_t *d_cycles, double &out_cycles) {
  mma_ldmatrix<NACC, kIters, REUSE><<<1, 32>>>(sink, d_cycles);
  CHECK(cudaDeviceSynchronize());
  uint64_t c = 0;
  CHECK(cudaMemcpy(&c, d_cycles, sizeof(c), cudaMemcpyDeviceToHost));
  out_cycles = double(c) / (double(kIters) * NACC * REUSE);
}

struct SustArg { int32_t grid, block; };
static void launch_sust(void *p) {
  SustArg *a = (SustArg *)p;
  static float *sink = nullptr;
  if (!sink) CHECK(cudaMalloc(&sink, sizeof(float)));
  mma_sustained<8, kIters><<<a->grid, a->block>>>(sink);
}

static void launch_sust_fp8(void *p) {
  SustArg *a = (SustArg *)p;
  static float *sink = nullptr;
  if (!sink) CHECK(cudaMalloc(&sink, sizeof(float)));
  mma_sustained_fp8<8, kIters><<<a->grid, a->block>>>(sink);
}

int main(int argc, char **argv) {
  const int32_t reps = (argc > 1) ? atoi(argv[1]) : 20;

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int32_t SMS = prop.multiProcessorCount;
  printf("# mma unit -- %s, %d SMs, %d threads/SM max\n", prop.name, SMS,
         prop.maxThreadsPerMultiProcessor);
  printf("# mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32, %d FLOP each\n",
         kFlopsPerMma);
  printf("# no wgmma on this target, so this is the ONLY tensor-core path\n\n");

  float *sink = nullptr;
  uint64_t *d_cycles = nullptr;
  CHECK(cudaMalloc(&sink, sizeof(float)));
  CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

  // ---- M1: cycles per mma vs independent accumulator sets ----------------
  printf("## M1 cycles per m16n8k16, one warp, register-resident operands\n");
  printf("%14s %14s %12s\n", "accum sets", "cycles/mma", "vs best");
  double c1 = 0, c2 = 0, c4 = 0, c8 = 0, c16 = 0;
  run_issue<1>(sink, d_cycles, c1);
  run_issue<2>(sink, d_cycles, c2);
  run_issue<4>(sink, d_cycles, c4);
  run_issue<8>(sink, d_cycles, c8);
  run_issue<16>(sink, d_cycles, c16);
  const double best = std::min({c1, c2, c4, c8, c16});
  const double vals[] = {c1, c2, c4, c8, c16};
  const int32_t naccs[] = {1, 2, 4, 8, 16};
  for (int32_t i = 0; i < 5; ++i)
    printf("%14d %14.2f %11.2fx\n", naccs[i], vals[i], vals[i] / best);
  printf("\n  latency (1 set) / issue interval (best) = %.2fx\n", c1 / best);

  // ---- M2: sustained device throughput ------------------------------------
  printf("\n## M2 sustained bf16 throughput (8 accumulator sets)\n");
  printf("%10s %10s %12s %12s %10s\n", "CTAs", "warps/CTA", "ms", "TFLOP/s",
         "of 419.4");
  for (int32_t ctas_per_sm : {1, 2}) {
    for (int32_t warps : {4, 8, 12}) {
      const int32_t grid = SMS * ctas_per_sm, block = warps * 32;
      if (warps * ctas_per_sm * 32 > prop.maxThreadsPerMultiProcessor) continue;
      SustArg arg{grid, block};
      launch_sust(&arg);
      CHECK(cudaDeviceSynchronize());
      const float ms = median_ms(launch_sust, &arg, reps);
      const double mmas = double(grid) * warps * kIters * 8.0;
      const double tflops = mmas * kFlopsPerMma / (ms * 1e-3) / 1e12;
      printf("%10d %10d %12.3f %12.1f %9.0f%%\n", grid, warps, ms, tflops,
             100.0 * tflops / 419.4);
    }
  }

  // ---- M3: the ldmatrix feed tax ------------------------------------------
  printf("\n## M3 ldmatrix feed tax, 8 accumulator sets\n");
  printf("%14s %14s %12s\n", "mmas per load", "cycles/mma", "vs M1");
  double l1 = 0, l2 = 0, l4 = 0, l8 = 0;
  run_ldm<8, 1>(sink, d_cycles, l1);
  run_ldm<8, 2>(sink, d_cycles, l2);
  run_ldm<8, 4>(sink, d_cycles, l4);
  run_ldm<8, 8>(sink, d_cycles, l8);
  const double lv[] = {l1, l2, l4, l8};
  const int32_t reuse[] = {8, 16, 32, 64};  // NACC * REUSE mmas per load pair
  for (int32_t i = 0; i < 4; ++i)
    printf("%14d %14.2f %11.2fx\n", reuse[i], lv[i], lv[i] / c8);

  // ---- M4: fp8, to cross-check the derived peak ladder --------------------
  printf("\n## M4 sustained fp8 (e4m3) throughput -- validates spec.py's ladder\n");
  printf("%10s %10s %12s %12s %10s\n", "CTAs", "warps/CTA", "ms", "TFLOP/s",
         "of 838.8");
  for (int32_t warps : {4, 8}) {
    SustArg arg{SMS, warps * 32};
    launch_sust_fp8(&arg);
    CHECK(cudaDeviceSynchronize());
    const float ms = median_ms(launch_sust_fp8, &arg, reps);
    const double mmas = double(SMS) * warps * kIters * 8.0;
    const double tflops = mmas * kFlopsPerMmaFp8 / (ms * 1e-3) / 1e12;
    printf("%10d %10d %12.3f %12.1f %9.0f%%\n", SMS, warps, ms, tflops,
           100.0 * tflops / 838.8);
  }

  printf("\n# Compare sm90: [mma.issue.warp] 6.26 cyc at >=4 accumulators,\n"
         "# [mma.stages.warp.knee] 4 sets, [mma.feedtax.warp.ldmatrix] <=1.18x.\n");
  return 0;
}
