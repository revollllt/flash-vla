// mma_rate.cu -- isolate the tensor core's issue rate from everything else.
//
// tile-dataflow's Phase 0 names this gap outright: none of its measurements
// yield per-engine throughput, and an L3 timeline that claims the math column
// covers the copy column is asserting a ratio nobody measured.  The TMA unit
// supplies one side of that ratio (270 ns per TMA per producer warp).  This is
// the other side.
//
// One warpgroup (or two) issues wgmma back to back out of resident shared
// memory.  No TMA, no global traffic, no barriers in the loop: the only thing
// being timed is how fast the tensor core retires instructions.
//
// The operands are re-issued from the same two smem tiles every iteration.
// That is deliberate: it keeps address arithmetic out of the loop, and a real
// GEMM mainloop accumulates every k-step into the same registers anyway, so
// the dependency structure is the realistic one.
//
// check_kernel exists because a rate measured on an instruction nobody
// verified is a measurement of an unknown.  It runs ONE wgmma on real data and
// writes D out in row/col order so the host can compare against a reference.
// If the descriptors or the accumulator mapping are wrong, that check fails
// before any number is believed.
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include mma_rate.cu -lcuda

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include <cuda.h>
#include <cstdint>

namespace mmaprobe {

using namespace cute;
using bf16 = cutlass::bfloat16_t;
namespace G = cute::SM90::GMMA;

template <int N> struct AtomFor;
#define ATOM(N) template <> struct AtomFor<N> { \
  using type = G::MMA_64x##N##x16_F32BF16BF16_SS<G::Major::K, G::Major::K>; }
ATOM(8); ATOM(16); ATOM(32); ATOM(64); ATOM(96); ATOM(128); ATOM(192); ATOM(256);
#undef ATOM

constexpr int M_TILE = 64;   // one warpgroup's rows
constexpr int K_TILE = 16;   // one wgmma k-step

// Without this, ptxas emits C7515 -- "wgmma.mma_async instructions are
// serialized due to non wgmma instructions defining accumulator registers" --
// and every rate below would be the SERIALIZED rate, not the pipelined one.
// The empty asm with "+f" tells the compiler wgmma owns these registers, so it
// stops scheduling anything else into them. A rate probe that ignored the
// warning would have measured a real number for the wrong machine.
template <int NREG>
__device__ __forceinline__ void fence_acc(float (&acc)[NREG]) {
#pragma unroll
  for (int i = 0; i < NREG; ++i) cute::warpgroup_fence_operand(acc[i]);
}

extern __shared__ uint8_t smem_pool[];

// Fill the two smem tiles through their cute layouts, so the descriptor and
// the data agree by construction rather than by a matching pair of hand
// calculations.
template <int N>
__device__ __forceinline__ void stage(const bf16* gA, const bf16* gB,
                                      uint64_t& da, uint64_t& db) {
  auto lA = tile_to_shape(G::Layout_K_INTER_Atom<bf16>{},
                          Shape<Int<M_TILE>, Int<K_TILE>>{});
  auto lB = tile_to_shape(G::Layout_K_INTER_Atom<bf16>{},
                          Shape<Int<N>, Int<K_TILE>>{});
  bf16* pA = reinterpret_cast<bf16*>(smem_pool);
  bf16* pB = pA + cosize(lA);
  Tensor sA = make_tensor(make_smem_ptr(pA), lA);
  Tensor sB = make_tensor(make_smem_ptr(pB), lB);
  for (int i = threadIdx.x; i < M_TILE * K_TILE; i += blockDim.x)
    sA(i / K_TILE, i % K_TILE) = gA[i];
  for (int i = threadIdx.x; i < N * K_TILE; i += blockDim.x)
    sB(i / K_TILE, i % K_TILE) = gB[i];
  __syncthreads();
  da = G::make_gmma_desc<G::Major::K>(sA);
  db = G::make_gmma_desc<G::Major::K>(sB);
}

// NGROUP wgmma are issued, then committed as one group; WAIT groups are
// allowed to stay outstanding. Instructions in flight is therefore
// NGROUP * (WAIT + 1), which is the axis M2 sweeps.
template <int N, int NGROUP, int WAIT, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void rate_kernel(const bf16* __restrict__ gA, const bf16* __restrict__ gB,
                 int trip, float* __restrict__ sink,
                 long long* __restrict__ cycles) {
  using Atom = MMA_Atom<typename AtomFor<N>::type>;
  constexpr int NREG = N / 2;
  uint64_t da, db;
  stage<N>(gA, gB, da, db);

  float acc[NREG];
#pragma unroll
  for (int i = 0; i < NREG; ++i) acc[i] = 0.f;
  auto tA = make_tensor(make_rmem_ptr(&da), Layout<_1>{});
  auto tB = make_tensor(make_rmem_ptr(&db), Layout<_1>{});
  auto tC = make_tensor(make_rmem_ptr(acc), Layout<Int<NREG>>{});

  fence_acc(acc);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  __syncthreads();
  const long long t0 = clock64();
  for (int i = 0; i < trip; ++i) {
#pragma unroll
    for (int j = 0; j < NGROUP; ++j) Atom{}.call(tA, tB, tC);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(WAIT) : "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  fence_acc(acc);
  const long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  // Never true; keeps every accumulator live so none of the wgmma is dropped.
  if (acc[0] == 1234.5678f) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NREG; ++i) s += acc[i];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

// One wgmma on real data, D written out in row/col order. The accumulator
// mapping below is the m64nNk16 f32 layout; if it is wrong the host comparison
// fails, which is the point -- it is checked, not recalled.
template <int N>
__global__ __launch_bounds__(128, 1)
void check_kernel(const bf16* __restrict__ gA, const bf16* __restrict__ gB,
                  float* __restrict__ out) {
  using Atom = MMA_Atom<typename AtomFor<N>::type>;
  constexpr int NREG = N / 2;
  uint64_t da, db;
  stage<N>(gA, gB, da, db);

  float acc[NREG];
#pragma unroll
  for (int i = 0; i < NREG; ++i) acc[i] = 0.f;
  auto tA = make_tensor(make_rmem_ptr(&da), Layout<_1>{});
  auto tB = make_tensor(make_rmem_ptr(&db), Layout<_1>{});
  auto tC = make_tensor(make_rmem_ptr(acc), Layout<Int<NREG>>{});

  fence_acc(acc);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  Atom{}.call(tA, tB, tC);
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  fence_acc(acc);

  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int r = 0; r < NREG; ++r) {
    const int g = r >> 2, s = r & 3;
    const int row = wid * 16 + (lane >> 2) + 8 * (s >> 1);
    const int col = g * 8 + (lane & 3) * 2 + (s & 1);
    out[row * N + col] = acc[r];
  }
}

template <int N>
__device__ constexpr size_t smem_bytes() {
  return sizeof(bf16) * (size_t)(M_TILE + N) * K_TILE;
}


// ---------------------------------------------------------------- mma.sync
//
// The warp-level instruction, for the regime wgmma is bad at. MMA-RATE found
// that wgmma below N=64 sits on a ~19-cycle per-instruction floor and reaches
// only 20% of peak at N=8, so the obvious question is whether the older
// warp-level mma.sync does better on a small output tile. It reads operands
// from REGISTERS rather than shared memory; that is the difference between the
// two instructions, and it is stated rather than controlled away.

__device__ __forceinline__ uint32_t pack2(bf16 lo, bf16 hi) {
  uint32_t l = *reinterpret_cast<const uint16_t*>(&lo);
  uint32_t h = *reinterpret_cast<const uint16_t*>(&hi);
  return l | (h << 16);
}

__device__ __forceinline__ void mma_16816(float& d0, float& d1, float& d2,
                                          float& d3, uint32_t a0, uint32_t a1,
                                          uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The m16n8k16 fragment layout, from the PTX ISA. Written out rather than
// buried in a helper because mma_sync_check_kernel is what proves it, and a
// reader has to be able to check the two against each other.
//   gid = lane/4, tg = lane%4
//   A(16x16) a0..a3 : rows gid and gid+8, cols tg*2 (+8), two elements per reg
//   B(8x16)  b0..b1 : col n = gid, k = tg*2 and tg*2+8
//   D(16x8)  d0..d3 : rows gid and gid+8, cols tg*2 and tg*2+1
__device__ __forceinline__ void load_frags(const bf16* A, const bf16* B,
                                           uint32_t (&a)[4], uint32_t (&b)[2]) {
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tg = lane & 3;
  a[0] = pack2(A[gid * 16 + tg * 2], A[gid * 16 + tg * 2 + 1]);
  a[1] = pack2(A[(gid + 8) * 16 + tg * 2], A[(gid + 8) * 16 + tg * 2 + 1]);
  a[2] = pack2(A[gid * 16 + tg * 2 + 8], A[gid * 16 + tg * 2 + 9]);
  a[3] = pack2(A[(gid + 8) * 16 + tg * 2 + 8], A[(gid + 8) * 16 + tg * 2 + 9]);
  b[0] = pack2(B[gid * 16 + tg * 2], B[gid * 16 + tg * 2 + 1]);
  b[1] = pack2(B[gid * 16 + tg * 2 + 8], B[gid * 16 + tg * 2 + 9]);
}

__global__ __launch_bounds__(32, 1)
void mma_sync_check_kernel(const bf16* __restrict__ gA,
                           const bf16* __restrict__ gB,
                           float* __restrict__ out) {
  uint32_t a[4], b[2];
  load_frags(gA, gB, a, b);
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  mma_16816(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1]);
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tg = lane & 3;
  out[gid * 8 + tg * 2 + 0] = d[0];
  out[gid * 8 + tg * 2 + 1] = d[1];
  out[(gid + 8) * 8 + tg * 2 + 0] = d[2];
  out[(gid + 8) * 8 + tg * 2 + 1] = d[3];
}

// NACC independent accumulators per warp. NACC == 1 chains every instruction
// into the previous one's result and so measures LATENCY; NACC > 1 measures
// issue throughput. One axis separating the two, the way ring depth did in the
// TMA unit.
template <int NACC, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void mma_sync_rate_kernel(const bf16* __restrict__ gA,
                          const bf16* __restrict__ gB, int trip,
                          float* __restrict__ sink,
                          long long* __restrict__ cycles) {
  uint32_t a[4], b[2];
  load_frags(gA, gB, a, b);
  float d[NACC][4];
#pragma unroll
  for (int i = 0; i < NACC; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) d[i][j] = 0.f;

  __syncthreads();
  const long long t0 = clock64();
  for (int i = 0; i < trip; ++i) {
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      mma_16816(d[j][0], d[j][1], d[j][2], d[j][3],
                a[0], a[1], a[2], a[3], b[0], b[1]);
  }
  const long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (d[0][0] == 1234.5678f) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) s += d[i][j];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

}  // namespace mmaprobe

// (index, N, NGROUP, WAIT). Kept as an explicit list so the instantiation
// count stays visible: every row is a kernel nvcc must compile and a register
// allocation that can fail on its own.
#define MMA_CONFIGS                        \
  X( 0,   8, 2, 1) X( 1,  16, 2, 1)        \
  X( 2,  32, 2, 1) X( 3,  64, 2, 1)        \
  X( 4,  96, 2, 1) X( 5, 128, 2, 1)        \
  X( 6, 192, 2, 1) X( 7, 256, 2, 1)        \
  X( 8,  64, 1, 0) X( 9,  64, 1, 1)        \
  X(10,  64, 1, 3) X(11,  64, 2, 0)        \
  X(12,  64, 4, 0) X(13,  64, 4, 1)        \
  X(14,  64, 8, 0) X(15, 128, 1, 0)        \
  X(16, 128, 4, 1) X(17, 256, 1, 0)

extern "C" {

int mma_probe_rate(int cfg, int n_ctas, int n_threads, const void* gA,
                   const void* gB, int trip, void* sink, void* cycles,
                   void* stream) {
  using namespace mmaprobe;
  auto s = (cudaStream_t)stream;
  const bf16* a = (const bf16*)gA;
  const bf16* b = (const bf16*)gB;
#define X(IDX, N, NG, W)                                                     \
  case IDX: {                                                                \
    const size_t sm = sizeof(bf16) * (size_t)(M_TILE + N) * K_TILE;          \
    if (n_threads == 128)                                                    \
      rate_kernel<N, NG, W, 128><<<n_ctas, 128, sm, s>>>(                    \
          a, b, trip, (float*)sink, (long long*)cycles);                     \
    else if (n_threads == 256)                                               \
      rate_kernel<N, NG, W, 256><<<n_ctas, 256, sm, s>>>(                    \
          a, b, trip, (float*)sink, (long long*)cycles);                     \
    else return 1302;                                                        \
    break;                                                                   \
  }
  switch (cfg) { MMA_CONFIGS default: return 1301; }
#undef X
  return (int)cudaGetLastError();
}

int mma_probe_check(int n, const void* gA, const void* gB, void* out,
                    void* stream) {
  using namespace mmaprobe;
  auto s = (cudaStream_t)stream;
  const bf16* a = (const bf16*)gA;
  const bf16* b = (const bf16*)gB;
#define C(N)                                                                 \
  case N: check_kernel<N><<<1, 128, sizeof(bf16) * (M_TILE + N) * K_TILE, s>>>( \
              a, b, (float*)out); break;
  switch (n) { C(8) C(16) C(32) C(64) C(128) C(256) default: return 1303; }
#undef C
  return (int)cudaGetLastError();
}

// (N, NGROUP, WAIT) for one config index, so the host never has to keep a
// duplicate of the table above.
int mma_probe_cfg(int cfg, int field) {
  switch (cfg) {
#define X(IDX, N, NG, W) \
  case IDX: return field == 0 ? N : (field == 1 ? NG : W);
    MMA_CONFIGS
#undef X
    default: return -1;
  }
}

int mma_probe_sync_rate(int nacc, int n_ctas, int n_threads, const void* gA,
                        const void* gB, int trip, void* sink, void* cycles,
                        void* stream) {
  using namespace mmaprobe;
  auto s = (cudaStream_t)stream;
  const bf16* a = (const bf16*)gA;
  const bf16* b = (const bf16*)gB;
#define S(NA, NT)                                                            \
  if (nacc == NA && n_threads == NT) {                                       \
    mma_sync_rate_kernel<NA, NT><<<n_ctas, NT, 0, s>>>(                      \
        a, b, trip, (float*)sink, (long long*)cycles);                       \
    return (int)cudaGetLastError();                                          \
  }
  S(1, 32) S(1, 64) S(1, 128) S(1, 256)
  S(2, 32) S(2, 64) S(2, 128) S(2, 256)
  S(4, 32) S(4, 64) S(4, 128) S(4, 256)
  S(8, 32) S(8, 64) S(8, 128) S(8, 256)
#undef S
  return 1304;
}

int mma_probe_sync_check(const void* gA, const void* gB, void* out,
                         void* stream) {
  using namespace mmaprobe;
  mma_sync_check_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
      (const bf16*)gA, (const bf16*)gB, (float*)out);
  return (int)cudaGetLastError();
}

int mma_probe_cfg_count() {
  int n = 0;
#define X(IDX, N, NG, W) ++n;
  MMA_CONFIGS
#undef X
  return n;
}

}  // extern "C"
