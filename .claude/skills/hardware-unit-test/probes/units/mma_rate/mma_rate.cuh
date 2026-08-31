// mma_rate.cuh -- tensor-core issue rate, isolated from everything else.
//
// One warpgroup (or two) issues wgmma back to back out of resident shared
// memory. No TMA, no global traffic, no barriers in the loop: the only thing
// timed is how fast the tensor core retires instructions. The operands are
// re-issued from the same two smem tiles every iteration -- that keeps address
// arithmetic out of the loop, and a real GEMM mainloop accumulates every k-step
// into the same registers anyway, so the dependency structure is realistic.
//
// Three instruction families, because the question "which instruction for this
// tile" only has an answer if all three are measured under one harness:
// wgmma SS, mma.sync from registers, and mma.sync fed by ldmatrix.

#pragma once

#include "hut/common.cuh"
#include "hut/mma.cuh"
#include "hut/unit.hpp"

namespace hut {
namespace mma_rate {

using namespace cute;

constexpr int32_t M_TILE = 64;   // one warpgroup's rows
constexpr int32_t K_TILE = 16;   // one wgmma k-step

// The instruction, its TiledMMA and the compile-time assertion that the
// selector landed on it all come from hut/mma.cuh -- three units were building
// these six aliases independently. Only the ATOM below is this unit's choice.
template <int N> using TileShape = hut::TileShape<M_TILE, N, K_TILE>;
template <int N> using MmaOp = hut::SsMmaOp<M_TILE, N, K_TILE>;
template <int N> using TiledMmaFor = hut::SsTiledMma<M_TILE, N, K_TILE>;

HUT_ASSERT_SS_ATOM_BF16(64, 8, 16);   HUT_ASSERT_SS_ATOM_BF16(64, 16, 16);
HUT_ASSERT_SS_ATOM_BF16(64, 32, 16);  HUT_ASSERT_SS_ATOM_BF16(64, 64, 16);
HUT_ASSERT_SS_ATOM_BF16(64, 96, 16);  HUT_ASSERT_SS_ATOM_BF16(64, 128, 16);
HUT_ASSERT_SS_ATOM_BF16(64, 192, 16); HUT_ASSERT_SS_ATOM_BF16(64, 256, 16);

// K-major interleaved smem, the layout the SS atom's descriptor expects. One
// k-step deep: the probe re-issues against the same tile every iteration on
// purpose (see the header), so there is no PIPE mode to carry.
// INTERLEAVED, not swizzled: this unit re-issues against one resident tile and
// no TMA is filling stages underneath, so the swizzle a pipelined mainloop
// needs would only change the descriptor without changing what is measured.
template <int N>
using SmemLayoutA = hut::SmemLayout<G::Layout_K_INTER_Atom<bf16>, M_TILE, K_TILE>;
template <int N>
using SmemLayoutB = hut::SmemLayout<G::Layout_K_INTER_Atom<bf16>, N, K_TILE>;

extern __shared__ uint8_t smem_pool[];

// One warpgroup of the TiledMMA. Every thread of a 256-thread launch slices
// the SAME 128-thread TiledMMA, so two warpgroups issue independent wgmma
// against one pair of tiles -- which is the M3 axis, and is what the
// hand-rolled version did.

template <int N>
__device__ __forceinline__ auto smem_A() {
  return make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(smem_pool)),
                     SmemLayoutA<N>{});
}

template <int N>
__device__ __forceinline__ auto smem_B() {
  return make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(smem_pool)
                                   + cosize(SmemLayoutA<N>{})),
                     SmemLayoutB<N>{});
}

// Fill the two smem tiles through their cute layouts, so the layout the
// descriptor is built from and the data actually written agree by construction
// rather than by a matching pair of hand calculations.  The descriptors
// themselves are built by the MMA traits inside make_fragment_A/B, not here.
template <int N>
__device__ __forceinline__ void stage(const bf16* gA, const bf16* gB) {
  auto sA = smem_A<N>();
  auto sB = smem_B<N>();
  for (int i = threadIdx.x; i < M_TILE * K_TILE; i += blockDim.x)
    sA(i / K_TILE, i % K_TILE) = gA[i];
  for (int i = threadIdx.x; i < N * K_TILE; i += blockDim.x)
    sB(i / K_TILE, i % K_TILE) = gB[i];
  __syncthreads();
}

// NGROUP wgmma are issued, then committed as one group; WAIT groups are
// allowed to stay outstanding. Instructions in flight is therefore
// NGROUP * (WAIT + 1), which is the axis M2 sweeps.
//
// One cute::gemm is exactly ONE wgmma here: the tile K equals the atom K, so
// MMA_M = MMA_N = MMA_K = 1 and the internal k-loop has a single k_tile_count. cute
// does NOT insert the warpgroup fence/commit/wait -- those stay the caller's,
// which is what keeps NGROUP and WAIT meaning what they meant before.
template <int N, int NGROUP, int WAIT, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void rate_kernel(const bf16* __restrict__ gA, const bf16* __restrict__ gB,
                 int32_t k_tile_count, float* __restrict__ sink,
                 long long* __restrict__ cycles) {
  using Mma = TiledMmaFor<N>;
  static_assert(size(Mma{}) == kWarpgroupThreads, "SS atom is warpgroup-wide");
  stage<N>(gA, gB);

  Mma tiled_mma;
  tiled_mma.accumulate_ = G::ScaleOut::One;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x % kWarpgroupThreads);
  Tensor tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(smem_A<N>()));
  Tensor tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(smem_B<N>()));
  Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape<N>{}));
  clear(accum);

  // Without this ptxas emits C7515 -- "wgmma.mma_async instructions are
  // serialized due to non wgmma instructions defining accumulator registers" --
  // and every rate below would be the SERIALIZED rate, not the pipelined one.
  // It tells the compiler wgmma owns these registers so nothing else is
  // scheduled into them. A rate probe that ignored the warning would have
  // measured a real number for the wrong machine.
  warpgroup_fence_operand(accum);
  warpgroup_arrive();
  __syncthreads();
  const long long t0 = clock64();
  for (int i = 0; i < k_tile_count; ++i) {
#pragma unroll
    for (int j = 0; j < NGROUP; ++j) cute::gemm(tiled_mma, tCrA, tCrB, accum);
    warpgroup_commit_batch();
    warpgroup_wait<WAIT>();
  }
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);
  const long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  // Never true; keeps every accumulator live so none of the wgmma is dropped.
  if (accum(0) == 1234.5678f) {
    float s = 0.f;
    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) s += accum(i);
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

// One wgmma on real data, D written out in row/col order for the host to
// compare against a reference.  The accumulator-to-(row, col) mapping is
// `partition_C`, i.e. it comes from the same MMA traits that issued the
// instruction -- so it is correct by construction for every N and every
// element type, where a hand-written m64nNk16 decode was one table to keep in
// step with the atom table above.
template <int N>
__global__ __launch_bounds__(128, 1)
void check_kernel(const bf16* __restrict__ gA, const bf16* __restrict__ gB,
                  float* __restrict__ out) {
  using Mma = TiledMmaFor<N>;
  stage<N>(gA, gB);

  Mma tiled_mma;
  tiled_mma.accumulate_ = G::ScaleOut::One;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(smem_A<N>()));
  Tensor tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(smem_B<N>()));
  Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape<N>{}));
  clear(accum);

  warpgroup_fence_operand(accum);
  warpgroup_arrive();
  cute::gemm(tiled_mma, tCrA, tCrB, accum);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

  // Row-major (M_TILE, N), matching the torch tensor the host allocates.
  Tensor gD = make_tensor(make_gmem_ptr(out),
                          make_shape(Int<M_TILE>{}, Int<N>{}),
                          make_stride(Int<N>{}, _1{}));
  cute::copy(accum, thr_mma.partition_C(gD));
}

template <int N>
__device__ constexpr size_t smem_bytes() {
  return sizeof(bf16) * (size_t)(M_TILE + N) * K_TILE;
}


// ---------------------------------------------------------------- mma.sync
//
// The warp-level instruction, for the regime wgmma is bad at. wgmma.issue.wg.ss found
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
// issue throughput. One axis separating the two, the way ring stages did in the
// TMA unit.
template <int NACC, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void mma_sync_rate_kernel(const bf16* __restrict__ gA,
                          const bf16* __restrict__ gB, int32_t k_tile_count,
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
  for (int i = 0; i < k_tile_count; ++i) {
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


// ------------------------------------------------------- mma.sync + ldmatrix
//
// mma.issue.warp and mma.xover.n.wgmma were measured with operands ALREADY IN
// REGISTERS: the probe loads fragments once and never reloads them. A real
// mma.sync mainloop pays `ldmatrix` on every k-step, and wgmma does not -- it
// reads shared memory directly. That tax is the term that can erase the
// crossover, so it is measured here rather than assumed away.
//
// The axis that matters is REUSE: a warp computing an (AM*16) x (BN*8) tile
// issues AM*BN mma.sync but only AM+BN ldmatrix, so the tax falls as the tile
// grows. Sweeping (AM, BN) gives the tax as a function of the ratio, which is
// what a tiling decision actually needs.

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(uint32_t (&d)[4], const void* p) {
  uint32_t a = smem_addr(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}

__device__ __forceinline__ void ldm_x2(uint32_t (&d)[2], const void* p) {
  uint32_t a = smem_addr(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}

// Per-lane row pointers. Derived from ldmatrix's own distribution rule --
// matrix i element (r,c) lands in lane r*4 + c/2, register i -- matched
// against the m16n8k16 fragment layout in load_frags above. The check kernel
// is what proves the two agree.
//   A(16x16) row-major, 32 B rows: lanes 0-7 -> rows 0-7 cols 0-7, 8-15 ->
//   rows 8-15 cols 0-7, 16-23 -> rows 0-7 cols 8-15, 24-31 -> rows 8-15 c 8-15
__device__ __forceinline__ const bf16* a_row_ptr(const bf16* base, int lane) {
  const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
  return base + r * 16 + ((lane >= 16) ? 8 : 0);
}
//   B(8x16) k-contiguous, 32 B rows: lanes 0-7 -> n 0-7 k 0-7,
//   lanes 8-15 -> n 0-7 k 8-15. Lanes 16-31 are unused by .x2 but must still
//   address in-bounds memory.
__device__ __forceinline__ const bf16* b_row_ptr(const bf16* base, int lane) {
  const int r = lane & 7, mat = (lane >> 3) & 1;
  return base + r * 16 + mat * 8;
}

extern __shared__ uint8_t ldm_pool[];

__global__ __launch_bounds__(32, 1)
void mma_sync_ldm_check_kernel(const bf16* __restrict__ gA,
                               const bf16* __restrict__ gB,
                               float* __restrict__ out) {
  bf16* sA = reinterpret_cast<bf16*>(ldm_pool);
  bf16* sB = sA + 16 * 16;
  for (int i = threadIdx.x; i < 16 * 16; i += 32) sA[i] = gA[i];
  for (int i = threadIdx.x; i < 8 * 16; i += 32) sB[i] = gB[i];
  __syncwarp();

  const int lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
  ldm_x4(a, a_row_ptr(sA, lane));
  ldm_x2(b, b_row_ptr(sB, lane));

  float d[4] = {0.f, 0.f, 0.f, 0.f};
  mma_16816(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1]);

  const int gid = lane >> 2, tg = lane & 3;
  out[gid * 8 + tg * 2 + 0] = d[0];
  out[gid * 8 + tg * 2 + 1] = d[1];
  out[(gid + 8) * 8 + tg * 2 + 0] = d[2];
  out[(gid + 8) * 8 + tg * 2 + 1] = d[3];
}

// AM A-fragments and BN B-fragments reloaded from smem every iteration, then
// AM*BN mma.sync issued from them -- the shape of one k-step of a real
// mainloop. Accumulators stay in registers across the loop, as they would.
template <int AM, int BN, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void mma_sync_ldm_rate_kernel(const bf16* __restrict__ gA,
                              const bf16* __restrict__ gB, int32_t k_tile_count,
                              float* __restrict__ sink,
                              long long* __restrict__ cycles) {
  bf16* sA = reinterpret_cast<bf16*>(ldm_pool);
  bf16* sB = sA + AM * 16 * 16;
  for (int i = threadIdx.x; i < AM * 16 * 16; i += blockDim.x)
    sA[i] = gA[i % (16 * 16)];
  for (int i = threadIdx.x; i < BN * 8 * 16; i += blockDim.x)
    sB[i] = gB[i % (8 * 16)];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  float acc[AM][BN][4];
#pragma unroll
  for (int m = 0; m < AM; ++m)
#pragma unroll
    for (int n = 0; n < BN; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.f;

  const long long t0 = clock64();
  for (int i = 0; i < k_tile_count; ++i) {
    uint32_t a[AM][4], b[BN][2];
#pragma unroll
    for (int m = 0; m < AM; ++m)
      ldm_x4(a[m], a_row_ptr(sA + m * 16 * 16, lane));
#pragma unroll
    for (int n = 0; n < BN; ++n)
      ldm_x2(b[n], b_row_ptr(sB + n * 8 * 16, lane));
#pragma unroll
    for (int m = 0; m < AM; ++m)
#pragma unroll
      for (int n = 0; n < BN; ++n)
        mma_16816(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                  a[m][0], a[m][1], a[m][2], a[m][3], b[n][0], b[n][1]);
  }
  const long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (acc[0][0][0] == 1234.5678f) {
    float s = 0.f;
#pragma unroll
    for (int m = 0; m < AM; ++m)
#pragma unroll
      for (int n = 0; n < BN; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) s += acc[m][n][j];
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

}  // namespace mma_rate
}  // namespace hut

// (index, N, n_groups, wait). An explicit list so the instantiation count stays
// visible: every row is a kernel nvcc must compile and a register allocation
// that can fail on its own.
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

// (index, a_tiles_m, b_tiles_n): a warp computes an (a_tiles_m*16) x
// (b_tiles_n*8) tile, so it issues a_tiles_m*b_tiles_n mma.sync against
// a_tiles_m+b_tiles_n ldmatrix. The list spans reuse ratios 0.5 to 2.7.
#define LDM_CONFIGS                        \
  Y( 0, 1, 1) Y( 1, 1, 4) Y( 2, 1, 7)      \
  Y( 3, 2, 2) Y( 4, 2, 4) Y( 5, 2, 7)      \
  Y( 6, 4, 2) Y( 7, 4, 4) Y( 8, 4, 7)      \
  Y( 9, 8, 4)

