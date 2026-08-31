// overlap.cuh -- do the copy engine and the tensor core actually overlap?
//
// One CTA runs both at once: a consumer warpgroup issuing wgmma against
// resident tiles while `num_producers` warps refill TMA rings underneath. The
// three MODES (tma_only, wgmma_only, both) share one kernel and one shared
// memory layout on purpose -- an isolated run and a contended run that differ
// in their descriptor or tile offsets are not a pair. [protocol.md rule 3]

#pragma once

#include "hut/barrier.cuh"
#include "hut/common.cuh"
#include "hut/mma.cuh"
#include "hut/tma.cuh"
#include "hut/unit.hpp"
#include "hut/watchdog.cuh"

namespace hut {
namespace overlap {

using namespace cute;

extern __shared__ uint8_t smem_pool[];

// One wgmma tile pair, interleaved: the consumer re-issues against resident
// tiles while the producers refill rings underneath. [hut/mma.cuh]
constexpr int32_t M_TILE = 64;
constexpr int32_t K_TILE = 16;
template <int N> using TileShape = hut::TileShape<M_TILE, N, K_TILE>;
template <int N> using TiledMmaFor = hut::SsTiledMma<M_TILE, N, K_TILE>;
template <int N>
using SmemLayoutA = hut::SmemLayout<G::Layout_K_INTER_Atom<bf16>, M_TILE, K_TILE>;
template <int N>
using SmemLayoutB = hut::SmemLayout<G::Layout_K_INTER_Atom<bf16>, N, K_TILE>;

HUT_ASSERT_SS_ATOM_BF16(64, 64, 16);
HUT_ASSERT_SS_ATOM_BF16(64, 128, 16);
HUT_ASSERT_SS_ATOM_BF16(64, 256, 16);


constexpr int kWarpgroupThreads = 128;      // the consumer warpgroup
constexpr int kConsumerWarps = 4;
constexpr int kMaxProducers = 8;
constexpr int kMaxStages = 8;
constexpr long long WATCHDOG_CYCLES = 1ll << 31;


// The consumer's operand tiles sit above the producer rings. The offset moves
// with num_producers, which is fine and necessary: sizing the rings for kMaxProducers would
// need 512 KB at an 8 KB box. What must NOT move is the offset between the
// three MODES of one config, and it does not -- num_producers is passed unchanged to
// all three, so `wgmma_only` reads the descriptor `both` reads.
__host__ __device__ __forceinline__ size_t ring_bytes(int num_producers, int stages,
                                             int box_bytes) {
  return (size_t)num_producers * stages * box_bytes;
}

template <int N, int NGROUP, int WAIT, int NTHREADS>
__global__ __launch_bounds__(NTHREADS, 1)
void overlap_kernel(const __grid_constant__ CUtensorMap tensor_map,
                    int stages, int box_bytes, int k_tiles_tma, int k_tiles_mma,
                    int num_producers, int mode,
                    int mask0, int shift0, int step0, int mask1, int step1,
                    long long* __restrict__ cyc_tma,
                    long long* __restrict__ cyc_mma,
                    float* __restrict__ sink,
                    long long* __restrict__ dbg) {
  const int warp = threadIdx.x >> 5;
  const bool is_consumer = warp < kConsumerWarps;
  const int prod = warp - kConsumerWarps;

  uint8_t* rings = smem_pool;
  uint8_t* mma_tiles = smem_pool + ring_bytes(num_producers, stages, box_bytes);
  uint64_t* bars = reinterpret_cast<uint64_t*>(
      mma_tiles + sizeof(bf16) * (M_TILE + N) * K_TILE);

  if (threadIdx.x == 0) {
    TransactionBarrier* b = reinterpret_cast<TransactionBarrier*>(bars);
    for (int i = 0; i < num_producers * stages; ++i) b[i].init(1);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // ---------------------------------------------------------- consumer side
  if (is_consumer) {
    using Mma = TiledMmaFor<N>;
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(mma_tiles)),
                          SmemLayoutA<N>{});
    auto sB = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16*>(mma_tiles)
                      + cosize(SmemLayoutA<N>{})), SmemLayoutB<N>{});
    for (int i = threadIdx.x; i < M_TILE * K_TILE; i += kWarpgroupThreads)
      sA(i / K_TILE, i % K_TILE) = bf16(0.01f);
    for (int i = threadIdx.x; i < N * K_TILE; i += kWarpgroupThreads)
      sB(i / K_TILE, i % K_TILE) = bf16(0.01f);
  }
  __syncthreads();

  if (is_consumer && mode != 1) {
    using Mma = TiledMmaFor<N>;
    Mma tiled_mma;
    tiled_mma.accumulate_ = G::ScaleOut::One;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x % kWarpgroupThreads);
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(mma_tiles)),
                          SmemLayoutA<N>{});
    auto sB = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16*>(mma_tiles)
                      + cosize(SmemLayoutA<N>{})), SmemLayoutB<N>{});
    Tensor tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
    Tensor tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape<N>{}));
    clear(accum);

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    const long long t0 = clock64();
    for (int i = 0; i < k_tiles_mma; ++i) {
#pragma unroll
      for (int j = 0; j < NGROUP; ++j) cute::gemm(tiled_mma, tCrA, tCrB, accum);
      warpgroup_commit_batch();
      warpgroup_wait<WAIT>();
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    const long long t1 = clock64();
    if (threadIdx.x == 0) cyc_mma[blockIdx.x] = t1 - t0;
    if (accum(0) == 1234.5678f) {
      float s = 0.f;
      CUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) s += accum(i);
      sink[blockIdx.x] = s;
    }
  }

  // ---------------------------------------------------------- producer side
  if (!is_consumer && prod < num_producers && mode != 2) {
    uint8_t* my_pool = rings + (size_t)prod * stages * box_bytes;
    TransactionBarrier* mybar = reinterpret_cast<TransactionBarrier*>(bars) + prod * stages;
    uint32_t ph = 0;
    const int base = blockIdx.x * num_producers + prod;
    const int lane0 = (threadIdx.x & 31) == 0;

    const long long t0 = clock64();
    for (int g = 0; g < k_tiles_tma; ++g) {
      const int s = g % stages;
      if (g >= stages) {
        wait(&mybar[s], (ph >> s) & 1, dbg, kSiteUnitBase + 1);
        ph ^= 1u << s;
      }
      __syncwarp();
      const int idx = base * k_tiles_tma + g;
      const int32_t c0 = (idx & mask0) * step0;
      const int32_t c1 = ((idx >> shift0) & mask1) * step1;
      if (lane0) {
        mybar[s].arrive_and_expect_tx(box_bytes);
        cp_async_bulk_tensor_2d(&tensor_map, my_pool + (size_t)s * box_bytes, c0, c1,
               &mybar[s]);
      }
    }
    const int tail = k_tiles_tma < stages ? k_tiles_tma : stages;
    for (int j = 0; j < tail; ++j) {
      const int s = (k_tiles_tma - tail + j) % stages;
      wait(&mybar[s], (ph >> s) & 1, dbg, kSiteUnitBase + 2);
      ph ^= 1u << s;
    }
    const long long t1 = clock64();
    if (prod == 0 && lane0) cyc_tma[blockIdx.x] = t1 - t0;
  }
  __syncthreads();
}

}  // namespace overlap
}  // namespace hut

#define OVERLAP_CONFIGS \
  X(0, 64, 2, 1) X(1, 128, 2, 1) X(2, 256, 2, 1) X(3, 64, 4, 1)
