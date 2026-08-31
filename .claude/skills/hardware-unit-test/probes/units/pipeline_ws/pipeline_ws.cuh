// pipeline_ws.cuh -- the warp-specialised mainloop, measured as CUTLASS builds it.
//
// Producer and consumer warpgroups over CUTLASS's PipelineTmaAsync, so the
// pipeline being measured is the library's rather than ours: producer_acquire /
// consumer_wait / consumer_release / producer_tail are the reference mainloop's
// calls, not a re-implementation that might synchronise differently.
//
// PipelineTmaAsync has exactly one consumer in this suite, so it stays here
// rather than in a shared header -- there is no duplication to factor out, and
// a shared abstraction with one user is harder to read, not easier.

#pragma once

// hut/mma.cuh FIRST: it pulls cute/tensor.hpp, and copy_traits_sm90_tma.hpp
// included ahead of that redeclares cute::copy_if.
#include "hut/mma.cuh"

// This unit's own dependencies -- the TMA copy traits and CUTLASS's pipeline.
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "hut/common.cuh"
#include "hut/unit.hpp"
#include "hut/watchdog.cuh"

namespace hut {
namespace pipeline_ws {

using namespace cute;

constexpr int M_TILE = 64;
constexpr int K_PIPE_MMAS = 1;          // matches the CUTLASS mainloop default

// Register reallocation between the two warpgroups, as the warp-specialized
// kernels do: the producer needs almost none, the consumer holds accumulators.
constexpr int PROD_REGS = 40;
constexpr int CONS_REGS = 232;

// Rule 9: a persistent probe hangs rather than fails unless every wait carries
// a deadline. The first version of this probe had none, and a barrier-count
// mistake cost a full Slurm slot and reported nothing. The pipeline's blocking
// calls are inside CUTLASS, so the deadline is built from its own try_* API --
// which the reference mainloop uses anyway (consumer_try_wait then
// consumer_wait), so this stays faithful rather than diverging to get a
// watchdog.

__device__ __forceinline__ void trap_at(long long* dbg, int site) {
  if (dbg) {
    dbg[blockIdx.x * 2] = site;
    dbg[blockIdx.x * 2 + 1] = threadIdx.x;
    __threadfence_system();
  }
  __trap();
}

template <int N, int BK> using TileShape = hut::TileShape<M_TILE, N, BK>;
template <int N, int BK> using TiledMmaFor = hut::SsTiledMma<M_TILE, N, BK>;

// SWIZZLED, unlike mma_rate and overlap: a TMA is filling stages underneath, so
// the descriptor and the wgmma operand layout have to agree on the swizzle.
// Layout_K_SW128_Atom<bf16> is (8, 64), so tile_to_shape static_asserts on any
// BK it does not divide -- the good failure. [hut/mma.cuh]
template <int BK, int S>
using SmemLayoutA = decltype(tile_to_shape(
    G::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<M_TILE>, Int<BK>, Int<S>>{}));
template <int N, int BK, int S>
using SmemLayoutB = decltype(tile_to_shape(
    G::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<N>, Int<BK>, Int<S>>{}));

template <int N, int BK, int S>
struct Storage {
  alignas(128) cute::ArrayEngine<bf16, cosize_v<SmemLayoutA<BK, S>>> A;
  alignas(128) cute::ArrayEngine<bf16, cosize_v<SmemLayoutB<N, BK, S>>> B;
  alignas(16) typename cutlass::PipelineTmaAsync<S>::SharedStorage pipe;
};

template <int N, int BK, int S, class TmaA, class TmaB>
__global__ __launch_bounds__(2 * kWarpgroupThreads, 1)
void pipeline_ws_kernel(CUTE_GRID_CONSTANT TmaA const tma_a,
                        CUTE_GRID_CONSTANT TmaB const tma_b,
                        int k_tiles, int mode,
                        long long* __restrict__ cyc_prod,
                        long long* __restrict__ cyc_cons,
                        float* __restrict__ sink,
                        long long* __restrict__ dbg) {
  using Pipeline = cutlass::PipelineTmaAsync<S>;
  using PipeState = cutlass::PipelineState<S>;
  using Store = Storage<N, BK, S>;

  extern __shared__ char raw[];
  Store& st = *reinterpret_cast<Store*>(raw);

  Tensor sA = make_tensor(make_smem_ptr(st.A.begin()), SmemLayoutA<BK, S>{});
  Tensor sB = make_tensor(make_smem_ptr(st.B.begin()), SmemLayoutB<N, BK, S>{});

  const int wg = cutlass::canonical_warp_group_idx();
  const bool is_producer = (wg == 0);

  // Barrier arrival counts follow the reference: every producer-warpgroup
  // thread is a producer for the empty barrier, every consumer thread for the
  // full barrier. Getting these wrong deadlocks rather than mismeasures.
  typename Pipeline::Params params;
  params.transaction_bytes =
      static_cast<uint32_t>(sizeof(bf16) * (M_TILE + N) * BK);
  params.role = is_producer ? Pipeline::ThreadCategory::Producer
                            : Pipeline::ThreadCategory::Consumer;
  params.is_leader = is_producer && (threadIdx.x % kWarpgroupThreads == 0);
  // CUTLASS counts `num_consumers` in THREADS, not in warpgroups: it is how
  // many arrivals the empty barrier waits for on consumer_release. One consumer
  // warpgroup is 128 threads, so the two are numerically equal here and would
  // stop being equal the moment a second consumer warpgroup is added.
  // `num_producers` is 1 because a single elected lane issues the TMA.
  params.num_consumers = kWarpgroupThreads;
  params.num_producers = 1;

  Pipeline pipeline(st.pipe, params, Shape<_1, _1, _1>{});
  __syncthreads();

  // The gmem tensors carry a K-TILE MODE and the producer advances it, so the
  // source is a stream rather than one resident tile. The first version of this
  // probe had no k mode: 132 CTAs re-read the SAME 24 KB, the walk never left
  // L2, and it measured a coupled pipeline fed at 9.9-14.8 TB/s -- above the
  // DRAM ceiling and therefore not a DRAM measurement at all. Partitioned once
  // outside the loop, as sm90_mma_tma_gmma_ss_warpspecialized.hpp does.
  Tensor gA = tma_a.get_tma_tensor(make_shape(Int<M_TILE>{}, Int<BK>{}, k_tiles));
  Tensor gB = tma_b.get_tma_tensor(make_shape(Int<N>{}, Int<BK>{}, k_tiles));
  auto ta = tma_a.get_slice(0);
  auto tb = tma_b.get_slice(0);
  Tensor tAgA = ta.partition_S(gA);            // (CPY,CPY_M,CPY_K,k)
  Tensor tBgB = tb.partition_S(gB);            // (CPY,CPY_N,CPY_K,k)
  Tensor tAsA = ta.partition_D(sA);            // (CPY,CPY_M,CPY_K,STAGE)
  Tensor tBsB = tb.partition_D(sB);            // (CPY,CPY_N,CPY_K,STAGE)

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<PROD_REGS>();
    if (mode == 2) return;   // no producer in consumer_only

    PipeState write = cutlass::make_producer_start_state<Pipeline>();
    const long long t0 = clock64();

    if (cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync()) {
      uint32_t ph = 0;                   // mode 1 only
      for (int k = 0; k < k_tiles; ++k) {
        int s;
        typename Pipeline::ProducerBarrierType* bar;
        if (mode == 0) {
          // The reference sequence: acquire (which also does expect_tx), take
          // the stage's barrier, and let the TMA itself complete it. There is
          // no producer_commit for PipelineTmaAsync.
          // Bounded producer_acquire: spin on try_acquire with a deadline
          // rather than the blocking form, so a wrong empty-barrier arrival
          // count reports its site instead of hanging the job.
          {
            long long t = clock64();
            while (pipeline.producer_try_acquire(write).get()
                   != cutlass::BarrierStatus::WaitDone) {
              if (clock64() - t > kWatchdogCycles) trap_at(dbg, 11);
            }
          }
          pipeline.producer_acquire(write);
          bar = pipeline.producer_get_barrier(write);
          s = write.index();
          ++write;
        } else {
          // No consumer releases anything here, so the producer drains on its
          // own FULL barrier -- tma_ring's ring, not half of the pipeline,
          // because make_producer_start_state leaves the phase inverted and
          // reusing it outside the protocol is how a probe deadlocks.
          s = k % S;
          // `full_barrier_[s]` is the barrier OBJECT; producer_get_barrier
          // hands back its raw mbarrier word, which is what tma.with() takes.
          // The two are the same storage seen at two levels.
          auto& fb = st.pipe.full_barrier_[s];
          if (k >= S) {
            long long t = clock64();
            while (!fb.try_wait((ph >> s) & 1u)) {
              if (clock64() - t > kWatchdogCycles) trap_at(dbg, 12);
            }
            ph ^= 1u << s;
          }
          fb.arrive_and_expect_tx(params.transaction_bytes);
          bar = reinterpret_cast<typename Pipeline::ProducerBarrierType*>(&fb);
        }
        // Each CTA starts at a different k so the grid streams distinct bytes
        // rather than 132 CTAs sharing one working set.
        const int kt = (blockIdx.x + k) % k_tiles;
        copy(tma_a.with(*bar), tAgA(_, _, _, kt), tAsA(_, _, _, s));
        copy(tma_b.with(*bar), tBgB(_, _, _, kt), tBsB(_, _, _, s));
      }
      if (mode == 0) {
        // producer_tail with a deadline: acquiring every stage is what it does,
        // and the blocking form has no try_ variant to bound.
        for (int j = 0; j < S; ++j) {
          long long t = clock64();
          while (pipeline.producer_try_acquire(write).get()
                 != cutlass::BarrierStatus::WaitDone) {
            if (clock64() - t > kWatchdogCycles) trap_at(dbg, 14);
          }
          ++write;
        }
      } else {
        const int tail = k_tiles < S ? k_tiles : S;
        for (int j = 0; j < tail; ++j) {
          const int s = (k_tiles - tail + j) % S;
          long long t = clock64();
          while (!st.pipe.full_barrier_[s].try_wait((ph >> s) & 1u)) {
            if (clock64() - t > kWatchdogCycles) trap_at(dbg, 13);
          }
          ph ^= 1u << s;
        }
      }
    }
    const long long t1 = clock64();
    if (threadIdx.x == 0) cyc_prod[blockIdx.x] = t1 - t0;
  } else {
    cutlass::arch::warpgroup_reg_alloc<CONS_REGS>();
    if (mode == 1) return;   // no consumer in producer_only

    TiledMmaFor<N, BK> tiled_mma;
    auto thr = tiled_mma.get_slice(threadIdx.x % kWarpgroupThreads);
    Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));
    Tensor tCrB = thr.make_fragment_B(thr.partition_B(sB));
    Tensor accum = partition_fragment_C(tiled_mma,
                                        take<0, 2>(TileShape<N, BK>{}));
    clear(accum);
    tiled_mma.accumulate_ = G::ScaleOut::One;

    PipeState read, release;
    const long long t0 = clock64();
    warpgroup_fence_operand(accum);
    for (int k = 0; k < k_tiles; ++k) {
      int s;
      if (mode == 0) {
        long long t = clock64();
        while (pipeline.consumer_try_wait(read).get()
               != cutlass::BarrierStatus::WaitDone) {
          if (clock64() - t > kWatchdogCycles) trap_at(dbg, 21);
        }
        s = read.index();
      } else {
        s = k % S;                       // no producer: re-read a resident stage
      }
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_, _, _, s), tCrB(_, _, _, s), accum);
      warpgroup_commit_batch();
      // Release LAGS read by K_PIPE_MMAS -- that lag is what keeps wgmma in
      // flight across the stage boundary.
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);
      if (mode == 0 && k >= K_PIPE_MMAS) {
        pipeline.consumer_release(release);
        ++release;
      }
      ++read;
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    // mma_tail: the mainloop skipped the first K_PIPE_MMAS releases so that
    // `release` could lag `read`; those stages are still locked and must be
    // handed back, or the producer's tail waits for a release that never comes.
    // sm90_mma_tma_gmma_ss_warpspecialized.hpp does this in mma_tail().
    if (mode == 0) {
      for (int j = 0; j < K_PIPE_MMAS; ++j) {
        pipeline.consumer_release(release);
        ++release;
      }
    }
    const long long t1 = clock64();
    if (threadIdx.x % kWarpgroupThreads == 0) cyc_cons[blockIdx.x] = t1 - t0;
    if (accum(0) == 1234.5678f) {
      float acc = 0.f;
      CUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) acc += accum(i);
      sink[blockIdx.x] = acc;
    }
  }
}

}  // namespace pipeline_ws
}  // namespace hut

// (index, N, BK, Stages). BK is fixed at 64 -- see the SmemLayout note above.
#define WS_CONFIGS                                   \
  X(0,  64, 64, 4) X(1, 128, 64, 4) X(2, 256, 64, 4) \
  X(3, 128, 64, 2) X(4, 128, 64, 3) X(5, 128, 64, 6)

namespace hut {
namespace pipeline_ws {

template <int N, int BK, int S>
static int32_t launch_cfg(const void* pa, const void* pb, int n_ctas, int k_tiles,
                      int mode, void* cyc_prod, void* cyc_cons, void* sink,
                      void* dbg, cudaStream_t stream) {
  // (M, BK, k_tiles): k-major tiles laid out contiguously, so k_tiles x the
  // per-tile bytes IS the footprint the walk touches. The caller sizes k_tiles
  // to put that footprint clear of L2.
  auto mA = make_tensor(make_gmem_ptr(reinterpret_cast<const bf16*>(pa)),
                        make_shape(Int<M_TILE>{}, Int<BK>{}, k_tiles),
                        make_stride(Int<BK>{}, _1{}, Int<M_TILE * BK>{}));
  auto mB = make_tensor(make_gmem_ptr(reinterpret_cast<const bf16*>(pb)),
                        make_shape(Int<N>{}, Int<BK>{}, k_tiles),
                        make_stride(Int<BK>{}, _1{}, Int<N * BK>{}));
  // The canonical descriptor construction: one stage of the staged smem layout
  // IS the TMA box, so the box and the wgmma operand layout agree by
  // construction rather than by two hand calculations that must match.
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA,
                             SmemLayoutA<BK, S>{}(_, _, Int<0>{}));
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB,
                             SmemLayoutB<N, BK, S>{}(_, _, Int<0>{}));
  size_t sm = sizeof(Storage<N, BK, S>);
  if (sm > 232448u) return HUT_ERR_SMEM;
  auto kern = pipeline_ws_kernel<N, BK, S, decltype(tma_a), decltype(tma_b)>;
  cudaError_t e = cudaFuncSetAttribute(
      kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  if (e != cudaSuccess) return (int32_t)e;
  kern<<<n_ctas, 2 * kWarpgroupThreads, sm, stream>>>(
      tma_a, tma_b, k_tiles, mode, (long long*)cyc_prod,
      (long long*)cyc_cons, (float*)sink, (long long*)dbg);
  return (int32_t)cudaGetLastError();
}

}  // namespace pipeline_ws
}  // namespace hut