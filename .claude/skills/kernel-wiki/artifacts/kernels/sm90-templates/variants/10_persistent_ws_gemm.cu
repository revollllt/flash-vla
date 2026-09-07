// Template 10 -- persistent warp-specialized GEMM with cluster multicast (sm90).
//
// The canonical high-performance sm90 GEMM shape, as DeepGEMM's sm90 kernels
// and CUTLASS's Hopper warp-specialized collectives build it.  Everything in
// templates 01-04 appears here at once, plus the four things that separate a
// real GEMM from a mainloop:
//
//   1. A persistent grid with an L2-aware raster order.  One CTA per SM runs
//      the whole problem, taking tiles from a scheduler instead of being one
//      tile.  The raster group width is chosen to minimise operand rows loaded
//      per wave -- see best_group_width below; walking pure row-major instead
//      re-reads the whole B panel every wave.
//   2. Cluster multicast.  Two CTAs sharing an A tile receive it from one
//      transaction: only rank 0 issues, the mask names both, and each CTA arms
//      its OWN barrier for the bytes it will receive.  This halves A traffic,
//      which is why it is worth the cluster constraint.
//   3. Per-warp barrier arrivals, not per-thread.  An empty barrier expecting
//      one arrival per math warp (times the multicast width) is cheaper than
//      one per thread and is what lets a consumer release with a single lane.
//   4. Frames released after the batch that reads them retires, not on issue
//      [pattern-serialized-wgmma, technique-release-on-retirement].
//
// It also has to use DYNAMIC shared memory.  Static __shared__ is capped at
// 48 KB per CTA; a real GEMM ring is several times that, so the pool is one
// extern array carved by constexpr offsets and the host opts in with
// cudaFuncSetAttribute before the first launch.  Forgetting that call is a
// launch failure, not a slow kernel.
//
// The cluster teardown wait at the end is not optional: a DSMEM barrier whose
// remote arrivals are still in flight cannot be safely deconstructed when the
// kernel exits.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-GRADE: structural
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.mbarrier::complete_tx::bytes\.multicast::cluster
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n128k16\.f32\.bf16\.bf16
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: mbarrier\.arrive\.shared::cluster\.b64
// CHECK-PTX: mapa\.shared::cluster\.u32
// CHECK-PTX: barrier\.cluster\.arrive
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.global\.shared::cta\.bulk_group

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kStages = 4;  // cold DRAM feed [pipeline.stages.wg.knee]

// Two CTAs per cluster share the A tile.  Wider clusters exist but cost
// scheduling freedom, and a 128-CTA grid at 1 CTA/SM cannot use cluster 4 or 8
// without deadlock risk [cluster.count.max].
constexpr int kCluster = 2;
constexpr uint16_t kClusterMask = (1u << kCluster) - 1;

// Two math warpgroups stack along M, so one CTA tile is 128 rows.  The second
// warpgroup adds no tensor-core throughput [wgmma.ratio.sm.wg2]; it is here to
// own half the M tile, not to add FLOPs.
constexpr int kMathWarpgroups = 2;
constexpr int kMathThreads = kMathWarpgroups * tmpl::kWarpgroupThreads;
constexpr int kMathWarps = kMathThreads / tmpl::kWarpThreads;
constexpr int kTmaThreads = tmpl::kWarpgroupThreads;
constexpr int kThreads = kMathThreads + kTmaThreads;

constexpr int kMathRegs = 224;  // 248 when there is a single math warpgroup
constexpr int kTmaRegs = 48;

// H100 SXM5.  The grid is exactly this, so the scheduler's stride is a
// constant and every CTA stays resident.
constexpr uint32_t kNumSMs = 132;
constexpr uint32_t kNumClusters = kNumSMs / kCluster;

constexpr int kTileElems = kBlockM * kBlockK;  // A and B frames are equal here
constexpr int kFrameBytes = kTileElems * static_cast<int>(sizeof(Element));

// One shared pool, carved at compile time.  Barriers last so the tiles keep
// their 1024-B alignment.
constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kFrameBytes;
constexpr int kOffD = kOffB + kStages * kFrameBytes;
constexpr int kOffBar = kOffD + kBlockM * kBlockN * static_cast<int>(sizeof(Element));
constexpr int kSmemBytes = kOffBar + 2 * kStages * static_cast<int>(sizeof(uint64_t));

// A wave of kNumSMs CTAs laid out `width` tiles along N and ceil(SMs/width)
// along M loads width*BLOCK_N + rows*BLOCK_M operand rows.  Minimising that
// sum is the whole of L2-aware rasterisation; the best width is rarely 1 (pure
// column-major) or kNumSMs (pure row-major).
constexpr uint32_t best_group_width() {
  uint32_t best = 1, best_cost = ~0u;
  for (uint32_t w = 1; w <= kNumSMs; ++w) {
    const uint32_t rows = (kNumSMs + w - 1) / w;
    const uint32_t cost = w * kBlockN + rows * kBlockM;
    if (cost < best_cost) { best_cost = cost; best = w; }
  }
  return best;
}
constexpr uint32_t kGroupWidth = best_group_width();

// Hands out tiles in that raster order.  Stateless apart from the iteration
// counter: no shared cursor, so no atomic and no contention [atom.rate.addr].
//
// The unit handed out is a CLUSTER tile, not a CTA tile: both CTAs of a cluster
// must land on the same (m_block, n_block) or they do not share an A tile and
// the multicast premise is gone.  Each CTA then takes sub-column `rank` of it.
struct Scheduler {
  uint32_t m_blocks, n_blocks, total;
  int32_t iter = -1;

  __device__ Scheduler(uint32_t m, uint32_t n)
      : m_blocks((m + kBlockM - 1) / kBlockM),
        n_blocks((n + kBlockN * kCluster - 1) / (kBlockN * kCluster)),
        total(m_blocks * n_blocks) {}

  __device__ bool next(uint32_t& m_block, uint32_t& n_block) {
    const uint32_t idx = static_cast<uint32_t>(++iter) * kNumClusters +
                         blockIdx.x / kCluster;
    if (idx >= total) { return false; }
    // Walk kGroupWidth columns of N fully before advancing M, so the A rows a
    // wave touches stay resident in L2 across the group.
    const uint32_t per_group = m_blocks * kGroupWidth;
    const uint32_t group = idx / per_group;
    const uint32_t in_group = idx % per_group;
    const uint32_t first_n = group * kGroupWidth;
    const uint32_t width = min(kGroupWidth, n_blocks - first_n);
    m_block = in_group / width;
    n_block = first_n + in_group % width;
    return true;
  }
};

using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutA =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));
using SmemLayoutB =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockK>>{}));

using MmaAtom = SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K,
                                                         GMMA::Major::K>;
using TiledMma =
    decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<Int<kMathWarpgroups>, _1, _1>>{}));

}  // namespace

__global__ __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kThreads, 1)
void persistent_ws_gemm_kernel(const __grid_constant__ CUtensorMap map_a,
                               const __grid_constant__ CUtensorMap map_b,
                               const __grid_constant__ CUtensorMap map_d,
                               uint32_t shape_m, uint32_t shape_n,
                               uint32_t shape_k) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<Element(*)[kTileElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<Element(*)[kTileElems]>(smem + kOffB);
  auto* const sd = reinterpret_cast<Element*>(smem + kOffD);
  auto* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  auto* const empty = full + kStages;

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / tmpl::kWarpThreads;
  const uint32_t rank = tmpl::cluster_ctarank();
  const bool is_math = tid < kMathThreads;

  if (warp == kMathWarps && tmpl::elect_one()) {
    for (int s = 0; s < kStages; ++s) {
      // One arrival: the single lane that issues this stage's copies.
      tmpl::mbarrier_init(&full[s], 1);
      // One arrival per math warp, from every CTA that shares the frame.
      tmpl::mbarrier_init(&empty[s], kCluster * kMathWarps);
    }
    tmpl::fence_barrier_init();
  }
  // Cluster-wide: a peer CTA may arrive on these barriers, so every CTA must
  // see them initialised before any of them starts.
  tmpl::cluster_sync();

  Scheduler scheduler(shape_m, shape_n);
  const uint32_t k_tiles = (shape_k + kBlockK - 1) / kBlockK;

  if (!is_math) {
    tmpl::setmaxnreg_dec<kTmaRegs>();
    if (warp != kMathWarps) { return; }  // registers already released

    uint32_t m_block, n_block;
    // One running frame counter for the whole persistent run: the ring does not
    // restart per tile, so slot and phase both derive from it.
    uint32_t g = 0;
    while (scheduler.next(m_block, n_block)) {
      for (uint32_t k = 0; k < k_tiles; ++k, ++g) {
        const uint32_t stage = g % kStages;
        const uint32_t use = g / kStages;
        if (use > 0) { tmpl::wait_parity(&empty[stage], (use - 1) & 1u); }
        if (tmpl::elect_one()) {
          // Each CTA arms its own barrier: it expects the bytes IT receives,
          // which is the multicast A tile plus its own private B tile.
          tmpl::arrive_and_expect_tx(&full[stage], 2 * kFrameBytes);
          // Only rank 0 issues the multicast; issuing from both would deliver
          // the tile twice and overflow every expect_tx.
          if (rank == 0) {
            tmpl::tma_load_2d_multicast(&map_a, sa[stage], k * kBlockK,
                                        m_block * kBlockM, &full[stage],
                                        kClusterMask);
          }
          // B differs per CTA of the cluster, so it is a plain load.
          tmpl::tma_load_2d(&map_b, sb[stage], k * kBlockK,
                            (n_block * kCluster + rank) * kBlockN, &full[stage]);
        }
      }
    }
    // Teardown: every empty phase this CTA's barriers will ever complete must
    // complete before the barriers go out of scope, or a remote arrival is
    // still in flight when the cluster dissolves.  Slot s was used
    // g/kStages times, plus once more if it is below the final partial pass.
    for (uint32_t s = 0; s < kStages; ++s) {
      const uint32_t uses = g / kStages + (s < g % kStages ? 1u : 0u);
      if (uses > 0) { tmpl::wait_parity(&empty[s], (uses - 1) & 1u); }
    }
    return;
  }

  tmpl::setmaxnreg_inc<kMathRegs>();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

  uint32_t m_block, n_block;
  uint32_t g = 0;  // must advance in lockstep with the producer's counter
  while (scheduler.next(m_block, n_block)) {
    clear(acc);
    for (uint32_t k = 0; k < k_tiles; ++k, ++g) {
      const uint32_t stage = g % kStages;
      const uint32_t use = g / kStages;
      tmpl::wait_parity(&full[stage], use & 1u);

      Tensor tile_a = make_tensor(make_smem_ptr(sa[stage]), SmemLayoutA{});
      Tensor tile_b = make_tensor(make_smem_ptr(sb[stage]), SmemLayoutB{});
      auto frag_a = thr_mma.make_fragment_A(thr_mma.partition_A(tile_a));
      auto frag_b = thr_mma.make_fragment_B(thr_mma.partition_B(tile_b));

      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      tiled_mma.accumulate_ = (k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
      CUTE_UNROLL
      for (int kb = 0; kb < size<2>(frag_a); ++kb) {
        cute::gemm(tiled_mma, frag_a(_, _, kb), frag_b(_, _, kb), acc);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);

      // Safe now, and only now: the batch that read this frame has retired.
      // One lane per math warp arrives, on every CTA that received the frame.
      if (tmpl::elect_one()) {
        CUTE_UNROLL
        for (uint32_t c = 0; c < kCluster; ++c) {
          tmpl::mbarrier_arrive_cluster(&empty[stage], c);
        }
      }
    }

    // Epilogue: partition_C gives the shared tile the same thread mapping the
    // accumulator already has, so no cross-thread shuffle is needed.
    Tensor tile_d = make_tensor(make_smem_ptr(sd),
                                make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{},
                                            LayoutRight{}));
    Tensor frag_d = thr_mma.partition_C(tile_d);
    CUTE_UNROLL
    for (int i = 0; i < size(acc); ++i) {
      frag_d(i) = static_cast<Element>(acc(i));
    }

    tmpl::fence_proxy_async_shared();
    tmpl::named_barrier_sync(1, kMathThreads);
    if (tid == 0) {
      tmpl::tma_store_2d(&map_d, sd, (n_block * kCluster + rank) * kBlockN,
                         m_block * kBlockM);
      tmpl::tma_store_commit();
      tmpl::tma_store_wait<0>();
    }
    tmpl::named_barrier_sync(1, kMathThreads);
  }
}

// Must run once before the first launch: dynamic shared memory above 48 KB is
// opt-in per kernel, and the launch fails without it.
cudaError_t configure_persistent_ws_gemm() {
  return cudaFuncSetAttribute(persistent_ws_gemm_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}

// Grid is the SM count exactly, and must be a whole number of clusters.
static_assert(kNumSMs % kCluster == 0, "grid must divide into whole clusters");
static_assert(kSmemBytes <= 227 * 1024, "exceeds the H100 per-CTA shared limit");
