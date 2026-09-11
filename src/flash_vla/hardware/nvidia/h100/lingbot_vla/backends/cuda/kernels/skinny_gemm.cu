// Weight-stationary bf16 GEMM for the action expert's four M = 51 projections.
//
// Per denoise step per layer the expert issues packed qkv [51,768]x[2560,768],
// o_proj [51,2048]x[768,2048], packed gate_up [51,768]x[5504,768] and
// down_proj [51,2752]x[768,2752].  Each moves 3.1 - 8.5 MB of weights to do
// 0.2 - 0.4 GFLOP, so they are cold reads, not math: the machine's measured
// model is t_us = 1.85 + MB/2.77 [unit-launch ld.bw.dev.dram], and 51 rows
// cannot reach a quarter of tensor peak under any tiling.
//
// One CTA is one warpgroup.  It owns a TileN slice of the output and a
// contiguous run of K tiles, streams the weight through a shared-memory ring
// and issues wgmma with both operands in shared memory.  M = 51 is padded to
// the wgmma M of 64: the pad rows are zeroed once, never copied into, and
// dropped in the epilogue.
//
// Two producers are compiled, because the first measurement said the choice
// is the whole kernel.  With cp.async the ring is filled by the four math
// warps themselves and one CTA reached only ~19.5 GB/s -- about what an SM's
// miss handling sustains from four warps -- so the kernel ran at the speed of
// however many SMs the N axis happened to fill (job 614332: 8.99 us on 40
// CTAs for packed qkv against cuBLAS's 5.28 us).  The TMA producer issues the
// same tile from one elected thread and lets the copy engine hold the
// requests, which is the only way a 4-warp CTA reaches its share of DRAM.
//
// N alone cannot supply 128 CTAs for o_proj and down_proj (N = 768 is 12 tiles
// of 64), so the K axis splits too.  Each split writes its own fp32 partial
// and the last CTA of an N tile sums them.  The first version accumulated
// with red.global.add instead, which needs no reduction pass but costs one L2
// atomic per split per output element: measured on job 614381, that epilogue
// added ~8 us to a 3.8 us mainloop for packed qkv -- about 100 G atomics/s,
// which is the L2's rate, not a tuning problem.  Plain partials plus one
// vectorised pass move the same bytes at streaming rate.
//
// Who sums the partials is the second half of that measurement.  One CTA per
// N tile reading k_split partials is MSHR-bound at ~13 GB/s (job 614394: the
// epilogue grew 4 us as the reducer's read grew from 26 to 78 KB), so all
// k_split CTAs of a tile reduce a slice each and every one reads the same
// 13 KB whatever k_split is.  That needs the tile's CTAs to be co-resident,
// which the host checks against the occupancy of this exact kernel; when they
// are not, the kernel falls back to the single-reducer path.
//
// Only the arrival counters carry state between launches, and the CTAs reset
// them, so a captured graph replays without a zeroing launch.  The counters
// must be zero before the first launch; the partial workspace need not be.
//
// SKINNY_PROBE_NO_REDUCE builds a variant that publishes partials and arrives
// but never sums them, to separate the arrival's cost from the read's.
//
// The split-K reduction has a third form that never leaves the chip.  When the
// splits of one N tile are the CTAs of a cluster, each stages its partial in
// its own shared memory and reads its siblings' through the cluster's
// distributed shared window, so there is no global round trip, no fp32
// workspace and no device-scope release fence -- only two cluster barriers.
// Measured cost of the global form it replaces: 1.8 - 2.5 us on a ~3.9 us
// mainloop (jobs 614426, 614476).
//
// SKINNY_PROBE_NO_EPILOGUE builds a variant whose mainloop is unchanged but
// whose epilogue is a single guarded store, so a run can price the split-K
// reduction against the stream it sits on.  It computes the wrong answer by
// construction and exists only for that measurement.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include <cute/tensor.hpp>

#include "tile/sm90/barrier.cuh"
#include "tile/sm90/common.cuh"
#include "tile/sm90/copy_g2s.cuh"
#include "tile/sm90/gemm.cuh"
#include "tile/sm90/smem_layout.cuh"
#include "tile/sm90/tma_host.cuh"

namespace flash_vla::lingbot::skinny {

using namespace cute;
namespace tile = flash_vla::sm90;

using Element = tile::BF16;

// One warpgroup: wgmma's M of 64 already covers all 51 rows, so a second
// warpgroup would have to split N and halve the per-instruction N, which is
// the shared-memory-bound direction [wgmma.issue.wg.ss].
constexpr int kThreads = tile::kWarpgroupThreads;
constexpr int kTileM = 64;
// One SW128 atom row is 128 B = 64 bf16, so a K tile is exactly one swizzle
// span: one TMA box, and no second shared-memory layout for the copy.
constexpr int kTileK = 64;
constexpr int kMaxSmemBytes = 226 * 1024;

template <int TileN, int Depth>
struct Config {
  using SmemA = typename tile::SmemTileLayout<Element, kTileM, kTileK,
                                              tile::Major::K, 128>::type;
  using SmemB = typename tile::SmemTileLayout<Element, TileN, kTileK,
                                              tile::Major::K, 128>::type;
  using Mma = tile::MmaSelector<Element, Element, kTileM, TileN, kTileK,
                                tile::Operand::kSmem, tile::Operand::kSmem,
                                tile::Major::K, tile::Major::K, kThreads, 1>;
  using TiledMma = typename Mma::TiledMma;

  static constexpr int kFrameA = kTileM * kTileK;
  static constexpr int kFrameB = TileN * kTileK;
  static constexpr int kFrameBytesB = kFrameB * static_cast<int>(sizeof(Element));
  static constexpr int kRingBytes =
      Depth * (kFrameA + kFrameB) * static_cast<int>(sizeof(Element));
  // The barriers follow the ring; 1024-byte alignment of the ring keeps every
  // frame's base legal for a swizzled TMA box.
  // The epilogue restages the accumulator in the ring; with the row pad that
  // is a little wider than an m x TileN fp32 tile, so the allocation is the
  // larger of the two uses.
  static constexpr int kStageBytes = kTileM * (TileN + 4) * 4;
  // full (TMA completion), empty (cluster: both CTAs have armed a frame) and
      // consumed (local: the math warps have retired a frame).
  static constexpr int kBarrierBytes = 3 * Depth * 8;
  static constexpr int kSmemBytes =
      (kRingBytes > kStageBytes ? kRingBytes : kStageBytes) + kBarrierBytes;
  static constexpr bool kFits = kSmemBytes <= kMaxSmemBytes;
};

// Release/acquire at device scope.  __threadfence() is fence.sc.gpu, which is
// stronger than the split-K handshake needs: the partials only have to be
// visible before the arrival that publishes them, and visible again after the
// arrival that observes them.
__device__ __forceinline__ void fence_release_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

// Distributed shared memory: `mapa` rewrites a local shared address into the
// same offset in another CTA of the cluster, and the ::cluster qualifier is
// what makes the load leave this SM.  A plain ld.shared would read this CTA's
// own bytes instead, silently.
__device__ __forceinline__ uint32_t map_shared_rank(uint32_t address,
                                                    uint32_t rank) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(address), "r"(rank));
  return mapped;
}

__device__ __forceinline__ float4 load_shared_cluster(uint32_t address) {
  float4 value;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "r"(address));
  return value;
}

// Converts and stores the finished N tile, or publishes this split's partial
// and lets the last arrival of the tile sum them.
template <int TileN, class AccTensor, class CoordTensor>
__device__ __forceinline__ void epilogue(const AccTensor& acc,
                                         const CoordTensor& tCcC, int32_t tid,
                                         int32_t n_tile, int32_t n_base,
                                         int32_t split, int32_t m, int32_t n,
                                         int32_t k_split, bool coresident,
                                         const __nv_bfloat16* __restrict__ bias,
                                         __nv_bfloat16* __restrict__ out,
                                         float* __restrict__ workspace,
                                         int32_t* __restrict__ counters,
                                         void* smem_scratch) {
  // The wgmma f32 accumulator holds each thread's values as (n-pair, m-pair,
  // n-block), so acc(i) and acc(i + 1) are always columns c and c + 1 of one
  // row: one 4-byte bf16 store, or one 8-byte fp32 store, per pair.
  if (k_split == 1) {
    CUTE_UNROLL
    for (int i = 0; i < size(acc); i += 2) {
      const int row = get<0>(tCcC(i));
      const int col = n_base + get<1>(tCcC(i));
      if (row >= m || col >= n) { continue; }
      float a = acc(i);
      float b = acc(i + 1);
      const bool pair = (col + 1) < n;
      if (bias != nullptr) {
        a += __bfloat162float(bias[col]);
        b += pair ? __bfloat162float(bias[col + 1]) : 0.0f;
      }
      __nv_bfloat16* dst = out + static_cast<int64_t>(row) * n + col;
      if (pair) {
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(a, b);
      } else {
        dst[0] = __float2bfloat16_rn(a);
      }
    }
    return;
  }

  // Partials are TileN wide whatever the N tail is, so the reduction reads a
  // dense block and only the store has to know where N ends.
  const int64_t tile_elems = static_cast<int64_t>(m) * TileN;
  float* const partial =
      workspace + (static_cast<int64_t>(n_tile) * k_split + split) * tile_elems;
  const int32_t vectors = static_cast<int32_t>(tile_elems) / 4;

  // The accumulator's per-thread image is 8-byte column pairs scattered over
  // eight rows, so storing it straight out is 16 two-float stores per thread
  // that the release fence then has to wait on.  Staging through the ring --
  // dead once the mainloop has drained -- turns the publication into one
  // dense float4 per thread per step.  The row pad keeps the eight rows a
  // warp writes off the same shared-memory banks.
  constexpr int kPad = 4;
  constexpr int kStage = TileN + kPad;
  float* const stage = reinterpret_cast<float*>(smem_scratch);
  // The staging area is the mainloop's ring; wgmma retirement is warpgroup
  // collective but this barrier is what orders it against the first warp's
  // write into a frame another warp may not have finished reading.
  __syncthreads();
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 2) {
    const int row = get<0>(tCcC(i));
    const int col = get<1>(tCcC(i));
    if (row >= m) { continue; }
    *reinterpret_cast<float2*>(stage + row * kStage + col) =
        make_float2(acc(i), acc(i + 1));
  }
  __syncthreads();
  for (int32_t v = tid; v < vectors; v += kThreads) {
    const int32_t element = v * 4;
    reinterpret_cast<float4*>(partial)[v] = *reinterpret_cast<const float4*>(
        stage + (element / TileN) * kStage + (element % TileN));
  }

  // The fence orders this CTA's partial before its arrival.  `arrive` counts
  // publications and `depart` completions: the pair is what lets the counters
  // return to zero without a CTA resetting `arrive` while a sibling is still
  // spinning on it.
  fence_release_gpu();
  __syncthreads();
  int32_t* const arrive = &counters[n_tile];
  int32_t* const depart = &counters[n_tile + gridDim.x];
  __shared__ int32_t reduce_here;
  if (tid == 0) {
    const int32_t prior = atomicAdd(arrive, 1);
    if (coresident) {
      // Every CTA of the tile reduces, so every CTA waits for all of them.
      while (__ldcg(arrive) < k_split) { __nanosleep(64); }
      reduce_here = 1;
    } else {
      reduce_here = (prior == k_split - 1) ? 1 : 0;
      if (reduce_here != 0) {
        *arrive = 0;
      }
    }
  }
  __syncthreads();
  if (reduce_here == 0) { return; }
  fence_release_gpu();

#ifndef SKINNY_PROBE_NO_REDUCE
  // One float4 per thread per step keeps k_split loads of one output vector in
  // flight at once, which is the whole reason this pass is not the atomics it
  // replaced.  TileN is a multiple of 4, so a vector never crosses a row.
  const float* const base =
      workspace + static_cast<int64_t>(n_tile) * k_split * tile_elems;
  const int32_t first = coresident ? split * kThreads + tid : tid;
  const int32_t step = coresident ? k_split * kThreads : kThreads;
  constexpr int kBatch = 8;
  for (int32_t v = first; v < vectors; v += step) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int32_t s = 0;
    // Issue the batch before consuming it: a plain loop over the splits leaves
    // one load in flight per thread and this pass has no other parallelism.
    // .cg bypasses L1, which is not coherent with the other splits' stores.
    for (; s + kBatch <= k_split; s += kBatch) {
      float4 part[kBatch];
      CUTE_UNROLL
      for (int j = 0; j < kBatch; ++j) {
        part[j] = __ldcg(reinterpret_cast<const float4*>(
            base + static_cast<int64_t>(s + j) * tile_elems) + v);
      }
      CUTE_UNROLL
      for (int j = 0; j < kBatch; ++j) {
        sum.x += part[j].x;
        sum.y += part[j].y;
        sum.z += part[j].z;
        sum.w += part[j].w;
      }
    }
    for (; s < k_split; ++s) {
      const float4 part = __ldcg(reinterpret_cast<const float4*>(
          base + static_cast<int64_t>(s) * tile_elems) + v);
      sum.x += part.x;
      sum.y += part.y;
      sum.z += part.z;
      sum.w += part.w;
    }
    const int32_t element = v * 4;
    const int32_t row = element / TileN;
    const int32_t col = n_base + (element % TileN);
    float value[4] = {sum.x, sum.y, sum.z, sum.w};
    __nv_bfloat16* dst = out + static_cast<int64_t>(row) * n + col;
    if (col + 3 < n) {
      if (bias != nullptr) {
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) { value[j] += __bfloat162float(bias[col + j]); }
      }
      const __nv_bfloat162 lo = __floats2bfloat162_rn(value[0], value[1]);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(value[2], value[3]);
      reinterpret_cast<__nv_bfloat162*>(dst)[0] = lo;
      reinterpret_cast<__nv_bfloat162*>(dst)[1] = hi;
    } else {
      for (int j = 0; j < 4; ++j) {
        if (col + j >= n) { break; }
        float scalar = value[j];
        if (bias != nullptr) { scalar += __bfloat162float(bias[col + j]); }
        dst[j] = __float2bfloat16_rn(scalar);
      }
    }
  }
#endif

  if (coresident) {
    // The CTA that finishes last clears the pair; by then every sibling has
    // passed the spin above, so neither counter can be read again this launch.
    __syncthreads();
    if (tid == 0 && atomicAdd(depart, 1) == k_split - 1) {
      *arrive = 0;
      *depart = 0;
    }
  }
}

// Split-K epilogue for a cluster whose CTAs are the splits of one N tile.
// Each CTA stages its partial in its own shared memory, one cluster barrier
// publishes all of them, and each CTA then sums and stores its own slice of
// the tile -- so the reduction reads (m x TileN x 4) / Cluster bytes per CTA
// out of the cluster's shared window rather than out of L2.
template <int TileN, int Cluster, class AccTensor, class CoordTensor>
__device__ __forceinline__ void cluster_epilogue(
    const AccTensor& acc, const CoordTensor& tCcC, int32_t tid, int32_t n_base,
    int32_t split, int32_t m, int32_t n,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    void* smem_scratch) {
  constexpr int kPad = 4;
  constexpr int kStage = TileN + kPad;
  float* const stage = reinterpret_cast<float*>(smem_scratch);
  // The staging area is the mainloop's ring; this barrier orders wgmma
  // retirement against the first warp's write into a frame another warp may
  // not have finished reading.
  __syncthreads();
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 2) {
    const int row = get<0>(tCcC(i));
    const int col = get<1>(tCcC(i));
    if (row >= m) { continue; }
    *reinterpret_cast<float2*>(stage + row * kStage + col) =
        make_float2(acc(i), acc(i + 1));
  }
  // barrier.cluster.arrive carries release and .wait carries acquire, so the
  // siblings' stages are visible to the reads below and nothing has to fence.
  cute::cluster_sync();

  const int32_t vectors = m * TileN / 4;
  const int32_t chunk = (vectors + Cluster - 1) / Cluster;
  const int32_t begin = split * chunk;
  const int32_t end = (begin + chunk) < vectors ? (begin + chunk) : vectors;
  const uint32_t base = tile::smem_u32(stage);
  for (int32_t v = begin + tid; v < end; v += kThreads) {
    const int32_t element = v * 4;
    const int32_t row = element / TileN;
    const int32_t col = n_base + (element % TileN);
    const uint32_t offset = base + static_cast<uint32_t>(
        (row * kStage + (element % TileN)) * sizeof(float));
    float4 sum = *reinterpret_cast<const float4*>(
        stage + row * kStage + (element % TileN));
    // Start at the next rank so the cluster's CTAs do not all read rank 0 at
    // the same moment.
    CUTE_UNROLL
    for (int q = 1; q < Cluster; ++q) {
      const uint32_t rank = static_cast<uint32_t>((split + q) % Cluster);
      const float4 part = load_shared_cluster(map_shared_rank(offset, rank));
      sum.x += part.x;
      sum.y += part.y;
      sum.z += part.z;
      sum.w += part.w;
    }
    float value[4] = {sum.x, sum.y, sum.z, sum.w};
    __nv_bfloat16* dst = out + static_cast<int64_t>(row) * n + col;
    if (col + 3 < n) {
      if (bias != nullptr) {
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) { value[j] += __bfloat162float(bias[col + j]); }
      }
      reinterpret_cast<__nv_bfloat162*>(dst)[0] =
          __floats2bfloat162_rn(value[0], value[1]);
      reinterpret_cast<__nv_bfloat162*>(dst)[1] =
          __floats2bfloat162_rn(value[2], value[3]);
    } else {
      for (int j = 0; j < 4; ++j) {
        if (col + j >= n) { break; }
        float scalar = value[j];
        if (bias != nullptr) { scalar += __bfloat162float(bias[col + j]); }
        dst[j] = __float2bfloat16_rn(scalar);
      }
    }
  }
  // No CTA may retire while a sibling still maps its shared memory.
  cute::cluster_sync();
}

// The K range this CTA owns.  The first k_tiles % k_split splits take one
// extra tile so no split is empty and every CTA reaches the arrival.
__device__ __forceinline__ void split_range(int32_t k_tiles, int32_t k_split,
                                            int32_t split, int32_t* begin,
                                            int32_t* count) {
  const int32_t per_split = k_tiles / k_split;
  const int32_t extra = k_tiles % k_split;
  *begin = split * per_split + (split < extra ? split : extra);
  *count = per_split + (split < extra ? 1 : 0);
}

template <int TileN, int Depth>
__global__ __launch_bounds__(kThreads) void skinny_gemm_cpasync_kernel(
    const Element* __restrict__ x, const Element* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    float* __restrict__ workspace, int32_t* __restrict__ counters, int32_t m,
    int32_t n, int32_t k, int32_t k_tiles, int32_t k_split, bool coresident) {
  using Cfg = Config<TileN, Depth>;
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t n_tile = static_cast<int32_t>(blockIdx.x);
  const int32_t n_base = n_tile * TileN;
  int32_t k_begin, k_count;
  split_range(k_tiles, k_split, static_cast<int32_t>(blockIdx.y), &k_begin,
              &k_count);

  // The staged copies skip rows >= m, so the pad rows the wgmma still reads
  // must be zero before the first copy lands.
  {
    uint4* const zero = reinterpret_cast<uint4*>(sa_base);
    constexpr int kZeroVecs = Depth * Cfg::kFrameA * sizeof(Element) / sizeof(uint4);
    for (int i = tid; i < kZeroVecs; i += kThreads) {
      zero[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();

  Tensor gA_all = make_tensor(make_gmem_ptr(x), make_shape(Int<kTileM>{}, k),
                              make_stride(k, _1{}));
  Tensor gB_all = make_tensor(make_gmem_ptr(w + static_cast<int64_t>(n_base) * k),
                              make_shape(Int<TileN>{}, k), make_stride(k, _1{}));
  Tensor gA = local_tile(gA_all, Shape<Int<kTileM>, Int<kTileK>>{},
                         make_coord(_0{}, _));
  Tensor gB = local_tile(gB_all, Shape<Int<TileN>, Int<kTileK>>{},
                         make_coord(_0{}, _));

  auto copy_a = tile::make_g2s_cp_async_copy<Element, kThreads, kTileK>();
  auto thr_a = copy_a.get_thread_slice(tid);
  auto thr_b = copy_a.get_thread_slice(tid);
  Tensor tAgA = thr_a.partition_S(gA);
  Tensor tBgB = thr_b.partition_S(gB);

  Tensor cA = make_identity_tensor(Shape<Int<kTileM>, Int<kTileK>>{});
  Tensor cB = make_identity_tensor(Shape<Int<TileN>, Int<kTileK>>{});
  Tensor tAcA = thr_a.partition_S(cA);
  Tensor tBcB = thr_b.partition_S(cB);
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAcA), size<2>(tAcA)));
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBcB), size<2>(tBcB)));
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tApA); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tApA); ++j) {
      tApA(i, j) = get<0>(tAcA(_0{}, i, j)) < m;
    }
  }
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tBpB); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(tBpB); ++j) {
      tBpB(i, j) = n_base + get<0>(tBcB(_0{}, i, j)) < n;
    }
  }

  auto issue = [&](int32_t stage) {
    const int32_t slot = stage % Depth;
    Tensor sA = make_tensor(make_smem_ptr(sa_base + slot * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    copy_if(copy_a, tApA, tAgA(_, _, _, k_begin + stage), thr_a.partition_D(sA));
    copy_if(copy_a, tBpB, tBgB(_, _, _, k_begin + stage), thr_b.partition_D(sB));
  };

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < Depth - 1; ++stage) {
    if (stage < k_count) { issue(stage); }
    tile::cp_async_commit_group();
  }

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<TileN>>{});
  clear(acc);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < k_count; ++stage) {
    // Depth - 2 groups may still be pending, so this thread's copies for
    // `stage` have landed; the __syncthreads makes every other thread's
    // copies for the same stage visible to the wgmma descriptors.
    tile::cp_async_wait_group<Depth - 2>();
    __syncthreads();

    const int32_t slot = stage % Depth;
    Tensor sA = make_tensor(make_smem_ptr(sa_base + slot * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    // WgWait = 1 retires the batch of `stage - 1` and leaves this stage's in
    // flight; waiting the batch to empty costs 20-30% [wgmma.stages.wg.knee].
    tile::gemm_ss<false, 1>(tiled_mma, tid, sA, sB, acc);

    const int32_t next = stage + Depth - 1;
    if (next < k_count) { issue(next); }
    tile::cp_async_commit_group();
  }
  cute::warpgroup_wait<0>();

  Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<TileN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
#ifdef SKINNY_PROBE_NO_EPILOGUE
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
#else
  epilogue<TileN>(acc, tCcC, tid, n_tile, n_base,
                  static_cast<int32_t>(blockIdx.y), m, n, k_split, coresident,
                  bias, out, workspace, counters, smem_raw);
#endif
}

template <int TileN, int Depth>
__global__ __launch_bounds__(kThreads) void skinny_gemm_tma_kernel(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    float* __restrict__ workspace, int32_t* __restrict__ counters, int32_t m,
    int32_t n, int32_t k_tiles, int32_t k_split, bool coresident) {
  using Cfg = Config<TileN, Depth>;
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;
  uint64_t* const full =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t n_tile = static_cast<int32_t>(blockIdx.x);
  const int32_t n_base = n_tile * TileN;
  int32_t k_begin, k_count;
  split_range(k_tiles, k_split, static_cast<int32_t>(blockIdx.y), &k_begin,
              &k_count);
  // The A box is m rows, so the wgmma's pad rows are never written and have to
  // start at zero.
  {
    uint4* const zero = reinterpret_cast<uint4*>(sa_base);
    constexpr int kZeroVecs = Depth * Cfg::kFrameA * sizeof(Element) / sizeof(uint4);
    for (int i = tid; i < kZeroVecs; i += kThreads) {
      zero[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < Depth; ++s) {
      reinterpret_cast<tile::FullBarrier*>(&full[s])->init(1);
    }
  }
  tile::fence_barrier_init();
  __syncthreads();

  const uint32_t tx_bytes =
      static_cast<uint32_t>(m) * kTileK * sizeof(Element) + Cfg::kFrameBytesB;
  auto issue = [&](int32_t stage) {
    const int32_t slot = stage % Depth;
    auto* bar = reinterpret_cast<tile::FullBarrier*>(&full[slot]);
    bar->arrive_and_expect_tx(tx_bytes);
    const int32_t k_off = (k_begin + stage) * kTileK;
    tile::tma_load_2d(&map_a, sa_base + slot * Cfg::kFrameA, k_off, 0,
                      &full[slot]);
    // The weight is read once by one CTA, so it must not evict the activation
    // tile every other CTA is re-reading from L2.
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
        &map_b, sb_base + slot * Cfg::kFrameB, k_off, n_base, &full[slot]);
  };

  if (tid == 0) {
    CUTE_NO_UNROLL
    for (int32_t stage = 0; stage < Depth - 1 && stage < k_count; ++stage) {
      issue(stage);
    }
  }

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<TileN>>{});
  clear(acc);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < k_count; ++stage) {
    const int32_t slot = stage % Depth;
    // The barrier completes when both boxes of this stage have landed, which
    // is also what makes them visible to every thread: no CTA barrier here.
    reinterpret_cast<tile::FullBarrier*>(&full[slot])
        ->wait(static_cast<uint32_t>(stage / Depth) & 1u);

    Tensor sA = make_tensor(make_smem_ptr(sa_base + slot * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    tile::gemm_ss<false, 1>(tiled_mma, tid, sA, sB, acc);

    // Every thread has retired the batch of stage - 1, so its frame is dead
    // and its barrier can be re-armed; the __syncthreads is what makes that
    // true for the issuing thread rather than only for its own warp.
    __syncthreads();
    const int32_t next = stage + Depth - 1;
    if (tid == 0 && next < k_count) { issue(next); }
  }
  cute::warpgroup_wait<0>();

  Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<TileN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
#ifdef SKINNY_PROBE_NO_EPILOGUE
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
#else
  epilogue<TileN>(acc, tCcC, tid, n_tile, n_base,
                  static_cast<int32_t>(blockIdx.y), m, n, k_split, coresident,
                  bias, out, workspace, counters, smem_raw);
#endif
}

// Same mainloop as skinny_gemm_tma_kernel, but the k_split CTAs of one N tile
// are a cluster and reduce through distributed shared memory.  blockIdx.x is
// the split so a cluster is exactly one N tile's splits; blockIdx.y is the
// tile.  No workspace and no counters: the cluster barrier replaces both.
template <int TileN, int Depth, int Cluster>
__global__ __launch_bounds__(kThreads) __cluster_dims__(Cluster, 1, 1) void
skinny_gemm_tma_dsmem_kernel(const __grid_constant__ CUtensorMap map_a,
                             const __grid_constant__ CUtensorMap map_b,
                             const __nv_bfloat16* __restrict__ bias,
                             __nv_bfloat16* __restrict__ out, int32_t m,
                             int32_t n, int32_t k_tiles) {
  using Cfg = Config<TileN, Depth>;
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;
  uint64_t* const full =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t split = static_cast<int32_t>(blockIdx.x);
  const int32_t n_base = static_cast<int32_t>(blockIdx.y) * TileN;
  int32_t k_begin, k_count;
  split_range(k_tiles, Cluster, split, &k_begin, &k_count);

  {
    uint4* const zero = reinterpret_cast<uint4*>(sa_base);
    constexpr int kZeroVecs = Depth * Cfg::kFrameA * sizeof(Element) / sizeof(uint4);
    for (int i = tid; i < kZeroVecs; i += kThreads) {
      zero[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < Depth; ++s) {
      reinterpret_cast<tile::FullBarrier*>(&full[s])->init(1);
    }
  }
  tile::fence_barrier_init();
  __syncthreads();

  const uint32_t tx_bytes =
      static_cast<uint32_t>(m) * kTileK * sizeof(Element) + Cfg::kFrameBytesB;
  auto issue = [&](int32_t stage) {
    const int32_t slot = stage % Depth;
    reinterpret_cast<tile::FullBarrier*>(&full[slot])->arrive_and_expect_tx(tx_bytes);
    const int32_t k_off = (k_begin + stage) * kTileK;
    tile::tma_load_2d(&map_a, sa_base + slot * Cfg::kFrameA, k_off, 0,
                      &full[slot]);
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
        &map_b, sb_base + slot * Cfg::kFrameB, k_off, n_base, &full[slot]);
  };

  if (tid == 0) {
    CUTE_NO_UNROLL
    for (int32_t stage = 0; stage < Depth - 1 && stage < k_count; ++stage) {
      issue(stage);
    }
  }

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<TileN>>{});
  clear(acc);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < k_count; ++stage) {
    const int32_t slot = stage % Depth;
    reinterpret_cast<tile::FullBarrier*>(&full[slot])
        ->wait(static_cast<uint32_t>(stage / Depth) & 1u);

    Tensor sA = make_tensor(make_smem_ptr(sa_base + slot * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    tile::gemm_ss<false, 1>(tiled_mma, tid, sA, sB, acc);

    __syncthreads();
    const int32_t next = stage + Depth - 1;
    if (tid == 0 && next < k_count) { issue(next); }
  }
  cute::warpgroup_wait<0>();

  Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<TileN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
#ifdef SKINNY_PROBE_NO_EPILOGUE
  // Both cluster barriers stay: without them a CTA can retire while a sibling
  // still maps its shared memory, and the probe would not be the same kernel.
  cute::cluster_sync();
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
  cute::cluster_sync();
#else
  cluster_epilogue<TileN, Cluster>(acc, tCcC, tid, n_base, split, m, n, bias,
                                   out, smem_raw);
#endif
}

// Same mainloop, but pairs of N tiles form a cluster and the activation tile
// arrives once per pair by TMA multicast, issued by a dedicated producer warp.
//
// Every CTA re-reads the whole activation from L2, and at the tile widths that
// give enough CTAs that re-read is most of the traffic: packed qkv at
// tile_n = 32 moves 6.2 MB of activation against 3.9 MB of weight, and the
// measured aggregate ceiling of ~2.1 TB/s of load traffic makes that time.  A
// cluster of two halves it.  Only k_split == 1 takes this path.
//
// The producer warp is the whole point.  A first version had the multicast
// issued by the same thread that ran the mainloop, so rank 0 waited on a
// cross-CTA empty barrier on the consumer's critical path once per stage; that
// lost outright (job 614485: packed qkv 7.44 us against the plain TMA
// producer's 5.12, gate_up 7.94 against 5.91), which is the placement lesson
// in [unit-launch cluster.lat.sync].  Here warp 4 owns the ring and runs Depth
// stages ahead, so the cluster round trip is hidden by the ring rather than
// serialised with the wgmma.  Three barrier sets carry it: `full` is the TMA
// completion the math warps wait on, `empty` (rank 0's, arrived by both CTAs)
// tells the multicast that both frames are armed, and `consumed` (local) tells
// the producer that the math warps have retired a frame.
template <int TileN, int Depth>
__global__ __launch_bounds__(kThreads + tile::kWarpThreads)
    __cluster_dims__(2, 1, 1) void
    skinny_gemm_tma_mcast_kernel(const __grid_constant__ CUtensorMap map_a,
                                 const __grid_constant__ CUtensorMap map_b,
                                 const __nv_bfloat16* __restrict__ bias,
                                 __nv_bfloat16* __restrict__ out, int32_t m,
                                 int32_t n, int32_t k_tiles) {
  using Cfg = Config<TileN, Depth>;
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;
  uint64_t* const full =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);
  uint64_t* const empty = full + Depth;
  uint64_t* const consumed = empty + Depth;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t n_tile = static_cast<int32_t>(blockIdx.x);
  const int32_t n_base = n_tile * TileN;
  const uint32_t rank = cute::block_rank_in_cluster();

  {
    uint4* const zero = reinterpret_cast<uint4*>(sa_base);
    constexpr int kZeroVecs = Depth * Cfg::kFrameA * sizeof(Element) / sizeof(uint4);
    for (int i = tid; i < kZeroVecs; i += kThreads + tile::kWarpThreads) {
      zero[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < Depth; ++s) {
      reinterpret_cast<tile::FullBarrier*>(&full[s])->init(1);
      reinterpret_cast<tile::EmptyBarrier*>(&empty[s])->init(2);
      reinterpret_cast<tile::EmptyBarrier*>(&consumed[s])->init(1);
    }
  }
  tile::fence_barrier_init();
  __syncthreads();
  // The barriers have to exist in both CTAs before either issues a remote
  // arrive or a multicast; one cluster sync, not one per stage.
  cute::cluster_sync();

  if (tid >= kThreads) {
    const uint32_t tx_bytes =
        static_cast<uint32_t>(m) * kTileK * sizeof(Element) + Cfg::kFrameBytesB;
    if (tid == kThreads) {
      CUTE_NO_UNROLL
      for (int32_t stage = 0; stage < k_tiles; ++stage) {
        const int32_t slot = stage % Depth;
        const uint32_t turn = static_cast<uint32_t>(stage / Depth);
        if (stage >= Depth) {
          reinterpret_cast<tile::EmptyBarrier*>(&consumed[slot])
              ->wait((turn + 1u) & 1u);
        }
        reinterpret_cast<tile::FullBarrier*>(&full[slot])
            ->arrive_and_expect_tx(tx_bytes);
        // The weight box is this CTA's alone; the activation box is shared.
        tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
            &map_b, sb_base + slot * Cfg::kFrameB, stage * kTileK, n_base,
            &full[slot]);
        reinterpret_cast<tile::EmptyBarrier*>(&empty[slot])->arrive(0u, 1u);
        if (rank == 0) {
          reinterpret_cast<tile::EmptyBarrier*>(&empty[slot])->wait(turn & 1u);
          tile::tma_load_2d_multicast(&map_a, sa_base + slot * Cfg::kFrameA,
                                      stage * kTileK, 0, &full[slot], 0x3u);
        }
      }
    }
    cute::cluster_sync();
    return;
  }

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<TileN>>{});
  clear(acc);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < k_tiles; ++stage) {
    const int32_t slot = stage % Depth;
    reinterpret_cast<tile::FullBarrier*>(&full[slot])
        ->wait(static_cast<uint32_t>(stage / Depth) & 1u);

    Tensor sA = make_tensor(make_smem_ptr(sa_base + slot * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    tile::gemm_ss<false, 1>(tiled_mma, tid, sA, sB, acc);

    // Named barrier 1 covers the math warps only: __syncthreads would pull the
    // producer warp back into lockstep with them, which is what this kernel
    // exists to avoid.
    tile::named_barrier_sync(1, kThreads);
    if (stage >= 1 && tid == 0) {
      reinterpret_cast<tile::EmptyBarrier*>(&consumed[(stage - 1) % Depth])->arrive();
    }
  }
  cute::warpgroup_wait<0>();

  Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<TileN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
#ifdef SKINNY_PROBE_NO_EPILOGUE
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
#else
  epilogue<TileN>(acc, tCcC, tid, n_tile, n_base, 0, m, n, 1, false, bias, out,
                  nullptr, nullptr, smem_raw);
#endif
  // No CTA retires while a partner still maps its shared memory.
  cute::cluster_sync();
}

// Tensor maps are host objects and cuTensorMapEncodeTiled is a driver entry
// point, so they are built once per (tensor, tiling) and reused: a captured
// graph must see a hit, which the caller guarantees by running each argument
// set once before capture.
struct MapSlot {
  const void* base = nullptr;
  uint64_t inner = 0;
  uint64_t outer = 0;
  uint32_t box_inner = 0;
  uint32_t box_outer = 0;
  alignas(64) CUtensorMap map;
};
constexpr int kMapSlots = 4096;
MapSlot g_maps[kMapSlots];

const CUtensorMap* tensor_map(const void* base, uint64_t inner, uint64_t outer,
                              uint32_t box_inner, uint32_t box_outer) {
  // Allocations of one size come back on a regular stride, so the bucket has
  // to come from a full avalanche of the address: a shift-and-multiply keeps
  // the low bits of the stride and sends every replica to the same slot.
  uint64_t hash = reinterpret_cast<uint64_t>(base);
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdull;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53ull;
  hash ^= hash >> 33;
  hash += outer * 0x9E3779B97F4A7C15ull + box_outer;
  for (int probe = 0; probe < 64; ++probe) {
    MapSlot& slot = g_maps[(hash + probe) & (kMapSlots - 1)];
    if (slot.base == base && slot.inner == inner && slot.outer == outer &&
        slot.box_inner == box_inner && slot.box_outer == box_outer) {
      return &slot.map;
    }
    if (slot.base == nullptr) {
      if (tile::encode_tensor_map_2d<Element>(&slot.map, base, inner, outer,
                                              inner * sizeof(Element), box_inner,
                                              box_outer, 128) != CUDA_SUCCESS) {
        return nullptr;
      }
      slot.inner = inner;
      slot.outer = outer;
      slot.box_inner = box_inner;
      slot.box_outer = box_outer;
      slot.base = base;
      return &slot.map;
    }
  }
  return nullptr;
}

// True when a grid of `blocks` CTAs of this kernel is entirely resident, which
// is what makes the spin in the distributed reduction terminate.  Queried per
// kernel because the ring's shared memory is what caps the occupancy.
template <class Kernel>
bool all_blocks_resident(Kernel kernel, int smem_bytes, int blocks) {
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads,
                                                    smem_bytes) != cudaSuccess) {
    return false;
  }
  int device = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess) {
    return false;
  }
  return per_sm > 0 && blocks <= per_sm * sms;
}

template <class Kernel>
cudaError_t opt_in_smem(Kernel kernel, int bytes, bool* done) {
  if (*done) { return cudaSuccess; }
  const cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status == cudaSuccess) { *done = true; }
  return status;
}

template <int TileN, int Depth>
cudaError_t launch_cpasync(const void* x, const void* w, const void* bias,
                           void* out, void* workspace, void* counters,
                           int32_t m, int32_t n, int32_t k, int32_t k_tiles,
                           int32_t k_split, cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    static bool opted_in = false;
    auto kernel = skinny_gemm_cpasync_kernel<TileN, Depth>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const dim3 grid(static_cast<unsigned>((n + TileN - 1) / TileN),
                    static_cast<unsigned>(k_split));
    const bool coresident =
        k_split > 1 &&
        all_blocks_resident(kernel, Cfg::kSmemBytes, grid.x * grid.y);
    kernel<<<grid, kThreads, Cfg::kSmemBytes, stream>>>(
        reinterpret_cast<const Element*>(x), reinterpret_cast<const Element*>(w),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<float*>(workspace), reinterpret_cast<int32_t*>(counters),
        m, n, k, k_tiles, k_split, coresident);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth>
cudaError_t launch_tma(const void* x, const void* w, const void* bias, void* out,
                       void* workspace, void* counters, int32_t m, int32_t n,
                       int32_t k, int32_t k_tiles, int32_t k_split,
                       cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    static bool opted_in = false;
    auto kernel = skinny_gemm_tma_kernel<TileN, Depth>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    // The A box is exactly the real rows, so no coordinate leaves the tensor;
    // the B box may, and TMA fills those rows with zero.
    const CUtensorMap* map_a = tensor_map(x, k, m, kTileK, m);
    const CUtensorMap* map_b = tensor_map(w, k, n, kTileK, TileN);
    if (map_a == nullptr || map_b == nullptr) { return cudaErrorUnknown; }
    const dim3 grid(static_cast<unsigned>((n + TileN - 1) / TileN),
                    static_cast<unsigned>(k_split));
    const bool coresident =
        k_split > 1 &&
        all_blocks_resident(kernel, Cfg::kSmemBytes, grid.x * grid.y);
    kernel<<<grid, kThreads, Cfg::kSmemBytes, stream>>>(
        *map_a, *map_b, reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<float*>(workspace), reinterpret_cast<int32_t*>(counters),
        m, n, k_tiles, k_split, coresident);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth, int Cluster>
cudaError_t launch_dsmem(const void* x, const void* w, const void* bias,
                         void* out, int32_t m, int32_t n, int32_t k,
                         int32_t k_tiles, cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    static bool opted_in = false;
    auto kernel = skinny_gemm_tma_dsmem_kernel<TileN, Depth, Cluster>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const CUtensorMap* map_a = tensor_map(x, k, m, kTileK, m);
    const CUtensorMap* map_b = tensor_map(w, k, n, kTileK, TileN);
    if (map_a == nullptr || map_b == nullptr) { return cudaErrorUnknown; }
    const dim3 grid(static_cast<unsigned>(Cluster),
                    static_cast<unsigned>((n + TileN - 1) / TileN));
    kernel<<<grid, kThreads, Cfg::kSmemBytes, stream>>>(
        *map_a, *map_b, reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), m, n, k_tiles);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth>
cudaError_t launch_cluster(const void* x, const void* w, const void* bias,
                           void* out, int32_t m, int32_t n, int32_t k,
                           int32_t k_tiles, cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    const int32_t tiles = (n + TileN - 1) / TileN;
    // A cluster of two needs an even N tile count; an odd one has no partner
    // for its last tile, so the caller gets the plain TMA kernel instead.
    if ((tiles & 1) != 0) { return cudaErrorInvalidValue; }
    static bool opted_in = false;
    auto kernel = skinny_gemm_tma_mcast_kernel<TileN, Depth>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const CUtensorMap* map_a = tensor_map(x, k, m, kTileK, m);
    const CUtensorMap* map_b = tensor_map(w, k, n, kTileK, TileN);
    if (map_a == nullptr || map_b == nullptr) { return cudaErrorUnknown; }
    kernel<<<dim3(static_cast<unsigned>(tiles)),
             kThreads + tile::kWarpThreads, Cfg::kSmemBytes, stream>>>(*map_a, *map_b,
                       reinterpret_cast<const __nv_bfloat16*>(bias),
                       reinterpret_cast<__nv_bfloat16*>(out), m, n, k_tiles);
    return cudaPeekAtLastError();
  }
}

}  // namespace flash_vla::lingbot::skinny

namespace skinny = flash_vla::lingbot::skinny;

// Returns the tile count for a TileN so the caller can size the workspace and
// the arrival counters without repeating the tiling rule.
extern "C" int skinny_gemm_tiles(int32_t n, int32_t tile_n) {
  if (tile_n <= 0) { return -1; }
  return (n + tile_n - 1) / tile_n;
}

// fp32 elements the split-K partials need: one dense m x tile_n block per
// (N tile, split).  Partials are fully rewritten every launch, so the buffer
// needs no initialisation.
extern "C" long long skinny_gemm_workspace_floats(int32_t m, int32_t n,
                                                  int32_t tile_n,
                                                  int32_t k_split) {
  if (tile_n <= 0 || k_split <= 0) { return -1; }
  const long long tiles = (n + tile_n - 1) / tile_n;
  return tiles * k_split * static_cast<long long>(m) * tile_n;
}

// C = x @ w^T (+ bias), bf16 in, fp32 accumulate, bf16 out.
//
// `x` is [m, k] and `w` is [n, k], both bf16 and row-major contiguous on the
// current device; `out` is [m, n] bf16 and is written in full.  `bias` is
// [n] bf16 or null.  `tile_n` is 32, 64, 128 or 256, `depth` is the ring depth and
// `producer` is 0 for cp.async, 1 for TMA, or 2 for TMA with a cluster of two
// sharing the activation tile (k_split must be 1 and the N tile count even),
// or 3 for TMA with the k_split splits of an N tile forming a cluster that
// reduces through distributed shared memory (k_split must be 2, 4 or 8, and
// neither a workspace nor counters are used).  `k_split` > 1 needs `workspace`
// (skinny_gemm_workspace_floats() fp32 elements, uninitialised) and `counters`
// (two int32 per N tile, zero on entry; the kernel restores them, so a
// captured graph can replay without re-zeroing).
// One launch, capture safe once the TMA producer has seen these tensors once.
extern "C" int skinny_gemm_launch(const void* x, const void* w, const void* bias,
                                  void* out, void* workspace, void* counters,
                                  int32_t m, int32_t n, int32_t k,
                                  int32_t tile_n, int32_t depth,
                                  int32_t k_split, int32_t producer,
                                  void* stream) {
  if (m <= 0 || m > skinny::kTileM || n <= 0 || k <= 0 ||
      (k % skinny::kTileK) != 0) {
    return 1;
  }
  const int32_t k_tiles = k / skinny::kTileK;
  if (k_split < 1 || k_split > k_tiles) { return 2; }
  // The cluster reduction below takes neither, so this guard is for the
  // global-partial paths only; it used to reject producer 3 outright.
  if (k_split > 1 && producer != 3 &&
      (workspace == nullptr || counters == nullptr)) { return 3; }
  // The cluster producer multicasts one activation tile to a pair of N tiles;
  // it carries no split-K handshake.
  if (producer == 2 && k_split != 1) { return 5; }
  // The cluster reduction makes the splits of one N tile a cluster, so the
  // split count is the cluster size and only the portable sizes are compiled.
  if (producer == 3 && k_split != 2 && k_split != 4 && k_split != 8) { return 6; }
  auto s = static_cast<cudaStream_t>(stream);

#define FLASH_VLA_SKINNY_CASE(TILE_N_, DEPTH_)                                \
  if (tile_n == TILE_N_ && depth == DEPTH_) {                                 \
    if (producer == 2) {                                                      \
      return static_cast<int>(skinny::launch_cluster<TILE_N_, DEPTH_>(        \
          x, w, bias, out, m, n, k, k_tiles, s));                             \
    }                                                                         \
    if (producer == 3) {                                                      \
      if (k_split == 2) {                                                     \
        return static_cast<int>(skinny::launch_dsmem<TILE_N_, DEPTH_, 2>(     \
            x, w, bias, out, m, n, k, k_tiles, s));                           \
      }                                                                       \
      if (k_split == 4) {                                                     \
        return static_cast<int>(skinny::launch_dsmem<TILE_N_, DEPTH_, 4>(     \
            x, w, bias, out, m, n, k, k_tiles, s));                           \
      }                                                                       \
      return static_cast<int>(skinny::launch_dsmem<TILE_N_, DEPTH_, 8>(       \
          x, w, bias, out, m, n, k, k_tiles, s));                             \
    }                                                                         \
    return static_cast<int>(                                                  \
        producer == 0                                                         \
            ? skinny::launch_cpasync<TILE_N_, DEPTH_>(                        \
                  x, w, bias, out, workspace, counters, m, n, k, k_tiles,     \
                  k_split, s)                                                 \
            : skinny::launch_tma<TILE_N_, DEPTH_>(                            \
                  x, w, bias, out, workspace, counters, m, n, k, k_tiles,     \
                  k_split, s));                                               \
  }
  FLASH_VLA_SKINNY_CASE(32, 6)
  FLASH_VLA_SKINNY_CASE(32, 8)
  FLASH_VLA_SKINNY_CASE(32, 12)
  FLASH_VLA_SKINNY_CASE(64, 4)
  FLASH_VLA_SKINNY_CASE(64, 6)
  FLASH_VLA_SKINNY_CASE(64, 8)
  FLASH_VLA_SKINNY_CASE(64, 12)
  FLASH_VLA_SKINNY_CASE(128, 4)
  FLASH_VLA_SKINNY_CASE(128, 6)
  FLASH_VLA_SKINNY_CASE(128, 8)
  FLASH_VLA_SKINNY_CASE(256, 3)
  FLASH_VLA_SKINNY_CASE(256, 4)
#undef FLASH_VLA_SKINNY_CASE
  return 4;
}
