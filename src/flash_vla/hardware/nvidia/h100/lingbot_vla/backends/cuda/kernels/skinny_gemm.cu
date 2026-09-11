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
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out,
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
    const int64_t flat = static_cast<int64_t>(row) * n + col;
    CUTE_UNROLL
    for (int j = 0; j < 4; ++j) {
      if (col + j >= n) { break; }
      if (bias != nullptr) { value[j] += __bfloat162float(bias[col + j]); }
      // The unfused chain rounds the GEMM output to bf16 and only then adds the
      // residual in f32 (ada_rms_add_kernel's `source + residual`), so the fused
      // form has to round twice as well or the 360-step chain drifts.
      if (residual != nullptr) {
        value[j] = __bfloat162float(__float2bfloat16_rn(value[j])) +
                   __bfloat162float(residual[flat + j]);
      }
    }
    if (col + 3 < n) {
      reinterpret_cast<__nv_bfloat162*>(dst)[0] =
          __floats2bfloat162_rn(value[0], value[1]);
      reinterpret_cast<__nv_bfloat162*>(dst)[1] =
          __floats2bfloat162_rn(value[2], value[3]);
    } else {
      for (int j = 0; j < 4; ++j) {
        if (col + j >= n) { break; }
        dst[j] = __float2bfloat16_rn(value[j]);
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

// Reduction for a paired N tile: the cluster sums the partials as
// cluster_epilogue does, but a CTA's slice is of the OUTPUT, and each output
// element pulls its gate and its up column out of every sibling's stage.  Same
// bytes per CTA as the unpaired form -- m x TileN x 4 -- because halving the
// slice and doubling the columns per element cancel.
template <int TileN, int Cluster, class AccTensor, class CoordTensor>
__device__ __forceinline__ void cluster_silu_epilogue(
    const AccTensor& acc, const CoordTensor& tCcC, int32_t tid, int32_t n_base,
    int32_t split, int32_t m, int32_t half_n,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    void* smem_scratch) {
  constexpr int kPad = 4;
  constexpr int kStage = TileN + kPad;
  constexpr int kHalfTile = TileN / 2;
  float* const stage = reinterpret_cast<float*>(smem_scratch);
  __syncthreads();
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 2) {
    const int row = get<0>(tCcC(i));
    const int col = get<1>(tCcC(i));
    if (row >= m) { continue; }
    *reinterpret_cast<float2*>(stage + row * kStage + col) =
        make_float2(acc(i), acc(i + 1));
  }
  cute::cluster_sync();

  const int32_t vectors = m * kHalfTile / 4;
  const int32_t chunk = (vectors + Cluster - 1) / Cluster;
  const int32_t begin = split * chunk;
  const int32_t end = (begin + chunk) < vectors ? (begin + chunk) : vectors;
  const uint32_t base = tile::smem_u32(stage);
  for (int32_t v = begin + tid; v < end; v += kThreads) {
    const int32_t element = v * 4;
    const int32_t row = element / kHalfTile;
    const int32_t tile_col = element % kHalfTile;
    const uint32_t gate_at =
        base + static_cast<uint32_t>((row * kStage + tile_col) * sizeof(float));
    const uint32_t up_at = gate_at + kHalfTile * sizeof(float);
    float4 gate = *reinterpret_cast<const float4*>(stage + row * kStage + tile_col);
    float4 up = *reinterpret_cast<const float4*>(
        stage + row * kStage + tile_col + kHalfTile);
    CUTE_UNROLL
    for (int q = 1; q < Cluster; ++q) {
      const uint32_t rank = static_cast<uint32_t>((split + q) % Cluster);
      const float4 g = load_shared_cluster(map_shared_rank(gate_at, rank));
      const float4 u = load_shared_cluster(map_shared_rank(up_at, rank));
      gate.x += g.x; gate.y += g.y; gate.z += g.z; gate.w += g.w;
      up.x += u.x; up.y += u.y; up.z += u.z; up.w += u.w;
    }
    const float gv[4] = {gate.x, gate.y, gate.z, gate.w};
    const float uv[4] = {up.x, up.y, up.z, up.w};
    float value[4];
    const int32_t col = n_base + tile_col;
    CUTE_UNROLL
    for (int j = 0; j < 4; ++j) {
      float g = gv[j];
      float u = uv[j];
      if (bias != nullptr) {
        g += __bfloat162float(bias[col + j]);
        u += __bfloat162float(bias[half_n + col + j]);
      }
      // Both operands are bf16 by the time silu_multiply sees them unfused.
      g = __bfloat162float(__float2bfloat16(g));
      u = __bfloat162float(__float2bfloat16(u));
      value[j] = __bfloat162float(__float2bfloat16(g / (1.0f + expf(-g)))) * u;
    }
    __nv_bfloat16* dst = out + static_cast<int64_t>(row) * half_n + col;
    reinterpret_cast<__nv_bfloat162*>(dst)[0] =
        __floats2bfloat162_rn(value[0], value[1]);
    reinterpret_cast<__nv_bfloat162*>(dst)[1] =
        __floats2bfloat162_rn(value[2], value[3]);
  }
  cute::cluster_sync();
}

// Paired gate/up GEMM whose K axis is split across the CTAs of a cluster.
//
// The unsplit paired kernel is correct and bit-exact but only break-even (job
// 615592: 1.03x), and the reason is visible in the CTA count: pairing gate with
// up halves it, 172 -> 86, and this kernel lives on CTA count.  Splitting K by
// the cluster size puts it back -- 86 x 2 = 172 -- for the price of the
// distributed-shared reduction already measured at 0.04 - 0.56 us on the
// o_proj and down_proj shapes.
template <int TileN, int Depth, int Cluster>
__global__ __launch_bounds__(kThreads) __cluster_dims__(Cluster, 1, 1) void
skinny_gemm_silu_dsmem_kernel(const __grid_constant__ CUtensorMap map_a,
                              const __grid_constant__ CUtensorMap map_b,
                              const __nv_bfloat16* __restrict__ bias,
                              __nv_bfloat16* __restrict__ out, int32_t m,
                              int32_t half_n, int32_t k_tiles) {
  using Cfg = Config<TileN, Depth>;
  static_assert(TileN % 16 == 0, "each half of a paired N tile is a TMA box");
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;
  uint64_t* const full =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t split = static_cast<int32_t>(blockIdx.x);
  constexpr int32_t kHalfTile = TileN / 2;
  const int32_t n_base = static_cast<int32_t>(blockIdx.y) * kHalfTile;
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
    tile::tma_load_2d(&map_a, sa_base + slot * Cfg::kFrameA, k_off, 0, &full[slot]);
    Element* const frame = sb_base + slot * Cfg::kFrameB;
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(&map_b, frame, k_off, n_base,
                                                 &full[slot]);
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
        &map_b, frame + kHalfTile * kTileK, k_off, half_n + n_base, &full[slot]);
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
  cute::cluster_sync();
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
  cute::cluster_sync();
#else
  cluster_silu_epilogue<TileN, Cluster>(acc, tCcC, tid, n_base, split, m, half_n,
                                        bias, out, smem_raw);
#endif
}

// Packed gate/up GEMM with silu(gate) * up folded into the store.
//
// The wrinkle is which columns a CTA owns.  `silu_multiply` pairs output column
// c with packed columns c and c + 2752, so a CTA that owns a contiguous N tile
// holds one side of every pair and none of the other.  This kernel therefore
// builds its B tile from TWO weight row ranges -- [n0, n0 + TileN/2) and
// [half + n0, half + n0 + TileN/2) -- as two TMA boxes into one tile, so the
// wgmma is still a single TileN-wide instruction and the pair lands inside one
// thread's accumulator: the fragment's n-blocks are 8 columns apart, so gate at
// index i has its up at i + TileN/4, same lane, same row.
//
// Unlike the AdaRMS prologue this costs nothing structurally -- every output
// element is independent, so the work is partitioned across CTAs rather than
// replicated in each of them.
//
// Rounding follows silu_multiply_kernel: the activation is rounded to bf16
// before it multiplies `up`, which is a double rounding the fused form has to
// keep.
template <int TileN, int Depth>
__global__ __launch_bounds__(kThreads) void skinny_gemm_silu_kernel(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int32_t m, int32_t half_n, int32_t k_tiles) {
  using Cfg = Config<TileN, Depth>;
  static_assert(TileN % 16 == 0, "each half of a paired N tile is a TMA box");
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + Depth * Cfg::kFrameA;
  uint64_t* const full =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t n_tile = static_cast<int32_t>(blockIdx.x);
  constexpr int32_t kHalfTile = TileN / 2;
  const int32_t n_base = n_tile * kHalfTile;

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
    const int32_t k_off = stage * kTileK;
    tile::tma_load_2d(&map_a, sa_base + slot * Cfg::kFrameA, k_off, 0, &full[slot]);
    // The gate rows, then the up rows, into the two halves of one B tile.
    Element* const frame = sb_base + slot * Cfg::kFrameB;
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(&map_b, frame, k_off, n_base,
                                                 &full[slot]);
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
        &map_b, frame + kHalfTile * kTileK, k_off, half_n + n_base, &full[slot]);
  };

  if (tid == 0) {
    CUTE_NO_UNROLL
    for (int32_t stage = 0; stage < Depth - 1 && stage < k_tiles; ++stage) {
      issue(stage);
    }
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

    __syncthreads();
    const int32_t next = stage + Depth - 1;
    if (tid == 0 && next < k_tiles) { issue(next); }
  }
  cute::warpgroup_wait<0>();

  Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<TileN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
#ifdef SKINNY_PROBE_NO_EPILOGUE
  if (acc(0) == 1.0e30f) { out[0] = __float2bfloat16_rn(acc(1)); }
#else
  // The gate half of the fragment; its up partner is TileN/4 entries along.
  constexpr int kPairStride = TileN / 4;
  CUTE_UNROLL
  for (int i = 0; i < size(acc) / 2; i += 2) {
    const int row = get<0>(tCcC(i));
    const int tile_col = get<1>(tCcC(i));
    if (row >= m) { continue; }
    float value[2];
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      float gate = acc(i + j);
      float up = acc(i + j + kPairStride);
      if (bias != nullptr) {
        gate += __bfloat162float(bias[n_base + tile_col + j]);
        up += __bfloat162float(bias[half_n + n_base + tile_col + j]);
      }
      // Unfused, the GEMM stores bf16 and silu_multiply reads that back, so
      // both operands are already rounded before the activation sees them.
      // Feeding the fp32 accumulator straight in is more accurate and wrong:
      // it disagrees with the chain this replaces by up to six bf16 ulps.
      gate = __bfloat162float(__float2bfloat16(gate));
      up = __bfloat162float(__float2bfloat16(up));
      const float activated =
          __bfloat162float(__float2bfloat16(gate / (1.0f + expf(-gate))));
      value[j] = activated * up;
    }
    const int col = n_base + tile_col;
    __nv_bfloat16* dst = out + static_cast<int64_t>(row) * half_n + col;
    if (col + 1 < half_n) {
      *reinterpret_cast<__nv_bfloat162*>(dst) =
          __floats2bfloat162_rn(value[0], value[1]);
    } else if (col < half_n) {
      dst[0] = __float2bfloat16(value[0]);
    }
  }
#endif
}

// GEMM with the AdaRMS normalisation of its own activation folded in.
//
// `ada_rms_add` is 2.55 ms of the deployed expert (job 615421) and no tiling
// fixes it: it is a row-wise reduction over 51 rows, so it is 51 CTAs whatever
// the block size.  A weight-stationary GEMM already reads the whole 51 x K
// activation in every CTA, so the normalisation is arithmetic on a tile that is
// already resident and the launch disappears.
//
// The cost is that the whole activation has to be resident before the first
// wgmma, because rsqrt(mean(x^2)) needs all of K.  At K = 768 that is a
// 64 x 768 bf16 tile, 96 KB, which fits beside the weight ring -- and it is not
// extra traffic, only a different shape for the same bytes: the ring no longer
// carries the activation at all, so the weight gets all of its slots and the
// prologue's normalisation overlaps the ring fill.
//
// Rounding follows ada_rms_add_kernel in cuda/kernels/pointwise.cu instruction
// for instruction, because 360 layer-steps compound any drift.  The delicate
// part is the reduction ORDER: the reference is 256 threads per row, each
// accumulating a stride-256 subsequence with fadd_rn, then a shared-memory tree
// of strides 128..1.  Here the same 256 partials are the 8 registers of each of
// 32 lanes, the tree's first five strides are __shfl_down_sync by 16..1, and
// the last three are register adds -- same operands in the same order, no
// shared-memory array, one warp per row.
template <int TileN, int Depth, int KTiles>
struct FusedConfig {
  static constexpr int kK = KTiles * kTileK;
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
  static constexpr int kABytes = KTiles * kFrameA * static_cast<int>(sizeof(Element));
  static constexpr int kBBytes = Depth * kFrameB * static_cast<int>(sizeof(Element));
  // weight, gamma and beta, staged once instead of re-read per element.
  static constexpr int kNormBytes = 3 * kK * static_cast<int>(sizeof(Element));
  static constexpr int kScaleBytes = kTileM * 4;
  static constexpr int kBarrierBytes = (1 + Depth) * 8;
  static constexpr int kSmemBytes =
      kABytes + kBBytes + kNormBytes + kScaleBytes + kBarrierBytes;
  static constexpr bool kFits = kSmemBytes <= kMaxSmemBytes;
};

// Address of (row, col) in the resident activation: which K frame, then that
// frame's swizzle.  It goes through the same make_tensor the wgmma's
// partition_A does rather than calling the layout directly -- the swizzle in a
// GMMA atom is anchored to the tensor's pointer, so indexing the bare layout
// reads a permuted tile that is wrong without ever being out of bounds.
// col must be a multiple of 8; the swizzle keeps those 8 elements contiguous.
template <class LayoutA>
__device__ __forceinline__ int activation_offset(Element* sa_base, int row,
                                                 int col) {
  Element* const base = sa_base + (col / kTileK) * (kTileM * kTileK);
  Tensor frame = make_tensor(make_smem_ptr(base), LayoutA{});
  // Returned as an offset, not a pointer: CuTe's smem_ptr hands back a generic
  // address, and a load through one costs the generic path instead of
  // ld.shared.  Re-adding it to sa_base, which the compiler knows is the
  // extern __shared__ block, keeps every access in the shared window.
  return static_cast<int>(&frame(row, col % kTileK) - sa_base);
}

// Threads is the prologue's width, not the MMA's: the normalisation is
// instruction-bound and a 128-thread CTA gives each scheduler one warp, so
// every dependent instruction in it pays full latency.  The extra warps do the
// prologue and then retire; the mainloop is still one warpgroup, which is why
// its frame release is a named barrier over threads 0..127 rather than
// __syncthreads.
template <int TileN, int Depth, int KTiles, int Threads>
__global__ __launch_bounds__(Threads) void skinny_gemm_adarms_kernel(
    const __grid_constant__ CUtensorMap map_x,
    const __grid_constant__ CUtensorMap map_w,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ norm_gamma,
    const __nv_bfloat16* __restrict__ norm_beta,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ normed_out, int32_t m, int32_t n, float epsilon) {
  using Cfg = FusedConfig<TileN, Depth, KTiles>;
  extern __shared__ __align__(1024) char smem_raw[];
  Element* const sa_base = reinterpret_cast<Element*>(smem_raw);
  Element* const sb_base = sa_base + KTiles * Cfg::kFrameA;
  // Typed as the CUDA bf16 so the conversions below are the reference's
      // intrinsics rather than CUTLASS's conversion operators.
  __nv_bfloat16* const s_weight =
      reinterpret_cast<__nv_bfloat16*>(sb_base + Depth * Cfg::kFrameB);
  __nv_bfloat16* const s_gamma = s_weight + Cfg::kK;
  __nv_bfloat16* const s_beta = s_gamma + Cfg::kK;
  float* const s_scale = reinterpret_cast<float*>(s_beta + Cfg::kK);
  uint64_t* const full_a =
      reinterpret_cast<uint64_t*>(smem_raw + Cfg::kSmemBytes - Cfg::kBarrierBytes);
  uint64_t* const full_w = full_a + 1;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t n_tile = static_cast<int32_t>(blockIdx.x);
  const int32_t n_base = n_tile * TileN;

  // The activation box is m rows, so the wgmma's pad rows are never written by
  // the copy and never touched by the normalisation.
  {
    uint4* const zero = reinterpret_cast<uint4*>(sa_base);
    constexpr int kZeroVecs = KTiles * Cfg::kFrameA * sizeof(Element) / sizeof(uint4);
    for (int i = tid; i < kZeroVecs; i += Threads) {
      zero[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid == 0) {
    reinterpret_cast<tile::FullBarrier*>(full_a)->init(1);
    CUTE_UNROLL
    for (int s = 0; s < Depth; ++s) {
      reinterpret_cast<tile::FullBarrier*>(&full_w[s])->init(1);
    }
  }
  tile::fence_barrier_init();
  __syncthreads();

  auto issue_w = [&](int32_t stage) {
    const int32_t slot = stage % Depth;
    reinterpret_cast<tile::FullBarrier*>(&full_w[slot])
        ->arrive_and_expect_tx(Cfg::kFrameB * sizeof(Element));
    tile::tma_load_2d<tile::L2Hint::kEvictFirst>(
        &map_w, sb_base + slot * Cfg::kFrameB, stage * kTileK, n_base,
        &full_w[slot]);
  };

  if (tid == 0) {
    // The activation is one barrier over every K frame; the weight ring starts
    // filling in the same breath, so the normalisation below runs under it.
    reinterpret_cast<tile::FullBarrier*>(full_a)->arrive_and_expect_tx(
        static_cast<uint32_t>(m) * Cfg::kK * sizeof(Element));
    CUTE_UNROLL
    for (int j = 0; j < KTiles; ++j) {
      tile::tma_load_2d(&map_x, sa_base + j * Cfg::kFrameA, j * kTileK, 0, full_a);
    }
    CUTE_NO_UNROLL
    for (int32_t stage = 0; stage < Depth - 1 && stage < KTiles; ++stage) {
      issue_w(stage);
    }
  }
  for (int i = tid; i < Cfg::kK; i += Threads) {
    s_weight[i] = norm_weight[i];
    s_gamma[i] = norm_gamma[i];
    s_beta[i] = norm_beta[i];
  }
  reinterpret_cast<tile::FullBarrier*>(full_a)->wait(0u);

  // Sum of squares: one warp per row, the reference's 256 partials held as 8
  // registers in each of 32 lanes.  Lane L owns partials 8L..8L+7 so the eight
  // columns it needs are one 16-byte chunk -- three loads per row instead of
  // twenty-four, which measured faster than the mapping that trades them for
  // fewer shuffles (job 615548: 20.9 us against 13.1).
  {
    const int warp = tid / tile::kWarpThreads;
    const int lane = tid % tile::kWarpThreads;
    constexpr int kBlocks = (Cfg::kK + 255) / 256;
    constexpr int kWarps = Threads / tile::kWarpThreads;
    for (int row = warp; row < m; row += kWarps) {
      float part[8];
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j) { part[j] = 0.0f; }
      CUTE_UNROLL
      for (int b = 0; b < kBlocks; ++b) {
        const int col = b * 256 + lane * 8;
        if (col >= Cfg::kK) { break; }
        const uint4 raw = *reinterpret_cast<const uint4*>(
            sa_base + activation_offset<typename Cfg::SmemA>(sa_base, row, col));
        const __nv_bfloat16* value = reinterpret_cast<const __nv_bfloat16*>(&raw);
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          const float x = __bfloat162float(value[j]);
          part[j] = __fadd_rn(part[j], __fmul_rn(x, x));
        }
      }
      // Tree strides 128, 64, 32, 16 and 8: partial t + 8*delta is in lane
      // (lane + delta), so these five shuffles are the reference's five steps.
      CUTE_UNROLL
      for (int delta = 16; delta >= 1; delta >>= 1) {
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          part[j] += __shfl_down_sync(0xffffffffu, part[j], delta);
        }
      }
      if (lane == 0) {
        // Strides 4, 2 and 1 are now inside one lane's registers.
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) { part[j] += part[j + 4]; }
        part[0] += part[2];
        part[1] += part[3];
        part[0] += part[1];
        s_scale[row] = rsqrtf(part[0] / (float)Cfg::kK + epsilon);
      }
    }
  }
  __syncthreads();

  // Normalise in place; the wgmma then reads the tile it would have read
  // anyway.  Consecutive threads take consecutive ROWS of one K chunk: the
  // swizzle makes eight such rows conflict-free, and a warp then shares one
  // chunk's weight/gamma/beta, which the hardware broadcasts.
  {
    constexpr int kVectors = Cfg::kK / 8;
    for (int idx = tid; idx < kTileM * kVectors; idx += Threads) {
      const int row = idx & (kTileM - 1);
      const int col = (idx / kTileM) * 8;
      if (row >= m) { continue; }
      const uint4 raw_w = *reinterpret_cast<const uint4*>(s_weight + col);
      const uint4 raw_g = *reinterpret_cast<const uint4*>(s_gamma + col);
      const uint4 raw_b = *reinterpret_cast<const uint4*>(s_beta + col);
      const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&raw_w);
      const __nv_bfloat16* gv = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
      const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&raw_b);
      Element* const at =
          sa_base + activation_offset<typename Cfg::SmemA>(sa_base, row, col);
      uint4 raw = *reinterpret_cast<const uint4*>(at);
      __nv_bfloat16* value = reinterpret_cast<__nv_bfloat16*>(&raw);
      const float scale = s_scale[row];
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j) {
        const float normalized =
            __bfloat162float(value[j]) * scale * __bfloat162float(wv[j]);
        value[j] = __float2bfloat16(
            (1.0f + __bfloat162float(gv[j])) * normalized + __bfloat162float(bv[j]));
      }
      *reinterpret_cast<uint4*>(at) = raw;
    }
  }
  // The normalisation wrote the tile with ordinary shared stores; wgmma reads
  // it through the async proxy, which does not see them without this fence.
  tile::fence_proxy_async_shared();
  __syncthreads();

  if (normed_out != nullptr) {
    // Prologue-only path: one N tile dumps the normalised activation so the
    // arithmetic can be compared against ada_rms_add_kernel on its own.
    if (n_tile == 0) {
      constexpr int kVectors = Cfg::kK / 8;
      for (int idx = tid; idx < m * kVectors; idx += Threads) {
        const int row = idx / kVectors;
        const int col = (idx - row * kVectors) * 8;
        *reinterpret_cast<uint4*>(normed_out + (int64_t)row * Cfg::kK + col) =
            *reinterpret_cast<const uint4*>(
                sa_base + activation_offset<typename Cfg::SmemA>(sa_base, row, col));
      }
    }
    return;
  }
  // The prologue's extra warps are done; the MMA is one warpgroup.
  if (tid >= kThreads) { return; }

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<TileN>>{});
  clear(acc);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  CUTE_NO_UNROLL
  for (int32_t stage = 0; stage < KTiles; ++stage) {
    const int32_t slot = stage % Depth;
    reinterpret_cast<tile::FullBarrier*>(&full_w[slot])
        ->wait(static_cast<uint32_t>(stage / Depth) & 1u);

    Tensor sA = make_tensor(make_smem_ptr(sa_base + stage * Cfg::kFrameA),
                            typename Cfg::SmemA{});
    Tensor sB = make_tensor(make_smem_ptr(sb_base + slot * Cfg::kFrameB),
                            typename Cfg::SmemB{});
    tile::gemm_ss<false, 1>(tiled_mma, tid, sA, sB, acc);

    // Named barrier 1 covers the warpgroup only; the prologue's extra warps
    // have already retired and __syncthreads would be undefined with them gone.
    tile::named_barrier_sync(1, kThreads);
    const int32_t next = stage + Depth - 1;
    if (tid == 0 && next < KTiles) { issue_w(next); }
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
                             const __nv_bfloat16* __restrict__ residual,
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
                                   residual, out, smem_raw);
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
  // The prologue's extra warps are done; the MMA is one warpgroup.
  if (tid >= kThreads) { return; }

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
cudaError_t launch_silu_dsmem(const void* x, const void* w, const void* bias,
                              void* out, int32_t m, int32_t n, int32_t k,
                              int32_t k_tiles, cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    const int32_t half_n = n / 2;
    if ((n & 1) != 0 || (half_n % (TileN / 2)) != 0) { return cudaErrorInvalidValue; }
    if (k_tiles < Cluster) { return cudaErrorInvalidValue; }
    static bool opted_in = false;
    auto kernel = skinny_gemm_silu_dsmem_kernel<TileN, Depth, Cluster>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const CUtensorMap* map_a = tensor_map(x, k, m, kTileK, m);
    const CUtensorMap* map_b = tensor_map(w, k, n, kTileK, TileN / 2);
    if (map_a == nullptr || map_b == nullptr) { return cudaErrorUnknown; }
    const dim3 grid(static_cast<unsigned>(Cluster),
                    static_cast<unsigned>(half_n / (TileN / 2)));
    kernel<<<grid, kThreads, Cfg::kSmemBytes, stream>>>(
        *map_a, *map_b, reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), m, half_n, k_tiles);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth>
cudaError_t launch_silu(const void* x, const void* w, const void* bias,
                        void* out, int32_t m, int32_t n, int32_t k,
                        int32_t k_tiles, cudaStream_t stream) {
  using Cfg = Config<TileN, Depth>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    const int32_t half_n = n / 2;
    if ((n & 1) != 0 || (half_n % (TileN / 2)) != 0) { return cudaErrorInvalidValue; }
    static bool opted_in = false;
    auto kernel = skinny_gemm_silu_kernel<TileN, Depth>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const CUtensorMap* map_a = tensor_map(x, k, m, kTileK, m);
    // Half-tile boxes, because the two halves come from different weight rows.
    const CUtensorMap* map_b = tensor_map(w, k, n, kTileK, TileN / 2);
    if (map_a == nullptr || map_b == nullptr) { return cudaErrorUnknown; }
    kernel<<<dim3(static_cast<unsigned>(half_n / (TileN / 2))), kThreads,
             Cfg::kSmemBytes, stream>>>(
        *map_a, *map_b, reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), m, half_n, k_tiles);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth, int KTiles, int Threads>
cudaError_t launch_adarms(const void* x, const void* w, const void* norm_weight,
                          const void* gamma, const void* beta, const void* bias,
                          void* out, void* normed_out, int32_t m, int32_t n,
                          int32_t k, float epsilon, cudaStream_t stream) {
  using Cfg = FusedConfig<TileN, Depth, KTiles>;
  if constexpr (!Cfg::kFits) {
    return cudaErrorInvalidValue;
  } else {
    if (k != Cfg::kK) { return cudaErrorInvalidValue; }
    static bool opted_in = false;
    auto kernel = skinny_gemm_adarms_kernel<TileN, Depth, KTiles, Threads>;
    const cudaError_t status = opt_in_smem(kernel, Cfg::kSmemBytes, &opted_in);
    if (status != cudaSuccess) { return status; }
    const CUtensorMap* map_x = tensor_map(x, k, m, kTileK, m);
    const CUtensorMap* map_w = tensor_map(w, k, n, kTileK, TileN);
    if (map_x == nullptr || map_w == nullptr) { return cudaErrorUnknown; }
    kernel<<<dim3(static_cast<unsigned>((n + TileN - 1) / TileN)), Threads,
             Cfg::kSmemBytes, stream>>>(
        *map_x, *map_w, reinterpret_cast<const __nv_bfloat16*>(norm_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<__nv_bfloat16*>(normed_out), m, n, epsilon);
    return cudaPeekAtLastError();
  }
}

template <int TileN, int Depth, int Cluster>
cudaError_t launch_dsmem(const void* x, const void* w, const void* bias,
                         const void* residual, void* out, int32_t m, int32_t n,
                         int32_t k, int32_t k_tiles, cudaStream_t stream) {
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
        reinterpret_cast<const __nv_bfloat16*>(residual),
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
static int skinny_gemm_dispatch(const void* x, const void* w, const void* bias,
                                const void* residual, void* out,
                                void* workspace, void* counters, int32_t m,
                                int32_t n, int32_t k, int32_t tile_n,
                                int32_t depth, int32_t k_split,
                                int32_t producer, void* stream) {
  if (m <= 0 || m > skinny::kTileM || n <= 0 || k <= 0 ||
      (k % skinny::kTileK) != 0) {
    return 1;
  }
  const int32_t k_tiles = k / skinny::kTileK;
  if (k_split < 1 || k_split > k_tiles) { return 2; }
  // The cluster reduction keeps its partials in shared memory, so it is the one
  // split-K path that needs neither buffer.
  if (k_split > 1 && producer != 3 &&
      (workspace == nullptr || counters == nullptr)) {
    return 3;
  }
  // The cluster producer multicasts one activation tile to a pair of N tiles;
  // it carries no split-K handshake.
  if (producer == 2 && k_split != 1) { return 5; }
  // The cluster reduction makes the splits of one N tile a cluster, so the
  // split count is the cluster size and only the portable sizes are compiled.
  if (producer == 3 && k_split != 2 && k_split != 4 && k_split != 8) { return 6; }
  // Only the cluster reduction carries the residual epilogue so far; the other
  // producers would silently drop it.
  if (residual != nullptr && producer != 3) { return 7; }
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
            x, w, bias, residual, out, m, n, k, k_tiles, s));                 \
      }                                                                       \
      if (k_split == 4) {                                                     \
        return static_cast<int>(skinny::launch_dsmem<TILE_N_, DEPTH_, 4>(     \
            x, w, bias, residual, out, m, n, k, k_tiles, s));                 \
      }                                                                       \
      return static_cast<int>(skinny::launch_dsmem<TILE_N_, DEPTH_, 8>(       \
          x, w, bias, residual, out, m, n, k, k_tiles, s));                   \
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

extern "C" int skinny_gemm_launch(const void* x, const void* w, const void* bias,
                                  void* out, void* workspace, void* counters,
                                  int32_t m, int32_t n, int32_t k,
                                  int32_t tile_n, int32_t depth,
                                  int32_t k_split, int32_t producer,
                                  void* stream) {
  return skinny_gemm_dispatch(x, w, bias, nullptr, out, workspace, counters, m,
                              n, k, tile_n, depth, k_split, producer, stream);
}

// As skinny_gemm_launch, plus `out = ... + residual` folded into the store.
//
// `residual` is [m, n] bf16 or null and is read, never written; `out` may alias
// it.  The GEMM result is rounded to bf16 before the add, which is what the
// unfused pair did, so the fused and unfused chains agree bit for bit.  Only
// producer 3 carries it today.
extern "C" int skinny_gemm_residual_launch(
    const void* x, const void* w, const void* bias, const void* residual,
    void* out, void* workspace, void* counters, int32_t m, int32_t n, int32_t k,
    int32_t tile_n, int32_t depth, int32_t k_split, int32_t producer,
    void* stream) {
  return skinny_gemm_dispatch(x, w, bias, residual, out, workspace, counters, m,
                              n, k, tile_n, depth, k_split, producer, stream);
}

// C = adarms(x) @ w^T (+ bias), with the normalisation of x folded into the
// GEMM's own activation tile.
//
// `x` is [m, k] bf16 and is the already-summed activation -- the residual add
// belongs to the producing GEMM's epilogue (skinny_gemm_residual_launch), which
// is what makes the pre-norm sum a value the chain already has rather than one
// this kernel would have to publish redundantly from every CTA.  `norm_weight`,
// `gamma` and `beta` are [k] bf16; `epsilon` matches the reference.  The
// arithmetic is ada_rms_add_kernel's, reduction order included.  `tile_n` is 32
// or 64, `depth` is the weight ring's; `k` must be 768.  One launch, capture
// safe once this `x`/`w` pair has been seen once outside capture.  Passing
// `normed_out` ([m, k] bf16) runs the prologue only and dumps the normalised
// activation there instead of doing the GEMM, which is how the fused
// arithmetic is checked against ada_rms_add_kernel on its own.
extern "C" int skinny_gemm_adarms_launch(
    const void* x, const void* w, const void* norm_weight, const void* gamma,
    const void* beta, const void* bias, void* out, void* normed_out, int32_t m,
    int32_t n, int32_t k, int32_t tile_n, int32_t depth, int32_t threads,
    float epsilon, void* stream) {
  if (m <= 0 || m > skinny::kTileM || n <= 0 || k != 768) { return 1; }
  if (norm_weight == nullptr || gamma == nullptr || beta == nullptr) { return 2; }
  auto s = static_cast<cudaStream_t>(stream);
#define FLASH_VLA_SKINNY_ADARMS_CASE(TILE_N_, DEPTH_, THREADS_)                \
  if (tile_n == TILE_N_ && depth == DEPTH_ && threads == THREADS_) {           \
    return static_cast<int>(                                                   \
        skinny::launch_adarms<TILE_N_, DEPTH_, 12, THREADS_>(                  \
            x, w, norm_weight, gamma, beta, bias, out, normed_out, m, n, k,    \
            epsilon, s));                                                      \
  }
  FLASH_VLA_SKINNY_ADARMS_CASE(32, 6, 128)
  FLASH_VLA_SKINNY_ADARMS_CASE(32, 6, 256)
  FLASH_VLA_SKINNY_ADARMS_CASE(32, 6, 512)
  FLASH_VLA_SKINNY_ADARMS_CASE(32, 8, 256)
  FLASH_VLA_SKINNY_ADARMS_CASE(32, 8, 512)
  FLASH_VLA_SKINNY_ADARMS_CASE(64, 6, 128)
  FLASH_VLA_SKINNY_ADARMS_CASE(64, 6, 256)
  FLASH_VLA_SKINNY_ADARMS_CASE(64, 6, 512)
  FLASH_VLA_SKINNY_ADARMS_CASE(64, 8, 256)
  FLASH_VLA_SKINNY_ADARMS_CASE(64, 8, 512)
#undef FLASH_VLA_SKINNY_ADARMS_CASE
  return 4;
}

// C = silu(gate) * up, where [gate | up] = x @ w^T (+ bias).
//
// `x` is [m, k] bf16, `w` is [n, k] with n even -- the packed gate/up weight --
// and `out` is [m, n/2] bf16, written in full.  `bias` is [n] bf16 or null and
// applies to both halves before the activation, which is where it would have
// landed unfused.  The activation is rounded to bf16 before it multiplies `up`,
// as silu_multiply_kernel does.  `tile_n` is the paired tile: a CTA owns
// tile_n/2 gate columns and their tile_n/2 up partners; `k_split` is 1, or 2 or
// 4 to split K across a cluster that reduces through distributed shared memory,
// which is what puts back the CTAs the pairing costs.  One launch, capture safe
// once this `x`/`w` pair has been seen once outside capture.
extern "C" int skinny_gemm_silu_launch(const void* x, const void* w,
                                       const void* bias, void* out, int32_t m,
                                       int32_t n, int32_t k, int32_t tile_n,
                                       int32_t depth, int32_t k_split,
                                       void* stream) {
  if (m <= 0 || m > skinny::kTileM || n <= 0 || k <= 0 ||
      (k % skinny::kTileK) != 0) {
    return 1;
  }
  const int32_t k_tiles = k / skinny::kTileK;
  auto s = static_cast<cudaStream_t>(stream);
#define FLASH_VLA_SKINNY_SILU_CASE(TILE_N_, DEPTH_)                            \
  if (tile_n == TILE_N_ && depth == DEPTH_) {                                  \
    if (k_split == 2) {                                                        \
      return static_cast<int>(skinny::launch_silu_dsmem<TILE_N_, DEPTH_, 2>(   \
          x, w, bias, out, m, n, k, k_tiles, s));                              \
    }                                                                          \
    if (k_split == 4) {                                                        \
      return static_cast<int>(skinny::launch_silu_dsmem<TILE_N_, DEPTH_, 4>(   \
          x, w, bias, out, m, n, k, k_tiles, s));                              \
    }                                                                          \
    return static_cast<int>(skinny::launch_silu<TILE_N_, DEPTH_>(              \
        x, w, bias, out, m, n, k, k_tiles, s));                                \
  }
  FLASH_VLA_SKINNY_SILU_CASE(32, 6)
  FLASH_VLA_SKINNY_SILU_CASE(32, 8)
  FLASH_VLA_SKINNY_SILU_CASE(64, 4)
  FLASH_VLA_SKINNY_SILU_CASE(64, 6)
  FLASH_VLA_SKINNY_SILU_CASE(64, 8)
  FLASH_VLA_SKINNY_SILU_CASE(64, 12)
  FLASH_VLA_SKINNY_SILU_CASE(128, 4)
  FLASH_VLA_SKINNY_SILU_CASE(128, 6)
#undef FLASH_VLA_SKINNY_SILU_CASE
  return 4;
}
