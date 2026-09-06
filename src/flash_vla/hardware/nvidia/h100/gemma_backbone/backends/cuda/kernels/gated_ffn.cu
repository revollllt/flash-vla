// gated_ffn.cu -- the Gemma backbone's RMSNorm + gated feed-forward, one launch
// per projection pair, persistent and warp-specialized.
//
//   out[m, n] = gelu_tanh(xn[m, :] . gate_w[:, n]) * (xn[m, :] . up_w[:, n])
//   xn        = x * rsqrt(mean_K(x^2) + 1e-6), rounded to bf16
//
// This call site is 56.6% of the Pi0.5 backbone's in-graph kernel time (job
// 599808) and the largest kernel in either Target.  The incumbent TileLang
// body reaches 207.8 us/call against a 152.85 us wgmma-clock ceiling
// [wgmma.clock.sm], i.e. 74% of the tensor core.
//
// THE THESIS, stated so it can be falsified.  Two properties of the incumbent
// are removable and neither needs a different tile:
//
//   1. Box count.  A row-major (rows x K) tile loaded as 2-D boxes needs K/64
//      transactions under SW128 [tma.bytes.txn.max]; as ONE 3-D box
//      {64, rows, K/64} it needs one [tma-3d-box-row-major].  The copy column
//      is transactions x 248 ns per SM [tma.issue.warp], and the rejected
//      short-K GEMM note measured that cost as per-SM, not per-warp: it does
//      not parallelize across producer warps.  At this tile 3 boxes per K-step
//      is 92 us of copy against a 153 us math column; 6 boxes is 185 us and
//      the kernel is copy-bound instead.
//   2. Wave quantization -- absent here, and that is why this shape is worth
//      attacking where the N=2048 sites are not.  N=16384 gives 128 N-tiles,
//      so 1024 tiles on Pi0.5 and 768 on Pi0 over 132 SMs: 7.76 and 5.82
//      waves, a 3% tail either way.  The rejected note's "one wave is worth
//      more than every other tiling property" bites at 96 or 128 tiles, not
//      here, which is exactly why it named persistent multi-tile CTAs as the
//      untried successor for tile counts above 132.
//
// GEOMETRY, and why each number is forced.
//
//   BN = 128.  Halving it doubles the tile count without halving the per-tile
//   box count (A is re-read at half the useful width), which puts the copy
//   column at 185 us.  BN >= 64 is also the wgmma smem-bandwidth floor
//   [wgmma-tile-n-floor].
//   BK = 128.  One 32 KB 3-D box per operand per K-step, the descriptor
//   maximum [tma.bytes.txn.max].  BK = 64 halves the box and doubles the
//   count, back to 185 us.
//   BM = 128 with TWO math warpgroups.  Two live f32 accumulators of
//   BM x BN = 16384 floats each; over one warpgroup that is 256 registers per
//   thread, past the 255 limit, so M is split across two warpgroups (64 rows
//   each, 64 + 64 = 128 accumulator registers).  The second warpgroup buys no
//   tensor-core throughput [wgmma.ratio.sm.wg2]; it is here to own half the M
//   tile, as in kernel-design template 10.
//   STAGES = 2.  A, gate and up frames are 32 KB each, so three rings of two
//   plus a 32 KB C staging tile is 224 KB of the 227 KB dynamic maximum.  This
//   is BELOW the 4-stage knee for a cold-fed mainloop
//   [pipeline.stages.wg.knee] and is the one property this kernel does NOT
//   improve over the incumbent; the gain, if any, is the box count.
//
// The RMSNorm is a separate launch, not fused.  Folding it needs whole-row
// statistics over K=2048 (64 rows is 256 KB, past smem) and the normalized
// activation must be rounded to bf16 BEFORE the projection -- the owning Agent
// Note records that order as not interchangeable with a folded per-row factor.
// The separate pass already runs at ~83% of [ld.bw.dev.dram], so the whole
// prize would be one launch ramp [launch.lat.dev.ramp].
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include
//        -I<repo>/src/flash_vla/hardware/nvidia/cuda gated_ffn.cu -lcuda

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include "tile/sm90/sm90.cuh"
#include "tile/sm90/tma_host.cuh"

namespace gated_ffn {

using namespace cute;
using BF = cutlass::bfloat16_t;
namespace tile = flash_vla::sm90;

// ------------------------------------------------------------- geometry
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int STAGES = 2;

constexpr int kMathWarpgroups = 2;
constexpr int kMathThreads = kMathWarpgroups * 128;   // 256
constexpr int kMathWarps = kMathThreads / 32;         // 8
constexpr int kProdThreads = 128;                     // one warpgroup, warp 0 issues
constexpr int kThreads = kMathThreads + kProdThreads; // 384
// setmaxnreg is warpgroup-wide and the CTA's total must fit 65536 registers:
// 256 * 224 + 128 * 48 = 63488.  A lone producer warp cannot hand its
// registers to the math side, so three of its warps idle after the handoff --
// the same trade enc_attn.cu records.
constexpr int kMathRegs = 224;
constexpr int kProdRegs = 48;

constexpr int kSwizzle = 128;
constexpr int kChunk = 64;                            // elements per SW128 span, bf16
constexpr int kFrameBytes = BM * BK * static_cast<int>(sizeof(BF));   // 32768, all three
constexpr int kStageBytes = 3 * kFrameBytes;          // one arm covers A, gate, up

// Ablation switches.  Each removes one column so the kernel's own decomposition
// can be read the way the rejected short-K GEMM note read its candidate's.
#ifndef GU_ABL_NO_EPILOGUE
#define GU_ABL_NO_EPILOGUE 0    // 1: keep the mainloop, drop activation + store
#endif
#ifndef GU_ABL_NO_MMA
#define GU_ABL_NO_MMA 0         // 1: keep the copy ring, drop the wgmma batches
#endif
#ifndef GU_BOX_2D
#define GU_BOX_2D 0             // 1: load each tile as BK/64 two-dimensional boxes
#endif
// Epilogue form. MEASURED (job 599861, ACD1-58, isolated cudagraph timer, the
// Pi0.5 shape): staging the result through a shared tile and storing it with
// TMA costs 171 us per call on top of a 160 us mainloop -- 52% of the kernel,
// against a mainloop already at 96% of the [wgmma.clock.sm] floor. The staged
// store serializes a TMA round trip and two named barriers per tile behind a
// single C buffer that the next tile's epilogue has to wait for, and a
// persistent kernel pays that 7.76 times per CTA instead of once. Storing the
// accumulator straight to global as bf16 pairs is what `ffn_taskloop.cu` does
// for the same reason; the vision and encoder GEMM notes' "stage the C tile"
// finding is about a NON-persistent kernel that pays it once.
#ifndef GU_EPI_TMA
#define GU_EPI_TMA 0            // 1: the staged shared tile + TMA store
#endif
#ifndef GU_ABL_NO_GELU
#define GU_ABL_NO_GELU 0        // 1: store the gate accumulator raw, no activation
#endif

// A is (M, K) row-major, K contiguous -> K-major, boxes walk K (default order).
using SmemA = typename tile::SmemTileLayout<BF, BM, BK, tile::Major::K, kSwizzle>::type;
// B is (K, N) row-major, N contiguous.  As a wgmma B operand its shape is
// (N, K) and N is the contiguous MN mode -> MN-major, and the 3-D box places
// its chunks along N, so the tile walks K first: Step<_2,_1>, the same pairing
// enc_attn.cu uses for its MN-major V frame.
using SmemB = typename tile::SmemTileLayout<BF, BN, BK, tile::Major::MN, kSwizzle,
                                            Step<_2, _1>>::type;
// C is (M, F) row-major, F contiguous.
using SmemC = typename tile::SmemTileLayout<BF, BM, BN, tile::Major::K, kSwizzle>::type;

using Sel = tile::MmaSelector<BF, BF, BM, BN, BK,
                              tile::Operand::kSmem, tile::Operand::kSmem,
                              tile::Major::K, tile::Major::MN, kMathThreads, 1>;
static_assert(Sel::kUseWgmma, "the gated FFN mainloop must land on wgmma");
static_assert(Sel::kWarpsM == kMathWarpgroups, "warpgroups stack along M");

// Two-dimensional fallback for the box-count ablation.
using TileA2D = tile::TmaTile2D<BF, BK, BM, kSwizzle>;   // inner = K (contiguous)
using TileB2D = tile::TmaTile2D<BF, BN, BK, kSwizzle>;   // inner = N (contiguous)
static_assert(TileA2D::kBytes == kFrameBytes && TileB2D::kBytes == kFrameBytes, "frame size");

// ------------------------------------------------------------- shared pool
constexpr int kOffA = 0;
constexpr int kOffG = kOffA + STAGES * kFrameBytes;
constexpr int kOffU = kOffG + STAGES * kFrameBytes;
constexpr int kOffC = kOffU + STAGES * kFrameBytes;
// The C staging tile exists only for the TMA-store epilogue; the direct-store
// form gives its 32 KB back, which is why that form is the default.
constexpr int kCBytes = GU_EPI_TMA ? BM * BN * static_cast<int>(sizeof(BF)) : 0;
constexpr int kOffBar = kOffC + kCBytes;
constexpr int kSmemBytes = kOffBar + 2 * STAGES * static_cast<int>(sizeof(uint64_t));
static_assert(kSmemBytes <= 232448, "over the H100 dynamic shared-memory maximum");

//: 2 * sqrt(2/pi); the tanh GELU in its sigmoid form, as the reference spells it.
__device__ __forceinline__ float gelu_tanh(float v) {
  return v * (1.0f / (1.0f + __expf(-(1.5957691216057308f * v * (1.0f + 0.044715f * v * v)))));
}

struct Params {
  const CUtensorMap* tm_a;
  const CUtensorMap* tm_g;
  const CUtensorMap* tm_u;
  const CUtensorMap* tm_c;
  int m_tiles;     // ceil(M / BM)
  int n_tiles;     // F / BN
  int k_steps;     // K / BK
};

__global__ void __launch_bounds__(kThreads, 1)
gated_ffn_kernel(const __grid_constant__ CUtensorMap tm_a,
                 const __grid_constant__ CUtensorMap tm_g,
                 const __grid_constant__ CUtensorMap tm_u,
                 const __grid_constant__ CUtensorMap tm_c,
                 BF* __restrict__ out, int rows, int f_dim,
                 int m_tiles, int n_tiles, int k_steps) {
  extern __shared__ __align__(1024) unsigned char pool[];
  BF* sA = reinterpret_cast<BF*>(pool + kOffA);
  BF* sG = reinterpret_cast<BF*>(pool + kOffG);
  BF* sU = reinterpret_cast<BF*>(pool + kOffU);
  BF* sC = reinterpret_cast<BF*>(pool + kOffC);
  uint64_t* bars = reinterpret_cast<uint64_t*>(pool + kOffBar);
  auto* full = reinterpret_cast<tile::FullBarrier*>(bars);
  auto* empty = reinterpret_cast<tile::EmptyBarrier*>(bars + STAGES);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int total_tiles = m_tiles * n_tiles;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      full[s].init(1);            // one elected producer lane arms each stage
      empty[s].init(kMathWarps);  // one arrival per math warp
    }
  }
  tile::fence_barrier_init();
  __syncthreads();

  // ---------------------------------------------------------------- producer
  if (warp >= kMathWarps) {
    cutlass::arch::warpgroup_reg_dealloc<kProdRegs>();
    if (warp != kMathWarps) return;   // only warp 0 of the group issues TMAs
    tile::PhaseRing<STAGES> empty_phase;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
      // M fastest: the 8 (Pi0.5) or 6 (Pi0) CTAs sharing one N tile run
      // concurrently, so each 32 KB weight frame is read from L2 that many
      // times.  The weights are 134 MB and cannot be resident; the activation
      // is 4 MB and is.
      const int m = t % m_tiles;
      const int n = t / m_tiles;
      for (int g = 0; g < k_steps; ++g) {
        const int s = g % STAGES;
        if (t > blockIdx.x || g >= STAGES) {
          empty[s].wait(empty_phase.take(s));
        }
        if (tile::elect_one()) {
          full[s].arrive_and_expect_tx(kStageBytes);
          BF* a = sA + s * (BM * BK);
          BF* gg = sG + s * (BM * BK);
          BF* uu = sU + s * (BM * BK);
#if GU_BOX_2D
          TileA2D::load(&tm_a, a, g * BK, m * BM, &bars[s]);
          TileB2D::template load<tile::L2Hint::kEvictFirst>(&tm_g, gg, n * BN, g * BK, &bars[s]);
          TileB2D::template load<tile::L2Hint::kEvictFirst>(&tm_u, uu, n * BN, g * BK, &bars[s]);
#else
          // {64, rows, chunks}: c0 is always 0, c1 the row offset, c2 the
          // 64-element chunk index of the contiguous axis.
          tile::tma_load_3d(&tm_a, a, 0, m * BM, g * (BK / kChunk), &bars[s]);
          tile::tma_load_3d<tile::L2Hint::kEvictFirst>(
              &tm_g, gg, 0, g * BK, n * (BN / kChunk), &bars[s]);
          tile::tma_load_3d<tile::L2Hint::kEvictFirst>(
              &tm_u, uu, 0, g * BK, n * (BN / kChunk), &bars[s]);
#endif
        }
      }
    }
    return;
  }

  // ------------------------------------------------------------------- math
  cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
  auto mma = Sel::make();
  Tensor accG = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
  Tensor accU = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
  tile::PhaseRing<STAGES> full_phase;
  const bool warp_leader = (tid & 31) == 0;

  for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
    const int m = t % m_tiles;
    const int n = t / m_tiles;
    clear(accG);
    clear(accU);

    for (int g = 0; g < k_steps; ++g) {
      const int s = g % STAGES;
      full[s].wait(full_phase.take(s));
#if !GU_ABL_NO_MMA
      Tensor sAt = make_tensor(make_smem_ptr(sA + s * (BM * BK)), SmemA{});
      Tensor sGt = make_tensor(make_smem_ptr(sG + s * (BM * BK)), SmemB{});
      Tensor sUt = make_tensor(make_smem_ptr(sU + s * (BM * BK)), SmemB{});
      // Both projections read the SAME A frame: that sharing is the whole
      // point of fusing the pair, and it is why the ring holds one A per two
      // weight frames rather than two.
      tile::gemm_ss<false, -1>(mma, tid, sAt, sGt, accG);
      tile::gemm_ss<false, -1>(mma, tid, sAt, sUt, accU);
      // Release the frame only after the batch that reads it has retired, not
      // when it was issued [release-on-retirement, c7518-wgmma-serialization].
      // Two groups per K-step, so waiting on 2 leaves this step in flight and
      // retires the previous one.
      if (g >= 1) {
        warpgroup_wait<2>();
        if (warp_leader) empty[(g - 1) % STAGES].arrive();
      }
#else
      if (warp_leader) empty[s].arrive();
#endif
    }
#if !GU_ABL_NO_MMA
    warpgroup_wait<0>();
    if (warp_leader) empty[(k_steps - 1) % STAGES].arrive();
#endif

#if !GU_ABL_NO_EPILOGUE
#if !GU_EPI_TMA
    // Direct store. Each thread recovers the (row, col) of its accumulator
    // elements from an identity tensor and writes bf16 pairs: elements e and
    // e+1 of a wgmma C fragment are adjacent columns. No shared tile, no
    // barrier, no store round trip to wait on -- the next tile's mainloop
    // starts as soon as the registers are free.
    {
      auto thr_c = mma.get_thread_slice(tid);
      Tensor cC = thr_c.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
      const int row0 = m * BM, col0 = n * BN;
      CUTE_UNROLL
      for (int e = 0; e < size(accG); e += 2) {
        const int r = row0 + get<0>(cC(e));
        // The last M tile is ragged (968 rows over 8 tiles of 128): the load
        // side is zero-filled by the tensor map, the store side is guarded.
        if (r >= rows) continue;
        const int c = col0 + get<1>(cC(e));
        __nv_bfloat162 v;
#if GU_ABL_NO_GELU
        // Separates the epilogue's transcendental arithmetic from its stores.
        v.x = __float2bfloat16(accG(e));
        v.y = __float2bfloat16(accG(e + 1));
#else
        v.x = __float2bfloat16(gelu_tanh(accG(e)) * accU(e));
        v.y = __float2bfloat16(gelu_tanh(accG(e + 1)) * accU(e + 1));
#endif
        *reinterpret_cast<__nv_bfloat162*>(out + static_cast<long>(r) * f_dim + c) = v;
      }
    }
#else
    // gelu_tanh(gate) * up on the two live fp32 accumulators, one rounding on
    // the store: the two (M, F) branch buffers never exist.
    CUTE_UNROLL
    for (int e = 0; e < size(accG); ++e) {
      accG(e) = gelu_tanh(accG(e)) * accU(e);
    }
    Tensor sCt = make_tensor(make_smem_ptr(sC), SmemC{});
    auto r2s = tile::make_r2s_copy_C<BF>(mma);
    tile::copy_r2s(r2s, tid, tile::convert_fragment<BF>(accG), sCt);
    tile::fence_proxy_async_shared();
    // Math warps only: the producer warpgroup is filling the next tile.
    tile::named_barrier_sync(1, kMathThreads);
    if (tid == 0) {
      // A swizzled store box row is at most 128 B, so a 128-wide bf16 tile is
      // two boxes; the smem source of box i starts BM * 64 elements in.
      CUTE_UNROLL
      for (int i = 0; i < BN / kChunk; ++i) {
        tile::tma_store_2d(&tm_c, sC + i * (BM * kChunk), n * BN + i * kChunk, m * BM);
      }
      tile::store_commit_group();
      // Only the smem source has to be reusable before the next tile writes it.
      tile::store_wait_group_read<0>();
    }
    tile::named_barrier_sync(1, kMathThreads);
#endif  // GU_EPI_TMA
#endif  // GU_ABL_NO_EPILOGUE
  }
}

// ------------------------------------------------------------------ rmsnorm
// A separate launch, deliberately: see the header. One CTA per row, whole row
// in registers across the block, fp32 statistics, one bf16 rounding on the
// store -- the rounding the projection must see. 8 MB of traffic at the Pi0.5
// shape, which the incumbent already moves at ~83% of [ld.bw.dev.dram], so
// this exists to keep the package free of any Target import, not to be faster.
constexpr int kRmsThreads = 256;

__global__ void __launch_bounds__(kRmsThreads)
rms_norm_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                int k_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const __nv_bfloat16* xr = x + static_cast<long>(row) * k_dim;
  __nv_bfloat16* orow = out + static_cast<long>(row) * k_dim;

  float sum = 0.0f;
  for (int i = tid * 2; i < k_dim; i += kRmsThreads * 2) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(xr + i);
    const float a = __bfloat162float(v.x), b = __bfloat162float(v.y);
    sum += a * a + b * b;
  }
  CUTE_UNROLL
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
  __shared__ float warp_sums[kRmsThreads / 32];
  if ((tid & 31) == 0) warp_sums[tid / 32] = sum;
  __syncthreads();
  if (tid < 32) {
    float s = tid < kRmsThreads / 32 ? warp_sums[tid] : 0.0f;
    CUTE_UNROLL
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (tid == 0) warp_sums[0] = rsqrtf(s / static_cast<float>(k_dim) + 1e-6f);
  }
  __syncthreads();
  const float factor = warp_sums[0];

  for (int i = tid * 2; i < k_dim; i += kRmsThreads * 2) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(xr + i);
    __nv_bfloat162 o;
    o.x = __float2bfloat16(__bfloat162float(v.x) * factor);
    o.y = __float2bfloat16(__bfloat162float(v.y) * factor);
    *reinterpret_cast<__nv_bfloat162*>(orow + i) = o;
  }
}

extern "C" int rms_norm_launch(const void* x, void* out, int rows, int k_dim, void* stream) {
  if (k_dim % (2 * kRmsThreads) != 0) return 7001;   // the vectorized stride must divide K
  rms_norm_kernel<<<rows, kRmsThreads, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)x, (__nv_bfloat16*)out, k_dim);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 7000 + static_cast<int>(e);
}

// ------------------------------------------------------------------- host ABI
extern "C" int gated_ffn_geometry(int* threads, int* smem_b, int* stages,
                                  int* bm, int* bn, int* bk, int* map_bytes) {
  *threads = kThreads;
  *smem_b = kSmemBytes;
  *stages = STAGES;
  *bm = BM; *bn = BN; *bk = BK;
  *map_bytes = 4 * static_cast<int>(sizeof(CUtensorMap));
  return 0;
}

struct Maps { CUtensorMap a, g, u, c; };

extern "C" int gated_ffn_make_maps(const void* x_norm, const void* gate_w, const void* up_w,
                                   void* out, int rows, int k_dim, int f_dim, void* blob) {
  Maps* m = reinterpret_cast<Maps*>(blob);
  CUresult rc;
  // Activation (rows, K), K contiguous: {64, rows, K/64}.
  rc = tile::encode_tensor_map_3d<BF>(&m->a, x_norm, kChunk, rows, k_dim / kChunk,
                                      static_cast<uint64_t>(k_dim) * sizeof(BF),
                                      kChunk * sizeof(BF), kChunk, BM, BK / kChunk, kSwizzle);
  if (rc) return 1000 + static_cast<int>(rc);
  // Weights (K, F), F contiguous: {64, K, F/64}; one box is BK rows x BN cols.
  rc = tile::encode_tensor_map_3d<BF>(&m->g, gate_w, kChunk, k_dim, f_dim / kChunk,
                                      static_cast<uint64_t>(f_dim) * sizeof(BF),
                                      kChunk * sizeof(BF), kChunk, BK, BN / kChunk, kSwizzle);
  if (rc) return 2000 + static_cast<int>(rc);
  rc = tile::encode_tensor_map_3d<BF>(&m->u, up_w, kChunk, k_dim, f_dim / kChunk,
                                      static_cast<uint64_t>(f_dim) * sizeof(BF),
                                      kChunk * sizeof(BF), kChunk, BK, BN / kChunk, kSwizzle);
  if (rc) return 3000 + static_cast<int>(rc);
  // Output (rows, F), F contiguous, stored as BN/64 boxes of 64 columns.
  rc = tile::encode_tensor_map_2d<BF>(&m->c, out, f_dim, rows,
                                      static_cast<uint64_t>(f_dim) * sizeof(BF),
                                      kChunk, BM, kSwizzle);
  if (rc) return 4000 + static_cast<int>(rc);
  return 0;
}

extern "C" int gated_ffn_launch(const void* blob, void* out, int rows, int k_dim, int f_dim,
                                int sm_count, void* stream) {
  const Maps* m = reinterpret_cast<const Maps*>(blob);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(gated_ffn_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    if (e != cudaSuccess) return 5000 + static_cast<int>(e);
    attr_set = true;
  }
  const int m_tiles = (rows + BM - 1) / BM;
  const int n_tiles = f_dim / BN;
  const int k_steps = k_dim / BK;
  const int grid = sm_count > 0 ? sm_count : 132;
  gated_ffn_kernel<<<grid, kThreads, kSmemBytes, (cudaStream_t)stream>>>(
      m->a, m->g, m->u, m->c, (BF*)out, rows, f_dim, m_tiles, n_tiles, k_steps);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 6000 + static_cast<int>(e);
}

}  // namespace gated_ffn
