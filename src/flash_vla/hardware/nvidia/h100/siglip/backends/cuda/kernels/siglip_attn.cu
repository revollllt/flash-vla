// Fused SigLIP vision attention, sm90.
//
//   out[v, t, h*DH + d] = sum_k softmax_k(Q[v,t,h,:] . K[v,k,h,:] * scale) V[v,k,h,d]
//
// One launch in place of the production route's three (a cuDNN workspace
// memset, the SDPA kernel, and the transpose+copy back into the graph buffer).
// Reads the packed QKV projection where it lies: qkv is (VIEWS, TOKENS, 3*DIM)
// with Q|K|V blocks of DIM and head-major inside a block, so head h of block b
// is the 72-wide column window [b*DIM + h*DH, +DH). The view axis is a batch
// axis; there is no attention across views, no mask and no causality.
//
// Why cp.async and not TMA. DH is 72, so a head's window is 144 B: not one of
// the 32/64/128 B swizzle widths, and not a multiple of the wgmma K step of
// 16 either. The contraction therefore runs at K=80 with columns 72..79 held
// at zero, and TMA cannot produce that image -- it writes a box contiguously,
// so it can deliver 72-wide rows or a wider aligned window, never 72 real
// columns strided by 80. cp.async can: nine 16 B chunks per row cover the 72
// real columns, and the pad column is written once at frame init and never
// again. This is the mechanism a prior TileLang attempt at this fusion lacked
// -- it fell back to 2-byte predicated loads and lost in the captured graph --
// and it is why the per-SM TMA issue cost that binds the vision GEMM sites
// does not apply here.
//
// Build (see siglip_attn.py, which owns the cache key and the exact flags):
//   nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I<cutlass>/include -I<tile root>
//
// CUDA-graph safety: the only host-side entry is a plain kernel launch on the
// caller's stream. There is no tensor map, so nothing on this path calls the
// driver, and the kernel allocates nothing.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>

#include "tile/sm90/common.cuh"
#include "tile/sm90/copy_g2s.cuh"
#include "tile/sm90/gemm.cuh"
#include "tile/sm90/smem_layout.cuh"

namespace {

using namespace cute;
namespace tile = flash_vla::sm90;
using BF = tile::BF16;

// ------------------------------------------------------------- geometry
// Fixed by models/pi05/spec.py and the packed-QKV view the production
// attention wrapper takes; siglip/geometry.py is the single mirror and
// siglip_attn.py checks these against it at load time.
constexpr int TOKENS = 256;   // queries and keys per view
constexpr int HEADS = 16;
constexpr int DH = 72;        // real head width
constexpr int DIM = HEADS * DH;      // 1152, one Q|K|V block
constexpr int QKV_DIM = 3 * DIM;     // 3456, the packed row stride

// The contraction extent. wgmma steps K by 16 and 72 is not a multiple of 16,
// so every tile carries eight zero columns; the parity harness computes its
// reference from explicitly zero-extended 80-wide heads, so a pad column that
// is not zero fails rel_rms rather than hiding.
constexpr int DH_PAD = 80;
static_assert(DH_PAD % 16 == 0, "wgmma K step");
static_assert(DH_PAD % 8 == 0, "a 16-byte chunk is 8 bf16");
constexpr int CHUNKS_REAL = DH / 8;      // 9 chunks of 8 elements cover 72
constexpr int CHUNKS_PAD = DH_PAD / 8;   // 10 with the zero chunk
static_assert(CHUNKS_REAL * 8 == DH, "72 is a whole number of 16-byte chunks");

constexpr int BM = 64;    // queries per CTA
constexpr int BKK = 64;   // keys per pipeline stage
constexpr int STAGES = 3;
static_assert(TOKENS % BM == 0 && TOKENS % BKK == 0, "no ragged tail at 256");
constexpr int Q_BLOCKS = TOKENS / BM;
constexpr int KEY_BLOCKS = TOKENS / BKK;

constexpr int kThreads = tile::kWarpgroupThreads;  // one math warpgroup
// A second math warpgroup buys no tensor-core throughput [wgmma.ratio.sm.wg2],
// and this kernel is bound by its loads and its softmax, not by wgmma issue.

// ------------------------------------------------------------ shared pool
// Q is live for the whole key loop, so it cannot alias the ring. The output
// staging tile can: it is written only after the last key block retires.
constexpr int TILE_ELEMS = BM * DH_PAD;          // also BKK * DH_PAD
constexpr int TILE_B = TILE_ELEMS * 2;           // 10240
constexpr int OFF_Q = 0;
constexpr int OFF_K = OFF_Q + TILE_B;
constexpr int OFF_V = OFF_K + STAGES * TILE_B;
constexpr int OFF_END = OFF_V + STAGES * TILE_B;
constexpr int OFF_OUT = OFF_K;                   // overlays the retired rings
constexpr int SMEM_B = OFF_END;
static_assert(SMEM_B <= 227 * 1024, "shared memory per block");
static_assert(OFF_END - OFF_K >= TILE_B, "output staging overlays the rings");

// --------------------------------------------------------- smem layouts
// 80 elements is 160 B, five 32-byte swizzle spans, so SW32 is the widest
// swizzle this width admits; SW64 and SW128 would need 32 or 64 columns.
// A 16-byte chunk is 8 elements and never straddles a 16-element span, which
// is what lets the cp.async destination be one contiguous store.
using LayoutQ = tile::SmemTileLayout<BF, BM, DH_PAD, tile::Major::K, 32>;
using LayoutK = tile::SmemTileLayout<BF, BKK, DH_PAD, tile::Major::K, 32>;
// B of O = P V is (K = keys, N = dh), so dh is the contiguous mode: MN-major
// with Step<_2,_1> gives the frame the same image as K's, which is what lets
// one copy routine fill either.
using LayoutV = tile::SmemTileLayout<BF, DH_PAD, BKK, tile::Major::MN, 32,
                                     Step<_2, _1>>;
static_assert(LayoutQ::kBytes == TILE_B && LayoutV::kBytes == TILE_B, "frame size");

// S = Q K^T: both operands K-major over the padded head width.
using SelS = tile::MmaSelector<BF, BF, BM, BKK, DH_PAD, tile::Operand::kSmem,
                               tile::Operand::kSmem, tile::Major::K,
                               tile::Major::K, kThreads>;
// O = P V: P is the S accumulator reinterpreted in place, V is MN-major.
// N is the padded width, so O's columns 72..79 are P times zero and the
// epilogue simply does not store them.
using SelO = tile::MmaSelector<BF, BF, BM, DH_PAD, BKK, tile::Operand::kReg,
                               tile::Operand::kSmem, tile::Major::K,
                               tile::Major::MN, kThreads>;
static_assert(SelS::kUseWgmma && SelO::kUseWgmma, "both stages are wgmma");

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The wgmma C fragment (2,2,N/8) reinterpreted as the RS A fragment
// ((2,2,2),N/16): the register order is the same, so P feeds the P.V wgmma
// with no shuffle and no smem round trip. Ported from FlashAttention-3
// hopper/utils.h (convert_layout_acc_Aregs) by way of the Pi0.5 encoder
// attention kernel; it belongs in tile/sm90 once a second consumer needs it.
template <class Layout>
__device__ __forceinline__ auto acc_to_aregs(Layout acc) {
  auto l = logical_divide(get<0>(acc), Shape<Underscore, Underscore, _2>{});
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
                     get<1>(acc),
                     coalesce(make_layout(get<2, 1>(l), get<2>(acc))));
}

// One 64-row frame of one Q|V|K block, global -> shared, issued by the whole
// warpgroup. `block` selects Q, K or V within the packed row; `row0` is the
// first token. The nine real chunks are cp.async; the tenth is the zero pad,
// stored directly because it is a constant and cp.async has no immediate form.
//
// The caller owns the group: cp_async_commit_group() after the call, and
// cp_async_wait_group<N>() plus a CTA barrier before any wgmma reads it.
template <class SLayout>
__device__ __forceinline__ void load_frame(BF* smem, const BF* __restrict__ qkv_view,
                                           int block, int row0, int head, int tid) {
  Tensor s = make_tensor(make_smem_ptr(smem), typename SLayout::type{});
  const int rows = SLayout::kMajor == tile::Major::K ? SLayout::kRows
                                                     : SLayout::kCols;
  // One thread per (row, chunk) pair per round; 64 rows x 10 chunks = 640
  // pairs over 128 threads is five rounds with no predication.
  constexpr int kPairs = 64 * CHUNKS_PAD;
  for (int i = tid; i < kPairs; i += kThreads) {
    const int row = i / CHUNKS_PAD;
    const int chunk = i % CHUNKS_PAD;
    if (row >= rows) { continue; }
    // The layout is (MN, K) for K-major and (K, MN) for MN-major, and the
    // logical element is (token, dh) either way.
    BF* dst = SLayout::kMajor == tile::Major::K ? &s(row, chunk * 8)
                                                : &s(chunk * 8, row);
    if (chunk < CHUNKS_REAL) {
      const BF* src = qkv_view + static_cast<long>(row0 + row) * QKV_DIM
                    + block * DIM + head * DH + chunk * 8;
      tile::cp_async_16(dst, src);
    } else {
      // Columns 72..79 must read as exact zero: 0 * NaN is NaN, and a padded
      // column that is merely small would still bias every softmax row.
      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

}  // namespace

// Outside the anonymous namespace: a __global__ parameter type must have
// external linkage, or nvcc's generated launch stub cannot name it.
namespace siglip_attn {
struct Params {
  const BF* __restrict__ qkv;   // (VIEWS, TOKENS, QKV_DIM)
  BF* __restrict__ out;         // (VIEWS, TOKENS, DIM)
  float scale_log2;             // DH^-0.5 * log2(e)
};
}  // namespace siglip_attn

// grid = (Q_BLOCKS, HEADS, views); one CTA owns one (view, head, query block).
// At three views that is 3 * 16 * 4 = 192 CTAs, above the 128-CTA delivery
// knee [ld.ctas.dev.knee] and 1.45 waves on 132 SMs; BM=128 would be one wave
// of 96 but doubles the S and O accumulators, and this kernel has register
// room to spare only at BM=64.
__global__ __launch_bounds__(kThreads, 1)
void siglip_attn_kernel(__grid_constant__ const siglip_attn::Params params) {
  extern __shared__ __align__(1024) uint8_t smem[];
  BF* const sq = reinterpret_cast<BF*>(smem + OFF_Q);
  BF* const sk = reinterpret_cast<BF*>(smem + OFF_K);
  BF* const sv = reinterpret_cast<BF*>(smem + OFF_V);

  const int tid = static_cast<int>(threadIdx.x);
  const int qb = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int view = static_cast<int>(blockIdx.z);
  const BF* const qkv_view = params.qkv + static_cast<long>(view) * TOKENS * QKV_DIM;

  // Q is loaded once and stays live for every key block.
  load_frame<LayoutQ>(sq, qkv_view, /*block=*/0, qb * BM, head, tid);
  // Prologue: fill STAGES-1 key frames so the first iteration finds one ready.
  CUTE_UNROLL
  for (int g = 0; g < STAGES; ++g) {
    if (g >= KEY_BLOCKS) { break; }
    load_frame<LayoutK>(sk + g * TILE_ELEMS, qkv_view, 1, g * BKK, head, tid);
    load_frame<LayoutV>(sv + g * TILE_ELEMS, qkv_view, 2, g * BKK, head, tid);
    tile::cp_async_commit_group();
  }

  auto mma_s = SelS::make();
  auto mma_o = SelO::make();
  auto acc_s = partition_fragment_C(mma_s, Shape<Int<BM>, Int<BKK>>{});
  auto acc_o = partition_fragment_C(mma_o, Shape<Int<BM>, Int<DH_PAD>>{});
  clear(acc_o);

  // Online softmax state, in the log2 domain so the rescale is one ex2.
  // The C fragment gives each thread two rows of every eight, so the running
  // max and sum are two scalars, not a vector.
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};
  // Carried across one iteration: the rescale that belongs to block g is
  // applied at the top of iteration g, after the previous block's P.V has
  // retired and before this block's is issued.
  float alpha[2] = {0.f, 0.f};

  Tensor tq = make_tensor(make_smem_ptr(sq), typename LayoutQ::type{});

  // The softmax of one key block, in place on acc_s, in the log2 domain.
  // Updates m_run, l_run and alpha; acc_s holds the unconverted probabilities
  // on return. Every reduction is over the four lanes of the quad that share
  // a row: each lane holds two of every eight columns of the C fragment.
  auto softmax_block = [&](float scale_log2) {
    float m_new[2];
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) { m_new[r] = m_run[r]; }
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s); ++i) {
      const int r = (i / 2) % 2;
      acc_s(i) *= scale_log2;
      m_new[r] = fmaxf(m_new[r], acc_s(i));
    }
    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      m_new[r] = fmaxf(m_new[r], __shfl_xor_sync(0xffffffffu, m_new[r], 1));
      m_new[r] = fmaxf(m_new[r], __shfl_xor_sync(0xffffffffu, m_new[r], 2));
      alpha[r] = (m_run[r] == -INFINITY) ? 0.f : fast_exp2(m_run[r] - m_new[r]);
      l_run[r] *= alpha[r];
      m_run[r] = m_new[r];
    }
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s); ++i) {
      const int r = (i / 2) % 2;
      const float p = fast_exp2(acc_s(i) - m_run[r]);
      acc_s(i) = p;
      l_run[r] += p;
    }
  };

  auto frame = [&](BF* ring, int slot) { return ring + slot * TILE_ELEMS; };

  // Prologue: block 0's scores and softmax, so the loop always enters with a
  // converted P in hand and can issue P.V before anything else. Block 0 is the
  // first of STAGES committed groups, so STAGES-1 may still be pending.
  tile::cp_async_wait_group<STAGES - 1>();
  __syncthreads();
  {
    Tensor tk = make_tensor(make_smem_ptr(frame(sk, 0)), typename LayoutK::type{});
    tile::gemm_ss<true, 0>(mma_s, tid, tq, tk, acc_s);
  }
  softmax_block(params.scale_log2);
  auto p_frag = make_fragment_like<BF>(acc_s);
  CUTE_UNROLL
  for (int i = 0; i < size(acc_s); ++i) { p_frag(i) = static_cast<BF>(acc_s(i)); }

  // Fully unrolled: KEY_BLOCKS is a compile-time 4, and a runtime back edge is
  // half of what makes ptxas serialize a wgmma pipeline stage it cannot prove
  // safe [c7518-wgmma-serialization].
  CUTE_UNROLL
  for (int g = 0; g < KEY_BLOCKS; ++g) {
    const int stage = g % STAGES;

    // 1. Rescale the running output by this block's factor. acc_o is settled:
    //    the previous iteration waited for its P.V to retire.
    CUTE_UNROLL
    for (int i = 0; i < size(acc_o); ++i) { acc_o(i) *= alpha[(i / 2) % 2]; }

    // 2. Issue the NEXT block's scores first, then this block's P.V. wgmma
    //    groups retire in commit order, so committing the score GEMM first is
    //    what makes warpgroup_wait<1> below mean "scores landed, P.V may still
    //    be running". Committing them the other way round would wait for the
    //    P.V and leave the scores in flight, which is the reverse of the point.
    const bool more = (g + 1 < KEY_BLOCKS);
    if (more) {
      // Block g+1's group; see the accounting beside NEW_PROLOGUE.
      tile::cp_async_wait_group<STAGES - 2>();
      __syncthreads();
      Tensor tk = make_tensor(make_smem_ptr(frame(sk, (g + 1) % STAGES)),
                              typename LayoutK::type{});
      tile::gemm_ss<true, -1>(mma_s, tid, tq, tk, acc_s);
    }

    // 3. O += P_g V_g, left in flight so the softmax below runs under it
    //    [wgmma.stages.wg.knee]. Slot g holds V_g, slot g+1 holds K_{g+1} and
    //    the prefetch below targets slot g+2: three live slots, STAGES = 3.
    Tensor tv = make_tensor(make_smem_ptr(frame(sv, stage)), typename LayoutV::type{});
    Tensor p_a = make_tensor(p_frag.data(), acc_to_aregs(p_frag.layout()));
    tile::gemm_rs<false, -1>(mma_o, tid, p_a, tv, acc_o);

    if (more) {
      // At most the P.V remains pending, so the score GEMM has landed.
      cute::warpgroup_wait<1>();
      softmax_block(params.scale_log2);
    }

    // 4. Retire P.V: only now is V's frame free and acc_o settled.
    cute::warpgroup_wait<0>();

    // 5. Convert the next block's probabilities. After the wait, because the
    //    P.V above read p_frag as a register operand and must not have it
    //    change underneath.
    if (more) {
      CUTE_UNROLL
      for (int i = 0; i < size(acc_s); ++i) { p_frag(i) = static_cast<BF>(acc_s(i)); }
    }

    __syncthreads();
    const int prefetch = g + STAGES;  // the prologue filled through g = STAGES-1
    if (prefetch < KEY_BLOCKS) {
      const int slot = prefetch % STAGES;
      load_frame<LayoutK>(frame(sk, slot), qkv_view, 1, prefetch * BKK, head, tid);
      load_frame<LayoutV>(frame(sv, slot), qkv_view, 2, prefetch * BKK, head, tid);
    }
    tile::cp_async_commit_group();
  }

  // ------------------------------------------------------------- epilogue
  // Normalise by the row sum and stage through shared memory: the accumulator
  // sits in the wgmma fragment layout, so a direct global store writes 4-byte
  // pieces per thread [epilogue-staging-short-k].
  float inv[2];
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    // The row sum, like the row max, is spread across the four lanes of the
    // quad that share a row: each lane holds two of every eight columns, so a
    // lane's own l_run is a quarter of the denominator. Reducing the max but
    // not the sum leaves the output scaled by about four -- close enough in
    // direction to keep cosine near 1 and far enough in magnitude to fail
    // rel_rms, which is what the two gates together are for.
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
    // Every key is attended, so the row sum is never zero; the guard is for a
    // degenerate shape rather than for real data.
    inv[r] = (l_run[r] > 0.f) ? __frcp_rn(l_run[r]) : 0.f;
  }
  CUTE_UNROLL
  for (int i = 0; i < size(acc_o); ++i) { acc_o(i) *= inv[(i / 2) % 2]; }

  // Every outstanding copy targets the rings the staging tile is about to
  // overlay, so the group must drain even though the last iterations issue
  // none: an empty commit still counts toward the group depth.
  tile::cp_async_wait_group<0>();
  __syncthreads();
  BF* const sout = reinterpret_cast<BF*>(smem + OFF_OUT);
  Tensor tout = make_tensor(make_smem_ptr(sout), typename LayoutQ::type{});
  auto thr_o = mma_o.get_thread_slice(tid);
  Tensor frag_out = thr_o.partition_C(tout);
  CUTE_UNROLL
  for (int i = 0; i < size(acc_o); ++i) { frag_out(i) = static_cast<BF>(acc_o(i)); }
  __syncthreads();

  // Store the 72 real columns; 72..79 are P times the zero pad and are dropped.
  BF* const out_view = params.out + static_cast<long>(view) * TOKENS * DIM;
  constexpr int kPairs = BM * CHUNKS_REAL;
  for (int i = tid; i < kPairs; i += kThreads) {
    const int row = i / CHUNKS_REAL;
    const int chunk = i % CHUNKS_REAL;
    const uint4 v = *reinterpret_cast<const uint4*>(&tout(row, chunk * 8));
    BF* dst = out_view + static_cast<long>(qb * BM + row) * DIM + head * DH + chunk * 8;
    *reinterpret_cast<uint4*>(dst) = v;
  }
}

// ------------------------------------------------------------------ host ABI
// One plain launch: no tensor map, so nothing here calls the driver and the
// whole path is safe inside CUDA-graph capture.
extern "C" int siglip_attn_launch(const void* qkv, void* out, int views,
                                  float scale, void* stream) {
  if (views <= 0) { return 1; }
  static bool attr_set = false;
  if (!attr_set) {
    const cudaError_t e = cudaFuncSetAttribute(
        siglip_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_B);
    if (e != cudaSuccess) { return static_cast<int>(e); }
    attr_set = true;
  }
  siglip_attn::Params params{static_cast<const BF*>(qkv), static_cast<BF*>(out),
                scale * 1.4426950408889634f};
  const dim3 grid(Q_BLOCKS, HEADS, static_cast<unsigned>(views));
  siglip_attn_kernel<<<grid, kThreads, SMEM_B, static_cast<cudaStream_t>(stream)>>>(params);
  return static_cast<int>(cudaGetLastError());
}

// Compiled-in geometry, so the Python side asserts against the model spec
// rather than restating it.
extern "C" void siglip_attn_geometry(int* tokens, int* heads, int* dh, int* dh_pad,
                                     int* bm, int* bkk, int* stages, int* threads,
                                     int* smem_b) {
  *tokens = TOKENS; *heads = HEADS; *dh = DH; *dh_pad = DH_PAD;
  *bm = BM; *bkk = BKK; *stages = STAGES; *threads = kThreads; *smem_b = SMEM_B;
}
