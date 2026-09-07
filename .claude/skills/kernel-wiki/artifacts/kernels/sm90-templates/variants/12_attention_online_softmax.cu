// Template 12 -- warp-specialized attention with online softmax (sm90).
//
// The FlashAttention-3 shape: two chained GEMMs per key tile with a softmax
// between them, and nothing ever materialised at sequence length.  What makes
// it fast on sm90, beyond the pipeline of templates 01-02:
//
//   1. P never leaves the register file.  The wgmma A-operand fragment and the
//      wgmma C-accumulator fragment give a thread the SAME rows and columns,
//      so S = QK^T comes out of GEMM 1 already partitioned the way GEMM 2 wants
//      its A operand.  The handoff is an fp32 -> bf16 convert in registers, no
//      shuffle and no shared-memory round trip, and it is why GEMM 2 is the RS
//      form.  The static_assert below is that claim, checked by the compiler.
//   2. The row reduction is a two-step shuffle, not a warp scan.  One row of
//      the fragment lives in the four lanes that share lane/4, so max and sum
//      reduce with xor masks 1 and 2 -- eight lanes' worth of work, not 32.
//   3. exp2f, not expf, with log2(e) folded into the QK scale.  The hardware
//      has a fast path for exp2; multiplying the scale once at the top costs
//      nothing and removes a multiply from every element of every tile.
//   4. Keep the key tile >= 64.  N here is the key-tile width, and a wgmma
//      below N=64 is shared-memory-bound rather than tensor-core-bound
//      [wgmma-tile-n-floor, wgmma.issue.wg.ss].
//
// V is read MN-major because O = P V contracts over keys while V is stored
// (keys, head_dim) with head_dim contiguous.  bf16 wgmma accepts either major;
// an fp8 attention kernel cannot, which is why fp8 FA3 needs a V transpose.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// Softmax accuracy and masking in particular need a parity harness.
//
// CHECK-GRADE: structural
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n128k16\.f32\.bf16\.bf16
// CHECK-PTX: ex2\.approx
// CHECK-PTX: shfl\.sync\.bfly
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

constexpr int kBlockQ = 64;     // queries per CTA; one math warpgroup's M
constexpr int kBlockKeys = 128; // key tile = GEMM 1's N, so it stays >= 64
// 64 keeps every tile's contiguous extent at exactly one SW128 span (64 bf16
// = 128 B).  A head dim of 128 is 256 B and must be loaded as two TMA boxes --
// correct, but it is template 01's rule, not this template's subject.
constexpr int kHeadDim = 64;
constexpr int kStages = 2;

constexpr int kMathThreads = tmpl::kWarpgroupThreads;
constexpr int kMathWarps = kMathThreads / tmpl::kWarpThreads;
constexpr int kThreads = kMathThreads + tmpl::kWarpgroupThreads;
constexpr int kMathRegs = 240;
constexpr int kProducerRegs = 32;

constexpr int kQElems = kBlockQ * kHeadDim;
constexpr int kKElems = kBlockKeys * kHeadDim;
constexpr int kQBytes = kQElems * static_cast<int>(sizeof(Element));
constexpr int kKBytes = kKElems * static_cast<int>(sizeof(Element));

constexpr int kOffQ = 0;
constexpr int kOffK = kOffQ + kQBytes;
constexpr int kOffV = kOffK + kStages * kKBytes;
constexpr int kOffBar = kOffV + kStages * kKBytes;
// Q gets its own barrier after the K/V full/empty pairs.
constexpr int kSmemBytes = kOffBar + (2 * kStages + 1) * static_cast<int>(sizeof(uint64_t));

using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutQ =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockQ>, Int<kHeadDim>>{}));
using SmemLayoutK =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockKeys>, Int<kHeadDim>>{}));
// V is the MN-major operand of GEMM 2, so it needs the MN atom: an MN-major
// descriptor over a K-major image is the classic silent-wrong-answer bug.
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<Int<kHeadDim>, Int<kBlockKeys>>{}));
// Only the LAYOUT of P is needed, to shape GEMM 2's register A operand.
using SmemLayoutP =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockQ>, Int<kBlockKeys>>{}));

// GEMM 1: S(q, key) = Q(q, d) K(key, d)^T -- contraction over d, both K-major.
using MmaAtomQK = SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K,
                                                           GMMA::Major::K>;
using TiledMmaQK = decltype(make_tiled_mma(MmaAtomQK{}, Layout<Shape<_1, _1, _1>>{}));

// GEMM 2: O(q, d) += P(q, key) V(key, d) -- A from registers, V MN-major.
using MmaAtomPV = SM90::GMMA::MMA_64x64x16_F32BF16BF16_RS<GMMA::Major::K,
                                                          GMMA::Major::MN>;
using TiledMmaPV = decltype(make_tiled_mma(MmaAtomPV{}, Layout<Shape<_1, _1, _1>>{}));

// One row of the C/A fragment lives in the four lanes sharing lane/4.
__device__ __forceinline__ float row_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float row_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

}  // namespace

__global__ __launch_bounds__(kThreads, 1) void attention_online_softmax_kernel(
    const __grid_constant__ CUtensorMap map_q,
    const __grid_constant__ CUtensorMap map_k,
    const __grid_constant__ CUtensorMap map_v, uint32_t key_tiles,
    float softmax_scale, float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sq = reinterpret_cast<Element*>(smem + kOffQ);
  auto* const sk = reinterpret_cast<Element(*)[kKElems]>(smem + kOffK);
  auto* const sv = reinterpret_cast<Element(*)[kKElems]>(smem + kOffV);
  auto* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  auto* const empty = full + kStages;
  auto* const q_ready = empty + kStages;

  const uint32_t tid = threadIdx.x;
  const bool is_math = tid < kMathThreads;

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      tmpl::mbarrier_init(&empty[s], kMathWarps);
    }
    tmpl::mbarrier_init(q_ready, 1);
    tmpl::fence_barrier_init();
  }
  __syncthreads();

  // exp2 costs one hardware instruction; folding log2(e) in here is the whole
  // reason the softmax below never calls expf.
  const float scale_log2e = softmax_scale * 1.4426950408889634f;

  if (!is_math) {
    tmpl::setmaxnreg_dec<kProducerRegs>();
    if (tid / tmpl::kWarpThreads != kMathWarps) { return; }

    if (tmpl::elect_one()) {
      tmpl::arrive_and_expect_tx(q_ready, kQBytes);
      tmpl::tma_load_2d(&map_q, sq, 0, blockIdx.x * kBlockQ, q_ready);
    }
    for (uint32_t t = 0; t < key_tiles; ++t) {
      const uint32_t stage = t % kStages;
      const uint32_t use = t / kStages;
      if (use > 0) { tmpl::wait_parity(&empty[stage], (use - 1) & 1u); }
      if (tmpl::elect_one()) {
        // K and V of one tile share a barrier: the math role needs both before
        // it can finish the tile, so splitting them buys nothing.
        tmpl::arrive_and_expect_tx(&full[stage], 2 * kKBytes);
        tmpl::tma_load_2d(&map_k, sk[stage], 0, t * kBlockKeys, &full[stage]);
        tmpl::tma_load_2d(&map_v, sv[stage], 0, t * kBlockKeys, &full[stage]);
      }
    }
    return;
  }

  tmpl::setmaxnreg_inc<kMathRegs>();

  TiledMmaQK mma_qk;
  TiledMmaPV mma_pv;
  auto thr_qk = mma_qk.get_thread_slice(tid);
  auto thr_pv = mma_pv.get_thread_slice(tid);

  auto acc_s = partition_fragment_C(mma_qk, Shape<Int<kBlockQ>, Int<kBlockKeys>>{});
  auto acc_o = partition_fragment_C(mma_pv, Shape<Int<kBlockQ>, Int<kHeadDim>>{});
  // P as GEMM 2's register A operand.  partition_fragment_A reads the layout
  // and allocates registers; it never dereferences, so a null view is enough
  // to say "a P tile of this shape" without spending shared memory on one.
  auto frag_p = thr_pv.partition_fragment_A(
      make_tensor(make_smem_ptr(static_cast<Element*>(nullptr)), SmemLayoutP{}));

  // Rule 1, checked: S's accumulator and P's A operand give this thread the
  // same number of elements, so the handoff is an element-wise convert.
  static_assert(decltype(size(acc_s))::value == decltype(size(frag_p))::value,
                "S accumulator and P A-operand must share the thread mapping");

  clear(acc_o);
  // Two rows per thread, so two running maxima and two running sums.
  float m_0 = -INFINITY, m_1 = -INFINITY;
  float l_0 = 0.f, l_1 = 0.f;

  Tensor tile_q = make_tensor(make_smem_ptr(sq), SmemLayoutQ{});
  tmpl::wait_parity(q_ready, 0);
  auto frag_q = thr_qk.make_fragment_A(thr_qk.partition_A(tile_q));

  for (uint32_t t = 0; t < key_tiles; ++t) {
    const uint32_t stage = t % kStages;
    const uint32_t use = t / kStages;
    tmpl::wait_parity(&full[stage], use & 1u);

    Tensor tile_k = make_tensor(make_smem_ptr(sk[stage]), SmemLayoutK{});
    auto frag_k = thr_qk.make_fragment_B(thr_qk.partition_B(tile_k));

    warpgroup_fence_operand(acc_s);
    warpgroup_arrive();
    mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_q); ++k) {
      cute::gemm(mma_qk, frag_q(_, _, k), frag_k(_, _, k), acc_s);
      mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();  // softmax reads acc_s immediately
    warpgroup_fence_operand(acc_s);

    // Row maxima over this tile.  Elements 0,1 of each group of four are the
    // thread's first row; 2,3 are its second.
    float t_0 = -INFINITY, t_1 = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s) / 4; ++i) {
      t_0 = fmaxf(t_0, fmaxf(acc_s(i * 4 + 0), acc_s(i * 4 + 1)));
      t_1 = fmaxf(t_1, fmaxf(acc_s(i * 4 + 2), acc_s(i * 4 + 3)));
    }
    t_0 = row_max(t_0);
    t_1 = row_max(t_1);

    const float new_m_0 = fmaxf(m_0, t_0 * softmax_scale);
    const float new_m_1 = fmaxf(m_1, t_1 * softmax_scale);
    // Correction for everything already accumulated under the old maximum.
    const float corr_0 = exp2f((m_0 - new_m_0) * 1.4426950408889634f);
    const float corr_1 = exp2f((m_1 - new_m_1) * 1.4426950408889634f);
    m_0 = new_m_0;
    m_1 = new_m_1;

    float s_0 = 0.f, s_1 = 0.f;
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s) / 4; ++i) {
      const float p0 = exp2f(acc_s(i * 4 + 0) * scale_log2e - m_0 * 1.4426950408889634f);
      const float p1 = exp2f(acc_s(i * 4 + 1) * scale_log2e - m_0 * 1.4426950408889634f);
      const float p2 = exp2f(acc_s(i * 4 + 2) * scale_log2e - m_1 * 1.4426950408889634f);
      const float p3 = exp2f(acc_s(i * 4 + 3) * scale_log2e - m_1 * 1.4426950408889634f);
      s_0 += p0 + p1;
      s_1 += p2 + p3;
      // The handoff: same index, fp32 -> bf16, still in registers.
      frag_p(i * 4 + 0) = static_cast<Element>(p0);
      frag_p(i * 4 + 1) = static_cast<Element>(p1);
      frag_p(i * 4 + 2) = static_cast<Element>(p2);
      frag_p(i * 4 + 3) = static_cast<Element>(p3);
    }
    l_0 = l_0 * corr_0 + row_sum(s_0);
    l_1 = l_1 * corr_1 + row_sum(s_1);

    // Rescale the running output before folding this tile in, so O and l stay
    // on the same maximum.
    CUTE_UNROLL
    for (int i = 0; i < size(acc_o) / 4; ++i) {
      acc_o(i * 4 + 0) *= corr_0;
      acc_o(i * 4 + 1) *= corr_0;
      acc_o(i * 4 + 2) *= corr_1;
      acc_o(i * 4 + 3) *= corr_1;
    }

    Tensor tile_v = make_tensor(make_smem_ptr(sv[stage]), SmemLayoutV{});
    auto frag_v = thr_pv.make_fragment_B(thr_pv.partition_B(tile_v));

    warpgroup_fence_operand(frag_p);
    warpgroup_fence_operand(acc_o);
    warpgroup_arrive();
    mma_pv.accumulate_ = GMMA::ScaleOut::One;  // acc_o carries across tiles
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_v); ++k) {
      cute::gemm(mma_pv, frag_p(_, _, k), frag_v(_, _, k), acc_o);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_o);
    warpgroup_fence_operand(frag_p);

    // Released only after the PV batch retired: frag_p is a register operand
    // of an in-flight wgmma until then [pattern-serialized-wgmma].
    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&empty[stage]); }
  }

  // Normalise once at the end -- dividing per tile would be key_tiles times the
  // work for the same answer.
  const float inv_0 = 1.f / l_0, inv_1 = 1.f / l_1;
  CUTE_UNROLL
  for (int i = 0; i < size(acc_o) / 4; ++i) {
    acc_o(i * 4 + 0) *= inv_0;
    acc_o(i * 4 + 1) *= inv_0;
    acc_o(i * 4 + 2) *= inv_1;
    acc_o(i * 4 + 3) *= inv_1;
  }
  CUTE_UNROLL
  for (int i = 0; i < size(acc_o); ++i) {
    out[blockIdx.x * kMathThreads * size(acc_o) + tid * size(acc_o) + i] = acc_o(i);
  }
}

cudaError_t configure_attention_online_softmax() {
  return cudaFuncSetAttribute(attention_online_softmax_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
