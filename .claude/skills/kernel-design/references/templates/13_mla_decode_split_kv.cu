// Template 13 -- split-KV decode with an LSE combine (sm90).
//
// Decode has almost no parallelism of its own: one query position per sequence
// means the M dimension is a handful of heads, so a batch of 32 fills a few SMs
// and leaves the rest idle no matter how good the mainloop is.  FlashMLA's
// answer, and every fast decode kernel's, is to split the KV length across CTAs
// and merge afterwards.  The protocol, not the attention math, is this
// template's subject -- template 12 owns the softmax.
//
//   1. Split along keys, never along heads.  Each CTA sweeps its own key range
//      and produces a PARTIAL output plus the log-sum-exp of its range.  Two
//      partials merge exactly, because softmax over a union is a weighted sum
//      of softmaxes over a partition, with LSE carrying the weights.
//   2. Write partials in fp32.  The merge is a rescale-and-add of values that
//      may differ by many orders of magnitude; rounding partials to bf16 before
//      merging loses accuracy the fused path never loses.
//   3. Keep the no-split fast path.  A request that fits one CTA writes the
//      final output directly and never touches the accumulator buffers -- the
//      combine kernel is pure overhead for it.
//   4. Chain the two kernels with PDL.  The combine grid's prologue -- index
//      arithmetic, its own tile coordinates -- has no dependency on the split
//      grid's output, so it can run during the split kernel's tail instead of
//      after a full launch gap [launch.lat.dev.ramp, producer-fusion-pdl].
//      The trigger goes as soon as the key sweep is done, before the partial
//      writes -- it publishes nothing, so there is no reason to hold it until
//      the stores.  The combine kernel's wait is what orders those.
//
// The number of splits is a scheduling decision made on the host from the SM
// count and the batch: enough that grid >= 3x SMs [sched.ctas.sm.knee], few
// enough that each split still amortises its Q load.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n128k16\.f32\.bf16\.bf16
// CHECK-PTX: griddepcontrol\.launch_dependents
// CHECK-PTX: griddepcontrol\.wait
// CHECK-PTX: ex2\.approx
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

constexpr int kBlockQ = 64;      // query positions x heads packed into M
constexpr int kBlockKeys = 128;
constexpr int kHeadDim = 64;
constexpr int kStages = 2;
constexpr int kThreads = tmpl::kWarpgroupThreads;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;

constexpr int kQElems = kBlockQ * kHeadDim;
constexpr int kKElems = kBlockKeys * kHeadDim;
constexpr int kQBytes = kQElems * static_cast<int>(sizeof(Element));
constexpr int kKBytes = kKElems * static_cast<int>(sizeof(Element));

constexpr int kOffQ = 0;
constexpr int kOffK = kOffQ + kQBytes;
constexpr int kOffBar = kOffK + kStages * kKBytes;
constexpr int kSmemBytes = kOffBar + (2 * kStages + 1) * static_cast<int>(sizeof(uint64_t));

using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutQ =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockQ>, Int<kHeadDim>>{}));
using SmemLayoutK =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockKeys>, Int<kHeadDim>>{}));

using MmaAtom = SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K,
                                                         GMMA::Major::K>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}));

__device__ __forceinline__ float row_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float row_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

}  // namespace

// One CTA per (request, split).  The host precomputes each split's key range so
// the kernel does no scheduling arithmetic of its own.
struct SplitTask {
  int32_t request;
  int32_t key_begin;
  int32_t key_end;
  int32_t split_idx;   // index within this request's splits
  int32_t is_no_split; // 1 when this CTA owns the whole request
};

__global__ __launch_bounds__(kThreads, 1) void mla_decode_split_kernel(
    const __grid_constant__ CUtensorMap map_q,
    const __grid_constant__ CUtensorMap map_k,
    const SplitTask* __restrict__ tasks, float softmax_scale,
    float* __restrict__ o_accum,   // (split, kBlockQ, kHeadDim) fp32 partials
    float* __restrict__ lse_accum, // (split, kBlockQ) fp32
    float* __restrict__ o_final) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sq = reinterpret_cast<Element*>(smem + kOffQ);
  auto* const sk = reinterpret_cast<Element(*)[kKElems]>(smem + kOffK);
  auto* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  auto* const empty = full + kStages;
  auto* const q_ready = empty + kStages;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % tmpl::kWarpThreads;
  const uint32_t warp = tid / tmpl::kWarpThreads;
  const SplitTask task = tasks[blockIdx.x];

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      tmpl::mbarrier_init(&empty[s], kWarps);
    }
    tmpl::mbarrier_init(q_ready, 1);
    tmpl::fence_barrier_init();
  }
  __syncthreads();

  const float scale_log2e = softmax_scale * 1.4426950408889634f;

  if (tid == 0) {
    tmpl::arrive_and_expect_tx(q_ready, kQBytes);
    tmpl::tma_load_2d(&map_q, sq, 0, task.request * kBlockQ, q_ready);
  }
  tmpl::wait_parity(q_ready, 0);

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockQ>, Int<kBlockKeys>>{});
  Tensor tile_q = make_tensor(make_smem_ptr(sq), SmemLayoutQ{});
  auto frag_q = thr_mma.make_fragment_A(thr_mma.partition_A(tile_q));

  // Running softmax state for the two rows this thread owns, over THIS SPLIT's
  // key range only.  m and l are what the combine kernel needs, so they are
  // outputs here, not scratch.
  float m_0 = -INFINITY, m_1 = -INFINITY, l_0 = 0.f, l_1 = 0.f;
  float o_0 = 0.f, o_1 = 0.f;  // stand-in for the PV accumulator of template 12

  const int32_t tiles = (task.key_end - task.key_begin + kBlockKeys - 1) / kBlockKeys;
  for (int32_t t = 0; t < tiles; ++t) {
    const uint32_t stage = t % kStages;
    const uint32_t use = t / kStages;
    if (tid == 0) {
      if (use > 0) { tmpl::wait_parity(&empty[stage], (use - 1) & 1u); }
      tmpl::arrive_and_expect_tx(&full[stage], kKBytes);
      tmpl::tma_load_2d(&map_k, sk[stage], 0, task.key_begin + t * kBlockKeys,
                        &full[stage]);
    }
    tmpl::wait_parity(&full[stage], use & 1u);

    Tensor tile_k = make_tensor(make_smem_ptr(sk[stage]), SmemLayoutK{});
    auto frag_k = thr_mma.make_fragment_B(thr_mma.partition_B(tile_k));

    warpgroup_fence_operand(acc_s);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_q); ++k) {
      cute::gemm(tiled_mma, frag_q(_, _, k), frag_k(_, _, k), acc_s);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_s);

    float t_0 = -INFINITY, t_1 = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s) / 4; ++i) {
      t_0 = fmaxf(t_0, fmaxf(acc_s(i * 4 + 0), acc_s(i * 4 + 1)));
      t_1 = fmaxf(t_1, fmaxf(acc_s(i * 4 + 2), acc_s(i * 4 + 3)));
    }
    const float new_m_0 = fmaxf(m_0, row_max(t_0) * softmax_scale);
    const float new_m_1 = fmaxf(m_1, row_max(t_1) * softmax_scale);
    const float corr_0 = exp2f((m_0 - new_m_0) * 1.4426950408889634f);
    const float corr_1 = exp2f((m_1 - new_m_1) * 1.4426950408889634f);
    m_0 = new_m_0;
    m_1 = new_m_1;

    float s_0 = 0.f, s_1 = 0.f;
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s) / 4; ++i) {
      s_0 += exp2f(acc_s(i * 4 + 0) * scale_log2e - m_0 * 1.4426950408889634f) +
             exp2f(acc_s(i * 4 + 1) * scale_log2e - m_0 * 1.4426950408889634f);
      s_1 += exp2f(acc_s(i * 4 + 2) * scale_log2e - m_1 * 1.4426950408889634f) +
             exp2f(acc_s(i * 4 + 3) * scale_log2e - m_1 * 1.4426950408889634f);
    }
    l_0 = l_0 * corr_0 + row_sum(s_0);
    l_1 = l_1 * corr_1 + row_sum(s_1);
    o_0 = o_0 * corr_0 + s_0;  // the PV wgmma of template 12 goes here
    o_1 = o_1 * corr_1 + s_1;

    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&empty[stage]); }
  }

  // The key sweep is done; the writes below are tail the combine kernel's
  // prologue can overlap.  Its wait, not this trigger, is what orders them.
  tmpl::pdl_trigger();

  const uint32_t r_0 = warp * 16 + lane / 4;
  const uint32_t r_1 = r_0 + 8;

  if (task.is_no_split) {
    // Whole request in one CTA: normalise here and skip the merge entirely.
    o_final[task.request * kBlockQ + r_0] = o_0 / l_0;
    o_final[task.request * kBlockQ + r_1] = o_1 / l_1;
  } else {
    // Partials stay UNNORMALISED and fp32; the LSE carries the weight.
    const int64_t base = static_cast<int64_t>(task.split_idx) * kBlockQ;
    o_accum[base + r_0] = o_0;
    o_accum[base + r_1] = o_1;
    lse_accum[base + r_0] = m_0 + log2f(l_0) * 0.6931471805599453f;
    lse_accum[base + r_1] = m_1 + log2f(l_1) * 0.6931471805599453f;
  }
}

// Merges the splits of one request.  Trivially parallel and bandwidth-bound;
// its whole cost should hide under the split kernel's tail.
__global__ __launch_bounds__(128) void mla_decode_combine_kernel(
    const float* __restrict__ o_accum, const float* __restrict__ lse_accum,
    const int32_t* __restrict__ split_begin, const int32_t* __restrict__ split_count,
    float* __restrict__ o_final) {
  const int32_t request = blockIdx.x;
  const int32_t row = threadIdx.x;

  // Prologue first: reading the task metadata does not depend on the partials,
  // so it belongs before the wait.
  const int32_t begin = split_begin[request];
  const int32_t count = split_count[request];

  tmpl::pdl_wait();  // from here on we read what the split kernel wrote

  // Two passes: the maximum LSE, then one rescaled accumulation.  Doing it in
  // one pass would need the partials twice anyway, and this way no term ever
  // exponentiates above zero.
  float lse_max = -INFINITY;
  for (int32_t s = 0; s < count; ++s) {
    lse_max = fmaxf(lse_max, lse_accum[static_cast<int64_t>(begin + s) * kBlockQ + row]);
  }

  float acc = 0.f, denom = 0.f;
  for (int32_t s = 0; s < count; ++s) {
    const int64_t idx = static_cast<int64_t>(begin + s) * kBlockQ + row;
    const float w = exp2f((lse_accum[idx] - lse_max) * 1.4426950408889634f);
    acc += w * o_accum[idx];
    denom += w;
  }
  o_final[static_cast<int64_t>(request) * kBlockQ + row] = acc / denom;
}

cudaError_t configure_mla_decode_split() {
  return cudaFuncSetAttribute(mla_decode_split_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
