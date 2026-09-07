// Template 14 -- grouped / masked GEMM for a fused MoE layer (sm90).
//
// An MoE expert layer is many small GEMMs sharing a shape: each expert sees a
// different, data-dependent number of tokens.  The naive form -- one launch per
// expert -- pays 1.24 us per launch [launch.lat.dev.ramp] and leaves the tail
// of every small GEMM exposed; with hundreds of experts the launches alone can
// exceed the math.  DeepGEMM's grouped kernels answer with ONE persistent
// launch whose scheduler walks all experts.
//
// Two layouts, one kernel shape:
//
//   * CONTIGUOUS -- tokens are permuted so each expert's rows are adjacent
//     along M, and an array gives the expert owning each M block.  Every block
//     is full work.  This is the prefill/training layout, and it costs a
//     permutation (a scatter before and a gather after).
//   * MASKED -- each expert gets a fixed capacity along M and a runtime count
//     says how many of its rows are real.  Blocks past the count exit before
//     issuing any copy.  This is the decode layout: no permutation, at the
//     price of scheduling slots that do nothing.
//
// Both need the same two things, which is the template's point:
//
//   1. The group index must reach the TMA, not just the epilogue.  Weights are
//      one 3-D tensor map whose outermost coordinate is the expert, so
//      switching experts is a coordinate change, not a new descriptor.  A
//      descriptor per expert would have to be built on the host per step.
//   2. Empty and partial groups must be cheap to skip.  The check happens
//      before the first arrive_and_expect_tx: a barrier armed for bytes that
//      are never issued hangs the consumer forever.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-GRADE: structural
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: cp\.async\.bulk\.tensor\.3d\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n128k16\.f32\.bf16\.bf16
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.global\.shared::cta\.bulk_group

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

constexpr int kBlockM = 64;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kStages = 4;

constexpr int kMathThreads = tmpl::kWarpgroupThreads;
constexpr int kMathWarps = kMathThreads / tmpl::kWarpThreads;
constexpr int kThreads = kMathThreads + tmpl::kWarpgroupThreads;
constexpr int kMathRegs = 240;
constexpr int kTmaRegs = 32;

constexpr uint32_t kNumSMs = 132;

constexpr int kAElems = kBlockM * kBlockK;
constexpr int kBElems = kBlockN * kBlockK;
constexpr int kABytes = kAElems * static_cast<int>(sizeof(Element));
constexpr int kBBytes = kBElems * static_cast<int>(sizeof(Element));

constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kABytes;
constexpr int kOffD = kOffB + kStages * kBBytes;
constexpr int kOffBar = kOffD + kBlockM * kBlockN * static_cast<int>(sizeof(Element));
constexpr int kSmemBytes = kOffBar + 2 * kStages * static_cast<int>(sizeof(uint64_t));

using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutA =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));
using SmemLayoutB =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockK>>{}));

using MmaAtom = SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K,
                                                         GMMA::Major::K>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}));

enum class GroupLayout : int { kContiguous = 0, kMasked = 1 };

}  // namespace

// Per-M-block metadata, built on the host from the routing result.  One array
// read replaces every per-expert launch decision.
struct GroupIndex {
  const int32_t* __restrict__ block_expert;  // contiguous: expert of each M block
  const int32_t* __restrict__ expert_rows;   // masked: valid rows per expert
  int32_t m_blocks;                          // total M blocks across all experts
  int32_t n_blocks;
  int32_t experts;
};

__global__ __launch_bounds__(kThreads, 1) void grouped_moe_gemm_kernel(
    const __grid_constant__ CUtensorMap map_a,  // 2-D: permuted tokens (M, K)
    const __grid_constant__ CUtensorMap map_b,  // 3-D: (K, N, expert)
    const __grid_constant__ CUtensorMap map_d,
    GroupIndex idx, GroupLayout layout, uint32_t k_tiles) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<Element(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<Element(*)[kBElems]>(smem + kOffB);
  auto* const sd = reinterpret_cast<Element*>(smem + kOffD);
  auto* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  auto* const empty = full + kStages;

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / tmpl::kWarpThreads;
  const bool is_math = tid < kMathThreads;

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      tmpl::mbarrier_init(&empty[s], kMathWarps);
    }
    tmpl::fence_barrier_init();
  }
  __syncthreads();

  const int32_t total = idx.m_blocks * idx.n_blocks;

  // Resolves one linear block index to (expert, m_block, n_block), or reports
  // that the block is padding.  Both roles must agree exactly, so it is one
  // function called from both rather than two copies of the arithmetic.
  auto resolve = [&](int32_t block, int32_t& expert, int32_t& m_block,
                     int32_t& n_block) -> bool {
    m_block = block / idx.n_blocks;
    n_block = block % idx.n_blocks;
    if (layout == GroupLayout::kContiguous) {
      expert = idx.block_expert[m_block];
      // A negative marker is how the host says "this block is padding".
      return expert >= 0;
    }
    // Masked: capacity is uniform, so the expert is arithmetic, and the block
    // is real only if the routing put at least one row in it.
    const int32_t blocks_per_expert = idx.m_blocks / idx.experts;
    expert = m_block / blocks_per_expert;
    const int32_t row_in_expert = (m_block % blocks_per_expert) * kBlockM;
    return row_in_expert < idx.expert_rows[expert];
  };

  if (!is_math) {
    tmpl::setmaxnreg_dec<kTmaRegs>();
    if (warp != kMathWarps) { return; }

    uint32_t g = 0;
    for (int32_t block = blockIdx.x; block < total; block += kNumSMs) {
      int32_t expert, m_block, n_block;
      // Skipped BEFORE any barrier is armed: arming for bytes that never
      // arrive is an unrecoverable hang, not a wasted tile.
      if (!resolve(block, expert, m_block, n_block)) { continue; }

      for (uint32_t k = 0; k < k_tiles; ++k, ++g) {
        const uint32_t stage = g % kStages;
        const uint32_t use = g / kStages;
        if (use > 0) { tmpl::wait_parity(&empty[stage], (use - 1) & 1u); }
        if (tmpl::elect_one()) {
          tmpl::arrive_and_expect_tx(&full[stage], kABytes + kBBytes);
          tmpl::tma_load_2d(&map_a, sa[stage], k * kBlockK, m_block * kBlockM,
                            &full[stage]);
          // The expert is the third coordinate: one descriptor, every expert.
          tmpl::tma_load_3d(&map_b, sb[stage], k * kBlockK, n_block * kBlockN,
                            expert, &full[stage]);
        }
      }
    }
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

  uint32_t g = 0;
  for (int32_t block = blockIdx.x; block < total; block += kNumSMs) {
    int32_t expert, m_block, n_block;
    if (!resolve(block, expert, m_block, n_block)) { continue; }

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

      if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&empty[stage]); }
    }

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
      // Output stays in the permuted row order; the un-permute rides the
      // combine that follows, not this kernel.
      tmpl::tma_store_2d(&map_d, sd, n_block * kBlockN, m_block * kBlockM);
      tmpl::tma_store_commit();
      tmpl::tma_store_wait<0>();
    }
    tmpl::named_barrier_sync(1, kMathThreads);
  }
}

cudaError_t configure_grouped_moe_gemm() {
  return cudaFuncSetAttribute(grouped_moe_gemm_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
