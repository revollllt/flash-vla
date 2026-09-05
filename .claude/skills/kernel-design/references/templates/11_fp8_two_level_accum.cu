// Template 11 -- fp8 wgmma with CUDA-core promotion and block scales (sm90).
//
// The one thing an sm90 fp8 GEMM must do that a bf16 one must not.  Hopper's
// fp8 wgmma accumulates at REDUCED precision -- it is not an fp32 accumulator
// -- so a long K reduction done entirely inside the tensor core loses accuracy
// that no tolerance choice recovers.  DeepGEMM's sm90 fp8 kernels answer with
// two-level accumulation:
//
//   * wgmma accumulates one BLOCK_K (128 channels) into `acc`, zero-initialised
//     every block;
//   * a CUDA-core pass folds `acc` into an ordinary fp32 `final`, applying the
//     per-row and per-column dequant scales while the values are in registers.
//
// The two jobs pay for each other.  Per-128-channel scaling has to break the K
// loop at that granularity anyway, so the promotion the accuracy needs is free
// at the point the scaling already forces a stop.  That is why BLOCK_K is 128
// here and not a tuning knob.
//
// The fragment mapping is the hardware wgmma C layout, not a convention:
// lane l of warp w owns rows w*16 + l/4 and that + 8, and its i-th group of
// four accumulators covers columns i*8 + (l%4)*2 and + 1.  Scales are indexed
// with exactly that, which is why no shuffle appears in the promotion.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// Accuracy claims in particular need a parity harness -- this template only
// shows where the promotion goes.
//
// CHECK-GRADE: structural
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n128k32\.f32\.e4m3\.e4m3
// CHECK-PTX: wgmma\.commit_group\.sync\.aligned
// CHECK-PTX: wgmma\.wait_group\.sync\.aligned
// CHECK-PTX: fma\.rn\.f32
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::float_e4m3_t;

constexpr int kBlockM = 64;
constexpr int kBlockN = 128;
// Fixed by the scaling granularity, not by tiling: per-128-channel scales mean
// the K loop must stop and promote every 128 channels.
constexpr int kBlockK = 128;
constexpr int kStages = 2;

constexpr int kThreads = tmpl::kWarpgroupThreads;
constexpr int kWarps = kThreads / tmpl::kWarpThreads;

constexpr int kAElems = kBlockM * kBlockK;
constexpr int kBElems = kBlockN * kBlockK;
constexpr int kABytes = kAElems * static_cast<int>(sizeof(Element));
constexpr int kBBytes = kBElems * static_cast<int>(sizeof(Element));
// Per-row scales for A, per-column scales for B: the "1d1d" scaling shape.
constexpr int kSfaBytes = kBlockM * static_cast<int>(sizeof(float));
constexpr int kSfbBytes = kBlockN * static_cast<int>(sizeof(float));

constexpr int kOffA = 0;
constexpr int kOffB = kOffA + kStages * kABytes;
constexpr int kOffSfa = kOffB + kStages * kBBytes;
constexpr int kOffSfb = kOffSfa + kStages * kSfaBytes;
constexpr int kOffBar = kOffSfb + kStages * kSfbBytes;
constexpr int kSmemBytes = kOffBar + 2 * kStages * static_cast<int>(sizeof(uint64_t));

// fp8 wgmma is K-major only (TN); there is no MN-major fp8 atom to choose.
using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutA =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));
using SmemLayoutB =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockK>>{}));

using MmaAtom = SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}));

}  // namespace

__global__ __launch_bounds__(kThreads, 1) void fp8_two_level_accum_kernel(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b,
    const __grid_constant__ CUtensorMap map_sfa,
    const __grid_constant__ CUtensorMap map_sfb, uint32_t k_blocks,
    float* __restrict__ out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto* const sa = reinterpret_cast<Element(*)[kAElems]>(smem + kOffA);
  auto* const sb = reinterpret_cast<Element(*)[kBElems]>(smem + kOffB);
  auto* const sfa = reinterpret_cast<float(*)[kBlockM]>(smem + kOffSfa);
  auto* const sfb = reinterpret_cast<float(*)[kBlockN]>(smem + kOffSfb);
  auto* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  auto* const empty = full + kStages;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % tmpl::kWarpThreads;
  const uint32_t warp = tid / tmpl::kWarpThreads;

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      tmpl::mbarrier_init(&empty[s], kWarps);
    }
    tmpl::fence_barrier_init();
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

  // The fp32 accumulator the tensor core does not give us.  It never enters a
  // wgmma, so it keeps full precision across the whole K reduction.
  float final_acc[decltype(size(acc))::value] = {};

  // Hardware wgmma C fragment mapping; see the header note.
  const uint32_t r_0 = warp * 16 + lane / 4;
  const uint32_t r_1 = r_0 + 8;
  const uint32_t col = (lane % 4) * 2;

  for (uint32_t kb = 0; kb < k_blocks; ++kb) {
    const uint32_t stage = kb % kStages;
    const uint32_t use = kb / kStages;

    if (tid == 0) {
      if (use > 0) { tmpl::wait_parity(&empty[stage], (use - 1) & 1u); }
      tmpl::arrive_and_expect_tx(&full[stage],
                                 kABytes + kBBytes + kSfaBytes + kSfbBytes);
      tmpl::tma_load_2d(&map_a, sa[stage], kb * kBlockK, 0, &full[stage]);
      tmpl::tma_load_2d(&map_b, sb[stage], kb * kBlockK, 0, &full[stage]);
      // Scales ride the same barrier: they are needed at exactly the moment
      // the tile they scale is.
      tmpl::tma_load_2d(&map_sfa, sfa[stage], 0, kb, &full[stage]);
      tmpl::tma_load_2d(&map_sfb, sfb[stage], 0, kb, &full[stage]);
    }
    tmpl::wait_parity(&full[stage], use & 1u);

    const float sa_0 = sfa[stage][r_0];
    const float sa_1 = sfa[stage][r_1];

    Tensor tile_a = make_tensor(make_smem_ptr(sa[stage]), SmemLayoutA{});
    Tensor tile_b = make_tensor(make_smem_ptr(sb[stage]), SmemLayoutB{});
    auto frag_a = thr_mma.make_fragment_A(thr_mma.partition_A(tile_a));
    auto frag_b = thr_mma.make_fragment_B(thr_mma.partition_B(tile_b));

    // Zero every block: `acc` is a per-block partial, never the running sum.
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_a); ++k) {
      cute::gemm(tiled_mma, frag_a(_, _, k), frag_b(_, _, k), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    // The promotion reads acc immediately, so this batch cannot stay in
    // flight -- the cost of two-level accumulation is exactly this stall,
    // which is why the promotion is per BLOCK_K and not per wgmma.
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc);

    // CUDA-core pass: scale and fold.  Four accumulators at a time, because
    // that group shares two rows and two columns.
    CUTE_UNROLL
    for (int i = 0; i < size(acc) / 4; ++i) {
      const float sb_0 = sfb[stage][i * 8 + col + 0];
      const float sb_1 = sfb[stage][i * 8 + col + 1];
      final_acc[i * 4 + 0] += sa_0 * sb_0 * acc(i * 4 + 0);
      final_acc[i * 4 + 1] += sa_0 * sb_1 * acc(i * 4 + 1);
      final_acc[i * 4 + 2] += sa_1 * sb_0 * acc(i * 4 + 2);
      final_acc[i * 4 + 3] += sa_1 * sb_1 * acc(i * 4 + 3);
    }

    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&empty[stage]); }
  }

  CUTE_UNROLL
  for (int i = 0; i < size(acc); ++i) {
    out[tid * size(acc) + i] = final_acc[i];
  }
}

cudaError_t configure_fp8_two_level_accum() {
  return cudaFuncSetAttribute(fp8_two_level_accum_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSmemBytes);
}
