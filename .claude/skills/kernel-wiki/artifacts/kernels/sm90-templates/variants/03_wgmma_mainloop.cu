// Template 03 -- wgmma mainloop over a swizzled smem ring (sm90).
//
// Turns the ring of templates 01-02 into a GEMM mainloop.  The rules it exists
// to show:
//
//   1. One swizzle, three agreeing parties.  The TMA box, the shared layout,
//      and the wgmma descriptor must name the same swizzle, and an SW128 box
//      row is exactly 128 B (64 bf16).  Get this wrong and the copy is fine but
//      the tensor core reads a permuted tile.
//   2. Descriptors are CuTe's job.  The 64-bit matrix descriptor packs address,
//      leading and stride byte offsets (all >> 4), a base offset and a layout
//      code; hand-rolling it is how kernels silently read the wrong tile.  Name
//      an atom and let MMA_Traits build it.
//   3. Never wgmma.wait_group 0 in a mainloop.  Waiting the batch to empty
//      costs 20-30% [wgmma.stages.wg.knee]; keep at least one in flight.
//   4. Keep N >= 64.  At N=32 the instruction re-reads the full A tile per
//      half-width B tile and runs ~3x [wgmma.issue.wg.ss], shared-memory-bound
//      rather than tensor-core-bound; below N=32 switch instruction, not tile
//      [mma.xover.n.wgmma].
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// This repo's kernels reach the same instructions through tile/sm90/gemm.cuh
// (MmaSelector + gemm), which is where the fence/arrive/commit/wait contract
// is owned.
//
// CHECK-GRADE: structural
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n64k16\.f32\.bf16\.bf16
// CHECK-PTX: wgmma\.fence\.sync\.aligned
// CHECK-PTX: wgmma\.commit_group\.sync\.aligned
// CHECK-PTX: wgmma\.wait_group\.sync\.aligned
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include "sm90_common.cuh"

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

// N = 64 is the floor for a wgmma that is not shared-memory-bound.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 64;
constexpr int kDepth = 2;  // L2-resident feed; a cold DRAM feed wants 4

constexpr int kFrameElems = kTileM * kTileK;
constexpr int kFrameBytes = kFrameElems * static_cast<int>(sizeof(Element));
constexpr int kThreads = tmpl::kWarpgroupThreads;

// SW128 K-major: one atom row is 128 B, which is also the widest legal swizzled
// TMA box row -- so kTileK = 64 bf16 lands as exactly one box.
using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutA =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kTileM>, Int<kTileK>>{}));
using SmemLayoutB =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kTileN>, Int<kTileK>>{}));

// SS form: both operands stay in shared memory and the instruction reads them
// through descriptors.  A in registers would be the RS atom instead.
using MmaAtom = SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K,
                                                        GMMA::Major::K>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}));

}  // namespace

__global__ __launch_bounds__(kThreads) void wgmma_mainloop_kernel(
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b, int32_t k_tiles,
    float* __restrict__ out) {
  __shared__ alignas(1024) Element sa[kDepth][kFrameElems];
  __shared__ alignas(1024) Element sb[kDepth][kFrameElems];
  __shared__ alignas(8) uint64_t full[kDepth];
  __shared__ alignas(8) uint64_t empty[kDepth];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  if (tid == 0) {
    for (int32_t s = 0; s < kDepth; ++s) {
      tmpl::mbarrier_init(&full[s], 1);
      tmpl::mbarrier_init(&empty[s], kThreads);
    }
  }
  tmpl::fence_barrier_init();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{});
  clear(acc);

  for (int32_t stage = 0; stage < k_tiles; ++stage) {
    const int32_t slot = stage % kDepth;
    const uint32_t use = static_cast<uint32_t>(stage / kDepth);

    if (tid == 0) {
      if (use > 0) { tmpl::wait_parity(&empty[slot], (use - 1) & 1u); }
      // One arrival covers both copies, so the count is both frames.
      tmpl::arrive_and_expect_tx(&full[slot], 2 * kFrameBytes);
      tmpl::tma_load_2d(&map_a, sa[slot], stage * kTileK,
                        static_cast<int32_t>(blockIdx.x) * kTileM, &full[slot]);
      tmpl::tma_load_2d(&map_b, sb[slot], stage * kTileK,
                        static_cast<int32_t>(blockIdx.y) * kTileN, &full[slot]);
    }
    tmpl::wait_parity(&full[slot], use & 1u);

    Tensor tile_a = make_tensor(make_smem_ptr(sa[slot]), SmemLayoutA{});
    Tensor tile_b = make_tensor(make_smem_ptr(sb[slot]), SmemLayoutB{});
    auto frag_a = thr_mma.make_fragment_A(thr_mma.partition_A(tile_a));
    auto frag_b = thr_mma.make_fragment_B(thr_mma.partition_B(tile_b));

    // The batch: fence the accumulator, arrive, issue every K block, commit.
    // accumulate_ is Zero only for the very first block of the whole mainloop;
    // leaving it at One afterwards is what makes the K loop accumulate.
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma.accumulate_ =
        (stage == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(frag_a); ++k) {
      cute::gemm(tiled_mma, frag_a(_, _, k), frag_b(_, _, k), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    // One batch stays in flight: the previous stage's math overlaps this
    // stage's copies.  warpgroup_wait<0> here would serialize the mainloop.
    warpgroup_wait<1>();
    warpgroup_fence_operand(acc);

    // Released only after the wait above retires the batch that reads it --
    // releasing on issue would let the producer overwrite a live operand.
    if (stage > 0) {
      const int32_t prev = (stage - 1) % kDepth;
      tmpl::mbarrier_arrive(&empty[prev]);
    }
  }

  // Retire the last batch before the accumulator is read.
  warpgroup_wait<0>();
  warpgroup_fence_operand(acc);

  const int32_t base = (static_cast<int32_t>(blockIdx.x) * kTileM) * kTileN;
  CUTE_UNROLL
  for (int i = 0; i < size(acc); ++i) {
    out[base + tid * size(acc) + i] = acc(i);
  }
}

CUresult make_operand_map(CUtensorMap* map, const Element* base, uint64_t k,
                          uint64_t rows, uint32_t box_rows) {
  // The swizzle here IS the SmemAtom's swizzle; they are one decision.
  return tmpl::encode_tile_map_2d(map, base, k, rows, kTileK, box_rows,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  CU_TENSOR_MAP_SWIZZLE_128B);
}
