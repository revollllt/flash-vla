#pragma once

// One GEMM entry for both tensor-core families.  MmaSelector maps a CTA
// tile (M, N, K), the element types, where each operand lives and its
// major-ness to an instruction at compile time; gemm() then issues with the
// choreography that instruction needs.  Kernels name the selector once and
// never spell an instruction.
//
// Selection rule:
//   wgmma     Threads is a whole number of warpgroups, TileM is a multiple
//             of 64 per warpgroup, B is in shared memory.  A in smem -> SS,
//             A in registers -> RS.  N is the widest table entry that
//             divides TileN (wgmma.cuh).
//   mma.sync  otherwise; both operands must then be register fragments
//             (copy_s2r.cuh stages them) and the warps tile M x N.

#include <cute/tensor.hpp>

#include "tile/sm90/common.cuh"
#include "tile/sm90/mma_sync.cuh"
#include "tile/sm90/wgmma.cuh"

namespace flash_vla::sm90 {

template <class TA, class TB, int TileM, int TileN, int TileK,
          Operand LocA = Operand::kSmem, Operand LocB = Operand::kSmem,
          Major MajA = Major::K, Major MajB = Major::K,
          int Threads = kWarpgroupThreads, int WarpsN = 1>
struct MmaSelector {
  static_assert(Threads % kWarpThreads == 0, "Threads is a whole number of warps");
  static constexpr int kWarpgroups = Threads / kWarpgroupThreads;
  static constexpr bool kUseWgmma =
      (Threads % kWarpgroupThreads == 0) && (kWarpgroups > 0) &&
      (TileM % (64 * (kWarpgroups > 0 ? kWarpgroups : 1)) == 0) &&
      (LocB == Operand::kSmem);
  static constexpr bool kUseMmaSync = !kUseWgmma;

  static_assert(kUseWgmma || (LocA == Operand::kReg && LocB == Operand::kReg),
                "mma.sync reads both operands from registers: stage them with copy_s2r");
  static_assert(kUseWgmma || (MajA == Major::K && MajB == Major::K),
                "mma.sync register fragments are K-major (TN)");
  static_assert(!kUseWgmma || WarpsN == 1,
                "wgmma warpgroups stack along M; split N across selectors instead");

  static constexpr int kWarps = Threads / kWarpThreads;
  static constexpr int kWarpsM = kUseWgmma ? kWarpgroups : kWarps / WarpsN;
  static constexpr int kWarpsN = kUseWgmma ? 1 : WarpsN;
  static_assert(kUseWgmma || kWarps % WarpsN == 0, "WarpsN must divide the warp count");

  static constexpr int kAtomN = kUseWgmma ? wgmma_atom_n<TileN>() : 8;

 private:
  static constexpr auto select() {
    if constexpr (kUseWgmma) {
      return WgmmaAtom<TA, TB, kAtomN, LocA, MajA, MajB>{};
    } else {
      return MmaSyncAtom<TA, TB>{};
    }
  }
  using Entry = decltype(select());

 public:
  using Atom = typename Entry::type;
  static constexpr int kAtomM = Entry::kM;
  static constexpr int kAtomK = Entry::kK;

  // Per-warp footprint of one repetition.  mma.sync pairs two 16 x 8 atoms
  // along N so a warp owns 16 x 16 and the x4 ldmatrix in copy_s2r.cuh fills
  // both operand fragments without a narrower atom.
  static constexpr int kWarpN = kUseWgmma ? kAtomN : 16;
  static_assert(TileM % (kAtomM * kWarpsM) == 0,
                "TileM must be a multiple of the atom M times the warps along M");
  static_assert(TileN % (kWarpN * kWarpsN) == 0,
                "TileN must be a multiple of the per-warp N times the warps along N");
  static_assert(TileK % kAtomK == 0, "TileK must be a multiple of the atom K");

  using AtomLayout =
      cute::Layout<cute::Shape<cute::Int<kWarpsM>, cute::Int<kWarpsN>, cute::_1>>;

 private:
  static constexpr auto make_tiled() {
    if constexpr (kUseWgmma) {
      return cute::make_tiled_mma(Atom{}, AtomLayout{});
    } else {
      return cute::make_tiled_mma(
          Atom{}, AtomLayout{},
          cute::Tile<cute::Int<kAtomM * kWarpsM>, cute::Int<kWarpN * kWarpsN>,
                     cute::Int<kAtomK>>{});
    }
  }

 public:
  using TiledMma = decltype(make_tiled());

  __device__ static __forceinline__ TiledMma make() { return TiledMma{}; }
};

// Issues the K blocks of one stage and accumulates into acc.
//
// wgmma: fence the operands, arrive, one wgmma per K block, commit, then
// wait for at most WgWait groups (WgWait < 0 leaves the batch in flight and
// the caller retires it with cute::warpgroup_wait<N>()).  Arrive/Commit let
// several stages share one batch.  Adapted from FlashAttention-3
// hopper/utils.h (Tri Dao, BSD-3) as carried by FlashMLA csrc/sm90/helpers.h.
//
// mma.sync: a plain unrolled K loop; the batch flags do not apply.
template <bool ZeroInit = false, int WgWait = -1, bool Arrive = true,
          bool Commit = true, class TiledMma, class TensorA, class TensorB,
          class TensorC>
__device__ __forceinline__ void gemm(TiledMma& tiled_mma, TensorA const& tCrA,
                                     TensorB const& tCrB, TensorC& acc) {
  using namespace cute;
  static_assert(size<2>(typename TensorA::layout_type{}) ==
                    size<2>(typename TensorB::layout_type{}),
                "A and B must carry the same number of K blocks");
  if constexpr (is_wgmma_v<TiledMma>) {
    constexpr bool kIsRS =
        !is_base_of<GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr (kIsRS) { warpgroup_fence_operand(const_cast<TensorA&>(tCrA)); }
    warpgroup_fence_operand(acc);
    if constexpr (Arrive) { warpgroup_arrive(); }
    // Only the zero-init entry touches the scale flag before the loop: every
    // call leaves it at One, so the accumulate path relies on that state.
    if constexpr (ZeroInit) { tiled_mma.accumulate_ = GMMA::ScaleOut::Zero; }
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    if constexpr (Commit) { warpgroup_commit_batch(); }
    if constexpr (WgWait >= 0) { warpgroup_wait<WgWait>(); }
    warpgroup_fence_operand(acc);
    if constexpr (kIsRS) { warpgroup_fence_operand(const_cast<TensorA&>(tCrA)); }
  } else {
    static_assert(WgWait == -1 && Arrive && Commit,
                  "batch flags are wgmma-only; leave them at their defaults");
    if constexpr (ZeroInit) { clear(acc); }
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
  }
}

// Both operands are smem tiles (SS wgmma): partition here and issue.
template <bool ZeroInit = false, int WgWait = -1, bool Arrive = true,
          bool Commit = true, class TiledMma, class STensorA, class STensorB,
          class TensorC>
__device__ __forceinline__ void gemm_ss(TiledMma& tiled_mma, int tid,
                                        STensorA const& sA, STensorB const& sB,
                                        TensorC& acc) {
  static_assert(is_wgmma_v<TiledMma>, "gemm_ss is a wgmma entry");
  auto thr = tiled_mma.get_thread_slice(tid);
  auto tCrA = thr.make_fragment_A(thr.partition_A(sA));
  auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
  gemm<ZeroInit, WgWait, Arrive, Commit>(tiled_mma, tCrA, tCrB, acc);
}

// A is a register fragment, B an smem tile (RS wgmma).
template <bool ZeroInit = false, int WgWait = -1, bool Arrive = true,
          bool Commit = true, class TiledMma, class RTensorA, class STensorB,
          class TensorC>
__device__ __forceinline__ void gemm_rs(TiledMma& tiled_mma, int tid,
                                        RTensorA const& rA, STensorB const& sB,
                                        TensorC& acc) {
  static_assert(is_wgmma_v<TiledMma>, "gemm_rs is a wgmma entry");
  auto thr = tiled_mma.get_thread_slice(tid);
  auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
  gemm<ZeroInit, WgWait, Arrive, Commit>(tiled_mma, rA, tCrB, acc);
}

}  // namespace flash_vla::sm90
