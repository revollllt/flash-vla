// hut/mma.cuh -- the tensor-core instruction under test, SELECTED not spelled.
//
// Three units built the same six type aliases independently (mma_rate,
// overlap, pipeline_ws), differing only in the shared-memory atom and whether
// K was fixed. They are here once, parameterised on both.
//
// Names are CUTLASS's: ss_op_selector, make_tiled_mma, tile_to_shape,
// partition_A/B, partition_fragment_C. [references/vocabulary.md]

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/numeric_types.h>

namespace hut {

namespace G = cute::SM90::GMMA;
using bf16 = cutlass::bfloat16_t;

// The instruction is chosen by the same code CUTLASS dispatches through, not
// named by us. `ss_op_selector` picks the widest atom dividing Tile_N for the
// (A, B, C) element types. That matters twice: retargeting to fp8/fp16/tf32 is
// a template argument rather than another hand-written table, and the probe
// cannot name an atom the library would not have used -- which would make the
// measured rate a property of the probe. [protocol.md rule 11b]
template <int M, int N, int K>
using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;

template <int M, int N, int K, class TA = bf16, class TB = bf16,
          class TC = float>
using SsMmaOp = decltype(G::ss_op_selector<TA, TB, TC, TileShape<M, N, K>,
                                           G::Major::K, G::Major::K>());

template <int M, int N, int K, class TA = bf16, class TB = bf16,
          class TC = float>
using SsTiledMma = decltype(cute::make_tiled_mma(SsMmaOp<M, N, K, TA, TB, TC>{}));

// Rule 11b at COMPILE time: assert the selector landed on the instruction whose
// rate the constants describe. This is an assertion, never the source of the
// instruction. If a CUTLASS update changes the dispatch, the probe fails to
// BUILD rather than quietly re-measuring a different instruction under the old
// tags.
// The parameters are spelled with a trailing underscore because `K` alone is
// captured inside `G::Major::K` by the preprocessor, which silently rewrites it
// to `G::Major::16` and produces four unreadable template errors.
#define HUT_ASSERT_SS_ATOM_BF16(M_, N_, K_)                                  \
  static_assert(                                                             \
      cute::is_same_v<::hut::SsMmaOp<M_, N_, K_>,                            \
                      ::hut::G::MMA_##M_##x##N_##x##K_##_F32BF16BF16_SS<     \
                          ::hut::G::Major::K, ::hut::G::Major::K>>,          \
      "ss_op_selector chose a different atom for " #M_ "x" #N_ "x" #K_)

// Shared-memory layout for one operand tile. The ATOM is a parameter because
// the units disagree on purpose: an interleaved atom when the probe re-issues
// against one resident tile, a swizzled one when a TMA is filling stages
// underneath. Passing the wrong one is a descriptor/data mismatch that still
// runs, so each unit states which it means.
template <class Atom, int ROWS, int COLS>
using SmemLayout = decltype(cute::tile_to_shape(
    Atom{}, cute::Shape<cute::Int<ROWS>, cute::Int<COLS>>{}));

// A warpgroup is 128 threads; the SS atoms are warpgroup-wide. Units that
// launch 256 threads slice the SAME 128-thread TiledMMA twice on purpose, so
// two warpgroups issue independently against one pair of tiles.
constexpr int kWarpgroupThreads = 128;

// The wgmma issue sequence, kept as free CUTLASS calls rather than wrapped:
//
//   warpgroup_fence_operand(accum);   // before
//   warpgroup_arrive();
//   ... cute::gemm(...) x NGROUP ...
//   warpgroup_commit_batch();
//   warpgroup_wait<WAIT>();
//   warpgroup_wait<0>(); warpgroup_fence_operand(accum);   // after
//
// The fences are NOT optional decoration. Without them ptxas emits C7515
// ("wgmma.mma_async instructions are serialized due to non wgmma instructions
// defining accumulator registers") and every rate measured is the SERIALIZED
// rate. A probe that ignored that warning would report a real number for the
// wrong machine. cute::gemm does not insert them -- they stay the caller's, so
// the in-flight count keeps meaning what the sweep says it means.
//
// Grep the build log for C7515 and C7518; both were live defects here.

}  // namespace hut
