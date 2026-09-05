// Template 44 -- DeepGEMM's Mega MoE on sm90: dispatch, expert GEMM 1 with
// SwiGLU, expert GEMM 2 and combine as one persistent kernel, with the
// two-stage scheduler that keeps it deadlock-free.
//
// DeepGEMM (MIT) ships Mega MoE for sm100 only (`impls/sm100_*_mega_moe.cuh`,
// `scheduler/mega_moe.cuh`, `layout/mega_moe.cuh`); the sm90 path SGLang uses
// comes from a build of DeepGEMM this repository does not pin, and neither the
// upstream nor the sgl-project fork carries an sm90 source. This file is the
// sm90 port of the sm100 design, single rank (the NVLink dispatch/combine
// become local copies through the same buffers and counters), bf16 weights,
// and no shared experts. Every scheduler and counter mechanism is the
// upstream one; the tensor-core path is wgmma in place of 2-SM UMMA.
//
// THE MACHINE, as upstream builds it (per CTA, 1 CTA/SM, grid = SM count)
//
//   dispatch warps   count each expert's tokens (smem atomics -> one gmem
//                    atomic per expert per SM that both allocates this SM's
//                    slot range and, in its high word, counts the SMs that
//                    have reported), write source indices, then PULL tokens
//                    into a pool ordered expert-major, one BLOCK_M block at a
//                    time, through a shared-memory bounce with bulk copies.
//   scheduler warp   turns pool blocks x N blocks into tasks: a WARMUP of L1
//                    tasks first, then L1 and L2 tasks interleaved, publishing
//                    each through a 2-slot shared-memory pipeline to the TMA
//                    warps and the math warpgroups.
//   token TMA warp   waits for the block's tokens (an l1_full / l2_full ring
//                    counter), then streams the token tile per K block.
//   weight TMA warp  streams the expert's 128-row weight tile per K block.
//   2 math warpgroups  wgmma, swapped A/B: weights are M, tokens are N, so a
//                    decode-sized block of 8..192 tokens is the wgmma N and
//                    the 128 weight rows split 64/64 over the two groups.
//                    Their epilogue is the SwiGLU (L1) or the combine scatter
//                    (L2), and they bump the ring counters.
//
// FIVE MECHANISMS THAT ARE THE POINT
//
//  1. TWO DEPENDENT STAGES OVER ONE WORKER POOL, WITHOUT DEADLOCK. An L2 task
//     for pool block b needs all 32 L1 tasks of b done. If every SM picked an
//     L2 task while b's L1 tasks were still unissued, nothing would progress.
//     `get_num_l1_warmup_waves` sizes an initial run of L1-only waves from
//     the per-block L1:L2 task ratio, and after it every scheduler issues one
//     L1 task per L2 task; an L2 task additionally waits until the L1 task
//     COUNTER has passed all of its block's L1 tasks (issued, not finished),
//     which is what keeps the schedule ahead of the dependency.
//  2. A RING OF POOL BLOCKS with four counters each -- l1_full (tokens
//     landed), l1_empty (L1 N blocks done: the slot's tokens may be
//     overwritten), l2_full (L1 N blocks done: the L2 input is complete),
//     l2_empty (L2 N blocks done: the L2 input may be overwritten). Targets
//     scale with the slot's use count, so nothing is ever reset mid-kernel.
//     `get_num_max_live_pool_blocks` sizes the ring so the warmup cannot
//     outrun it.
//  3. GRANULARITY-8 INTERLEAVED GATE/UP ROWS. Within a 128-row L1 weight
//     block rows alternate 8 gate, 8 up. In the wgmma accumulator a thread
//     holds rows r and r+8 of the same 16-row group, so (gate, up) of one
//     output feature sit in the same thread's registers and SwiGLU needs no
//     shuffle and no shared memory. SGLang's `_interleave_l1_weight_only`
//     is the offline half of this.
//  4. THE TOP-K WEIGHT IS APPLIED IN THE L1 EPILOGUE, so the combine is a
//     plain sum over top-k slots and needs no per-token metadata beyond the
//     slot layout; the L2 epilogue scatters each pool token's row straight
//     into its (token, top-k slot) of the combine buffer.
//  5. TASK INFO AS A SHARED-MEMORY PIPELINE. The scheduler is one warp; four
//     consumers (two TMA warps, two math warpgroups) read each task, so a
//     2-slot ring with a full barrier (1 arrival) and an empty barrier (4
//     elected arrivals) is what decouples them.
//
// WHAT WAS SIMPLIFIED (deviations from upstream, not from the design):
//
//   * One rank: no symmetric memory, no NVLink barrier, no rank peeling in
//     the dispatch (the loop is kept, over one rank).
//   * Counters are reset by a memset before each launch instead of by the
//     dispatch warps at the end of the previous one.
//   * The combine reads the top-k slots with plain vectorized loads; upstream
//     uses a 2-stage 1-D TMA pipeline per warp.
//   * bf16 weights (the `sm100_bf16` variant); SGLang's production sm90 path
//     is fp8 with 128-group scales.
//
// UPSTREAM NUMBERS (DeepGEMM PR #316, B200, EP8, fp8/fp4, 1 token per rank):
// DeepSeek-V4-Flash (256 experts, top-6, hidden 4096, inter 2048) 56.5 us,
// 1311 GB/s, 1.96x the unfused path; V4-Pro (384 experts, top-6, hidden 7168,
// inter 3072) 108.1 us, 1758 GB/s, 1.61x. Not the same GPU, precision or
// rank count as here; the comparable figure is the fraction of the weight
// stream's bandwidth floor.
//
// STATUS (H100 SXM5, CUDA 13.1, clocks not pinned, 132 SMs; 16 experts x
// (7168, 2048) bf16, top-8, graph-captured, min of 3 x 20; every row matches
// the double reference with no element beyond a quarter of the rms):
//
//   tokens  BLOCK_M  pool  streamed   time      rate       floor [ld.bw.dev.dram]
//      8       8      16    1.41 GB    497 us   2.84 TB/s   103%
//    128      64      22    1.97 GB    673 us   2.93 TB/s   106%
//    128     128      16    1.44 GB    563 us   2.56 TB/s    93%
//    512      64      73    6.56 GB   1863 us   3.52 TB/s   127%
//    512     128      41    3.74 GB   1141 us   3.28 TB/s   119%
//
// "Streamed" counts an expert's weights once per pool block, which is what
// the kernel reads; rates above the load floor are the TMA stream
// [tma.bw.dev.dram] plus L2 hits when the same expert's next block follows
// closely. The lever is BLOCK_M: the largest block that keeps each expert in
// one pool block minimises the stream, which is why upstream JIT-compiles it
// per call (its candidates run to 192). Upstream's published rows are fp8/fp4
// on B200 across 8 ranks (V4-Pro at 1 token per rank: 108 us, 1758 GB/s) and
// are not directly comparable; the comparable claim is the fraction of the
// weight stream's floor.
//
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -I third_party/cutlass/include -o mk44 44_megamoe_sm90.cu -lcuda && ./mk44
//   -DMK44_TOKENS=512 -DMK44_BLOCK_M=128 -DMK44_STAGES=5   larger blocks, fewer streams
//
// CHECK-GRADE: reference
// CHECK-INCLUDE: third_party/cutlass/include
// CHECK-PTX: wgmma\.mma_async\.sync\.aligned\.m64n64k16\.f32\.bf16\.bf16
// CHECK-PTX: cp\.async\.bulk\.tensor\.3d\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: cp\.async\.bulk\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: cp\.async\.bulk\.global\.shared::cta\.bulk_group
// CHECK-PTX: red\.release\.gpu\.global\.add\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: atom\.global\.add\.u64

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#include "sm90_common.cuh"

// ------------------------------------------------------------------ shape
// DeepSeek-V3-shaped experts on one rank: 16 local experts of hidden 7168,
// intermediate 2048, top-8 routing over the local experts, 128 tokens.
#ifndef MK44_EXPERTS
#define MK44_EXPERTS 16
#endif
#ifndef MK44_HIDDEN
#define MK44_HIDDEN 7168
#endif
#ifndef MK44_INTER
#define MK44_INTER 2048
#endif
#ifndef MK44_TOPK
#define MK44_TOPK 8
#endif
#ifndef MK44_TOKENS
#define MK44_TOKENS 128
#endif
#ifndef MK44_BLOCK_M
#define MK44_BLOCK_M 64
#endif
#ifndef MK44_STAGES
#define MK44_STAGES 6
#endif

namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;

constexpr int kNumExperts = MK44_EXPERTS;
constexpr int kHidden = MK44_HIDDEN;
constexpr int kInter = MK44_INTER;
constexpr int kTopk = MK44_TOPK;
constexpr int kNumTokens = MK44_TOKENS;
constexpr int kBlockM = MK44_BLOCK_M;   // tokens per pool block = wgmma N
constexpr int kBlockN = 128;            // weight rows per task = wgmma M (2 x 64)
constexpr int kBlockK = 64;
constexpr int kStages = MK44_STAGES;
constexpr int kL1ShapeN = 2 * kInter;   // gate and up, interleaved
constexpr int kL1ShapeK = kHidden;
constexpr int kL2ShapeN = kHidden;
constexpr int kL2ShapeK = kInter;
constexpr int kNumL1Blocks = kL1ShapeN / kBlockN;   // 32
constexpr int kNumL2Blocks = kL2ShapeN / kBlockN;   // 56
constexpr int kL1OutBlockN = kBlockN / 2;           // 64 SwiGLU features per L1 task
static_assert(kNumExperts <= 32, "one expert per lane in the scheduler");
static_assert(kHidden % kBlockK == 0 && kInter % kBlockK == 0, "K blocks");
static_assert(kL1ShapeN % kBlockN == 0 && kL2ShapeN % kBlockN == 0, "N blocks");
static_assert(kL2ShapeK % kL1OutBlockN == 0, "L2's K is assembled from L1 output blocks");
static_assert(kBlockM % 8 == 0 && kBlockM <= 256, "wgmma N");

// ---------------------------------------------------------------- warps
constexpr int kDispatchWarps = 4;
constexpr int kProducerWarps = 4;   // scheduler, token TMA, weight TMA, idle
constexpr int kMathWarpgroups = 2;
constexpr int kMathWarps = kMathWarpgroups * 4;
constexpr int kNumWarps = kDispatchWarps + kProducerWarps + kMathWarps;  // 16
constexpr int kNumThreads = kNumWarps * 32;                               // 512
constexpr int kDispatchThreads = kDispatchWarps * 32;
constexpr int kMathThreads = kMathWarps * 32;
constexpr int kWarpScheduler = kDispatchWarps + 0;
constexpr int kWarpTokenTma = kDispatchWarps + 1;
constexpr int kWarpWeightTma = kDispatchWarps + 2;
constexpr int kFirstMathWarp = kDispatchWarps + kProducerWarps;
constexpr int kDispatchRegs = 40;
constexpr int kProducerRegs = 40;
constexpr int kMathRegs = 216;
static_assert(kDispatchThreads * kDispatchRegs + kProducerWarps * 32 * kProducerRegs + kMathThreads * kMathRegs <= 65536, "");

// named barriers
constexpr uint32_t kBarDispatch = 1;
constexpr uint32_t kBarMath = 2;
constexpr uint32_t kBarMathWg0 = 3;  // + wg

// ----------------------------------------------------------------- smem
constexpr int kWBytes = kBlockN * kBlockK * 2;   // 16 KB weight tile
constexpr int kTBytes = kBlockM * kBlockK * 2;   // token tile
constexpr int kEpiBytes = kBlockM * kBlockN * 2;  // [tokens][128] bf16 transpose tile
constexpr int kPullChunk = 7168;                  // dispatch bounce, per warp
static_assert(kHidden * 2 % kPullChunk == 0, "hidden bytes must be whole pull chunks");
constexpr int kOffW = 0;
constexpr int kOffT = kOffW + kStages * kWBytes;
constexpr int kOffEpi = kOffT + kStages * kTBytes;
constexpr int kOffPull = kOffEpi + kEpiBytes;
constexpr int kOffBar = kOffPull + kDispatchWarps * kPullChunk;
constexpr int kNumBarriers = 2 * kStages + 2 * 2 + kDispatchWarps;  // full/empty, task full/empty, pull
constexpr int kOffTask = kOffBar + kNumBarriers * 8;
constexpr int kOffCounts = kOffTask + 2 * 64;
constexpr int kSmemBytes = kOffCounts + kNumExperts * 4 + 1024;  // + alignment slack
static_assert(kSmemBytes <= 227 * 1024, "shared memory");

using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
using SmemLayoutW = decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockK>>{}));
using SmemLayoutT = decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));
using MmaAtom = decltype(SM90::GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<kBlockM>, _64>,
                                                    GMMA::Major::K, GMMA::Major::K>());
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<Int<kMathWarpgroups>, _1, _1>>{}));

}  // namespace

// ------------------------------------------------------------- workspace
// Kernel parameter types live outside the anonymous namespace: nvcc's host
// stub has to name them.
struct TokenSrc { uint32_t token, topk; };

enum class Phase : uint32_t { kNone = 0, kLinear1 = 1, kLinear2 = 2 };

struct alignas(16) TaskInfo {
  uint32_t phase;
  uint32_t expert;
  uint32_t m_block;       // block within the expert
  uint32_t n_block;       // 128-row block of the expert's weights
  uint32_t pool_block;    // block within the whole pool (expert-major)
  uint32_t valid_m;
  uint32_t shape_k;
  uint32_t pad;
};

struct Workspace {
  uint32_t* grid_sync;          // [2]
  uint32_t* task_count;         // [2]: L1, L2
  uint64_t* expert_send_count;  // [E]: low = tokens, high = SMs reported
  uint64_t* expert_recv_sum;    // [E]: the same, published by SM 0 (one rank)
  uint32_t* l1_full;            // [ring_blocks]
  uint32_t* l1_empty;
  uint32_t* l2_full;
  uint32_t* l2_empty;
  uint32_t* src_token_topk;     // [E][max_recv]: token * topk + k
  TokenSrc* token_src;          // [max_pool_tokens]
  float* l1_topk_w;             // [ring_tokens]
  __nv_bfloat16* l1_tokens;     // [ring_tokens][hidden]
  __nv_bfloat16* l2_tokens;     // [ring_tokens][inter]
  __nv_bfloat16* combine;       // [topk][tokens][hidden]
  uint32_t ring_blocks;
  uint32_t max_recv;            // slots per expert in src_token_topk
};

struct Params {
  const __nv_bfloat16* x;        // [tokens][hidden]
  const int64_t* topk_idx;       // [tokens][topk]
  const float* topk_w;           // [tokens][topk]
  __nv_bfloat16* y;              // [tokens][hidden]
  CUtensorMap l1_w;              // [E][2*inter][hidden], box 64 x 128 x 1
  CUtensorMap l2_w;              // [E][hidden][inter]
  CUtensorMap l1_tokens;         // [ring_tokens][hidden], box 64 x BLOCK_M
  CUtensorMap l2_tokens;         // [ring_tokens][inter]
  Workspace ws;
  int32_t num_tokens;
  int32_t num_sms;
};

namespace {

// ---------------------------------------------------------------- atoms
__device__ __forceinline__ uint32_t ld_acquire_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_volatile_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_add(uint32_t* p, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t atom_add_u64(uint64_t* p, uint64_t v) {
  uint64_t old;
  asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v) : "memory");
  return old;
}
__device__ __forceinline__ void bulk_store_1d(void* gmem_dst, const void* smem_src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(gmem_dst), "r"(tmpl::smem_u32(smem_src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void spin_until(const uint32_t* p, uint32_t target) {
  while (ld_acquire_u32(p) < target) { __nanosleep(20); }
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Grid sync among one warp group of every CTA: counter target = SMs x use.
template <uint32_t BarId, int GroupThreads>
__device__ __forceinline__ void grid_sync(uint32_t* counter, uint32_t target) {
  tmpl::named_barrier_sync(BarId, GroupThreads);
  if (threadIdx.x % GroupThreads == 0) {
    __threadfence();
    red_release_add(counter, 1);
    spin_until(counter, target);
  }
  tmpl::named_barrier_sync(BarId, GroupThreads);
}

// ------------------------------------------------------------- scheduler
//
// Upstream's `get_num_l1_warmup_waves`: the number of L1-only waves before
// L1/L2 interleaving is safe. See the header for the argument.
__host__ __device__ constexpr int ceil_div_i(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ constexpr int num_l1_warmup_waves(int total_m_blocks, int num_sms, int l1_n, int l2_n) {
  const int first_l2_wave_m_blocks = ceil_div_i(num_sms, l2_n);
  const int for_first_l2_wave = ceil_div_i(first_l2_wave_m_blocks * l1_n, num_sms);
  const int diff = l1_n > l2_n ? l1_n - l2_n : 0;
  const int for_interleave = ceil_div_i(l1_n + (total_m_blocks - 1) * diff, num_sms) + 1;
  return for_first_l2_wave > for_interleave ? for_first_l2_wave : for_interleave;
}
// Upstream's `get_num_max_live_pool_blocks`: the ring capacity the schedule
// can have in flight, so the dispatch never blocks a slot the L1 warmup needs.
__host__ __device__ constexpr int max_live_pool_blocks(int total_m_blocks, int num_sms, int l1_n, int l2_n) {
  const int l1_tasks = total_m_blocks * l1_n;
  const int l1_waves = ceil_div_i(l1_tasks, num_sms);
  const int warm = num_l1_warmup_waves(total_m_blocks, num_sms, l1_n, l2_n);
  const int warm_waves = warm < l1_waves ? warm : l1_waves;
  const int warm_tasks = warm_waves * num_sms < l1_tasks ? warm_waves * num_sms : l1_tasks;
  const int live_after_warmup = ceil_div_i(warm_tasks, l1_n);
  const int growth = l2_n > l1_n ? ceil_div_i(total_m_blocks * (l2_n - l1_n), l2_n) : 0;
  const int margin = ceil_div_i(num_sms, l1_n < l2_n ? l1_n : l2_n);
  const int est = live_after_warmup + growth + margin;
  return est < total_m_blocks ? est : total_m_blocks;
}

// Per-expert token counts live one per lane; everything the scheduler and the
// dispatch need (pool block offsets, the owner of a pool block) is a warp scan.
struct ExpertCounts {
  uint32_t mine = 0;   // tokens of expert `lane`
  __device__ void fetch(const Workspace& ws, uint32_t num_sms) {
    const int lane = threadIdx.x % 32;
    uint64_t v = 0;
    if (lane < kNumExperts) {
      do { v = ld_volatile_u64(ws.expert_recv_sum + lane); } while (static_cast<uint32_t>(v >> 32) != num_sms);
    }
    mine = static_cast<uint32_t>(v);
    __syncwarp();
  }
  __device__ uint32_t blocks() const { return (mine + kBlockM - 1) / kBlockM; }
  __device__ uint32_t inclusive_blocks() const {
    uint32_t b = blocks();
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t o = __shfl_up_sync(0xffffffffu, b, off);
      if ((threadIdx.x % 32) >= off) { b += o; }
    }
    return b;
  }
  __device__ uint32_t total_blocks() const { return __shfl_sync(0xffffffffu, inclusive_blocks(), 31); }
  __device__ uint32_t tokens_of(int e) const { return __shfl_sync(0xffffffffu, mine, e); }
  __device__ uint32_t block_offset_of(int e) const {
    const uint32_t incl = inclusive_blocks();
    return __shfl_sync(0xffffffffu, incl - blocks(), e);
  }
  // Which expert owns pool block `pb`; also its block index within the expert.
  __device__ void owner(uint32_t pb, uint32_t& expert, uint32_t& m_block, uint32_t& valid_m) const {
    const int lane = threadIdx.x % 32;
    const uint32_t incl = inclusive_blocks();
    const uint32_t lo = incl - blocks();
    const bool is_owner = lane < kNumExperts && pb >= lo && pb < incl;
    const uint32_t mask = __ballot_sync(0xffffffffu, is_owner);
    expert = __ffs(mask) - 1;
    m_block = pb - __shfl_sync(0xffffffffu, lo, expert);
    const uint32_t tok = tokens_of(expert);
    valid_m = min(tok - m_block * kBlockM, static_cast<uint32_t>(kBlockM));
  }
};

}  // namespace

// ================================================================= kernel

__global__ __launch_bounds__(kNumThreads, 1) void mega_moe_sm90_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  auto* const sw = reinterpret_cast<Element(*)[kBlockN * kBlockK]>(smem + kOffW);
  auto* const st = reinterpret_cast<Element(*)[kBlockM * kBlockK]>(smem + kOffT);
  auto* const epi = reinterpret_cast<__nv_bfloat16*>(smem + kOffEpi);
  uint64_t* const full = reinterpret_cast<uint64_t*>(smem + kOffBar);
  uint64_t* const empty = full + kStages;
  uint64_t* const task_full = empty + kStages;     // [2]
  uint64_t* const task_empty = task_full + 2;      // [2]
  uint64_t* const pull_bar = task_empty + 2;       // [kDispatchWarps]
  TaskInfo* const task_infos = reinterpret_cast<TaskInfo*>(smem + kOffTask);
  uint32_t* const expert_count = reinterpret_cast<uint32_t*>(smem + kOffCounts);

  const Workspace& ws = p.ws;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t sm = blockIdx.x, num_sms = gridDim.x;
  const uint32_t ring_blocks = ws.ring_blocks;
  const uint32_t ring_tokens = ring_blocks * kBlockM;

  if (warp == 0) {
    if (lane == 0) {
      for (int s = 0; s < kStages; ++s) {
        tmpl::mbarrier_init(&full[s], 2);                  // token warp + weight warp
        tmpl::mbarrier_init(&empty[s], kMathWarpgroups);   // one elected lane per warpgroup
      }
      for (int s = 0; s < 2; ++s) {
        tmpl::mbarrier_init(&task_full[s], 1);
        tmpl::mbarrier_init(&task_empty[s], 2 + kMathWarpgroups);
      }
      for (int w = 0; w < kDispatchWarps; ++w) { tmpl::mbarrier_init(&pull_bar[w], 1); }
      tmpl::fence_barrier_init();
    }
    for (int e = lane; e < kNumExperts; e += 32) { expert_count[e] = 0; }
  }
  tmpl::fence_proxy_async_shared();
  __syncthreads();

  // ============================================================ dispatch
  if (warp < kDispatchWarps) {
    tmpl::setmaxnreg_dec<kDispatchRegs>();
    const int tokens_per_warp = 32 / kTopk;
    const int active_lanes = tokens_per_warp * kTopk;
    // Phase 1: this SM's share of tokens, counted per expert.
    auto each_topk = [&](auto&& fn) {
      for (int t = (sm * kDispatchWarps + warp) * tokens_per_warp; t < p.num_tokens;
           t += num_sms * kDispatchWarps * tokens_per_warp) {
        const int tok = t + lane / kTopk;
        if (lane < active_lanes && tok < p.num_tokens) {
          const int e = static_cast<int>(p.topk_idx[static_cast<int64_t>(tok) * kTopk + lane % kTopk]);
          if (e >= 0) { fn(tok * kTopk + lane % kTopk, e); }
        }
        __syncwarp();
      }
    };
    each_topk([&](int, int e) { atomicAdd_block(expert_count + e, 1u); });
    tmpl::named_barrier_sync(kBarDispatch, kDispatchThreads);
    // One atomic per expert per SM: the low word hands this SM a slot range,
    // the high word counts the SMs that have reported.
    for (int e = tid; e < kNumExperts; e += kDispatchThreads) {
      const uint64_t v = (1ull << 32) | static_cast<uint64_t>(expert_count[e]);
      expert_count[e] = static_cast<uint32_t>(atom_add_u64(ws.expert_send_count + e, v));
    }
    tmpl::named_barrier_sync(kBarDispatch, kDispatchThreads);
    each_topk([&](int token_topk, int e) {
      const uint32_t slot = atomicAdd_block(expert_count + e, 1u);
      ws.src_token_topk[static_cast<uint64_t>(e) * ws.max_recv + slot] = static_cast<uint32_t>(token_topk);
    });
    grid_sync<kBarDispatch, kDispatchThreads>(ws.grid_sync + 0, num_sms);
    if (sm == 0) {
      // "Recv" counts: on one rank, the send counts, republished with the
      // SM tally in the high word so the waiters below can tell "complete".
      for (int e = tid; e < kNumExperts; e += kDispatchThreads) {
        atom_add_u64(ws.expert_recv_sum + e, ld_volatile_u64(ws.expert_send_count + e));
      }
    }
    // Phase 2: pull tokens into pool order.
    ExpertCounts counts;
    counts.fetch(ws, num_sms);
    const uint32_t total_blocks = counts.total_blocks();
    uint32_t pull_phase = 0;
    uint8_t* bounce = smem + kOffPull + warp * kPullChunk;
    uint64_t* bar = &pull_bar[warp];
    // Pool tokens, expert-major; each warp walks a strided subset.
    for (uint32_t pb = 0; pb < total_blocks; ++pb) {
      uint32_t expert, m_block, valid_m;
      counts.owner(pb, expert, m_block, valid_m);
      const uint32_t expert_tokens = counts.tokens_of(expert);
      for (uint32_t i = sm * kDispatchWarps + warp; i < valid_m; i += num_sms * kDispatchWarps) {
        const uint32_t idx_in_expert = m_block * kBlockM + i;
        const uint32_t src = ws.src_token_topk[static_cast<uint64_t>(expert) * ws.max_recv + idx_in_expert];
        const uint32_t src_token = src / kTopk, src_k = src % kTopk;
        const uint32_t pool_token = pb * kBlockM + i;
        const uint32_t ring_block = pb % ring_blocks, ring_token = pool_token % ring_tokens;
        // The slot's previous occupant must have been consumed by all its L1 tasks.
        if (lane == 0) { spin_until(ws.l1_empty + ring_block, (pb / ring_blocks) * kNumL1Blocks); }
        __syncwarp();
        // Bounce through shared memory, as the NVLink pull does: bulk load,
        // bulk store, one chunk at a time.
        if (lane == 0) {
          const uint8_t* s = reinterpret_cast<const uint8_t*>(p.x + static_cast<int64_t>(src_token) * kHidden);
          uint8_t* d = reinterpret_cast<uint8_t*>(ws.l1_tokens + static_cast<int64_t>(ring_token) * kHidden);
          #pragma unroll 1
          for (int c = 0; c < kHidden * 2 / kPullChunk; ++c) {
            tmpl::arrive_and_expect_tx(bar, kPullChunk);
            tmpl::bulk_load_1d(bounce, s + c * kPullChunk, kPullChunk, bar);
            tmpl::wait_parity(bar, pull_phase);
            pull_phase ^= 1;
            tmpl::fence_proxy_async_shared();
            bulk_store_1d(d + c * kPullChunk, bounce, kPullChunk);
            bulk_commit();
            bulk_wait_all();
          }
          ws.l1_topk_w[ring_token] = p.topk_w[static_cast<int64_t>(src_token) * kTopk + src_k];
          ws.token_src[pool_token] = TokenSrc{src_token, src_k};
          const bool last = (idx_in_expert == expert_tokens - 1);
          // Padding rows count as arrived so the block fills to BLOCK_M.
          red_release_add(ws.l1_full + ring_block, last ? kBlockM - (idx_in_expert % kBlockM) : 1u);
        }
        __syncwarp();
      }
    }
    return;
  }

  // ============================================================ producers
  if (warp < kFirstMathWarp) {
    tmpl::setmaxnreg_dec<kProducerRegs>();
    if (warp == kWarpScheduler) {
      ExpertCounts counts;
      counts.fetch(ws, num_sms);
      const uint32_t total_blocks = counts.total_blocks();
      const uint32_t total_l1 = total_blocks * kNumL1Blocks, total_l2 = total_blocks * kNumL2Blocks;
      const uint32_t l1_waves = (total_l1 + num_sms - 1) / num_sms;
      const uint32_t warm = static_cast<uint32_t>(num_l1_warmup_waves(total_blocks, num_sms, kNumL1Blocks, kNumL2Blocks));
      constexpr uint32_t kWavesDone = 0xffffffffu;
      uint32_t sched_l1_waves = min(warm, l1_waves);
      uint32_t stage = 0, phase = 0;
      auto publish = [&](const TaskInfo& ti) {
        tmpl::wait_parity(&task_empty[stage], phase ^ 1u);
        if (lane == 0) {
          task_infos[stage] = ti;
          __threadfence_block();
          tmpl::mbarrier_arrive(&task_full[stage]);
        }
        __syncwarp();
        phase ^= (stage == 1);
        stage ^= 1;
      };
      auto claim = [&](uint32_t* counter) {
        uint32_t v = 0;
        if (lane == 0) { v = atomicAdd(counter, 1u); }
        return __shfl_sync(0xffffffffu, v, 0);
      };
      auto make_task = [&](Phase ph, uint32_t idx, uint32_t n_blocks, uint32_t shape_k) {
        TaskInfo ti{};
        ti.phase = static_cast<uint32_t>(ph);
        ti.pool_block = idx / n_blocks;
        ti.n_block = idx % n_blocks;
        counts.owner(ti.pool_block, ti.expert, ti.m_block, ti.valid_m);
        ti.shape_k = shape_k;
        return ti;
      };
      while (true) {
        if (sched_l1_waves != kWavesDone && sched_l1_waves) {
          --sched_l1_waves;
          const uint32_t idx = claim(ws.task_count + 0);
          if (idx >= total_l1) { sched_l1_waves = kWavesDone; continue; }
          publish(make_task(Phase::kLinear1, idx, kNumL1Blocks, kL1ShapeK));
        } else {
          const uint32_t idx = claim(ws.task_count + 1);
          if (idx >= total_l2) { break; }
          if (sched_l1_waves != kWavesDone) { sched_l1_waves = 1; }  // next one is an L1 task
          const TaskInfo ti = make_task(Phase::kLinear2, idx, kNumL2Blocks, kL2ShapeK);
          // Every L1 task of this block must have been CLAIMED before an L2
          // task for it occupies an SM.
          if (lane == 0) { while (ld_volatile_u32(ws.task_count + 0) < (ti.pool_block + 1) * kNumL1Blocks) {} }
          __syncwarp();
          publish(ti);
        }
      }
      publish(TaskInfo{});  // sentinel
    } else if (warp == kWarpTokenTma || warp == kWarpWeightTma) {
      const bool tokens = (warp == kWarpTokenTma);
      uint32_t tstage = 0, tphase = 0;   // task pipeline
      uint32_t g = 0;                    // K blocks issued, for the stage ring
      while (true) {
        tmpl::wait_parity(&task_full[tstage], tphase);
        const TaskInfo ti = task_infos[tstage];
        __syncwarp();
        if (lane == 0) { tmpl::mbarrier_arrive(&task_empty[tstage]); }
        tphase ^= (tstage == 1); tstage ^= 1;
        if (ti.phase == static_cast<uint32_t>(Phase::kNone)) { break; }
        const uint32_t ring_block = ti.pool_block % ring_blocks;
        const uint32_t use = ti.pool_block / ring_blocks;
        const uint32_t k_blocks = ti.shape_k / kBlockK;
        if (tokens && lane == 0) {
          // The block's input: tokens landed (L1) or all 32 L1 outputs (L2).
          if (ti.phase == static_cast<uint32_t>(Phase::kLinear1)) {
            spin_until(ws.l1_full + ring_block, kBlockM * (use + 1));
          } else {
            spin_until(ws.l2_full + ring_block, (kL2ShapeK / kL1OutBlockN) * (use + 1));
          }
        }
        __syncwarp();
        for (uint32_t k = 0; k < k_blocks; ++k, ++g) {
          const uint32_t s = g % kStages, u = g / kStages;
          if (u > 0) { tmpl::wait_parity(&empty[s], (u - 1) & 1u); }
          if (lane == 0) {
            if (tokens) {
              const CUtensorMap* map = ti.phase == static_cast<uint32_t>(Phase::kLinear1) ? &p.l1_tokens : &p.l2_tokens;
              tmpl::arrive_and_expect_tx(&full[s], kTBytes);
              tmpl::tma_load_2d(map, st[s], k * kBlockK, ring_block * kBlockM, &full[s]);
            } else {
              const CUtensorMap* map = ti.phase == static_cast<uint32_t>(Phase::kLinear1) ? &p.l1_w : &p.l2_w;
              tmpl::arrive_and_expect_tx(&full[s], kWBytes);
              tmpl::tma_load_3d(map, sw[s], k * kBlockK, ti.n_block * kBlockN, ti.expert, &full[s]);
            }
          }
          __syncwarp();
        }
      }
      // Teardown: every stage's last release must land before the CTA exits.
      for (uint32_t s = 0; s < kStages; ++s) {
        const uint32_t uses = g / kStages + (s < g % kStages ? 1u : 0u);
        if (uses > 0) { tmpl::wait_parity(&empty[s], (uses - 1) & 1u); }
      }
    }
    return;
  }

  // ================================================================ math
  tmpl::setmaxnreg_inc<kMathRegs>();
  const int mtid = tid - kFirstMathWarp * 32;   // 0..255
  const int wg = mtid / 128;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(mtid);
  auto acc = partition_fragment_C(tiled_mma, Shape<Int<kBlockN>, Int<kBlockM>>{});
  // (row, token) of every accumulator element, for the epilogues.
  auto coords = thr_mma.partition_C(make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{}));

  uint32_t tstage = 0, tphase = 0, g = 0;
  while (true) {
    tmpl::wait_parity(&task_full[tstage], tphase);
    const TaskInfo ti = task_infos[tstage];
    tmpl::named_barrier_sync(kBarMathWg0 + wg, 128);
    if (mtid % 128 == 0) { tmpl::mbarrier_arrive(&task_empty[tstage]); }
    tphase ^= (tstage == 1); tstage ^= 1;
    if (ti.phase == static_cast<uint32_t>(Phase::kNone)) { break; }
    const bool is_l1 = ti.phase == static_cast<uint32_t>(Phase::kLinear1);
    const uint32_t ring_block = ti.pool_block % ring_blocks;
    const uint32_t use = ti.pool_block / ring_blocks;
    const uint32_t k_blocks = ti.shape_k / kBlockK;

    clear(acc);
    for (uint32_t k = 0; k < k_blocks; ++k, ++g) {
      const uint32_t s = g % kStages, u = g / kStages;
      tmpl::wait_parity(&full[s], u & 1u);
      Tensor tw = make_tensor(make_smem_ptr(sw[s]), SmemLayoutW{});
      Tensor tt = make_tensor(make_smem_ptr(st[s]), SmemLayoutT{});
      auto frag_w = thr_mma.make_fragment_A(thr_mma.partition_A(tw));
      auto frag_t = thr_mma.make_fragment_B(thr_mma.partition_B(tt));
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      tiled_mma.accumulate_ = (k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
      CUTE_UNROLL
      for (int kb = 0; kb < size<2>(frag_w); ++kb) {
        cute::gemm(tiled_mma, frag_w(_, _, kb), frag_t(_, _, kb), acc);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      tmpl::named_barrier_sync(kBarMathWg0 + wg, 128);
      if (mtid % 128 == 0) { tmpl::mbarrier_arrive(&empty[s]); }
    }

    if (is_l1) {
      // The L2 input slot this block writes must be free of its previous use.
      if (mtid == 0) { spin_until(ws.l2_empty + ring_block, kNumL2Blocks * use); }
      tmpl::named_barrier_sync(kBarMath, kMathThreads);
      // SwiGLU on (gate = row r, up = row r + 8) pairs in registers, times the
      // token's top-k weight, into the [token][feature] transpose tile. In the
      // wgmma C fragment, element i + 2 is the same column 8 rows down, which
      // the coordinate tensor confirms (a mismatch traps: the interleave would
      // otherwise silently pair the wrong rows).
      CUTE_UNROLL
      for (int i = 0; i < size(acc); i += 4) {
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
          const int row = get<0>(coords(i + c)), tok = get<1>(coords(i + c));
          if (get<0>(coords(i + c + 2)) != row + 8 || get<1>(coords(i + c + 2)) != tok) { asm volatile("trap;"); }
          const float w = ws.l1_topk_w[ring_block * kBlockM + tok];
          const int feat = (row / 16) * 8 + row % 8;
          epi[tok * kBlockN + feat] = __float2bfloat16(silu(acc(i + c)) * acc(i + c + 2) * w);
        }
      }
      tmpl::named_barrier_sync(kBarMath, kMathThreads);
      // Rows of 64 features (128 B) into l2_tokens[ring token][n_block * 64].
      for (int i = mtid; i < kBlockM * 8; i += kMathThreads) {
        const int tok = i / 8, chunk = i % 8;
        if (static_cast<uint32_t>(tok) < ti.valid_m) {
          *reinterpret_cast<uint4*>(ws.l2_tokens + static_cast<int64_t>(ring_block * kBlockM + tok) * kInter +
                                    ti.n_block * kL1OutBlockN + chunk * 8) =
              *reinterpret_cast<const uint4*>(epi + tok * kBlockN + chunk * 8);
        }
      }
      __threadfence();
      tmpl::named_barrier_sync(kBarMath, kMathThreads);
      if (mtid == 0) {
        red_release_add(ws.l2_full + ring_block, 1u);
        red_add(ws.l1_empty + ring_block, 1u);
      }
    } else {
      // L2: [128 hidden][tokens] -> the combine buffer's (token, slot) rows.
      if (mtid == 0) { red_add(ws.l2_empty + ring_block, 1u); }
      CUTE_UNROLL
      for (int i = 0; i < size(acc); ++i) {
        const int row = get<0>(coords(i)), tok = get<1>(coords(i));
        epi[tok * kBlockN + row] = __float2bfloat16(acc(i));
      }
      tmpl::named_barrier_sync(kBarMath, kMathThreads);
      for (int i = mtid; i < kBlockM * 16; i += kMathThreads) {
        const int tok = i / 16, chunk = i % 16;
        if (static_cast<uint32_t>(tok) < ti.valid_m) {
          const TokenSrc src = ws.token_src[ti.pool_block * kBlockM + tok];
          __nv_bfloat16* dst = ws.combine + (static_cast<int64_t>(src.topk) * p.num_tokens + src.token) * kHidden +
                               ti.n_block * kBlockN + chunk * 8;
          *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(epi + tok * kBlockN + chunk * 8);
        }
      }
      tmpl::named_barrier_sync(kBarMath, kMathThreads);
    }
  }

  // ============================================================== combine
  // Every L2 output of every SM must have landed: grid sync over the math
  // groups, then each warp sums the top-k slots of its tokens.
  grid_sync<kBarMath, kMathThreads>(ws.grid_sync + 1, num_sms);
  const int mwarp = (mtid / 32);
  for (int tok = sm * kMathWarps + mwarp; tok < p.num_tokens; tok += num_sms * kMathWarps) {
    for (int c = lane; c < kHidden / 8; c += 32) {
      float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const uint4 raw = *reinterpret_cast<const uint4*>(
            ws.combine + (static_cast<int64_t>(k) * p.num_tokens + tok) * kHidden + c * 8);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
        #pragma unroll
        for (int e = 0; e < 8; ++e) { sum[e] += __bfloat162float(v[e]); }
      }
      uint4 out;
      __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
      #pragma unroll
      for (int e = 0; e < 8; ++e) { o[e] = __float2bfloat16(sum[e]); }
      *reinterpret_cast<uint4*>(p.y + static_cast<int64_t>(tok) * kHidden + c * 8) = out;
    }
  }
}

// ================================================================= harness

#ifndef MK44_NO_MAIN

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CKD(x)                                                                 \
  do {                                                                         \
    CUresult r = (x);                                                          \
    if (r != CUDA_SUCCESS) { printf("CU error %d at line %d\n", int(r), __LINE__); exit(1); } \
  } while (0)

namespace {

float r_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }
float host_rand(uint32_t& s, float scale) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.f - 0.5f) * 2.f * scale;
}
std::vector<float> rand_vec(uint32_t& s, size_t n, float scale) {
  std::vector<float> v(n);
  for (auto& e : v) e = r_bf16(host_rand(s, scale));
  return v;
}
__nv_bfloat16* upload(const std::vector<float>& h) {
  std::vector<__nv_bfloat16> t(h.size());
  for (size_t i = 0; i < h.size(); ++i) t[i] = __float2bfloat16(h[i]);
  __nv_bfloat16* d = nullptr;
  CK(cudaMalloc(&d, t.size() * 2));
  CK(cudaMemcpy(d, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
  return d;
}
template <class T>
T* upload_raw(const std::vector<T>& h) {
  T* d = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
T* alloc_zero(size_t n) {
  T* d = nullptr;
  CK(cudaMalloc(&d, n * sizeof(T)));
  CK(cudaMemset(d, 0, n * sizeof(T)));
  return d;
}

void make_map_3d(CUtensorMap* map, const __nv_bfloat16* base, uint64_t cols, uint64_t rows, uint64_t experts,
                 uint32_t box_rows) {
  uint64_t dims[3] = {cols, rows, experts};
  uint64_t strides[2] = {cols * 2, cols * rows * 2};
  uint32_t box[3] = {kBlockK, box_rows, 1};
  uint32_t estr[3] = {1, 1, 1};
  CKD(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(base), dims, strides,
                             box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
void make_map_2d(CUtensorMap* map, const __nv_bfloat16* base, uint64_t cols, uint64_t rows, uint32_t box_rows) {
  CKD(tmpl::encode_tile_map_2d(map, base, cols, rows, kBlockK, box_rows, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
}

// Reference in double with the device's roundings: the L1 output after
// SwiGLU x top-k weight is bf16, each L2 output row is bf16, the sum is bf16.
void reference(const std::vector<float>& x, const std::vector<int64_t>& topk_idx, const std::vector<float>& topk_w,
               const std::vector<float>& l1, const std::vector<float>& l2, std::vector<float>& y) {
  y.assign(size_t(kNumTokens) * kHidden, 0.f);
  std::vector<double> h(kInter), acc(kHidden);
  for (int t = 0; t < kNumTokens; ++t) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int k = 0; k < kTopk; ++k) {
      const int e = int(topk_idx[size_t(t) * kTopk + k]);
      if (e < 0) continue;
      const double w = topk_w[size_t(t) * kTopk + k];
      const float* W1 = l1.data() + size_t(e) * kL1ShapeN * kHidden;
      for (int f = 0; f < kInter; ++f) {
        // Interleaved rows: gate at 16 * (f / 8) + f % 8, up 8 rows later.
        const float* g = W1 + (size_t(16 * (f / 8) + f % 8)) * kHidden;
        const float* u = g + size_t(8) * kHidden;
        double sg = 0.0, su = 0.0;
        for (int c = 0; c < kHidden; ++c) {
          sg += double(g[c]) * x[size_t(t) * kHidden + c];
          su += double(u[c]) * x[size_t(t) * kHidden + c];
        }
        h[f] = r_bf16(float((sg / (1.0 + std::exp(-sg))) * su * w));
      }
      const float* W2 = l2.data() + size_t(e) * kHidden * kInter;
      for (int n = 0; n < kHidden; ++n) {
        double s = 0.0;
        for (int f = 0; f < kInter; ++f) s += double(W2[size_t(n) * kInter + f]) * h[f];
        acc[n] += r_bf16(float(s));
      }
    }
    for (int n = 0; n < kHidden; ++n) y[size_t(t) * kHidden + n] = r_bf16(float(acc[n]));
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = (argc > 1 && std::string(argv[1]) == "quick");
  int sm_count = 0;
  CK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaFree(nullptr));
  printf("Mega MoE sm90  (%d SMs; %d experts, hidden %d, inter %d, top-%d, %d tokens, BLOCK_M %d, %d stages)\n",
         sm_count, kNumExperts, kHidden, kInter, kTopk, kNumTokens, kBlockM, kStages);

  // ---- host data
  uint32_t seed = 4242u;
  std::vector<float> hx = rand_vec(seed, size_t(kNumTokens) * kHidden, 1.f);
  std::vector<float> hl1 = rand_vec(seed, size_t(kNumExperts) * kL1ShapeN * kHidden, 1.f / std::sqrt(float(kHidden)));
  std::vector<float> hl2 = rand_vec(seed, size_t(kNumExperts) * kHidden * kInter, 1.f / std::sqrt(float(kInter)));
  std::vector<int64_t> hidx(size_t(kNumTokens) * kTopk);
  std::vector<float> hw(size_t(kNumTokens) * kTopk);
  for (int t = 0; t < kNumTokens; ++t) {
    // top-k distinct experts per token, weights positive and normalised.
    std::vector<int> perm(kNumExperts);
    for (int e = 0; e < kNumExperts; ++e) perm[e] = e;
    for (int e = kNumExperts - 1; e > 0; --e) {
      seed = seed * 1664525u + 1013904223u;
      std::swap(perm[e], perm[(seed >> 8) % (e + 1)]);
    }
    float sum = 0.f;
    for (int k = 0; k < kTopk; ++k) {
      hidx[size_t(t) * kTopk + k] = perm[k];
      hw[size_t(t) * kTopk + k] = 0.5f + std::fabs(host_rand(seed, 0.5f));
      sum += hw[size_t(t) * kTopk + k];
    }
    for (int k = 0; k < kTopk; ++k) hw[size_t(t) * kTopk + k] /= sum;
  }
  std::vector<int> tokens_per_expert(kNumExperts, 0);
  for (auto e : hidx) tokens_per_expert[e]++;
  int total_blocks = 0, max_recv = 0;
  for (int e = 0; e < kNumExperts; ++e) {
    total_blocks += (tokens_per_expert[e] + kBlockM - 1) / kBlockM;
    max_recv = std::max(max_recv, tokens_per_expert[e]);
  }
  const int ring_blocks = max_live_pool_blocks(total_blocks, sm_count, kNumL1Blocks, kNumL2Blocks);
  const int ring_tokens = ring_blocks * kBlockM;
  printf("  pool: %d blocks of %d over %d experts (min %d, max %d tokens); ring %d blocks; warmup %d L1 waves\n",
         total_blocks, kBlockM, kNumExperts,
         *std::min_element(tokens_per_expert.begin(), tokens_per_expert.end()), max_recv, ring_blocks,
         num_l1_warmup_waves(total_blocks, sm_count, kNumL1Blocks, kNumL2Blocks));

  // ---- device
  Params p{};
  p.x = upload(hx);
  p.topk_idx = upload_raw(hidx);
  p.topk_w = upload_raw(hw);
  p.y = alloc_zero<__nv_bfloat16>(size_t(kNumTokens) * kHidden);
  __nv_bfloat16* d_l1 = upload(hl1);
  __nv_bfloat16* d_l2 = upload(hl2);
  Workspace& ws = p.ws;
  ws.grid_sync = alloc_zero<uint32_t>(4);
  ws.task_count = alloc_zero<uint32_t>(4);
  ws.expert_send_count = alloc_zero<uint64_t>(kNumExperts);
  ws.expert_recv_sum = alloc_zero<uint64_t>(kNumExperts);
  ws.l1_full = alloc_zero<uint32_t>(ring_blocks);
  ws.l1_empty = alloc_zero<uint32_t>(ring_blocks);
  ws.l2_full = alloc_zero<uint32_t>(ring_blocks);
  ws.l2_empty = alloc_zero<uint32_t>(ring_blocks);
  ws.max_recv = static_cast<uint32_t>(std::max(max_recv, 1));
  ws.src_token_topk = alloc_zero<uint32_t>(size_t(kNumExperts) * ws.max_recv);
  ws.token_src = alloc_zero<TokenSrc>(size_t(total_blocks) * kBlockM);
  ws.l1_topk_w = alloc_zero<float>(ring_tokens);
  ws.l1_tokens = alloc_zero<__nv_bfloat16>(size_t(ring_tokens) * kHidden);
  ws.l2_tokens = alloc_zero<__nv_bfloat16>(size_t(ring_tokens) * kInter);
  ws.combine = alloc_zero<__nv_bfloat16>(size_t(kTopk) * kNumTokens * kHidden);
  ws.ring_blocks = ring_blocks;
  make_map_3d(&p.l1_w, d_l1, kHidden, kL1ShapeN, kNumExperts, kBlockN);
  make_map_3d(&p.l2_w, d_l2, kInter, kHidden, kNumExperts, kBlockN);
  make_map_2d(&p.l1_tokens, ws.l1_tokens, kHidden, ring_tokens, kBlockM);
  make_map_2d(&p.l2_tokens, ws.l2_tokens, kInter, ring_tokens, kBlockM);
  p.num_tokens = kNumTokens;
  p.num_sms = sm_count;

  CK(cudaFuncSetAttribute(mega_moe_sm90_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  auto run = [&](cudaStream_t s) {
    // Counters are reset per launch; upstream's dispatch warps do this at the
    // end of the kernel instead.
    CK(cudaMemsetAsync(ws.grid_sync, 0, 16, s));
    CK(cudaMemsetAsync(ws.task_count, 0, 16, s));
    CK(cudaMemsetAsync(ws.expert_send_count, 0, kNumExperts * 8, s));
    CK(cudaMemsetAsync(ws.expert_recv_sum, 0, kNumExperts * 8, s));
    CK(cudaMemsetAsync(ws.l1_full, 0, ring_blocks * 4, s));
    CK(cudaMemsetAsync(ws.l1_empty, 0, ring_blocks * 4, s));
    CK(cudaMemsetAsync(ws.l2_full, 0, ring_blocks * 4, s));
    CK(cudaMemsetAsync(ws.l2_empty, 0, ring_blocks * 4, s));
    mega_moe_sm90_kernel<<<sm_count, kNumThreads, kSmemBytes, s>>>(p);
  };

  // ---- correctness
  printf("  reference (double) ...\n");
  fflush(stdout);
  std::vector<float> ref;
  reference(hx, hidx, hw, hl1, hl2, ref);
  run(stream);
  CK(cudaStreamSynchronize(stream));
  CK(cudaGetLastError());
  std::vector<__nv_bfloat16> got(size_t(kNumTokens) * kHidden);
  CK(cudaMemcpy(got.data(), p.y, got.size() * 2, cudaMemcpyDeviceToHost));
  // bf16 rounding noise (the L1 output, each L2 row, the top-k sum) puts the
  // worst of a few million elements a few percent of the rms off; a wrong
  // block or slot puts many elements a quarter of the rms off. Gate on the
  // mean and on that count, report the max.
  double rms = 0.0, worst = 0.0, mean = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) rms += double(ref[i]) * ref[i];
  rms = std::sqrt(rms / ref.size()) + 1e-12;
  size_t gross = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(__bfloat162float(got[i])) - ref[i]);
    worst = std::max(worst, d); mean += d;
    if (d > 0.25 * rms) ++gross;
  }
  mean /= ref.size();
  const bool ok = mean / rms < 5e-3 && gross == 0 && worst / rms < 2e-1;
  printf("  y   max err / rms %.3e  mean err / rms %.3e  elements > rms/4: %zu  %s\n", worst / rms, mean / rms, gross,
         ok ? "PASS" : "FAIL");
  if (!ok) { printf("numerics failed; timings withheld\n"); return 1; }
  if (quick) return 0;

  // ---- timing
  // Weights are streamed once per POOL BLOCK, not once per expert: an expert
  // whose tokens span two blocks of BLOCK_M is read twice, which is why
  // upstream JIT-compiles the largest BLOCK_M that keeps most experts in one
  // block. Both counts are reported; the floor uses the streamed one.
  const double expert_bytes = double(kL1ShapeN) * kHidden * 2 + double(kHidden) * kInter * 2;
  double unique = 0.0, bytes = 0.0;
  for (int e = 0; e < kNumExperts; ++e) {
    if (tokens_per_expert[e] > 0) {
      unique += expert_bytes;
      bytes += expert_bytes * ((tokens_per_expert[e] + kBlockM - 1) / kBlockM);
    }
  }
  bytes += double(kNumTokens) * kHidden * 2 * (2 + 2 * kTopk);  // x in, pool copy, combine out/in
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run(stream);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 3; ++r) {
    CK(cudaEventRecord(a, stream));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaEventRecord(b, stream));
    CK(cudaEventSynchronize(b));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms / 20.f * 1000.f);
  }
  const double floor_us = 1.85 + bytes / 2.77e6;
  const double flops = 2.0 * kNumTokens * kTopk * (double(kL1ShapeN) * kHidden + double(kHidden) * kInter);
  printf("\n  mega moe: %.1f us  (%.2f GB streamed, %.2f GB unique -> %.2f TB/s, %.0f TFLOP/s;\n"
         "            floor %.1f us [ld.bw.dev.dram] -> %.0f%% of floor; a TMA stream can reach [tma.bw.dev.dram])\n",
         best, bytes / 1e9, unique / 1e9, bytes / best / 1e6, flops / best / 1e6, floor_us, 100.0 * floor_us / best);
  printf("  (%d L1 tasks + %d L2 tasks over %d SMs)\n", total_blocks * kNumL1Blocks, total_blocks * kNumL2Blocks, sm_count);
  return 0;
}

#endif  // MK44_NO_MAIN
