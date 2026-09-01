// ffn_taskloop.cu -- Phase 5 of specs/tile/ffn_taskloop_minimal.md.
//
// Persistent task-loop prototype consuming an upstream K-major XFS buffer:
//   GatedUp task (128x): hidden[:, n:n+32] = gelu(XFS @ W1 + b1) * (XFS @ W2 + b2)
//   DownResidual task (32x):  out[:, n:n+32]   += ((hidden @ Wd)[:, n:n+32]) * g[n:n+32]
// with out pre-filled with the residual. GatedUp->DownResidual ordering runs through 32 gmem
// counters (one per 128-col slice of hidden, arrive count 4).
//
// Structure follows the CuTe conventions used by FlashMLA and DeepGEMM:
// typed SM90 geometry, explicit warp roles, named TMA/barrier helpers, and a
// small GEMM wrapper around the warpgroup fence/arrive/commit protocol. The
// task loop remains repository-specific: blockIdx.x indexes a preallocated
// per-CTA descriptor row, with no runtime scheduler or work queue.
//
// Build: nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -arch=sm_90a
//        --expt-relaxed-constexpr -I$CUTLASS_DIR/include ffn_taskloop.cu -lcuda

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdio>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "sm90_ffn_task_desc.cuh"
#include "sm90_ffn_barriers.cuh"
#include "sm90_ffn_gemm.cuh"
#include "sm90_ffn_warp_roles.cuh"
#include "sm90/helpers.h"

namespace ffn {

using namespace cute;
using BF = cutlass::bfloat16_t;
namespace ref = flash_vla::pi05::sm90::ffn;
using TaskDescriptor = ref::TaskDescriptor;
using TaskKind = ref::TaskKind;
using WarpRoles = ref::WarpRoles;

// ------------------------------------------------------------- spec constants
// spec: problem.dims / grid.cta_tile / mainloop.step
constexpr int M_PAD = 64;                 // cta_tile.M, pads M=50
constexpr int D  = 1024;                  // hidden width (GatedUp contraction)
constexpr int FF = 4096;                  // ffn width per branch (DownResidual contraction)
constexpr int BN = 32;                    // cta_tile.N
// Keep the two mainloop profiles independent. GatedUp consumes one 32 KiB
// BK256 activation box. DownResidual computes a BK128 stage; its hidden
// producer composes that stage from two legal SW128 BK64 TMA boxes.
constexpr int GATED_UP_BLOCK_K = 256;
constexpr int DOWN_RESIDUAL_BLOCK_K = 128;
constexpr int DOWN_RESIDUAL_HIDDEN_TMA_K = 64;
constexpr int GATED_UP_TRIP = D / GATED_UP_BLOCK_K;                 // 4
constexpr int DOWN_RESIDUAL_TRIP = FF / DOWN_RESIDUAL_BLOCK_K;      // 32
// Keep the complete [K=1024, M_PAD=64] XFS activation stationary in shared
// memory. Four fixed 32 KiB activation frames never rotate or get overwritten.
// Weights use a separate three-deep 32 KiB macro ring; each frame feeds one
// BK256/N64 WGMMA stage.
constexpr int GATED_UP_ACTIVATION_FRAMES = D / GATED_UP_BLOCK_K; // 4
constexpr int GATED_UP_WEIGHT_DEPTH = 3;
// Preserve PR3's non-draining policy: one committed group stays outstanding.
// A BK=256 stage contains sixteen m64n64k16 operations, so this experiment
// intentionally changes the per-group instruction count but not wait depth.
constexpr int GATED_UP_WGMMA_WAIT = 1;
constexpr int DOWN_RESIDUAL_WEIGHT_DEPTH = 4;              // DownResidual weight ring, dep-free
constexpr int DOWN_RESIDUAL_ACTIVATION_DEPTH = 4;          // DownResidual activation ring, counter-gated
// Split-K on DownResidual's FF contraction. The copy column is `txns_per_warp x 248 ns`
// [hardware-unit-test tma.issue.warp], and txns_per_warp = K_per_CTA / BK -- so
// splitting K is one of only three levers that divides a copy floor, and the
// only one available here (more CTAs does NOT move it; every CTA still walks
// its own K). S=4 takes DownResidual from 32 stages to 8 and places the TMA
// issue column below the DRAM wall. Larger S adds partial traffic without
// reducing the memory floor.
constexpr int DOWN_RESIDUAL_SPLIT   = 4;
constexpr int DOWN_RESIDUAL_K_SPAN  = FF / DOWN_RESIDUAL_SPLIT;      // 1024 contraction rows per split
constexpr int NUM_DOWN_RESIDUAL_TILES = D / BN;             // 32 output tiles
constexpr int PARTIAL_ELEMS = M_PAD * BN;      // 2048 f32 per partial tile
constexpr int DOWN_RESIDUAL_TRIP_PER_SPLIT =
    DOWN_RESIDUAL_TRIP / DOWN_RESIDUAL_SPLIT;  // 8 stages per split at BK=128
constexpr int kNumEpilogueWarps = 4;      // math warpgroup consumes each stage
constexpr int MAX_TASKS_PER_CTA = 2;      // rows may end early with type = -1
constexpr int N_CTAS = 132;               // fixed H100 worker grid
constexpr int COUNTER_K = 128;             // dependency granularity in hidden K
constexpr int N_COUNTERS = FF / COUNTER_K; // one per 128-col hidden slice
constexpr int COUNTER_ARRIVE = COUNTER_K / BN; // 4 GatedUp tasks fill one slice
static_assert(D % GATED_UP_BLOCK_K == 0,
              "GatedUp BLOCK_K must divide the contraction");
static_assert(FF % DOWN_RESIDUAL_BLOCK_K == 0,
              "DownResidual BLOCK_K must divide the contraction");
static_assert(FF % COUNTER_K == 0 &&
                  COUNTER_K % DOWN_RESIDUAL_BLOCK_K == 0,
              "counter geometry must divide DownResidual BLOCK_K");
static_assert(WarpRoles::kThreads == 224,
              "the current ABI/profile is a 224-thread CTA");

// spec: pipeline.staged_buffers -- shared-memory pool offsets in bytes.
// The experimental [K, M_PAD] input makes M the 128-byte TMA row, so one
// BK256 box fills the complete activation frame.
constexpr int GATED_UP_ACTIVATION_FRAME_BYTES =
    M_PAD * GATED_UP_BLOCK_K * sizeof(BF);                       // 32768
constexpr int GATED_UP_WEIGHT_FRAME_BYTES =
    2 * BN * GATED_UP_BLOCK_K * sizeof(BF);                      // 32768
constexpr int DOWN_RESIDUAL_ACTIVATION_FRAME_BYTES =
    M_PAD * DOWN_RESIDUAL_BLOCK_K * sizeof(BF);                 // 16384
constexpr int DOWN_RESIDUAL_WEIGHT_FRAME_BYTES =
    BN * DOWN_RESIDUAL_BLOCK_K * sizeof(BF);                     // 8192
constexpr int DOWN_RESIDUAL_HIDDEN_TMA_BYTES =
    M_PAD * DOWN_RESIDUAL_HIDDEN_TMA_K * sizeof(BF);             // 8192

constexpr int GATED_UP_WEIGHT_OFFSET =
    GATED_UP_ACTIVATION_FRAMES * GATED_UP_ACTIVATION_FRAME_BYTES; // 131072
constexpr int BARRIER_OFFSET =
    GATED_UP_WEIGHT_OFFSET +
    GATED_UP_WEIGHT_DEPTH * GATED_UP_WEIGHT_FRAME_BYTES;          // 229376
constexpr int BARRIER_WORDS = 17;
constexpr int DOWN_RESIDUAL_STAGE4_PREFETCH_BARRIER = 16;
// The barrier pool is now producer-owned: no scheduler->TMA sequence arrays
// are needed.  Each producer warp waits on its empty slot, calls
// arrive_and_expect_tx for its own byte count, and emits the TMA operation.
constexpr int SHARED_MEMORY_BYTES =
    BARRIER_OFFSET + BARRIER_WORDS * sizeof(uint64_t);             // 229512

// The two task bodies never execute concurrently, so their data planes alias.
// Describe that alias once instead of rebuilding it with byte offsets at every
// TMA and WGMMA call site. The external dynamic-SMEM allocation remains exact.
struct GatedProjectionSharedData {
  BF activation_frames[GATED_UP_ACTIVATION_FRAMES]
                      [GATED_UP_ACTIVATION_FRAME_BYTES / sizeof(BF)];
  BF weight_frames[GATED_UP_WEIGHT_DEPTH]
                  [GATED_UP_WEIGHT_FRAME_BYTES / sizeof(BF)];
};

struct DownProjectionResidualSharedData {
  BF weight_frames[DOWN_RESIDUAL_WEIGHT_DEPTH]
                  [DOWN_RESIDUAL_WEIGHT_FRAME_BYTES / sizeof(BF)];
  BF activation_frames[DOWN_RESIDUAL_ACTIVATION_DEPTH]
                      [DOWN_RESIDUAL_ACTIVATION_FRAME_BYTES / sizeof(BF)];
};

struct SharedStorage {
  union {
    GatedProjectionSharedData gated_projection;
    DownProjectionResidualSharedData down_projection_residual;
  } mainloop;
  // BARRIER_OFFSET is already 16-byte aligned. Keeping the aggregate at its
  // natural 8-byte alignment avoids padding after the seventeenth word.
  uint64_t barrier_words[BARRIER_WORDS];
};

static_assert(offsetof(GatedProjectionSharedData, weight_frames) ==
                  GATED_UP_WEIGHT_OFFSET,
              "stationary activation and weight-ring offsets changed");
static_assert(sizeof(GatedProjectionSharedData) == BARRIER_OFFSET,
              "GatedProjection data plane must end at the barrier pool");
static_assert(sizeof(DownProjectionResidualSharedData) <= BARRIER_OFFSET,
              "DownProjectionResidual data must fit the aliased pool");
static_assert(offsetof(SharedStorage, barrier_words) == BARRIER_OFFSET,
              "barrier-pool offset changed");
static_assert(sizeof(SharedStorage) == SHARED_MEMORY_BYTES,
              "dynamic shared-memory footprint changed");
static_assert(M_PAD * sizeof(BF) == 128,
              "internal activation TMA row must remain 128 bytes");
static_assert(GATED_UP_WEIGHT_FRAME_BYTES == 32768,
              "GatedUp weight TMA must stay at the measured 32 KB frame cap");
static_assert(GATED_UP_ACTIVATION_FRAME_BYTES == 32768,
              "internal GatedUp activation must stay at the 32 KB frame cap");
static_assert(DOWN_RESIDUAL_BLOCK_K == 2 * DOWN_RESIDUAL_HIDDEN_TMA_K &&
                  DOWN_RESIDUAL_ACTIVATION_FRAME_BYTES ==
                      2 * DOWN_RESIDUAL_HIDDEN_TMA_BYTES,
              "BK128 hidden stage must contain exactly two BK64 TMA boxes");
static_assert(GATED_UP_WGMMA_WAIT < GATED_UP_WEIGHT_DEPTH,
              "WGMMA retirement distance must fit the weight ring");
static_assert(BARRIER_OFFSET % 16 == 0,
              "mbarrier storage must remain 16-byte aligned");
static_assert(SHARED_MEMORY_BYTES <= 232448,
              "stationary XFS and weight ring must fit H100 opt-in SMEM");

// ------------------------------------------------------- canonical smem layouts
// The internal GatedUp input is [K, M_PAD] with M contiguous, allowing one
// legal 32 KB SW128 TMA box. DownResidual A tiles two K-contiguous 64x64
// atoms into one BK128 compute stage while retaining the row-major input.
using GatedUpSmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{},
    Shape<Int<M_PAD>, Int<GATED_UP_BLOCK_K>>{}));
// Wd uses a 32-column SW64 tile. Gate/up is interleaved offline as a
// 64-column row [W1|W2], loaded by one 128B TMA and consumed as two B views.
using DownResidualSmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{},
    Shape<Int<M_PAD>, Int<DOWN_RESIDUAL_BLOCK_K>>{}));
using DownResidualSmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_MN_SW64_Atom<BF>{},
    Shape<Int<BN>, Int<DOWN_RESIDUAL_BLOCK_K>>{}));
using GatedUpSmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{},
    Shape<Int<2 * BN>, Int<GATED_UP_BLOCK_K>>{}));

// spec: math[].unit / inst_shape -- wgmma.m64n32k16, A K-major, B MN-major
// GatedUp consumes the interleaved [W1|W2] slab as ONE 64-wide B operand instead of
// two 32-wide views of it. The slab was already contiguous -- the old code
// sliced it back apart -- and N=64 is where the tensor core actually runs:
// [hardware-unit-test wgmma.issue.wg.ss] measures m64n32k16 at 24.7 cycles against an
// architectural 15.3 (62% of peak) and m64n64k16 at 33.1 against 30.7 (93%).
// Two of the former cost 197.6 cycles per stage where one of the latter costs
// 132.4 -- 1.49x, and register-neutral: a 64x64 accumulator is 32 f32/thread,
// exactly what acc1 + acc2 cost. DownResidual keeps the N=32 atom; its output tile is
// 32 wide and it has no second operand to pair.
using FullBar  = cutlass::arch::ClusterTransactionBarrier;  // mbarrier-tx
using EmptyBar = cutlass::arch::ClusterBarrier;             // mbarrier
using TiledMma = typename ref::DownResidualGemm::TiledMma;
// Global XFS is logical [K,M] and TMA makes M the contiguous 128-byte row.
// The resulting shared operand is therefore MN-major for GMMA, matching the
// measured BK256 upper-bound profile rather than the row-major production A.
using TiledMmaWide = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN,
                                  GMMA::Major::MN>{}));
using TaskDesc = TaskDescriptor;  // taskgraph.queue; binary ABI is unchanged
using BarrierViews = ref::BarrierViews<FullBar, EmptyBar, GATED_UP_WEIGHT_DEPTH,
                                       DOWN_RESIDUAL_WEIGHT_DEPTH, DOWN_RESIDUAL_ACTIVATION_DEPTH>;

// Task-local views own every address calculation. Producers and consumers see
// named frames/barrier pools, never a raw shared-memory byte offset.
struct GatedProjectionSharedStorageView {
  SharedStorage* storage;

  __device__ __forceinline__ BF* activation_frame(int stage) const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    return reinterpret_cast<BF*>(
        base + stage * GATED_UP_ACTIVATION_FRAME_BYTES);
  }
  __device__ __forceinline__ BF* weight_frame(int stage) const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    return reinterpret_cast<BF*>(
        base + GATED_UP_WEIGHT_OFFSET + stage * GATED_UP_WEIGHT_FRAME_BYTES);
  }
  __device__ __forceinline__ uint64_t* barrier_words() const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    return reinterpret_cast<uint64_t*>(base + BARRIER_OFFSET);
  }
  __device__ __forceinline__ FullBar* activation_full() const {
    return reinterpret_cast<FullBar*>(barrier_words());
  }
  __device__ __forceinline__ FullBar* weight_full() const {
    return reinterpret_cast<FullBar*>(
        barrier_words() + GATED_UP_ACTIVATION_FRAMES);
  }
  __device__ __forceinline__ EmptyBar* weight_empty() const {
    return reinterpret_cast<EmptyBar*>(
        barrier_words() + GATED_UP_ACTIVATION_FRAMES +
        GATED_UP_WEIGHT_DEPTH);
  }
};

struct DownProjectionResidualSharedStorageView {
  SharedStorage* storage;

  __device__ __forceinline__ BF* weight_frame(int stage) const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    return reinterpret_cast<BF*>(
        base + stage * DOWN_RESIDUAL_WEIGHT_FRAME_BYTES);
  }
  __device__ __forceinline__ BF* activation_frame(int stage) const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    constexpr int kActivationOffset =
        offsetof(DownProjectionResidualSharedData, activation_frames);
    return reinterpret_cast<BF*>(
        base + kActivationOffset +
        stage * DOWN_RESIDUAL_ACTIVATION_FRAME_BYTES);
  }
  __device__ __forceinline__ uint64_t* barrier_words() const {
    auto* base = reinterpret_cast<uint8_t*>(storage);
    return reinterpret_cast<uint64_t*>(base + BARRIER_OFFSET);
  }
  __device__ __forceinline__ FullBar* weight_full() const {
    return BarrierViews::down_residual_weight_full(barrier_words());
  }
  __device__ __forceinline__ EmptyBar* weight_empty() const {
    return BarrierViews::down_residual_weight_empty(barrier_words());
  }
  __device__ __forceinline__ FullBar* activation_full() const {
    return BarrierViews::down_residual_activation_full(barrier_words());
  }
  __device__ __forceinline__ EmptyBar* activation_empty() const {
    return BarrierViews::down_residual_activation_empty(barrier_words());
  }
  __device__ __forceinline__ EmptyBar* stage4_prefetch_ready() const {
    return reinterpret_cast<EmptyBar*>(
        barrier_words() + DOWN_RESIDUAL_STAGE4_PREFETCH_BARRIER);
  }
};

// --------------------------------------------------------------- device utils
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

namespace tma {

// Device-side issue layer. Host descriptor construction is kept separate.
// Every helper is called by one elected producer lane; coords are {inner, outer}.
__device__ __forceinline__ void load_2d(
    const CUtensorMap* map, void* dst, int32_t c_inner, int32_t c_outer,
    uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(d), "l"(map), "r"(c_inner), "r"(c_outer), "r"(b) : "memory");
}

// DeepGEMM marks one-shot TMA loads EVICT_FIRST so streamed weights do not
// displace the reused XFS working set from L2. Keep activation loads normal.
__device__ __forceinline__ void load_weight_2d_evict_first(
    const CUtensorMap* map, void* dst, int32_t c_inner, int32_t c_outer,
    uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  constexpr uint64_t hint = static_cast<uint64_t>(
      cute::TMA::CacheHintSm90::EVICT_FIRST);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
      :: "r"(d), "l"(map), "r"(c_inner), "r"(c_outer), "r"(b), "l"(hint)
      : "memory");
}

// Tensor prefetch has no shared-memory destination and no completion barrier.
// It moves the tensor-map tile toward L2; the later ordinary TMA load remains
// the sole operation that owns the weight transaction barrier.
__device__ __forceinline__ void prefetch_2d_to_l2(
    const CUtensorMap* map, int32_t c_inner, int32_t c_outer) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];"
      :: "l"(map), "r"(c_inner), "r"(c_outer) : "memory");
}

// Plain 1D bulk-copy helper retained for profiles that transport side data.
__device__ __forceinline__ void load_1d(
    void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}

}  // namespace tma

// spec: grid.persistence.phase_ordering -- release: fence then red.add
__device__ __forceinline__ void counter_release(uint32_t* c) {
  __threadfence();  // GatedUp's st.global of the hidden tile precedes the count
  atomicAdd(c, 1u);
}

// Watchdog: a persistent-kernel bug hangs rather than fails (PLAN 4.3's
// lesson), so every wait carries a deadline. On expiry it records
// {site, g, tid} into a host-pinned buffer and traps -- the hang becomes a
// named barrier instead of a 40-minute silent job.
constexpr long long WATCHDOG_CYCLES = 1ll << 31;  // ~1.2 s at 1.8 GHz

__device__ __forceinline__ void wd_fire(long long* dbg, int site, int g) {
  if (dbg) {
    long long* d = dbg + blockIdx.x * 4;
    d[0] = site; d[1] = g; d[2] = threadIdx.x; d[3] = 1;
    __threadfence_system();
  }
  __trap();
}

// mbarrier parity wait with the watchdog; replaces the cutlass .wait() loop
__device__ __forceinline__ void wait_bar_wd(uint64_t* bar, uint32_t phase,
                                            int site, int g, long long* dbg) {
  uint32_t addr = smem_u32(bar);
  uint32_t done = 0;
  long long t0 = clock64();
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done) : "r"(addr), "r"(phase));
    if (!done && clock64() - t0 > WATCHDOG_CYCLES) wd_fire(dbg, site, g);
  }
}

// acquire: elected producer lane polls; nanosleep keeps the poll off the LSU
__device__ __forceinline__ void counter_wait(const uint32_t* c, uint32_t need,
                                             int g, long long* dbg) {
  uint32_t v;
  long long t0 = clock64();
  while (true) {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c));
    if (v >= need) break;
    if (clock64() - t0 > WATCHDOG_CYCLES) wd_fire(dbg, 7, g);
    __nanosleep(64);
  }
}

// Watchdogged shared-memory waits are retained for producer-side empty slots.
// Full transaction barriers are consumed directly with ClusterBarrier::wait;
// producer warps use __syncwarp() before electing the lane that arrives and
// emits TMA, so no scheduler-to-producer command handoff is required.
// gelu_tanh in sigmoid form -- mirrors tilelang/kernels/base.py:_gelu exactly
// (rounding_contract: f32 throughout the epilogue)
__device__ __forceinline__ float gelu_sig(float v) {
  float u = 1.5957691216057308f * v * (1.0f + 0.044715f * v * v);
  return v * (1.0f / (1.0f + __expf(-u)));
}

// math-WG barrier (warps 0-3 only; producer warp never arrives)
__device__ __forceinline__ void mathwg_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

// ------------------------------------------------------------------- GatedUp body
// spec L2/L3: GatedUp steady state. tid 0..127 = math0; warp 4 is the weight
// producer and warp 5 is the activation producer (same roles as DownResidual);
// warp 6 is reserved for future task-queue scheduling.
struct GatedProjectionTask {
  const TaskDesc* tasks;
  int task_count;
  const CUtensorMap* xfs_tensor_map;
  const CUtensorMap* weight_tensor_map;
  const __nv_bfloat16* legacy_norm_factor;
  const __nv_bfloat16* legacy_adaptive_scale;
  const __nv_bfloat16* gate_bias;
  const __nv_bfloat16* up_bias;
  const __nv_bfloat16* legacy_output_scale;
  __nv_bfloat16* hidden_output;
  uint32_t* hidden_ready_counters;
  SharedStorage* shared;
  long long* dbg;
};

// The activation producer fills four fixed frames once. The frames stay live
// until the task finishes, so there is no activation empty/reuse protocol.
__device__ __forceinline__ void gated_projection_activation_loader(
    const GatedProjectionTask& task) {
  GatedProjectionSharedStorageView shared{task.shared};
  auto* full_a = shared.activation_full();
  for (int i = 0; i < GATED_UP_ACTIVATION_FRAMES; ++i) {
    int k = i * GATED_UP_BLOCK_K;
    __syncwarp();
    BF* activation_frame = shared.activation_frame(i);
    if (cute::elect_one_sync()) {
      full_a[i].arrive_and_expect_tx(GATED_UP_ACTIVATION_FRAME_BYTES);
      tma::load_2d(task.xfs_tensor_map, activation_frame, 0, k,
                   reinterpret_cast<uint64_t*>(&full_a[i]));
    }
  }
}

__device__ __forceinline__ void gated_projection_weight_loader(
    const GatedProjectionTask& task) {
  GatedProjectionSharedStorageView shared{task.shared};
  auto* full_w = shared.weight_full();
  auto* empty_w = shared.weight_empty();
  uint32_t empty_phase[GATED_UP_WEIGHT_DEPTH] = {};
  for (int g = 0; g < task.task_count * GATED_UP_TRIP; ++g) {
    int t = g / GATED_UP_TRIP, i = g % GATED_UP_TRIP;
    int n = task.tasks[t].column;
    int s = g % GATED_UP_WEIGHT_DEPTH;
    if (g >= GATED_UP_WEIGHT_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty_w[s]), empty_phase[s],
                  2, g, task.dbg);
      empty_phase[s] ^= 1;
    }
    __syncwarp();
    // Interleaved gate/up weights: each blocked row is [W1(32), W2(32)].
    if (cute::elect_one_sync()) {
      full_w[s].arrive_and_expect_tx(GATED_UP_WEIGHT_FRAME_BYTES);
      tma::load_weight_2d_evict_first(
          task.weight_tensor_map, shared.weight_frame(s),
          0, (n >> 5) * D + i * GATED_UP_BLOCK_K,
          reinterpret_cast<uint64_t*>(&full_w[s]));
      // Let stage 0's real TMA enter the urgent path before competing for L2
      // resources, then move stage 1 toward L2 immediately before its load.
      // Prefetching at loader entry delays the critical stage-0 transaction.
      if (i == 0) {
        tma::prefetch_2d_to_l2(
            task.weight_tensor_map, 0,
            (n >> 5) * D + (i + 1) * GATED_UP_BLOCK_K);
      }
    }
  }
}

__device__ __forceinline__ void gated_projection_math(
    const GatedProjectionTask& task, int tid) {
  GatedProjectionSharedStorageView shared{task.shared};
  auto* full_a = shared.activation_full();
  auto* full_w = shared.weight_full();
  auto* empty_w = shared.weight_empty();
  uint32_t weight_phase[GATED_UP_WEIGHT_DEPTH] = {};

  TiledMmaWide mma;
  auto thr = mma.get_thread_slice(tid);
  // Columns [0, BN) are the gate GEMM, [BN, 2*BN) the up GEMM.
  Tensor acc = partition_fragment_C(mma, Shape<Int<M_PAD>, Int<2 * BN>>{});
  Tensor cC = thr.partition_C(
      make_identity_tensor(Shape<Int<M_PAD>, Int<2 * BN>>{}));

  for (int t = 0; t < task.task_count; ++t) {
    int n = task.tasks[t].column;
    clear(acc);
    for (int i = 0; i < GATED_UP_TRIP; ++i) {
      int g = t * GATED_UP_TRIP + i;
      int s = g % GATED_UP_WEIGHT_DEPTH;
      // The XFS frame is filled exactly once and remains stationary. The
      // weight frame rotates independently and changes phase on every reuse.
      full_a[i].wait(0);
      full_w[s].wait(weight_phase[s]);
      weight_phase[s] ^= 1;

      // The upper-bound input is already scaled by F and S in BF16 before it
      // is materialized as [K, M_PAD]. One BK=256 stage then issues sixteen
      // N=64 WGMMA instructions in one commit.
      Tensor sA = make_tensor(make_smem_ptr(shared.activation_frame(i)),
                              GatedUpSmemLayoutA{});
      Tensor sBwide = make_tensor(make_smem_ptr(shared.weight_frame(s)),
                                  GatedUpSmemLayoutB{});
      Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));
      Tensor tCrB = thr.make_fragment_B(thr.partition_B(sBwide));
      // FlashMLA's CuTe wrapper keeps the fence/arrive/commit ordering next
      // to the GEMM contract instead of repeating raw choreography here.
      ::sm90::gemm<false, -1, true, true>(mma, tCrA, tCrB, acc);
      // Keep one committed group outstanding. Once g-1 retires, release its
      // shared-memory frame independently of the three-stage TMA ring depth.
      if (g >= GATED_UP_WGMMA_WAIT) {
        warpgroup_wait<GATED_UP_WGMMA_WAIT>();
        __syncwarp();
        if ((tid & 31) == 0)
          empty_w[(g - GATED_UP_WGMMA_WAIT) % GATED_UP_WEIGHT_DEPTH].arrive();
      }
    }
    // Retire everything before the epilogue reads acc.  A worker-queue CTA
    // may have only one GatedUp slot, so drain the tail frames explicitly before
    // its shared barrier pool is reused by the following DownResidual slot.
    warpgroup_wait<0>();
    __syncwarp();
    if ((tid & 31) == 0) {
      for (int tail = GATED_UP_TRIP - GATED_UP_WGMMA_WAIT;
           tail < GATED_UP_TRIP; ++tail)
        empty_w[tail % GATED_UP_WEIGHT_DEPTH].arrive();
    }

    // spec non_mma.gelu_gate epilogue: f32 bias/gelu/product, bf16 pair store
    __nv_bfloat16* Cg = task.hidden_output + n;
    // Gate and up share one accumulator now. For m64n64k16, register r maps to
    // col = (r/4)*8 + (lane%4)*2 + (r%2): r and r+UP differ only in the column
    // group, so they are the SAME (row, col) with col apart by BN -- the
    // gate/up pair for one output element. Parity against torch is what checks
    // this, not the derivation.
    constexpr int UP = M_PAD * BN / 128;          // 16 = half the accumulator
    CUTE_UNROLL
    for (int e = 0; e < UP; e += 2) {
      int row = get<0>(cC(e)), col = get<1>(cC(e));
      float h0 = gelu_sig(acc(e) + __bfloat162float(task.gate_bias[n + col]))
               * (acc(e + UP) + __bfloat162float(task.up_bias[n + col]));
      float h1 = gelu_sig(
                     acc(e + 1) +
                     __bfloat162float(task.gate_bias[n + col + 1]))
               * (acc(e + 1 + UP) +
                  __bfloat162float(task.up_bias[n + col + 1]));
      __nv_bfloat162 hv{__float2bfloat16(h0), __float2bfloat16(h1)};
      *reinterpret_cast<__nv_bfloat162*>(Cg + row * FF + col) = hv;
    }
    // all 128 threads' stores precede the single release
    mathwg_sync();
    if (tid == 0) {
      counter_release(
          &task.hidden_ready_counters[task.tasks[t].dependency]);
    }
  }
}

// ------------------------------------------------------------------- DownResidual body
// DownResidual split rings: W_d depth 4 dep-free, A_h depth 4 counter-gated
// on counter[k-slice] >= 4. The decoupling IS the design thesis.
struct DownProjectionResidualTask {
  const TaskDesc* tasks;
  int task_count;
  const CUtensorMap* weight_tensor_map;
  const CUtensorMap* hidden_tensor_map;
  const __nv_bfloat16* output_gate;
  __nv_bfloat16* output;
  uint32_t* hidden_ready_counters;
  float* split_partials;
  uint32_t* split_ready_counters;
  SharedStorage* shared;
  long long* dbg;
};

__device__ __forceinline__ void down_projection_weight_loader(
    const DownProjectionResidualTask& task) {
  DownProjectionResidualSharedStorageView shared{task.shared};
  auto* full_w = shared.weight_full();
  auto* empty_w = shared.weight_empty();
  auto* full_a = shared.activation_full();
  auto* stage4_prefetch_ready = shared.stage4_prefetch_ready();
  uint32_t ewph[DOWN_RESIDUAL_WEIGHT_DEPTH] = {};
  for (int g = 0; g < task.task_count * DOWN_RESIDUAL_TRIP_PER_SPLIT; ++g) {
    int t = g / DOWN_RESIDUAL_TRIP_PER_SPLIT, i = g % DOWN_RESIDUAL_TRIP_PER_SPLIT;
    // `pad` carries the split index; this CTA walks only its own K span.
    int n = task.tasks[t].column;
    int k = task.tasks[t].split * DOWN_RESIDUAL_K_SPAN +
            i * DOWN_RESIDUAL_BLOCK_K;
    // dep-free weight ring first: it runs ahead through any counter stall
    int sw = g % DOWN_RESIDUAL_WEIGHT_DEPTH;
    if (g >= DOWN_RESIDUAL_WEIGHT_DEPTH) {
      // Stage 4 is the only steady-state weight tile whose completion remains
      // exposed. Activation stage 0 completes before weight slot 0 can be
      // released, so use that otherwise-dead interval to move stage 4 toward
      // L2 without attaching another transaction to the shared-memory ring.
      if (i == DOWN_RESIDUAL_WEIGHT_DEPTH) {
        full_a[0].wait(0);
        if (cute::elect_one_sync()) {
          // Release the activation producer before issuing the barrier-free
          // prefetch. The handshake protects the full_a[0] phase from ABA;
          // it does not make the prefetch part of activation's dependency.
          stage4_prefetch_ready->arrive();
          tma::prefetch_2d_to_l2(
              task.weight_tensor_map, 0, (n >> 5) * FF + k);
        }
      }
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty_w[sw]), ewph[sw], 5, g, task.dbg);
      ewph[sw] ^= 1;
    }
    // All lanes may wait on the producer ring, but only the elected lane owns
    // the transaction-barrier arrival and the corresponding TMA issue.  The
    // old single-thread dispatch implicitly provided this election; once the
    // whole warp runs the producer body it must be explicit here.
    if (cute::elect_one_sync()) {
      full_w[sw].arrive_and_expect_tx(DOWN_RESIDUAL_WEIGHT_FRAME_BYTES);
      tma::load_2d(
          task.weight_tensor_map, shared.weight_frame(sw),
          0, (n >> 5) * FF + k,
          reinterpret_cast<uint64_t*>(&full_w[sw]));
    }
  }
}

// The DownResidual activation producer owns its dependency poll and TMA issue in one
// warp.  Keeping these operations together avoids a scheduler->TMA sequence
// handoff (and the block fence that would otherwise be needed to order it).
__device__ __forceinline__ void down_projection_activation_loader(
    const DownProjectionResidualTask& task) {
  DownProjectionResidualSharedStorageView shared{task.shared};
  auto* full_a = shared.activation_full();
  auto* empty_a = shared.activation_empty();
  auto* stage4_prefetch_ready = shared.stage4_prefetch_ready();
  uint32_t eaph[DOWN_RESIDUAL_ACTIVATION_DEPTH] = {};
  for (int g = 0; g < task.task_count * DOWN_RESIDUAL_TRIP_PER_SPLIT; ++g) {
    int t = g / DOWN_RESIDUAL_TRIP_PER_SPLIT, i = g % DOWN_RESIDUAL_TRIP_PER_SPLIT;
    // Absolute stage index over the full FF: the counter this split waits on
    // covers ITS k span, not the span a full-K task would have walked.
    int kb = task.tasks[t].split * DOWN_RESIDUAL_TRIP_PER_SPLIT + i;
    // gated activation ring: poll BEFORE reserving the hidden slice (spec L2)
    int counter_id = kb / (COUNTER_K / DOWN_RESIDUAL_BLOCK_K);
    counter_wait(&task.hidden_ready_counters[counter_id], COUNTER_ARRIVE,
                 g, task.dbg);
    int sa = g % DOWN_RESIDUAL_ACTIVATION_DEPTH;
    // Do not rearm full_a[0] for activation stage 4 until the weight producer
    // has observed activation stage 0. This prevents a late phase-0 waiter
    // from confusing a future barrier generation for the original A0.
    if (i == DOWN_RESIDUAL_ACTIVATION_DEPTH) {
      stage4_prefetch_ready->wait(0);
    }
    if (g >= DOWN_RESIDUAL_ACTIVATION_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty_a[sa]), eaph[sa], 6, g, task.dbg);
      eaph[sa] ^= 1;
    }
    __syncwarp();
    int k = kb * DOWN_RESIDUAL_BLOCK_K;
    BF* Ah = shared.activation_frame(sa);
    if (cute::elect_one_sync()) {
      // GatedProjection publishes hidden through generic global stores while
      // this TMA load reads it through the async proxy.
      asm volatile("fence.proxy.async.global;" ::: "memory");
      full_a[sa].arrive_and_expect_tx(
          DOWN_RESIDUAL_ACTIVATION_FRAME_BYTES);
      // SW128 limits the hidden descriptor's innermost box to [K64, M64].
      // Two adjacent 8 KiB loads fill this BK128 frame and share one 16 KiB
      // completion barrier, so math observes the stage atomically.
      tma::load_2d(task.hidden_tensor_map, Ah, k, 0,
                   reinterpret_cast<uint64_t*>(&full_a[sa]));
      tma::load_2d(
          task.hidden_tensor_map,
          reinterpret_cast<uint8_t*>(Ah) +
              DOWN_RESIDUAL_HIDDEN_TMA_BYTES,
          k + DOWN_RESIDUAL_HIDDEN_TMA_K, 0,
          reinterpret_cast<uint64_t*>(&full_a[sa]));
    }
  }
}

__device__ __forceinline__ void down_projection_math(
    const DownProjectionResidualTask& task, int tid) {
  DownProjectionResidualSharedStorageView shared{task.shared};
  auto* full_w = shared.weight_full();
  auto* empty_w = shared.weight_empty();
  auto* full_a = shared.activation_full();
  auto* empty_a = shared.activation_empty();
  uint32_t fwph[DOWN_RESIDUAL_WEIGHT_DEPTH] = {}, faph[DOWN_RESIDUAL_ACTIVATION_DEPTH] = {};

  TiledMma mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor acc = partition_fragment_C(mma, Shape<Int<M_PAD>, Int<BN>>{});
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<M_PAD>, Int<BN>>{}));

  for (int t = 0; t < task.task_count; ++t) {
    int n = task.tasks[t].column;
    clear(acc);
    for (int i = 0; i < DOWN_RESIDUAL_TRIP_PER_SPLIT; ++i) {
      int g = t * DOWN_RESIDUAL_TRIP_PER_SPLIT + i;
      int sw = g % DOWN_RESIDUAL_WEIGHT_DEPTH, sa = g % DOWN_RESIDUAL_ACTIVATION_DEPTH;
      full_w[sw].wait(fwph[sw]);
      fwph[sw] ^= 1;
      full_a[sa].wait(faph[sa]);
      faph[sa] ^= 1;
      Tensor sAh = make_tensor(make_smem_ptr(shared.activation_frame(sa)),
                               DownResidualSmemLayoutA{});
      Tensor sWd = make_tensor(make_smem_ptr(shared.weight_frame(sw)),
                               DownResidualSmemLayoutB{});
      Tensor tCrA = thr.make_fragment_A(thr.partition_A(sAh));
      Tensor tCrB = thr.make_fragment_B(thr.partition_B(sWd));
      ::sm90::gemm<false, -1, true, true>(mma, tCrA, tCrB, acc);
      if (g > 0) {
        warpgroup_wait<1>();
        __syncwarp();
        if ((tid & 31) == 0) {
          empty_w[(g - 1) % DOWN_RESIDUAL_WEIGHT_DEPTH].arrive();
          empty_a[(g - 1) % DOWN_RESIDUAL_ACTIVATION_DEPTH].arrive();
        }
      }
    }
    // Drain the final activation/weight frames before this CTA reuses the
    // aliased pool for another task slot or for the next kernel replay.
    warpgroup_wait<0>();
    __syncwarp();
    if ((tid & 31) == 0) {
      empty_w[(DOWN_RESIDUAL_TRIP_PER_SPLIT - 1) % DOWN_RESIDUAL_WEIGHT_DEPTH].arrive();
      empty_a[(DOWN_RESIDUAL_TRIP_PER_SPLIT - 1) % DOWN_RESIDUAL_ACTIVATION_DEPTH].arrive();
    }

    // Split-K join. `out` is both residual and destination -- each thread
    // read-modify-writes its own element -- so exactly ONE CTA per output tile
    // may touch it. Splits 1..S-1 publish an f32 partial and arrive; split 0
    // waits, folds them in a FIXED order, then runs the unchanged epilogue.
    // Fixed order keeps the result bit-identical to the un-split kernel, which
    // an atomic accumulate would not (f32 addition is not associative).
    const int split = task.tasks[t].split;
    const int tile  = n / BN;
    if (split != 0) {
      float* P = task.split_partials
               + ((size_t)(split - 1) * NUM_DOWN_RESIDUAL_TILES + tile) * PARTIAL_ELEMS;
      CUTE_UNROLL
      for (int e = 0; e < size(acc); ++e)
        P[get<0>(cC(e)) * BN + get<1>(cC(e))] = acc(e);
      mathwg_sync();          // every math lane's store issued before the count
      if (tid == 0) counter_release(&task.split_ready_counters[tile]);
      continue;               // only split 0 owns `out`
    }
    if (tid == 0)
      counter_wait(&task.split_ready_counters[tile],
                   DOWN_RESIDUAL_SPLIT - 1, t, task.dbg);
    mathwg_sync();            // thread 0's gpu-scope acquire covers the CTA
    CUTE_UNROLL
    for (int sp = 1; sp < DOWN_RESIDUAL_SPLIT; ++sp) {
      const float* P = task.split_partials
                     + ((size_t)(sp - 1) * NUM_DOWN_RESIDUAL_TILES + tile) * PARTIAL_ELEMS;
      CUTE_UNROLL
      for (int e = 0; e < size(acc); ++e)
        acc(e) += P[get<0>(cC(e)) * BN + get<1>(cC(e))];
    }

    // DownResidual epilogue: f32 gate multiply + residual add, bf16 store.
    // out is both R and C; each thread reads its element before writing it.
    __nv_bfloat16* Og = task.output + n;
    CUTE_UNROLL
    for (int e = 0; e < size(acc); e += 2) {
      int row = get<0>(cC(e)), col = get<1>(cC(e));
      __nv_bfloat162 r =
          *reinterpret_cast<const __nv_bfloat162*>(Og + row * D + col);
      float o0 = __bfloat162float(r.x)
               + acc(e)     * __bfloat162float(task.output_gate[n + col]);
      float o1 = __bfloat162float(r.y)
               + acc(e + 1) * __bfloat162float(task.output_gate[n + col + 1]);
      __nv_bfloat162 ov{__float2bfloat16(o0), __float2bfloat16(o1)};
      *reinterpret_cast<__nv_bfloat162*>(Og + row * D + col) = ov;
    }
  }
}

__device__ __forceinline__ void initialize_barriers(
    const GatedProjectionSharedStorageView& shared) {
  auto* activation_full = shared.activation_full();
  auto* weight_full = shared.weight_full();
  auto* weight_empty = shared.weight_empty();
  for (int stage = 0; stage < GATED_UP_ACTIVATION_FRAMES; ++stage)
    activation_full[stage].init(1);
  for (int stage = 0; stage < GATED_UP_WEIGHT_DEPTH; ++stage) {
    weight_full[stage].init(1);
    weight_empty[stage].init(kNumEpilogueWarps);
  }
}

__device__ __forceinline__ void initialize_barriers(
    const DownProjectionResidualSharedStorageView& shared) {
  auto* weight_full = shared.weight_full();
  auto* weight_empty = shared.weight_empty();
  auto* activation_full = shared.activation_full();
  auto* activation_empty = shared.activation_empty();
  auto* stage4_prefetch_ready = shared.stage4_prefetch_ready();
  for (int stage = 0; stage < DOWN_RESIDUAL_WEIGHT_DEPTH; ++stage) {
    weight_full[stage].init(1);
    weight_empty[stage].init(kNumEpilogueWarps);
  }
  for (int stage = 0; stage < DOWN_RESIDUAL_ACTIVATION_DEPTH; ++stage) {
    activation_full[stage].init(1);
    activation_empty[stage].init(kNumEpilogueWarps);
  }
  stage4_prefetch_ready->init(1);
}

// ------------------------------------------------------------------ the kernel
__global__ void __launch_bounds__(WarpRoles::kThreads, 1)
    ffn_taskloop_kernel(
    const TaskDesc* __restrict__ table,
    const __grid_constant__ CUtensorMap xfs_tensor_map,
    const __grid_constant__ CUtensorMap gated_up_weight_tensor_map,
    const __grid_constant__ CUtensorMap packed_gate_up_legacy_unused,
    const __grid_constant__ CUtensorMap down_weight_tensor_map,
    const __grid_constant__ CUtensorMap hidden_tensor_map,
    const __nv_bfloat16* __restrict__ F,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ b2,
    const __nv_bfloat16* __restrict__ g_gate,
    __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ out,
    uint32_t* __restrict__ counters,
    float* __restrict__ down_residual_partial,
    uint32_t* __restrict__ down_residual_counters,
    long long* __restrict__ dbg,
    int32_t xfs_wait_mode) {
  // The dynamic extent is supplied by the launch; typed views below own all
  // frame and barrier addressing.
  extern __shared__ uint8_t smem_buffer[];
  auto* shared_storage = reinterpret_cast<SharedStorage*>(smem_buffer);
  GatedProjectionSharedStorageView gated_shared{shared_storage};
  DownProjectionResidualSharedStorageView down_shared{shared_storage};
  const int tid = threadIdx.x;
  const TaskDesc* cta_tasks = table + blockIdx.x * MAX_TASKS_PER_CTA;
  // The launch geometry is fixed at 132 workers.  Bisection schedules keep
  // that geometry and mark unused workers with a sentinel row; idle workers
  // must return before initializing their private barrier pool.
  if (ref::is_sentinel(cta_tasks[0].kind)) return;
  for (int slot = 0; slot < MAX_TASKS_PER_CTA; ++slot) {
    const TaskKind kind = cta_tasks[slot].kind;
    if (ref::is_sentinel(kind)) break;
    // Warp roles are uniform across task types: warp 4 loads weights and warp
    // 5 fills the stationary activation. Reinitialize the aliased barrier pool
    // at the GatedUp->DownResidual worker seam.
    if (tid == 0) {
      if (ref::is_gated_up(kind)) {
        initialize_barriers(gated_shared);
      } else {
        initialize_barriers(down_shared);
      }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    // Address calculation and barrier initialization above are independent of
    // the primary XFS producer.  Mode 1 holds every warp here (shipped
    // behavior).  Mode 2 releases the weight-loader and reserved warps: both
    // weight loaders read only static weight tensor maps and smem rings, so
    // their TMA issue (and the gate/up L2 prefetch) runs under the producer's
    // tail, while every reader of producer-written state -- XFS activations,
    // the readiness-counter arrays the producer resets, the residual -- still
    // waits per thread.  This wait must stay after the __syncthreads() above:
    // moved before it, the waiting warps would hold the weight warps at the
    // barrier and the early release would be void.  Wait only at slot 0; warp
    // roles are tid-fixed across slots, so program order covers slot 1.
    if (slot == 0 && xfs_wait_mode != 0) {
      const bool releases_early =
          xfs_wait_mode == 2 && (WarpRoles::is_weight_producer(tid) ||
                                 WarpRoles::is_reserved(tid));
      if (!releases_early) {
        cudaGridDependencySynchronize();
      }
    }
    if (ref::is_gated_up(kind)) {
      GatedProjectionTask task{
          cta_tasks + slot, 1, &xfs_tensor_map, &gated_up_weight_tensor_map,
          F, S, b1, b2, S, hidden, counters, shared_storage, dbg};
      if (WarpRoles::is_math(tid)) {
        gated_projection_math(task, tid);
      } else if (WarpRoles::is_weight_producer(tid)) {
        gated_projection_weight_loader(task);
      } else if (WarpRoles::is_activation_producer(tid)) {
        gated_projection_activation_loader(task);
      }
    } else {
      DownProjectionResidualTask task{
          cta_tasks + slot, 1, &down_weight_tensor_map, &hidden_tensor_map,
          g_gate, out, counters, down_residual_partial,
          down_residual_counters, shared_storage, dbg};
      if (WarpRoles::is_math(tid)) {
        down_projection_math(task, tid);
      } else if (WarpRoles::is_weight_producer(tid)) {
        down_projection_weight_loader(task);
      } else if (WarpRoles::is_activation_producer(tid)) {
        down_projection_activation_loader(task);
      }
    }
    // All active warps finish before the CTA reuses the shared-memory pool for
    // the next task type.
    __syncthreads();
  }
}

// -------------------------------------------------- Phase 0 probe: counter RTT
// Pairs of CTAs: even releases (timestamp then fence+red), odd polls with
// acquire and records globaltimer delta. 40 concurrent pairs approximates the
// contended case the spec's counter figures are [I] about.
__global__ void counter_probe_kernel(uint32_t* __restrict__ c,
                                     long long* __restrict__ t0s,
                                     long long* __restrict__ out_ns) {
  if (threadIdx.x != 0) return;
  int pair = blockIdx.x / 2;
  if ((blockIdx.x & 1) == 0) {
    long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    t0s[pair] = t0;
    __threadfence();
    atomicAdd(&c[pair], 1u);
  } else {
    uint32_t v;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&c[pair]));
    } while (v < 1);
    long long t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    out_ns[pair] = t1 - t0s[pair];
  }
}

__global__ void reset_ffn_counters_kernel(
    uint32_t* __restrict__ hidden_ready,
    uint32_t* __restrict__ down_residual_ready) {
  const int index = threadIdx.x;
  if (index < N_COUNTERS) hidden_ready[index] = 0;
  if (index < NUM_DOWN_RESIDUAL_TILES) down_residual_ready[index] = 0;
}

// ------------------------------------------------------ host TMA descriptors
// Descriptor preparation is separate from the device-side `tma::load_*` layer.
static CUtensorMap encode_bf16_tensor_map_2d(
    const void* global_address, uint64_t inner_extent, uint64_t outer_extent,
    uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swizzle,
    CUresult* result) {
  CUtensorMap tensor_map{};
  uint64_t dims[2] = {inner_extent, outer_extent};
  uint64_t strides[1] = {inner_extent * 2};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t element_strides[2] = {1, 1};
  *result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<void*>(global_address), dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tensor_map;
}

}  // namespace ffn

extern "C" {

int ffn_taskloop_launch(const void* table, int n_ctas,
                        int use_programmatic_dependency,
                        const void* x_pad, const void* F, const void* S,
                        const void* W1, const void* W2,
                        const void* b1, const void* b2,
                        const void* Wd, const void* g_gate,
                        void* hidden, void* out, void* counters,
                        void* down_residual_partial, void* down_residual_counters,
                        void* dbg, void* stream) {
  using namespace ffn;
  if (n_ctas != N_CTAS) return 1101;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        ffn_taskloop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        SHARED_MEMORY_BYTES);
    if (e != cudaSuccess) return (int)e;
    attr_set = true;
  }
  int device = 0;
  int multiprocessors = 0;
  int active_blocks_per_multiprocessor = 0;
  cudaError_t occupancy_error = cudaGetDevice(&device);
  if (occupancy_error != cudaSuccess) return (int)occupancy_error;
  occupancy_error = cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device);
  if (occupancy_error != cudaSuccess) return (int)occupancy_error;
  occupancy_error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_multiprocessor, ffn_taskloop_kernel,
      WarpRoles::kThreads, SHARED_MEMORY_BYTES);
  if (occupancy_error != cudaSuccess) return (int)occupancy_error;
  if (multiprocessors * active_blocks_per_multiprocessor < N_CTAS) {
    // DownResidual workers wait on counters published by GatedProjection
    // workers. Refuse launches that cannot make every fixed worker resident.
    return 1102;
  }
  CUresult rc = CUDA_SUCCESS;
  // GatedProjection input contract: x_pad points to a contiguous
  // [D, M_PAD] buffer, so M is the 128-byte TMA row and one BK256 box fills a
  // complete GatedUp activation stage.
  CUtensorMap xfs_tensor_map = encode_bf16_tensor_map_2d(
      x_pad, M_PAD, D, M_PAD, GATED_UP_BLOCK_K,
      CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;
  // Weights arrive PRE-BLOCKED and gate/up interleaved. One 128B row is
  // [W1_tile(32), W2_tile(32)].
  // The natural (K, N) layout reads 64 B strips at an 8 KB stride, which caps
  // the machine near ~1 TB/s (job 541407: gu-only 33 us, dr-only 25 us, and
  // the TileLang composition sits at the same ~1.1 TB/s -- PLAN 4.9's 30-36%
  // MBU is the same pattern). Static weights make the relayout free, offline,
  // and planner-owned.
  CUtensorMap gated_up_weight_tensor_map = encode_bf16_tensor_map_2d(
      W1, 2 * BN, (uint64_t)(FF / BN) * D, 2 * BN,
      GATED_UP_BLOCK_K, CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;
  CUtensorMap packed_gate_up_legacy_unused{};
  if (rc) return 1000 + (int)rc;
  // DownResidual uses one [K128, N32] 8 KiB weight box per compute stage.
  CUtensorMap down_weight_tensor_map = encode_bf16_tensor_map_2d(
      Wd, BN, (uint64_t)(D / BN) * FF, BN, DOWN_RESIDUAL_BLOCK_K,
      CU_TENSOR_MAP_SWIZZLE_64B, &rc);
  if (rc) return 1000 + (int)rc;
  // The hidden operand keeps a legal [K64, M64] SW128 box. The activation
  // producer issues this descriptor twice into each BK128 shared frame.
  CUtensorMap hidden_tensor_map = encode_bf16_tensor_map_2d(
      hidden, FF, M_PAD, M_PAD, DOWN_RESIDUAL_HIDDEN_TMA_K,
      CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;

  // 0 = plain launch, no wait; 1 = programmatic launch, every warp waits at
  // entry; 2 = programmatic launch, weight-loader warps released early.
  if (use_programmatic_dependency < 0 || use_programmatic_dependency > 2) {
    return 1103;
  }
  const int32_t xfs_wait_mode = use_programmatic_dependency;
  if (xfs_wait_mode != 0) {
    cudaLaunchAttribute dependency_attribute{};
    dependency_attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    dependency_attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(N_CTAS, 1, 1);
    launch_config.blockDim = dim3(WarpRoles::kThreads, 1, 1);
    launch_config.dynamicSmemBytes = SHARED_MEMORY_BYTES;
    launch_config.stream = (cudaStream_t)stream;
    launch_config.attrs = &dependency_attribute;
    launch_config.numAttrs = 1;
    cudaError_t launch_error = cudaLaunchKernelEx(
        &launch_config, ffn_taskloop_kernel,
        (const TaskDesc*)table, xfs_tensor_map, gated_up_weight_tensor_map,
        packed_gate_up_legacy_unused, down_weight_tensor_map,
        hidden_tensor_map, (const __nv_bfloat16*)F,
        (const __nv_bfloat16*)S,
        (const __nv_bfloat16*)b1, (const __nv_bfloat16*)b2,
        (const __nv_bfloat16*)g_gate, (__nv_bfloat16*)hidden,
        (__nv_bfloat16*)out, (uint32_t*)counters,
        (float*)down_residual_partial, (uint32_t*)down_residual_counters,
        (long long*)dbg, xfs_wait_mode);
    if (launch_error != cudaSuccess) return (int)launch_error;
  } else {
    ffn_taskloop_kernel<<<N_CTAS, WarpRoles::kThreads, SHARED_MEMORY_BYTES,
                          (cudaStream_t)stream>>>(
        (const TaskDesc*)table, xfs_tensor_map, gated_up_weight_tensor_map,
        packed_gate_up_legacy_unused, down_weight_tensor_map,
        hidden_tensor_map, (const __nv_bfloat16*)F,
        (const __nv_bfloat16*)S,
        (const __nv_bfloat16*)b1, (const __nv_bfloat16*)b2,
        (const __nv_bfloat16*)g_gate, (__nv_bfloat16*)hidden,
        (__nv_bfloat16*)out, (uint32_t*)counters,
        (float*)down_residual_partial, (uint32_t*)down_residual_counters,
        (long long*)dbg, xfs_wait_mode);
  }
  return (int)cudaGetLastError();
}

int counter_probe_launch(void* c, void* t0s, void* out_ns, int pairs,
                         void* stream) {
  ffn::counter_probe_kernel<<<pairs * 2, 32, 0, (cudaStream_t)stream>>>(
      (uint32_t*)c, (long long*)t0s, (long long*)out_ns);
  return (int)cudaGetLastError();
}

int ffn_counters_reset_launch(void* hidden_ready,
                              void* down_residual_ready,
                              void* stream) {
  ffn::reset_ffn_counters_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
      (uint32_t*)hidden_ready, (uint32_t*)down_residual_ready);
  return (int)cudaGetLastError();
}

int ffn_taskloop_smem_bytes() { return ffn::SHARED_MEMORY_BYTES; }

}  // extern "C"
