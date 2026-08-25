// ffn_taskloop.cu -- Phase 5 of specs/tile/ffn_taskloop_minimal.md.
//
// Persistent task-loop prototype fusing the pi0.5 decoder FFN chain in ONE
// launch:
//   GatedUp task (128x): hidden[:, n:n+32] = gelu((x*F*S) @ W1 + b1) * ((x*F*S) @ W2 + b2)
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
// BF16 profile: keep the K tile aligned with the m64n32k16 WGMMA atom. BK=128
// is reserved for the later FP8 profile; using it here over-stages the CTA.
constexpr int BK = 64;                    // mainloop.step, both task types
constexpr int GATED_UP_TRIP = D / BK;           // 8
constexpr int DOWN_RESIDUAL_TRIP = FF / BK;          // 64 stages over the full FF
// Depth 4 everywhere, revised from the spec's 2 after job 541358 measured
// 81 us: at this fan-out the per-CTA stage cadence is TMA-LATENCY-bound
// (~L/depth per stage), not bandwidth-bound -- the spec's "depth buys only
// jitter" argument priced fills at bandwidth and was wrong. Recorded in the
// spec's deviations.
constexpr int GATED_UP_DEPTH = 4;
constexpr int DOWN_RESIDUAL_WEIGHT_DEPTH = 4;              // DownResidual weight ring, dep-free
constexpr int DOWN_RESIDUAL_ACTIVATION_DEPTH = 4;          // DownResidual activation ring, counter-gated
// Split-K on DownResidual's FF contraction. The copy column is `txns_per_warp x 248 ns`
// [hardware-unit-test TMA-ISSUE], and txns_per_warp = K_per_CTA / BK -- so
// splitting K is one of only three levers that divides a copy floor, and the
// only one available here (more CTAs does NOT move it; every CTA still walks
// its own K). S=4 takes DownResidual from 64 stages to 16, 15.87 -> 3.97 us, and lands
// under the 2.71 us DRAM wall (W_down 8.39 MB / 3.09 TB/s [TMA-CEIL]) as soon
// as BK reaches 128. Larger S buys nothing -- the issue column is already
// below the wall and every extra split is pure partial traffic.
constexpr int DOWN_RESIDUAL_SPLIT   = 4;
constexpr int DOWN_RESIDUAL_K_SPAN  = FF / DOWN_RESIDUAL_SPLIT;      // 1024 contraction rows per split
constexpr int NUM_DOWN_RESIDUAL_TILES = D / BN;             // 32 output tiles
constexpr int PARTIAL_ELEMS = M_PAD * BN;      // 2048 f32 per partial tile
constexpr int DOWN_RESIDUAL_TRIP_PER_SPLIT  = DOWN_RESIDUAL_TRIP / DOWN_RESIDUAL_SPLIT; // 16 stages per split at BK=64
constexpr int kNumEpilogueWarps = 4;      // math warpgroup consumes each stage
constexpr int MAX_TASKS_PER_CTA = 2;      // rows may end early with type = -1
constexpr int N_CTAS = 132;               // fixed H100 worker grid
constexpr int COUNTER_K = 128;             // dependency granularity in hidden K
constexpr int N_COUNTERS = FF / COUNTER_K; // one per 128-col hidden slice
constexpr int COUNTER_ARRIVE = COUNTER_K / BN; // 4 GatedUp tasks fill one slice
static_assert(FF % COUNTER_K == 0 && COUNTER_K % BK == 0,
              "counter and BK geometry must divide FF");
static_assert(WarpRoles::kThreads == 224,
              "the current ABI/profile is a 224-thread CTA");

// spec: pipeline.staged_buffers -- smem pool offsets (bytes)
constexpr int A_FRAME_B = M_PAD * BK * 2; // 8192 at BK=64
constexpr int W_FRAME_B = BK * BN * 2;    // 4096 at BK=64
constexpr int WUP_FRAME_B = 2 * W_FRAME_B; // interleaved gate/up tile
constexpr int S_FRAME_B = BK * 2;         // 256
constexpr int OFF_A  = 0;                              // GATED_UP_DEPTH x 8192
constexpr int OFF_W1 = GATED_UP_DEPTH * A_FRAME_B;           // 32768
constexpr int OFF_S  = OFF_W1 + GATED_UP_DEPTH * WUP_FRAME_B; // 65536
constexpr int OFF_BARS = OFF_S + GATED_UP_DEPTH * S_FRAME_B; // 66048
// The barrier pool is now producer-owned: no scheduler->TMA sequence arrays
// are needed.  Each producer warp waits on its empty slot, calls
// arrive_and_expect_tx for its own byte count, and emits the TMA operation.
constexpr int SMEM_B = OFF_BARS + 16 * sizeof(uint64_t);
// DownResidual pool aliases the GatedUp pool (spec: non_staged_buffers.dr_pool)
constexpr int OFF_WD = 0;                              // DOWN_RESIDUAL_WEIGHT_DEPTH x 8192
constexpr int OFF_AH = DOWN_RESIDUAL_WEIGHT_DEPTH * W_FRAME_B;          // 32768; + 4 x 16384 = 98304
static_assert(OFF_AH + DOWN_RESIDUAL_ACTIVATION_DEPTH * A_FRAME_B <= OFF_BARS, "dr pool inside gu pool");

// ------------------------------------------------------- canonical smem layouts
// A_s / A_h: 64 x 64 bf16, K-major, 128B swizzle (one TMA box).
// The BK=64 profile needs one 64x64 SW128 atom per stage.
using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF>{}, Shape<Int<M_PAD>, Int<BK>>{}));
// Wd uses a 32-column SW64 tile. Gate/up is interleaved offline as a
// 64-column row [W1|W2], loaded by one 128B TMA and consumed as two B views.
using SmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_MN_SW64_Atom<BF>{}, Shape<Int<BN>, Int<BK>>{}));
using SmemLayoutBWide = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<Int<2 * BN>, Int<BK>>{}));

// spec: math[].unit / inst_shape -- wgmma.m64n32k16, A K-major, B MN-major
// GatedUp consumes the interleaved [W1|W2] slab as ONE 64-wide B operand instead of
// two 32-wide views of it. The slab was already contiguous -- the old code
// sliced it back apart -- and N=64 is where the tensor core actually runs:
// [hardware-unit-test MMA-RATE] measures m64n32k16 at 24.7 cycles against an
// architectural 15.3 (62% of peak) and m64n64k16 at 33.1 against 30.7 (93%).
// Two of the former cost 197.6 cycles per stage where one of the latter costs
// 132.4 -- 1.49x, and register-neutral: a 64x64 accumulator is 32 f32/thread,
// exactly what acc1 + acc2 cost. DownResidual keeps the N=32 atom; its output tile is
// 32 wide and it has no second operand to pair.
using FullBar  = cutlass::arch::ClusterTransactionBarrier;  // mbarrier-tx
using EmptyBar = cutlass::arch::ClusterBarrier;             // mbarrier
using TiledMma = typename ref::DownResidualGemm::TiledMma;
using TiledMmaWide = typename ref::GatedUpGemm::TiledMma;
using TaskDesc = TaskDescriptor;  // taskgraph.queue; binary ABI is unchanged
using BarrierViews = ref::BarrierViews<FullBar, EmptyBar, GATED_UP_DEPTH,
                                       DOWN_RESIDUAL_WEIGHT_DEPTH, DOWN_RESIDUAL_ACTIVATION_DEPTH>;

// --------------------------------------------------------------- device utils
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// spec L2: TMA-2D issue, one elected thread; coords {inner, outer}
__device__ __forceinline__ void issue_tma_2d(
    const CUtensorMap* map, void* dst, int32_t c_inner, int32_t c_outer,
    uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(d), "l"(map), "r"(c_inner), "r"(c_outer), "r"(b) : "memory");
}

// spec L2: the 256 B S slice -- plain 1D bulk copy, same barrier
__device__ __forceinline__ void issue_bulk_1d(
    void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}

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
// Physical SW128 element offset inside one 64x64 A frame. The 16B chunk index
// XORs with the row (byte address bits [4:7) ^ [7:10)). Deliberately do not
// use SmemLayoutA{}(m, kk) for flat indexing: the explicit physical mapping
// is what the TMA write and the in-place BF16 scale pass share.
__device__ __forceinline__ int a_phys_elem(int m, int kk) {
  int c = kk >> 3;                  // 16B chunk within the 128 B row
  return m * BK + ((c ^ (m & 7)) << 3) + (kk & 7);
}

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
struct GatedUpTask {
  const TaskDesc* my;
  int ntask;
  const CUtensorMap *tmx, *tmwup;
  const __nv_bfloat16 *F, *S, *b1, *b2, *Sg;
  __nv_bfloat16* hidden;
  uint32_t* counters;
  uint8_t* pool;
  uint64_t* bars;
  long long* dbg;
};

// The two GatedUp producer warps own the transaction barrier directly.  Each stage
// has two arrivals (A and W), and the transaction byte counts are split across
// those arrivals.  The empty barrier is released by all four math warps.
__device__ __forceinline__ void gated_up_activation_producer(
    const GatedUpTask& task) {
  auto* full  = BarrierViews::gated_up_full(task.bars);
  auto* empty = BarrierViews::gated_up_empty(task.bars);
  uint32_t eph[GATED_UP_DEPTH] = {};
  for (int g = 0; g < task.ntask * GATED_UP_TRIP; ++g) {
    int i = g % GATED_UP_TRIP;
    int k = i * BK, s = g % GATED_UP_DEPTH;
    if (g >= GATED_UP_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty[s]), eph[s], 2, g, task.dbg);
      eph[s] ^= 1;
    }
    __syncwarp();
    uint8_t* A0 = task.pool + OFF_A + s * A_FRAME_B;
    if (cute::elect_one_sync()) {
      full[s].arrive_and_expect_tx(A_FRAME_B);
      issue_tma_2d(task.tmx, A0, k, 0, reinterpret_cast<uint64_t*>(&full[s]));
    }
  }
}

__device__ __forceinline__ void gated_up_weight_producer(
    const GatedUpTask& task) {
  auto* full  = BarrierViews::gated_up_full(task.bars);
  auto* empty = BarrierViews::gated_up_empty(task.bars);
  uint32_t eph[GATED_UP_DEPTH] = {};
  for (int g = 0; g < task.ntask * GATED_UP_TRIP; ++g) {
    int t = g / GATED_UP_TRIP, i = g % GATED_UP_TRIP;
    int n = task.my[t].column, k = i * BK, s = g % GATED_UP_DEPTH;
    if (g >= GATED_UP_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty[s]), eph[s], 2, g, task.dbg);
      eph[s] ^= 1;
    }
    __syncwarp();
    // Interleaved gate/up weights: each blocked row is [W1(32), W2(32)].
    if (cute::elect_one_sync()) {
      full[s].arrive_and_expect_tx(WUP_FRAME_B + S_FRAME_B);
      issue_tma_2d(task.tmwup, task.pool + OFF_W1 + s * WUP_FRAME_B,
             0, (n >> 5) * D + k, reinterpret_cast<uint64_t*>(&full[s]));
    }
    if (cute::elect_one_sync()) {
      issue_bulk_1d(task.pool + OFF_S + s * S_FRAME_B, task.Sg + k, S_FRAME_B,
              reinterpret_cast<uint64_t*>(&full[s]));
    }
  }
}

__device__ __forceinline__ void gated_up_math(
    const GatedUpTask& task, int tid) {
  auto* full  = BarrierViews::gated_up_full(task.bars);
  auto* empty = BarrierViews::gated_up_empty(task.bars);
  uint32_t fph[GATED_UP_DEPTH] = {};

  TiledMmaWide mma;
  auto thr = mma.get_thread_slice(tid);
  // Columns [0, BN) are the gate GEMM, [BN, 2*BN) the up GEMM.
  Tensor acc = partition_fragment_C(mma, Shape<Int<M_PAD>, Int<2 * BN>>{});
  Tensor cC = thr.partition_C(
      make_identity_tensor(Shape<Int<M_PAD>, Int<2 * BN>>{}));

  for (int t = 0; t < task.ntask; ++t) {
    int n = task.my[t].column;
    clear(acc);
    for (int i = 0; i < GATED_UP_TRIP; ++i) {
      int g = t * GATED_UP_TRIP + i, s = g % GATED_UP_DEPTH;
      full[s].wait(fph[s]);
      fph[s] ^= 1;

      // spec non_mma.a_scale: in place, (A*F)*S in bf16, 16 B vectors.
      // Same buffer wgmma reads through its descriptor -> async-proxy fence
      // plus a math-WG barrier before any wgmma touches the frame.
      __nv_bfloat16* Abase =
          reinterpret_cast<__nv_bfloat16*>(task.pool + OFF_A + s * A_FRAME_B);
      const __nv_bfloat162* S2 = reinterpret_cast<const __nv_bfloat162*>(
          task.pool + OFF_S + s * S_FRAME_B);
      #pragma unroll
      for (int it = 0; it < (M_PAD * BK) / (8 * 128); ++it) {  // 4 segs/thread
        int seg = tid + it * 128;
        int m = seg / (BK / 8), kk = (seg % (BK / 8)) << 3;
        int off = a_phys_elem(m, kk);                // physical swizzled offset
        uint4* p = reinterpret_cast<uint4*>(Abase + off);
        uint4 v = *p;
        __nv_bfloat162 f2 = __bfloat162bfloat162(task.F[m]);
        __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          h2[j] = __hmul2(__hmul2(h2[j], f2), S2[(kk >> 1) + j]);
        *p = v;
      }
      cutlass::arch::fence_view_async_shared();
      mathwg_sync();

      // spec math[0..1]: two 8-instruction batches off one A fill, one commit
      Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(
                      task.pool + OFF_A + s * A_FRAME_B)), SmemLayoutA{});
      Tensor sBwide = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(
                          task.pool + OFF_W1 + s * WUP_FRAME_B)), SmemLayoutBWide{});
      Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));
      Tensor tCrB = thr.make_fragment_B(thr.partition_B(sBwide));
      // FlashMLA's CuTe wrapper keeps the fence/arrive/commit ordering next
      // to the GEMM contract instead of repeating raw choreography here.
      ::sm90::gemm<false, -1, true, true>(mma, tCrA, tCrB, acc);
      // Release on retirement, PIPELINED depth-1 batches deep (CUTLASS's
      // wait<K_PIPE_MMAS>). wait_group<1> per stage made batch RETIREMENT the
      // cadence (~2 us/stage measured, job 541432 -- per-CTA rate identical to
      // the 4-warp TileLang kernels); consecutive wgmma batches on one
      // accumulator pipeline fine, so only the frame release needs the wait.
      if (g >= GATED_UP_DEPTH - 1) {
        warpgroup_wait<GATED_UP_DEPTH - 1>();
        __syncwarp();
        if ((tid & 31) == 0)
          empty[(g - (GATED_UP_DEPTH - 1)) % GATED_UP_DEPTH].arrive();
      }
    }
    // Retire everything before the epilogue reads acc.  A worker-queue CTA
    // may have only one GatedUp slot, so drain the tail frames explicitly before
    // its shared barrier pool is reused by the following DownResidual slot.
    warpgroup_wait<0>();
    __syncwarp();
    if ((tid & 31) == 0) {
      for (int tail = GATED_UP_TRIP - GATED_UP_DEPTH + 1; tail < GATED_UP_TRIP; ++tail)
        empty[tail % GATED_UP_DEPTH].arrive();
    }

    // spec non_mma.gelu_gate epilogue: f32 bias/gelu/product, bf16 pair store
    __nv_bfloat16* Cg = task.hidden + n;
    // Gate and up share one accumulator now. For m64n64k16, register r maps to
    // col = (r/4)*8 + (lane%4)*2 + (r%2): r and r+UP differ only in the column
    // group, so they are the SAME (row, col) with col apart by BN -- the
    // gate/up pair for one output element. Parity against torch is what checks
    // this, not the derivation.
    constexpr int UP = M_PAD * BN / 128;          // 16 = half the accumulator
    CUTE_UNROLL
    for (int e = 0; e < UP; e += 2) {
      int row = get<0>(cC(e)), col = get<1>(cC(e));
      float h0 = gelu_sig(acc(e)     + __bfloat162float(task.b1[n + col]))
               * (acc(e + UP)     + __bfloat162float(task.b2[n + col]));
      float h1 = gelu_sig(acc(e + 1) + __bfloat162float(task.b1[n + col + 1]))
               * (acc(e + 1 + UP) + __bfloat162float(task.b2[n + col + 1]));
      __nv_bfloat162 hv{__float2bfloat16(h0), __float2bfloat16(h1)};
      *reinterpret_cast<__nv_bfloat162*>(Cg + row * FF + col) = hv;
    }
    // all 128 threads' stores precede the single release
    mathwg_sync();
    if (tid == 0) counter_release(&task.counters[task.my[t].dependency]);
  }
}

// ------------------------------------------------------------------- DownResidual body
// DownResidual split rings: W_d depth 4 dep-free, A_h depth 4 counter-gated
// on counter[k-slice] >= 4. The decoupling IS the design thesis.
struct DownResidualTask {
  const TaskDesc* my;
  int ntask;
  const CUtensorMap *tmwd, *tmh;
  const __nv_bfloat16* g_gate;
  __nv_bfloat16* out;
  uint32_t* counters;
  float* partial;            // (DOWN_RESIDUAL_SPLIT-1, NUM_DOWN_RESIDUAL_TILES, M_PAD*BN) f32 scratch
  uint32_t* down_residual_counters;     // one per output tile; splits 1..S-1 arrive
  uint8_t* pool;
  uint64_t* bars;
  long long* dbg;
};

__device__ __forceinline__ void down_residual_weight_producer(
    const DownResidualTask& task) {
  auto* full_w  = BarrierViews::down_residual_weight_full(task.bars);
  auto* empty_w = BarrierViews::down_residual_weight_empty(task.bars);
  uint32_t ewph[DOWN_RESIDUAL_WEIGHT_DEPTH] = {};
  for (int g = 0; g < task.ntask * DOWN_RESIDUAL_TRIP_PER_SPLIT; ++g) {
    int t = g / DOWN_RESIDUAL_TRIP_PER_SPLIT, i = g % DOWN_RESIDUAL_TRIP_PER_SPLIT;
    // `pad` carries the split index; this CTA walks only its own K span.
    int n = task.my[t].column, k = task.my[t].split * DOWN_RESIDUAL_K_SPAN + i * BK;
    // dep-free weight ring first: it runs ahead through any counter stall
    int sw = g % DOWN_RESIDUAL_WEIGHT_DEPTH;
    if (g >= DOWN_RESIDUAL_WEIGHT_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty_w[sw]), ewph[sw], 5, g, task.dbg);
      ewph[sw] ^= 1;
    }
    // All lanes may wait on the producer ring, but only the elected lane owns
    // the transaction-barrier arrival and the corresponding TMA issue.  The
    // old single-thread dispatch implicitly provided this election; once the
    // whole warp runs the producer body it must be explicit here.
    if (cute::elect_one_sync()) {
      full_w[sw].arrive_and_expect_tx(W_FRAME_B);
      issue_tma_2d(task.tmwd, task.pool + OFF_WD + sw * W_FRAME_B,
             0, (n >> 5) * FF + k, reinterpret_cast<uint64_t*>(&full_w[sw]));
    }
  }
}

// The DownResidual activation producer owns its dependency poll and TMA issue in one
// warp.  Keeping these operations together avoids a scheduler->TMA sequence
// handoff (and the block fence that would otherwise be needed to order it).
__device__ __forceinline__ void down_residual_activation_producer(
    const DownResidualTask& task) {
  auto* full_a  = BarrierViews::down_residual_activation_full(task.bars);
  auto* empty_a = BarrierViews::down_residual_activation_empty(task.bars);
  uint32_t eaph[DOWN_RESIDUAL_ACTIVATION_DEPTH] = {};
  for (int g = 0; g < task.ntask * DOWN_RESIDUAL_TRIP_PER_SPLIT; ++g) {
    int t = g / DOWN_RESIDUAL_TRIP_PER_SPLIT, i = g % DOWN_RESIDUAL_TRIP_PER_SPLIT;
    // Absolute stage index over the full FF: the counter this split waits on
    // covers ITS k span, not the span a full-K task would have walked.
    int kb = task.my[t].split * DOWN_RESIDUAL_TRIP_PER_SPLIT + i;
    // gated activation ring: poll BEFORE reserving the hidden slice (spec L2)
    int counter_id = kb / (COUNTER_K / BK);
    counter_wait(&task.counters[counter_id], COUNTER_ARRIVE, g, task.dbg);
    int sa = g % DOWN_RESIDUAL_ACTIVATION_DEPTH;
    if (g >= DOWN_RESIDUAL_ACTIVATION_DEPTH) {
      wait_bar_wd(reinterpret_cast<uint64_t*>(&empty_a[sa]), eaph[sa], 6, g, task.dbg);
      eaph[sa] ^= 1;
    }
    __syncwarp();
    int k = kb * BK;
    uint8_t* Ah = task.pool + OFF_AH + sa * A_FRAME_B;
    if (cute::elect_one_sync()) {
      full_a[sa].arrive_and_expect_tx(A_FRAME_B);
      issue_tma_2d(task.tmh, Ah, k, 0, reinterpret_cast<uint64_t*>(&full_a[sa]));
    }
  }
}

__device__ __forceinline__ void down_residual_math(
    const DownResidualTask& task, int tid) {
  auto* full_w  = BarrierViews::down_residual_weight_full(task.bars);
  auto* empty_w = BarrierViews::down_residual_weight_empty(task.bars);
  auto* full_a  = BarrierViews::down_residual_activation_full(task.bars);
  auto* empty_a = BarrierViews::down_residual_activation_empty(task.bars);
  uint32_t fwph[DOWN_RESIDUAL_WEIGHT_DEPTH] = {}, faph[DOWN_RESIDUAL_ACTIVATION_DEPTH] = {};

  TiledMma mma;
  auto thr = mma.get_thread_slice(tid);
  Tensor acc = partition_fragment_C(mma, Shape<Int<M_PAD>, Int<BN>>{});
  Tensor cC = thr.partition_C(make_identity_tensor(Shape<Int<M_PAD>, Int<BN>>{}));

  for (int t = 0; t < task.ntask; ++t) {
    int n = task.my[t].column;
    clear(acc);
    for (int i = 0; i < DOWN_RESIDUAL_TRIP_PER_SPLIT; ++i) {
      int g = t * DOWN_RESIDUAL_TRIP_PER_SPLIT + i;
      int sw = g % DOWN_RESIDUAL_WEIGHT_DEPTH, sa = g % DOWN_RESIDUAL_ACTIVATION_DEPTH;
      full_w[sw].wait(fwph[sw]);
      fwph[sw] ^= 1;
      full_a[sa].wait(faph[sa]);
      faph[sa] ^= 1;
      Tensor sAh = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(
                       task.pool + OFF_AH + sa * A_FRAME_B)), SmemLayoutA{});
      Tensor sWd = make_tensor(make_smem_ptr(reinterpret_cast<BF*>(
                       task.pool + OFF_WD + sw * W_FRAME_B)), SmemLayoutB{});
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
    const int split = task.my[t].split;
    const int tile  = n / BN;
    if (split != 0) {
      float* P = task.partial
               + ((size_t)(split - 1) * NUM_DOWN_RESIDUAL_TILES + tile) * PARTIAL_ELEMS;
      CUTE_UNROLL
      for (int e = 0; e < size(acc); ++e)
        P[get<0>(cC(e)) * BN + get<1>(cC(e))] = acc(e);
      mathwg_sync();          // every math lane's store issued before the count
      if (tid == 0) counter_release(&task.down_residual_counters[tile]);
      continue;               // only split 0 owns `out`
    }
    if (tid == 0)
      counter_wait(&task.down_residual_counters[tile], DOWN_RESIDUAL_SPLIT - 1, t, task.dbg);
    mathwg_sync();            // thread 0's gpu-scope acquire covers the CTA
    CUTE_UNROLL
    for (int sp = 1; sp < DOWN_RESIDUAL_SPLIT; ++sp) {
      const float* P = task.partial
                     + ((size_t)(sp - 1) * NUM_DOWN_RESIDUAL_TILES + tile) * PARTIAL_ELEMS;
      CUTE_UNROLL
      for (int e = 0; e < size(acc); ++e)
        acc(e) += P[get<0>(cC(e)) * BN + get<1>(cC(e))];
    }

    // DownResidual epilogue: f32 gate multiply + residual add, bf16 store.
    // out is both R and C; each thread reads its element before writing it.
    __nv_bfloat16* Og = task.out + n;
    CUTE_UNROLL
    for (int e = 0; e < size(acc); e += 2) {
      int row = get<0>(cC(e)), col = get<1>(cC(e));
      __nv_bfloat162 r =
          *reinterpret_cast<const __nv_bfloat162*>(Og + row * D + col);
      float o0 = __bfloat162float(r.x)
               + acc(e)     * __bfloat162float(task.g_gate[n + col]);
      float o1 = __bfloat162float(r.y)
               + acc(e + 1) * __bfloat162float(task.g_gate[n + col + 1]);
      __nv_bfloat162 ov{__float2bfloat16(o0), __float2bfloat16(o1)};
      *reinterpret_cast<__nv_bfloat162*>(Og + row * D + col) = ov;
    }
  }
}

// ------------------------------------------------------------------ the kernel
__global__ void __launch_bounds__(WarpRoles::kThreads, 1)
    ffn_taskloop_kernel(
    const TaskDesc* __restrict__ table,
    const __grid_constant__ CUtensorMap tmx,
    const __grid_constant__ CUtensorMap tmwup,
    const __grid_constant__ CUtensorMap packed_gate_up_legacy_unused,
    const __grid_constant__ CUtensorMap tmwd,
    const __grid_constant__ CUtensorMap tmh,
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
    long long* __restrict__ dbg) {
  extern __shared__ uint8_t pool[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(pool + OFF_BARS);
  const int tid = threadIdx.x;
  const TaskDesc* my = table + blockIdx.x * MAX_TASKS_PER_CTA;
  // The launch geometry is fixed at 132 workers.  Bisection schedules keep
  // that geometry and mark unused workers with a sentinel row; idle workers
  // must return before initializing their private barrier pool.
  if (ref::is_sentinel(my[0].kind)) return;
  for (int slot = 0; slot < MAX_TASKS_PER_CTA; ++slot) {
    const TaskKind kind = my[slot].kind;
    if (ref::is_sentinel(kind)) break;
    // Warp roles are uniform across task types: warp 4 loads weights, warp 5
    // loads activations.  GatedUp shares one full barrier between them (two
    // arrivals), while DownResidual gives each ring its own.  Reinitialize the aliased
    // pool at the GatedUp->DownResidual worker seam.
      if (tid == 0) {
      auto* full_w = BarrierViews::down_residual_weight_full(bars);
      auto* empty_w = BarrierViews::down_residual_weight_empty(bars);
      if (ref::is_gated_up(kind)) {
        for (int i = 0; i < GATED_UP_DEPTH; ++i) {
          full_w[i].init(2);
          empty_w[i].init(kNumEpilogueWarps);
        }
      } else {
        for (int i = 0; i < DOWN_RESIDUAL_WEIGHT_DEPTH; ++i) {
          full_w[i].init(1);
          empty_w[i].init(kNumEpilogueWarps);
        }
        auto* full_a = BarrierViews::down_residual_activation_full(bars);
        auto* empty_a = BarrierViews::down_residual_activation_empty(bars);
        for (int i = 0; i < DOWN_RESIDUAL_ACTIVATION_DEPTH; ++i) {
          full_a[i].init(1);
          empty_a[i].init(kNumEpilogueWarps);
        }
      }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    if (ref::is_gated_up(kind)) {
      GatedUpTask task{my + slot, 1, &tmx, &tmwup, F, S, b1, b2, S, hidden,
                       counters, pool, bars, dbg};
      if (WarpRoles::is_math(tid)) {
        gated_up_math(task, tid);
      } else if (WarpRoles::is_weight_producer(tid)) {
        gated_up_weight_producer(task);
      } else if (WarpRoles::is_activation_producer(tid)) {
        gated_up_activation_producer(task);
      }
    } else {
      DownResidualTask task{my + slot, 1, &tmwd, &tmh, g_gate, out, counters,
                              down_residual_partial, down_residual_counters, pool, bars, dbg};
      if (WarpRoles::is_math(tid)) {
        down_residual_math(task, tid);
      } else if (WarpRoles::is_weight_producer(tid)) {
        down_residual_weight_producer(task);
      } else if (WarpRoles::is_activation_producer(tid)) {
        down_residual_activation_producer(task);
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

// ------------------------------------------------------------------- host side
static CUtensorMap enc2d(const void* p, uint64_t inner, uint64_t outer,
                         uint32_t box_inner, uint32_t box_outer,
                         CUtensorMapSwizzle sw, CUresult* rc) {
  CUtensorMap m{};
  uint64_t dims[2] = {inner, outer};
  uint64_t strides[1] = {inner * 2};  // bf16, row-major, 16B-multiple required
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t es[2] = {1, 1};
  *rc = cuTensorMapEncodeTiled(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims,
      strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return m;
}

}  // namespace ffn

extern "C" {

int ffn_taskloop_launch(const void* table, int n_ctas,
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
        ffn_taskloop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_B);
    if (e != cudaSuccess) return (int)e;
    attr_set = true;
  }
  CUresult rc = CUDA_SUCCESS;
  CUtensorMap tmx  = enc2d(x_pad,  D,  M_PAD, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;
  // Weights arrive PRE-BLOCKED and interleaved by the host: each row is
  // (N/32, K, 64) = [W1_tile(32), W2_tile(32)], so one 128B TMA replaces the
  // two 64B gate/up copies while preserving a contiguous task slab.
  // The natural (K, N) layout reads 64 B strips at an 8 KB stride, which caps
  // the machine near ~1 TB/s (job 541407: gu-only 33 us, dr-only 25 us, and
  // the TileLang composition sits at the same ~1.1 TB/s -- PLAN 4.9's 30-36%
  // MBU is the same pattern). Static weights make the relayout free, offline,
  // and planner-owned.
  CUtensorMap tmwup = enc2d(W1, 2 * BN, (uint64_t)(FF / BN) * D,
                            2 * BN, BK, CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;
  CUtensorMap packed_gate_up_legacy_unused{};
  if (rc) return 1000 + (int)rc;
  CUtensorMap tmwd = enc2d(Wd, BN, (uint64_t)(D / BN) * FF, BN, BK,
                           CU_TENSOR_MAP_SWIZZLE_64B, &rc);
  if (rc) return 1000 + (int)rc;
  CUtensorMap tmh  = enc2d(hidden, FF, M_PAD, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, &rc);
  if (rc) return 1000 + (int)rc;

  ffn_taskloop_kernel<<<N_CTAS, 224, SMEM_B, (cudaStream_t)stream>>>(
      (const TaskDesc*)table, tmx, tmwup, packed_gate_up_legacy_unused, tmwd, tmh,
      (const __nv_bfloat16*)F, (const __nv_bfloat16*)S,
      (const __nv_bfloat16*)b1, (const __nv_bfloat16*)b2,
      (const __nv_bfloat16*)g_gate,
      (__nv_bfloat16*)hidden, (__nv_bfloat16*)out, (uint32_t*)counters,
      (float*)down_residual_partial, (uint32_t*)down_residual_counters,
      (long long*)dbg);
  return (int)cudaGetLastError();
}

int counter_probe_launch(void* c, void* t0s, void* out_ns, int pairs,
                         void* stream) {
  ffn::counter_probe_kernel<<<pairs * 2, 32, 0, (cudaStream_t)stream>>>(
      (uint32_t*)c, (long long*)t0s, (long long*)out_ns);
  return (int)cudaGetLastError();
}

int ffn_taskloop_smem_bytes() { return ffn::SMEM_B; }

}  // extern "C"
