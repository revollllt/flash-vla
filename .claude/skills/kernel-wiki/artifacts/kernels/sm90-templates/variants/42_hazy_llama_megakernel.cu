// Template 42 -- the HazyResearch low-latency Llama megakernel, ported to the
// toolkit (sm90).
//
// A whole decode step of a Llama-3.2-1B-shaped model -- 16 layers of
// {rmsnorm+QKV+RoPE+KV-append, attention, o_proj+residual,
// rmsnorm+up/gate+SiLU, down+residual} and the rmsnorm+lm_head -- as ONE
// persistent kernel interpreting a per-SM instruction stream that a host
// planner emitted. Every mechanism of HazyResearch/Megakernels (MIT) is here
// under its own name, written against sm90_common.cuh instead of
// ThunderKittens, and the file builds, checks itself against a double
// reference, and times itself. Its header is kept current with the harness
// output (STATUS below).
//
// THE MACHINE, as upstream builds it
//
//   20 warps: 16 consumers, and one each of loader, storer, launcher,
//   controller. 104 registers per consumer thread, 64 per non-consumer.
//   A 2-deep instruction ring; each ring slot owns 32 dynamic semaphores,
//   a 128-slot timing record, a 4 KB scratch, and a physical-page order.
//   13 pages of 16 KB in dynamic shared memory.
//
//   controller  fetch instruction -> compute this instruction's page order
//               from the PREVIOUS instruction's declared release order ->
//               init the op's semaphores -> publish. Reclaims a ring slot by
//               waiting for every other warp's "finished" arrival, then
//               invalidates that slot's semaphores and flushes its timings.
//   loader      issues the op's TMA loads (weights, KV) into pages.
//   launcher    the tensor-core issue warp on Blackwell; here it is the KV
//               loader for attention and idle elsewhere, exactly as upstream
//               on H100.
//   consumers   the math; 16 warps.
//   storer      reduces partials from scratch, applies the epilogue, stores
//               through the async proxy, bumps the dependency counter.
//
// FIVE MECHANISMS THAT ARE THE POINT -- static per-SM instruction streams,
// pages released individually in an op-declared order, weights prefetched
// across the dependency, one global counter per (layer, op, slice), and the
// planner (round robin or cost-weighted list scheduling) -- are the portable
// rules on [kernel-megakernel-forms]; here each appears under its own name:
// `instructions[worker][i][32]`, `release_lid` / `page_finished`, a loader
// that never waits on a counter while the consumer's `gmem_wait` does,
// `Bar[layer][op][k]`, and `rr` / `dag`.  Upstream indexes by %smid;
// blockIdx.x is used here (grid == SM count, 1 CTA/SM, same thing without
// the trap).  Template 40 waited before loading anything, and lost.
//
// WHAT WAS CHANGED, AND WHY (each a deviation from upstream, none from the
// design):
//
//   * Activations and norm weights are read straight from global memory into
//     registers by the consumer warps (16 B per lane) instead of being staged
//     through page 0. They are L2-resident and a few KB; the page round trip
//     bought nothing on H100 and page 0 is released on entry.
//   * Attention uses all 16 consumer warps (one 16-key block each per stage)
//     with an LSE merge through page 0, instead of upstream's single warp on
//     mma. The instruction semantics, pages and counters are unchanged.
//   * KV cache layout is [layer][kv_head][pos][64] so a 16-key block is one
//     contiguous 2 KB bulk copy rather than a 4-D tensor map.
//   * The residual adds (o_proj, down) go through `cp.reduce.async.bulk
//     ... .add.noftz.bf16`, the toolkit spelling of upstream's
//     `tma::store_add_async`. Accumulation order across the 4 reduction
//     columns is therefore nondeterministic, in bf16; the reference rounds
//     each partial the same way and the check tolerance carries the rest.
//   * RoPE is the interleaved-pair form upstream requires (`interleave_rope`),
//     with cos/sin tables already pair-duplicated per element.
//
// UPSTREAM NUMBERS (its blog, 2025-05-27): Llama-3.2-1B bf16, batch 1, H100:
// "under one millisecond" per forward pass at "78% of memory bandwidth",
// ~2.5x vLLM and ~1.5x SGLang; B200 under 680 us. The model moves 2.47 GB of
// weights per token, so 78% of 3.35 TB/s is ~0.95 ms.
//
// STATUS (H100 SXM5, CUDA 13.1, clocks not pinned, 132 SMs; pos 1023, whole
// 16-layer step + lm_head, graph-captured, min of 3 x 20; correctness: logits
// within 2.7% of their rms of the double reference, argmax identical):
//
//   partials=1 (upstream's demo schedule)   1.244 ms   2.01 TB/s   73% of floor
//   partials=4                              1.085 ms   2.31 TB/s   84%
//   partials=8                              1.081 ms   2.32 TB/s   84%
//   partials=16                             1.125 ms   2.23 TB/s   81%
//   partials=8, sched=dag                   1.147 ms   2.18 TB/s   79%
//
// Floor: 0.906 ms for 2.51 GB [ld.bw.dev.dram]. Upstream's "under 1 ms at
// 78% of bandwidth" is 2.6 TB/s of the 3.35 TB/s datasheet peak; this file
// reaches 2.32 TB/s (69% of that peak) with all of the machine but none of
// ThunderKittens' register-tile matvec. The VM profiler (-DMK42_TIMING=1)
// shows lm_head instructions -- 60 blocks each -- streaming at the peak, and
// the loss concentrated in the short per-layer instructions and in attention
// with too few partials; a run with a whole layer's attention on 8 SMs idles
// the weight stream once the 3-stage prefetch is full, which is what the
// partials knob recovers.
//
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -o mk42 42_hazy_llama_megakernel.cu -lcuda && ./mk42
//   ./mk42 layers=2                run 2 layers + lm_head (fast check)
//   ./mk42 stop=qkv layers=1       truncated table: bisection per op kind
//   ./mk42 sched=dag               cost-weighted list scheduling
//   ./mk42 partials=4              split attention 4 ways + reduction op
//   -DMK42_TIMING=1                the VM's own per-instruction profiler
//
// CHECK-GRADE: reference
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: mbarrier\.inval\.shared::cta\.b64
// CHECK-PTX: cp\.async\.bulk\.tensor\.3d\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: cp\.async\.bulk\.shared::cluster\.global\.mbarrier::complete_tx::bytes
// CHECK-PTX: cp\.reduce\.async\.bulk\.global\.shared::cta\.bulk_group\.add\.noftz\.bf16
// CHECK-PTX: cp\.async\.bulk\.global\.shared::cta\.bulk_group
// CHECK-PTX: red\.release\.gpu\.global\.add\.u32
// CHECK-PTX: ld\.acquire\.gpu\.u32
// CHECK-PTX-COUNT: 4 mbarrier\.try_wait\.parity

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>
#include <functional>

#include "sm90_common.cuh"

// ------------------------------------------------------------------ model
// Llama-3.2-1B. Layer count is the one knob that shortens a check run.
#ifndef MK42_LAYERS
#define MK42_LAYERS 16
#endif
#ifndef MK42_TIMING
#define MK42_TIMING 0
#endif

namespace {

constexpr int kNumLayers = MK42_LAYERS;
constexpr int kHidden = 2048;
constexpr int kInter = 8192;
constexpr int kHeadDim = 64;
constexpr int kNumHeads = 32;
constexpr int kNumKvHeads = 8;
constexpr int kGqa = kNumHeads / kNumKvHeads;  // 4
constexpr int kVocab = 128256;
constexpr int kBlock = 16;        // output rows per matvec block, keys per KV block
constexpr int kQkvDim = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;  // 3072
constexpr int kNumQkvHeads = kNumHeads + 2 * kNumKvHeads;          // 48
constexpr float kRopeBase = 500000.f;
constexpr float kEps = 1e-5f;
constexpr int kMaxPartials = 132;  // one per SM at most
constexpr int kLseStride = 144;    // kMaxPartials rounded to 16

enum Opcode : int32_t {
  kOpNoOp = 0,
  kOpRmsQkvRopeAppend = 1,
  kOpPartialAttention = 2,
  kOpAttentionReduction = 3,
  kOpOProjResidual = 4,
  kOpRmsUpgateSilu = 5,
  kOpDownProjResidual = 6,
  kOpRmsLmHead = 7,
  kNumOpcodes = 8,
};

// --------------------------------------------------------------- VM config

struct Cfg {
  static constexpr int kStages = 2;            // instruction ring depth
  static constexpr int kInstrWidth = 32;       // 128 B per instruction
  static constexpr int kTimingWidth = 128;
  static constexpr int kDynSems = 32;
  static constexpr int kConsumerWarps = 16;
  static constexpr int kNumWarps = kConsumerWarps + 4;
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kConsumerRegs = 104;
  static constexpr int kNonConsumerRegs = 64;
  static constexpr int kScratchBytes = 4096;
  static constexpr int kPageSize = 16384;
  static constexpr int kNumPages = 13;
  static constexpr int kDynamicSmem = kNumPages * kPageSize + 1024;  // + alignment slack
  static constexpr uint32_t kSpinNanos = 20;
  static constexpr bool kTiming = MK42_TIMING != 0;
};

// Timing event slots, upstream's convention.
enum TEvent : int {
  kTControllerStart = 0, kTIFetchDone = 1, kTPageAllocDone = 2, kTSemsSetup = 3,
  kTControllerEnd = 4, kTLoaderStart = 5, kTLauncherStart = 7, kTStorerStart = 9,
  kTConsumerStart = 11,  // + 2 * warp, and + 1 for the end
  kTAtGmemWait = 44, kTDoneGmemWait = 45, kTAtGmemStore = 46, kTDoneGmemStore = 47,
  kTFirstLoad = 48, kTFirstUse = 49, kTFirstStore = 50,
  kTLastLoad = 51, kTLastUse = 52, kTLastStore = 53, kTOutputReady = 54,
};

struct alignas(128) InstructionState {
  int32_t instruction[Cfg::kInstrWidth];
  int32_t timings[Cfg::kTimingWidth];
  int32_t pid_order[Cfg::kNumPages];
  int32_t pad[((Cfg::kNumPages + 31) & ~31) - Cfg::kNumPages];
  uint64_t semaphores[Cfg::kDynSems];
  int32_t scratch[Cfg::kScratchBytes / 4];
};

// -------------------------------------------------------------- primitives

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(tmpl::smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_n(uint64_t* bar, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
               ::"r"(tmpl::smem_u32(bar)), "r"(n) : "memory");
}

// Bulk stores and reductions from shared memory; completion is the bulk group.
__device__ __forceinline__ void bulk_store_1d(void* gmem_dst, const void* smem_src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(gmem_dst), "r"(tmpl::smem_u32(smem_src)), "r"(bytes) : "memory");
}
// The residual add: the copy engine adds bf16 into global memory.
__device__ __forceinline__ void bulk_reduce_add_bf16(void* gmem_dst, const void* smem_src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;"
               ::"l"(gmem_dst), "r"(tmpl::smem_u32(smem_src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
// .read: the shared source may be reused. Full: the global write has landed.
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

// Global counters. The storer's release orders its completed bulk stores
// before the count; the consumer's acquire orders its activation reads after.
__device__ __forceinline__ void counter_add_release(uint32_t* p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t counter_load_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void spin_until(const uint32_t* p, uint32_t target) {
  while (counter_load_acquire(p) < target) { __nanosleep(Cfg::kSpinNanos); }
}

__device__ __forceinline__ uint64_t clock_now() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}

__device__ __forceinline__ void bf16x8_unpack(const uint4& raw, float* f) {
  const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm("shl.b32 %0, %2, 16;\n and.b32 %1, %2, 0xFFFF0000;" : "=f"(f[2 * i]), "=f"(f[2 * i + 1]) : "r"(r[i]));
  }
}
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Sixteen lanes (a half warp) each hold 8 partial sums for the same 8 rows;
// after this, lane l holds the 16-lane total for row (l >> 1) & 7. Eight
// shuffles instead of the forty a per-row reduction would take.
__device__ __forceinline__ float transpose_reduce_8x16(float (&v)[8]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const bool up = lane & 8;
    const float send = up ? v[i] : v[i + 4];
    const float keep = up ? v[i + 4] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const bool up = lane & 4;
    const float send = up ? v[i] : v[i + 2];
    const float keep = up ? v[i + 2] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
  }
  {
    const bool up = lane & 2;
    const float send = up ? v[0] : v[1];
    const float keep = up ? v[1] : v[0];
    v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
  }
  return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

}  // namespace

// ================================================================= globals
// Passed by value as a grid constant, tensor maps included.
struct Globals {
  // vm
  uint32_t* bar;                 // [layers][10][48]
  const int32_t* instructions;   // [workers][max_instr][32]
  int32_t* timings;              // [workers][max_instr][128]
  int32_t num_instructions;      // per worker (padded with NoOps)
  // weights: tensor maps, 3-D (col, row, layer) with a 256 x 16 box
  CUtensorMap qkv_w;    // [L][3072][2048]
  CUtensorMap o_w;      // [L][2048][2048]
  CUtensorMap up_w;     // [L][8192][2048]
  CUtensorMap gate_w;   // [L][8192][2048]
  CUtensorMap down_w;   // [L][2048][8192]
  CUtensorMap lm_head_w;  // [128256][2048], 2-D map wrapped as 3-D with 1 layer
  const __nv_bfloat16* attn_norm;     // [L][2048]
  const __nv_bfloat16* mlp_norm;      // [L][2048]
  const __nv_bfloat16* lm_head_norm;  // [2048]
  __nv_bfloat16* k_cache;  // [L][kv][max_seq][64]
  __nv_bfloat16* v_cache;
  const float* rope_cos;   // [max_seq][64], pair-duplicated
  const float* rope_sin;
  // activations
  __nv_bfloat16* hidden;      // [2048] residual stream, updated in place
  __nv_bfloat16* q_post_rope; // [2048]
  __nv_bfloat16* attn_out;    // [2048]
  float* attn_o_part;         // [32 heads][kMaxPartials][64]
  float* attn_lse_part;       // [32 heads][kLseStride]
  __nv_bfloat16* silu_out;    // [8192]
  __nv_bfloat16* logits;      // [128256]
  int32_t pos_id;
  int32_t max_seq;
  float attn_scale;
  int32_t skip_attn_reduction;
};

namespace {

__device__ __forceinline__ uint32_t* bar_ptr(const Globals& g, int layer, int op, int k) {
  return g.bar + (static_cast<int64_t>(layer) * 10 + op) * 48 + k;
}

// ================================================================ VM state

struct Vm {
  InstructionState* is;      // [kStages]
  uint64_t* instr_arrived;   // [kStages], 1 arrival (controller)
  uint64_t* instr_finished;  // [kStages], kNumWarps-1 arrivals
  uint64_t* page_finished;   // [kNumPages], 16 arrivals per instruction
  uint8_t* pages;
  int32_t index = 0, ring = 0;

  __device__ __forceinline__ int32_t* instruction() const { return is[ring].instruction; }
  __device__ __forceinline__ int32_t* timing() const { return is[ring].timings; }
  __device__ __forceinline__ uint64_t* sem(int i) const { return &is[ring].semaphores[i]; }
  __device__ __forceinline__ uint8_t* scratch() const { return reinterpret_cast<uint8_t*>(is[ring].scratch); }
  __device__ __forceinline__ int32_t pid(int lid) const { return is[ring].pid_order[lid]; }
  __device__ __forceinline__ uint8_t* page(int pid_) const { return pages + static_cast<int64_t>(pid_) * Cfg::kPageSize; }

  __device__ __forceinline__ uint32_t phase() const { return static_cast<uint32_t>(index / Cfg::kStages) & 1u; }
  __device__ __forceinline__ void await_instruction() { tmpl::wait_parity_backoff<>(&instr_arrived[ring], phase()); }
  __device__ __forceinline__ void next_instruction() {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) { tmpl::mbarrier_arrive(&instr_finished[ring]); }
    ++index;
    ring = (ring + 1) % Cfg::kStages;
  }
  // Instruction n may use a page once its phase n is complete: the page's
  // 16 releases by instruction n-1 (phase 0 is completed at init).
  __device__ __forceinline__ void wait_page_ready(int pid_) {
    tmpl::wait_parity(&page_finished[pid_], static_cast<uint32_t>(index) & 1u);
  }
  __device__ __forceinline__ void finish_page(int pid_, uint32_t count) { mbarrier_arrive_n(&page_finished[pid_], count); }
  __device__ __forceinline__ void warp_finish_page(int pid_, uint32_t count) {
    if ((threadIdx.x & 31) == 0) { finish_page(pid_, count); }
  }
  __device__ __forceinline__ void record(int event) const {
    if constexpr (Cfg::kTiming) {
      timing()[event] = static_cast<int32_t>(clock_now() - start_clock);
    }
  }
  uint64_t start_clock = 0;
};

// Consumer-only barrier ids; 0 is __syncthreads and off limits to a subset.
constexpr uint32_t kBarConsumers = 1;
constexpr uint32_t kBarConsumersB = 2;
__device__ __forceinline__ void consumer_sync(uint32_t id = kBarConsumers) {
  tmpl::named_barrier_sync(id, Cfg::kConsumerWarps * 32);
}

__device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }

// ============================================================ matvec pipe
//
// The shared body of five ops: stream 16-row weight blocks through a 3-stage
// ring of 4 pages each, dot them against an activation vector the consumers
// hold in registers, reduce the 16 per-warp partials in the storer.
//
// Page layout per stage: 4 pages, page p holds rows [16) x cols [p*512,+512)
// of the block, as two 256 x 16 TMA boxes side by side ([2][16][256]).
// Consumer warp w works page w/4, column slice (w%4)*128 of it: lanes 0-15
// even rows, 16-31 odd rows, lane%16 the 16-byte chunk within the slice.

struct MatVecSpec {
  int32_t layer;
  int32_t iters;         // number of 16-row blocks this instruction streams
  int32_t red_col0;      // first reduction column (down-proj: r * 2048)
};

template <class Op>
struct MatVec {
  static constexpr int kInStages = 3;
  static constexpr int kOutStages = 3;
  static constexpr int kStagePages = 4;
  static constexpr int kActPage = 0;      // released on entry (see header)
  static constexpr int kWeightPage0 = 1;
  static constexpr int kSemWeightsArrived = 1;   // +stage
  static constexpr int kSemWeightsFinished = 4;  // +stage
  static constexpr int kSemOutputsArrived = 7;   // +stage
  static constexpr int kSemOutputsFinished = 10; // +stage
  static constexpr int kSemCount = 13;
  static constexpr int kScratchPerWarp = 16 * 4;               // 16 rows of partials
  static constexpr int kScratchPerStage = kScratchPerWarp * Cfg::kConsumerWarps;  // 1 KB
  static constexpr int kStageBytes = kStagePages * Cfg::kPageSize;
  static_assert(kOutStages * kScratchPerStage + 64 <= Cfg::kScratchBytes, "scratch");

  __device__ static int release_lid(const int32_t* ins, int query) {
    // Pages not used by this instruction are released first, then the
    // activation page, then the weight stages in the order they drain --
    // upstream's tables, keyed on how many stages the instruction touches.
    const int iters = Op::spec_of(ins).iters;
    const int rem = iters % kInStages;
    if (iters == 1) { constexpr int o[13] = {5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4}; return o[query]; }
    if (iters == 2) { constexpr int o[13] = {9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8}; return o[query]; }
    if (rem == 1)   { constexpr int o[13] = {0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4}; return o[query]; }
    if (rem == 2)   { constexpr int o[13] = {0, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8}; return o[query]; }
    return query;
  }

  __device__ static int init_semaphores(const Vm& vm) {
    for (int i = 0; i < kInStages; ++i) {
      tmpl::mbarrier_init(vm.sem(kSemWeightsArrived + i), 1);
      tmpl::mbarrier_init(vm.sem(kSemWeightsFinished + i), Cfg::kConsumerWarps);
    }
    for (int i = 0; i < kOutStages; ++i) {
      tmpl::mbarrier_init(vm.sem(kSemOutputsArrived + i), Cfg::kConsumerWarps);
      tmpl::mbarrier_init(vm.sem(kSemOutputsFinished + i), 1);
    }
    return kSemCount;
  }

  // Loader: lane 0 streams weights, the other lanes release the pages this
  // instruction will never touch, one page per lane.
  __device__ static void loader(Vm& vm, const Globals& g) {
    const MatVecSpec sp = Op::spec_of(vm.instruction());
    const int needed = 1 + min(sp.iters, kInStages) * kStagePages;
    const int lane = lane_id();
    if (lane == 0) {
      // Page 0 is not used here: release it at once (see header).
      vm.wait_page_ready(vm.pid(kActPage));
      vm.finish_page(vm.pid(kActPage), Cfg::kConsumerWarps);
      int stage = 0;
      for (int it = 0; it < sp.iters; ++it) {
        // Reuse of a stage waits for the consumers' release of its previous
        // use; a fresh barrier passes the "previous phase" parity at once.
        tmpl::wait_parity(vm.sem(kSemWeightsFinished + stage), ((it / kInStages) & 1u) ^ 1u);
        tmpl::arrive_and_expect_tx(vm.sem(kSemWeightsArrived + stage), kStageBytes);
        #pragma unroll
        for (int p = 0; p < kStagePages; ++p) {
          const int pid_ = vm.pid(kWeightPage0 + stage * kStagePages + p);
          if (it < kInStages) { vm.wait_page_ready(pid_); }
          if (it == 0 && p == 0) { vm.record(kTFirstLoad); }
          if (it == sp.iters - 1 && p == kStagePages - 1) { vm.record(kTLastLoad); }
          Op::load_page(vm, g, sp, it, p, vm.page(pid_), vm.sem(kSemWeightsArrived + stage));
        }
        stage = (stage + 1) % kInStages;
      }
    } else if (lane >= needed && lane < Cfg::kNumPages) {
      const int pid_ = vm.pid(lane);
      vm.wait_page_ready(pid_);
      vm.finish_page(pid_, Cfg::kConsumerWarps);
    }
  }

  // Consumers: `act` is this lane's 8 activation values for its chunk.
  __device__ static void consumer(Vm& vm, const Globals& g, const float (&act)[8]) {
    const MatVecSpec sp = Op::spec_of(vm.instruction());
    const int warp = warp_id(), lane = lane_id();
    const int page_in_stage = warp / 4;
    const int slice = warp % 4;
    const int box = slice / 2;
    const int col_off = (slice % 2) * 128 + (lane % 16) * 8;   // element offset in the box row
    const int row_par = lane / 16;
    int in_stage = 0, out_stage = 0;
    for (int it = 0; it < sp.iters; ++it) {
      tmpl::wait_parity(vm.sem(kSemWeightsArrived + in_stage), (it / kInStages) & 1u);
      tmpl::wait_parity(vm.sem(kSemOutputsFinished + out_stage), ((it / kOutStages) & 1u) ^ 1u);
      if (it == 0) { vm.record(kTFirstUse); }
      const uint8_t* pg = vm.page(vm.pid(kWeightPage0 + in_stage * kStagePages + page_in_stage));
      const __nv_bfloat16* base = reinterpret_cast<const __nv_bfloat16*>(pg) + box * (16 * 256) + col_off;
      float acc[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = row_par + 2 * j;
        const uint4 raw = *reinterpret_cast<const uint4*>(base + row * 256);
        float w[8];
        bf16x8_unpack(raw, w);
        float a = 0.f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { a = fmaf(w[e], act[e], a); }
        acc[j] = a;
      }
      const float total = transpose_reduce_8x16(acc);
      float* out = reinterpret_cast<float*>(vm.scratch() + out_stage * kScratchPerStage + warp * kScratchPerWarp);
      if ((lane & 1) == 0) { out[2 * ((lane >> 1) & 7) + row_par] = total; }
      __syncwarp();
      if (lane == 0) {
        tmpl::mbarrier_arrive(vm.sem(kSemOutputsArrived + out_stage));
        tmpl::mbarrier_arrive(vm.sem(kSemWeightsFinished + in_stage));
      }
      if (it == sp.iters - 1) { vm.record(kTLastUse); }
      // Last use of this stage's pages: hand them to the next instruction.
      if (it >= sp.iters - kInStages) {
        #pragma unroll
        for (int p = 0; p < kStagePages; ++p) {
          vm.warp_finish_page(vm.pid(kWeightPage0 + in_stage * kStagePages + p), 1);
        }
      }
      in_stage = (in_stage + 1) % kInStages;
      out_stage = (out_stage + 1) % kOutStages;
    }
  }

  // Storer: lanes 0-15 each own one of the 16 output rows.
  __device__ static float reduce_partials(const Vm& vm, int out_stage) {
    const int lane = lane_id();
    const float* base = reinterpret_cast<const float*>(vm.scratch() + out_stage * kScratchPerStage);
    float s = 0.f;
    if (lane < 16) {
      #pragma unroll
      for (int w = 0; w < Cfg::kConsumerWarps; ++w) { s += base[w * 16 + lane]; }
    }
    return s;
  }
  __device__ static __nv_bfloat16* stage_out_smem(const Vm& vm, int out_stage) {
    return reinterpret_cast<__nv_bfloat16*>(vm.scratch() + out_stage * kScratchPerStage);
  }

  // Per block the storer only waits for its bulk store to have READ the
  // scratch stage (so the consumers may reuse it); the counters that
  // publish the instruction's outputs are bumped once at the end, after one
  // wait for every write to have landed. Upstream waits for each block's
  // write in turn, and the VM profiler showed that wait, at several
  // microseconds a block, throttling the whole up/gate pipeline.
  template <int IterScale>
  __device__ static void storer(Vm& vm, const Globals& g) {
    const MatVecSpec sp = Op::spec_of(vm.instruction());
    int out_stage = 0;
    for (int it = 0; it < sp.iters; ++it) {
      tmpl::wait_parity(vm.sem(kSemOutputsArrived + out_stage), (it / kOutStages) & 1u);
      if (it == 0) { vm.record(kTFirstStore); }
      if (it == sp.iters - 1) { vm.record(kTLastStore); }
      Op::store(vm, g, sp, it, out_stage);
      if ((it + 1) % IterScale == 0) {
        if (lane_id() == 0) {
          bulk_wait_read();
          for (int j = 0; j < IterScale; ++j) {
            tmpl::mbarrier_arrive(vm.sem(kSemOutputsFinished + (it - j) % kOutStages));
          }
        }
      }
      out_stage = (out_stage + 1) % kOutStages;
    }
    __syncwarp();
    if (lane_id() == 0) {
      vm.record(kTAtGmemStore);
      bulk_wait_all();
      Op::publish(vm, g, sp);
      vm.record(kTDoneGmemStore);
    }
  }
};

// Every consumer lane's activation chunk: elements [c, c+8) of a 2048-vector,
// where c is fixed by the warp's page/slice and the lane's chunk.
__device__ __forceinline__ int lane_act_offset() {
  const int warp = warp_id(), lane = lane_id();
  return (warp / 4) * 512 + (warp % 4) * 128 + (lane % 16) * 8;
}

// RMSNorm over a 2048-vector held 8-per-lane across 16 warps (lanes 0-15
// and 16-31 hold the same chunk; only one half contributes to the sum).
__device__ __forceinline__ void rms_norm_in_regs(float (&x)[8], const __nv_bfloat16* w, uint8_t* scratch_red) {
  const int warp = warp_id(), lane = lane_id();
  float ss = 0.f;
  if (lane < 16) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) { ss += x[e] * x[e]; }
  }
  ss = warp_sum(ss);
  float* red = reinterpret_cast<float*>(scratch_red);
  if (lane == 0) { red[warp] = ss; }
  consumer_sync(kBarConsumersB);
  float total = 0.f;
  #pragma unroll
  for (int i = 0; i < Cfg::kConsumerWarps; ++i) { total += red[i]; }
  const float scale = rsqrtf(total / static_cast<float>(kHidden) + kEps);
  float wf[8];
  bf16x8_unpack(*reinterpret_cast<const uint4*>(w + lane_act_offset()), wf);
  #pragma unroll
  for (int e = 0; e < 8; ++e) { x[e] = x[e] * scale * wf[e]; }
}

// The consumer prologue every matvec op shares: warp 0 lane 0 waits for the
// activation's producer (the dependency hop), everyone then loads its chunk.
template <class WaitFn>
__device__ __forceinline__ void consumer_load_act(Vm& vm, const __nv_bfloat16* src, float (&act)[8], WaitFn wait) {
  if (warp_id() == 0 && lane_id() == 0) {
    vm.record(kTAtGmemWait);
    wait();
    vm.record(kTDoneGmemWait);
  }
  consumer_sync();
  bf16x8_unpack(*reinterpret_cast<const uint4*>(src + lane_act_offset()), act);
}

// ------------------------------------------------------------- op 1: QKV
// instruction: [1] layer, [2] start_block, [3] end_block. Blocks of 16 rows
// of the stacked (q | k | v) projection; 4 blocks make one head.

struct OpRmsQkv {
  static constexpr int kExpectedArrivals = 512;  // down-proj blocks of the previous layer
  __device__ static MatVecSpec spec_of(const int32_t* ins) {
    return MatVecSpec{ins[1], ins[3] - ins[2], 0};
  }
  __device__ static void load_page(const Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int p,
                                   uint8_t* dst, uint64_t* sem) {
    const int32_t* ins = vm.instruction();
    const int row = (ins[2] + it) * kBlock;
    tmpl::tma_load_3d(&g.qkv_w, dst, p * 512, row, sp.layer, sem);
    tmpl::tma_load_3d(&g.qkv_w, dst + 8192, p * 512 + 256, row, sp.layer, sem);
  }
  __device__ static void store(Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int out_stage) {
    using P = MatVec<OpRmsQkv>;
    const int32_t* ins = vm.instruction();
    const int block = ins[2] + it;
    const int lane = lane_id();
    float v = P::reduce_partials(vm, out_stage);
    // RoPE on q and k blocks: interleaved pairs, cos/sin per element.
    constexpr int kVBlockStart = (kNumHeads + kNumKvHeads) * kHeadDim / kBlock;
    if (block < kVBlockStart) {
      const int d0 = (block % 4) * kBlock;
      const float c = g.rope_cos[static_cast<int64_t>(g.pos_id) * kHeadDim + d0 + (lane & 15)];
      const float s = g.rope_sin[static_cast<int64_t>(g.pos_id) * kHeadDim + d0 + (lane & 15)];
      const float pair = __shfl_xor_sync(0xffffffffu, v, 1);
      v = (lane & 1) ? (v * c + pair * s) : (v * c - pair * s);
    }
    __nv_bfloat16* out = P::stage_out_smem(vm, out_stage);
    if (lane < 16) { out[lane] = __float2bfloat16(v); }
    __syncwarp();
    if (lane == 0) {
      tmpl::fence_proxy_async_shared();
      constexpr int kKBlockStart = kNumHeads * kHeadDim / kBlock;
      if (block < kKBlockStart) {
        bulk_store_1d(g.q_post_rope + block * kBlock, out, 32);
      } else {
        const bool is_k = block < kVBlockStart;
        const int b = block - (is_k ? kKBlockStart : kVBlockStart);
        const int head = b / 4, d0 = (b % 4) * kBlock;
        __nv_bfloat16* cache = is_k ? g.k_cache : g.v_cache;
        cache += ((static_cast<int64_t>(sp.layer) * kNumKvHeads + head) * g.max_seq + g.pos_id) * kHeadDim + d0;
        bulk_store_1d(cache, out, 32);
      }
      bulk_commit();
    }
    __syncwarp();
  }
  __device__ static void publish(Vm& vm, const Globals& g, const MatVecSpec& sp) {
    const int32_t* ins = vm.instruction();
    for (int it = 0; it < sp.iters; ++it) {
      counter_add_release(bar_ptr(g, sp.layer, kOpRmsQkvRopeAppend - 1, (ins[2] + it) / 4), 1);
    }
  }
  // roles
  __device__ static int release_lid(const int32_t* ins, int q) { return MatVec<OpRmsQkv>::release_lid(ins, q); }
  __device__ static int init_semaphores(const Vm& vm) { return MatVec<OpRmsQkv>::init_semaphores(vm); }
  __device__ static void loader(Vm& vm, const Globals& g) { MatVec<OpRmsQkv>::loader(vm, g); }
  __device__ static void launcher(Vm&, const Globals&) {}
  __device__ static void consumer(Vm& vm, const Globals& g) {
    const MatVecSpec sp = spec_of(vm.instruction());
    float act[8];
    consumer_load_act(vm, g.hidden, act, [&] {
      if (sp.layer > 0) { spin_until(bar_ptr(g, sp.layer - 1, kOpDownProjResidual - 1, 0), kExpectedArrivals); }
    });
    rms_norm_in_regs(act, g.attn_norm + static_cast<int64_t>(sp.layer) * kHidden,
                     vm.scratch() + MatVec<OpRmsQkv>::kOutStages * MatVec<OpRmsQkv>::kScratchPerStage);
    MatVec<OpRmsQkv>::consumer(vm, g, act);
  }
  __device__ static void storer(Vm& vm, const Globals& g) { MatVec<OpRmsQkv>::storer<1>(vm, g); }
};

// ------------------------------------------------ op 4 / op 6: matvec-add
// instruction: [1] layer, [2] start_block, [3] end_block, [4] reduction_block.
// Output rows [start, end) x 16 of a K=2048 slice starting at
// reduction_block * 2048; the result is ADDED into hidden_states.

template <int kOpcode, int kPrevOpcode, int kExpected, bool kIsDown>
struct OpMatVecAdd {
  __device__ static MatVecSpec spec_of(const int32_t* ins) {
    return MatVecSpec{ins[1], ins[3] - ins[2], ins[4] * kHidden};
  }
  __device__ static void load_page(const Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int p,
                                   uint8_t* dst, uint64_t* sem) {
    const int32_t* ins = vm.instruction();
    const int row = (ins[2] + it) * kBlock;
    const CUtensorMap* map = kIsDown ? &g.down_w : &g.o_w;
    tmpl::tma_load_3d(map, dst, sp.red_col0 + p * 512, row, sp.layer, sem);
    tmpl::tma_load_3d(map, dst + 8192, sp.red_col0 + p * 512 + 256, row, sp.layer, sem);
  }
  __device__ static void store(Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int out_stage) {
    using P = MatVec<OpMatVecAdd>;
    const int32_t* ins = vm.instruction();
    const int block = ins[2] + it;
    const int lane = lane_id();
    const float v = P::reduce_partials(vm, out_stage);
    __nv_bfloat16* out = P::stage_out_smem(vm, out_stage);
    if (lane < 16) { out[lane] = __float2bfloat16(v); }
    __syncwarp();
    if (lane == 0) {
      tmpl::fence_proxy_async_shared();
      bulk_reduce_add_bf16(g.hidden + block * kBlock, out, 32);
      bulk_commit();
    }
    __syncwarp();
  }
  __device__ static void publish(Vm&, const Globals& g, const MatVecSpec& sp) {
    counter_add_release(bar_ptr(g, sp.layer, kOpcode - 1, 0), static_cast<uint32_t>(sp.iters));
  }
  __device__ static int release_lid(const int32_t* ins, int q) { return MatVec<OpMatVecAdd>::release_lid(ins, q); }
  __device__ static int init_semaphores(const Vm& vm) { return MatVec<OpMatVecAdd>::init_semaphores(vm); }
  __device__ static void loader(Vm& vm, const Globals& g) { MatVec<OpMatVecAdd>::loader(vm, g); }
  __device__ static void launcher(Vm&, const Globals&) {}
  __device__ static void consumer(Vm& vm, const Globals& g) {
    const MatVecSpec sp = spec_of(vm.instruction());
    const int32_t* ins = vm.instruction();
    const __nv_bfloat16* src = (kIsDown ? g.silu_out : g.attn_out) + sp.red_col0;
    float act[8];
    consumer_load_act(vm, src, act, [&] {
      spin_until(bar_ptr(g, sp.layer, kPrevOpcode - 1, ins[4]), kExpected);
    });
    MatVec<OpMatVecAdd>::consumer(vm, g, act);
  }
  __device__ static void storer(Vm& vm, const Globals& g) { MatVec<OpMatVecAdd>::template storer<1>(vm, g); }
};
using OpOProj = OpMatVecAdd<kOpOProjResidual, kOpAttentionReduction, kNumHeads, false>;
using OpDownProj = OpMatVecAdd<kOpDownProjResidual, kOpRmsUpgateSilu, kHidden / kBlock, true>;

// ------------------------------------------------------------ op 5: up/gate
// instruction: [1] layer, [2] n, [3..3+n) block indices. Each block is
// streamed twice, up then gate; the storer combines the pair.

struct OpRmsUpgate {
  static constexpr int kExpectedArrivals = kHidden / kBlock;  // o_proj blocks
  __device__ static MatVecSpec spec_of(const int32_t* ins) { return MatVecSpec{ins[1], 2 * ins[2], 0}; }
  __device__ static void load_page(const Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int p,
                                   uint8_t* dst, uint64_t* sem) {
    const int32_t* ins = vm.instruction();
    const int row = ins[3 + it / 2] * kBlock;
    const CUtensorMap* map = (it % 2 == 0) ? &g.up_w : &g.gate_w;
    tmpl::tma_load_3d(map, dst, p * 512, row, sp.layer, sem);
    tmpl::tma_load_3d(map, dst + 8192, p * 512 + 256, row, sp.layer, sem);
  }
  __device__ static void store(Vm& vm, const Globals& g, const MatVecSpec& sp, int it, int out_stage) {
    using P = MatVec<OpRmsUpgate>;
    if (it % 2 == 0) { return; }  // up: stays in scratch until gate arrives
    const int32_t* ins = vm.instruction();
    const int block = ins[3 + it / 2];
    const int lane = lane_id();
    const float up = P::reduce_partials(vm, (it - 1) % P::kOutStages);
    const float gate = P::reduce_partials(vm, out_stage);
    __nv_bfloat16* out = P::stage_out_smem(vm, out_stage);
    if (lane < 16) { out[lane] = __float2bfloat16(silu(gate) * up); }
    __syncwarp();
    if (lane == 0) {
      tmpl::fence_proxy_async_shared();
      bulk_store_1d(g.silu_out + block * kBlock, out, 32);
      bulk_commit();
    }
    __syncwarp();
  }
  // Counted per reduction column of the down projection, so a down
  // instruction on column c starts as soon as column c's blocks are done.
  __device__ static void publish(Vm& vm, const Globals& g, const MatVecSpec& sp) {
    const int32_t* ins = vm.instruction();
    uint32_t per_col[kInter / kHidden] = {};
    for (int i = 0; i < sp.iters / 2; ++i) { per_col[ins[3 + i] * kBlock / kHidden]++; }
    for (int c = 0; c < kInter / kHidden; ++c) {
      if (per_col[c]) { counter_add_release(bar_ptr(g, sp.layer, kOpRmsUpgateSilu - 1, c), per_col[c]); }
    }
  }
  __device__ static int release_lid(const int32_t* ins, int q) { return MatVec<OpRmsUpgate>::release_lid(ins, q); }
  __device__ static int init_semaphores(const Vm& vm) { return MatVec<OpRmsUpgate>::init_semaphores(vm); }
  __device__ static void loader(Vm& vm, const Globals& g) { MatVec<OpRmsUpgate>::loader(vm, g); }
  __device__ static void launcher(Vm&, const Globals&) {}
  __device__ static void consumer(Vm& vm, const Globals& g) {
    const MatVecSpec sp = spec_of(vm.instruction());
    float act[8];
    consumer_load_act(vm, g.hidden, act, [&] {
      spin_until(bar_ptr(g, sp.layer, kOpOProjResidual - 1, 0), kExpectedArrivals);
    });
    rms_norm_in_regs(act, g.mlp_norm + static_cast<int64_t>(sp.layer) * kHidden,
                     vm.scratch() + MatVec<OpRmsUpgate>::kOutStages * MatVec<OpRmsUpgate>::kScratchPerStage);
    MatVec<OpRmsUpgate>::consumer(vm, g, act);
  }
  __device__ static void storer(Vm& vm, const Globals& g) { MatVec<OpRmsUpgate>::storer<2>(vm, g); }
};

// ------------------------------------------------------------ op 7: lm_head
// instruction: [1] start_block, [2] end_block over the 8016 vocab blocks.

struct OpRmsLmHead {
  static constexpr int kExpectedArrivals = 512;
  __device__ static MatVecSpec spec_of(const int32_t* ins) { return MatVecSpec{0, ins[2] - ins[1], 0}; }
  __device__ static void load_page(const Vm& vm, const Globals& g, const MatVecSpec&, int it, int p,
                                   uint8_t* dst, uint64_t* sem) {
    const int32_t* ins = vm.instruction();
    const int row = (ins[1] + it) * kBlock;
    tmpl::tma_load_3d(&g.lm_head_w, dst, p * 512, row, 0, sem);
    tmpl::tma_load_3d(&g.lm_head_w, dst + 8192, p * 512 + 256, row, 0, sem);
  }
  __device__ static void store(Vm& vm, const Globals& g, const MatVecSpec&, int it, int out_stage) {
    using P = MatVec<OpRmsLmHead>;
    const int32_t* ins = vm.instruction();
    const int block = ins[1] + it;
    const int lane = lane_id();
    const float v = P::reduce_partials(vm, out_stage);
    __nv_bfloat16* out = P::stage_out_smem(vm, out_stage);
    if (lane < 16) { out[lane] = __float2bfloat16(v); }
    __syncwarp();
    if (lane == 0) {
      tmpl::fence_proxy_async_shared();
      if (it == 0) { vm.record(kTOutputReady); }
      bulk_store_1d(g.logits + block * kBlock, out, 32);
      bulk_commit();
    }
    __syncwarp();
  }
  __device__ static void publish(Vm&, const Globals&, const MatVecSpec&) {}
  __device__ static int release_lid(const int32_t* ins, int q) { return MatVec<OpRmsLmHead>::release_lid(ins, q); }
  __device__ static int init_semaphores(const Vm& vm) { return MatVec<OpRmsLmHead>::init_semaphores(vm); }
  __device__ static void loader(Vm& vm, const Globals& g) { MatVec<OpRmsLmHead>::loader(vm, g); }
  __device__ static void launcher(Vm&, const Globals&) {}
  __device__ static void consumer(Vm& vm, const Globals& g) {
    float act[8];
    consumer_load_act(vm, g.hidden, act, [&] {
      spin_until(bar_ptr(g, kNumLayers - 1, kOpDownProjResidual - 1, 0), kExpectedArrivals);
    });
    rms_norm_in_regs(act, g.lm_head_norm,
                     vm.scratch() + MatVec<OpRmsLmHead>::kOutStages * MatVec<OpRmsLmHead>::kScratchPerStage);
    MatVec<OpRmsLmHead>::consumer(vm, g, act);
  }
  __device__ static void storer(Vm& vm, const Globals& g) { MatVec<OpRmsLmHead>::storer<1>(vm, g); }
};

// ------------------------------------------------------- op 2: attention
// instruction: [1] layer, [2] kv_head, [3] num_partials, [4] partial_idx.
// One KV head, its 4 query heads (GQA), a contiguous range of 16-key blocks.
//
// Pages: 0 holds the per-warp partial outputs for the merge; 1..12 are a
// 3-stage ring of 16 KV blocks (4 KB: K then V) each. Scratch: q (512 B),
// per-warp (m, l) (512 B), the merged O (1 KB), the merged LSE (64 B).
//
// The launcher streams KV blocks; the LAST block of the sequence holds the
// key/value this layer's QKV op is still writing, so only that load waits on
// the counter -- every earlier block is in flight before the dependency is
// even checked.

struct OpAttention {
  static constexpr int kBlocksPerStage = 16;
  static constexpr int kStages = 3;
  static constexpr int kStagePages = 4;
  static constexpr int kPartialPage = 0;
  static constexpr int kSemKvArrived = 0;    // +stage
  static constexpr int kSemKvFinished = 3;   // +stage
  static constexpr int kSemOReady = 6;
  static constexpr int kScrQ = 0;
  static constexpr int kScrML = 512;
  static constexpr int kScrO = 1024;
  static constexpr int kScrL = 2048;
  static constexpr int kBlockBytes = 2 * kBlock * kHeadDim * 2;  // K and V, 4 KB

  struct Range { int start, end, stage_iters; };
  __device__ static Range range_of(const Globals& g, const int32_t* ins) {
    const int total = (g.pos_id + 1 + kBlock - 1) / kBlock;
    const int per = (total + ins[3] - 1) / ins[3];
    const int start = ins[4] * per;
    const int end = min(start + per, total);
    const int n = max(0, end - start);
    return Range{start, end, (n + kBlocksPerStage - 1) / kBlocksPerStage};
  }
  __device__ static uint8_t* kv_slot(const Vm& vm, int stage, int j) {
    return vm.page(vm.pid(1 + stage * kStagePages + j / 4)) + (j % 4) * kBlockBytes;
  }

  __device__ static int release_lid(const int32_t*, int q) {
    constexpr int o[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0};
    return o[q];
  }
  __device__ static int init_semaphores(const Vm& vm) {
    for (int i = 0; i < kStages; ++i) {
      tmpl::mbarrier_init(vm.sem(kSemKvArrived + i), 1);
      tmpl::mbarrier_init(vm.sem(kSemKvFinished + i), Cfg::kConsumerWarps);
    }
    tmpl::mbarrier_init(vm.sem(kSemOReady), Cfg::kConsumerWarps);
    return 7;
  }
  __device__ static void loader(Vm& vm, const Globals& g) {
    // The launcher owns the KV stream here; the loader only releases the
    // ring stages this instruction will not use.
    const Range r = range_of(g, vm.instruction());
    const int lane = lane_id();
    const int used = min(r.stage_iters, kStages);
    if (lane >= 1 + used * kStagePages && lane < Cfg::kNumPages) {
      vm.wait_page_ready(vm.pid(lane));
      vm.finish_page(vm.pid(lane), Cfg::kConsumerWarps);
    }
  }
  __device__ static void launcher(Vm& vm, const Globals& g) {
    if (lane_id() != 0) { return; }
    const int32_t* ins = vm.instruction();
    const int layer = ins[1], kvh = ins[2];
    const Range r = range_of(g, ins);
    const int last_block = (g.pos_id + 1 + kBlock - 1) / kBlock - 1;
    const __nv_bfloat16* kbase = g.k_cache + (static_cast<int64_t>(layer) * kNumKvHeads + kvh) * g.max_seq * kHeadDim;
    const __nv_bfloat16* vbase = g.v_cache + (static_cast<int64_t>(layer) * kNumKvHeads + kvh) * g.max_seq * kHeadDim;
    for (int t = 0; t < r.stage_iters; ++t) {
      const int stage = t % kStages;
      tmpl::wait_parity(vm.sem(kSemKvFinished + stage), ((t / kStages) & 1u) ^ 1u);
      if (t < kStages) {
        for (int p = 0; p < kStagePages; ++p) { vm.wait_page_ready(vm.pid(1 + stage * kStagePages + p)); }
      }
      const int first = r.start + t * kBlocksPerStage;
      const int n = min(kBlocksPerStage, r.end - first);
      tmpl::arrive_and_expect_tx(vm.sem(kSemKvArrived + stage), n * kBlockBytes);
      for (int j = 0; j < n; ++j) {
        const int blk = first + j;
        if (blk == last_block) {
          // This block holds position pos_id: wait for this layer's K and V
          // append (4 blocks of 16 dims per head) before touching it.
          vm.record(kTAtGmemWait);
          spin_until(bar_ptr(g, layer, kOpRmsQkvRopeAppend - 1, kNumHeads + kvh), 4);
          spin_until(bar_ptr(g, layer, kOpRmsQkvRopeAppend - 1, kNumHeads + kNumKvHeads + kvh), 4);
          vm.record(kTDoneGmemWait);
        }
        uint8_t* slot = kv_slot(vm, stage, j);
        if (t == 0 && j == 0) { vm.record(kTFirstLoad); }
        tmpl::bulk_load_1d(slot, kbase + static_cast<int64_t>(blk) * kBlock * kHeadDim, kBlock * kHeadDim * 2,
                           vm.sem(kSemKvArrived + stage));
        tmpl::bulk_load_1d(slot + kBlock * kHeadDim * 2, vbase + static_cast<int64_t>(blk) * kBlock * kHeadDim,
                           kBlock * kHeadDim * 2, vm.sem(kSemKvArrived + stage));
      }
    }
    vm.record(kTLastLoad);
  }

  __device__ static void consumer(Vm& vm, const Globals& g) {
    const int32_t* ins = vm.instruction();
    const int layer = ins[1], kvh = ins[2];
    const int warp = warp_id(), lane = lane_id();
    const Range r = range_of(g, ins);
    const int q_head0 = kvh * kGqa;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(vm.scratch() + kScrQ);

    if (warp == 0) {
      if (lane == 0) {
        vm.record(kTAtGmemWait);
        for (int h = 0; h < kGqa; ++h) { spin_until(bar_ptr(g, layer, kOpRmsQkvRopeAppend - 1, q_head0 + h), 4); }
        vm.record(kTDoneGmemWait);
      }
      __syncwarp();
      // 4 heads x 64 dims of q: 512 B, one 16-byte load per lane.
      *reinterpret_cast<uint4*>(q_s + lane * 8) =
          *reinterpret_cast<const uint4*>(g.q_post_rope + q_head0 * kHeadDim + lane * 8);
    }
    consumer_sync();

    // Lane l: key l/2 of the block, dims [(l&1)*32, +32); softmax in exp2.
    const float temp = g.attn_scale * 1.4426950408889634f;
    const int key = lane >> 1, half = lane & 1;
    float m[kGqa], l[kGqa], o[kGqa][2];
    #pragma unroll
    for (int h = 0; h < kGqa; ++h) { m[h] = -1e30f; l[h] = 0.f; o[h][0] = 0.f; o[h][1] = 0.f; }

    for (int t = 0; t < r.stage_iters; ++t) {
      const int stage = t % kStages;
      tmpl::wait_parity(vm.sem(kSemKvArrived + stage), (t / kStages) & 1u);
      if (t == 0) { vm.record(kTFirstUse); }
      const int blk = r.start + t * kBlocksPerStage + warp;
      if (blk < r.end) {
        const uint8_t* slot = kv_slot(vm, stage, warp);
        const __nv_bfloat16* K = reinterpret_cast<const __nv_bfloat16*>(slot);
        const __nv_bfloat16* V = reinterpret_cast<const __nv_bfloat16*>(slot + kBlock * kHeadDim * 2);
        // S[h] for this lane's key over its 32 dims, then the partner's half.
        float s[kGqa] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float kf[8];
          bf16x8_unpack(*reinterpret_cast<const uint4*>(K + key * kHeadDim + half * 32 + c * 8), kf);
          #pragma unroll
          for (int h = 0; h < kGqa; ++h) {
            float qf[8];
            bf16x8_unpack(*reinterpret_cast<const uint4*>(q_s + h * kHeadDim + half * 32 + c * 8), qf);
            #pragma unroll
            for (int e = 0; e < 8; ++e) { s[h] = fmaf(qf[e], kf[e], s[h]); }
          }
        }
        const int pos = blk * kBlock + key;
        #pragma unroll
        for (int h = 0; h < kGqa; ++h) {
          s[h] += __shfl_xor_sync(0xffffffffu, s[h], 1);
          s[h] = (pos <= g.pos_id) ? s[h] * temp : -1e30f;
        }
        // Block max and sum over the 16 keys (lane bits 1..4), per head.
        #pragma unroll
        for (int h = 0; h < kGqa; ++h) {
          float bm = s[h];
          #pragma unroll
          for (int off = 2; off < 32; off <<= 1) { bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off)); }
          const float m_new = fmaxf(m[h], bm);
          const float p = exp2f(s[h] - m_new);
          float bs = p;
          #pragma unroll
          for (int off = 2; off < 32; off <<= 1) { bs += __shfl_xor_sync(0xffffffffu, bs, off); }
          const float rescale = exp2f(m[h] - m_new);
          l[h] = l[h] * rescale + bs;
          o[h][0] *= rescale; o[h][1] *= rescale;
          m[h] = m_new;
          s[h] = p;  // keep p for the PV pass
        }
        // O[h][2 dims per lane] += sum over keys p[h][k] * V[k][d].
        #pragma unroll
        for (int k = 0; k < kBlock; ++k) {
          float vf[2];
          {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(V + k * kHeadDim + lane * 2);
            asm("shl.b32 %0, %2, 16;\n and.b32 %1, %2, 0xFFFF0000;" : "=f"(vf[0]), "=f"(vf[1]) : "r"(raw));
          }
          #pragma unroll
          for (int h = 0; h < kGqa; ++h) {
            const float p = __shfl_sync(0xffffffffu, s[h], 2 * k);
            o[h][0] = fmaf(p, vf[0], o[h][0]);
            o[h][1] = fmaf(p, vf[1], o[h][1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) { tmpl::mbarrier_arrive(vm.sem(kSemKvFinished + stage)); }
      if (t >= r.stage_iters - kStages) {
        #pragma unroll
        for (int p = 0; p < kStagePages; ++p) { vm.warp_finish_page(vm.pid(1 + stage * kStagePages + p), 1); }
      }
    }
    vm.record(kTLastUse);

    // Merge the 16 warps: partial O to page 0 as [warp][head][64] fp32,
    // (m, l) to scratch; then 256 outputs spread over 16 warps x 16 lanes.
    float* part = reinterpret_cast<float*>(vm.page(vm.pid(kPartialPage)));
    float* ml = reinterpret_cast<float*>(vm.scratch() + kScrML);
    #pragma unroll
    for (int h = 0; h < kGqa; ++h) {
      *reinterpret_cast<float2*>(part + (warp * kGqa + h) * kHeadDim + lane * 2) = make_float2(o[h][0], o[h][1]);
    }
    if (lane < kGqa) { ml[warp * 8 + lane] = m[lane]; ml[warp * 8 + 4 + lane] = l[lane]; }
    consumer_sync();
    if (lane < 16) {
      const int idx = warp * 16 + lane;
      const int h = idx / kHeadDim, d = idx % kHeadDim;
      float mm = -1e30f;
      #pragma unroll
      for (int w = 0; w < Cfg::kConsumerWarps; ++w) { mm = fmaxf(mm, ml[w * 8 + h]); }
      float ll = 0.f, oo = 0.f;
      #pragma unroll
      for (int w = 0; w < Cfg::kConsumerWarps; ++w) {
        const float sc = exp2f(ml[w * 8 + h] - mm);
        ll += ml[w * 8 + 4 + h] * sc;
        oo += part[(w * kGqa + h) * kHeadDim + d] * sc;
      }
      float* O = reinterpret_cast<float*>(vm.scratch() + kScrO);
      float* L = reinterpret_cast<float*>(vm.scratch() + kScrL);
      const bool empty = (r.end <= r.start);
      O[idx] = empty ? 0.f : oo / ll;
      if (d == 0) { L[h] = empty ? -1e30f : mm + log2f(ll); }
    }
    __syncwarp();
    // Page 0 and the merged result are both done with; hand them on.
    vm.warp_finish_page(vm.pid(kPartialPage), 1);
    if (lane == 0) { tmpl::mbarrier_arrive(vm.sem(kSemOReady)); }
  }

  __device__ static void storer(Vm& vm, const Globals& g) {
    const int32_t* ins = vm.instruction();
    const int layer = ins[1], kvh = ins[2], num_partials = ins[3], partial = ins[4];
    const int lane = lane_id();
    const int q_head0 = kvh * kGqa;
    tmpl::wait_parity(vm.sem(kSemOReady), 0);
    vm.record(kTOutputReady);
    const float* O = reinterpret_cast<const float*>(vm.scratch() + kScrO);
    const float* L = reinterpret_cast<const float*>(vm.scratch() + kScrL);
    if (g.skip_attn_reduction) {
      // 4 x 64 bf16 = 512 B: one 16-byte store per lane.
      uint4 packed;
      uint32_t* pw = reinterpret_cast<uint32_t*>(&packed);
      #pragma unroll
      for (int i = 0; i < 4; ++i) { pw[i] = pack_bf16x2(O[lane * 8 + 2 * i], O[lane * 8 + 2 * i + 1]); }
      *reinterpret_cast<uint4*>(g.attn_out + q_head0 * kHeadDim + lane * 8) = packed;
      __syncwarp();
      if (lane == 0) {
        __threadfence();
        vm.record(kTAtGmemStore);
        counter_add_release(bar_ptr(g, layer, kOpAttentionReduction - 1, 0), kGqa);
        vm.record(kTDoneGmemStore);
      }
    } else {
      // fp32 partial per head, plus its LSE, for the reduction op.
      #pragma unroll
      for (int h = 0; h < kGqa; ++h) {
        float* dst = g.attn_o_part + (static_cast<int64_t>(q_head0 + h) * kMaxPartials + partial) * kHeadDim;
        *reinterpret_cast<float2*>(dst + lane * 2) = make_float2(O[h * kHeadDim + lane * 2], O[h * kHeadDim + lane * 2 + 1]);
      }
      if (lane < kGqa) { g.attn_lse_part[(q_head0 + lane) * kLseStride + partial] = L[lane]; }
      __syncwarp();
      if (lane < kGqa) {
        __threadfence();
        counter_add_release(bar_ptr(g, layer, kOpPartialAttention - 1, q_head0 + lane), 1);
      }
    }
    (void)num_partials;
  }
};

// ------------------------------------------------- op 3: attention reduce
// instruction: [1] layer, [2] q_head_start, [3] num_partials, [4] is_terminal,
// [5] list length n, [6..6+n) partial indices. Four heads per instruction.
// Page 0 holds the gathered O partials [4][n][64] fp32; scratch the LSEs.

struct OpAttentionReduction {
  static constexpr int kSemArrived = 0;
  static constexpr int kSemOReady = 1;
  static constexpr int kMaxList = Cfg::kInstrWidth - 6;
  __device__ static int release_lid(const int32_t*, int q) {
    constexpr int o[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0};
    return o[q];
  }
  __device__ static int init_semaphores(const Vm& vm) {
    tmpl::mbarrier_init(vm.sem(kSemArrived), 1);
    tmpl::mbarrier_init(vm.sem(kSemOReady), kGqa);
    return 2;
  }
  __device__ static void loader(Vm& vm, const Globals&) {
    const int lane = lane_id();
    if (lane >= 1 && lane < Cfg::kNumPages) {
      vm.wait_page_ready(vm.pid(lane));
      vm.finish_page(vm.pid(lane), Cfg::kConsumerWarps);
    }
  }
  __device__ static void launcher(Vm& vm, const Globals& g) {
    if (lane_id() != 0) { return; }
    const int32_t* ins = vm.instruction();
    const int layer = ins[1], head0 = ins[2], n = ins[5];
    vm.record(kTAtGmemWait);
    for (int h = 0; h < kGqa; ++h) { spin_until(bar_ptr(g, layer, kOpPartialAttention - 1, head0 + h), ins[3]); }
    vm.record(kTDoneGmemWait);
    vm.wait_page_ready(vm.pid(0));
    uint8_t* pg = vm.page(vm.pid(0));
    tmpl::arrive_and_expect_tx(vm.sem(kSemArrived), kGqa * n * kHeadDim * 4);
    for (int h = 0; h < kGqa; ++h) {
      for (int i = 0; i < n; ++i) {
        const float* src = g.attn_o_part + (static_cast<int64_t>(head0 + h) * kMaxPartials + ins[6 + i]) * kHeadDim;
        tmpl::bulk_load_1d(pg + (h * n + i) * kHeadDim * 4, src, kHeadDim * 4, vm.sem(kSemArrived));
      }
    }
  }
  __device__ static void consumer(Vm& vm, const Globals& g) {
    const int32_t* ins = vm.instruction();
    const int head0 = ins[2], n = ins[5];
    const int warp = warp_id(), lane = lane_id();
    if (warp < kGqa) {
      tmpl::wait_parity(vm.sem(kSemArrived), 0);
      const float* part = reinterpret_cast<const float*>(vm.page(vm.pid(0))) + warp * n * kHeadDim;
      const float* lse = g.attn_lse_part + (head0 + warp) * kLseStride;
      float mm = -1e30f;
      for (int i = 0; i < n; ++i) { mm = fmaxf(mm, lse[ins[6 + i]]); }
      float ll = 0.f, o0 = 0.f, o1 = 0.f;
      for (int i = 0; i < n; ++i) {
        const float sc = exp2f(lse[ins[6 + i]] - mm);
        ll += sc;
        o0 += part[i * kHeadDim + lane * 2] * sc;
        o1 += part[i * kHeadDim + lane * 2 + 1] * sc;
      }
      __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(vm.scratch()) + warp * kHeadDim;
      *reinterpret_cast<uint32_t*>(out + lane * 2) = pack_bf16x2(o0 / ll, o1 / ll);
      __syncwarp();
      if (lane == 0) { tmpl::mbarrier_arrive(vm.sem(kSemOReady)); }
    }
    vm.warp_finish_page(vm.pid(0), 1);
  }
  __device__ static void storer(Vm& vm, const Globals& g) {
    const int32_t* ins = vm.instruction();
    const int layer = ins[1], head0 = ins[2];
    const int lane = lane_id();
    tmpl::wait_parity(vm.sem(kSemOReady), 0);
    const __nv_bfloat16* out = reinterpret_cast<const __nv_bfloat16*>(vm.scratch());
    *reinterpret_cast<uint4*>(g.attn_out + head0 * kHeadDim + lane * 8) = *reinterpret_cast<const uint4*>(out + lane * 8);
    __syncwarp();
    if (lane == 0) {
      __threadfence();
      counter_add_release(bar_ptr(g, layer, kOpAttentionReduction - 1, 0), kGqa);
    }
  }
};

// ------------------------------------------------------------- op 0: NoOp
// Releases every page at once, so a padded tail costs the next SM nothing.
struct OpNoOp {
  __device__ static int release_lid(const int32_t*, int q) { return q; }
  __device__ static int init_semaphores(const Vm&) { return 0; }
  __device__ static void loader(Vm& vm, const Globals&) {
    const int lane = lane_id();
    if (lane < Cfg::kNumPages) {
      vm.wait_page_ready(vm.pid(lane));
      vm.finish_page(vm.pid(lane), Cfg::kConsumerWarps);
    }
  }
  __device__ static void launcher(Vm&, const Globals&) {}
  __device__ static void consumer(Vm&, const Globals&) {}
  __device__ static void storer(Vm&, const Globals&) {}
};

// ---------------------------------------------------------------- dispatch

#define MK42_DISPATCH(opcode, EXPR)                                            \
  switch (opcode) {                                                            \
    case kOpRmsQkvRopeAppend:  { using Op = OpRmsQkv;              EXPR; break; } \
    case kOpPartialAttention:  { using Op = OpAttention;           EXPR; break; } \
    case kOpAttentionReduction:{ using Op = OpAttentionReduction;  EXPR; break; } \
    case kOpOProjResidual:     { using Op = OpOProj;               EXPR; break; } \
    case kOpRmsUpgateSilu:     { using Op = OpRmsUpgate;           EXPR; break; } \
    case kOpDownProjResidual:  { using Op = OpDownProj;            EXPR; break; } \
    case kOpRmsLmHead:         { using Op = OpRmsLmHead;           EXPR; break; } \
    default:                   { using Op = OpNoOp;                EXPR; break; } \
  }

__device__ __forceinline__ int dispatch_release_lid(int opcode, const int32_t* ins, int q) {
  int r = q;
  MK42_DISPATCH(opcode, r = Op::release_lid(ins, q));
  return r;
}
__device__ __forceinline__ int dispatch_init_semaphores(int opcode, const Vm& vm) {
  int n = 0;
  MK42_DISPATCH(opcode, n = Op::init_semaphores(vm));
  return n;
}

enum class Role { kLoader, kStorer, kLauncher, kConsumer };

template <Role R>
__device__ __forceinline__ void dispatch_role(int opcode, Vm& vm, const Globals& g) {
  MK42_DISPATCH(opcode,
    if constexpr (R == Role::kLoader) Op::loader(vm, g);
    else if constexpr (R == Role::kStorer) Op::storer(vm, g);
    else if constexpr (R == Role::kLauncher) Op::launcher(vm, g);
    else Op::consumer(vm, g));
}

// ============================================================== controller

__device__ void store_timings_and_reset(Vm& vm, const Globals& g, int ring, int instr_index) {
  if constexpr (Cfg::kTiming) {
    int32_t* t = vm.is[ring].timings;
    if (lane_id() == 0) {
      int32_t* dst = g.timings + (static_cast<int64_t>(blockIdx.x) * g.num_instructions + instr_index) * Cfg::kTimingWidth;
      tmpl::fence_proxy_async_shared();
      bulk_store_1d(dst, t, Cfg::kTimingWidth * 4);
      bulk_commit();
      bulk_wait_read();
    }
    __syncwarp();
    for (int i = lane_id(); i < Cfg::kTimingWidth; i += 32) { t[i] = 0; }
  }
}

__device__ void controller_loop(Vm& vm, const Globals& g) {
  const int lane = lane_id();
  int num_sems[Cfg::kStages] = {0, 0};
  for (vm.index = 0, vm.ring = 0; vm.index < g.num_instructions; ++vm.index, vm.ring = (vm.ring + 1) % Cfg::kStages) {
    // Step 0: reclaim the slot two instructions back.
    if (vm.index >= Cfg::kStages) {
      const int last = vm.index - Cfg::kStages;
      tmpl::wait_parity_backoff<>(&vm.instr_finished[vm.ring], static_cast<uint32_t>(last / Cfg::kStages) & 1u);
      if (lane < num_sems[vm.ring]) { mbarrier_inval(&vm.is[vm.ring].semaphores[lane]); }
      __syncwarp();
      if (lane == 0) { vm.record(kTControllerEnd); }
      store_timings_and_reset(vm, g, vm.ring, last);
    }
    if (lane == 0) { vm.record(kTControllerStart); }
    // Step 1: fetch the instruction, 128 B, one int per lane.
    const int32_t* src = g.instructions + (static_cast<int64_t>(blockIdx.x) * g.num_instructions + vm.index) * Cfg::kInstrWidth;
    vm.instruction()[lane] = src[lane];
    __syncwarp();
    if (lane == 0) { vm.record(kTIFetchDone); }
    // Step 2: physical page order from the previous instruction's release order.
    if (vm.index == 0) {
      if (lane < Cfg::kNumPages) { vm.is[vm.ring].pid_order[lane] = lane; }
    } else {
      const int prev_ring = (vm.ring + Cfg::kStages - 1) % Cfg::kStages;
      const int32_t* prev = vm.is[prev_ring].instruction;
      if (lane < Cfg::kNumPages) {
        const int lid = dispatch_release_lid(prev[0], prev, lane);
        vm.is[vm.ring].pid_order[lane] = vm.is[prev_ring].pid_order[lid];
      }
    }
    __syncwarp();
    if (lane == 0) { vm.record(kTPageAllocDone); }
    // Step 3: the op's semaphores, by one lane, then made visible to the async proxy.
    const int opcode = vm.instruction()[0];
    int n = 0;
    if (lane == 0) { n = dispatch_init_semaphores(opcode, vm); }
    num_sems[vm.ring] = __shfl_sync(0xffffffffu, n, 0);
    tmpl::fence_barrier_init();
    tmpl::fence_proxy_async_shared();
    __syncwarp();
    // Step 4: publish.
    if (lane == 0) {
      vm.record(kTSemsSetup);
      tmpl::mbarrier_arrive(&vm.instr_arrived[vm.ring]);
    }
  }
  // Drain: the last kStages slots.
  for (int i = 0; i < Cfg::kStages; ++i) {
    const int idx = g.num_instructions - Cfg::kStages + i;
    if (idx < 0) { continue; }
    const int ring = idx % Cfg::kStages;
    tmpl::wait_parity_backoff<>(&vm.instr_finished[ring], static_cast<uint32_t>(idx / Cfg::kStages) & 1u);
    if (lane < num_sems[ring]) { mbarrier_inval(&vm.is[ring].semaphores[lane]); }
    vm.index = idx; vm.ring = ring;
    if (lane == 0) { vm.record(kTControllerEnd); }
    store_timings_and_reset(vm, g, ring, idx);
  }
}

template <Role R>
__device__ void worker_loop(Vm& vm, const Globals& g, int start_event) {
  for (vm.index = 0, vm.ring = 0; vm.index < g.num_instructions; vm.next_instruction()) {
    vm.await_instruction();
    const int opcode = vm.instruction()[0];
    if (lane_id() == 0) { vm.record(start_event); }
    dispatch_role<R>(opcode, vm, g);
    if (lane_id() == 0) { vm.record(start_event + 1); }
  }
}

}  // namespace

// =================================================================== kernel

__global__ __launch_bounds__(Cfg::kNumThreads, 1) void mk_llama(const __grid_constant__ Globals g) {
  __shared__ alignas(128) InstructionState is[Cfg::kStages];
  __shared__ alignas(8) uint64_t instr_arrived[Cfg::kStages];
  __shared__ alignas(8) uint64_t instr_finished[Cfg::kStages];
  __shared__ alignas(8) uint64_t page_finished[Cfg::kNumPages];
  extern __shared__ uint8_t smem_raw[];
  uint8_t* pages = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));

  Vm vm{is, instr_arrived, instr_finished, page_finished, pages};
  vm.start_clock = clock_now();
  const int tid = threadIdx.x;

  if (tid < Cfg::kTimingWidth) {
    for (int i = 0; i < Cfg::kStages; ++i) { is[i].timings[tid] = 0; }
  }
  if (tid < Cfg::kStages) {
    tmpl::mbarrier_init(&instr_arrived[tid], 1);
    tmpl::mbarrier_init(&instr_finished[tid], Cfg::kNumWarps - 1);
  }
  if (tid < Cfg::kNumPages) {
    // Phase 0 completes at once: instruction 0 finds every page free.
    tmpl::mbarrier_init(&page_finished[tid], Cfg::kConsumerWarps);
    mbarrier_arrive_n(&page_finished[tid], Cfg::kConsumerWarps);
  }
  tmpl::fence_barrier_init();
  tmpl::fence_proxy_async_shared();
  __syncthreads();

  const int warp = warp_id();
  if (warp < Cfg::kConsumerWarps) {
    tmpl::setmaxnreg_inc<Cfg::kConsumerRegs>();
    worker_loop<Role::kConsumer>(vm, g, kTConsumerStart + 2 * warp);
  } else {
    tmpl::setmaxnreg_dec<Cfg::kNonConsumerRegs>();
    switch (warp - Cfg::kConsumerWarps) {
      case 0: worker_loop<Role::kLoader>(vm, g, kTLoaderStart); break;
      case 1: worker_loop<Role::kStorer>(vm, g, kTStorerStart); break;
      case 2: worker_loop<Role::kLauncher>(vm, g, kTLauncherStart); break;
      default: controller_loop(vm, g); break;
    }
  }
  __syncthreads();
}

// ================================================================= planner
//
// Host side, as upstream's scheduler.py: instructions are built per op with
// the block lists that make per-SM work even, chained into a DAG with the
// dependencies the counters express, then assigned to SMs.

#ifndef MK42_NO_MAIN

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

struct Instr {
  std::vector<int32_t> words;   // serialized, opcode first
  double cost = 0.0;            // bytes moved: the list scheduler's weight
  std::vector<int> deps;        // indices into the DAG
  int opcode() const { return words.empty() ? 0 : words[0]; }
};

struct Plan {
  std::vector<Instr> dag;        // topological order
  std::vector<std::vector<int>> per_sm;  // instruction indices per SM
};

int pick_partials(int pos_id, int requested) {
  if (requested > 0) { return std::min(requested, OpAttentionReduction::kMaxList); }
  return 1;
}

// One layer of the DAG. Returns the indices of the layer's last ops.
void plan_layer(std::vector<Instr>& dag, int layer, int sm_count, int pos_id, int partials,
                std::vector<int>& prev_outputs, const std::string& stop_after) {
  auto add = [&](std::vector<int32_t> w, double cost, std::vector<int> deps) {
    Instr in; in.words = std::move(w); in.cost = cost; in.deps = std::move(deps);
    dag.push_back(std::move(in));
    return static_cast<int>(dag.size()) - 1;
  };
  // QKV: 192 blocks over the SMs, contiguous ranges.
  std::vector<int> qkv;
  {
    const int blocks = kQkvDim / kBlock;
    const double per_sm = double(blocks) / sm_count;
    for (int s = 0; s < sm_count; ++s) {
      const int b0 = int(std::lround(s * per_sm)), b1 = int(std::lround((s + 1) * per_sm));
      if (b1 <= b0) { continue; }
      qkv.push_back(add({kOpRmsQkvRopeAppend, layer, b0, b1}, double(b1 - b0) * kBlock * kHidden * 2, prev_outputs));
    }
  }
  if (stop_after == "qkv") { prev_outputs = qkv; return; }
  // Attention: one instruction per (kv head, partial).
  std::vector<int> attn;
  for (int kvh = 0; kvh < kNumKvHeads; ++kvh) {
    for (int p = 0; p < partials; ++p) {
      attn.push_back(add({kOpPartialAttention, layer, kvh, partials, p},
                         double(pos_id + 1) / partials * kHeadDim * 2 * 2, qkv));
    }
  }
  std::vector<int> attn_done = attn;
  if (partials > 1) {
    std::vector<int> red;
    for (int h0 = 0; h0 < kNumHeads; h0 += kGqa) {
      std::vector<int32_t> w = {kOpAttentionReduction, layer, h0, partials, 1, partials};
      for (int p = 0; p < partials; ++p) { w.push_back(p); }
      red.push_back(add(w, double(partials) * kGqa * kHeadDim * 4, attn));
    }
    attn_done = red;
  }
  if (stop_after == "attn") { prev_outputs = attn_done; return; }
  // o_proj: 128 single-block instructions.
  std::vector<int> oproj;
  for (int b = 0; b < kHidden / kBlock; ++b) {
    oproj.push_back(add({kOpOProjResidual, layer, b, b + 1, 0}, double(kBlock) * kHidden * 2, attn_done));
  }
  if (stop_after == "oproj") { prev_outputs = oproj; return; }
  // up/gate: 512 blocks, strided over SMs so every instruction spans the
  // intermediate dimension (and so every down-proj column block fills evenly).
  std::vector<int> upgate;
  {
    const int blocks = kInter / kBlock;
    for (int s = 0; s < sm_count; ++s) {
      std::vector<int32_t> w = {kOpRmsUpgateSilu, layer, 0};
      for (int b = s; b < blocks; b += sm_count) { w.push_back(b); }
      w[2] = int(w.size()) - 3;
      if (w[2] == 0) { continue; }
      if (w.size() > size_t(Cfg::kInstrWidth)) { printf("upgate list overflow\n"); exit(1); }
      upgate.push_back(add(w, double(w[2]) * 2 * kBlock * kHidden * 2, oproj));
    }
  }
  if (stop_after == "upgate") { prev_outputs = upgate; return; }
  // down: (4 reduction columns x 128 row blocks) jobs, contiguous per SM
  // without crossing a column boundary.
  std::vector<int> down;
  {
    std::vector<std::pair<int, int>> jobs;
    for (int c = 0; c < kInter / kHidden; ++c) {
      for (int b = 0; b < kHidden / kBlock; ++b) { jobs.push_back({c, b}); }
    }
    size_t assigned = 0;
    for (int s = 0; s < sm_count && assigned < jobs.size(); ++s) {
      const double left = double(jobs.size() - assigned) / (sm_count - s);
      int take = std::max(1, int(std::lround(left)));
      const int col = jobs[assigned].first;
      int n = 0;
      while (n < take && assigned + n < jobs.size() && jobs[assigned + n].first == col) { ++n; }
      const int b0 = jobs[assigned].second;
      down.push_back(add({kOpDownProjResidual, layer, b0, b0 + n, col}, double(n) * kBlock * kHidden * 2, upgate));
      assigned += n;
    }
  }
  prev_outputs = down;
}

Plan build_plan(int sm_count, int pos_id, int layers, int partials, const std::string& stop_after,
                const std::string& sched) {
  Plan plan;
  std::vector<int> prev;
  for (int L = 0; L < layers; ++L) {
    plan_layer(plan.dag, L, sm_count, pos_id, partials, prev, L == layers - 1 ? stop_after : "");
  }
  if (stop_after.empty() && layers == kNumLayers) {
    const int blocks = kVocab / kBlock;
    const double per_sm = double(blocks) / sm_count;
    for (int s = 0; s < sm_count; ++s) {
      const int b0 = int(std::lround(s * per_sm)), b1 = int(std::lround((s + 1) * per_sm));
      if (b1 <= b0) { continue; }
      Instr in; in.words = {kOpRmsLmHead, b0, b1}; in.cost = double(b1 - b0) * kBlock * kHidden * 2; in.deps = prev;
      plan.dag.push_back(in);
    }
  }
  plan.per_sm.assign(sm_count, {});
  if (sched == "dag") {
    // List scheduling: ready instructions by descending cost onto the SM
    // that frees up first. A node becomes ready when all its deps are placed.
    const int n = int(plan.dag.size());
    std::vector<int> remaining(n);
    std::vector<std::vector<int>> children(n);
    std::vector<double> end_time(n, 0.0);
    for (int i = 0; i < n; ++i) {
      remaining[i] = int(plan.dag[i].deps.size());
      for (int d : plan.dag[i].deps) { children[d].push_back(i); }
    }
    using PQ = std::priority_queue<std::pair<double, int>>;
    PQ ready;  // (cost, idx): biggest first
    for (int i = 0; i < n; ++i) { if (remaining[i] == 0) { ready.push({plan.dag[i].cost, i}); } }
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> sms;
    for (int s = 0; s < sm_count; ++s) { sms.push({0.0, s}); }
    while (!ready.empty()) {
      const int idx = ready.top().second; ready.pop();
      auto [t, s] = sms.top(); sms.pop();
      end_time[idx] = t + plan.dag[idx].cost;
      plan.per_sm[s].push_back(idx);
      sms.push({end_time[idx], s});
      for (int c : children[idx]) { if (--remaining[c] == 0) { ready.push({plan.dag[c].cost, c}); } }
    }
  } else {
    for (int i = 0; i < int(plan.dag.size()); ++i) { plan.per_sm[i % sm_count].push_back(i); }
  }
  return plan;
}

// ------------------------------------------------------------- reference
// Doubles; bf16 rounding wherever the device rounds.

float r_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

struct HostModel {
  std::vector<float> qkv, o, up, gate, down, lm_head, attn_norm, mlp_norm, lm_norm, rope_cos, rope_sin;
  std::vector<float> k_cache, v_cache;  // [L][kv][max_seq][64]
  int max_seq = 0;
};

struct RefOut {
  std::vector<float> hidden_after_layer;  // final residual stream
  std::vector<float> q_last, attn_last, silu_last, hidden_last;  // last layer intermediates
  std::vector<float> logits;
};

void rms_ref(const std::vector<float>& x, const float* w, std::vector<double>& y) {
  double ss = 0.0;
  for (float v : x) ss += double(v) * v;
  const double sc = 1.0 / std::sqrt(ss / x.size() + kEps);
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * sc * w[i];
}

// out[n] = sum_k w[n][k] * x[k], w row-major with row stride K.
void matvec_ref(const float* w, int64_t rows, int64_t K, const std::vector<double>& x, std::vector<double>& out) {
  out.resize(rows);
  for (int64_t n = 0; n < rows; ++n) {
    const float* row = w + n * K;
    double acc = 0.0;
    for (int64_t k = 0; k < K; ++k) acc += double(row[k]) * x[k];
    out[n] = acc;
  }
}

RefOut reference(HostModel& m, std::vector<float> hidden, int pos_id, int layers, const std::string& stop_after) {
  RefOut r;
  const int64_t S = m.max_seq;
  for (int L = 0; L < layers; ++L) {
    const bool last = (L == layers - 1);
    std::vector<double> xn, qkv;
    rms_ref(hidden, m.attn_norm.data() + int64_t(L) * kHidden, xn);
    matvec_ref(m.qkv.data() + int64_t(L) * kQkvDim * kHidden, kQkvDim, kHidden, xn, qkv);
    std::vector<float> q(kNumHeads * kHeadDim);
    for (int h = 0; h < kNumQkvHeads; ++h) {
      // RoPE is applied to the fp32 projection and rounded once, as the device does.
      if (h >= kNumHeads + kNumKvHeads) {
        for (int d = 0; d < kHeadDim; ++d) qkv[h * kHeadDim + d] = r_bf16(float(qkv[h * kHeadDim + d]));
      }
      if (h < kNumHeads + kNumKvHeads) {
        for (int d = 0; d < kHeadDim; d += 2) {
          const double c = m.rope_cos[int64_t(pos_id) * kHeadDim + d], s = m.rope_sin[int64_t(pos_id) * kHeadDim + d];
          const double a = qkv[h * kHeadDim + d], b = qkv[h * kHeadDim + d + 1];
          qkv[h * kHeadDim + d] = r_bf16(float(a * c - b * s));
          qkv[h * kHeadDim + d + 1] = r_bf16(float(b * c + a * s));
        }
      }
    }
    for (int i = 0; i < kNumHeads * kHeadDim; ++i) q[i] = float(qkv[i]);
    for (int h = 0; h < kNumKvHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) {
        m.k_cache[((int64_t(L) * kNumKvHeads + h) * S + pos_id) * kHeadDim + d] = float(qkv[(kNumHeads + h) * kHeadDim + d]);
        m.v_cache[((int64_t(L) * kNumKvHeads + h) * S + pos_id) * kHeadDim + d] = float(qkv[(kNumHeads + kNumKvHeads + h) * kHeadDim + d]);
      }
    }
    if (last) r.q_last = q;
    if (last && stop_after == "qkv") break;
    // attention
    std::vector<double> attn(kNumHeads * kHeadDim, 0.0);
    const double scale = 1.0 / std::sqrt(double(kHeadDim));
    for (int h = 0; h < kNumHeads; ++h) {
      const int kvh = h / kGqa;
      const float* K = m.k_cache.data() + (int64_t(L) * kNumKvHeads + kvh) * S * kHeadDim;
      const float* V = m.v_cache.data() + (int64_t(L) * kNumKvHeads + kvh) * S * kHeadDim;
      std::vector<double> s(pos_id + 1);
      double mx = -1e300;
      for (int p = 0; p <= pos_id; ++p) {
        double acc = 0.0;
        for (int d = 0; d < kHeadDim; ++d) acc += double(q[h * kHeadDim + d]) * K[int64_t(p) * kHeadDim + d];
        s[p] = acc * scale; mx = std::max(mx, s[p]);
      }
      double l = 0.0;
      std::vector<double> o(kHeadDim, 0.0);
      for (int p = 0; p <= pos_id; ++p) {
        const double e = std::exp(s[p] - mx);
        l += e;
        for (int d = 0; d < kHeadDim; ++d) o[d] += e * V[int64_t(p) * kHeadDim + d];
      }
      for (int d = 0; d < kHeadDim; ++d) attn[h * kHeadDim + d] = r_bf16(float(o[d] / l));
    }
    if (last) { r.attn_last.resize(attn.size()); for (size_t i = 0; i < attn.size(); ++i) r.attn_last[i] = float(attn[i]); }
    if (last && stop_after == "attn") break;
    // o_proj + residual (bf16 add)
    std::vector<double> oo;
    matvec_ref(m.o.data() + int64_t(L) * kHidden * kHidden, kHidden, kHidden, attn, oo);
    for (int i = 0; i < kHidden; ++i) hidden[i] = r_bf16(hidden[i] + r_bf16(float(oo[i])));
    if (last && stop_after == "oproj") break;
    // up/gate
    std::vector<double> xn2, u, gt;
    rms_ref(hidden, m.mlp_norm.data() + int64_t(L) * kHidden, xn2);
    matvec_ref(m.up.data() + int64_t(L) * kInter * kHidden, kInter, kHidden, xn2, u);
    matvec_ref(m.gate.data() + int64_t(L) * kInter * kHidden, kInter, kHidden, xn2, gt);
    std::vector<double> silu_v(kInter);
    for (int i = 0; i < kInter; ++i) {
      const double gg = gt[i];
      silu_v[i] = r_bf16(float((gg / (1.0 + std::exp(-gg))) * u[i]));
    }
    if (last) { r.silu_last.resize(kInter); for (int i = 0; i < kInter; ++i) r.silu_last[i] = float(silu_v[i]); }
    if (last && stop_after == "upgate") break;
    // down: four column partials, each rounded and added in bf16
    for (int c = 0; c < kInter / kHidden; ++c) {
      std::vector<double> part(kHidden);
      std::vector<double> xs(silu_v.begin() + c * kHidden, silu_v.begin() + (c + 1) * kHidden);
      for (int n = 0; n < kHidden; ++n) {
        const float* row = m.down.data() + (int64_t(L) * kHidden + n) * kInter + c * kHidden;
        double acc = 0.0;
        for (int k = 0; k < kHidden; ++k) acc += double(row[k]) * xs[k];
        part[n] = acc;
      }
      for (int n = 0; n < kHidden; ++n) hidden[n] = r_bf16(hidden[n] + r_bf16(float(part[n])));
    }
    if (last) r.hidden_last = hidden;
  }
  r.hidden_after_layer = hidden;
  if (stop_after.empty() && layers == kNumLayers) {
    std::vector<double> xn, lg;
    rms_ref(hidden, m.lm_norm.data(), xn);
    matvec_ref(m.lm_head.data(), kVocab, kHidden, xn, lg);
    r.logits.resize(kVocab);
    for (int i = 0; i < kVocab; ++i) r.logits[i] = r_bf16(float(lg[i]));
  }
  return r;
}

float host_rand(uint32_t& s, float scale) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.f - 0.5f) * 2.f * scale;
}
std::vector<float> rand_vec(uint32_t& s, size_t n, float scale, float offset = 0.f) {
  std::vector<float> v(n);
  for (auto& e : v) e = r_bf16(offset + host_rand(s, scale));
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
std::vector<float> download(const __nv_bfloat16* d, size_t n) {
  std::vector<__nv_bfloat16> t(n);
  CK(cudaMemcpy(t.data(), d, n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(t[i]);
  return h;
}
// Errors normalised by the reference's RMS: a bf16 residual stream carries
// ~1 ulp of order-dependent rounding per add, which a per-element relative
// error at small elements reports as tens of percent.
struct Err { float max_norm, mean_norm; };
Err compare(const std::vector<float>& a, const std::vector<float>& b) {
  double rms = 0.0, mean = 0.0, worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    rms += double(b[i]) * b[i];
    const double d = std::fabs(double(a[i]) - b[i]);
    mean += d; worst = std::max(worst, d);
  }
  rms = std::sqrt(rms / a.size()) + 1e-12;
  return Err{float(worst / rms), float(mean / a.size() / rms)};
}

// 3-D map over a [layers][rows][cols] bf16 stack with a 256 x 16 x 1 box.
void make_weight_map(CUtensorMap* map, const __nv_bfloat16* base, uint64_t cols, uint64_t rows, uint64_t layers) {
  uint64_t dims[3] = {cols, rows, layers};
  uint64_t strides[2] = {cols * 2, cols * rows * 2};
  uint32_t box[3] = {256, 16, 1};
  uint32_t estr[3] = {1, 1, 1};
  CKD(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(base), dims, strides,
                             box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

std::string arg_value(int argc, char** argv, const char* key, const char* dflt) {
  const std::string k = std::string(key) + "=";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).rfind(k, 0) == 0) { return std::string(argv[i]).substr(k.size()); }
  }
  return dflt;
}

}  // namespace

int main(int argc, char** argv) {
  const int layers = std::stoi(arg_value(argc, argv, "layers", std::to_string(kNumLayers).c_str()));
  const int pos_id = std::stoi(arg_value(argc, argv, "pos", "1023"));
  const int partials = pick_partials(pos_id, std::stoi(arg_value(argc, argv, "partials", "0")));
  const std::string stop_after = arg_value(argc, argv, "stop", "");
  const std::string sched = arg_value(argc, argv, "sched", "rr");
  const int timing_iters = std::stoi(arg_value(argc, argv, "iters", "20"));
  if (layers < 1 || layers > kNumLayers) { printf("layers must be 1..%d\n", kNumLayers); return 1; }

  int sm_count = 0;
  CK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaFree(nullptr));
  const int max_seq = pos_id + 1;
  const bool full = stop_after.empty() && layers == kNumLayers;

  printf("HazyResearch-style Llama-1B megakernel  (%d SMs, %d layers%s, pos %d, partials %d, sched %s)\n",
         sm_count, layers, full ? " + lm_head" : "", pos_id, partials, sched.c_str());

  // ---- host model (the whole 16-layer weight stack is allocated whatever `layers` is)
  uint32_t seed = 77u;
  HostModel m;
  m.max_seq = max_seq;
  const float wsc = 1.f / std::sqrt(float(kHidden));
  m.qkv = rand_vec(seed, size_t(kNumLayers) * kQkvDim * kHidden, wsc);
  m.o = rand_vec(seed, size_t(kNumLayers) * kHidden * kHidden, wsc);
  m.up = rand_vec(seed, size_t(kNumLayers) * kInter * kHidden, wsc);
  m.gate = rand_vec(seed, size_t(kNumLayers) * kInter * kHidden, wsc);
  m.down = rand_vec(seed, size_t(kNumLayers) * kHidden * kInter, 1.f / std::sqrt(float(kInter)));
  m.lm_head = rand_vec(seed, size_t(kVocab) * kHidden, wsc);
  m.attn_norm = rand_vec(seed, size_t(kNumLayers) * kHidden, 0.25f, 1.f);
  m.mlp_norm = rand_vec(seed, size_t(kNumLayers) * kHidden, 0.25f, 1.f);
  m.lm_norm = rand_vec(seed, kHidden, 0.25f, 1.f);
  m.k_cache = rand_vec(seed, size_t(kNumLayers) * kNumKvHeads * max_seq * kHeadDim, 1.f);
  m.v_cache = rand_vec(seed, size_t(kNumLayers) * kNumKvHeads * max_seq * kHeadDim, 1.f);
  m.rope_cos.resize(size_t(max_seq) * kHeadDim);
  m.rope_sin.resize(size_t(max_seq) * kHeadDim);
  for (int p = 0; p < max_seq; ++p) {
    for (int i = 0; i < kHeadDim / 2; ++i) {
      const double theta = std::pow(double(kRopeBase), -2.0 * i / kHeadDim);
      const double f = p * theta;
      m.rope_cos[size_t(p) * kHeadDim + 2 * i] = m.rope_cos[size_t(p) * kHeadDim + 2 * i + 1] = float(std::cos(f));
      m.rope_sin[size_t(p) * kHeadDim + 2 * i] = m.rope_sin[size_t(p) * kHeadDim + 2 * i + 1] = float(std::sin(f));
    }
  }
  std::vector<float> hidden0 = rand_vec(seed, kHidden, 1.f);

  // ---- device
  Globals g{};
  __nv_bfloat16* d_qkv = upload(m.qkv);
  __nv_bfloat16* d_o = upload(m.o);
  __nv_bfloat16* d_up = upload(m.up);
  __nv_bfloat16* d_gate = upload(m.gate);
  __nv_bfloat16* d_down = upload(m.down);
  __nv_bfloat16* d_lm = upload(m.lm_head);
  make_weight_map(&g.qkv_w, d_qkv, kHidden, kQkvDim, kNumLayers);
  make_weight_map(&g.o_w, d_o, kHidden, kHidden, kNumLayers);
  make_weight_map(&g.up_w, d_up, kHidden, kInter, kNumLayers);
  make_weight_map(&g.gate_w, d_gate, kHidden, kInter, kNumLayers);
  make_weight_map(&g.down_w, d_down, kInter, kHidden, kNumLayers);
  make_weight_map(&g.lm_head_w, d_lm, kHidden, kVocab, 1);
  g.attn_norm = upload(m.attn_norm);
  g.mlp_norm = upload(m.mlp_norm);
  g.lm_head_norm = upload(m.lm_norm);
  g.k_cache = upload(m.k_cache);
  g.v_cache = upload(m.v_cache);
  float *d_cos = nullptr, *d_sin = nullptr;
  CK(cudaMalloc(&d_cos, m.rope_cos.size() * 4)); CK(cudaMemcpy(d_cos, m.rope_cos.data(), m.rope_cos.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_sin, m.rope_sin.size() * 4)); CK(cudaMemcpy(d_sin, m.rope_sin.data(), m.rope_sin.size() * 4, cudaMemcpyHostToDevice));
  g.rope_cos = d_cos; g.rope_sin = d_sin;
  g.hidden = upload(hidden0);
  __nv_bfloat16* d_hidden0 = upload(hidden0);
  CK(cudaMalloc((void**)&g.q_post_rope, kHidden * 2));
  CK(cudaMalloc((void**)&g.attn_out, kHidden * 2));
  CK(cudaMalloc((void**)&g.attn_o_part, size_t(kNumHeads) * kMaxPartials * kHeadDim * 4));
  CK(cudaMalloc((void**)&g.attn_lse_part, size_t(kNumHeads) * kLseStride * 4));
  CK(cudaMalloc((void**)&g.silu_out, kInter * 2));
  CK(cudaMalloc((void**)&g.logits, size_t(kVocab) * 2));
  CK(cudaMemset(g.logits, 0, size_t(kVocab) * 2));
  const size_t bar_words = size_t(kNumLayers) * 10 * 48;
  CK(cudaMalloc((void**)&g.bar, bar_words * 4));
  g.pos_id = pos_id; g.max_seq = max_seq;
  g.attn_scale = 1.f / std::sqrt(float(kHeadDim));
  g.skip_attn_reduction = (partials == 1) ? 1 : 0;

  // ---- plan
  const Plan plan = build_plan(sm_count, pos_id, layers, partials, stop_after, sched);
  size_t max_len = 0;
  for (auto& q : plan.per_sm) max_len = std::max(max_len, q.size());
  std::vector<int32_t> table(size_t(sm_count) * max_len * Cfg::kInstrWidth, 0);
  for (int s = 0; s < sm_count; ++s) {
    for (size_t i = 0; i < plan.per_sm[s].size(); ++i) {
      const Instr& in = plan.dag[plan.per_sm[s][i]];
      std::copy(in.words.begin(), in.words.end(), table.begin() + (size_t(s) * max_len + i) * Cfg::kInstrWidth);
    }
  }
  int32_t* d_table = nullptr;
  CK(cudaMalloc(&d_table, table.size() * 4));
  CK(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
  g.instructions = d_table;
  g.num_instructions = int32_t(max_len);
  int32_t* d_timings = nullptr;
  CK(cudaMalloc(&d_timings, size_t(sm_count) * max_len * Cfg::kTimingWidth * 4));
  CK(cudaMemset(d_timings, 0, size_t(sm_count) * max_len * Cfg::kTimingWidth * 4));
  g.timings = d_timings;
  int op_count[kNumOpcodes] = {};
  for (auto& in : plan.dag) op_count[in.opcode()]++;
  printf("  program: %zu instructions, %zu per SM; per op:", plan.dag.size(), max_len);
  const char* opname[kNumOpcodes] = {"noop", "qkv", "attn", "attn_red", "o_proj", "upgate", "down", "lm_head"};
  for (int i = 1; i < kNumOpcodes; ++i) if (op_count[i]) printf(" %s=%d", opname[i], op_count[i]);
  printf("\n");

  CK(cudaFuncSetAttribute(mk_llama, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kDynamicSmem));
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  auto run_step = [&](cudaStream_t s) {
    CK(cudaMemcpyAsync(g.hidden, d_hidden0, kHidden * 2, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemsetAsync(g.bar, 0, bar_words * 4, s));
    mk_llama<<<sm_count, Cfg::kNumThreads, Cfg::kDynamicSmem, s>>>(g);
  };

  // ---- correctness
  printf("  reference (double) ...\n");
  fflush(stdout);
  RefOut ref = reference(m, hidden0, pos_id, layers, stop_after);
  run_step(stream);
  CK(cudaStreamSynchronize(stream));
  CK(cudaGetLastError());
  bool ok = true;
  auto report = [&](const char* what, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    const Err e = compare(got, want);
    const bool pass = e.max_norm < tol && e.mean_norm < tol / 5.f;
    printf("  %-14s max err / rms %.3e  mean err / rms %.3e  %s\n", what, e.max_norm, e.mean_norm, pass ? "PASS" : "FAIL");
    ok = ok && pass;
  };
  if (stop_after == "qkv") {
    report("q_post_rope", download(g.q_post_rope, kHidden), ref.q_last, 2e-2f);
  } else if (stop_after == "attn") {
    report("attn_out", download(g.attn_out, kHidden), ref.attn_last, 2e-2f);
  } else if (stop_after == "upgate") {
    report("silu_out", download(g.silu_out, kInter), ref.silu_last, 2e-2f);
  } else {
    report("hidden", download(g.hidden, kHidden), ref.hidden_after_layer, 5e-2f);
    if (full) {
      std::vector<float> lg = download(g.logits, kVocab);
      report("logits", lg, ref.logits, 5e-2f);
      const int a_dev = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
      const int a_ref = int(std::max_element(ref.logits.begin(), ref.logits.end()) - ref.logits.begin());
      printf("  argmax device %d reference %d %s\n", a_dev, a_ref, a_dev == a_ref ? "MATCH" : "DIFFER");
    }
  }
  if (!ok) { printf("numerics failed; timings withheld\n"); return 1; }

  // ---- timing
  double bytes = 0.0;
  for (int L = 0; L < layers; ++L) {
    bytes += double(kQkvDim + kHidden + 2 * kInter) * kHidden * 2 + double(kHidden) * kInter * 2;
    bytes += double(kNumKvHeads) * (pos_id + 1) * kHeadDim * 2 * 2;
  }
  if (full) bytes += double(kVocab) * kHidden * 2;
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run_step(stream);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 3; ++r) {
    CK(cudaEventRecord(a, stream));
    for (int i = 0; i < timing_iters; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaEventRecord(b, stream));
    CK(cudaEventSynchronize(b));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms / timing_iters);
  }
  const double floor_ms = (1.85 + bytes / 2.77e6) / 1000.0;
  if (stop_after.empty()) {
    printf("\n  decode step: %.3f ms  (%.2f GB moved -> %.2f TB/s; floor %.3f ms [ld.bw.dev.dram] -> %.0f%% of floor)\n",
           best, bytes / 1e9, bytes / best / 1e9, floor_ms, 100.0 * floor_ms / best);
    if (full) printf("  upstream: under 1 ms at 78%% of bandwidth on H100 for this model.\n");
  } else {
    printf("\n  truncated program: %.3f ms (no bandwidth figure: the byte count is for a whole layer)\n", best);
  }

  // ---- the VM's own profiler
  if (Cfg::kTiming) {
    CK(cudaMemset(d_timings, 0, size_t(sm_count) * max_len * Cfg::kTimingWidth * 4));
    run_step(stream);
    CK(cudaStreamSynchronize(stream));
    std::vector<int32_t> t(size_t(sm_count) * max_len * Cfg::kTimingWidth);
    CK(cudaMemcpy(t.data(), d_timings, t.size() * 4, cudaMemcpyDeviceToHost));
    struct Agg { double ctrl, fetch, page, sems, ldr_span, gmem_wait, cons_first, cons_span, store_span, total; int n; };
    Agg agg[kNumOpcodes] = {};
    for (int s = 0; s < sm_count; ++s) {
      for (size_t i = 0; i < plan.per_sm[s].size(); ++i) {
        const int32_t* r = &t[(size_t(s) * max_len + i) * Cfg::kTimingWidth];
        const int op = plan.dag[plan.per_sm[s][i]].opcode();
        Agg& a2 = agg[op];
        // An op that never records an event leaves its slot at 0; such an
        // interval is skipped rather than reported as a large negative.
        auto span = [&](int b, int e) { return (r[b] && r[e]) ? double(r[e] - r[b]) : 0.0; };
        a2.fetch += span(kTControllerStart, kTIFetchDone);
        a2.page += span(kTIFetchDone, kTPageAllocDone);
        a2.sems += span(kTPageAllocDone, kTSemsSetup);
        a2.ldr_span += span(kTFirstLoad, kTLastLoad);
        a2.gmem_wait += span(kTAtGmemWait, kTDoneGmemWait);
        a2.cons_first += span(kTConsumerStart, kTFirstUse);
        a2.cons_span += span(kTFirstUse, kTLastUse);
        a2.store_span += span(kTFirstStore, kTLastStore);
        a2.total += span(kTControllerStart, kTControllerEnd);
        a2.ctrl += span(kTControllerStart, kTSemsSetup);
        a2.n++;
      }
    }
    printf("\n  per-instruction phases, mean cycles (VM profiler; residency, not critical path)\n");
    printf("  %-9s %5s %8s %8s %8s %9s %9s %9s %9s %9s %10s\n", "op", "n", "fetch", "pages", "sems",
           "ldr-span", "gmem-wait", "cons-1st", "cons-span", "store-spn", "TOTAL");
    for (int op = 1; op < kNumOpcodes; ++op) {
      const Agg& a2 = agg[op];
      if (!a2.n) continue;
      printf("  %-9s %5d %8.0f %8.0f %8.0f %9.0f %9.0f %9.0f %9.0f %9.0f %10.0f\n", opname[op], a2.n,
             a2.fetch / a2.n, a2.page / a2.n, a2.sems / a2.n, a2.ldr_span / a2.n, a2.gmem_wait / a2.n,
             a2.cons_first / a2.n, a2.cons_span / a2.n, a2.store_span / a2.n, a2.total / a2.n);
    }
  }
  return 0;
}

#endif  // MK42_NO_MAIN
