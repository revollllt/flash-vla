// Template 43 -- the Mirage Persistent Kernel (MPK) runtime: a task graph
// executed by worker CTAs and scheduler warps inside one launch (sm90).
//
// MPK (mirage-project/mirage, Apache-2.0; arXiv 2512.22219) compiles a model
// into SM-level TASKS joined by EVENTS and runs the whole thing -- every
// layer, and every decode step after the first -- in a single persistent
// kernel. This file is that runtime, written against the toolkit, driving the
// same Llama-3.2-1B decode step as template 42 so the two schedulers can be
// compared on one workload. It builds, checks itself against a double
// reference over several decode iterations, and times itself.
//
// THE MACHINE, as upstream builds it
//
//   Grid = SM count. The first `num_workers` CTAs are WORKERS (256 threads on
//   Hopper). The remaining CTAs hold the SCHEDULERS: 4 warps each, one
//   scheduler per warp, lane 0 only (H100: 128 workers, 16 scheduler warps on
//   4 SMs). Roles are fixed at launch.
//
//   TASK  {type, trigger_event, dependent_event, input/output pointers, meta}.
//         A task has at most ONE event it waits on and ONE it triggers.
//   EVENT {type, num_triggers, [first_task, last_task)}. A counter in global
//         memory; the task that brings it to num_triggers x iteration pushes
//         the event onto a scheduler's queue.
//   WORKER QUEUE  per worker, a ring of TaskIds (iteration << 32 | position),
//         published by a scheduler with a release-add on `last_ready`.
//   SCHEDULER QUEUE  per scheduler, a ring of event indices, published by
//         workers with an atomic slot claim, a relaxed store, and a CAS loop
//         on `last_ready` that keeps the publishes in slot order.
//
//   worker loop     fetch up to 16 TaskIds (acquire on last_ready), cp.async
//                   the descriptors into shared memory, then per task: spin
//                   on the dependent event's counter, execute, trigger.
//   scheduler loop  pop an event; LAUNCH_TASKS -> enqueue [first, last) round
//                   robin over MY workers; LAUNCH_MASSIVE (>= 8 tasks) -> the
//                   range is split across all local schedulers via the
//                   broadcast queue; LAUNCH_DEPENDENT_TASKS -> the whole
//                   graph, iteration + 1; END_OF_TASK_GRAPH -> prepare the
//                   next decode step (position + 1) and launch its begin task,
//                   or terminate everyone.
//
// FOUR MECHANISMS THAT ARE THE POINT -- cumulative event counters (iteration
// i waits for num_triggers x i, so nothing is reset between decode steps),
// two launch modes of one graph (`aot` pre-enqueues the whole graph round
// robin; `jit` lets events launch their dependents), event granularity as a
// compiler decision (qkv -> attention is cut per KV head, which needs a
// head's q, k and v weight rows contiguous), and cross-task pre-loading of
// the next task's weights -- are the portable rules on
// [kernel-megakernel-forms].  The harness runs both launch modes and toggles
// the pre-load (prefetch=0).
//
// WHAT WAS SIMPLIFIED (deviations from upstream, not from the design):
//
//   * One process, one GPU: no NVSHMEM, no remote schedulers, no paged KV
//     request batching. Tasks are the seven decode-step kinds; the Hopper
//     linear tasks are warp-per-row GEMVs (batch 1 wants no tensor core:
//     wgmma at N = 8 is 5x waste [wgmma.issue.wg.ss]).
//   * Single kernel (workers and schedulers in one grid) rather than the two
//     concurrent kernels on two streams upstream defaults to; the queues are
//     the same, only the launch differs.
//   * Residual adds are exact: each o_proj / down task owns whole output rows,
//     so there is no split-K accumulation to order.
//
// UPSTREAM NUMBERS (paper): Qwen3-8B decode on A100 goes from 14.5 ms/token
// (SGLang/vLLM) to 12.5 ms with MPK against a ~10 ms bandwidth bound, i.e.
// 80% of the bound; H100 128 workers + 16 scheduler warps.
//
// STATUS (H100 SXM5, CUDA 13.1, clocks not pinned; 128 workers + 16 scheduler
// warps; the template-42 workload: Llama-1B, pos 1023, 16 layers + lm_head;
// 8 decode steps in one launch, per-step time = kernel time / 8; correctness
// over the 8 steps: logits within 3.7% of rms, argmax identical):
//
//   mode=aot (upstream's current runtime)    1.227 ms   2.04 TB/s   74% of floor
//   mode=aot prefetch=0                      1.208 ms   2.07 TB/s   75%
//   mode=jit                                 1.883 ms   1.33 TB/s   48%
//   mode=aot, 1 step per launch              1.257 ms   1.99 TB/s   72%
//
// Floor 0.906 ms [ld.bw.dev.dram]. The task profile (-DMK43_PROFILE=1) shows
// the long tasks at the per-SM bandwidth share (up/gate 512 KB in 20 us, down
// 256 KB in 10 us, lm_head 4 MB in 157 us) and the short ones -- qkv, o_proj,
// attention at 64 KB or less -- at 3.2-3.4 us each, latency-bound; workers are
// busy 67% of the span, the rest is dependency waits. Launch-on-fire costs a
// scheduler hop per task and loses 1.5x, which is why upstream pre-launches.
// The L2 pre-load of the next task's weights buys nothing here: the tasks
// that would profit are the short ones, and the pre-load reaches only one
// task ahead within a 16-task fetch batch.
//
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -o mk43 43_mpk_task_graph_runtime.cu -lcuda && ./mk43
//   ./mk43 layers=2 iters=4           2 layers + lm_head, 4 decode steps in one launch
//   ./mk43 mode=jit                   events launch their dependents
//   ./mk43 prefetch=0                 no cross-task pre-loading
//   -DMK43_PROFILE=1                  per-task timeline from %globaltimer
//
// CHECK-GRADE: reference
// CHECK-PTX: atom\.add\.release\.gpu\.u64
// CHECK-PTX: atom\.cas\.release\.gpu\.b64
// CHECK-PTX: ld\.acquire\.gpu\.u64
// CHECK-PTX: st\.relaxed\.gpu\.u64
// CHECK-PTX: cp\.async\.bulk\.prefetch\.L2\.global
// CHECK-PTX: cp\.async\.cg\.shared\.global
// CHECK-PTX: ld\.global\.L1::no_allocate\.v4\.b32
// CHECK-PTX: globaltimer

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
#include <map>

#ifndef MK43_LAYERS
#define MK43_LAYERS 16
#endif
#ifndef MK43_PROFILE
#define MK43_PROFILE 0
#endif
#ifndef MK43_ATTN_SPLITS
#define MK43_ATTN_SPLITS 16
#endif

namespace {

// ------------------------------------------------------------------ model
constexpr int kNumLayers = MK43_LAYERS;
constexpr int kHidden = 2048;
constexpr int kInter = 8192;
constexpr int kHeadDim = 64;
constexpr int kNumHeads = 32;
constexpr int kNumKvHeads = 8;
constexpr int kGqa = kNumHeads / kNumKvHeads;
constexpr int kVocab = 128256;
constexpr int kQkvDim = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;  // 3072
constexpr int kQkvGroupRows = (kGqa + 2) * kHeadDim;               // 384: one KV head's q, k, v rows
constexpr float kRopeBase = 500000.f;
constexpr float kEps = 1e-5f;

// ---------------------------------------------------------------- runtime
constexpr int kWorkerThreads = 256;
constexpr int kWorkerWarps = kWorkerThreads / 32;
constexpr int kSchedWarpsPerCta = 4;
constexpr int kTaskBatch = 16;             // descriptors fetched per queue read
constexpr int kMaxInputs = 8;
constexpr int kMaxOutputs = 3;
constexpr uint64_t kTaskInvalid = 0x7fffffffffffffffull;
constexpr uint64_t kEventInvalid = 0x7ffffffffffffffeull;

enum TaskType : int32_t {
  kTaskTerminate = 0,
  kTaskBeginTaskGraph = 10,
  kTaskRmsNormQkvRope = 101,     // norm + qkv rows + RoPE + KV append
  kTaskAttentionSplitKv = 102,   // one KV head, a range of keys, 4 q heads
  kTaskAttentionMerge = 103,     // LSE merge of the splits of one KV head
  kTaskLinearResidual = 104,     // rows of W x in, added into hidden (o_proj)
  kTaskRmsNormUpgateSilu = 105,  // norm + up/gate rows + SiLU
  kTaskDownResidual = 106,       // rows of down x silu_out, added into hidden
  kTaskRmsNormLmHead = 107,
};

enum EventType : int32_t {
  kEventEmpty = 900,
  kEventLaunchTasks = 901,
  kEventLaunchMassiveTasks = 902,
  kEventLaunchDependentTasks = 903,
  kEventEndOfTaskGraph = 910,
  kEventTermination = 911,
};

struct alignas(16) TaskDesc {
  int32_t task_type;
  int32_t variant;      // task kind's own parameter (layer, or kv head)
  int32_t meta[4];      // [row0, row1); attention: key0, key1, layer, split
  uint64_t trigger_event;
  uint64_t dependent_event;
  void* inputs[kMaxInputs];
  void* outputs[kMaxOutputs];
};
static_assert(sizeof(TaskDesc) % 16 == 0, "descriptor is copied in 16-byte pieces");

struct EventDesc {
  int32_t event_type;
  int32_t num_triggers;
  uint64_t first_task, last_task;
};

struct RuntimeConfig {
  int32_t num_workers, num_schedulers, num_events, num_tasks;
  uint64_t per_worker_queue_len, per_sched_queue_len;
  uint64_t* worker_queue_last_ready;   // [num_workers]
  uint64_t* sched_queue_last_ready;    // [num_schedulers + 1]
  uint64_t* sched_queue_next_free;     // [num_schedulers + 1]
  uint64_t* event_counters;            // [num_events], cumulative
  int32_t* event_num_triggers;         // [num_events]
  TaskDesc* tasks;
  EventDesc* events;
  uint64_t* worker_queues;             // [num_workers][per_worker_queue_len]
  uint64_t* sched_queues;              // [num_schedulers + 1][per_sched_queue_len]
  int32_t* step;                       // decode position of the current iteration
  int32_t num_iterations;
  int32_t prefetch_next;               // cross-task pre-loading on/off
  uint64_t* profile;                   // [num_workers][kProfileEntries] or null
  int32_t profile_entries;
};

// Model buffers every task can reach; the graph builder fills TaskDesc
// pointers from these so a task never dereferences a table.
struct Model {
  const __nv_bfloat16* qkv_w;       // [L][3072][2048], KV-head-grouped rows
  const __nv_bfloat16* o_w;         // [L][2048][2048]
  const __nv_bfloat16* up_w;        // [L][8192][2048]
  const __nv_bfloat16* gate_w;      // [L][8192][2048]
  const __nv_bfloat16* down_w;      // [L][2048][8192]
  const __nv_bfloat16* lm_head_w;   // [128256][2048]
  const __nv_bfloat16* attn_norm;   // [L][2048]
  const __nv_bfloat16* mlp_norm;    // [L][2048]
  const __nv_bfloat16* lm_norm;     // [2048]
  __nv_bfloat16* k_cache;           // [L][kv][max_seq][64]
  __nv_bfloat16* v_cache;
  const float* rope_cos;            // [max_seq][64], pair-duplicated
  const float* rope_sin;
  __nv_bfloat16* hidden;            // [2048]
  __nv_bfloat16* q;                 // [2048] post-RoPE
  float* attn_part;                 // [kv][splits][4][64]
  float* attn_ml;                   // [kv][splits][4][2]
  __nv_bfloat16* attn_out;          // [2048]
  __nv_bfloat16* silu_out;          // [8192]
  __nv_bfloat16* logits;            // [128256]
  int32_t max_seq;
  float attn_scale;
};

// ------------------------------------------------------------- atoms
// Upstream's mpk_atoms.cuh, verbatim in spirit: the queue protocol needs
// exactly these orderings and no others.

__device__ __forceinline__ uint64_t atom_add_release_gpu_u64(uint64_t* p, uint64_t v) {
  uint64_t old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v) : "memory");
  return old;
}
__device__ __forceinline__ uint64_t atom_cas_release_gpu_u64(uint64_t* p, uint64_t cmp, uint64_t v) {
  uint64_t old;
  asm volatile("atom.cas.release.gpu.b64 %0, [%1], %2, %3;" : "=l"(old) : "l"(p), "l"(cmp), "l"(v) : "memory");
  return old;
}
__device__ __forceinline__ uint64_t ld_acquire_gpu_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ int32_t ld_acquire_gpu_s32(const int32_t* p) {
  int32_t v;
  asm volatile("ld.acquire.gpu.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// The pre-loading phase: no page, no barrier, just requests in flight.
__device__ __forceinline__ void prefetch_l2_bulk(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ uint64_t task_iteration(uint64_t id) { return id >> 32; }
__device__ __forceinline__ uint64_t task_position(uint64_t id) { return id & 0xffffffffull; }
__device__ __forceinline__ uint64_t make_task_id(uint64_t iter, uint64_t pos) { return (iter << 32) | pos; }
__device__ __forceinline__ uint64_t event_index(uint64_t id) { return id & 0xffffffffull; }

// ----------------------------------------------------------- math bits

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
__device__ __forceinline__ uint4 ldg_weight(const void* p) {
  uint4 v;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// RMSNorm of the 2048-vector into shared memory as bf16, by all 256 threads
// (8 elements per thread). Every task that needs it does it: one hop
// [atom.lat.dev.hop] costs more than 128 CTAs re-reading 4 KB from L2.
__device__ __forceinline__ void rms_norm_to_smem(const __nv_bfloat16* __restrict__ x,
                                                 const __nv_bfloat16* __restrict__ w,
                                                 __nv_bfloat16* xs, float* red) {
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  float f[8];
  bf16x8_unpack(*reinterpret_cast<const uint4*>(x + tid * 8), f);
  float ss = 0.f;
  #pragma unroll
  for (int e = 0; e < 8; ++e) { ss += f[e] * f[e]; }
  ss = warp_sum(ss);
  if (lane == 0) { red[warp] = ss; }
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int i = 0; i < kWorkerWarps; ++i) { total += red[i]; }
  const float scale = rsqrtf(total / static_cast<float>(kHidden) + kEps);
  float wf[8];
  bf16x8_unpack(*reinterpret_cast<const uint4*>(w + tid * 8), wf);
  uint4 out;
  uint32_t* po = reinterpret_cast<uint32_t*>(&out);
  #pragma unroll
  for (int i = 0; i < 4; ++i) { po[i] = pack_bf16x2(f[2 * i] * scale * wf[2 * i], f[2 * i + 1] * scale * wf[2 * i + 1]); }
  *reinterpret_cast<uint4*>(xs + tid * 8) = out;
  __syncthreads();
}

// One warp, one weight row of length K against a bf16 vector in shared
// memory. 8 loads per lane are in flight per 2048 columns.
template <int K>
__device__ __forceinline__ float warp_dot_row(const __nv_bfloat16* __restrict__ wrow, const __nv_bfloat16* xs, int lane) {
  static_assert(K % 256 == 0, "");
  float acc = 0.f;
  #pragma unroll 8
  for (int i = 0; i < K / 256; ++i) {
    const int col = (i * 32 + lane) * 8;
    float w[8], x[8];
    bf16x8_unpack(ldg_weight(wrow + col), w);
    bf16x8_unpack(*reinterpret_cast<const uint4*>(xs + col), x);
    #pragma unroll
    for (int e = 0; e < 8; ++e) { acc = fmaf(w[e], x[e], acc); }
  }
  return warp_sum(acc);
}

// ================================================================== tasks
//
// Each task is a __device__ function over 256 threads with the descriptor
// and the model in hand. Output partitions are contiguous row ranges
// [meta[0], meta[1]); `variant` is the layer, or the KV head for attention.

struct SmemWorker {
  alignas(16) __nv_bfloat16 xs[kHidden];  // normalized input, or attention scratch
  float red[kWorkerWarps];
  float attn_o[kWorkerWarps][kGqa][kHeadDim];
  float attn_ml[kWorkerWarps][kGqa][2];
};

// 101: rows [r0, r1) of the grouped QKV projection. Rows come in pairs so a
// warp holds both elements of a RoPE pair.
__device__ void task_rmsnorm_qkv_rope(const TaskDesc& t, const Model& m, SmemWorker& sm, int pos) {
  const int layer = t.variant;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  rms_norm_to_smem(m.hidden, m.attn_norm + static_cast<int64_t>(layer) * kHidden, sm.xs, sm.red);
  const __nv_bfloat16* W = m.qkv_w + static_cast<int64_t>(layer) * kQkvDim * kHidden;
  for (int r = t.meta[0] + warp * 2; r < t.meta[1]; r += kWorkerWarps * 2) {
    const float a = warp_dot_row<kHidden>(W + static_cast<int64_t>(r) * kHidden, sm.xs, lane);
    const float b = warp_dot_row<kHidden>(W + static_cast<int64_t>(r + 1) * kHidden, sm.xs, lane);
    if (lane == 0) {
      // Grouped row order: [q heads 4g..4g+3 | k head g | v head g] per KV head g.
      const int g = r / kQkvGroupRows, in = r % kQkvGroupRows;
      const int d = in % kHeadDim;  // even; d+1 is the pair
      float x0 = a, x1 = b;  // RoPE on the fp32 projection, one rounding at the store
      const bool is_v = in >= (kGqa + 1) * kHeadDim;
      if (!is_v) {
        const float c = m.rope_cos[static_cast<int64_t>(pos) * kHeadDim + d];
        const float s = m.rope_sin[static_cast<int64_t>(pos) * kHeadDim + d];
        const float y0 = x0 * c - x1 * s, y1 = x1 * c + x0 * s;
        x0 = y0; x1 = y1;
      }
      const uint32_t packed = pack_bf16x2(x0, x1);
      if (in < kGqa * kHeadDim) {
        const int head = g * kGqa + in / kHeadDim;
        *reinterpret_cast<uint32_t*>(m.q + head * kHeadDim + d) = packed;
      } else {
        __nv_bfloat16* cache = is_v ? m.v_cache : m.k_cache;
        cache += ((static_cast<int64_t>(layer) * kNumKvHeads + g) * m.max_seq + pos) * kHeadDim + d;
        *reinterpret_cast<uint32_t*>(cache) = packed;
      }
    }
  }
}

// 102: keys [k0, k1) of KV head `variant`, all 4 of its query heads. Eight
// lanes share a key (8 dims each), four keys per warp per step, eight warps.
__device__ void task_attention_split(const TaskDesc& t, const Model& m, SmemWorker& sm, int pos) {
  const int kvh = t.variant, layer = t.meta[2], split = t.meta[3];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int slot = lane / 8, chunk = lane % 8;
  const int k0 = t.meta[0], k1 = min(t.meta[1], pos + 1);
  const __nv_bfloat16* K = m.k_cache + (static_cast<int64_t>(layer) * kNumKvHeads + kvh) * m.max_seq * kHeadDim;
  const __nv_bfloat16* V = m.v_cache + (static_cast<int64_t>(layer) * kNumKvHeads + kvh) * m.max_seq * kHeadDim;
  const float temp = m.attn_scale * 1.4426950408889634f;
  float q[kGqa][8];
  #pragma unroll
  for (int h = 0; h < kGqa; ++h) {
    bf16x8_unpack(*reinterpret_cast<const uint4*>(m.q + (kvh * kGqa + h) * kHeadDim + chunk * 8), q[h]);
    #pragma unroll
    for (int e = 0; e < 8; ++e) { q[h][e] *= temp; }
  }
  float mm[kGqa], ll[kGqa], o[kGqa][8];
  #pragma unroll
  for (int h = 0; h < kGqa; ++h) {
    mm[h] = -1e30f; ll[h] = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { o[h][e] = 0.f; }
  }
  constexpr int kUnroll = 4;
  for (int base = k0 + warp * 4 + slot; base < k1; base += kWorkerWarps * 4 * kUnroll) {
    uint4 kr[kUnroll], vr[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int key = base + u * kWorkerWarps * 4;
      kr[u] = make_uint4(0, 0, 0, 0); vr[u] = kr[u];
      if (key < k1) {
        kr[u] = *reinterpret_cast<const uint4*>(K + static_cast<int64_t>(key) * kHeadDim + chunk * 8);
        vr[u] = *reinterpret_cast<const uint4*>(V + static_cast<int64_t>(key) * kHeadDim + chunk * 8);
      }
    }
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int key = base + u * kWorkerWarps * 4;
      float kf[8], vf[8];
      bf16x8_unpack(kr[u], kf);
      bf16x8_unpack(vr[u], vf);
      #pragma unroll
      for (int h = 0; h < kGqa; ++h) {
        float s = 0.f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { s = fmaf(q[h][e], kf[e], s); }
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        if (key >= k1) { s = -1e30f; }
        const float m_new = fmaxf(mm[h], s);
        const float rescale = exp2f(mm[h] - m_new);
        const float p = exp2f(s - m_new);
        mm[h] = m_new;
        ll[h] = ll[h] * rescale + p;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { o[h][e] = fmaf(p, vf[e], o[h][e] * rescale); }
      }
    }
  }
  // Merge the 4 key slots of the warp, then the 8 warps through shared memory.
  #pragma unroll
  for (int off = 8; off < 32; off <<= 1) {
    #pragma unroll
    for (int h = 0; h < kGqa; ++h) {
      const float m2 = __shfl_xor_sync(0xffffffffu, mm[h], off);
      const float l2 = __shfl_xor_sync(0xffffffffu, ll[h], off);
      const float mx = fmaxf(mm[h], m2);
      const float r1 = exp2f(mm[h] - mx), r2 = exp2f(m2 - mx);
      ll[h] = ll[h] * r1 + l2 * r2;
      #pragma unroll
      for (int e = 0; e < 8; ++e) { o[h][e] = o[h][e] * r1 + __shfl_xor_sync(0xffffffffu, o[h][e], off) * r2; }
      mm[h] = mx;
    }
  }
  if (lane < 8) {
    #pragma unroll
    for (int h = 0; h < kGqa; ++h) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) { sm.attn_o[warp][h][chunk * 8 + e] = o[h][e]; }
      if (lane == 0) { sm.attn_ml[warp][h][0] = mm[h]; sm.attn_ml[warp][h][1] = ll[h]; }
    }
  }
  __syncthreads();
  float* part = m.attn_part + ((static_cast<int64_t>(kvh) * MK43_ATTN_SPLITS + split) * kGqa) * kHeadDim;
  float* ml = m.attn_ml + ((static_cast<int64_t>(kvh) * MK43_ATTN_SPLITS + split) * kGqa) * 2;
  if (threadIdx.x < kGqa * kHeadDim) {
    const int h = threadIdx.x / kHeadDim, d = threadIdx.x % kHeadDim;
    float mx = -1e30f;
    #pragma unroll
    for (int w = 0; w < kWorkerWarps; ++w) { mx = fmaxf(mx, sm.attn_ml[w][h][0]); }
    float lsum = 0.f, osum = 0.f;
    #pragma unroll
    for (int w = 0; w < kWorkerWarps; ++w) {
      const float sc = exp2f(sm.attn_ml[w][h][0] - mx);
      lsum += sm.attn_ml[w][h][1] * sc;
      osum += sm.attn_o[w][h][d] * sc;
    }
    part[h * kHeadDim + d] = osum;
    if (d == 0) { ml[h * 2] = mx; ml[h * 2 + 1] = lsum; }
  }
}

// 103: merge the splits of KV head `variant` into attn_out, one warp per head.
__device__ void task_attention_merge(const TaskDesc& t, const Model& m) {
  const int kvh = t.variant;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (warp >= kGqa) { return; }
  const float* part = m.attn_part + (static_cast<int64_t>(kvh) * MK43_ATTN_SPLITS * kGqa) * kHeadDim;
  const float* ml = m.attn_ml + (static_cast<int64_t>(kvh) * MK43_ATTN_SPLITS * kGqa) * 2;
  float mx = -1e30f;
  for (int s = 0; s < MK43_ATTN_SPLITS; ++s) { mx = fmaxf(mx, ml[(s * kGqa + warp) * 2]); }
  float lsum = 0.f, o0 = 0.f, o1 = 0.f;
  for (int s = 0; s < MK43_ATTN_SPLITS; ++s) {
    const float sc = exp2f(ml[(s * kGqa + warp) * 2] - mx);
    lsum += ml[(s * kGqa + warp) * 2 + 1] * sc;
    o0 += part[(s * kGqa + warp) * kHeadDim + lane * 2] * sc;
    o1 += part[(s * kGqa + warp) * kHeadDim + lane * 2 + 1] * sc;
  }
  *reinterpret_cast<uint32_t*>(m.attn_out + (kvh * kGqa + warp) * kHeadDim + lane * 2) = pack_bf16x2(o0 / lsum, o1 / lsum);
}

// 104 / 106: rows [r0, r1) of W (K = 2048 or 8192) x a bf16 vector, added
// into hidden. Whole rows per task, so the residual add is a plain store.
template <int K>
__device__ void task_linear_residual(const TaskDesc& t, const Model& m, SmemWorker& sm, const __nv_bfloat16* W,
                                     const __nv_bfloat16* in) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if constexpr (K == kHidden) {
    for (int i = threadIdx.x; i < K / 8; i += kWorkerThreads) {
      *reinterpret_cast<uint4*>(sm.xs + i * 8) = *reinterpret_cast<const uint4*>(in + i * 8);
    }
    __syncthreads();
    for (int r = t.meta[0] + warp; r < t.meta[1]; r += kWorkerWarps) {
      const float acc = warp_dot_row<K>(W + static_cast<int64_t>(r) * K, sm.xs, lane);
      if (lane == 0) { m.hidden[r] = __float2bfloat16(__bfloat162float(m.hidden[r]) + __bfloat162float(__float2bfloat16(acc))); }
    }
  } else {
    // K = 8192: the input does not fit the norm buffer; read it from L2.
    for (int r = t.meta[0] + warp; r < t.meta[1]; r += kWorkerWarps) {
      const __nv_bfloat16* wrow = W + static_cast<int64_t>(r) * K;
      float acc = 0.f;
      #pragma unroll 8
      for (int i = 0; i < K / 256; ++i) {
        const int col = (i * 32 + lane) * 8;
        float w[8], x[8];
        bf16x8_unpack(ldg_weight(wrow + col), w);
        bf16x8_unpack(*reinterpret_cast<const uint4*>(in + col), x);
        #pragma unroll
        for (int e = 0; e < 8; ++e) { acc = fmaf(w[e], x[e], acc); }
      }
      acc = warp_sum(acc);
      if (lane == 0) { m.hidden[r] = __float2bfloat16(__bfloat162float(m.hidden[r]) + __bfloat162float(__float2bfloat16(acc))); }
    }
  }
}

// 105: rows [r0, r1) of up and gate, SiLU-gated into silu_out.
__device__ void task_rmsnorm_upgate_silu(const TaskDesc& t, const Model& m, SmemWorker& sm) {
  const int layer = t.variant;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  rms_norm_to_smem(m.hidden, m.mlp_norm + static_cast<int64_t>(layer) * kHidden, sm.xs, sm.red);
  const __nv_bfloat16* U = m.up_w + static_cast<int64_t>(layer) * kInter * kHidden;
  const __nv_bfloat16* G = m.gate_w + static_cast<int64_t>(layer) * kInter * kHidden;
  for (int r = t.meta[0] + warp; r < t.meta[1]; r += kWorkerWarps) {
    const float u = warp_dot_row<kHidden>(U + static_cast<int64_t>(r) * kHidden, sm.xs, lane);
    const float g = warp_dot_row<kHidden>(G + static_cast<int64_t>(r) * kHidden, sm.xs, lane);
    if (lane == 0) { m.silu_out[r] = __float2bfloat16(silu(g) * u); }
  }
}

// 107: rows [r0, r1) of the vocabulary projection.
__device__ void task_rmsnorm_lm_head(const TaskDesc& t, const Model& m, SmemWorker& sm) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  rms_norm_to_smem(m.hidden, m.lm_norm, sm.xs, sm.red);
  for (int r = t.meta[0] + warp; r < t.meta[1]; r += kWorkerWarps) {
    const float v = warp_dot_row<kHidden>(m.lm_head_w + static_cast<int64_t>(r) * kHidden, sm.xs, lane);
    if (lane == 0) { m.logits[r] = __float2bfloat16(v); }
  }
}

// The weight bytes a task will stream, for the pre-loading phase.
__device__ __forceinline__ void task_weight_range(const TaskDesc& t, const Model& m, const void*& p, uint32_t& bytes) {
  const int rows = t.meta[1] - t.meta[0];
  switch (t.task_type) {
    case kTaskRmsNormQkvRope:
      p = m.qkv_w + (static_cast<int64_t>(t.variant) * kQkvDim + t.meta[0]) * kHidden; bytes = rows * kHidden * 2; break;
    case kTaskLinearResidual:
      p = m.o_w + (static_cast<int64_t>(t.variant) * kHidden + t.meta[0]) * kHidden; bytes = rows * kHidden * 2; break;
    case kTaskRmsNormUpgateSilu:
      p = m.up_w + (static_cast<int64_t>(t.variant) * kInter + t.meta[0]) * kHidden; bytes = rows * kHidden * 2; break;
    case kTaskDownResidual:
      p = m.down_w + (static_cast<int64_t>(t.variant) * kHidden + t.meta[0]) * kInter; bytes = rows * kInter * 2; break;
    case kTaskRmsNormLmHead:
      p = m.lm_head_w + static_cast<int64_t>(t.meta[0]) * kHidden; bytes = rows * kHidden * 2; break;
    default: p = nullptr; bytes = 0; break;
  }
}

__device__ void execute_task(const TaskDesc& t, const Model& m, SmemWorker& sm, int pos) {
  switch (t.task_type) {
    case kTaskRmsNormQkvRope: task_rmsnorm_qkv_rope(t, m, sm, pos); break;
    case kTaskAttentionSplitKv: task_attention_split(t, m, sm, pos); break;
    case kTaskAttentionMerge: task_attention_merge(t, m); break;
    case kTaskLinearResidual:
      task_linear_residual<kHidden>(t, m, sm, m.o_w + static_cast<int64_t>(t.variant) * kHidden * kHidden, m.attn_out); break;
    case kTaskRmsNormUpgateSilu: task_rmsnorm_upgate_silu(t, m, sm); break;
    case kTaskDownResidual:
      task_linear_residual<kInter>(t, m, sm, m.down_w + static_cast<int64_t>(t.variant) * kHidden * kInter, m.silu_out); break;
    case kTaskRmsNormLmHead: task_rmsnorm_lm_head(t, m, sm); break;
    default: break;
  }
}

// ================================================================= worker

__device__ void execute_worker(const RuntimeConfig& cfg, const Model& m) {
  __shared__ alignas(16) TaskDesc descs[kTaskBatch];
  __shared__ uint64_t ids[kTaskBatch];
  __shared__ uint64_t next_pos, last_pos;
  __shared__ SmemWorker sm;
  const int worker = blockIdx.x;
  uint64_t* queue = cfg.worker_queues + static_cast<uint64_t>(worker) * cfg.per_worker_queue_len;
  if (threadIdx.x == 0) { next_pos = 0; last_pos = 0; }
  __syncthreads();
  int qpos = 0, qlen = 0;
  uint64_t* prof = cfg.profile ? cfg.profile + static_cast<uint64_t>(worker) * cfg.profile_entries : nullptr;
  int prof_n = 0;
  while (true) {
    if (qpos == qlen) {
      if (threadIdx.x == 0) {
        while (next_pos == last_pos) {
          last_pos = ld_acquire_gpu_u64(&cfg.worker_queue_last_ready[worker]);
          if (next_pos < last_pos) { break; }
          __nanosleep(10);
        }
      }
      __syncthreads();
      const int n = min(static_cast<int>(last_pos - next_pos), kTaskBatch);
      if (threadIdx.x < n) { ids[threadIdx.x] = ld_relaxed_gpu_u64(&queue[(next_pos + threadIdx.x) % cfg.per_worker_queue_len]); }
      __syncthreads();
      if (threadIdx.x == 0) { next_pos += n; }
      // Descriptors, 16 bytes per thread per copy, through the async proxy.
      constexpr int kPieces = sizeof(TaskDesc) / 16;
      for (int i = threadIdx.x; i < n * kPieces; i += kWorkerThreads) {
        const int ti = i / kPieces, piece = i % kPieces;
        cp_async_16(reinterpret_cast<char*>(&descs[ti]) + piece * 16,
                    reinterpret_cast<const char*>(&cfg.tasks[task_position(ids[ti])]) + piece * 16);
      }
      cp_async_wait_all();
      __syncthreads();
      qpos = 0; qlen = n;
    }
    const TaskDesc& t = descs[qpos];
    const uint64_t iter = task_iteration(ids[qpos]);
    // Pre-loading phase for the NEXT task, before this one's dependency wait.
    if (cfg.prefetch_next && qpos + 1 < qlen && threadIdx.x < 32) {
      const void* p = nullptr; uint32_t bytes = 0;
      task_weight_range(descs[qpos + 1], m, p, bytes);
      if (p != nullptr) {
        const uint32_t chunk = 16384;
        for (uint32_t off = threadIdx.x * chunk; off < bytes; off += 32 * chunk) {
          prefetch_l2_bulk(static_cast<const char*>(p) + off, min(chunk, bytes - off));
        }
      }
    }
    // Dependency: the event's counter must reach num_triggers x iteration.
    if (threadIdx.x == 0 && t.dependent_event != kEventInvalid) {
      const uint64_t e = event_index(t.dependent_event);
      const uint64_t need = static_cast<uint64_t>(cfg.event_num_triggers[e]) * iter;
      while (ld_acquire_gpu_u64(&cfg.event_counters[e]) < need) { __nanosleep(10); }
    }
    __syncthreads();
    if (t.task_type == kTaskTerminate) { return; }
    const uint64_t t0 = (prof && threadIdx.x == 0) ? globaltimer_ns() : 0;
    if (t.task_type != kTaskBeginTaskGraph) {
      const int pos = ld_acquire_gpu_s32(cfg.step);
      execute_task(t, m, sm, pos);
    }
    __syncthreads();
    if (prof && threadIdx.x == 0 && prof_n + 2 <= cfg.profile_entries) {
      // Two words per task: (type << 32 | position), (start << 32 | duration) in ns.
      const uint64_t t1 = globaltimer_ns();
      prof[prof_n++] = (static_cast<uint64_t>(t.task_type) << 32) | task_position(ids[qpos]);
      prof[prof_n++] = ((t0 & 0xffffffffull) << 32) | ((t1 - t0) & 0xffffffffull);
    }
    // Trigger: release-add, and the last trigger publishes the event.
    if (threadIdx.x == 0 && t.trigger_event != kEventInvalid) {
      const uint64_t e = event_index(t.trigger_event);
      const uint64_t count = atom_add_release_gpu_u64(&cfg.event_counters[e], 1);
      const uint64_t need = static_cast<uint64_t>(cfg.event_num_triggers[e]) * iter;
      if (count + 1 == need) {
        const EventDesc ev = cfg.events[e];
        if (ev.event_type != kEventEmpty) {
          const bool bcast = (ev.event_type == kEventLaunchMassiveTasks || ev.event_type == kEventLaunchDependentTasks);
          const int sched = bcast ? cfg.num_schedulers
                                  : worker / ((cfg.num_workers + cfg.num_schedulers - 1) / cfg.num_schedulers);
          uint64_t* sq = cfg.sched_queues + static_cast<uint64_t>(sched) * cfg.per_sched_queue_len;
          const uint64_t slot = atom_add_release_gpu_u64(&cfg.sched_queue_next_free[sched], 1);
          st_relaxed_gpu_u64(&sq[slot % cfg.per_sched_queue_len], e);
          // Publish in slot order: whoever claimed slot k waits for k to be
          // the last ready before making k + 1 ready.
          uint64_t old;
          do { old = atom_cas_release_gpu_u64(&cfg.sched_queue_last_ready[sched], slot, slot + 1); } while (old != slot);
        }
      }
    }
    ++qpos;
  }
}

// ============================================================== scheduler

__device__ __forceinline__ void push_task(const RuntimeConfig& cfg, int worker, uint64_t* next_free_pos, uint64_t id) {
  uint64_t* wq = cfg.worker_queues + static_cast<uint64_t>(worker) * cfg.per_worker_queue_len;
  const uint64_t slot = (*next_free_pos)++;
  st_relaxed_gpu_u64(&wq[slot % cfg.per_worker_queue_len], id);
  atom_add_release_gpu_u64(&cfg.worker_queue_last_ready[worker], 1);
}

__device__ void terminate_schedulers(const RuntimeConfig& cfg) {
  for (int s = 0; s < cfg.num_schedulers; ++s) {
    uint64_t* sq = cfg.sched_queues + static_cast<uint64_t>(s) * cfg.per_sched_queue_len;
    const uint64_t slot = atom_add_release_gpu_u64(&cfg.sched_queue_next_free[s], 1);
    st_relaxed_gpu_u64(&sq[slot % cfg.per_sched_queue_len], 0);  // event 0 = termination
    uint64_t old;
    do { old = atom_cas_release_gpu_u64(&cfg.sched_queue_last_ready[s], slot, slot + 1); } while (old != slot);
  }
}

__device__ void execute_scheduler(const RuntimeConfig& cfg, int first_sched_cta) {
  const int warp = threadIdx.x / 32;
  if (threadIdx.x % 32 != 0 || warp >= kSchedWarpsPerCta) { return; }
  const int sched = (blockIdx.x - first_sched_cta) * kSchedWarpsPerCta + warp;
  if (sched >= cfg.num_schedulers) { return; }
  // My workers: a contiguous slice; plus the broadcast queue every local
  // scheduler drains.
  const int per = (cfg.num_workers + cfg.num_schedulers - 1) / cfg.num_schedulers;
  const int w0 = sched * per, w1 = min(cfg.num_workers, w0 + per);
  uint64_t next_free[64];  // per-worker enqueue cursor; this scheduler is their only producer
  for (int i = 0; i < 64; ++i) { next_free[i] = 0; }
  const int nq = 2;
  const int qids[2] = {sched, cfg.num_schedulers};
  uint64_t cur[2] = {0, 0}, last[2] = {0, 0};
  int qi = 0;
  int next_worker = w0;
  uint64_t iteration = 0;
  while (true) {
    while (cur[qi] == last[qi]) {
      last[qi] = ld_acquire_gpu_u64(&cfg.sched_queue_last_ready[qids[qi]]);
      if (cur[qi] < last[qi]) { break; }
      qi = (qi + 1) % nq;
      __nanosleep(10);
    }
    const uint64_t* sq = cfg.sched_queues + static_cast<uint64_t>(qids[qi]) * cfg.per_sched_queue_len;
    const uint64_t e = ld_relaxed_gpu_u64(&sq[cur[qi] % cfg.per_sched_queue_len]);
    const EventDesc ev = cfg.events[e];
    if (e == 0) {  // termination: hand every worker of mine the terminate task
      for (int w = w0; w < w1; ++w) { push_task(cfg, w, &next_free[w - w0], make_task_id(iteration, 0)); }
      return;
    }
    // Event 1 opens an iteration; it reaches every scheduler (broadcast
    // queue), so the per-scheduler iteration counters agree.
    if (e == 1) { ++iteration; }
    if (ev.event_type == kEventEndOfTaskGraph) {
      // prepare_next_batch: another decode step, or done. The seeded first
      // END (iteration 0) starts step pos0 as the host set it.
      if (static_cast<int32_t>(iteration) >= cfg.num_iterations) {
        terminate_schedulers(cfg);
      } else {
        if (iteration > 0) {
          const int32_t step = *cfg.step + 1;
          asm volatile("st.release.gpu.s32 [%0], %1;" ::"l"(cfg.step), "r"(step) : "memory");
        }
        push_task(cfg, next_worker, &next_free[next_worker - w0], make_task_id(iteration + 1, 1));
        next_worker = (next_worker + 1 == w1) ? w0 : next_worker + 1;
      }
    } else if (ev.event_type == kEventLaunchDependentTasks) {
      // The whole graph, interleaved over all workers by task index so that
      // consecutive tasks land on consecutive SMs.
      const uint64_t n = ev.last_task - ev.first_task;
      for (uint64_t i = 0; i < (n + cfg.num_workers - 1) / cfg.num_workers; ++i) {
        for (int w = w0; w < w1; ++w) {
          const uint64_t pos = ev.first_task + i * cfg.num_workers + w;
          if (pos < ev.last_task) { push_task(cfg, w, &next_free[w - w0], make_task_id(iteration, pos)); }
        }
      }
    } else {
      uint64_t f = ev.first_task, l = ev.last_task;
      if (ev.event_type == kEventLaunchMassiveTasks) {
        // Split the range across the local schedulers.
        const uint64_t n = l - f, per_s = n / cfg.num_schedulers, rem = n % cfg.num_schedulers;
        const uint64_t my0 = sched < static_cast<int>(rem) ? (per_s + 1) * sched : per_s * sched + rem;
        const uint64_t my1 = my0 + per_s + (sched < static_cast<int>(rem) ? 1 : 0);
        f += my0; l = ev.first_task + my1;
      }
      for (uint64_t pos = f; pos < l; ++pos) {
        push_task(cfg, next_worker, &next_free[next_worker - w0], make_task_id(iteration, pos));
        next_worker = (next_worker + 1 == w1) ? w0 : next_worker + 1;
      }
    }
    cur[qi] += 1;
  }
}

}  // namespace

__global__ __launch_bounds__(kWorkerThreads, 1) void mpk_persistent_kernel(RuntimeConfig cfg, Model m) {
  if (static_cast<int>(blockIdx.x) < cfg.num_workers) {
    execute_worker(cfg, m);
  } else {
    execute_scheduler(cfg, cfg.num_workers);
  }
}

// Reset queue cursors and counters, and seed scheduler 0 with the
// end-of-graph event so the first step starts (upstream's prepare_kernel).
__global__ void mpk_prepare_kernel(RuntimeConfig cfg, int end_event) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cfg.num_workers) { cfg.worker_queue_last_ready[i] = 0; }
  if (i < cfg.num_schedulers + 1) { cfg.sched_queue_last_ready[i] = 0; cfg.sched_queue_next_free[i] = 0; }
  for (int e = i; e < cfg.num_events; e += gridDim.x * blockDim.x) { cfg.event_counters[e] = 0; }
  if (i == 0) {
    cfg.sched_queue_next_free[0] = 1;
    cfg.sched_queues[0] = end_event;
    cfg.sched_queue_last_ready[0] = 1;
  }
}

// ============================================================ graph builder
//
// Host side, as upstream's register_mugraph + dfs_create_events_add_tasks:
// ops in topological order, tasks partition each op's output rows, and each
// edge is cut into event_dim events over contiguous producer/consumer ranges.

#ifndef MK43_NO_MAIN

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace {

struct OpTasks { int first = 0, last = 0; };

struct GraphBuilder {
  std::vector<TaskDesc> tasks;
  std::vector<EventDesc> events;
  int workers;

  explicit GraphBuilder(int w) : workers(w) {
    // task 0 terminates, task 1 begins a graph iteration; event 0 terminates,
    // event 1 launches (the first op's tasks, or the whole graph in aot mode).
    TaskDesc t{}; t.task_type = kTaskTerminate; t.trigger_event = kEventInvalid; t.dependent_event = kEventInvalid;
    tasks.push_back(t);
    t.task_type = kTaskBeginTaskGraph; t.trigger_event = 1;
    tasks.push_back(t);
    events.push_back(EventDesc{kEventTermination, 0, 0, 0});
    events.push_back(EventDesc{kEventLaunchTasks, 1, 0, 0});
  }
  TaskDesc blank(int32_t type, int32_t variant, int32_t m0, int32_t m1, int32_t m2 = 0, int32_t m3 = 0) {
    TaskDesc t{}; t.task_type = type; t.variant = variant; t.meta[0] = m0; t.meta[1] = m1; t.meta[2] = m2; t.meta[3] = m3;
    t.trigger_event = kEventInvalid; t.dependent_event = kEventInvalid;
    return t;
  }
  // Adds an op's tasks and, per producer range i of `deps` (each split into
  // event_dim contiguous pieces), one event launching consumer piece i.
  OpTasks add_op(std::vector<TaskDesc> op_tasks, const std::vector<OpTasks>& deps, int event_dim) {
    OpTasks out;
    out.first = int(tasks.size());
    for (auto& t : op_tasks) tasks.push_back(t);
    out.last = int(tasks.size());
    const int n = out.last - out.first;
    if (deps.empty()) {
      // Roots are launched by event 1.
      events[1].first_task = out.first; events[1].last_task = out.last;
      events[1].event_type = n >= 8 ? kEventLaunchMassiveTasks : kEventLaunchTasks;
      return out;
    }
    for (int i = 0; i < event_dim; ++i) {
      EventDesc e{};
      e.first_task = out.first + int64_t(n) * i / event_dim;
      e.last_task = out.first + int64_t(n) * (i + 1) / event_dim;
      e.num_triggers = 0;
      for (const OpTasks& d : deps) {
        const int dn = d.last - d.first;
        const int p0 = d.first + int(int64_t(dn) * i / event_dim), p1 = d.first + int(int64_t(dn) * (i + 1) / event_dim);
        for (int p = p0; p < p1; ++p) {
          if (tasks[p].trigger_event != kEventInvalid) { printf("task %d triggers two events\n", p); exit(1); }
          tasks[p].trigger_event = events.size();
          e.num_triggers++;
        }
      }
      e.event_type = (e.last_task - e.first_task) >= 8 ? kEventLaunchMassiveTasks : kEventLaunchTasks;
      events.push_back(e);
    }
    return out;
  }
  void finish(const OpTasks& leaves, bool aot) {
    const int end_event = int(events.size());
    for (int p = leaves.first; p < leaves.last; ++p) tasks[p].trigger_event = end_event;
    events.push_back(EventDesc{kEventEndOfTaskGraph, leaves.last - leaves.first, 0, 0});
    if (aot) {
      // Upstream's current behaviour: pre-launch everything, demote launches
      // to counters the tasks wait on.
      events[1] = EventDesc{kEventLaunchDependentTasks, 1, 2, static_cast<uint64_t>(tasks.size())};
      for (size_t e = 2; e + 1 < events.size(); ++e) {
        if (events[e].event_type == kEventLaunchTasks || events[e].event_type == kEventLaunchMassiveTasks) {
          for (uint64_t t = events[e].first_task; t < events[e].last_task; ++t) tasks[t].dependent_event = e;
          events[e].event_type = kEventEmpty;
        }
      }
    }
  }
};

std::vector<TaskDesc> partition_rows(GraphBuilder& b, int32_t type, int32_t variant, int rows, int per_task, int align) {
  std::vector<TaskDesc> v;
  per_task = std::max(align, (per_task / align) * align);
  for (int r = 0; r < rows; r += per_task) v.push_back(b.blank(type, variant, r, std::min(rows, r + per_task)));
  return v;
}

// ------------------------------------------------------------- reference

float r_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

struct HostModel {
  std::vector<float> qkv, o, up, gate, down, lm_head, attn_norm, mlp_norm, lm_norm, rope_cos, rope_sin, k_cache, v_cache;
  int max_seq = 0;
};

void rms_ref(const std::vector<float>& x, const float* w, std::vector<double>& y) {
  double ss = 0.0;
  for (float v : x) ss += double(v) * v;
  const double sc = 1.0 / std::sqrt(ss / x.size() + kEps);
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * sc * w[i];
}
double dot_ref(const float* w, int64_t K, const std::vector<double>& x) {
  double acc = 0.0;
  for (int64_t k = 0; k < K; ++k) acc += double(w[k]) * x[k];
  return acc;
}

// One decode step at `pos`; hidden updated in place, KV appended.
void reference_step(HostModel& m, std::vector<float>& hidden, int pos, int layers, std::vector<float>* logits) {
  const int64_t S = m.max_seq;
  for (int L = 0; L < layers; ++L) {
    std::vector<double> xn;
    rms_ref(hidden, m.attn_norm.data() + int64_t(L) * kHidden, xn);
    std::vector<float> q(kNumHeads * kHeadDim);
    for (int r = 0; r < kQkvDim; r += 2) {
      const float* W = m.qkv.data() + (int64_t(L) * kQkvDim + r) * kHidden;
      float x0 = float(dot_ref(W, kHidden, xn)), x1 = float(dot_ref(W + kHidden, kHidden, xn));
      const int g = r / kQkvGroupRows, in = r % kQkvGroupRows, d = in % kHeadDim;
      const bool is_v = in >= (kGqa + 1) * kHeadDim;
      if (!is_v) {
        const double c = m.rope_cos[int64_t(pos) * kHeadDim + d], s = m.rope_sin[int64_t(pos) * kHeadDim + d];
        const float y0 = r_bf16(float(x0 * c - x1 * s)), y1 = r_bf16(float(x1 * c + x0 * s));
        x0 = y0; x1 = y1;
      } else {
        x0 = r_bf16(x0); x1 = r_bf16(x1);
      }
      if (in < kGqa * kHeadDim) {
        const int head = g * kGqa + in / kHeadDim;
        q[head * kHeadDim + d] = x0; q[head * kHeadDim + d + 1] = x1;
      } else {
        std::vector<float>& cache = is_v ? m.v_cache : m.k_cache;
        const int64_t off = ((int64_t(L) * kNumKvHeads + g) * S + pos) * kHeadDim + d;
        cache[off] = x0; cache[off + 1] = x1;
      }
    }
    std::vector<double> attn(kNumHeads * kHeadDim);
    const double scale = 1.0 / std::sqrt(double(kHeadDim));
    for (int h = 0; h < kNumHeads; ++h) {
      const int kvh = h / kGqa;
      const float* K = m.k_cache.data() + (int64_t(L) * kNumKvHeads + kvh) * S * kHeadDim;
      const float* V = m.v_cache.data() + (int64_t(L) * kNumKvHeads + kvh) * S * kHeadDim;
      std::vector<double> s(pos + 1);
      double mx = -1e300;
      for (int p = 0; p <= pos; ++p) {
        double acc = 0.0;
        for (int d = 0; d < kHeadDim; ++d) acc += double(q[h * kHeadDim + d]) * K[int64_t(p) * kHeadDim + d];
        s[p] = acc * scale; mx = std::max(mx, s[p]);
      }
      double l = 0.0;
      std::vector<double> o(kHeadDim, 0.0);
      for (int p = 0; p <= pos; ++p) {
        const double e = std::exp(s[p] - mx);
        l += e;
        for (int d = 0; d < kHeadDim; ++d) o[d] += e * V[int64_t(p) * kHeadDim + d];
      }
      for (int d = 0; d < kHeadDim; ++d) attn[h * kHeadDim + d] = r_bf16(float(o[d] / l));
    }
    for (int n = 0; n < kHidden; ++n) {
      const double v = dot_ref(m.o.data() + (int64_t(L) * kHidden + n) * kHidden, kHidden, attn);
      hidden[n] = r_bf16(hidden[n] + r_bf16(float(v)));
    }
    std::vector<double> xn2, silu_v(kInter);
    rms_ref(hidden, m.mlp_norm.data() + int64_t(L) * kHidden, xn2);
    for (int r = 0; r < kInter; ++r) {
      const double u = dot_ref(m.up.data() + (int64_t(L) * kInter + r) * kHidden, kHidden, xn2);
      const double gt = dot_ref(m.gate.data() + (int64_t(L) * kInter + r) * kHidden, kHidden, xn2);
      silu_v[r] = r_bf16(float((gt / (1.0 + std::exp(-gt))) * u));
    }
    for (int n = 0; n < kHidden; ++n) {
      const double v = dot_ref(m.down.data() + (int64_t(L) * kHidden + n) * kInter, kInter, silu_v);
      hidden[n] = r_bf16(hidden[n] + r_bf16(float(v)));
    }
  }
  if (logits) {
    std::vector<double> xn;
    rms_ref(hidden, m.lm_norm.data(), xn);
    logits->resize(kVocab);
    for (int r = 0; r < kVocab; ++r) (*logits)[r] = r_bf16(float(dot_ref(m.lm_head.data() + int64_t(r) * kHidden, kHidden, xn)));
  }
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
std::string arg_value(int argc, char** argv, const char* key, const char* dflt) {
  const std::string k = std::string(key) + "=";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).rfind(k, 0) == 0) { return std::string(argv[i]).substr(k.size()); }
  }
  return dflt;
}
const char* task_name(int t) {
  switch (t) {
    case kTaskRmsNormQkvRope: return "qkv";
    case kTaskAttentionSplitKv: return "attn";
    case kTaskAttentionMerge: return "merge";
    case kTaskLinearResidual: return "o_proj";
    case kTaskRmsNormUpgateSilu: return "upgate";
    case kTaskDownResidual: return "down";
    case kTaskRmsNormLmHead: return "lm_head";
    case kTaskBeginTaskGraph: return "begin";
    default: return "?";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int layers = std::stoi(arg_value(argc, argv, "layers", std::to_string(kNumLayers).c_str()));
  const int pos0 = std::stoi(arg_value(argc, argv, "pos", "1023"));
  const int iters = std::stoi(arg_value(argc, argv, "iters", "8"));
  const std::string mode = arg_value(argc, argv, "mode", "aot");
  const int prefetch = std::stoi(arg_value(argc, argv, "prefetch", "1"));
  const int timing_reps = std::stoi(arg_value(argc, argv, "reps", "5"));
  if (layers < 1 || layers > kNumLayers) { printf("layers must be 1..%d\n", kNumLayers); return 1; }
  const bool full = (layers == kNumLayers);

  int sm_count = 0;
  CK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const int num_schedulers = std::stoi(arg_value(argc, argv, "schedulers", "16"));
  const int sched_ctas = (num_schedulers + kSchedWarpsPerCta - 1) / kSchedWarpsPerCta;
  const int num_workers = std::stoi(arg_value(argc, argv, "workers", std::to_string(sm_count - sched_ctas).c_str()));
  if (num_workers > 64 * num_schedulers) { printf("at most 64 workers per scheduler\n"); return 1; }
  const int max_seq = pos0 + iters;

  printf("MPK-style task-graph runtime  (%d SMs: %d workers + %d scheduler warps on %d CTAs; %d layers%s, "
         "pos %d, %d decode steps per launch, mode %s, prefetch %d)\n",
         sm_count, num_workers, num_schedulers, sched_ctas, layers, full ? " + lm_head" : "", pos0, iters,
         mode.c_str(), prefetch);

  // ---- host model
  uint32_t seed = 91u;
  HostModel hm;
  hm.max_seq = max_seq;
  const float wsc = 1.f / std::sqrt(float(kHidden));
  hm.qkv = rand_vec(seed, size_t(kNumLayers) * kQkvDim * kHidden, wsc);
  hm.o = rand_vec(seed, size_t(kNumLayers) * kHidden * kHidden, wsc);
  hm.up = rand_vec(seed, size_t(kNumLayers) * kInter * kHidden, wsc);
  hm.gate = rand_vec(seed, size_t(kNumLayers) * kInter * kHidden, wsc);
  hm.down = rand_vec(seed, size_t(kNumLayers) * kHidden * kInter, 1.f / std::sqrt(float(kInter)));
  hm.lm_head = rand_vec(seed, size_t(kVocab) * kHidden, wsc);
  hm.attn_norm = rand_vec(seed, size_t(kNumLayers) * kHidden, 0.25f, 1.f);
  hm.mlp_norm = rand_vec(seed, size_t(kNumLayers) * kHidden, 0.25f, 1.f);
  hm.lm_norm = rand_vec(seed, kHidden, 0.25f, 1.f);
  hm.k_cache = rand_vec(seed, size_t(kNumLayers) * kNumKvHeads * max_seq * kHeadDim, 1.f);
  hm.v_cache = rand_vec(seed, size_t(kNumLayers) * kNumKvHeads * max_seq * kHeadDim, 1.f);
  hm.rope_cos.resize(size_t(max_seq) * kHeadDim);
  hm.rope_sin.resize(size_t(max_seq) * kHeadDim);
  for (int p = 0; p < max_seq; ++p) {
    for (int i = 0; i < kHeadDim / 2; ++i) {
      const double f = p * std::pow(double(kRopeBase), -2.0 * i / kHeadDim);
      hm.rope_cos[size_t(p) * kHeadDim + 2 * i] = hm.rope_cos[size_t(p) * kHeadDim + 2 * i + 1] = float(std::cos(f));
      hm.rope_sin[size_t(p) * kHeadDim + 2 * i] = hm.rope_sin[size_t(p) * kHeadDim + 2 * i + 1] = float(std::sin(f));
    }
  }
  std::vector<float> hidden0 = rand_vec(seed, kHidden, 1.f);

  // ---- device model
  Model m{};
  m.qkv_w = upload(hm.qkv); m.o_w = upload(hm.o); m.up_w = upload(hm.up); m.gate_w = upload(hm.gate);
  m.down_w = upload(hm.down); m.lm_head_w = upload(hm.lm_head);
  m.attn_norm = upload(hm.attn_norm); m.mlp_norm = upload(hm.mlp_norm); m.lm_norm = upload(hm.lm_norm);
  m.k_cache = upload(hm.k_cache); m.v_cache = upload(hm.v_cache);
  __nv_bfloat16* d_kv0[2] = {upload(hm.k_cache), upload(hm.v_cache)};  // pristine, restored per timed run
  float *d_cos = nullptr, *d_sin = nullptr;
  CK(cudaMalloc(&d_cos, hm.rope_cos.size() * 4)); CK(cudaMemcpy(d_cos, hm.rope_cos.data(), hm.rope_cos.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_sin, hm.rope_sin.size() * 4)); CK(cudaMemcpy(d_sin, hm.rope_sin.data(), hm.rope_sin.size() * 4, cudaMemcpyHostToDevice));
  m.rope_cos = d_cos; m.rope_sin = d_sin;
  m.hidden = upload(hidden0);
  __nv_bfloat16* d_hidden0 = upload(hidden0);
  CK(cudaMalloc((void**)&m.q, kHidden * 2));
  CK(cudaMalloc((void**)&m.attn_part, size_t(kNumKvHeads) * MK43_ATTN_SPLITS * kGqa * kHeadDim * 4));
  CK(cudaMalloc((void**)&m.attn_ml, size_t(kNumKvHeads) * MK43_ATTN_SPLITS * kGqa * 2 * 4));
  CK(cudaMalloc((void**)&m.attn_out, kHidden * 2));
  CK(cudaMalloc((void**)&m.silu_out, kInter * 2));
  CK(cudaMalloc((void**)&m.logits, size_t(kVocab) * 2));
  CK(cudaMemset(m.logits, 0, size_t(kVocab) * 2));
  m.max_seq = max_seq;
  m.attn_scale = 1.f / std::sqrt(float(kHeadDim));

  // ---- the task graph: one decode step
  GraphBuilder gb(num_workers);
  OpTasks prev;
  bool have_prev = false;
  const int keys_per_split = (max_seq + MK43_ATTN_SPLITS - 1) / MK43_ATTN_SPLITS;
  for (int L = 0; L < layers; ++L) {
    // qkv: 16 rows per task (8 warps x one RoPE pair), 24 tasks per KV head.
    OpTasks qkv = gb.add_op(partition_rows(gb, kTaskRmsNormQkvRope, L, kQkvDim, 16, 2),
                            have_prev ? std::vector<OpTasks>{prev} : std::vector<OpTasks>{}, 1);
    // attention: per KV head, MK43_ATTN_SPLITS key ranges; event per KV head.
    std::vector<TaskDesc> at;
    for (int kvh = 0; kvh < kNumKvHeads; ++kvh) {
      for (int sp = 0; sp < MK43_ATTN_SPLITS; ++sp) {
        at.push_back(gb.blank(kTaskAttentionSplitKv, kvh, sp * keys_per_split, (sp + 1) * keys_per_split, L, sp));
      }
    }
    OpTasks attn = gb.add_op(at, {qkv}, kNumKvHeads);
    std::vector<TaskDesc> mg;
    for (int kvh = 0; kvh < kNumKvHeads; ++kvh) mg.push_back(gb.blank(kTaskAttentionMerge, kvh, 0, 0));
    OpTasks merge = gb.add_op(mg, {attn}, kNumKvHeads);
    OpTasks oproj = gb.add_op(partition_rows(gb, kTaskLinearResidual, L, kHidden, 16, 1), {merge}, 1);
    OpTasks upgate = gb.add_op(partition_rows(gb, kTaskRmsNormUpgateSilu, L, kInter, 64, 1), {oproj}, 1);
    OpTasks down = gb.add_op(partition_rows(gb, kTaskDownResidual, L, kHidden, 16, 1), {upgate}, 1);
    prev = down; have_prev = true;
  }
  if (full) {
    prev = gb.add_op(partition_rows(gb, kTaskRmsNormLmHead, 0, kVocab, (kVocab + num_workers - 1) / num_workers, 8), {prev}, 1);
  }
  gb.finish(prev, mode == "aot");
  std::map<int, int> per_type;
  for (auto& t : gb.tasks) per_type[t.task_type]++;
  printf("  graph: %zu tasks, %zu events;", gb.tasks.size(), gb.events.size());
  for (auto& kv : per_type) if (kv.first >= 100) printf(" %s=%d", task_name(kv.first), kv.second);
  printf("\n");

  // ---- runtime state
  RuntimeConfig cfg{};
  cfg.num_workers = num_workers; cfg.num_schedulers = num_schedulers;
  cfg.num_events = int(gb.events.size()); cfg.num_tasks = int(gb.tasks.size());
  cfg.per_worker_queue_len = 8192; cfg.per_sched_queue_len = 4096;
  cfg.num_iterations = iters; cfg.prefetch_next = prefetch;
  CK(cudaMalloc(&cfg.worker_queue_last_ready, num_workers * 8));
  CK(cudaMalloc(&cfg.sched_queue_last_ready, (num_schedulers + 1) * 8));
  CK(cudaMalloc(&cfg.sched_queue_next_free, (num_schedulers + 1) * 8));
  CK(cudaMalloc(&cfg.event_counters, gb.events.size() * 8));
  CK(cudaMalloc(&cfg.event_num_triggers, gb.events.size() * 4));
  std::vector<int32_t> trig(gb.events.size());
  for (size_t i = 0; i < gb.events.size(); ++i) trig[i] = gb.events[i].num_triggers;
  CK(cudaMemcpy(cfg.event_num_triggers, trig.data(), trig.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&cfg.tasks, gb.tasks.size() * sizeof(TaskDesc)));
  CK(cudaMemcpy(cfg.tasks, gb.tasks.data(), gb.tasks.size() * sizeof(TaskDesc), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&cfg.events, gb.events.size() * sizeof(EventDesc)));
  CK(cudaMemcpy(cfg.events, gb.events.data(), gb.events.size() * sizeof(EventDesc), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&cfg.worker_queues, size_t(num_workers) * cfg.per_worker_queue_len * 8));
  CK(cudaMalloc(&cfg.sched_queues, size_t(num_schedulers + 1) * cfg.per_sched_queue_len * 8));
  CK(cudaMalloc(&cfg.step, 4));
  const size_t per_iter_tasks = (gb.tasks.size() + num_workers - 1) / num_workers + 2;
  if (per_iter_tasks * iters > cfg.per_worker_queue_len) { printf("worker queue too short for %d iterations\n", iters); return 1; }
  cfg.profile = nullptr; cfg.profile_entries = 0;
  if (MK43_PROFILE) {
    cfg.profile_entries = int(2 * per_iter_tasks * iters + 16);
    CK(cudaMalloc(&cfg.profile, size_t(num_workers) * cfg.profile_entries * 8));
    CK(cudaMemset(cfg.profile, 0, size_t(num_workers) * cfg.profile_entries * 8));
  }
  const int end_event = int(gb.events.size()) - 1;
  const int grid = num_workers + sched_ctas;
  // Unused dynamic shared memory, reserved so the hardware places exactly
  // one CTA per SM: the grid is the SM count and every CTA must be resident.
  constexpr int kResidencySmem = 200 * 1024;
  CK(cudaFuncSetAttribute(mpk_persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kResidencySmem));
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  // The first pos is written with a pageable H2D above only once; keep a device copy for the graph.
  int32_t* d_pos0 = nullptr;
  CK(cudaMalloc(&d_pos0, 4)); CK(cudaMemcpy(d_pos0, &pos0, 4, cudaMemcpyHostToDevice));
  auto run_graph_safe = [&](cudaStream_t s) {
    CK(cudaMemcpyAsync(m.hidden, d_hidden0, kHidden * 2, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(cfg.step, d_pos0, 4, cudaMemcpyDeviceToDevice, s));
    mpk_prepare_kernel<<<(std::max(num_workers, cfg.num_events) + 127) / 128, 128, 0, s>>>(cfg, end_event);
    mpk_persistent_kernel<<<grid, kWorkerThreads, kResidencySmem, s>>>(cfg, m);
  };

  // ---- correctness over `iters` decode steps
  printf("  reference (double, %d steps) ...\n", iters);
  fflush(stdout);
  std::vector<float> hidden = hidden0, logits_ref;
  for (int i = 0; i < iters; ++i) reference_step(hm, hidden, pos0 + i, layers, (full && i == iters - 1) ? &logits_ref : nullptr);
  run_graph_safe(stream);
  CK(cudaStreamSynchronize(stream));
  CK(cudaGetLastError());
  bool ok = true;
  {
    const Err e = compare(download(m.hidden, kHidden), hidden);
    // The residual stream after iters x layers bf16 layer applications: the
    // max gate scales with that depth (about 1 bf16 ulp per 40 applications).
    const float max_gate = 3e-2f + 2e-4f * float(iters * layers);
    const bool pass = e.max_norm < max_gate && e.mean_norm < 1e-2f;
    printf("  %-8s max err / rms %.3e  mean err / rms %.3e  %s\n", "hidden", e.max_norm, e.mean_norm, pass ? "PASS" : "FAIL");
    ok = ok && pass;
    if (full) {
      std::vector<float> lg = download(m.logits, kVocab);
      const Err e2 = compare(lg, logits_ref);
      const bool pass2 = e2.max_norm < 5e-2f && e2.mean_norm < 1e-2f;
      printf("  %-8s max err / rms %.3e  mean err / rms %.3e  %s\n", "logits", e2.max_norm, e2.mean_norm, pass2 ? "PASS" : "FAIL");
      const int a = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
      const int b = int(std::max_element(logits_ref.begin(), logits_ref.end()) - logits_ref.begin());
      printf("  argmax device %d reference %d %s\n", a, b, a == b ? "MATCH" : "DIFFER");
      ok = ok && pass2;
    }
  }
  if (!ok) { printf("numerics failed; timings withheld\n"); return 1; }

  // ---- timing: the whole launch, divided by decode steps
  double bytes = 0.0;
  for (int L = 0; L < layers; ++L) {
    bytes += double(kQkvDim + kHidden + 2 * kInter) * kHidden * 2 + double(kHidden) * kInter * 2;
    bytes += double(kNumKvHeads) * (pos0 + 1) * kHeadDim * 2 * 2;
  }
  if (full) bytes += double(kVocab) * kHidden * 2;
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CK(cudaMemcpyAsync(m.k_cache, d_kv0[0], hm.k_cache.size() * 2, cudaMemcpyDeviceToDevice, stream));
  CK(cudaMemcpyAsync(m.v_cache, d_kv0[1], hm.v_cache.size() * 2, cudaMemcpyDeviceToDevice, stream));
  run_graph_safe(stream);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  // Time the persistent kernel alone (the KV restore is not part of a step).
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < timing_reps; ++r) {
    CK(cudaMemcpyAsync(m.k_cache, d_kv0[0], hm.k_cache.size() * 2, cudaMemcpyDeviceToDevice, stream));
    CK(cudaMemcpyAsync(m.v_cache, d_kv0[1], hm.v_cache.size() * 2, cudaMemcpyDeviceToDevice, stream));
    CK(cudaMemcpyAsync(m.hidden, d_hidden0, kHidden * 2, cudaMemcpyDeviceToDevice, stream));
    CK(cudaMemcpyAsync(cfg.step, d_pos0, 4, cudaMemcpyDeviceToDevice, stream));
    mpk_prepare_kernel<<<(std::max(num_workers, cfg.num_events) + 127) / 128, 128, 0, stream>>>(cfg, end_event);
    CK(cudaEventRecord(a, stream));
    mpk_persistent_kernel<<<grid, kWorkerThreads, kResidencySmem, stream>>>(cfg, m);
    CK(cudaEventRecord(b, stream));
    CK(cudaEventSynchronize(b));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms / iters);
  }
  const double floor_ms = (1.85 + bytes / 2.77e6) / 1000.0;
  printf("\n  per decode step: %.3f ms  (%.2f GB -> %.2f TB/s; floor %.3f ms [ld.bw.dev.dram] -> %.0f%% of floor)\n",
         best, bytes / 1e9, bytes / best / 1e9, floor_ms, 100.0 * floor_ms / best);
  printf("  upstream: 12.5 ms vs a ~10 ms bound (80%%) for Qwen3-8B on A100.\n");

  if (MK43_PROFILE && cfg.profile) {
    std::vector<uint64_t> p(size_t(num_workers) * cfg.profile_entries);
    CK(cudaMemcpy(p.data(), cfg.profile, p.size() * 8, cudaMemcpyDeviceToHost));
    std::map<int, std::pair<double, int>> agg;
    double busy = 0.0;
    uint64_t t_min = ~0ull, t_max = 0;
    for (int w = 0; w < num_workers; ++w) {
      for (int i = 0; i + 1 < cfg.profile_entries; i += 2) {
        const uint64_t k = p[size_t(w) * cfg.profile_entries + i], v = p[size_t(w) * cfg.profile_entries + i + 1];
        if (k == 0) break;
        const int type = int(k >> 32);
        const uint64_t start = v >> 32, dur = v & 0xffffffffull;
        agg[type].first += dur; agg[type].second++;
        busy += dur;
        t_min = std::min(t_min, start); t_max = std::max(t_max, start + dur);
      }
    }
    printf("\n  task profile (%%globaltimer, last timed launch, %d steps): mean us per task\n", iters);
    for (auto& kv : agg) printf("    %-8s n=%6d  %8.2f us\n", task_name(kv.first), kv.second.second, kv.second.first / kv.second.second / 1000.0);
    const double span = double(t_max - t_min);
    printf("    worker busy fraction %.1f%% over a %.3f ms span (timer wraps at 4.29 s; ignore if negative)\n",
           100.0 * busy / (span * num_workers), span / 1e6);
  }
  return 0;
}

#endif  // MK43_NO_MAIN
