// Template 40 -- megakernel: one persistent interpreter over a task graph (sm90).
//
// The endgame of launch-cost removal.  A chain of small ops whose grid ramps sit
// inside their own self-time cannot be fixed by tuning any one of them; the
// answer is to stop launching. An offline planner emits a task table plus
// dependency counters -- SCHEDULE AS DATA -- and one persistent kernel walks it
// -- DISPATCH AS CODE.
//
// What actually makes this work, and what makes it fail:
//
//   1. Ordering is GLOBAL-MEMORY COUNTERS, never a cycle-count schedule.  A
//      task publishes by decrementing its successors' counters; a task starts
//      when its counter reaches zero.  Observers are free, so one counter can
//      gate many waiters [atom.lat.dev.hop].
//   2. The acquire/release pair IS the correctness argument.  The decrement is
//      a RELEASE so the task's writes precede it; the poll is an ACQUIRE so the
//      successor's reads follow it.  A plain atomicAdd and a plain load compile
//      and run and produce wrong answers under contention.
//   3. No cooperative launch.  A grid barrier costs ~1.09 us [coop.lat.dev.sync]
//      and caps occupancy; the counters already express the ordering, and they
//      express a DAG rather than a sequence of phases.
//   4. The register budget is the MAX OVER TASK KINDS, not the average.  One
//      heavy kind taxes every worker for the whole run.  This is the tax that
//      most often makes a megakernel lose to the launches it replaced.
//   5. The planner must emit TRUNCATED tables from day one.  A persistent
//      kernel with a wrong dependency hangs rather than fails, so `max_tasks`
//      is a debugging necessity, not a convenience: it makes the table
//      bisectable and gives parity a runnable prefix per task kind.
//
// WHY THIS ONE CANNOT DEADLOCK, AND WHEN A MEGAKERNEL CAN.
//
// A worker here claims a task and then blocks on its counter, which looks like
// the classic way to hang: fill every worker with unready tasks and nobody is
// left to make them ready.  It is safe only because of two things together --
// the table is topologically ordered and the claim cursor is monotonic.  Then
// the lowest-indexed task still in flight has all its predecessors already
// completed, so SOME worker can always advance.  Remove either property and
// that argument is gone.
//
// The argument also breaks the moment a BOUNDED RESOURCE enters: a ring of live
// blocks, a shared-memory pool, a fixed number of accumulator slots.  A worker
// holding a slot while waiting for a task that needs a slot is a resource
// deadlock, and topological order does not help.  This is the real hazard when
// a megakernel fuses two DEPENDENT STAGES -- an MoE layer's two linears, say --
// as interleaved task phases over one worker pool: the second stage's tasks can
// occupy the pool waiting on first-stage tasks that were never issued.
//
// DeepGEMM's MegaMoE scheduler answers it with a WARMUP WAVE: issue enough
// first-stage tasks before any second-stage task may be scheduled, sized from
// the per-block ratio of the two stages' task counts, plus one extra wave for
// partial-wave rounding.  The ring capacity gets a closed-form bound from how
// far the two stages can drift apart.  If a fused-layer megakernel is the goal,
// that sizing is the design, not a detail.
//
// Before believing a projected win, price the ledger both ways
// [fusion-economics, price-the-direction]: launches removed x
// [launch.lat.dev.ramp] against hops added x [atom.lat.dev.hop].  At one-op
// scope it usually comes out negative.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
//
// CHECK-PTX: atom\.add\.release\.gpu\.u32
// CHECK-PTX: ld\.acquire\.gpu\.u32
// CHECK-PTX: atom\.global\.add\.u32
// CHECK-PTX: st\.global

#include <cstdint>

#include "sm90_common.cuh"

namespace {

constexpr int kThreads = 256;
// Workers are sized to the machine, not to the graph: the table is walked, so
// more CTAs than tasks is waste and fewer is fine.
constexpr uint32_t kNumSMs = 132;

enum class TaskKind : int32_t { kElementwise = 0, kReduce = 1, kNumKinds = 2 };

// Release: everything this task wrote must precede the decrement that lets a
// successor run.  `atom.add` of -1 rather than `red` because the returning
// value is what tells this thread it fired the LAST decrement.
__device__ __forceinline__ uint32_t counter_dec_release(uint32_t* p) {
  uint32_t old;
  asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(p), "r"(0xffffffffu)
               : "memory");
  return old;
}

// Acquire: the successor's reads must not be hoisted above the observation that
// its dependencies are done.
__device__ __forceinline__ uint32_t counter_load_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

}  // namespace

// Emitted by the offline planner.  Flat and fixed-size on purpose: a worker
// reads one of these per task, so a variable-size descriptor would put a
// dependent load on the critical path of every dispatch.
struct Task {
  int32_t kind;
  int32_t counter_idx;    // this task's own readiness counter
  int32_t succ_begin;     // range into the successor-index array
  int32_t succ_count;
  int32_t in_offset;
  int32_t out_offset;
  int32_t elems;
  int32_t reserved;       // keeps the descriptor 32 B, one sector
};

__global__ __launch_bounds__(kThreads, 1) void megakernel_interpreter(
    const Task* __restrict__ tasks, const int32_t* __restrict__ successors,
    uint32_t* __restrict__ counters, uint32_t* __restrict__ next_task,
    const float* __restrict__ in, float* __restrict__ out,
    int32_t num_tasks,
    // The truncation hook.  A hang is the normal failure mode here, so the
    // ability to run a prefix of the table is what makes the thing debuggable.
    int32_t max_tasks) {
  __shared__ Task shared_task;
  __shared__ uint32_t claimed;

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t limit = (max_tasks > 0 && max_tasks < num_tasks) ? max_tasks : num_tasks;

  while (true) {
    // One CTA-wide claim per task: a per-thread atomic on a shared cursor would
    // be kThreads times the contention for the same answer [atom.rate.addr].
    if (tid == 0) { claimed = atomicAdd(next_task, 1u); }
    __syncthreads();
    const uint32_t my_task = claimed;
    if (my_task >= static_cast<uint32_t>(limit)) { break; }

    if (tid == 0) { shared_task = tasks[my_task]; }
    __syncthreads();
    const Task task = shared_task;

    // Wait for predecessors.  One thread polls: a whole CTA spinning on the
    // same line would serialize the line's traffic against every other waiter.
    if (tid == 0) {
      while (counter_load_acquire(&counters[task.counter_idx]) != 0u) {}
    }
    __syncthreads();

    // Dispatch as code.  Every kind's register use is charged to every worker,
    // so a kind that needs a big tile makes the whole interpreter expensive --
    // splitting the heavy kind into its own kernel is a legitimate answer.
    switch (static_cast<TaskKind>(task.kind)) {
      case TaskKind::kElementwise: {
        for (int32_t i = tid; i < task.elems; i += kThreads) {
          out[task.out_offset + i] = in[task.in_offset + i] * 2.f;
        }
        break;
      }
      case TaskKind::kReduce: {
        float acc = 0.f;
        for (int32_t i = tid; i < task.elems; i += kThreads) {
          acc += in[task.in_offset + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (tid % tmpl::kWarpThreads == 0) {
          atomicAdd(&out[task.out_offset], acc);
        }
        break;
      }
      default: break;
    }

    // Publish.  The barrier orders this CTA's stores before the release below;
    // the release orders them before any successor's acquire observes zero.
    __syncthreads();
    if (tid == 0) {
      for (int32_t s = 0; s < task.succ_count; ++s) {
        counter_dec_release(&counters[successors[task.succ_begin + s]]);
      }
    }
  }
}

// The planner's side of the contract, stated here because the kernel cannot
// check it: counters[i] must start at the in-degree of task i, successor ranges
// must be disjoint, and the table MUST be topologically ordered.  That last one
// is not stylistic -- it is the whole progress argument above, and violating it
// hangs the kernel with no error and no output.
cudaError_t configure_megakernel() {
  return cudaFuncSetAttribute(megakernel_interpreter,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
}
