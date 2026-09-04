// Template 40 -- a megakernel VM: one persistent kernel interpreting a program.
//
// This is the longest template in the directory on purpose. A megakernel is not
// a kernel with a switch in it; it is a small virtual machine, and everything
// hard about it is machinery that a sketch leaves out. The architecture below
// follows HazyResearch/Megakernels (MIT), reduced to the toolkit so the
// mechanism is visible without a tile library in the way.
//
// THE SHAPE
//
//     an offline planner emits a PROGRAM        -- schedule as data
//     one persistent kernel INTERPRETS it       -- dispatch as code
//
// and the interpreter is itself warp-specialized, which is the part that
// surprises people. A megakernel is not a flat loop over tasks: it is a
// pipelined machine with five roles, and each instruction flows through them.
//
//     controller  fetches instructions, allocates pages, arms semaphores
//     loader      global -> shared for the current instruction
//     consumer    the math, on 16 warps
//     storer      shared -> global
//     launcher    issues the async work that must come from one warp
//
// FIVE THINGS THAT MAKE IT WORK
//
//  1. A FIXED-WIDTH INSTRUCTION. 128 bytes, 32 ints, streamed from global
//     memory like any other tensor. Fixed width is what lets the fetch be
//     pipelined and vectorized; a variable-length descriptor would put a
//     dependent load on the critical path of every dispatch.
//  2. AN INSTRUCTION RING. Instruction i+1 is fetched and its resources are
//     arranged while instruction i executes -- the ring of template 01, with
//     instructions as the payload instead of tiles. `arrived` and `finished`
//     are its full/empty pair.
//  3. SHARED MEMORY AS PAGES, RECYCLED IN AN OP-DECLARED ORDER. This is the
//     mechanism most sketches omit and the one that makes the pipeline real.
//     Each op declares which page it releases next (`release_page`), so the
//     next instruction -- of a DIFFERENT KIND, with a different footprint --
//     can start filling pages the previous one has finished with. Without it,
//     every instruction boundary is a __syncthreads() and the machine degrades
//     into a sequence of small kernels with extra steps.
//  4. PER-INSTRUCTION SEMAPHORES, ARMED BY THE OP. The VM owns a fixed pool;
//     each op says how many it needs and with what arrival counts. Ops get
//     internal pipelining without the VM knowing anything about their dataflow.
//  5. A BUILT-IN PROFILER. Per-instruction timing slots, off by default. This
//     is not a nicety: a profiler cannot attribute time inside a megakernel --
//     every op is the same kernel -- so if the VM does not record it, nobody
//     can see which instruction is slow.
//
// WHY THIS ONE CANNOT DEADLOCK, AND WHEN A MEGAKERNEL CAN
//
// Workers claim instructions monotonically from a topologically ordered
// program, so the lowest-indexed instruction still in flight always has its
// predecessors complete and SOME worker can advance. Remove either property and
// the argument is gone.
//
// It also breaks the moment a BOUNDED RESOURCE enters -- and this VM has one:
// pages. A role holding a page while waiting for an instruction that needs a
// page is a resource deadlock, which is why the page order is declared per op
// and rotated by the controller rather than being grabbed on demand. The same
// hazard is what makes fusing two DEPENDENT STAGES over one worker pool hard:
// the second stage's instructions can occupy the machine waiting on
// first-stage instructions nobody has issued. DeepGEMM's MegaMoE answers it
// with a warmup wave sized from the per-block ratio of the two stages, plus a
// closed-form ring bound. That sizing is the design, not a detail.
//
// THE LEDGER, BEFORE ANY OF THIS
//
// launches removed x [launch.lat.dev.ramp] against hops added x
// [atom.lat.dev.hop], plus the register tax below. At one-op scope it comes out
// negative [fusion-economics, price-the-direction].
//
// THE REGISTER TAX IS THE USUAL KILLER. Consumer warps are sized once, for the
// MAX over every op the VM can run. One op that wants a big accumulator makes
// every instruction in the program expensive. Splitting that op back out into
// its own kernel is a legitimate answer and is often the right one.
//
// Structural only; see 01 for what the PTX assertions do and do not prove.
// A VM is also the worst possible thing to debug by inspection: build the
// truncation hook (`max_instructions`) on day one, because a wrong dependency
// hangs with no output and bisecting the program is the only way in.
//
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: mbarrier\.init\.shared::cta\.b64
// CHECK-PTX: mbarrier\.try_wait\.parity\.shared::cta\.b64
// CHECK-PTX: mbarrier\.arrive\.shared::cta\.b64
// CHECK-PTX: nanosleep\.u32
// CHECK-PTX: ld\.global(\.nc)?\.v4
// CHECK-PTX: atom\.add\.release\.gpu\.u32

#include <cstdint>

#include "sm90_common.cuh"

namespace {

// ---------------------------------------------------------------- VM config

struct VmConfig {
  // 32 ints = 128 B. One instruction is two 64-B sectors, fetched as vectors.
  static constexpr int kInstructionWidth = 32;
  // Two stages: fetch i+1 while i runs. Deeper buys nothing, because the fetch
  // is a single 128-B load and the arranging is a few semaphore writes.
  static constexpr int kPipelineStages = 2;
  // Pool the VM owns; ops draw from it and declare their own arrival counts.
  static constexpr int kDynamicSemaphores = 32;
  // Per-instruction timing slots. Off by default -- when on, this is the only
  // way to see which instruction in a megakernel is slow.
  static constexpr int kTimingWidth = 32;
  static constexpr bool kTimingEnabled = false;

  static constexpr int kConsumerWarps = 16;
  static constexpr int kNonConsumerWarps = 4;  // loader, storer, launcher, controller
  static constexpr int kNumWarps = kConsumerWarps + kNonConsumerWarps;
  static constexpr int kNumThreads = kNumWarps * 32;

  // Consumers get the registers the non-consumers release. Both numbers are
  // the MAX over every op the VM can run, which is the tax the header warns of.
  static constexpr int kConsumerRegisters = 104;
  static constexpr int kNonConsumerRegisters = 64;

  static constexpr int kPageSize = 16384;
  static constexpr int kNumPages = 12;
  static constexpr int kScratchBytes = 1024;

  // Spin loops back off rather than hammering the line; a megakernel has many
  // waiters on few addresses.
  static constexpr uint32_t kSpinSleepNanos = 20;
};

// Opcode 0 is always NoOp so a zero-filled instruction is legal. That makes a
// program paddable to any length and makes truncation a matter of writing
// zeros rather than recomputing the table.
enum Opcode : int32_t { kOpNoOp = 0, kOpElementwise = 1, kOpReduce = 2 };

// Instruction layout. Word 0 is the opcode; the rest is the op's business.
// Named accessors rather than raw indices, because a program is emitted by a
// planner in another language and the two must agree exactly.
struct InstructionView {
  const int32_t* w;
  __device__ __forceinline__ int32_t opcode() const { return w[0]; }
  __device__ __forceinline__ int32_t in_offset() const { return w[1]; }
  __device__ __forceinline__ int32_t out_offset() const { return w[2]; }
  __device__ __forceinline__ int32_t elems() const { return w[3]; }
  __device__ __forceinline__ int32_t page_count() const { return w[4]; }
  __device__ __forceinline__ int32_t counter_idx() const { return w[5]; }
  __device__ __forceinline__ int32_t succ_begin() const { return w[6]; }
  __device__ __forceinline__ int32_t succ_count() const { return w[7]; }
};

// Per-stage control block. One of these per pipeline stage, all in static
// shared memory; the pages live in dynamic shared memory separately.
struct alignas(128) StageState {
  int32_t instruction[VmConfig::kInstructionWidth];
  int32_t timings[VmConfig::kTimingWidth];
  // Order in which this instruction hands its pages on. Written by the
  // controller from the PREVIOUS instruction's release_page(), which is what
  // lets pages cross an instruction boundary without a barrier.
  int32_t page_order[VmConfig::kNumPages];
};

__device__ __forceinline__ void spin_backoff() {
  asm volatile("nanosleep.u32 %0;" ::"n"(VmConfig::kSpinSleepNanos));
}

// Release: an instruction's writes must precede the decrement that frees a
// successor. The returned value tells this thread whether it fired the last one.
__device__ __forceinline__ uint32_t counter_dec_release(uint32_t* p) {
  uint32_t old;
  asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(p), "r"(0xffffffffu) : "memory");
  return old;
}

// Acquire: the successor's reads must not be hoisted above observing zero.
__device__ __forceinline__ uint32_t counter_load_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t stage_phase(int32_t index) {
  return static_cast<uint32_t>(index / VmConfig::kPipelineStages) & 1u;
}

}  // namespace

// The VM's whole shared state, gathered so every role takes one reference
// rather than a dozen pointers.
struct VmState {
  StageState* stage;              // [kPipelineStages]
  uint64_t* instr_arrived;        // [kPipelineStages]
  uint64_t* instr_finished;       // [kPipelineStages]
  uint64_t* page_free;            // [kNumPages]
  uint64_t* op_semaphores;        // [kPipelineStages][kDynamicSemaphores]
  uint8_t* pages;                 // dynamic smem, kNumPages * kPageSize
  int32_t* scratch;

  __device__ __forceinline__ uint8_t* page(int32_t pid) const {
    return pages + static_cast<int64_t>(pid) * VmConfig::kPageSize;
  }
  __device__ __forceinline__ uint64_t* sem(int32_t ring, int32_t i) const {
    return op_semaphores + ring * VmConfig::kDynamicSemaphores + i;
  }
};

// ---------------------------------------------------------------- op interface
//
// Every op implements the same five entry points. The VM never learns what an
// op does; it only knows that these exist, which is what keeps the machine's
// cost independent of the number of ops (except for the register max).
//
//   arm_semaphores  how many semaphores, with what arrival counts
//   release_page    which page this op hands on next -- the page order
//   loader          global -> shared
//   consumer        the math
//   storer          shared -> global

namespace op_elementwise {

__device__ __forceinline__ void arm_semaphores(const VmState& vm, int32_t ring) {
  // One semaphore: the loader signals, the consumers wait.
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);
}

// Pages are handed on in order; an op with a different footprint would
// reorder here so its first-touched page is freed first.
__device__ __forceinline__ int32_t release_page(int32_t lane) { return lane; }

__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView ins,
                                       const float* __restrict__ in) {
  float* dst = reinterpret_cast<float*>(vm.page(vm.stage[ring].page_order[0]));
  const int32_t n = ins.elems();
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < n; i += 32) {
    dst[i] = in[ins.in_offset() + i];
  }
  __threadfence_block();
  if (tmpl::elect_one()) { tmpl::mbarrier_arrive(vm.sem(ring, 0)); }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, int32_t warp) {
  // arm_semaphores re-inits this every instruction, so the phase is always 0.
  tmpl::wait_parity(vm.sem(ring, 0), 0);
  float* buf = reinterpret_cast<float*>(vm.page(vm.stage[ring].page_order[0]));
  const int32_t n = ins.elems();
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  for (int32_t i = warp * 32 + lane; i < n; i += VmConfig::kConsumerWarps * 32) {
    buf[i] = buf[i] * 2.f;
  }
}

__device__ __forceinline__ void storer(const VmState& vm, int32_t ring,
                                       InstructionView ins,
                                       float* __restrict__ out) {
  const float* src =
      reinterpret_cast<const float*>(vm.page(vm.stage[ring].page_order[0]));
  const int32_t n = ins.elems();
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < n; i += 32) {
    out[ins.out_offset() + i] = src[i];
  }
}

}  // namespace op_elementwise

namespace op_reduce {

__device__ __forceinline__ void arm_semaphores(const VmState& vm, int32_t ring) {
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);
}

// This op touches its pages back to front, so it releases them that way.
__device__ __forceinline__ int32_t release_page(int32_t lane) {
  return VmConfig::kNumPages - 1 - lane;
}

__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView ins,
                                       const float* __restrict__ in) {
  float* dst = reinterpret_cast<float*>(vm.page(vm.stage[ring].page_order[0]));
  const int32_t n = ins.elems();
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < n; i += 32) {
    dst[i] = in[ins.in_offset() + i];
  }
  __threadfence_block();
  if (tmpl::elect_one()) { tmpl::mbarrier_arrive(vm.sem(ring, 0)); }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, int32_t warp) {
  // arm_semaphores re-inits this every instruction, so the phase is always 0.
  tmpl::wait_parity(vm.sem(ring, 0), 0);
  const float* buf =
      reinterpret_cast<const float*>(vm.page(vm.stage[ring].page_order[0]));
  const int32_t n = ins.elems();
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  float acc = 0.f;
  for (int32_t i = warp * 32 + lane; i < n; i += VmConfig::kConsumerWarps * 32) {
    acc += buf[i];
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) { atomicAdd(&vm.scratch[warp], __float_as_int(acc)); }
}

__device__ __forceinline__ void storer(const VmState& vm, int32_t ring,
                                       InstructionView ins,
                                       float* __restrict__ out) {
  if (tmpl::elect_one()) {
    float total = 0.f;
    for (int32_t w = 0; w < VmConfig::kConsumerWarps; ++w) {
      total += __int_as_float(vm.scratch[w]);
    }
    out[ins.out_offset()] = total;
  }
}

}  // namespace op_reduce

// ------------------------------------------------------------------ the VM

__global__ __launch_bounds__(VmConfig::kNumThreads, 1) void megakernel_vm(
    const int32_t* __restrict__ program,      // (num_instructions, 32)
    const int32_t* __restrict__ successors,
    uint32_t* __restrict__ counters,
    uint32_t* __restrict__ next_instruction,
    const float* __restrict__ in, float* __restrict__ out,
    int32_t num_instructions,
    // Truncation hook. Build it on day one: a wrong dependency hangs with no
    // output, and running a prefix is the only way to bisect a program.
    int32_t max_instructions) {
  __shared__ alignas(128) StageState stage[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_arrived[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_finished[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t page_free[VmConfig::kNumPages];
  __shared__ alignas(8) uint64_t
      op_semaphores[VmConfig::kPipelineStages * VmConfig::kDynamicSemaphores];
  __shared__ alignas(16) int32_t scratch[VmConfig::kScratchBytes / 4];
  __shared__ uint32_t claimed[VmConfig::kPipelineStages];
  extern __shared__ __align__(1024) uint8_t pages_raw[];

  VmState vm{stage,        instr_arrived, instr_finished, page_free,
             op_semaphores, pages_raw,     scratch};

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / 32;

  if (tid < VmConfig::kPipelineStages) {
    // The controller alone announces arrival; every other warp reports done.
    tmpl::mbarrier_init(&instr_arrived[tid], 1);
    tmpl::mbarrier_init(&instr_finished[tid], VmConfig::kNumWarps - 1);
  }
  if (tid < VmConfig::kNumPages) {
    // Pages start FREE, so the first instruction does not wait on a release
    // that never happened.
    tmpl::mbarrier_init(&page_free[tid], 1);
    tmpl::mbarrier_arrive(&page_free[tid]);
  }
  if (tid < VmConfig::kConsumerWarps) { scratch[tid] = 0; }
  tmpl::fence_barrier_init();
  __syncthreads();

  const int32_t limit =
      (max_instructions > 0 && max_instructions < num_instructions)
          ? max_instructions : num_instructions;

  // ------------------------------------------------------------ controller
  if (warp == VmConfig::kConsumerWarps + 3) {
    tmpl::setmaxnreg_dec<VmConfig::kNonConsumerRegisters>();
    for (int32_t idx = 0;; ++idx) {
      const int32_t ring = idx % VmConfig::kPipelineStages;

      // Reclaim the stage two instructions back before overwriting it.
      if (idx >= VmConfig::kPipelineStages) {
        tmpl::wait_parity(&instr_finished[ring], stage_phase(idx) ^ 1u);
      }

      uint32_t claim = 0;
      if (tmpl::elect_one()) { claim = atomicAdd(next_instruction, 1u); }
      claim = __shfl_sync(0xffffffffu, claim, 0);
      if (claim >= static_cast<uint32_t>(limit)) {
        // Publish a NoOp so the other roles retire in lockstep rather than
        // being left waiting on a stage that never arrives.
        if (tmpl::elect_one()) {
          stage[ring].instruction[0] = kOpNoOp;
          tmpl::mbarrier_arrive(&instr_arrived[ring]);
        }
        break;
      }

      // Fetch: 128 B, vectorized, one instruction per lane group.
      const int4* src = reinterpret_cast<const int4*>(
          program + static_cast<int64_t>(claim) * VmConfig::kInstructionWidth);
      int4* dst = reinterpret_cast<int4*>(stage[ring].instruction);
      const int32_t lane = tid % 32;
      if (lane < VmConfig::kInstructionWidth / 4) { dst[lane] = src[lane]; }
      __syncwarp();

      const InstructionView ins{stage[ring].instruction};

      // Wait for this instruction's dependencies before anyone touches data.
      if (tmpl::elect_one()) {
        while (counter_load_acquire(&counters[ins.counter_idx()]) != 0u) {
          spin_backoff();
        }
      }

      // Arrange resources: the page order comes from the op, so the next
      // instruction can begin on pages this one has finished with.
      if (lane < VmConfig::kNumPages) {
        int32_t pid = lane;
        switch (ins.opcode()) {
          case kOpElementwise: pid = op_elementwise::release_page(lane); break;
          case kOpReduce:      pid = op_reduce::release_page(lane); break;
          default: break;
        }
        stage[ring].page_order[lane] = pid;
      }
      if (tmpl::elect_one()) {
        switch (ins.opcode()) {
          case kOpElementwise: op_elementwise::arm_semaphores(vm, ring); break;
          case kOpReduce:      op_reduce::arm_semaphores(vm, ring); break;
          default: break;
        }
        if (VmConfig::kTimingEnabled) {
          stage[ring].timings[0] = static_cast<int32_t>(clock64());
        }
        claimed[ring] = claim;
      }
      __syncwarp();
      tmpl::fence_proxy_async_shared();
      if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&instr_arrived[ring]); }
    }
    return;
  }

  // ------------------------------------------- loader / storer / launcher
  if (warp >= VmConfig::kConsumerWarps) {
    tmpl::setmaxnreg_dec<VmConfig::kNonConsumerRegisters>();
    const int32_t role = warp - VmConfig::kConsumerWarps;  // 0 load, 1 store, 2 launch
    for (int32_t idx = 0;; ++idx) {
      const int32_t ring = idx % VmConfig::kPipelineStages;
      tmpl::wait_parity(&instr_arrived[ring], stage_phase(idx));
      const InstructionView ins{stage[ring].instruction};
      if (ins.opcode() == kOpNoOp) { break; }

      if (role == 0) {
        switch (ins.opcode()) {
          case kOpElementwise: op_elementwise::loader(vm, ring, ins, in); break;
          case kOpReduce:      op_reduce::loader(vm, ring, ins, in); break;
          default: break;
        }
      } else if (role == 1) {
        switch (ins.opcode()) {
          case kOpElementwise: op_elementwise::storer(vm, ring, ins, out); break;
          case kOpReduce:      op_reduce::storer(vm, ring, ins, out); break;
          default: break;
        }
      }
      // role 2 (launcher) issues async work that must come from one warp; the
      // ops here have none, but the role stays so the warp count and the
      // register split do not change when an op acquires some.

      if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&instr_finished[ring]); }
    }
    return;
  }

  // -------------------------------------------------------------- consumer
  tmpl::setmaxnreg_inc<VmConfig::kConsumerRegisters>();
  for (int32_t idx = 0;; ++idx) {
    const int32_t ring = idx % VmConfig::kPipelineStages;
    tmpl::wait_parity(&instr_arrived[ring], stage_phase(idx));
    const InstructionView ins{stage[ring].instruction};
    if (ins.opcode() == kOpNoOp) { break; }

    switch (ins.opcode()) {
      case kOpElementwise: op_elementwise::consumer(vm, ring, ins, warp); break;
      case kOpReduce:      op_reduce::consumer(vm, ring, ins, warp); break;
      default: break;
    }

    // Publish to the program's successors. One warp does it, after the
    // instruction's own stores are ordered by the barrier below.
    __threadfence();
    if (warp == 0 && tmpl::elect_one()) {
      for (int32_t s = 0; s < ins.succ_count(); ++s) {
        counter_dec_release(&counters[successors[ins.succ_begin() + s]]);
      }
    }
    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&instr_finished[ring]); }
  }
}

// The planner owns what the kernel cannot check: counters[i] starts at the
// in-degree of instruction i, successor ranges are disjoint, the program is
// TOPOLOGICALLY ORDERED (that is the whole progress argument), and every
// instruction's page_count fits kNumPages. Violate any one and the VM hangs
// with no error and no output.
cudaError_t configure_megakernel_vm() {
  return cudaFuncSetAttribute(megakernel_vm,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              VmConfig::kNumPages * VmConfig::kPageSize);
}

static_assert(VmConfig::kNumPages * VmConfig::kPageSize <= 227 * 1024,
              "page pool exceeds the H100 per-CTA shared limit");
static_assert(VmConfig::kInstructionWidth % 4 == 0,
              "instruction must be a whole number of 16-byte vectors");
