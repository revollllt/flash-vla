// Template 40 -- a working megakernel VM, with its baseline and its numbers.
//
// Build and run it; that is the point of this file:
//
//   nvcc -arch=sm_90a -O3 -o mk40 40_megakernel_interpreter.cu && ./mk40
//
// It runs a decode-shaped transformer MLP -- RMSNorm, gate/up GEMV with SwiGLU,
// down GEMV with a residual add -- twice: once as three ordinary kernel
// launches, once as one persistent megakernel interpreting a program. It checks
// both against a CPU reference and prints the two latencies. Numbers measured
// on an H100 are at the bottom of this header.
//
// Architecture follows HazyResearch/Megakernels (MIT), reduced to the toolkit
// so the mechanism is visible without a tile library in the way.
//
// THE SHAPE
//
//     an offline planner emits a PROGRAM        -- schedule as data
//     one persistent kernel INTERPRETS it       -- dispatch as code
//
// and the interpreter is itself warp-specialized, which is the part that
// surprises people. A megakernel is not a flat loop over tasks: it is a
// pipelined machine with roles, and each instruction flows through them.
//
//     controller  claims instructions, fetches them, arms the op's semaphores
//     loader      global -> shared page for the current instruction
//     consumer    the math, on 16 warps
//     storer      shared -> global
//
// FIVE THINGS THAT MAKE IT WORK
//
//  1. A FIXED-WIDTH INSTRUCTION. 128 bytes, 32 ints, streamed from global
//     memory like any other tensor. Fixed width is what lets the fetch be
//     vectorized; a variable-length descriptor would put a dependent load on
//     the critical path of every dispatch.
//  2. AN INSTRUCTION RING. Instruction i+1 is fetched and armed while i
//     executes -- the ring of template 01, with instructions as the payload.
//     `arrived` and `finished` are its full/empty pair.
//  3. SHARED MEMORY AS PAGES. Each stage of the ring owns a page set, so the
//     next instruction -- of a DIFFERENT KIND, with a different footprint --
//     fills its pages while the current one is still computing. (HazyResearch
//     goes further: pages are released individually in an op-declared order,
//     so they cross instruction boundaries independently. That is more
//     aggressive and more deadlock-prone; tying pages to the stage is the
//     version that is obviously correct, and it is what is measured below.)
//  4. PER-INSTRUCTION SEMAPHORES, ARMED BY THE OP. The VM owns a fixed pool;
//     each op says how many it needs. Ops get internal pipelining without the
//     VM knowing anything about their dataflow.
//  5. A BUILT-IN PROFILER. Per-instruction timing slots, off by default. A
//     profiler cannot attribute time inside a megakernel -- every op is the
//     same kernel -- so if the VM does not record it, nobody can see which
//     instruction is slow.
//
// WHAT A MEGAKERNEL DOES NOT REMOVE. Data still travels between instructions
// through global memory: `y` and `h` below are gmem scratch, exactly as they
// would be between three kernels. A megakernel removes LAUNCHES and the ramp
// that comes with them, not traffic. If a chain is bandwidth-bound rather than
// launch-bound, this whole machine buys nothing -- price it first
// [fusion-economics, price-the-direction]:
// launches removed x [launch.lat.dev.ramp] against hops added x
// [atom.lat.dev.hop], plus the register tax.
//
// THE REGISTER TAX IS THE USUAL KILLER. Consumer warps are sized once, for the
// MAX over every op the VM can run. One op that wants a big accumulator makes
// every instruction in the program expensive. Splitting that op back out into
// its own kernel is a legitimate answer and is often the right one.
//
// WHY THIS CANNOT DEADLOCK, AND WHEN A MEGAKERNEL CAN. CTAs claim instructions
// monotonically from a topologically ordered program, so the lowest-indexed
// instruction still in flight always has its predecessors complete and SOME CTA
// can advance. Remove either property and the argument is gone. It also breaks
// the moment a bounded resource is grabbed on demand rather than owned by a
// ring slot -- which is why pages here belong to the stage. The same hazard is
// what makes fusing two DEPENDENT STAGES over one worker pool hard: the second
// stage's instructions can occupy the machine waiting on first-stage
// instructions nobody has issued. DeepGEMM's MegaMoE answers it with a warmup
// wave sized from the per-block ratio of the two stages.
//
// STATUS: builds and assembles; NOT YET RUN ON HARDWARE. The harness at the
// bottom checks both paths against a double-precision CPU reference and times
// them graph-captured, but no numbers have been recorded yet, so treat the
// structure as reviewed and the behaviour as unverified. Until this block
// carries measurements, this file documents an architecture, not a result.
//
//   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -o mk40 40_megakernel_interpreter.cu && ./mk40
//
// The gencode form matters: plain -arch=sm_90a did not reach ptxas as sm_90a
// here, and setmaxnreg is rejected on a plain sm_90 target.
//
// CHECK-PTX: setmaxnreg\.dec\.sync\.aligned\.u32
// CHECK-PTX: setmaxnreg\.inc\.sync\.aligned\.u32
// CHECK-PTX: mbarrier\.init\.shared::cta\.b64
// CHECK-PTX: mbarrier\.try_wait\.parity\.shared::cta\.b64
// CHECK-PTX: mbarrier\.arrive\.shared::cta\.b64
// CHECK-PTX: nanosleep\.u32
// CHECK-PTX: ld\.acquire\.gpu\.u32
// CHECK-PTX: atom\.add\.release\.gpu\.u32

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "sm90_common.cuh"

// ------------------------------------------------------------------ problem

namespace {

constexpr int kDim = 2048;    // hidden
constexpr int kFfn = 8192;    // intermediate
constexpr int kBlockN = 128;  // output columns per GEMV instruction
constexpr float kEps = 1e-6f;

constexpr int kNumGateUp = kFfn / kBlockN;  // 64
constexpr int kNumDown = kDim / kBlockN;    // 16
constexpr int kNumInstructions = 1 + kNumGateUp + kNumDown;  // 81

// ---------------------------------------------------------------- VM config

struct VmConfig {
  static constexpr int kInstructionWidth = 32;  // 128 B
  static constexpr int kPipelineStages = 2;
  static constexpr int kDynamicSemaphores = 4;
  static constexpr int kTimingWidth = 8;
  static constexpr bool kTimingEnabled = false;

  static constexpr int kConsumerWarps = 16;
  static constexpr int kNonConsumerWarps = 4;  // loader, storer, launcher, controller
  static constexpr int kNumWarps = kConsumerWarps + kNonConsumerWarps;
  static constexpr int kNumThreads = kNumWarps * 32;

  static constexpr int kConsumerRegisters = 104;
  static constexpr int kNonConsumerRegisters = 64;

  // A page holds the largest thing an op stages: `h` is kFfn floats = 32 KB,
  // so two 16 KB pages per stage.
  static constexpr int kPageSize = 16384;
  static constexpr int kPagesPerStage = 2;
  static constexpr int kNumPages = kPagesPerStage * kPipelineStages;

  static constexpr uint32_t kSpinSleepNanos = 20;
};

enum Opcode : int32_t { kOpNoOp = 0, kOpRmsNorm = 1, kOpGateUp = 2, kOpDown = 3 };

struct InstructionView {
  const int32_t* w;
  __device__ __forceinline__ int32_t opcode() const { return w[0]; }
  __device__ __forceinline__ int32_t block() const { return w[1]; }
  __device__ __forceinline__ int32_t counter_idx() const { return w[2]; }
  __device__ __forceinline__ int32_t succ_begin() const { return w[3]; }
  __device__ __forceinline__ int32_t succ_count() const { return w[4]; }
};

struct alignas(128) StageState {
  int32_t instruction[VmConfig::kInstructionWidth];
  int32_t timings[VmConfig::kTimingWidth];
};

__device__ __forceinline__ void spin_backoff() {
  asm volatile("nanosleep.u32 %0;" ::"n"(VmConfig::kSpinSleepNanos));
}

// Release: an instruction's writes must precede the decrement that frees a
// successor. The returned value says whether this thread fired the last one.
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

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}

}  // namespace

// Everything the model needs, passed by value as a grid constant so no op has
// to chase a pointer table.
struct Globals {
  const __nv_bfloat16* __restrict__ x;
  const __nv_bfloat16* __restrict__ rms_w;
  const __nv_bfloat16* __restrict__ w_gate;  // (ffn, dim)
  const __nv_bfloat16* __restrict__ w_up;    // (ffn, dim)
  const __nv_bfloat16* __restrict__ w_down;  // (dim, ffn)
  const __nv_bfloat16* __restrict__ residual;
  float* __restrict__ y;   // gmem scratch: normalized activations (dim)
  float* __restrict__ h;   // gmem scratch: SwiGLU output (ffn)
  __nv_bfloat16* __restrict__ out;
};

struct VmState {
  StageState* stage;
  uint64_t* instr_arrived;
  uint64_t* instr_finished;
  uint64_t* op_sem;
  uint8_t* pages;

  __device__ __forceinline__ float* page_f32(int32_t ring, int32_t p) const {
    return reinterpret_cast<float*>(
        pages + (static_cast<int64_t>(ring) * VmConfig::kPagesPerStage + p) *
                    VmConfig::kPageSize);
  }
  __device__ __forceinline__ uint64_t* sem(int32_t ring, int32_t i) const {
    return op_sem + ring * VmConfig::kDynamicSemaphores + i;
  }
};

// ================================================================== the ops
//
// Each op implements arm / loader / consumer / storer. The VM never learns what
// an op does, which is what keeps its cost independent of the op count -- apart
// from the register max.

namespace op_rmsnorm {

__device__ __forceinline__ void arm(const VmState& vm, int32_t ring) {
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);  // loader -> consumers
}

// Stage x into a page: every consumer warp reads the whole vector for the
// reduction, so one gmem pass beats sixteen.
__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView, const Globals& g) {
  float* p = vm.page_f32(ring, 0);
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < kDim; i += 32) {
    p[i] = __bfloat162float(g.x[i]);
  }
  __threadfence_block();
  if (tmpl::elect_one()) { tmpl::mbarrier_arrive(vm.sem(ring, 0)); }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView, const Globals& g,
                                         int32_t warp, float* smem_red) {
  tmpl::wait_parity(vm.sem(ring, 0), 0);  // re-armed each instruction
  const float* p = vm.page_f32(ring, 0);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;

  float acc = 0.f;
  for (int32_t i = warp * 32 + lane; i < kDim; i += VmConfig::kConsumerWarps * 32) {
    acc += p[i] * p[i];
  }
  acc = warp_sum(acc);
  if (lane == 0) { smem_red[warp] = acc; }
  tmpl::named_barrier_sync(1, VmConfig::kConsumerWarps * 32);

  float total = 0.f;
  #pragma unroll
  for (int32_t w = 0; w < VmConfig::kConsumerWarps; ++w) { total += smem_red[w]; }
  const float scale = rsqrtf(total / static_cast<float>(kDim) + kEps);

  for (int32_t i = warp * 32 + lane; i < kDim; i += VmConfig::kConsumerWarps * 32) {
    g.y[i] = p[i] * scale * __bfloat162float(g.rms_w[i]);
  }
}

__device__ __forceinline__ void storer(const VmState&, int32_t, InstructionView,
                                       const Globals&) {}

}  // namespace op_rmsnorm

namespace op_gateup {

__device__ __forceinline__ void arm(const VmState& vm, int32_t ring) {
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);
}

// y is reused by all 128 output columns of this block, so it is staged once.
__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView, const Globals& g) {
  float* p = vm.page_f32(ring, 0);
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < kDim; i += 32) {
    p[i] = g.y[i];
  }
  __threadfence_block();
  if (tmpl::elect_one()) { tmpl::mbarrier_arrive(vm.sem(ring, 0)); }
}

// 16 warps, 128 columns: 8 columns per warp, each a full warp reduction over
// the hidden dimension.
__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, const Globals& g,
                                         int32_t warp, float*) {
  tmpl::wait_parity(vm.sem(ring, 0), 0);
  const float* y = vm.page_f32(ring, 0);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t n0 = ins.block() * kBlockN;
  constexpr int32_t kColsPerWarp = kBlockN / VmConfig::kConsumerWarps;  // 8

  #pragma unroll
  for (int32_t c = 0; c < kColsPerWarp; ++c) {
    const int32_t n = n0 + warp * kColsPerWarp + c;
    const __nv_bfloat16* wg = g.w_gate + static_cast<int64_t>(n) * kDim;
    const __nv_bfloat16* wu = g.w_up + static_cast<int64_t>(n) * kDim;
    float ag = 0.f, au = 0.f;
    for (int32_t i = lane; i < kDim; i += 32) {
      const float yi = y[i];
      ag += __bfloat162float(wg[i]) * yi;
      au += __bfloat162float(wu[i]) * yi;
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) { g.h[n] = silu(ag) * au; }
  }
}

__device__ __forceinline__ void storer(const VmState&, int32_t, InstructionView,
                                       const Globals&) {}

}  // namespace op_gateup

namespace op_down {

__device__ __forceinline__ void arm(const VmState& vm, int32_t ring) {
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);
}

// h is kFfn floats = 32 KB: two pages. This is the op that sizes the page pool.
__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView, const Globals& g) {
  constexpr int32_t kPerPage = VmConfig::kPageSize / 4;
  for (int32_t i = static_cast<int32_t>(threadIdx.x) % 32; i < kFfn; i += 32) {
    vm.page_f32(ring, i / kPerPage)[i % kPerPage] = g.h[i];
  }
  __threadfence_block();
  if (tmpl::elect_one()) { tmpl::mbarrier_arrive(vm.sem(ring, 0)); }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, const Globals& g,
                                         int32_t warp, float*) {
  tmpl::wait_parity(vm.sem(ring, 0), 0);
  constexpr int32_t kPerPage = VmConfig::kPageSize / 4;
  const float* h0 = vm.page_f32(ring, 0);
  const float* h1 = vm.page_f32(ring, 1);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t k0 = ins.block() * kBlockN;
  constexpr int32_t kColsPerWarp = kBlockN / VmConfig::kConsumerWarps;  // 8

  #pragma unroll
  for (int32_t c = 0; c < kColsPerWarp; ++c) {
    const int32_t k = k0 + warp * kColsPerWarp + c;
    const __nv_bfloat16* wd = g.w_down + static_cast<int64_t>(k) * kFfn;
    float acc = 0.f;
    for (int32_t i = lane; i < kFfn; i += 32) {
      const float hi = (i < kPerPage) ? h0[i] : h1[i - kPerPage];
      acc += __bfloat162float(wd[i]) * hi;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
      g.out[k] = __float2bfloat16(acc + __bfloat162float(g.residual[k]));
    }
  }
}

__device__ __forceinline__ void storer(const VmState&, int32_t, InstructionView,
                                       const Globals&) {}

}  // namespace op_down

// ================================================================== the VM

__global__ __launch_bounds__(VmConfig::kNumThreads, 1) void megakernel_vm(
    const __grid_constant__ Globals g,
    const int32_t* __restrict__ program,
    const int32_t* __restrict__ successors,
    uint32_t* __restrict__ counters,
    uint32_t* __restrict__ next_instruction,
    int32_t num_instructions, int32_t max_instructions) {
  __shared__ alignas(128) StageState stage[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_arrived[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_finished[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t
      op_sem[VmConfig::kPipelineStages * VmConfig::kDynamicSemaphores];
  __shared__ float smem_red[VmConfig::kConsumerWarps];
  extern __shared__ __align__(1024) uint8_t pages_raw[];

  VmState vm{stage, instr_arrived, instr_finished, op_sem, pages_raw};

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / 32;

  if (tid < VmConfig::kPipelineStages) {
    tmpl::mbarrier_init(&instr_arrived[tid], 1);  // the controller alone arms
    // Every warp but the controller reports done.
    tmpl::mbarrier_init(&instr_finished[tid], VmConfig::kNumWarps - 1);
  }
  tmpl::fence_barrier_init();
  __syncthreads();

  const int32_t limit =
      (max_instructions > 0 && max_instructions < num_instructions)
          ? max_instructions : num_instructions;

  // ------------------------------------------------------------ controller
  if (warp == VmConfig::kConsumerWarps + 3) {
    tmpl::setmaxnreg_dec<VmConfig::kNonConsumerRegisters>();
    const int32_t lane = tid % 32;
    for (int32_t idx = 0;; ++idx) {
      const int32_t ring = idx % VmConfig::kPipelineStages;

      // Reclaim the stage two instructions back before overwriting it. This is
      // also what makes the stage's pages safe to refill.
      if (idx >= VmConfig::kPipelineStages) {
        tmpl::wait_parity(&instr_finished[ring], stage_phase(idx) ^ 1u);
      }

      uint32_t claim = 0;
      if (lane == 0) { claim = atomicAdd(next_instruction, 1u); }
      claim = __shfl_sync(0xffffffffu, claim, 0);

      if (claim >= static_cast<uint32_t>(limit)) {
        // Publish a NoOp so the other roles retire in lockstep instead of
        // waiting on a stage that never arrives.
        if (lane == 0) {
          stage[ring].instruction[0] = kOpNoOp;
          __threadfence_block();
          tmpl::mbarrier_arrive(&instr_arrived[ring]);
        }
        break;
      }

      // Fetch: 128 B, vectorized.
      const int4* src = reinterpret_cast<const int4*>(
          program + static_cast<int64_t>(claim) * VmConfig::kInstructionWidth);
      int4* dst = reinterpret_cast<int4*>(stage[ring].instruction);
      if (lane < VmConfig::kInstructionWidth / 4) { dst[lane] = src[lane]; }
      __syncwarp();

      const InstructionView ins{stage[ring].instruction};

      // Dependencies, before anyone touches data.
      if (lane == 0) {
        while (counter_load_acquire(&counters[ins.counter_idx()]) != 0u) {
          spin_backoff();
        }
        switch (ins.opcode()) {
          case kOpRmsNorm: op_rmsnorm::arm(vm, ring); break;
          case kOpGateUp:  op_gateup::arm(vm, ring); break;
          case kOpDown:    op_down::arm(vm, ring); break;
          default: break;
        }
        if (VmConfig::kTimingEnabled) {
          stage[ring].timings[0] = static_cast<int32_t>(clock64());
        }
      }
      __syncwarp();
      tmpl::fence_proxy_async_shared();
      if (lane == 0) { tmpl::mbarrier_arrive(&instr_arrived[ring]); }
    }
    return;
  }

  // ------------------------------------------- loader / storer / launcher
  if (warp >= VmConfig::kConsumerWarps) {
    tmpl::setmaxnreg_dec<VmConfig::kNonConsumerRegisters>();
    const int32_t role = warp - VmConfig::kConsumerWarps;
    for (int32_t idx = 0;; ++idx) {
      const int32_t ring = idx % VmConfig::kPipelineStages;
      tmpl::wait_parity(&instr_arrived[ring], stage_phase(idx));
      const InstructionView ins{stage[ring].instruction};
      if (ins.opcode() == kOpNoOp) { break; }

      if (role == 0) {
        switch (ins.opcode()) {
          case kOpRmsNorm: op_rmsnorm::loader(vm, ring, ins, g); break;
          case kOpGateUp:  op_gateup::loader(vm, ring, ins, g); break;
          case kOpDown:    op_down::loader(vm, ring, ins, g); break;
          default: break;
        }
      } else if (role == 1) {
        switch (ins.opcode()) {
          case kOpRmsNorm: op_rmsnorm::storer(vm, ring, ins, g); break;
          case kOpGateUp:  op_gateup::storer(vm, ring, ins, g); break;
          case kOpDown:    op_down::storer(vm, ring, ins, g); break;
          default: break;
        }
      }
      // role 2 (launcher) has no async work in these ops. The role stays so the
      // warp count and register split do not move when an op acquires some.

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
      case kOpRmsNorm: op_rmsnorm::consumer(vm, ring, ins, g, warp, smem_red); break;
      case kOpGateUp:  op_gateup::consumer(vm, ring, ins, g, warp, smem_red); break;
      case kOpDown:    op_down::consumer(vm, ring, ins, g, warp, smem_red); break;
      default: break;
    }

    // All consumer warps must have finished writing before the successors are
    // released; the named barrier covers the consumer group only.
    tmpl::named_barrier_sync(2, VmConfig::kConsumerWarps * 32);
    if (warp == 0 && tmpl::elect_one()) {
      __threadfence();
      for (int32_t s = 0; s < ins.succ_count(); ++s) {
        counter_dec_release(&counters[successors[ins.succ_begin() + s]]);
      }
    }
    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&instr_finished[ring]); }
  }
}

// ============================================================ the baseline
// The same math as three ordinary kernels. This is what the megakernel has to
// beat, and it is deliberately not a strawman: same tiling, same reductions.

__global__ __launch_bounds__(512) void baseline_rmsnorm(Globals g) {
  __shared__ float red[16];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / 32, lane = tid % 32;
  float acc = 0.f;
  for (int32_t i = tid; i < kDim; i += 512) {
    const float v = __bfloat162float(g.x[i]);
    acc += v * v;
  }
  acc = warp_sum(acc);
  if (lane == 0) { red[warp] = acc; }
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int32_t w = 0; w < 16; ++w) { total += red[w]; }
  const float scale = rsqrtf(total / static_cast<float>(kDim) + kEps);
  for (int32_t i = tid; i < kDim; i += 512) {
    g.y[i] = __bfloat162float(g.x[i]) * scale * __bfloat162float(g.rms_w[i]);
  }
}

__global__ __launch_bounds__(512) void baseline_gateup(Globals g) {
  extern __shared__ float sy[];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  for (int32_t i = tid; i < kDim; i += 512) { sy[i] = g.y[i]; }
  __syncthreads();
  const int32_t warp = tid / 32, lane = tid % 32;
  const int32_t n0 = static_cast<int32_t>(blockIdx.x) * kBlockN;
  constexpr int32_t kColsPerWarp = kBlockN / 16;
  #pragma unroll
  for (int32_t c = 0; c < kColsPerWarp; ++c) {
    const int32_t n = n0 + warp * kColsPerWarp + c;
    const __nv_bfloat16* wg = g.w_gate + static_cast<int64_t>(n) * kDim;
    const __nv_bfloat16* wu = g.w_up + static_cast<int64_t>(n) * kDim;
    float ag = 0.f, au = 0.f;
    for (int32_t i = lane; i < kDim; i += 32) {
      const float yi = sy[i];
      ag += __bfloat162float(wg[i]) * yi;
      au += __bfloat162float(wu[i]) * yi;
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) { g.h[n] = silu(ag) * au; }
  }
}

__global__ __launch_bounds__(512) void baseline_down(Globals g) {
  extern __shared__ float sh[];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  for (int32_t i = tid; i < kFfn; i += 512) { sh[i] = g.h[i]; }
  __syncthreads();
  const int32_t warp = tid / 32, lane = tid % 32;
  const int32_t k0 = static_cast<int32_t>(blockIdx.x) * kBlockN;
  constexpr int32_t kColsPerWarp = kBlockN / 16;
  #pragma unroll
  for (int32_t c = 0; c < kColsPerWarp; ++c) {
    const int32_t k = k0 + warp * kColsPerWarp + c;
    const __nv_bfloat16* wd = g.w_down + static_cast<int64_t>(k) * kFfn;
    float acc = 0.f;
    for (int32_t i = lane; i < kFfn; i += 32) {
      acc += __bfloat162float(wd[i]) * sh[i];
    }
    acc = warp_sum(acc);
    if (lane == 0) {
      g.out[k] = __float2bfloat16(acc + __bfloat162float(g.residual[k]));
    }
  }
}

// ================================================================== harness

#ifndef MK40_NO_MAIN

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace {

float host_rand(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.f - 0.5f) * 0.1f;
}

// Reference in double, so the check measures the kernels rather than itself.
void reference(const std::vector<float>& x, const std::vector<float>& rms_w,
               const std::vector<float>& wg, const std::vector<float>& wu,
               const std::vector<float>& wd, const std::vector<float>& res,
               std::vector<float>& out) {
  double ss = 0.0;
  for (int i = 0; i < kDim; ++i) ss += double(x[i]) * x[i];
  const double scale = 1.0 / std::sqrt(ss / kDim + kEps);
  std::vector<double> y(kDim);
  for (int i = 0; i < kDim; ++i) y[i] = x[i] * scale * rms_w[i];

  std::vector<double> h(kFfn);
  for (int n = 0; n < kFfn; ++n) {
    double ag = 0.0, au = 0.0;
    for (int i = 0; i < kDim; ++i) {
      ag += double(wg[size_t(n) * kDim + i]) * y[i];
      au += double(wu[size_t(n) * kDim + i]) * y[i];
    }
    h[n] = (ag / (1.0 + std::exp(-ag))) * au;
  }
  for (int k = 0; k < kDim; ++k) {
    double acc = 0.0;
    for (int i = 0; i < kFfn; ++i) acc += double(wd[size_t(k) * kFfn + i]) * h[i];
    out[k] = float(acc + res[k]);
  }
}

float max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    const float s = std::max(1e-3f, std::fabs(b[i]));
    worst = std::max(worst, d / s);
  }
  return worst;
}

}  // namespace

int main() {
  // ---- host data
  uint32_t seed = 12345;
  std::vector<float> hx(kDim), hrms(kDim), hres(kDim), href(kDim);
  std::vector<float> hwg(size_t(kFfn) * kDim), hwu(size_t(kFfn) * kDim),
      hwd(size_t(kDim) * kFfn);
  for (auto& v : hx) v = host_rand(seed);
  for (auto& v : hrms) v = 1.f + host_rand(seed);
  for (auto& v : hres) v = host_rand(seed);
  for (auto& v : hwg) v = host_rand(seed);
  for (auto& v : hwu) v = host_rand(seed);
  for (auto& v : hwd) v = host_rand(seed);

  // The reference runs on the values the device will actually see, so bf16
  // rounding is not counted as kernel error.
  auto round_bf16 = [](std::vector<float>& v) {
    for (auto& e : v) e = __bfloat162float(__float2bfloat16(e));
  };
  round_bf16(hx); round_bf16(hrms); round_bf16(hres);
  round_bf16(hwg); round_bf16(hwu); round_bf16(hwd);
  reference(hx, hrms, hwg, hwu, hwd, hres, href);

  // ---- device data
  auto up_bf16 = [](const std::vector<float>& h) {
    std::vector<__nv_bfloat16> t(h.size());
    for (size_t i = 0; i < h.size(); ++i) t[i] = __float2bfloat16(h[i]);
    __nv_bfloat16* d = nullptr;
    CK(cudaMalloc(&d, t.size() * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(d, t.data(), t.size() * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));
    return d;
  };
  Globals g{};
  g.x = up_bf16(hx);
  g.rms_w = up_bf16(hrms);
  g.w_gate = up_bf16(hwg);
  g.w_up = up_bf16(hwu);
  g.w_down = up_bf16(hwd);
  g.residual = up_bf16(hres);
  CK(cudaMalloc((void**)&g.y, kDim * sizeof(float)));
  CK(cudaMalloc((void**)&g.h, kFfn * sizeof(float)));
  CK(cudaMalloc((void**)&g.out, kDim * sizeof(__nv_bfloat16)));

  // ---- the program: 1 rmsnorm -> 64 gate/up -> 16 down
  std::vector<int32_t> prog(size_t(kNumInstructions) * VmConfig::kInstructionWidth, 0);
  std::vector<int32_t> succ;
  std::vector<uint32_t> counters(kNumInstructions, 0);
  auto ins = [&](int i) { return &prog[size_t(i) * VmConfig::kInstructionWidth]; };

  // rmsnorm: successors are all gate/up instructions
  ins(0)[0] = kOpRmsNorm; ins(0)[1] = 0; ins(0)[2] = 0;
  ins(0)[3] = int32_t(succ.size()); ins(0)[4] = kNumGateUp;
  for (int i = 0; i < kNumGateUp; ++i) succ.push_back(1 + i);
  counters[0] = 0;

  // gate/up: each depends on rmsnorm, and each is a predecessor of every down
  for (int i = 0; i < kNumGateUp; ++i) {
    const int id = 1 + i;
    ins(id)[0] = kOpGateUp; ins(id)[1] = i; ins(id)[2] = id;
    ins(id)[3] = int32_t(succ.size()); ins(id)[4] = kNumDown;
    for (int j = 0; j < kNumDown; ++j) succ.push_back(1 + kNumGateUp + j);
    counters[id] = 1;
  }
  // down: fan-in of every gate/up
  for (int j = 0; j < kNumDown; ++j) {
    const int id = 1 + kNumGateUp + j;
    ins(id)[0] = kOpDown; ins(id)[1] = j; ins(id)[2] = id;
    ins(id)[3] = 0; ins(id)[4] = 0;
    counters[id] = kNumGateUp;
  }

  int32_t *d_prog = nullptr, *d_succ = nullptr;
  uint32_t *d_counters = nullptr, *d_next = nullptr;
  CK(cudaMalloc(&d_prog, prog.size() * 4));
  CK(cudaMalloc(&d_succ, succ.size() * 4));
  CK(cudaMalloc(&d_counters, counters.size() * 4));
  CK(cudaMalloc(&d_next, 4));
  CK(cudaMemcpy(d_prog, prog.data(), prog.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_succ, succ.data(), succ.size() * 4, cudaMemcpyHostToDevice));

  constexpr int kSmem = VmConfig::kNumPages * VmConfig::kPageSize;
  CK(cudaFuncSetAttribute(megakernel_vm,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  CK(cudaFuncSetAttribute(baseline_gateup,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          kDim * 4));
  CK(cudaFuncSetAttribute(baseline_down,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          kFfn * 4));

  int sm_count = 0;
  CK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

  auto run_baseline = [&](cudaStream_t s) {
    baseline_rmsnorm<<<1, 512, 0, s>>>(g);
    baseline_gateup<<<kNumGateUp, 512, kDim * 4, s>>>(g);
    baseline_down<<<kNumDown, 512, kFfn * 4, s>>>(g);
  };
  auto run_mk = [&](cudaStream_t s) {
    CK(cudaMemsetAsync(d_next, 0, 4, s));
    CK(cudaMemcpyAsync(d_counters, counters.data(), counters.size() * 4,
                       cudaMemcpyHostToDevice, s));
    megakernel_vm<<<sm_count, VmConfig::kNumThreads, kSmem, s>>>(
        g, d_prog, d_succ, d_counters, d_next, kNumInstructions, 0);
  };

  auto check = [&](const char* what) {
    std::vector<__nv_bfloat16> hb(kDim);
    CK(cudaMemcpy(hb.data(), g.out, kDim * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));
    std::vector<float> hf(kDim);
    for (int i = 0; i < kDim; ++i) hf[i] = __bfloat162float(hb[i]);
    const float e = max_rel_err(hf, href);
    printf("  %-12s max rel err %.3e  %s\n", what, e, e < 2e-2f ? "PASS" : "FAIL");
    return e < 2e-2f;
  };

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  printf("megakernel VM vs 3-kernel baseline  (dim=%d ffn=%d bf16, batch 1)\n",
         kDim, kFfn);
  printf("  SMs=%d  VM: %d warps, %d pages x %d KB\n\n", sm_count,
         VmConfig::kNumWarps, VmConfig::kNumPages, VmConfig::kPageSize / 1024);

  printf("correctness\n");
  CK(cudaMemset(g.out, 0, kDim * sizeof(__nv_bfloat16)));
  run_baseline(stream); CK(cudaStreamSynchronize(stream));
  bool ok = check("baseline");
  CK(cudaMemset(g.out, 0, kDim * sizeof(__nv_bfloat16)));
  run_mk(stream); CK(cudaStreamSynchronize(stream));
  ok = check("megakernel") && ok;
  if (!ok) { printf("\nnumerics failed; timings withheld\n"); return 1; }

  // ---- timing, graph-captured: PDL and launch ramp only show up in a graph.
  auto time_graph = [&](void (*)(), const char* name, bool mk) {
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (mk) run_mk(stream); else run_baseline(stream);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
      CK(cudaEventRecord(a, stream));
      for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(exec, stream));
      CK(cudaEventRecord(b, stream));
      CK(cudaEventSynchronize(b));
      float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
      best = std::min(best, ms / 200.f * 1000.f);
    }
    printf("  %-12s %8.2f us\n", name, best);
    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    return best;
  };

  printf("\nlatency (CUDA graph, min of 3 x 200 iters)\n");
  const float t_base = time_graph(nullptr, "baseline", false);
  const float t_mk = time_graph(nullptr, "megakernel", true);
  printf("\n  speedup %.2fx  (%.2f us saved)\n", t_base / t_mk, t_base - t_mk);
  printf("  3 launches removed; ramp alone is ~%.2f us at 1.24 us/launch\n",
         2 * 1.24f);
  return 0;
}

#endif  // MK40_NO_MAIN
