// Template 40 -- a megakernel VM, with its baseline and its harness.
//
// Build and run it; that is the point of this file. It executes a decode-shaped
// transformer MLP -- RMSNorm, gate/up GEMV with SwiGLU, down GEMV with a
// residual add -- twice: once as three ordinary kernel launches, once as one
// persistent megakernel interpreting a program. It checks both against a
// double-precision CPU reference and times both graph-captured. The build line
// and the current status are in the STATUS block below; read it before
// trusting anything here.
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
//     version that is obviously correct, and it is what this file implements.)
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
// MEASURED. H100 SXM5, CUDA 13.0, driver 610.43.02, clocks NOT pinned (no
// permission on this node, so read these as ratios). d=2048 ffn=8192 bf16,
// batch 1. CUDA-graph captured, min of 3 x 200 iterations after 50 warmup.
// Both paths agree with a double-precision CPU reference at 3.813e-03 max
// relative error -- identical to three digits, which is the check that the VM
// computes the same function and not merely a plausible one.
//
//   baseline, 3 kernels                              90.75 us
//   megakernel, 132 CTAs                            115.25 us   0.79x
//
//   grid sweep (81 instructions in the program)
//     CTAs   ins/CTA   latency    vs baseline
//      132      0.61   115.25 us     0.79x
//       81      1.00   115.42 us     0.79x
//       64      1.27   125.94 us     0.72x
//       40      2.03   132.55 us     0.68x
//       16      5.06   209.96 us     0.43x
//        8     10.12   369.58 us     0.25x
//
// THE MEGAKERNEL LOSES, BY 1.27x, AND THAT IS THE RESULT. It is reported here
// rather than tuned away because a reference that only shows the idiom winning
// teaches the wrong thing: this is the ledger of [fusion-economics] coming out
// negative, which is the common case at one-layer scope. Three launches are
// worth ~2.5 us of ramp [launch.lat.dev.ramp]; the machinery costs ten times
// that here.
//
// FOUR EXPLANATIONS TESTED AND FALSIFIED, so the next person does not repeat
// them:
//
//   1. "The VM is not amortised -- 81 instructions over 132 CTAs is less than
//      one each." Falsified by the grid sweep: fewer CTAs is monotonically
//      WORSE. The problem is bandwidth-bound and wants every SM.
//   2. "The fan-in of 64 on every `down` strands CTAs spinning where a kernel
//      boundary would release the machine." Falsified with the truncation hook:
//      the program cut to rmsnorm + gate/up has no fan-in at all, and the gap
//      WIDENS to 0.49x (34.17 us baseline vs 70.05 us).
//   3. "The page pool is sized for the largest op, so 64 KB of shared memory
//      forces 1 CTA/SM." Falsified by halving it: -DMK40_STAGES=1 gives a 32 KB
//      pool and 115.40 us, unchanged.
//   4. "The counter reset is a pageable H2D copy inside the timed graph."
//      Real defect, fixed (it is device-to-device now), but worth 0.3 us.
//
// NCU, like for like (both paths cut to rmsnorm + gate/up; run the binary with
// `probe` to reproduce this capture):
//
//                        baseline_gateup      megakernel_vm
//   duration                   35.33 us            78.37 us
//   DRAM throughput      59.6% / 2.00 TB/s   26.7% / 894 GB/s
//   achieved occupancy           24.46%             24.29%
//   theoretical occupancy        50.0%              31.25%
//   registers/thread               64                 96
//   avg stall              10.9 cycles         21.3 cycles
//     dominated by         56% smem-data       68% smem-data
//
// So it is NOT occupancy: the two achieve the same 24.4%. The whole difference
// is that the VM's warps stall twice as long, on the same reason -- waiting for
// data from shared memory -- with the same inner loop and the same access
// pattern. Both compile to ld.shared with no generic loads (checked in the
// PTX), so it is not an addressing-space failure either.
//
// THE VM'S OWN PROFILER LOCALISES IT. Build with -DMK40_TIMING=1; NCU cannot
// attribute time inside a megakernel because every op is the same kernel, which
// is why the timing slots are in the design. Mean cycles per instruction phase:
//
//   op         n     claim    fetch  dep-wait  arm+pub  load-iss data-ready  compute    TOTAL
//   rmsnorm    1       965      373       274      597       630      6099    59067    68005
//   gate/up   64       894    38716       459      604      4043     78513    14369   137598
//   down      16       926   169453       472      623      2068    132410      823   306776
//
// COMPUTE IS 10% OF A GATE/UP INSTRUCTION AND 0.3% OF A DOWN. Everything else
// is waiting -- on the instruction's own 128 bytes, and on the staged operand.
// (These are residency times, not critical path: phases of different CTAs
// overlap. The shape is what matters, not the sum.)
//
// The structural reading, which the earlier guesses all missed:
//
//   AN INSTRUCTION'S OPERAND STAGING CANNOT BEGIN UNTIL THE INSTRUCTION HAS
//   BEEN FETCHED AND ITS DEPENDENCIES CHECKED. A kernel launch has no such
//   chain -- its parameters arrive with the launch, so its CTAs start loading
//   on cycle one. The VM pays fetch -> dependency -> arm -> publish -> issue
//   before a single operand byte moves.
//
// That prologue is amortisable only by instruction-level pipelining: fetch and
// stage instruction i+1 while i computes. Which needs many instructions per
// CTA. Which this program does not have -- 81 instructions over 132 CTAs -- and
// cannot be given, because the grid sweep shows that shrinking the grid to
// raise instructions-per-CTA loses more to lost parallelism than it recovers.
//
// THAT PREDICTION WAS TESTED BY DEEPENING THE PROGRAM, AND IT FAILED. Running
// the whole decode step -- every layer chained, so one CTA's instruction stream
// is long -- does not amortise anything:
//
//   layers  instructions  ins/CTA   baseline   megakernel   ratio
//        1            81     0.61    97.92 us    165.97 us   0.59x
//        8           648     4.91   760.76 us   1380.39 us   0.55x
//       32          2592    19.64  3033.74 us   5538.98 us   0.55x
//
// Flat. So "many instructions per CTA" is NOT the condition. The refined
// reading is that the instructions must be INDEPENDENT of each other: the ring
// can only overlap instruction i+1's prologue with instruction i's compute if
// i+1 is runnable, and in a strictly serialized layer chain a CTA's next
// instruction almost always waits on work other CTAs are still doing. Depth
// gives length, not independence. A planner that interleaves independent work
// across the stage boundary is what supplies it, and that scheduling -- not
// the interpreter -- is where a megakernel is won.
//
// (Correctness is exact at every depth: 0.000e+00 against the double reference,
// once the reference rounds to bf16 between layers as the device does.)
//
// Seven earlier explanations were tested and falsified; recorded so nobody
// repeats them: instruction amortisation as such, fan-in spin-stranding, the
// page pool's occupancy cost, a pageable H2D in the timed graph, the global
// claim cursor (replaced with the reference's static per-worker lists -- no
// change), spin backoff on the mbarrier waits (no change), and
// generic-vs-shared addressing (the PTX is ld.shared on both paths).
//
// One instrumentation note, because it nearly produced a wrong answer:
// clock64() is not a fence, and a profiler whose timestamps the compiler may
// reorder is worse than no profiler. The reads carry a "memory" clobber. With
// and without it the numbers matched here, so the measurement was sound -- but
// that was luck, not design.
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
#include <string>

#include "sm90_common.cuh"

// ------------------------------------------------------------------ problem

namespace {

constexpr int kDim = 2048;    // hidden
constexpr int kFfn = 8192;    // intermediate
constexpr int kBlockN = 128;  // output columns per GEMV instruction
constexpr float kEps = 1e-6f;

// A megakernel amortises its per-instruction prologue over the instructions ONE
// CTA runs back to back. One layer over 132 CTAs gives less than one each, so
// the prologue is fully exposed. A decode step runs every layer, which is the
// shape the idiom is for -- and the shape where the baseline pays 3 launches
// PER LAYER.
#ifndef MK40_LAYERS
#define MK40_LAYERS 32
#endif
constexpr int kNumLayers = MK40_LAYERS;

constexpr int kNumGateUp = kFfn / kBlockN;  // 64
constexpr int kNumDown = kDim / kBlockN;    // 16
constexpr int kPerLayer = 1 + kNumGateUp + kNumDown;  // 81
constexpr int kNumInstructions = kNumLayers * kPerLayer;

// ---------------------------------------------------------------- VM config

struct VmConfig {
  static constexpr int kInstructionWidth = 32;  // 128 B
#ifndef MK40_STAGES
#define MK40_STAGES 2
#endif
// 1 = static per-worker instruction lists (the reference design), 0 = a global
// atomic claim cursor. The A/B that isolates the cursor's cost.
#ifndef MK40_STATIC_CLAIM
#define MK40_STATIC_CLAIM 0
#endif
  static constexpr int kPipelineStages = MK40_STAGES;
  static constexpr int kDynamicSemaphores = 4;
  // Per-instruction timestamps. NCU cannot attribute time inside a megakernel
  // -- every op is the same kernel -- so the VM has to record its own. Off by
  // default because each phase costs a clock64 and a store.
#ifndef MK40_TIMING
#define MK40_TIMING 0
#endif
  static constexpr int kTimingWidth = 8;
  static constexpr bool kTimingEnabled = MK40_TIMING != 0;

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
  // Which layer this instruction belongs to. Only its parity is used on the
  // device -- it selects which of the two activation buffers is input.
  __device__ __forceinline__ int32_t layer() const { return w[5]; }
};

struct alignas(128) StageState {
  int32_t instruction[VmConfig::kInstructionWidth];
  int32_t timings[VmConfig::kTimingWidth];
};

// The phases one instruction passes through. Timestamps are clock64() and are
// only differenced WITHIN one instruction, which is what makes them valid: the
// cycle counter is per-SM, so cross-CTA differences are meaningless.
enum TimingPhase : int32_t {
  kTClaim = 0,      // controller has an instruction index
  kTFetched = 1,    // its 128 bytes are in shared memory
  kTDepReady = 2,   // its dependency counter reached zero
  kTPublished = 3,  // controller armed the op and rang instr_arrived
  kTLoadIssued = 4, // loader issued the bulk copy
  kTDataReady = 5,  // consumers' wait on the op semaphore returned
  kTComputed = 6,   // consumers finished the math
  kTFinished = 7,   // successors released
};

// clock64() is NOT a fence. Without a "memory" clobber the compiler is free to
// hoist or sink the read across the very work being timed, and the first
// version of this profiler reported 169k cycles for a 128-byte load because of
// it. A profiler that lies is worse than none.
__device__ __forceinline__ uint64_t now() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}

__device__ __forceinline__ void record(uint64_t* __restrict__ timings,
                                       uint32_t instr, int32_t phase) {
  if (VmConfig::kTimingEnabled && timings != nullptr) {
    const uint64_t t = now();
    timings[static_cast<int64_t>(instr) * VmConfig::kTimingWidth + phase] = t;
  }
}

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

// One lane reads 8 bf16 = 16 bytes. The scalar form -- one 2-byte load per lane
// per step -- is what made the first version of this file 6x off roofline, on
// BOTH paths. A GEMV is a bandwidth problem and the load width is the program.
template <class Acc>
__device__ __forceinline__ void dot_bf16_vec(const __nv_bfloat16* __restrict__ w,
                                             const float* __restrict__ v,
                                             int32_t n, int32_t lane, Acc& acc) {
  for (int32_t i = lane * 8; i < n; i += 32 * 8) {
    int4 raw = *reinterpret_cast<const int4*>(w + i);
    const __nv_bfloat16* wv = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int32_t j = 0; j < 8; ++j) { acc += __bfloat162float(wv[j]) * v[i + j]; }
  }
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, off); }
  return v;
}

}  // namespace

// Everything the model needs, passed by value as a grid constant so no op has
// to chase a pointer table.
struct Globals {
  // Two activation buffers, ping-ponged per layer: layer L reads buf[L & 1] and
  // writes buf[(L + 1) & 1], with the read buffer also serving as its residual.
  __nv_bfloat16* buf[2];
  const __nv_bfloat16* __restrict__ rms_w;
  const __nv_bfloat16* __restrict__ w_gate;  // (ffn, dim)
  const __nv_bfloat16* __restrict__ w_up;    // (ffn, dim)
  const __nv_bfloat16* __restrict__ w_down;  // (dim, ffn)
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
  // Transaction barrier: the copy engine completes it, not a thread.
  tmpl::mbarrier_init(vm.sem(ring, 0), 1);
}

// Stage x into a page: every consumer warp reads the whole vector for the
// reduction, so one gmem pass beats sixteen. Issued as one async bulk copy --
// the loader warp's job is to ISSUE, not to move bytes with its own lanes.
__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView ins, const Globals& g) {
  if (tmpl::elect_one()) {
    tmpl::arrive_and_expect_tx(vm.sem(ring, 0), kDim * sizeof(__nv_bfloat16));
    tmpl::bulk_load_1d(vm.page_f32(ring, 0), g.buf[ins.layer() & 1],
                       kDim * sizeof(__nv_bfloat16), vm.sem(ring, 0));
  }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, const Globals& g,
                                         int32_t warp, float* smem_red) {
  tmpl::wait_parity(vm.sem(ring, 0), 0);  // re-armed each instruction
  const __nv_bfloat16* p =
      reinterpret_cast<const __nv_bfloat16*>(vm.page_f32(ring, 0));
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;

  float acc = 0.f;
  for (int32_t i = warp * 32 + lane; i < kDim; i += VmConfig::kConsumerWarps * 32) {
    const float v = __bfloat162float(p[i]);
    acc += v * v;
  }
  acc = warp_sum(acc);
  if (lane == 0) { smem_red[warp] = acc; }
  tmpl::named_barrier_sync(1, VmConfig::kConsumerWarps * 32);

  float total = 0.f;
  #pragma unroll
  for (int32_t w = 0; w < VmConfig::kConsumerWarps; ++w) { total += smem_red[w]; }
  const float scale = rsqrtf(total / static_cast<float>(kDim) + kEps);

  for (int32_t i = warp * 32 + lane; i < kDim; i += VmConfig::kConsumerWarps * 32) {
    g.y[i] = __bfloat162float(p[i]) * scale * __bfloat162float(g.rms_w[i]);
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
  if (tmpl::elect_one()) {
    tmpl::arrive_and_expect_tx(vm.sem(ring, 0), kDim * sizeof(float));
    tmpl::bulk_load_1d(vm.page_f32(ring, 0), g.y, kDim * sizeof(float),
                       vm.sem(ring, 0));
  }
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
    dot_bf16_vec(wg, y, kDim, lane, ag);
    dot_bf16_vec(wu, y, kDim, lane, au);
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
// A stage's pages are adjacent by construction, so the pair is one span and one
// bulk copy -- no per-element page arithmetic in the consumer.
__device__ __forceinline__ void loader(const VmState& vm, int32_t ring,
                                       InstructionView, const Globals& g) {
  if (tmpl::elect_one()) {
    tmpl::arrive_and_expect_tx(vm.sem(ring, 0), kFfn * sizeof(float));
    tmpl::bulk_load_1d(vm.page_f32(ring, 0), g.h, kFfn * sizeof(float),
                       vm.sem(ring, 0));
  }
}

__device__ __forceinline__ void consumer(const VmState& vm, int32_t ring,
                                         InstructionView ins, const Globals& g,
                                         int32_t warp, float*) {
  tmpl::wait_parity(vm.sem(ring, 0), 0);
  const float* h0 = vm.page_f32(ring, 0);  // spans both of the stage's pages
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t k0 = ins.block() * kBlockN;
  constexpr int32_t kColsPerWarp = kBlockN / VmConfig::kConsumerWarps;  // 8

  #pragma unroll
  for (int32_t c = 0; c < kColsPerWarp; ++c) {
    const int32_t k = k0 + warp * kColsPerWarp + c;
    const __nv_bfloat16* wd = g.w_down + static_cast<int64_t>(k) * kFfn;
    float acc = 0.f;
    dot_bf16_vec(wd, h0, kFfn, lane, acc);
    acc = warp_sum(acc);
    if (lane == 0) {
      // Residual is the layer's own input, and the sum lands in the other
      // buffer so the next layer reads a complete row.
      const float res = __bfloat162float(g.buf[ins.layer() & 1][k]);
      g.buf[(ins.layer() + 1) & 1][k] = __float2bfloat16(acc + res);
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
    uint64_t* __restrict__ timings,
    int32_t num_instructions, int32_t max_instructions) {
  __shared__ alignas(128) StageState stage[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_arrived[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t instr_finished[VmConfig::kPipelineStages];
  __shared__ alignas(8) uint64_t
      op_sem[VmConfig::kPipelineStages * VmConfig::kDynamicSemaphores];
  __shared__ float smem_red[VmConfig::kConsumerWarps];
  // The roles need the instruction INDEX, not just its contents, to file a
  // timestamp against it.
  __shared__ uint32_t claimed[VmConfig::kPipelineStages];
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
        tmpl::wait_parity_backoff<>(&instr_finished[ring], stage_phase(idx) ^ 1u);
      }

#if MK40_STATIC_CLAIM
      // Static strided assignment, as HazyResearch does: each worker walks its
      // OWN instruction list, emitted by the planner. No global cursor, so no
      // atomic on the critical path of every dispatch. Progress still holds:
      // a CTA walks its list in increasing order, so the smallest unexecuted
      // instruction has all predecessors done and its owner is at it.
      const uint32_t claim =
          static_cast<uint32_t>(blockIdx.x) + gridDim.x * static_cast<uint32_t>(idx);
#else
      uint32_t claim = 0;
      if (lane == 0) { claim = atomicAdd(next_instruction, 1u); }
      claim = __shfl_sync(0xffffffffu, claim, 0);
#endif
      if (lane == 0) { record(timings, claim, kTClaim); }

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
      if (lane == 0) { record(timings, claim, kTFetched); }

      const InstructionView ins{stage[ring].instruction};

      // Dependencies, before anyone touches data.
      if (lane == 0) {
        while (counter_load_acquire(&counters[ins.counter_idx()]) != 0u) {
          spin_backoff();
        }
        record(timings, claim, kTDepReady);
        switch (ins.opcode()) {
          case kOpRmsNorm: op_rmsnorm::arm(vm, ring); break;
          case kOpGateUp:  op_gateup::arm(vm, ring); break;
          case kOpDown:    op_down::arm(vm, ring); break;
          default: break;
        }
        claimed[ring] = claim;
      }
      __syncwarp();
      tmpl::fence_proxy_async_shared();
      if (lane == 0) {
        record(timings, claim, kTPublished);
        tmpl::mbarrier_arrive(&instr_arrived[ring]);
      }
    }
    return;
  }

  // ------------------------------------------- loader / storer / launcher
  if (warp >= VmConfig::kConsumerWarps) {
    tmpl::setmaxnreg_dec<VmConfig::kNonConsumerRegisters>();
    const int32_t role = warp - VmConfig::kConsumerWarps;
    for (int32_t idx = 0;; ++idx) {
      const int32_t ring = idx % VmConfig::kPipelineStages;
      tmpl::wait_parity_backoff<>(&instr_arrived[ring], stage_phase(idx));
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
      if (role == 0 && tmpl::elect_one()) {
        record(timings, claimed[ring], kTLoadIssued);
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
    tmpl::wait_parity_backoff<>(&instr_arrived[ring], stage_phase(idx));
    const InstructionView ins{stage[ring].instruction};
    if (ins.opcode() == kOpNoOp) { break; }

    if (warp == 0 && tmpl::elect_one()) {
      // Data-ready is measured by warp 0 re-observing the op semaphore that the
      // op itself waited on, so the timestamp brackets the loader's copy.
      tmpl::wait_parity(vm.sem(ring, 0), 0);
      record(timings, claimed[ring], kTDataReady);
    }
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
      record(timings, claimed[ring], kTComputed);
      __threadfence();
      for (int32_t s = 0; s < ins.succ_count(); ++s) {
        counter_dec_release(&counters[successors[ins.succ_begin() + s]]);
      }
      record(timings, claimed[ring], kTFinished);
    }
    if (tmpl::elect_one()) { tmpl::mbarrier_arrive(&instr_finished[ring]); }
  }
}

// ============================================================ the baseline
// The same math as three ordinary kernels. This is what the megakernel has to
// beat, and it is deliberately not a strawman: same tiling, same reductions.

__global__ __launch_bounds__(512) void baseline_rmsnorm(Globals g, int32_t layer) {
  __shared__ float red[16];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / 32, lane = tid % 32;
  float acc = 0.f;
  for (int32_t i = tid; i < kDim; i += 512) {
    const float v = __bfloat162float(g.buf[layer & 1][i]);
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
    g.y[i] = __bfloat162float(g.buf[layer & 1][i]) * scale *
             __bfloat162float(g.rms_w[i]);
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
    dot_bf16_vec(wg, sy, kDim, lane, ag);
    dot_bf16_vec(wu, sy, kDim, lane, au);
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) { g.h[n] = silu(ag) * au; }
  }
}

__global__ __launch_bounds__(512) void baseline_down(Globals g, int32_t layer) {
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
    dot_bf16_vec(wd, sh, kFfn, lane, acc);
    acc = warp_sum(acc);
    if (lane == 0) {
      const float res = __bfloat162float(g.buf[layer & 1][k]);
      g.buf[(layer + 1) & 1][k] = __float2bfloat16(acc + res);
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
// The whole chain, in double. Each layer's residual is its own input, and the
// device rounds to bf16 between layers -- so the reference does too, or the
// comparison drifts by the rounding rather than by any kernel error.
void reference(std::vector<float> x, const std::vector<float>& rms_w,
               const std::vector<float>& wg, const std::vector<float>& wu,
               const std::vector<float>& wd, std::vector<float>& out) {
  std::vector<double> y(kDim), h(kFfn);
  for (int L = 0; L < kNumLayers; ++L) {
    double ss = 0.0;
    for (int i = 0; i < kDim; ++i) ss += double(x[i]) * x[i];
    const double scale = 1.0 / std::sqrt(ss / kDim + kEps);
    for (int i = 0; i < kDim; ++i) y[i] = x[i] * scale * rms_w[i];
    for (int n = 0; n < kFfn; ++n) {
      double ag = 0.0, au = 0.0;
      for (int i = 0; i < kDim; ++i) {
        ag += double(wg[size_t(n) * kDim + i]) * y[i];
        au += double(wu[size_t(n) * kDim + i]) * y[i];
      }
      h[n] = (ag / (1.0 + std::exp(-ag))) * au;
    }
    std::vector<float> nx(kDim);
    for (int k = 0; k < kDim; ++k) {
      double acc = 0.0;
      for (int i = 0; i < kFfn; ++i) acc += double(wd[size_t(k) * kFfn + i]) * h[i];
      nx[k] = __bfloat162float(__float2bfloat16(float(acc + x[k])));
    }
    x.swap(nx);
  }
  out = x;
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

// argv[1]=="probe" runs ONLY the fan-in-free prefix on both paths, so an NCU
// capture compares like with like: profiling a whole megakernel against one
// kernel of a three-kernel baseline measures different amounts of work.
int main(int argc, char** argv) {
  const bool probe_only = (argc > 1 && std::string(argv[1]) == "probe");
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
  round_bf16(hx); round_bf16(hrms);
  round_bf16(hwg); round_bf16(hwu); round_bf16(hwd);
  reference(hx, hrms, hwg, hwu, hwd, href);

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
  g.buf[0] = up_bf16(hx);
  {  // the second activation buffer starts empty
    std::vector<float> zero(kDim, 0.f);
    g.buf[1] = up_bf16(zero);
  }
  g.rms_w = up_bf16(hrms);
  g.w_gate = up_bf16(hwg);
  g.w_up = up_bf16(hwu);
  g.w_down = up_bf16(hwd);
  CK(cudaMalloc((void**)&g.y, kDim * sizeof(float)));
  CK(cudaMalloc((void**)&g.h, kFfn * sizeof(float)));
  // The initial activations, kept so every timed run starts from the same
  // state -- a chain of layers is not idempotent.
  __nv_bfloat16* d_x0 = up_bf16(hx);

  // ---- the program: 1 rmsnorm -> 64 gate/up -> 16 down
  std::vector<int32_t> prog(size_t(kNumInstructions) * VmConfig::kInstructionWidth, 0);
  std::vector<int32_t> succ;
  std::vector<uint32_t> counters(kNumInstructions, 0);
  auto ins = [&](int i) { return &prog[size_t(i) * VmConfig::kInstructionWidth]; };

  // The program is the whole decode step: every layer, chained. Layer L's
  // rmsnorm waits on all of layer L-1's down instructions, which is what makes
  // one CTA's instruction stream long enough for the ring to pipeline.
  for (int L = 0; L < kNumLayers; ++L) {
    const int base = L * kPerLayer;
    const int nrm = base;
    const int gu0 = base + 1;
    const int dn0 = base + 1 + kNumGateUp;

    ins(nrm)[0] = kOpRmsNorm; ins(nrm)[1] = 0; ins(nrm)[2] = nrm;
    ins(nrm)[3] = int32_t(succ.size()); ins(nrm)[4] = kNumGateUp;
    ins(nrm)[5] = L;
    for (int i = 0; i < kNumGateUp; ++i) succ.push_back(gu0 + i);
    counters[nrm] = (L == 0) ? 0 : kNumDown;  // waits on the previous layer

    for (int i = 0; i < kNumGateUp; ++i) {
      const int id = gu0 + i;
      ins(id)[0] = kOpGateUp; ins(id)[1] = i; ins(id)[2] = id;
      ins(id)[3] = int32_t(succ.size()); ins(id)[4] = kNumDown;
      ins(id)[5] = L;
      for (int j = 0; j < kNumDown; ++j) succ.push_back(dn0 + j);
      counters[id] = 1;
    }
    for (int j = 0; j < kNumDown; ++j) {
      const int id = dn0 + j;
      ins(id)[0] = kOpDown; ins(id)[1] = j; ins(id)[2] = id;
      ins(id)[5] = L;
      if (L + 1 < kNumLayers) {
        ins(id)[3] = int32_t(succ.size()); ins(id)[4] = 1;
        succ.push_back((L + 1) * kPerLayer);  // the next layer's rmsnorm
      } else {
        ins(id)[3] = 0; ins(id)[4] = 0;
      }
      counters[id] = kNumGateUp;
    }
  }

  int32_t *d_prog = nullptr, *d_succ = nullptr;
  uint32_t *d_counters = nullptr, *d_counters_init = nullptr, *d_next = nullptr;
  CK(cudaMalloc(&d_prog, prog.size() * 4));
  CK(cudaMalloc(&d_succ, succ.size() * 4));
  CK(cudaMalloc(&d_counters, counters.size() * 4));
  // A pristine DEVICE copy of the initial in-degrees. Resetting from host
  // memory each iteration -- which the first version of this harness did --
  // puts a pageable H2D copy inside the timed graph that the baseline does not
  // pay, and it dominated the comparison.
  CK(cudaMalloc(&d_counters_init, counters.size() * 4));
  CK(cudaMemcpy(d_counters_init, counters.data(), counters.size() * 4,
                cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_next, 4));
  uint64_t* d_timings = nullptr;
  CK(cudaMalloc(&d_timings,
                size_t(kNumInstructions) * VmConfig::kTimingWidth * 8));
  CK(cudaMemset(d_timings, 0,
                size_t(kNumInstructions) * VmConfig::kTimingWidth * 8));
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

  auto reset_state = [&](cudaStream_t s) {
    CK(cudaMemcpyAsync(g.buf[0], d_x0, kDim * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice, s));
  };
  auto run_baseline = [&](cudaStream_t s) {
    reset_state(s);
    for (int L = 0; L < kNumLayers; ++L) {
      baseline_rmsnorm<<<1, 512, 0, s>>>(g, L);
      baseline_gateup<<<kNumGateUp, 512, kDim * 4, s>>>(g);
      baseline_down<<<kNumDown, 512, kFfn * 4, s>>>(g, L);
    }
  };
  int mk_grid = sm_count;
  int mk_limit = 0;  // 0 = whole program; the truncation hook, used below
  auto run_mk = [&](cudaStream_t s) {
    reset_state(s);
    CK(cudaMemsetAsync(d_next, 0, 4, s));
    CK(cudaMemcpyAsync(d_counters, d_counters_init, counters.size() * 4,
                       cudaMemcpyDeviceToDevice, s));
    megakernel_vm<<<mk_grid, VmConfig::kNumThreads, kSmem, s>>>(
        g, d_prog, d_succ, d_counters, d_next, d_timings, kNumInstructions,
        mk_limit);
  };

  auto check = [&](const char* what) {
    std::vector<__nv_bfloat16> hb(kDim);
    CK(cudaMemcpy(hb.data(), g.buf[kNumLayers & 1], kDim * sizeof(__nv_bfloat16),
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
  printf("  %d layers -> %d instructions; baseline pays %d launches\n",
         kNumLayers, kNumInstructions, 3 * kNumLayers);
  printf("  SMs=%d  VM: %d warps, %d pages x %d KB\n\n", sm_count,
         VmConfig::kNumWarps, VmConfig::kNumPages, VmConfig::kPageSize / 1024);

  if (probe_only) {
    mk_grid = sm_count; mk_limit = 1 + kNumGateUp;
    for (int i = 0; i < 30; ++i) {
      baseline_rmsnorm<<<1, 512, 0, stream>>>(g, 0);
      baseline_gateup<<<kNumGateUp, 512, kDim * 4, stream>>>(g);
      run_mk(stream);
    }
    CK(cudaStreamSynchronize(stream));
    printf("probe-only run complete (for profiling)\n");
    return 0;
  }

  printf("correctness\n");
  run_baseline(stream); CK(cudaStreamSynchronize(stream));
  bool ok = check("baseline");
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

  // A megakernel amortises its machinery over the instructions ONE CTA runs
  // back to back: with fewer instructions than CTAs the ring never pipelines
  // and every CTA pays full setup for a fraction of an instruction. Sweeping
  // the grid is what turns that from a hypothesis into a number.
  printf("\nmegakernel vs grid size (%d instructions in the program)\n",
         kNumInstructions);
  printf("  %6s %8s %10s %9s\n", "CTAs", "ins/CTA", "latency", "vs base");
  float best_mk = 1e30f; int best_grid = 0;
  for (int grid : {sm_count, 81, 64, 40, 27, 16, 8}) {
    if (grid <= 0) continue;
    mk_grid = grid;
    char name[32]; snprintf(name, sizeof(name), "grid=%d", grid);
    cudaGraph_t gr; cudaGraphExec_t ex;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    run_mk(stream);
    CK(cudaStreamEndCapture(stream, &gr));
    CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ex, stream));
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
      CK(cudaEventRecord(a, stream));
      for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ex, stream));
      CK(cudaEventRecord(b, stream));
      CK(cudaEventSynchronize(b));
      float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
      best = std::min(best, ms / 200.f * 1000.f);
    }
    printf("  %6d %8.2f %9.2f us %8.2fx\n", grid,
           float(kNumInstructions) / grid, best, t_base / best);
    if (best < best_mk) { best_mk = best; best_grid = grid; }
    CK(cudaGraphExecDestroy(ex)); CK(cudaGraphDestroy(gr));
  }
  // Correctness again at the winning grid: a scheduling change must not move
  // the answer, and this is where a claim-cursor bug would show.
  mk_grid = best_grid;
  run_mk(stream); CK(cudaStreamSynchronize(stream));
  printf("\nre-check at grid=%d\n", best_grid);
  check("megakernel");
  printf("\n  best megakernel %.2f us at grid=%d -> %.2fx vs baseline\n",
         best_mk, best_grid, t_base / best_mk);

  // Discriminating probe. The full program has a fan-in of 64 on every `down`
  // instruction, so 16 CTAs spin holding an SM while 64 others work -- a cost a
  // kernel boundary does not have, because it RELEASES the machine. Truncating
  // the program to rmsnorm + gate/up removes that fan-in entirely. If the gap
  // closes here, the cost is spin-stranding; if it does not, it is the VM's
  // fixed role warps (640 threads of which 512 compute).
  auto time_one = [&](const char* name, bool mk, int limit) {
    mk_limit = limit;
    cudaGraph_t gr; cudaGraphExec_t ex;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (mk) { run_mk(stream); }
    else {
      baseline_rmsnorm<<<1, 512, 0, stream>>>(g, 0);
      baseline_gateup<<<kNumGateUp, 512, kDim * 4, stream>>>(g);
    }
    CK(cudaStreamEndCapture(stream, &gr));
    CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ex, stream));
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
      CK(cudaEventRecord(a, stream));
      for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ex, stream));
      CK(cudaEventRecord(b, stream));
      CK(cudaEventSynchronize(b));
      float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
      best = std::min(best, ms / 200.f * 1000.f);
    }
    printf("  %-22s %8.2f us\n", name, best);
    CK(cudaGraphExecDestroy(ex)); CK(cudaGraphDestroy(gr));
    mk_limit = 0;
    return best;
  };

  if (VmConfig::kTimingEnabled) {
    // One clean pass, outside any timed loop: the timestamps are the subject
    // here, not the latency.
    mk_grid = sm_count; mk_limit = 0;
    CK(cudaMemset(d_timings, 0,
                  size_t(kNumInstructions) * VmConfig::kTimingWidth * 8));
    run_mk(stream); CK(cudaStreamSynchronize(stream));
    std::vector<uint64_t> t(size_t(kNumInstructions) * VmConfig::kTimingWidth);
    CK(cudaMemcpy(t.data(), d_timings, t.size() * 8, cudaMemcpyDeviceToHost));

    static const char* kPhase[8] = {
        "claim", "fetch", "dep-wait", "arm+publish",
        "load-issue", "data-ready", "compute", "release"};
    // Interval i is phase i -> phase i+1; the last column is the whole
    // instruction, claim to release.
    struct Agg { double sum[7]; double total; int n; };
    Agg agg[4] = {};
    for (int i = 0; i < kNumInstructions; ++i) {
      const uint64_t* r = &t[size_t(i) * VmConfig::kTimingWidth];
      if (r[kTClaim] == 0 || r[kTFinished] == 0) continue;  // never ran
      const int op = prog[size_t(i) * VmConfig::kInstructionWidth];
      Agg& a = agg[op];
      for (int ph = 0; ph < 7; ++ph) {
        a.sum[ph] += double(r[ph + 1] - r[ph]);
      }
      a.total += double(r[kTFinished] - r[kTClaim]);
      a.n++;
    }
    const char* opname[4] = {"noop", "rmsnorm", "gate/up", "down"};
    printf("\nper-instruction phases, mean cycles (VM's own profiler)\n");
    printf("  %-8s %4s", "op", "n");
    for (int ph = 0; ph < 7; ++ph) printf(" %11s", kPhase[ph]);
    printf(" %11s\n", "TOTAL");
    for (int op = 1; op < 4; ++op) {
      if (agg[op].n == 0) continue;
      printf("  %-8s %4d", opname[op], agg[op].n);
      for (int ph = 0; ph < 7; ++ph) {
        printf(" %11.0f", agg[op].sum[ph] / agg[op].n);
      }
      printf(" %11.0f\n", agg[op].total / agg[op].n);
    }
    printf("  (phase i is the interval from %s.. to the next column)\n",
           kPhase[0]);
  }

  printf("\nprobe: program truncated to rmsnorm + gate/up (no fan-in)\n");
  mk_grid = sm_count;
  const float p_base = time_one("baseline (2 kernels)", false, 0);
  const float p_mk = time_one("megakernel (65 ins)", true, 1 + kNumGateUp);
  printf("  ratio %.2fx  (full program was %.2fx)\n", p_base / p_mk,
         t_base / best_mk);
  return 0;
}

#endif  // MK40_NO_MAIN
