# Pi0.5 action-expert persistent megakernel optimization plan

Status: active plan for the SM90 BF16 profile

## Goal and invariant

The target is one persistent H100 kernel for the fixed Pi0.5 action-expert
FFN shape. The kernel must keep a static per-CTA task list: `blockIdx.x`
selects the descriptor row directly, so no device-side scheduler, atomic work
queue, or CTA work stealing is allowed in the production path. Numerical
parity and graph replay are hard gates for every optimization step.

The current reference geometry is BK=64, BN=32, M_pad=64, 224 threads, and
132 CTAs. This document is a decision/measurement plan; source remains the
authority for exact layouts and ABI fields.

## Hardware denominators

The H100 unit tests establish the following constraints:

- `MMA-RATE`: BF16 `m64n32k16` is about 24.7 cycles/instruction, versus a
  15.3-cycle architectural ideal; `m64n64k16` is about 33.1 cycles versus
  30.7 ideal. Therefore pairing gate/up at N=64 is the preferred compute
  shape, subject to register and occupancy checks.
- `TMA-ISSUE`: the corrected single producer-warp interval is about 248 ns per
  transaction. A stage ring must cover the measured round trip; depth 4 is the
  default starting point.
- `TMA-CEIL`: large cold-DRAM transfers reach about 3.09 TB/s with sufficient
  in-flight bytes. Short kernels should use `max(issue_time, bytes/bandwidth)`
  as the lower bound instead of a datasheet peak.
- `LAUNCH-RAMP` and barrier/counter probes are treated as fixed terms. They
  cannot be amortized away by adding more CTAs to a single fixed-size request.

## Phased work

### Phase 0 — freeze the contract and measurements

Record the exact packed gate/up weight layout, descriptor ABI, stream semantics,
SMEM budget, and current benchmark command. Run `gu`, `dr`, `full`, replay, and
TileLang composition in one process. Capture clocks, CUDA, Torch, GPU, and
artifact paths. Do not change geometry in this phase.

### Phase 1 — readable CuTe structure (landed)

Use `GatedUp` and `DownResidual` in implementation names. Keep `gu`/`dr` only
as compatibility mode strings. Isolate `TaskDescriptor`, `WarpRoles`, typed
barrier-ring views, and GEMM traits in headers. Reuse FlashMLA's `sm90::gemm`
wrapper for fence/arrive/commit/wait choreography. Keep the static task table,
224-thread role assignment, and C ABI unchanged.

Acceptance: nvcc build, all parity modes, replay, and same-process benchmark.

### Phase 2 — planner proof before kernel changes

Build an offline planner that emits one descriptor row per CTA and proves:

1. every task has a unique owner and a bounded slot;
2. dependency counters have a statically known producer count;
3. producer and consumer stage rings do not alias while a TMA or WGMMA is live;
4. the SMEM/TMA/warp/register budget is valid for the selected profile; and
5. the schedule has no runtime scheduler path.

The planner should use DeepGEMM-style names (`m_block_idx`, `n_block_idx`,
`current_iter`, `kNum*`) but replace its runtime scheduler with precomputed
per-CTA lists. Emit a machine-readable plan plus a validator report.

### Phase 3 — isolate the FFN experiments

Measure one change at a time in this order:

1. GatedUp N=64 paired WGMMA versus the N=32 two-accumulator form;
2. BK=128 with stage-depth and register/occupancy checks;
3. TMA producer/consumer overlap and barrier depth 3/4/5;
4. DownResidual split-K S=2/4 and fixed-order partial reduction;
5. epilogue vectorization, scale placement, and packed-weight swizzle; and
6. task-table ordering for L2 reuse and wave balance.

Every candidate gets a separate generated source/hash and reports achieved
MMA issue rate, TMA delivery, active warps, registers, SMEM, and end-to-end
latency. A candidate is rejected if it improves an isolated stage but loses the
full fused path or violates replay/parity.

### Phase 4 — planner-generated persistent launch

Replace hand-authored table construction with the validated offline artifact.
The launch remains `grid=(132,1,1)` and each CTA reads only its own row. Add
sentinel and dependency diagnostics in debug builds; benchmarks use the same
schedule with diagnostics disabled. No central scheduler is introduced.

### Phase 5 — hardware-limit closure

Use the unit-test denominators to classify the remaining gap as MMA issue,
TMA issue, DRAM delivery, launch/barrier overhead, or under-filled waves. Only
then tune CTA count, task order, stage depth, and register budget. The target is
to approach the measured lower bound without trading away readability or the
static-planner invariant.

## Review gates

Each phase is a separate commit/PR slice. Required evidence is the exact build
and benchmark command, parity/replay output, and a short before/after table.
Performance claims must include the hardware-unit-test tag used as the
denominator; otherwise the result is marked exploratory.
