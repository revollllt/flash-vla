# Minimal FFN persistent task-loop (H100)

This is the first executable megakernel slice for Pi0/Pi0.5. It deliberately
implements only the gated FFN, while keeping the mechanisms needed by the
future decoder task graph.

## Fixed contract

- H100 SM90A, launch grid: **132 CTA**.
- CTA 0..127: one GU tile each (128 GU tiles total).
- CTA 0..31: a second, dependency-gated DR tile in slot 1.
- CTA 128..131: sentinel-idle workers in this minimal schedule.
- The task list is preallocated, but a CTA advances through its slots after
  the previous body and CTA barrier complete; this is persistent worker
  reuse without global atomics or runtime stealing.
- Task table shape: `int32[132, 2, 4]`.
- Task descriptor: `{type, n, counter, pad}`.
- Four GU tasks release each DR counter.
- BF16 mainloop K tile: `BK=64`; GU uses 16 K stages and DR uses 64.
- Logical tiles: `A/hidden=64x64`, `W=32x64`, output `M×N=64x32`.
- Phase 1 launch geometry: **192 threads/CTA**. Warps 0..3 are the existing
  WGMMA math warpgroup; warp 4 owns weight TMA (GU: W1/W2 + scales, DR: Wd)
  and warp 5 owns activation TMA (GU: x, DR: gated hidden). The 132-CTA grid,
  task table and counter ABI are unchanged.
- Counter reset is a stream operation before every launch, so CUDA Graph
  replay starts from a clean dependency state.

## Kernel structure

```text
for slot in per_cta_task_lists[blockIdx.x]:
    if slot is GU:
        execute GU body
        release successor counter
    else:  # DR slot 1 on rows 0..31
        wait(counter)
        execute DR body
```

The GU and DR bodies retain their existing mbarrier rings and WGMMA math. In
Phase 1, TMA issue is split across two producer warps that carry the same
role in both task types: warp 4 issues the weight (and scale) copies, warp 5
issues the activation copies. Every TMA warp enters the body as a full warp
and wraps its TMA issue with `cute::elect_one_sync()`, so exactly one lane
emits each copy. For DR-W, the election also owns `arrive_and_expect_tx`; the
other lanes only participate in the common wait path. A small per-slot sequence handoff makes the GU A-TMA
warp publish the transaction-barrier setup before the W-TMA warp attaches its
copies. The DR slot still waits on the original four-GU counter, so the dependency is
not removed: rows 0..31 run GU first, publish the counter, then enter DR. The
slot seam explicitly drains the remaining mbarrier phases before the aliased
shared-memory pool is reused. This version does not yet solve variable
pipeline depth, CTA-local A reuse, dynamic planning, multi-stream VLM/action
overlap, or fine-grained L2 prefetch.

## Validation

`eval/correctness/pi05/ffn_taskloop_parity.py` checks GU-only, DR-only and
full-graph parity, then replays the full schedule with fresh residual/counter
state. `--bench` compares the 132-CTA fused launch with the existing TileLang
two-kernel composition after parity passes.

## Phase 1 result (BK=64, unfixed clocks)

Compared with the single-producer 160-thread baseline, Phase 1 passed parity
and replay and measured:

| variant | fused | GU-only | DR-only |
| --- | ---: | ---: | ---: |
| 160-thread baseline | 84.63 us | 48.35 us | 40.87 us |
| 192-thread split TMA | 79.54 us | 51.10 us | 32.04 us |

The split producer improves the dependency-gated DR path substantially, while
the GU path regresses slightly because its shared transaction barrier needs an
additional A-to-W handoff. The next phase should add a scheduler/lookahead
queue and measure whether that handoff can be hidden rather than immediately
increasing ring depth.

## Phase 2 result (BK=64, unfixed clocks)

Phase 2 uses **224 threads/CTA**: warps 0..3 remain the WGMMA math group,
warps 4 and 5 issue A/W TMA, and warp 6 owns slot reservation and dependency
publication. The scheduler configures transaction barriers before publishing
per-slot sequences to the TMA warps. DR counter polling is now performed by
the scheduler, so the hidden-A TMA warp only consumes ready sequence entries.
Ring depths, task clustering and counter granularity are unchanged.

| variant | fused | GU-only | DR-only |
| --- | ---: | ---: | ---: |
| 160-thread single producer | 84.63 us | 48.35 us | 40.87 us |
| 192-thread split TMA | 79.54 us | 51.10 us | 32.04 us |
| 224-thread scheduler | 78.04 us | 51.08 us | 31.47 us |

Phase 2 preserves parity/replay and improves the fused path another 1.9% over
Phase 1. The GU path remains limited by its four WGMMA stages and the current
serialized task order; the next change should be pipeline-depth/lookahead
tuning rather than adding more CTA threads.

## Phase 3 depth probe

With the Phase 2 scheduler kept unchanged, `GU_DEPTH=6` was compared with the
canonical `GU_DEPTH=4` using the same BK=64 kernel and 50 timed repetitions:

| GU depth | fused | GU-only | DR-only |
| ---: | ---: | ---: | ---: |
| 4 (kept) | 77.29 us | 50.46 us | 31.11 us |
| 6 (probe) | 77.97 us | 49.29 us | 31.07 us |

The per-path differences are within clock/noise variation and the fused result
favors depth 4. The canonical implementation therefore keeps depth 4 while
the next optimization targets a real lookahead queue and CTA-local activation
reuse instead of allocating more shared-memory stages.

## Dependency-aware static placement probe

A 96-GU-CTA/32-DR-CTA probe was run while preserving single-type dispatch. It
did not improve latency because 32 of the GU CTAs still carried two GU tasks;
those dual-task CTAs remained on the critical path while the 64 single-task
CTAs finished early. Reducing this tail requires a phase transition that lets
a CTA claim another task after its first task.

## Phase 4 preallocated worker queue

The first worker-queue schedule gives every GU tile its own initial CTA row,
then reuses rows 0..31 for the dependent DR tiles. This removes the 32 dual-GU
CTA tail while retaining the exact `GU -> counter[0..31] -> DR` protocol. Each
CTA initializes both barrier pools once, executes slot 0, drains its ring tail,
hits a CTA barrier, and executes slot 1 if present. No global queue or atomic
work stealing is needed because the FFN DAG and task count are known offline.

| variant | fused | GU-only | DR-only |
| --- | ---: | ---: | ---: |
| Phase 3 static placement | 77.29 us | 50.46 us | 31.11 us |
| Phase 4 worker queue | **59.71 us** | **28.60 us** | **34.33 us** |

Phase 4 passed GU/DR/full parity and replay×3. The fused path is 22.7% below
the Phase 3 placement and the GU critical path is 43.3% lower because all 128
GU tiles start concurrently. DR-only is slightly slower from the one-task
worker layout and explicit seam drain; the next step is to tune queue mapping
and overlap DR with the GU tail without changing dependency semantics.

## Phase 5 producer-owned mbarrier protocol

The scheduler-to-TMA sequence handoff was removed from the hot path. For GU,
the A and W producer warps own the same transaction barrier directly:

```text
full[stage].init(2)
empty[stage].init(kNumEpilogueWarps)

producer A: __syncwarp(); elect; full.arrive_and_expect_tx(A_bytes); tma_issue(A)
producer W: __syncwarp(); elect; full.arrive_and_expect_tx(W_bytes); tma_issue(W1/W2/S)
math warp:  full[stage].wait(phase); ...; empty[stage].arrive()
```

The four math warps each contribute one empty-barrier arrival. DR uses the same
producer-owned structure for its independent W and dependency-gated A rings;
the A producer performs the counter poll and TMA issue in one warp, avoiding a
cross-warp sequence publication and its block fence. `cute::elect_one_sync()`
selects the sole lane that emits TMA and performs each producer arrival.

| variant | fused | GU-only | DR-only |
| --- | ---: | ---: | ---: |
| Phase 4 worker queue + elected TMA | 59.40 us | 28.55 us | 34.23 us |
| Phase 5 producer-owned barriers | **51.54 us** | **21.74 us** | **32.83 us** |

Phase 5 passed GU/DR/full parity and replay×3. Relative to Phase 4, fused
latency falls by 13.2%, and the shared-memory command/sequence metadata is no
longer needed. The current implementation keeps the 132-CTA preallocated task
table; the next step is to use the SGLang references in `third_party/sglang`
to prototype a richer warp-specialized queue rather than reintroducing a
cross-warp scheduler handoff.

## SGLang reference checkout

The shallow checkout is pinned to the current SGLang main commit and is kept
as a read-only implementation reference. Relevant files are:

- `third_party/sglang/python/sglang/kernels/jit/include/sgl_kernel/mbarrier.cuh`
- `third_party/sglang/python/sglang/kernels/jit/include/sgl_kernel/warp.cuh`
- `third_party/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`

## Phase 6 gate/up interleave and activation-residency probe

Gate/up weights are now interleaved offline as one blocked row
`[W1_tile(32), W2_tile(32)]`. The GU W producer issues one 128B TMA per K
stage, and the WGMMA path consumes the two halves through one SW128 shared
frame. Correctness and replay passed, but the 50-repetition benchmark was
essentially flat:

| variant | fused | GU-only | DR-only |
| --- | ---: | ---: | ---: |
| Phase 5 producer-owned | 51.54 us | 21.74 us | 32.83 us |
| Phase 6 interleaved W1/W2 | 52.00 us | 21.87 us | 33.32 us |

This confirms that the current bottleneck is not the number of gate/up TMA
instructions; it is the WGMMA/CTA dependency critical path.

An activation-resident probe paired adjacent GU tiles on one CTA and appended
the corresponding DR task, so one 64-column hidden-K frame could remain in
shared memory and bypass one hidden TMA. It passed parity/replay, but fused
latency regressed to **68.66 us** (GU-only **37.58 us**) because GU parallelism
dropped from 128 workers to 64. This mapping is therefore not the default.
The correct next design is a finer split-K DAG: keep all GU producers resident,
emit activation-tile tasks into a preallocated queue, and let DR partial tasks
consume those tiles without requiring one CTA to own the full 4096-wide hidden
reduction. That requires an explicit partial-output/reduction protocol rather
than simply adding a third per-CTA slot.

## Phase 7 split-K queue schema (planner landed, kernel not enabled)

The offline planner now emits a separate ``int32[132, 34, 6]`` queue format.
It is intentionally not accepted by the current C ABI; this prevents an
experimental descriptor from being mistaken for the validated ``[132, 2, 4]``
table.

For hidden tile ``h`` (32 columns), the owning worker receives:

```text
slot 0:       GU(h)                         # produces H[:, h*32:(h+1)*32]
slots 1..32:  DR_PARTIAL(h, output_tile=d)   # d = 0..31, local H stays in SMEM
slot 33:      REDUCE(output_tile=d)          # workers 0..31 only
```

There are 128 GU nodes, 4096 partial nodes and 32 reductions. Every reduction
waits for 128 partial arrivals and writes the residual/gated output only after
an ordered FP32 accumulation. The planner also records the future resource
contract (four math warps, two TMA warps, one scheduler warp, a 64x32 BF16
local activation frame, and a global FP32 partial buffer of about 32 MiB).

This queue is the first version that preserves all 128 GU producers while
giving each hidden tile a local consumer list. The remaining implementation
work is deliberately explicit: add a partial-output pointer to the C ABI,
implement a BK=32/16 WGMMA partial body, publish per-output partial counters,
and add deterministic reduction tasks. Until those pieces are present, the
Phase 6 interleaved worker queue remains the production default.
