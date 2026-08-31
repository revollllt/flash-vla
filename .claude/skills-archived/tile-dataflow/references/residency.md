# Residency — persistent, cooperative, clusters, and the CTA/warp trade

SKILL.md's Phase 1 keeps the three-decision summary (persistent / residency /
clusters) and the one rule `Grid = SM_count × cta_per_sm`. This file is the
detail behind them: when cooperative launch actually pays, why CTA count and
warp count hide *different* things, how to size the producer, and how phase
ordering follows from the tile shape. Read it when the kernel is persistent,
warp-specialized, latency-bound, or clustered — i.e. whenever `cta_per_sm`,
`cooperative`, or the producer width is not obvious.

## Cooperative: one guarantee, three use cases

The guarantee is residency, and nothing else. It matters because the GPU does
not preempt within a kernel: if CTA 500 is unscheduled and CTA 0 spins on it,
the failure is a **hang**, not an error.

At `SM_count × cta_per_sm` residency already holds in practice; cooperative only
makes it *checked* — the launch fails instead of hanging after someone's edit
pushes registers over a threshold. A safety property, not a reason on its own.

| Worth it for | Because two launches cannot |
|---|---|
| **cross-phase state in registers/smem** | a launch boundary destroys both. If an accumulator or resident tile would otherwise round-trip through HBM, a grid barrier keeps it. **Usually the only argument that pays** — cost it in bytes against ~one launch |
| device-decided phase counts | a data-dependent trip count cannot be unrolled into launches without a host round-trip |
| global work stealing | a CTA taking from a queue needs every producer live |

Not worth it for *"fewer launches"* — a grid barrier is *believed* to cost about
what a launch costs `[I, UNMEASURED]`, so only retained state wins. **Check this
before relying on it**: Phase 0's measurement 1 gives the launch cost and 5 gives
the *cluster* barrier, but nothing here has costed a grid barrier. Also not worth
it for *"more persistent"*, a grid-sizing decision and orthogonal.

**Cooperative and clusters are not mutually exclusive — but together they need an
explicit placement check.** A cooperative launch is rejected with
`cudaErrorCooperativeLaunchTooLarge` when its grid exceeds what can be
co-resident, and with a cluster that ceiling is
`cudaOccupancyMaxActiveClusters × cluster_size`, not `SM_count × cta_per_sm`.
Measured here `[MEAS-B]`: cluster 8 at 207360 B gives 15 placeable clusters
against the 16 the grid needed, so the pair was rejected — a failure of *that
footprint*, not a prohibition. Deep pipelines make it the common case on sm90,
so budget for it; at cluster 2 with modest smem the pair launches.
Both are individually capturable into a CUDA graph. Since multicast usually beats retained state, the
default for a static fused chain is still **persistent yes, cooperative no** —
but as a default, not because the hardware forbids the pair.

## CTA count hides barriers; warp count does not

A warp stalled on DRAM is hidden by any eligible warp, including one in the same
CTA. **A warp stalled on a block-level barrier is not** — `__syncthreads`, an
mbarrier wait, a producer/consumer handoff stop *every warp in the CTA at once*.
Only **another CTA**, whose barriers do not align with yours, fills that gap.

So at equal warps per SM, **more CTAs of fewer warps beats fewer CTAs of more
warps** whenever barriers sit on the critical path — which every software-pipelined
fused kernel has, once per stage. Aim for `cta_per_sm` 2 or 4 when latency-bound.

Measured: 1 CTA/SM at 8 warps gave **0.30 eligible warps per scheduler**, top
stall `short scoreboard` at 42.7%. Widening the CTA would not have helped.

**This opposes the warp-specialisation idiom**, and the tension should be
resolved explicitly. Producer/consumer groups want large CTAs and deep pipelines
want large per-CTA smem; both drive `cta_per_sm` to 1. Right at large M, usually
wrong at small M — four 128-thread CTAs at 56 KB can beat one 256-thread CTA at
204 KB.

**Two orthogonal knobs, and they buy different things.** `cta_per_sm` hides
**barriers**, as above. `warps_per_cta` hides **memory latency** — more eligible
warps *between* barriers, which is what covers a DRAM miss. A latency-bound
kernel wants both, and neither is capped by the other: raise total warps toward
the SM's 64 and let the register and smem budgets say where you stop.

H100 per SM: 233472 B smem, 65536 registers, **64 warps / 2048 threads**, 32 CTAs.
The budget is joint, so read the row you want and check all three columns:

| `cta_per_sm` × threads/CTA | warps/SM | regs/thread | smem/CTA |
|---|---|---|---|
| 1 × 256 | 8 | 255 | 227 KB |
| 2 × 256 | 16 | 128 | 114 KB |
| 4 × 256 | 32 | 64 | 57 KB |
| 2 × 512 | 32 | 64 | 114 KB |
| 4 × 512 | 64 | 32 | 57 KB |
| 8 × 256 | 64 | 32 | 28 KB |

`regs/thread = 65536 / (cta_per_sm × threads_per_cta)`, so **the real trade is
warps against accumulator size**: 64 warps/SM leaves 32 registers per thread,
which a 64-f32-per-thread accumulator alone already exceeds. Small-M kernels are
the ones that can afford it — an accumulator of `BLOCK_M×BLOCK_N/threads` is
16 f32/thread at 64×32 over 128 threads, so ~64 regs/thread is comfortable and
32 warps/SM is reachable. Large-tile throughput kernels cannot go there and
should not try.

## Size the producer; do not inherit 128

**wgmma** needs whole 128-thread warp groups — that binds the *math* side.
**TMA does not**: `cp.async.bulk.tensor` is issued by a single elected thread
(`cute::elect_one_sync()`, `sm90_mma_tma_gmma_ss_warpspecialized.hpp:320`), so a
producer can be **one warp**. CUTLASS producers are 128 threads because
`setmaxnreg.{inc,dec}.sync.**aligned**.u32` is warpgroup-granular
(`arch/reg_reconfig.h:76,84`) — a register-redistribution constraint, not a TMA
one. So `threads_per_cta` moves in steps of **32**, with the math side a
multiple of 128.

Two reasons to spend more than one warp on the producer, and you should name
which applies:

- **you want `setmaxnreg`** — to run the producer at 24-40 registers so the math
  groups can hold 232-240. Needs the producer to be a full warp group;
- **the producer does real work** — an in-mainloop transform, a squares
  reduction, smem staging. That is CUDA-core work and wants warps — and it is
  what puts the producer on L3's critical path.

If neither applies, one warp is enough, and the saving is not just 96 threads:
**it converts register budget into CTA count, which is what hides barriers.** At
128 registers per thread, a 256-thread CTA fits 2 per SM (16 warps) while a
160-thread CTA fits 3 (15 warps) — nearly the same warps, but three independent
barrier schedules instead of two. (Registers allocate in units of 8 per thread,
so round down: 65536/(4×160) = 102.4 becomes **96**.)

One cost of a lone producer warp: `setmaxnreg` becomes unusable **for the whole
CTA**, not just the producer — PTX requires every warp of a warpgroup to execute
it, and the register pool is per-CTA.

| variant | threads | warps/SM @4 CTAs | regs/thread @4 CTAs |
|---|---|---|---|
| producer WG + math WG | 256 | 32 | 64 |
| producer warp + math WG | 160 | 20 | 96 |
| producer warp + 2 math WGs | 288 | 36 | 56 |

The costs are real: smaller tiles re-read the shared operand more (price it at
the Phase 0 L2 slope), and less smem caps depth — which only pays to the knee
anyway. A spec that raises either knob without naming what it gave up has made
half a trade.

## Ordering between phases

**The mechanism is a consequence of the tile shape, not an independent choice.**
An unsplit reduction over the full K makes every consumer depend on every
producer — only a grid barrier serves that, and it costs about a launch, so just
launch twice. Split-K makes each consumer depend on `1/splits` of the producers,
and then per-tile semaphores are far cheaper.

Inside a captured graph: a semaphore must be **self-resetting** (replay reruns
identical nodes with identical arguments and there is no host reset in the replay
path, so the last CTA through clears the counters), and anything derived from a
launch-time counter must be a kernel **parameter**, not captured state.

## A static offline schedule

- **No work queue, no atomic.** Claiming a tile is index arithmetic or a
  constant-memory lookup. Say so; a reviewer should not have to guess.
- **The schedule must respect the cluster shape.** Co-clustered CTAs must be
  assigned tiles that actually share the multicast operand. Cluster boundaries
  constrain the offline optimiser; getting it wrong turns multicast into a
  broadcast nobody needs.
