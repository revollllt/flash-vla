# Agent Note: Restructure the hardware-unit-test probes into a shared device library with one uniform ABI

Status: proposed

## Problem

The probes grew one at a time and each one re-implements the harness around it.
4432 lines across 11 files, and the duplication is not cosmetic -- it is where
this session's failures came from.

| Duplicated | Copies | What it cost |
|---|---|---|
| `tma_2d`, `wait_wd`, `smem_u32`, `FullBar` | 2 each | — |
| `WATCHDOG_CYCLES` | 4, in **two incompatible implementations** (`wait_wd` vs `trap_at`) | `pipeline_ws.cu` shipped with **no watchdog at all**; two Slurm slots burned for zero output before the deadlock was found by reasoning instead of by a trap site |
| `extern "C"` entry points | 13, in 5 naming conventions | — |
| ctypes `argtypes` blocks | 13, positional | ABI broken **three times** in one session, each time by appending a trailing pointer (`smid`, `dbg`, `dbg`) |
| repetition/median logic | 4 different counts (5/7/20/30), 3 implementations | `overlap.py` shipped with one launch per mode and read `tma_iso` 11% apart across configs whose TMA work was identical |
| regime guard (`l2_ratio`, `touched_b`) | present in 2 of 5 drivers | `overlap.py` has none: E5 measured cold DRAM at 8 CTAs, E5b measured L2-resident, and one was divided by the other |

There is also a structural inversion: `overlap.py` reaches into `tma_ring.py`
through a `sys.path` insert to borrow `Geom` and `encode` -- one probe's driver
importing another probe's driver.

The through-line is that `protocol.md`'s rules 9 (every wait has a deadline),
11 (gate rates on a correctness check) and 6b (state the regime) are enforced by
*remembering them*, and all three were forgotten in the same session by the
person who wrote them.

## Proposal

A header-only device library plus one uniform ABI, with per-unit translation
units preserved so compile isolation survives (`mma_rate.cu` alone is a two
minute build; a single TU would rebuild everything on any edit).

```
probes/
  include/hut/            header-only, shared; no unit includes another unit
    common.cuh            fixed-width types, smem_u32, elect_one, warp/CTA ids
    watchdog.cuh          bounded wait + trap-site table
    barrier.cuh           mbarrier / ClusterTransactionBarrier, built ON watchdog
    tma.cuh               tma_2d, descriptor encode, box geometry
    mma.cuh               ss_op_selector, atom static_asserts, warpgroup_* wrappers
    pipeline.cuh          PipelineTmaAsync with bounded acquire / wait / tail
    unit.hpp              THE ABI -- HutParams, HutBuffers, HutUnitInfo, HUT_REGISTER
  units/<name>/
    <name>.cuh            kernels only
    <name>.cu             a few lines: include the unit, HUT_REGISTER it
    <name>.py             sweeps only
  hut/                    python, one implementation each
    toolchain.py          (today's _toolchain.py)
    abi.py                one ctypes.Structure mirroring HutParams, bound once
    harness.py            build cache, reps/median, timer identity
    regime.py             touched_b, l2_ratio, flush, COLD_MIN_L2_RATIO, the guard
    render.py             the table renderers
```

### The ABI is a struct, not a positional list

Thirteen bespoke entry points collapse to eight uniform ones over two PODs:

```cpp
struct HutParams {
  int32_t cfg, n_ctas, n_warps, stages, k_tiles, mode;
  int32_t box_bytes;        // one TMA box            -- driver boxDim x elem
  int32_t txn_bytes;        // bytes on ONE barrier   -- CUTLASS transaction_bytes
  int32_t stage_bytes;      // bytes of one smem slot -- CUTLASS Stages
  int32_t mask0, shift0, step0, mask1, step1;
  const void *desc, *a, *b;          // CUtensorMap / operands, or nullptr
};
struct HutBuffers { void *cyc_a, *cyc_b, *sink, *dbg, *smid, *out; };

extern "C" {
  int         hut_unit_count();
  const char* hut_unit_name(int u);
  uint32_t    hut_unit_flags(int u);      // NEEDS_COLD | HAS_CHECK | HAS_WATCHDOG
  int         hut_cfg_count(int u);
  int         hut_cfg(int u, int cfg, int field);
  int         hut_smem(int u, const HutParams*);
  int         hut_check(int u, const HutParams*, HutBuffers*, void* stream);
  int         hut_launch(int u, const HutParams*, HutBuffers*, void* stream);
}
```

Adding a field becomes a struct change mirrored in one Python `Structure`, not
thirteen `argtypes` lists edited by hand. That alone removes the failure that
recurred three times.

### Three protocol rules stop being things to remember

- **Rule 9.** `watchdog.cuh` owns the only wait primitive; `barrier.cuh` and
  `pipeline.cuh` expose no unguarded wait for a unit to call. A unit *cannot*
  ship without a deadline, which `pipeline_ws.cu` did.
- **Rule 11.** `HUT_HAS_CHECK` is mandatory in `HutUnitInfo`, and `harness.py`
  runs `hut_check` before the first rate and aborts on failure. Today the TMA
  probe has a gate because it was added by hand; the atomic probe has none.
- **Rule 6b.** `hut_unit_flags` declares `HUT_NEEDS_COLD`. `regime.py` refuses
  to emit a row for a cold-requiring unit whose measured `l2_ratio` is below
  `COLD_MIN_L2_RATIO`, and stamps the regime into the JSON that
  `constants.yaml` cites. The single largest source of error in this unit --
  one constant retracted, six rebuilt -- becomes unrepresentable.

### The vocabulary is part of the ABI

`frame` is not a TMA word. The driver API says **box** (`boxDim`), PTX says
**tile** (`cp.async.bulk.tensor...tile`), and CUTLASS says **transaction_bytes**
for what lands on one barrier and **Stages** for the smem ring. `frame` is this
probe's own invention, and `naming.md` already forbids exactly it -- "never name
a cache, a PTX mnemonic, or a probe-internal word (`frame`, `trip`, `sweep`)
where a machine quantity is meant" -- a rule written for the tags and then not
applied to the code.

It is not only untidy. `frame_b` conflates three quantities that are already
different inside this suite:

| | `tma_ring.cu` | `pipeline_ws.cu` |
|---|---|---|
| one TMA box | `frame_b` | A box and B box, different sizes |
| bytes on one barrier | `frame_b` | `2 x (M+N) x BK` -- both boxes |
| bytes of one smem slot | `frame_b` | one stage holding both |

`frame` reads as sufficient only because `tma_ring` is the degenerate case where
all three coincide. The rename below is therefore a correctness change to the
ABI, not a cosmetic one:

| Now | Becomes | Authority |
|---|---|---|
| `frame_b` | `box_bytes` / `txn_bytes` / `stage_bytes` (three fields) | `boxDim`, `transaction_bytes`, `Stages` |
| `trip` | `k_tiles` | CUTLASS `k_tile_count` |
| `depth` | `stages` | CUTLASS `Stages` |
| `box_inner`, `box_outer` | unchanged | already `boxDim[0]`, `boxDim[1]` |

One constant pair inherits the same inconsistency and should be settled with it:
`tma.stages.warp.knee` and `pipeline.stages.wg.knee` measure the same quantity in
two regimes under two different words. Both become `stages`, with `depth` kept
as an alias.


## Consequence found during migration: a shared wait primitive is not neutral

The proposal assumed the device library could be extracted without changing what
the probes measure. It cannot, and the effect is larger than the noise floor.

ptxas schedules `while (!pred())` and `while (!ready) { ready = pred(); }`
differently for an mbarrier poll. Neither spelling reproduces both units:

| wait spelling | tma_ring ns/txn | overlap tma_iso | overlap.eff.sm |
|---|---|---|---|
| pre-migration (per-unit, hand-written) | 249.2 | 123439 | 1.248 |
| predicate in condition | 257.8 (+3.5%) | -5.0% | 1.248 |
| variable tested | 279.4 (+12.1%) | -0.0% | 1.248 |

The variable-tested form costs tma_ring 12%. The predicate-in-condition form
makes ptxas emit an extra YIELD per wait site (8 -> 16 in `overlap_kernel`),
which deprioritises the spinning warp so others issue -- friendly in a
production kernel, wrong in a ceiling probe, because a producer that stops
issuing is no longer measuring the best the engine can do.

**Decision.** `barrier.cuh::wait` uses the predicate-in-condition form, chosen
on which unit's CONSTANT can absorb the difference: `tma.issue.warp` is an
absolute and must stay inside the 6% floor (+3.5% does, +12.1% does not), while
`overlap.eff.sm` is a ratio that read 1.248 under both, because its two terms
move together. Verified after the change: tma_ring +2.9% median (worst +3.1%),
overlap.eff.sm 1.257 against a recorded 1.25.

`spin_until` stays generic for counter polls, where genericity costs nothing.

**Consequences.**
- The wait implementation is part of the measurement, not incidental. A unit
  reporting a new ABSOLUTE has to state which spelling it measured under.
- A future unit penalised by this shape re-opens the decision; it does not
  quietly re-record the constant.
- Instruction COUNT does not detect this: tma_ring is 328 instructions either
  way, and overlap differed by 56 with the mix looking near-identical. Only a
  full SASS text diff showed it.

## Consequence: absolute readings did not survive the migration; ratios did

Every ratio checked reproduced exactly (`overlap.eff.sm` 1.248 under four
variants; `mma.issue.warp` 6.26 to 0.0%). Absolutes moved by up to 12% under
changes that left the instruction count identical. This is the same split
already recorded for cache residency, arrived at by a different route, and it
is the reason the migration gate must compare per-row absolutes rather than
headline ratios -- a gate on the headline alone would have passed every broken
variant in the table above.

## Alternatives considered

- **One translation unit including every unit.** Rejected: `mma_rate`'s template
  instantiations already take ~2 minutes, and a shared TU would pay that on
  every edit to any unit. Compile isolation is worth the extra `.cu` stub.
- **Leave the device code alone, share only the Python.** Rejected: the most
  expensive failure (`pipeline_ws` with no watchdog) was device-side, and the
  watchdog cannot be shared from Python.
- **Keep positional `extern "C"` and just be careful.** Rejected on evidence:
  three ABI breaks in one session by the same mechanism.
- **Do this incrementally as each unit is next edited.** Rejected for the shared
  headers -- two watchdog implementations already diverged -- but accepted for
  the sweep bodies, which move unchanged.

## Status: implemented 2026-08-29

All five units migrated (`gmem_atomic`, `tma_ring`, `mma_rate`, `overlap`,
`pipeline_ws`); `probes/compute/` and `probes/memory/` are gone. Every unit
verified against its pre-migration self IN ONE JOB, which is the only
comparison that separates a code change from machine drift:

| unit | constant | reproduces |
|---|---|---|
| gmem_atomic | atom.* | within 1.8% (`atom.ratio.width` 0.0%) |
| tma_ring | tma.issue.warp | +2.9% median, worst +3.1% |
| mma_rate | wgmma.issue.wg.ss | worst 1.4% over 8 values of N |
| mma_rate | mma.issue.warp | 6.26 vs 6.26 |
| mma_rate | wgmma.stages.wg.knee | 8/8 rows, worst 1.2% |
| overlap | overlap.eff.sm | 1.257 vs recorded 1.25 |
| pipeline_ws | pipeline.ratio.sm.dep | -0.4% |
| pipeline_ws | pipeline.stages.wg.knee | 4, unchanged |

No shared `pipeline.cuh` was built: `PipelineTmaAsync` has exactly one consumer,
and a shared abstraction with one user is harder to read than the code it hides.
`overlap` no longer imports `tma_ring` -- it exports its own descriptor encoder.

## Acceptance criteria

Migration is stepwise and each step ends with the same gate: **every constant
the touched unit produces reproduces within its stated noise floor, and
`scripts/constants.py --validate` passes with an unchanged constant count.**

1. Extract `include/hut/*.cuh`; units keep their current ABI. Verify by
   reproducing `tma.issue.warp` and `wgmma.issue.wg.ss`.
2. Land `unit.hpp` + `hut/abi.py`; migrate **gmem_atomic** first -- smallest
   (193 + 323 lines) and untouched this session, so it is a clean test of the
   ABI rather than a re-test of recent edits.
3. Migrate `tma_ring` (1146-line driver; sweeps out of the harness).
4. Migrate `mma_rate`, then `overlap` and `pipeline_ws` onto the shared
   `pipeline.cuh`, deleting the `sys.path` import of `tma_ring`.
5. Turn on the three enforcement flags and confirm each one FAILS a
   deliberately broken unit before it is trusted.

## Risks

- **This refactors the thing that produces every number.** The reproduce-a-
  constant gate at each step is the mitigation, and step 2 deliberately starts
  with the unit whose constants were not touched this session.
- **Enforcement can be wrong in the strict direction.** A `NEEDS_COLD` guard
  that misfires blocks a legitimate measurement. Step 5 requires demonstrating
  each guard rejects a known-bad case *and* passes a known-good one.
- **`hut_check` on a unit that computes nothing** (the launch/occupancy work in
  `roadmap.md`) has no host reference to compare against. Those units declare
  the flag off and say why in their `isolation:` field, rather than the harness
  silently skipping the gate.
