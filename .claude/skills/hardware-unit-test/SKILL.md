---
name: hardware-unit-test
description: Hardware unit tests that measure what this GPU actually costs, and the consultable table of constants they produce — TMA delivery rate, latency and the CTA/warp/frame saturation frontier; global-atomic throughput by width, scope, placement and contention; gmem-counter arrive→observe latency; wgmma issue rate against N, pipeline depth and whether a second math warpgroup helps; launch and grid-ramp cost, streaming bandwidth ceilings, cluster barrier and placement limits, occupancy. Use whenever a kernel decision needs a machine number rather than a datasheet one: how many CTAs or SMs it takes to saturate bandwidth, how big one TMA must be, how deep a ring buffer should go, whether to accumulate with atomics or a second kernel, how to lay out a reduction, what tile N a wgmma needs or how deep its pipeline must be, whether a small tile should use mma.sync instead of wgmma, what a counter hop or a cluster_sync or a launch costs, whether a floor or roofline target is reachable, or when asked to measure/microbenchmark/calibrate a GPU primitive. Also read it before writing a new microbenchmark, so the result lands as a falsifiable constant rather than a number in a log.
---

# Machine Limits — measure the machine, then design against it

A kernel target derived from a datasheet peak cannot be missed, because it was
never reachable. Two of three kernel targets in this repo were once set *below
their own floors* by dividing bytes by 3.35 TB/s. This skill exists so that
never happens again: it holds **hardware unit tests** — probes that isolate one
engine and measure it — and the **constants** they produce, each with the range
it holds over and the sweep that would have refuted it.

## Layout — one axis is the category, the other is the machine

**Nothing above an `<arch>/` directory is architecture-specific, and everything
inside one is.** Porting is "add `sm120/`, re-run the probes", not "edit every
file and hope".

```
references/     WHAT to measure and how -- arch-independent
  protocol.md             what makes a probe a unit test, not a benchmark
  category-memory.md      访存 -- copy engine, atomics, caches, address maps
  category-compute.md     计算 -- tensor cores, CUDA cores, instruction choice
  category-execution.md   执行 -- launch, occupancy, clusters, barriers, counters
probes/         the experiments, filed by category
  memory/   tma_ring, gmem_atomic
  compute/  mma_rate
sm90/           THE RESULTS for one machine
  README.md         the arch index: machine, toolchains, headline results
  constants.yaml    the extracted claims -- read via scripts/, not by eye
  unit-*.md         narrative, sweep tables, design rules, open gaps
scripts/        design-time query tools -- no GPU, no torch
```

A number that is a **claim** lives in `<arch>/constants.yaml`; a number that is
**evidence** lives in the unit reference beside the rows it came from. The
scripts read the YAML, so nothing is transcribed twice.

## Consulting it — the common case

You are choosing a tile, a grid, or a ring depth and need a real number. Start
at `sm90/README.md` for the map, then:

```bash
python3 scripts/constants.py                    # every constant, one line each
python3 scripts/constants.py --tag TMA-ISSUE    # one, with what would refute it
python3 scripts/frontier.py --table             # the saturation frontier, both ways
python3 scripts/frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304
```

Then **cite the tag in the spec** — `[TMA-ISSUE]`, `[BW-CEIL]` — so a reviewer
can trace the floor to a job id. A measured constant retires an `[I]`.

Read `--tag` output before spending a constant, not after. Every one carries a
`valid:` range, and quoting it outside that range is the mistake this format was
built to catch: `TMA-ISSUE` is 270 ns *while fewer than ~256 KB are in flight*,
and the same probe reads 2863 ns where bandwidth binds instead.

## What is measured, and what is not

| Unit | Status | Answers |
|---|---|---|
| **tma** | measured | delivery rate per producer warp, round-trip latency and the ring depth that covers it, box geometry, descriptor caps, the CTA × warp × frame saturation frontier, the per-CTA 133 GB/s ceiling and what a second producer warp is worth, the 3.02 TB/s steady-state ceiling |
| **launch** | measured | grid ramp per launch, `1.85 + MB/2.77` cold-read model, bandwidth vs CTA count, cluster barrier cost and placement limits, occupancy ≠ residency |
| **atomic** | measured | atomic throughput by instruction, width, scope, placement and sharing; the 6.3× layout lever; gmem-counter hop latency and why observers are free |
| **mma** | measured | both tensor-core instructions. `wgmma`: cycles per instruction against N, the in-flight depth and `wait_group` setting that fill the pipeline, whether a second math warpgroup buys anything, the clock under sustained load, the bytes-per-ns ratio against the copy engine. `mma.sync`: issue rate, latency, the accumulator and warp counts that saturate it, and the tile size at which the two cross over |

All four units are measured. That does not mean they are complete: **each
unit's reference ends with its own open gaps**, and the largest one is named
below. `scripts/constants.py --validate` prints a `GAP` line for any unit whose
status is still `unmeasured`.

**The biggest thing still untested:** `MMA-VS-TMA` says a CTA needs ~4.2
producer warps per math warpgroup, but it is *arithmetic over two constants
measured in separate kernels*. Whether the copy engine and the tensor core
actually run concurrently at those rates — rather than contending — is the
assumption every fused kernel rests on, and no probe here has run them
together.

### Headline results

```
delivered = min( n_ctas × per_cta ,  curve(product) )
per_cta   = min( n_warps × frame / 270 ns ,  133 GB/s )
```

Aggregate TMA bandwidth is a function of the **product `n_ctas × n_warps ×
frame_bytes`** and nothing else — 8 CTAs × 32 KB and 264 CTAs × 1 KB land on one
curve, within 6.9% across a 132× range, and 22 matched `(N × 2 warps)` vs
`(2N × 1 warp)` pairs agree within 3.4%. **But one CTA saturates at ~133 GB/s**,
which a single producer warp at the 32 KB descriptor cap already reaches 91% of
— so the second producer warp is worth ~10% and the third nothing. See
`sm90/unit-tma.md`.

**Atomics:** address layout is worth **6.3×** and every other lever ≤1.3×. The
unit is per-transaction, so `red.global.add.v4.f32` moves 3.8× the bytes for
free; scope is free; one contended address is 386× down. A gmem-counter hop is
~650 ns and **observers are free** — one counter can gate the whole machine.

**Tensor core:** `wgmma` N must be **≥ 64** (at N=8 it runs at 20% of peak);
four instructions in flight with `wait_group ≥ 1` (never `wait_group 0`, which
costs 20–30%); and **one warpgroup already saturates it** — a second exactly
doubles per-instruction cycles. **Below tile N = 32 use the warp-level
`mma.sync` instead — worth up to 3.1×** — though its own ceiling is 63% of peak
against wgmma's 95%. The clock sits at 1.05–1.58 GHz under load, so the
practical bf16 ceiling is ~850 TFLOP/s, not the datasheet's 989.

## Running a probe

Probes need a GPU; scripts do not. On this cluster:

```bash
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/memory/tma_ring.py \
    --sweeps A,E,F --json profiles/hardware-unit-test/tma_frontier.json
python3 scripts/curve_from_json.py profiles/hardware-unit-test/tma_frontier.json --yaml
```

Then write the constants into `<arch>/constants.yaml` **with the job id**, and
run `python3 scripts/constants.py --validate`.

Read `references/protocol.md` first if you are writing or extending a probe. Its
nine rules are what separate a unit test from a benchmark; the two that get
skipped most are **"a probe constant is a ceiling, not a prediction"** (a real
kernel gets at most the constant, so the gap is a quantity to explain, never
evidence the constant is wrong) and **"design the decisive pair first"** (a
sweep that could not have come out differently measured nothing).

## Porting to another machine

Constants **do not transfer** — not across GPUs, not across toolchains, and not,
for a *slope* measured on a kernel, across kernel bodies.

1. Read the three `references/category-*.md` files. They are the questions,
   with none of this machine's answers in them.
2. `mkdir <arch>/`, re-run the probes, write `<arch>/constants.yaml`. Any
   directory holding a `constants.yaml` is discovered automatically, and
   `--machine <arch>` selects it.
3. Write `<arch>/unit-*.md` and an `<arch>/README.md` index.
4. Where a probe's PTX is arch-specific, branch inside it rather than forking
   it — the sweep design is the portable part and is worth keeping shared.

Every number in `sm90/` is wrong for another architecture until re-measured, and
the category files mark which of them were *surprises* here and should therefore
be re-tested rather than assumed.

## Related skills

- **`tile-dataflow`** — Phase 0 *is* this skill. Run these probes, cite the tags
  in the spec's `toolchain` block, and every floor traces to a measurement.
- **`benchmark-kernel`** — measures *a kernel*. This skill measures *the
  machine*. Use its harness inside a probe (the probes prefer its CUPTI timer
  and say so when they fall back to CUDA events).
- **`megakernel-taskgraph`** — its counter-ordering protocol now has its
  constant: `ATOM-HOP`, ~650 ns per arrive→observe, with observers free.
- **`gpu-profiler-analysis`** — explains where a kernel's time went. This skill
  says where it *could* have gone.

## Files

| Path | Read when |
|---|---|
| `sm90/README.md` | **start here** — the machine, its toolchains, and the headline result per unit |
| `sm90/constants.yaml` | the source of truth; read via `scripts/constants.py` rather than by eye |
| `references/protocol.md` | writing or extending a probe, or judging whether a number is usable |
| `references/category-memory.md` | adding or porting a memory-side unit — the questions without this machine's answers |
| `references/category-compute.md` | adding or porting a compute unit |
| `references/category-execution.md` | adding or porting a launch / occupancy / sync unit |
| `sm90/unit-tma.md` | choosing a tile, grid, ring depth, or `BK`; anything TMA-fed |
| `sm90/unit-launch.md` | fusion decisions, grid sizing, clusters, occupancy |
| `sm90/unit-atomic.md` | split-K reduction, histogram-shaped accumulates, or the megakernel counter protocol |
| `probes/memory/gmem_atomic.{cu,py}` | re-measuring the atomic unit, or adding a sweep to it |
| `sm90/unit-mma.md` | choosing a wgmma N, a pipeline depth, or the number of math warpgroups; checking an overlap claim by arithmetic |
| `probes/compute/mma_rate.{cu,py}` | re-measuring the tensor core, or adding a sweep to it |
| `scripts/constants.py` | consult or validate the constants |
| `scripts/frontier.py` | "how few CTAs", "how small a TMA", "what is my copy floor" |
| `scripts/curve_from_json.py` | after a probe run, to reduce raw sweeps to a curve |
| `probes/memory/tma_ring.{cu,py}` | re-measuring the TMA unit, or adding a sweep to it |
