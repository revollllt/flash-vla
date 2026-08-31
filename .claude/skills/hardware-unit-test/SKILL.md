---
name: hardware-unit-test
description: Hardware unit tests that measure what this GPU actually costs, and the consultable table of constants they produce — TMA delivery rate, latency and the CTA/warp/box saturation frontier; global-atomic throughput by width, scope, placement and contention; gmem-counter arrive→observe latency; wgmma issue rate against N, pipeline stages and whether a second math warpgroup helps; launch and grid-ramp cost, streaming bandwidth ceilings, cluster barrier and placement limits, occupancy. Use whenever a kernel decision needs a machine number rather than a datasheet one: how many CTAs or SMs it takes to saturate bandwidth, how big one TMA must be, how deep a ring buffer should go, whether to accumulate with atomics or a second kernel, how to lay out a reduction, what tile N a wgmma needs or how deep its pipeline must be, whether a small tile should use mma.sync instead of wgmma, what a counter hop or a cluster_sync or a launch costs, whether a floor or roofline target is reachable, or when asked to measure/microbenchmark/calibrate a GPU primitive. Also read it before writing a new microbenchmark, so the result lands as a falsifiable constant rather than a number in a log.
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
  naming.md               the tag grammar every constant is filed under
  vocabulary.md           every identifier's authority -- driver API, PTX, CUTLASS
  roadmap.md              units not yet measured, and the pairs that settle them
  category-memory.md      访存 -- copy engine, atomics, caches, address maps
  category-compute.md     计算 -- tensor cores, CUDA cores, instruction choice
  category-execution.md   执行 -- launch, occupancy, clusters, barriers, counters
probes/         the experiments -- one directory per unit over a shared library
  include/hut/    the device library: common, watchdog, barrier, tma, mma
  include/hut/unit.hpp  THE ABI -- nine symbols every unit exports
  hut/            the host side: abi, harness, regime, tma, toolchain
  units/<name>/   <name>.cuh (kernels), <name>.cu (ABI), <name>.py (sweeps)
                  tma_ring, gmem_atomic, mma_rate, overlap, pipeline_ws
sm90/           THE RESULTS for one machine
  README.md         the arch index: machine, toolchains, headline results
  constants.yaml    the extracted claims -- read via scripts/, not by eye
  unit-*.md         narrative, sweep tables, design rules, open gaps
scripts/        design-time query tools -- no GPU, no torch
```

`<arch>/constants.yaml` is a RESULTS TABLE -- value, the condition it holds
under, the one-line answer, the rule to apply, and nothing else. The evidence
that justifies a number -- what was held fixed, what would have refuted it, which
job produced it -- lives in the unit reference beside the rows it came from. The
scripts read the YAML, so nothing is transcribed twice.

## Consulting it — the common case

You are choosing a tile, a grid, or a ring's stage count and need a real number. Start
at `sm90/README.md` for the map, then:

```bash
python3 scripts/constants.py                    # every constant, one line each
python3 scripts/constants.py --tag tma.issue.warp    # one, with what would refute it
python3 scripts/frontier.py --table             # the saturation frontier, both ways
python3 scripts/frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304
```

Then **cite the tag in the spec** — `[tma.issue.warp]`, `[ld.bw.dev.dram]` — so a reviewer
can trace the floor to a job id. A measured constant retires an `[I]`.

Read `--tag` output before spending a constant, not after. Every one carries a
`valid:` range, and quoting it outside that range is the mistake this format was
built to catch: `tma.issue.warp` is 248 ns *while fewer than ~256 KB are in flight*,
and the same probe reads 2863 ns where bandwidth binds instead.

## What is measured, and what is not

| Unit | Status | Answers |
|---|---|---|
| **tma** | measured | delivery rate per producer warp, round-trip latency and the ring stages that cover it, box geometry, descriptor caps, the CTA × warp × box saturation frontier, what a second producer warp is worth, the 3.17 TB/s steady-state ceiling |
| **launch** | measured | grid ramp per launch, `1.85 + MB/2.77` cold-read model, bandwidth vs CTA count, cluster barrier cost and placement limits, occupancy ≠ residency |
| **coop** | measured | cooperative launch: what one `grid_sync` costs (1.09 µs, flat in grid size), the co-resident block limit and that it is exactly the occupancy query, and whether a grid barrier is cheaper than the relaunch it replaces (1.29×, so barely) |
| **atomic** | measured | atomic throughput by instruction, width, scope, placement and sharing; the 6.3× layout lever; gmem-counter hop latency and why observers are free |
| **mma** | measured | both tensor-core instructions. `wgmma`: cycles per instruction against N, the in-flight stages and `wait_group` setting that fill the pipeline, whether a second math warpgroup buys anything, the clock under sustained load, the bytes-per-ns ratio against the copy engine. `mma.sync`: issue rate, latency, the accumulator and warp counts that saturate it, and the tile size at which the two cross over |

All four units are measured. That does not mean they are complete: **each
unit's reference ends with its own open gaps**, and the largest one is named
below. `scripts/constants.py --validate` prints a `GAP` line for any unit whose
status is still `unmeasured`.

**That gap is now closed, and the answer was no.** `overlap.eff.sm` ran a TMA
producer and a wgmma consumer in one CTA, uncoupled: **TMA issue slows 1.25×**
and wgmma slows 1.05×, flat across N, NGROUP and producer count. So
`wgmma.bytes.wg.tma` — which used to read "~4.2 producer warps per math
warpgroup" — is now stated as what it always was, **~36 KB of in-flight
`num_producers × box`**, with that contention included. The warp count was never the
quantity: at an 8 KB box it is 4.5 warps and at the 32 KB descriptor cap it is
1.1, and those are the same constant. The pipelining cost is now measured too
(`pipeline.ratio.sm.dep`, E5b, CUTLASS `PipelineTmaAsync`): the barrier round
k_tile_count adds only **~1.07×** on top of contention, while **ring stages** dominates —
two stages cost 2.39× and three already match six.

### Headline results

**Everything here is stated per SOURCE.** The same probe on the same machine
gives 3.17 TB/s cold and 18.3 TB/s cache-hot, and constants that did not say
which they measured were the single largest source of error in this unit — one
several were rebuilt on 2026-08-29.

```
per-warp issue = 248 ns per TMA, SOURCE-INDEPENDENT   [tma.issue.warp]
delivered      = num_producers × box / 248 ns, until a ceiling binds
ceiling        = 3.17 TB/s cold  |  18.3 TB/s from L2
```

There is **no per-CTA bandwidth ceiling** — the issue interval stays flat while
delivery rises linearly to at least 40 KB per CTA. Ring stages is likewise per-source:
**4 stages for DRAM, 2 for L2**. The product law holds to ~10% at the 90th
percentile but has a 19–29% tail driven by CTA count, so use it to size a
configuration and then measure the one you chose. See `sm90/unit-tma.md`.

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

Probes need a GPU, `nvcc`, torch and CUTLASS headers; the `scripts/` need none
of those. Anywhere:

```bash
python3 probes/units/tma_ring/tma_ring.py --sweeps A,E,F --json /tmp/tma.json
python3 scripts/curve_from_json.py /tmp/tma.json --yaml
```

On a Slurm cluster, put that behind whatever launcher the host repo uses — in
this one:

```bash
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/tma_ring/tma_ring.py \
    --sweeps A,E,F --json profiles/hardware-unit-test/tma_frontier.json
```

Then write the constants into `<arch>/constants.yaml` **with the job id**, and
run `python3 scripts/constants.py --validate`.

### What the skill needs from its host

It is a self-contained plugin: copy the directory into any repo and it works.
Nothing outside it is imported, and no path to one machine is compiled in.
Three environment variables, each discovered when unset and each **failing with
the variable's name** rather than falling back to a guess:

| Variable | Used for | If unset |
|---|---|---|
| `CUDA_HOME` | `-L$CUDA_HOME/lib64/stubs -lcuda` | derived from `nvcc` on PATH, then `/usr/local/cuda` |
| `CUTLASS_DIR` | `cute/` and `cutlass/arch/` headers | a few conventional locations, then an error naming the variable |
| `HW_UNIT_TEST_CACHE` | where built `.so` files land | the host repo's `.cache/cuda_ext`, else `~/.cache/hardware-unit-test` |

Pinning `CUDA_HOME` and `CUTLASS_DIR` explicitly is the **supported** path, not
a fallback: `provenance.toolchain` has to record what actually built the probe,
and a discovered toolchain is one that can change under you between jobs.

Built artefacts never land inside the skill directory. A stale `.so` there would
be copied along with the skill to the next machine, where it is a binary for the
wrong GPU.

Two optional couplings, both of which degrade loudly rather than silently:

- **The timer.** Probes prefer the host repo's CUPTI harness
  (`flash_vla.bench.bench_gpu_time` here) and fall back to CUDA events, printing
  which one they used. The two disagree by the launch overhead events include,
  so a number taken under one is not comparable with a constant recorded under
  the other — hence rule 8, and hence the fallback announces itself.
- **The alias scan.** `constants.py --validate` reports files in the enclosing
  git repo that still cite a retired tag. Standalone, there is no repo and it
  reports nothing, which is correct rather than degraded.

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

All optional — nothing here imports them, and the skill is complete without any
of them installed.

- **`benchmark-kernel`** — measures *a kernel*. This skill measures *the
  machine*. Use its harness inside a probe (the probes prefer its CUPTI timer
  and say so when they fall back to CUDA events).
- **`megakernel-taskgraph`** — its counter-ordering protocol now has its
  constant: `atom.lat.dev.hop`, ~650 ns per arrive→observe, with observers free.
- **`gpu-profiler-analysis`** — explains where a kernel's time went. This skill
  says where it *could* have gone.

## A note on the examples

The narrative in `references/` and `<arch>/` cites concrete kernels — an FFN
task-loop, a qkv projection, targets that were once set below their own floors.
Those are **evidence**, not coupling: rule 3 asks for the decisive pair that
settled a question, and a rule stated without the case that produced it is the
folklore this skill exists to prevent. They are safe to read on any machine and
safe to delete only when a better example replaces them.

Anything that is a *host* dependency rather than an example — a launcher, an
output path, a timer harness — is marked as one where it appears.

## Files

| Path | Read when |
|---|---|
| `sm90/README.md` | **start here** — the machine, its toolchains, and the headline result per unit |
| `sm90/constants.yaml` | the source of truth; read via `scripts/constants.py` rather than by eye |
| `references/protocol.md` | writing or extending a probe, or judging whether a number is usable |
| `references/naming.md` | naming a new constant, or decoding an existing tag; the retired-spelling map |
| `references/vocabulary.md` | naming anything in probe code — the authority for each term, and what is deliberately ours |
| `references/roadmap.md` | what is deliberately not measured yet, and the decisive pair each gap needs |
| `references/category-memory.md` | adding or porting a memory-side unit — the questions without this machine's answers |
| `references/category-compute.md` | adding or porting a compute unit |
| `references/category-execution.md` | adding or porting a launch / occupancy / sync unit |
| `sm90/unit-tma.md` | choosing a tile, grid, ring stages, or `BK`; anything TMA-fed |
| `sm90/unit-launch.md` | fusion decisions, grid sizing, clusters, occupancy |
| `sm90/unit-atomic.md` | split-K reduction, histogram-shaped accumulates, or the megakernel counter protocol |
| `probes/units/gmem_atomic/gmem_atomic.{cu,py}` | re-measuring the atomic unit, or adding a sweep to it |
| `sm90/unit-mma.md` | choosing a wgmma N, a pipeline stages, or the number of math warpgroups; checking an overlap claim by arithmetic |
| `probes/units/mma_rate/mma_rate.{cu,py}` | re-measuring the tensor core, or adding a sweep to it |
| `scripts/constants.py` | consult or validate the constants |
| `scripts/frontier.py` | "how few CTAs", "how small a TMA", "what is my copy floor" |
| `scripts/curve_from_json.py` | after a probe run, to reduce raw sweeps to a curve |
| `probes/units/tma_ring/tma_ring.{cu,py}` | re-measuring the TMA unit, or adding a sweep to it |
