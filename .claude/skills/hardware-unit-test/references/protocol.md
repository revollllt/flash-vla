# The protocol — what makes a probe a unit test

A microbenchmark that produces a number is not a unit test. A unit test asserts
something about the machine that a later reading could **contradict**, and says
what would have contradicted it. Everything in this file exists to keep the
constants in `constants/` from decaying into folklore — numbers everyone repeats
and nobody can re-derive.

## The shape of a unit

| Field | Is | The failure it prevents |
|---|---|---|
| **claim** | one sentence, in the machine's terms | "TMA is fast" is not a claim; "one warp sustains one TMA per 270 ns" is |
| **value + units + short** | the constant | a value whose units are implicit gets multiplied by the wrong thing |
| **valid** | the range it was measured over | the constant is quoted at a shape nobody tested |
| **isolation** | what else was held fixed, and what was stripped out | the number is really about something else |
| **falsifier** | the sweep row that would have shown otherwise | a sweep that could not have come out differently measured nothing |
| **design_rule** | how it enters a kernel budget | a constant nobody knows how to spend |
| **provenance** | machine, job, node, toolchain, timer, clocks | two numbers combined across toolchains, silently |

`scripts/constants.py --validate` enforces every one of these. A constant missing
a falsifier is the common case and the worst one: it means the sweep was a
demonstration, not an experiment.

## Eleven rules the probes here follow

**1. Strip the confounds, then say what you stripped.** The TMA probe removes
wgmma, the consumer warp, counter polling and `__syncthreads` from the issue
path. That is what makes the number a *machine* constant rather than a property
of one kernel body.

**2. A probe constant is a CEILING, not a prediction.** Having removed
everything else, the probe measures the best the unit can do. A real kernel gets
at most that. So the gap between a kernel and its constant is a *quantity to
explain*, never evidence the constant is wrong.

**3. Move one axis at a time, and keep a decisive pair.** The point of the sweep
is not coverage, it is the pair of rows that discriminates between two
hypotheses. In-flight *bytes* versus in-flight *transactions* is settled by
holding `depth x frame` fixed and varying the split: at equal bytes, depth 16 x
2 KB and depth 2 x 16 KB measure 246 and 1319 GB/s, so it is transactions.
Design that pair first and the rest of the sweep is context.

**4. Compute the constant by the shortest path from what you timed.** The TMA
issue interval is `us * 1000 / trip` — a division that never passes through the
byte accounting, so a miscounted footprint or a wrong frame size cannot produce
it. Deriving it from GB/s instead would have made every bookkeeping bug look
like a hardware finding.

**5. Two points do not make a slope, and a knee needs points on both sides.**
The bandwidth-vs-size fit uses an 8x range; extrapolating from the smallest
point alone would have understated the largest case by 40%. Sweep past the knee
too: 264 CTAs is *worse* than 128, which no monotone model predicts.

**6. Report against the measured ceiling, not the datasheet peak.** 2.77 TB/s is
what this machine reaches; 3.35 is what the box says. A row printed as "83% of
peak" invites a hunt for the missing 17% that does not exist. The probe prints
both, and the constants are defined against the measured one.

**7. State the noise floor and refuse to read below it.** Clocks are unpinnable
for this user on this partition, so ~6% is noise. A 3% difference is not a
finding, however many digits the harness prints. Take comparisons side by side
in one process, never across jobs, whenever the difference matters.

**8. Name the timer.** CUPTI and CUDA events disagree by the launch overhead
events include — 13% at 0.18 ms, most of the measurement at 5 us. The probes
report which one they used, and a number from one is not comparable with a
constant recorded under the other.

**9. A persistent probe hangs rather than fails.** Every wait gets a deadline
and writes its site to a debug buffer before trapping, or a wrong ring index
costs an hour of Slurm time and tells you nothing.

**10. On an unpinnable machine, measure CYCLES, not rates.** Clocks here move
1.05–1.58 GHz under load, so a throughput figure carries the clock inside it.
The `mma` unit's second-warpgroup comparison shows what that costs: achieved
TFLOP/s said a second math warpgroup was worth **+20%**, and the cycle counts
showed aggregate throughput did not move at all — the 20% was entirely a clock
difference between two runs. Report cycles where the hardware exposes a counter
covering the measured window, and derive the clock from `cycles / time` as a
by-product rather than assuming one.

**11. Make the toolchain prove it built what you wrote.** Two traps in the
`mma` unit, either of which yields a plausible wrong number rather than an
error:

- `-arch=sm_90a` resolves to virtual arch `compute_90` on this toolchain, and
  ptxas then rejects every 90a-only instruction. `-gencode
  arch=compute_90a,code=sm_90a` is what actually selects it. This one fails
  loudly — but only because the probe used those instructions *directly*; a
  probe reaching them through a library would have silently taken the library's
  fallback path.
- Without `warpgroup_fence_operand` on every accumulator, ptxas emits `C7515`
  and **serialises** the wgmma pipeline. The kernel still runs and still
  produces a number: the serialised rate. The probe now treats C7515 as a build
  **failure**.

The general rule: when a probe targets a specific instruction, verify the
instruction is what ran — check the build diagnostics, and where the
instruction computes something, check the result (the `mma` probe gates every
rate on a torch comparison at every N).

## What does not transfer

Three boundaries, each of which has already burned this project once:

- **Across bodies.** A knob's slope on one kernel does not carry to another:
  pipeline depth was worth 1.6x on one and negative on another. The constants
  here are machine properties; a *slope* measured on a kernel is not.
- **Across machines and toolchains.** Every constant carries its node and
  toolchain. The tma unit and the launch unit on this machine were taken five
  days and one torch minor apart — they agree on the one constant they share,
  which is the only reason the two can be quoted together.
- **Across regimes.** `valid:` is not decoration. `TMA-ISSUE` drifts 265 → 281 ns
  across 2–16 KB; quoting it at 32 KB is an extrapolation and
  `scripts/frontier.py` says so on every line that does it.

## Adding a unit

**First decide which category it belongs to** — `references/category-memory.md`,
`category-compute.md` or `category-execution.md` — and read that file. Each
carries the questions its units must answer and the traps specific to it, with
none of any particular machine's answers in them. Add the unit to that
category's roster.

1. Write the questions first, in the order they change a design decision — not
   the order they are easy to measure. A question that changes nothing is not
   worth a probe.
2. For each question, name the **decisive pair** of configurations and the
   hypothesis each one kills. If you cannot, the question is not sharp yet.
3. Build the probe so every axis is a runtime argument: one binary, one sweep
   harness, no recompiles between points (a recompile between points is a
   confound you cannot see). It goes in `probes/<category>/`.
4. Run it, then write the constants into `<arch>/constants.yaml` with all
   seven fields.
5. `python3 scripts/constants.py --validate`, and put the job id in
   `provenance`.
6. Write `<arch>/unit-<name>.md`: the questions, the sweep design, the
   results narrative, and — the part that pays for the whole exercise — **the
   design rules and the open gaps**. Add a line to `<arch>/README.md`.
   Anything you learned that is *not* about this machine — a question worth
   asking anywhere, a trap in the method — belongs in the **category** file
   instead, or it is lost the next time someone ports the skill.
7. If a kernel floor now depends on the new constant, say so in the spec's
   `toolchain` block by tag. A measured constant retires an `[I]`.

Until a unit is measured, leave it in `units:` with `status: unmeasured` and
its questions written out. That way a floor that needs it is *visibly blocked*
instead of quietly guessed — which is the whole difference between this skill
and a folder of benchmarks.
