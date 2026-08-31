# The protocol — what makes a probe a unit test

A microbenchmark that produces a number is not a unit test. A unit test asserts
something about the machine that a later reading could **contradict**, and says
what would have contradicted it. Everything in this file exists to keep the
constants in `constants/` from decaying into folklore — numbers everyone repeats
and nobody can re-derive.

## The shape of a unit

Two different things, kept apart on purpose since 2026-08-29.

**What you must establish to trust a number.** This is the discipline, and it
does not relax:

| Must have | Is | The failure it prevents |
|---|---|---|
| **claim** | one sentence, in the machine's terms | "TMA is fast" is not a claim; "one warp sustains one TMA per 248 ns" is |
| **valid** | the range it was measured over | the constant is quoted at a shape nobody tested |
| **isolation** | what else was held fixed, and what was stripped out | the number is really about something else |
| **falsifier** | the sweep row that would have shown otherwise | a sweep that could not have come out differently measured nothing |
| **provenance** | machine, job, node, toolchain, timer, clocks | two numbers combined across toolchains, silently |

A constant with no falsifier is the common case and the worst one: it means the
sweep was a demonstration, not an experiment. Establish these, then write the
narrative into `<arch>/unit-<name>.md` and the decision into an Agent Note --
the places built to hold reasoning.

**What the published table stores.** `<arch>/constants.yaml` is a distributable
results manual, not a lab notebook. It carries only what a reader needs to USE
the number:

| Field | Is |
|---|---|
| **value + units** | the number and the condition it holds under |
| **short** | the one-line answer, for scanning a column |
| **rule** | how it enters a kernel budget |
| **status** | present only when the entry is retracted or derived |

`scripts/constants.py --validate` enforces the second table. It cannot enforce
the first, which is why the first is a discipline and not a schema: a probe that
skips it produces an entry that validates and means nothing.

## Fourteen rules the probes here follow

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
holding `stages x box` fixed and varying the split: at equal bytes, stages 16 x
2 KB and stages 2 x 16 KB measure 246 and 1319 GB/s, so it is transactions.
Design that pair first and the rest of the sweep is context.

**4. Compute the constant by the shortest path from what you timed.** The TMA
issue interval is `us * 1000 / k_tile_count` — a division that never passes through the
byte accounting, so a miscounted footprint or a wrong box size cannot produce
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

**6b. `valid:` must state the REGIME, not just the configuration.** The seven
fields exist so a constant cannot be quoted where it does not hold, and the
field that does that work is `valid:`. Filling it with the sweep's parameters --
CTAs, warps, stages, box -- is the easy half. The half that gets skipped is
*what regime those parameters put the machine in*, and skipping it has now cost
this skill four constants.

For anything memory-sourced that means naming **the source and its size against
the cache below it**. "Cold 256 MB footprint" is not a regime statement if the
walk only reaches 68 MB of that footprint: at 1.3x a 50 MB L2 the result depends
on what the PREVIOUS sweep row left resident, and two byte-identical
configurations read 1.51x apart for exactly that reason. A whole "ceiling with a
knee" can be that artefact and nothing more.

Two habits follow.

- **Report what the walk TOUCHES, never what it could address.** The probe
  printed 235 MB for a walk that reached 68. `tma_ring.py` now computes
  `touched_b()` from the walk's own index arithmetic, prints it as the
  footprint, and flags any row within `COLD_MIN_L2_RATIO` of L2 with `!`.
- **Hold the regime constant across the rows being compared.** Rows whose L2
  ratio differs are not a controlled sweep even when every other parameter
  matches, because cache residency is one of the parameters.

The same applies away from memory: a compute constant's `valid:` should say
whether the unit was issue-limited or bandwidth-limited, because rule 10's
choice of cycles or wall time depends on it.

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

**But cycles are the wrong unit for a side that is bandwidth-bound, and the
overlap probe found that the hard way.** A DRAM-bound wait is a WALL-CLOCK
quantity: it does not shrink when the clock rises. So when a concurrent wgmma
warpgroup pulled the clock down, the TMA side's fixed real-time waits spanned
FEWER cycles and the probe reported TMA getting 11% *faster* from added load --
a systematic 0.89x that reproduced across twelve configurations and is entirely
an artefact. The rule refines to: **cycles for whatever is issue- or
compute-limited, wall time for whatever is bandwidth-limited**, and a probe
spanning both regimes has to say which regime each row is in. The same probe's
issue-limited rows are trustworthy in cycles and its bandwidth-limited rows are
not.

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
instruction computes **or moves** something, check the result. Both probes now
gate every rate on a comparison against a host reference: `mma` on a torch
matmul at every N, `tma` on the delivered bytes at every geometry.

That check is what makes a rate a measurement rather than a demonstration.
A wrong descriptor, box or coordinate walk delivers the **wrong bytes at the
right speed** — no row of any sweep looks suspicious, and nothing the probe
prints could contradict the constant. The `tma` unit ran that way for its whole
the gate is `verify()` and it aborts the job rather than reporting.

**11b. Do not name the instruction yourself — select it, then assert the
selection.** A probe that spells its own atom can measure something the library
would never have dispatched, and the constant then describes the probe. Reach
the instruction the way production code does (`GMMA::ss_op_selector` +
`make_tiled_mma` + `cute::gemm` for wgmma), and pin the choice with a
`static_assert` naming the atom the recorded constants were measured on. A
toolchain update that changes the dispatch then fails the **build**, instead of
quietly re-measuring a different instruction under the existing tags.

Two limits worth stating rather than hiding. Selecting through a library is not
free of rule 11's first trap — a library reaching a fallback path is exactly the
silent failure it warns about, which is why the assertion is not optional. And
where the vendor abstraction would reintroduce the confound the probe exists to
strip (a TMA `PipelineTmaAsync` puts producer/consumer semantics back into a
copy-engine probe; a `TiledMMA` puts partitioning back into a raw `mma.sync`
probe), keep the direct form and **say in the unit reference why**.

**12. Inspect the compiled probe, not just its output.** Rule 11 asks whether
the toolchain built what you wrote. This one asks whether *what you wrote* put
anything in the timed loop that is not the unit under test. Run `-Xptxas -v` on
every probe and read three things: **a non-zero stack frame with zero spills is
a local-memory array** — an array indexed by a runtime value — and every access
to it is a round trip through L1TEX sitting inside your measurement; the
register count, because it moves occupancy; and the instruction mix.

**13. A defect found in a probe is not automatically a defect in a kernel.**
The same edit can be a fix in one place and a regression in another, so a
profiler finding is a candidate to A/B, never a conclusion to apply.

**13b. A library measurement calibrates a model; it does not bound one.** Run
the best existing implementation at the same shapes and check your model lands
on it -- that is what says the model is neither optimistic nor pessimistic. It
is NOT a target to beat or a floor to trust: a specialised kernel is supposed to
go below a library call, and for a FUSED kernel the honest reference is the
composition it replaces, not the one library call it looks like.

**14. A baseline is only a baseline if it reproduces ITSELF.** Before
concluding that a change moved a number, re-run the *unchanged* code and check
it still gives the old number. Comparing new-code-today against
old-code-last-week attributes every machine difference to the edit.

**14b. Say where a measurement is not reproducible.** A single noise floor
quoted for a whole unit is a claim about every row in it. Regions can be far
noisier than the unit's median, and a constant read from one of them carries
that spread whether or not it is written down.

**14b2. Check the loop shape of every poll, not just the contended ones.** The
same two spellings of a bounded spin -- predicate in the loop condition, or a
variable assigned inside it -- compile to different code and measure
differently, in every unit here that polls:

| poll | predicate-in-condition | variable-tested |
|---|---|---|
| tma_ring, mbarrier | 257.8 ns/txn | 279.4 |
| gmem_atomic, gmem counter | 661.7 ns/hop | 680.1 |

The counter case was expected to be immune -- a release->acquire hop is
dominated by the round trip through L2, not by issue slots -- and moved 3.4%
anyway. Both primitives here use the predicate-in-condition form for that
reason: a poll that costs more reports the thing it waits for as slower than it
is.

**14c. A probe's own implementation is part of its number.** Two spellings of
the same wait loop, one library boundary, one helper made generic -- each can
move an absolute reading further than the noise floor while leaving the
instruction COUNT identical. When a constant is an absolute, record what it was
measured under; when it is a ratio, say so, because ratios survive what
absolutes do not.

## What does not transfer

Three boundaries, each of which has already burned this project once:

- **Across bodies.** A knob's slope on one kernel does not carry to another:
  pipeline stages was worth 1.6x on one and negative on another. The constants
  here are machine properties; a *slope* measured on a kernel is not.
- **Across machines and toolchains.** Every constant carries its node and
  toolchain. The tma unit and the launch unit on this machine were taken five
  days and one torch minor apart — they agree on the one constant they share,
  which is the only reason the two can be quoted together.
- **Across regimes.** `valid:` is not decoration. `tma.issue.warp` drifts 265 → 281 ns
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
   confound you cannot see). It goes in `probes/units/<name>/`.
4. Run it, then write the RESULT into `<arch>/constants.yaml`: value, units,
   short, rule. Nothing else goes in that file.
5. `python3 scripts/constants.py --validate`.
6. Write `<arch>/unit-<name>.md`: the questions, the sweep design, the results
   narrative, the claim/valid/isolation/falsifier for each constant, the job
   ids, and — the part that pays for the whole exercise — **the design rules
   and the open gaps**. This is where the evidence lives now, and a constant
   whose evidence is not here is folklore with a tag. Add a line to
   `<arch>/README.md`.
   Anything you learned that is *not* about this machine — a question worth
   asking anywhere, a trap in the method — belongs in the **category** file
   instead, or it is lost the next time someone ports the skill.
7. If a kernel floor now depends on the new constant, say so in the spec's
   `toolchain` block by tag. A measured constant retires an `[I]`.

Until a unit is measured, leave it in `units:` with `status: unmeasured` and
its questions written out. That way a floor that needs it is *visibly blocked*
instead of quietly guessed — which is the whole difference between this skill
and a folder of benchmarks.
