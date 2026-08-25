# Category: compute — 计算

**The math engines and the instructions that drive them.** Tensor cores, CUDA
cores, the special-function units — how fast each retires instructions, what
must be in flight to keep it fed, and which instruction to choose when more
than one reaches the same hardware.

Arch-independent. Results live in `<arch>/unit-*.md`.

## Units in this category

| Unit | Probe | Measures |
|---|---|---|
| **mma** | `probes/compute/mma_rate.{cu,py}` | both tensor-core instructions: issue cadence against tile size, latency, in-flight depth, warp/warpgroup scaling, the clock under sustained load, and the crossover between them |

**Not yet a unit anywhere:** CUDA-core fp32/int throughput, transcendentals and
the SFU, fp8 and f16-accumulate variants, operand-from-register forms, and the
newer tensor-core instruction families on later architectures.

## Measure cycles, not FLOP/s

This is the category's defining rule, and it is not a stylistic preference.
Clocks move under load — on sm90, 1.05–1.61 GHz — so a throughput figure carries
the clock inside it and two runs of the same kernel are not comparable.

> The sm90 warpgroup sweep makes the case: achieved TFLOP/s said a second math
> warpgroup was worth **+20%**, and the cycle counts showed aggregate throughput
> did not move at all. The 20% was entirely a clock difference between two runs.
> A rate-only probe would have shipped a wrong design rule.

Use an in-kernel cycle counter over the measured window, and derive the clock
from `cycles / time` as a **by-product** — it is then a free observation about
the machine rather than an assumption inside the result.

## Gate every rate on a correctness check

A rate measured on an instruction nobody verified is a measurement of an
unknown. Math instructions make this acute: operand descriptors, fragment
layouts and accumulator register mappings are all easy to get subtly wrong, and
a wrong one still issues, still retires, and still produces a number.

**Run the instruction once on real data, write the result out through the
mapping under test, and compare against a reference.** Make the probe *fail*
rather than report if it does not match. Both sm90 instructions were gated this
way and both passed at ~1e-7 relative error — which is what makes the rates
citable.

## Make the toolchain prove it built what you wrote

Compute probes target one specific instruction, so a build that quietly
substituted something else invalidates everything. Two sm90 traps, both of
which produce a plausible wrong number rather than an error:

- an architecture flag that silently resolves to a *less* capable virtual arch,
  after which the assembler rejects or downgrades the instruction under test;
- a missing compiler hint that causes the pipeline to be **serialised** — the
  loop still runs, and the number it produces is the serialised rate.

Check the build diagnostics, and treat a serialisation warning as a build
failure. The details for this arch are in `sm90/unit-mma.md`.

## The questions a compute unit must answer

1. **Cost per instruction against tile size.** Flat means the cost is issue
   overhead and a bigger tile is free; scaling means the engine is the limit.
   The knee between the two is a tiling constraint, not a codegen detail.
2. **Latency versus issue interval**, separated by a depth knob — how many
   independent accumulators or outstanding groups are needed to cover it. This
   is simultaneously a register-budget decision, so it belongs to tiling.
3. **What the wait/synchronisation primitive costs** at each depth. Draining
   the pipeline every iteration is the natural thing to write and it is
   expensive.
4. **Does one warp/warpgroup already saturate the engine?** If yes, a second is
   pure register pressure and needs an independent justification.
5. **When two instructions reach the same hardware, where do they cross over?**
   Put both on one axis — FLOP per cycle per SM — which is the only fair
   comparison between a warp-scoped and a warpgroup-scoped instruction. Measure
   them in the same job, on the same SMs, through the same clock source, so the
   comparison survives the clock moving.

> The sm90 answer to (5) was that the *older* warp-level instruction wins below
> a tile N of 32, by up to 3.1× — the opposite of the "reach for the newest
> instruction" instinct. Nothing about that is predictable from a datasheet.

## Where this category meets the memory category

A fused kernel's whole thesis is that the copy engine and the math engine run
concurrently. Checking that claim needs one number from each category, on the
same axis: **bytes consumed per unit time against bytes delivered per unit
time.** Compute it, put it in the arch's constants, and mark it as arithmetic
over two constants rather than a third measurement.

**And note what it is not:** two rates measured in separate kernels do not
establish that the engines overlap when run together. That is a distinct
measurement — one kernel, both engines — and on sm90 it has **not** been made.
