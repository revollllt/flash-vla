# Unit: MMA / wgmma — the tensor core's issue rate

**Probe** `probes/compute/mma_rate.{cu,py}` · **Constants** `MMA-RATE`, `MMA-DEPTH`,
`MMA-WARPGROUPS`, `MMA-CLOCK`, `MMA-VS-TMA`, `MMA-SYNC-RATE`,
`MMA-SYNC-DEPTH`, `MMA-CROSSOVER`

Two instructions, measured in one job on the same SMs: the warpgroup
`wgmma.mma_async` and the warp-level `mma.sync`. They are not
interchangeable and the choice between them is a tiling decision.

```
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/compute/mma_rate.py \
    --json profiles/hardware-unit-test/mma.json
```

One warpgroup issues `wgmma.mma_async.m64nNk16.f32.bf16.bf16` back to back out
of resident shared memory — no TMA, no global traffic, no barriers in the loop.
**Every rate is gated on a correctness check** (`M0`): one wgmma on random data,
D written out through the accumulator mapping, compared against torch at every
N. Measured relative error 5e-8 to 1e-7. A rate measured on an instruction
nobody verified is a measurement of an unknown.

## The four things to take away

**1. N must be ≥ 64.** Below it a per-instruction floor of ~19 cycles dominates
and most of the tensor core is idle:

| N | 8 | 16 | 32 | **64** | 96 | 128 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cycles/instruction | 18.9 | 20.7 | 24.7 | **32.8** | 48.7 | 65.0 | 97.1 | 129.3 |
| architectural ideal | 3.8 | 7.7 | 15.3 | 30.7 | 46.0 | 61.4 | 92.1 | 122.8 |
| % of peak | 20% | 37% | 62% | **94%** | 95% | 94% | 95% | 95% |

At N = 8 the instruction costs 18.9 cycles to do 3.8 cycles of work. Above 64,
cycles scale exactly with FLOPs — **N is free**, so take the largest the
register budget allows. Identical at 1 CTA and 132 CTAs, so this is a per-SM
rate. Landing at 94–95% of the architectural 2135 MAC/cycle/SM is itself the
evidence that no operand-path effect is throttling the measurement.

**2. Never write `wgmma.wait_group 0` in a mainloop.** It drains the pipeline
every iteration and costs 20–30%:

| per group → | 1 | 1 | 1 | 2 | 2 | 4 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wait depth | 0 | 1 | 3 | 0 | 1 | 0 | 1 | 0 |
| in flight | 1 | 2 | 4 | 2 | **4** | 4 | **8** | 8 |
| cycles/instruction | 79.0 | 42.0 | 33.7 | 55.6 | **33.1** | 44.6 | **32.3** | 38.2 |

At *equal* in-flight count the configurations disagree — 4 in flight is 33.1
with `wait 1` and 44.6 with `wait 0` — so the wait depth is its own axis, and
it is the one that matters. **Four in flight with wait ≥ 1** is the knee. That
is also a register decision: at N = 64 each group in flight is 32 accumulator
registers per thread, so pipeline depth and register budget are one choice.

**3. One warpgroup saturates the tensor core.** A second exactly doubles the
cycles each spends per instruction (33.1 → 64.4, 65.0 → 128.7, 32.3 → 64.3,
64.4 → 128.5 — all 1.94–1.99×), so aggregate throughput does not move.

> **The TFLOP/s column says the opposite and it is wrong.** Those same 2-warpgroup
> runs measure 703.6 → 842.7 TFLOP/s, a 20% "gain", purely because they happened
> to sit at 1.57 GHz against 1.31. A probe reporting only achieved TFLOP/s would
> have concluded a second math warpgroup is worth 20%. This is the strongest
> argument in this skill for measuring **cycles** on a machine whose clocks
> cannot be pinned.

Add a second math warpgroup only to hold more accumulators than 255 registers
allow, or to overlap an epilogue — never on the theory that it feeds the tensor
core faster.

**4. Below tile N = 32, use `mma.sync` instead — it is worth up to 3.1×.**
Which is the opposite of the usual "reach for the newest instruction" instinct.
On one axis, FLOP per cycle per SM, both measured in the same job:

| wgmma tile N | 8 | 16 | **32** | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| wgmma | 871 | 1577 | **2623** | 3959 | 4036 | 4053 |
| best `mma.sync` | 2680 | 2680 | **2680** | 2680 | 2680 | 2680 |
| winner | sync **3.1×** | sync **1.7×** | **tie** | wgmma 1.5× | wgmma 1.5× | wgmma 1.5× |

The comparison is like-for-like in coverage: `wgmma m64nNk16` covers a 64×N tile
with one instruction, and the `mma.sync` column covers the same tile with 4
warps of `m16n8k16`. The N = 32 row differs by 2%, inside the noise floor, so it
is a tie rather than a win either way.

That the wgmma small-N penalty is real and not a probe artefact is settled by
this same table: `mma.sync`, measured in the same job on the same SMs through
the same clock source, is *flat* in tile size and beats wgmma by 3.1× exactly
where wgmma's per-instruction floor bites.

## The warp-level instruction

`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`, operands in registers, verified
against torch at 1.1e-7 relative error before any rate was read.

| independent accumulators | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| cycles/instruction | 25.14 | 12.54 | **6.26** | 6.29 |

**Latency is ~25 cycles, the issue interval is 6.26, so a warp needs exactly 4
independent accumulator sets** (25.14 / 6.26 = 4.02) — 16 f32 registers per
thread. One set is a latency reading, not a slow throughput reading; two halve
it and four quarter it, which is the 1/n a latency-bound pipeline gives.

| warps per SM | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| FLOP/cycle/SM | 655 | 1310 | **2618** | 2680 |

Linear to 4 warps, then flat. **The ceiling is ~2680 FLOP/cycle/SM = 63% of the
architectural peak**, against wgmma's 95% — reaching for `mma.sync` costs a
third of the tensor core even when everything else is perfect. That the 63% is
the instruction and not the probe is settled by the same probe reaching 95%
through wgmma.

Its lower utilisation lets the GPU clock *higher* (1.49–1.61 GHz against
wgmma's 1.06–1.34), so its achieved TFLOP/s flatters it relative to its
FLOP/cycle. Read cycles, not rates — again.

**Every engine on this machine wants four things outstanding**: 4 wgmma in
flight (`MMA-DEPTH`), 4 accumulator sets per warp (`MMA-SYNC-DEPTH`), a TMA ring
of depth 4 (`TMA-DEPTH`). A useful default to reach for, and a cheap one to
check.

## The ratio this unit exists for

An L3 timeline that claims the math column covers the copy column is asserting
a ratio. Here it is:

```
one wgmma m64n128k16 : 64.4 cycles @ 1.32 GHz = 48.8 ns
  consumes A(64×16) + B(128×16) bf16 = 6144 B   ->  126 B/ns
one TMA producer warp: 8192 B per 270 ns        ->   30 B/ns   [TMA-ISSUE]
```

**A CTA needs ~4.2 producer warps per math warpgroup** at `BLOCK_K = 16` with no
reuse — equivalently, each delivered byte must be reused ~4× before one producer
warp can keep the tensor core fed. At N = 64 the ratio is *worse* (166 B/ns
consumed), not better.

And it collides with the TMA unit's other ceiling: `TMA-CTA-CEIL` caps one CTA
at 133 B/ns, which is **4.4 producer warps' worth**. The two numbers meet almost
exactly, so a one-CTA-per-SM fused kernel at `BLOCK_K = 16` with no reuse sits
right on the edge of feedable. The fix is reuse — a larger `BLOCK_M`/`BLOCK_N`
so each tile feeds more wgmma — not more producer warps.

## The clock, and why cycles are the unit

Under sustained wgmma load this GPU runs at **1.05–1.58 GHz**, not the 1.755 GHz
the datasheet's 989 TFLOP/s assumes. So the architectural peak is reached *in
cycles* and missed *in wall clock*: best observed ≈ **850 TFLOP/s**. Target
cycles against `MMA-RATE`; treat 850 as the practical ceiling, the way `BW-CEIL`
replaces the 3.35 TB/s datasheet bandwidth.

## Two build traps, either of which yields a plausible wrong answer

- **`-arch=sm_90a` is not enough.** On this toolchain it resolves to virtual
  arch `compute_90`, and ptxas then rejects `wgmma.fence`,
  `wgmma.commit_group` and `wgmma.wait_group` outright. Use
  `-gencode arch=compute_90a,code=sm_90a`. (It fails loudly here — but only
  because the probe uses those instructions directly.)
- **Accumulators need `warpgroup_fence_operand`.** Without it ptxas emits
  `C7515` and *serialises* the wgmma pipeline; the loop still runs and still
  produces a number, and that number is the serialised rate. The probe treats
  C7515 as a **build failure**, not a warning.

## Open gaps

- **fp8 and f16 accumulate.** Only bf16-in/f32-out is measured, for both
  instructions. The published ridge point doubles with fp8, so a single bf16
  number would condemn every fp8 tile ever written — measure both or claim
  neither.
- **`mma.sync` operand loading.** The probe holds fragments in registers and
  never reloads them, so it measures the instruction and not `ldmatrix`. A real
  small-N kernel pays for the fragment loads too, and that cost is not in
  `MMA-CROSSOVER` — which therefore states the crossover under conditions
  favourable to `mma.sync`.
- **A from registers.** Only the smem–smem (`_SS`) form is measured; the `_RS`
  form's cost, and whether it is worth its register pressure, is untested.
- **Swizzled operand layouts.** The probe uses the unswizzled `Layout_K_INTER`
  atom. Landing at 94–95% of peak says the layout is not throttling *this*
  measurement, but it does not measure what a swizzled layout costs or saves.
- **The overlap itself.** `MMA-VS-TMA` is arithmetic over two constants measured
  in separate kernels. Whether the copy engine and the tensor core actually run
  concurrently at these rates — rather than contending — is the assumption every
  fused kernel in this repo rests on, and **it is still untested**. The probe
  needs a mode that runs the TMA ring and the wgmma loop in one kernel and
  reports both rates against their solo values.
- **Why the clock moves** with configuration in ways the work does not explain.
