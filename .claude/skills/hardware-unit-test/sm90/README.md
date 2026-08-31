# sm90 — H100 SXM5, `acd_u` partition

The measured constants for this machine. **Nothing here transfers to another
architecture**; the arch-independent layer — what to measure and how — is in
`references/category-*.md` and `references/protocol.md`.

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, 132 SMs, `sm90a` |
| smem | 232448 B per CTA (227 KB opt-in), 233472 B per SM |
| L2 | 50 MB · HBM datasheet peak 3.35 TB/s |
| clocks | **not pinnable** for this user — ~6% noise floor, and 1.05–1.61 GHz observed under load |
| nodes | ACD1-4, -18, -31, -36 |

**Toolchain caution.** One machine, two toolchains: the `tma`, `atomic` and
`mma` units are torch 2.13.0+cu130 / CUDA 13.1 (2026-08-25); the `launch` unit
is torch 2.11.0+cu130 / CUDA 13.0 (2026-08-20). They agree independently on the
streaming-bandwidth ceiling, which is the only cross-check between them. Do not
combine a number from one with a number from the other without saying so.

## Consulting

```bash
python3 scripts/constants.py --machine sm90            # every constant, one line each
python3 scripts/constants.py --tag tma.issue.warp           # one, with what would refute it
python3 scripts/constants.py --unit mma                # one unit, in full
python3 scripts/frontier.py --table                    # the TMA saturation frontier
python3 scripts/frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304
```

Cite tags — `[tma.issue.warp]`, `[mma.xover.n.wgmma]` — so a floor traces to a job id.
**Read the `valid:` range before spending a constant**: quoting one outside the
range it was measured over is the mistake this format exists to catch.

## What is measured here

### 访存 — memory (`references/category-memory.md`)

**`unit-tma.md`** — delivered bandwidth is a function of the product
`CTAs × warps × bytes-per-box` and nothing else, to ~10% at the 90th percentile
with a 19–29% tail driven by CTA count. There is **no per-CTA ceiling**: the issue interval stays flat while delivery
rises linearly to at least 40 KB per CTA, so a second producer warp is worth
close to 2x there. Issue interval **248 ns** per warp, box-independent; latency **595 ns +
5.2 ns/KB** on DRAM (217 + 4.8 from L2), so ring **stages 4 for DRAM, 2 for L2**.
Sustained ceiling **≤3.17 TB/s** (an upper bound; the cold rate is lower). Plus a
real, unexplained **11.5%** anti-scaling dip at 44–56 CTAs, DRAM only.

**`unit-atomic.md`** — **address layout is worth 6.3×, every other lever ≤1.3×.**
Per-transaction, so `red.global.add.v4.f32` moves 3.8× the bytes for free;
`red` beats `atom` by 1.30×; scope is free; one contended address is 386× down.
A gmem-counter hop is ~650 ns and **observers are free** — one counter can gate
the whole machine.

### 计算 — compute (`references/category-compute.md`)

**`unit-mma.md`** — `wgmma` tile N must be **≥ 64** (at N=8 it runs at 20% of
peak); four in flight with `wait_group ≥ 1` (never `wait_group 0`, worth
20–30%); **one warpgroup already saturates the tensor core**. **Below tile
N = 32 the warp-level `mma.sync` wins by up to 3.1×**, though its own ceiling is
63% of peak against wgmma's 95%. Practical bf16 ceiling ~850 TFLOP/s, not the
datasheet's 989, because the clock drops under load.

### 执行 — execution (`references/category-execution.md`)

> The `launch` unit's six constants carry **no Slurm job ids** -- only a date,
> harness and toolchain. That is the weakest provenance in this directory;
> `unit-launch.md` is where they and their tables live.

**`unit-launch.md`** — every kernel starts ~1.24 µs in debt (grid ramp, not
removed by graph capture), so launch count is a first-class fusion term. A cold
read costs `1.85 + MB/2.77` µs; `bytes / 3.35 TB/s` is not a floor. Declaring a
cluster is free, synchronising one is 0.65 µs at cluster 8 — and placement
dominates that floor. Occupancy capacity is not residency.

**`unit-coop.md`** — a `cooperative_groups` `grid_sync` costs **1.09 µs** and
barely moves with grid size (1.02× over a 4× block range). A cooperative launch
accepts exactly `max_active_blocks_per_sm × SMs` blocks — 1056 here, with 1057
refused — so the bound can be queried rather than discovered. A device-side
relaunch costs 1.40 µs, only **1.29×** a barrier: replacing launches with grid
barriers buys ~22%, not an order of magnitude.

## The biggest thing still untested here

**No per-SM bandwidth ceiling has been measured cold.** `tma.bw.dev.dram` bounds
the device and `tma.issue.warp` bounds one producer warp, but nothing measures
what one SM can absorb, so the middle of the hierarchy is empty.
The frontier still climbs past `tma.bw.dev.dip` at 44-56 CTAs with no model that
explains the shape.

*Previously in this slot:* whether the copy engine and the tensor core actually
run concurrently rather than contending. They were measured together --
`overlap.eff.sm` puts TMA 1.25x slower and wgmma 1.05x under contention, and
`pipeline.ratio.sm.dep` adds ~1.05x for the barrier on top. Budget ~1.32x over
the slower engine, not the 1.00x a timeline assumes.
