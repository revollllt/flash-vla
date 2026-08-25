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
python3 scripts/constants.py --tag TMA-ISSUE           # one, with what would refute it
python3 scripts/constants.py --unit mma                # one unit, in full
python3 scripts/frontier.py --table                    # the TMA saturation frontier
python3 scripts/frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304
```

Cite tags — `[TMA-ISSUE]`, `[MMA-CROSSOVER]` — so a floor traces to a job id.
**Read the `valid:` range before spending a constant**: quoting one outside the
range it was measured over is the mistake this format exists to catch.

## What is measured here

### 访存 — memory (`references/category-memory.md`)

**`unit-tma.md`** — delivered bandwidth is a function of the product
`CTAs × warps × bytes-per-box` and nothing else (22 bins, ±6.9%), **but one CTA
saturates at ~133 GB/s**, which one producer warp at the 32 KB descriptor cap
already reaches 91% of — so a second producer warp is worth ~10% and a third
nothing. Issue interval 270 ns per warp, frame-independent; latency 598 ns +
4.8 ns/KB, so ring depth 3 suffices at ≤16 KB. Sustained ceiling 3.02 TB/s.
Plus a real, unexplained 5–8% anti-scaling dip at 36–56 CTAs.

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

**`unit-launch.md`** — every kernel starts ~1.24 µs in debt (grid ramp, not
removed by graph capture), so launch count is a first-class fusion term. A cold
read costs `1.85 + MB/2.77` µs; `bytes / 3.35 TB/s` is not a floor. Declaring a
cluster is free, synchronising one is 0.65 µs at cluster 8 — and placement
dominates that floor. Occupancy capacity is not residency.

`example-phase0-run.md` is the original calibration run behind the `launch`
unit and its only provenance — **those constants carry no job ids**, which is
the weakest evidence in this directory and the reason that unit still wants a
probe of its own.

## The biggest thing still untested here

`MMA-VS-TMA` says a CTA needs ~4.2 producer warps per math warpgroup, and it
collides almost exactly with `TMA-CTA-CEIL`'s 4.4 warps' worth of per-CTA
bandwidth. **But it is arithmetic over two constants measured in separate
kernels.** Whether the copy engine and the tensor core actually run concurrently
at those rates — rather than contending — is what every fused kernel here
assumes, and no probe has run them together.
