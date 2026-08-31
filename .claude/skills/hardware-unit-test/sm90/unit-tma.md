# Unit: TMA — the copy engine's delivery rate

**Probe** `probes/units/tma_ring/tma_ring.{cu,py}` · **Constants** `tma.issue.warp`, `tma.stages.warp.knee`,
`tma.bw.cta.geom`, `tma.bytes.txn.max`, `tma.bw.dev.l2`, `tma.bw.cta.warps`, `tma.bw.dev.dram`,
`tma.bw.dev.curve` · **Curve** `tma-bw-vs-product`

```
# launcher and output path are this host's; the probe itself takes neither
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/tma_ring/tma_ring.py \
    --sweeps A,E,F --json profiles/hardware-unit-test/tma_frontier.json
python3 scripts/curve_from_json.py profiles/hardware-unit-test/tma_frontier.json --yaml
```

## The two things to take away

**1. Aggregate delivery depends on the product `n_ctas × num_producers ×
box_bytes` and on nothing else.** 8 CTAs issuing 32 KB TMAs and 264 CTAs
issuing 1 KB TMAs sit on the same curve; across 22 bins spanning a 132× range
the worst spread at equal product is **6.9%**, against a 6% noise floor. So CTA
count, producer warps and bytes-per-TMA are **interchangeable currency** — 22
matched `(N CTAs × 2 warps)` vs `(2N × 1 warp)` pairs agree within 3.4%, twenty
of them within 1.3%.

**2. There is no per-CTA ceiling.** The per-warp issue interval stays flat and
delivery rises LINEARLY to at least 40 KB per CTA, so a second producer warp
buys close to 2x there:

```
delivered = min( n_ctas × per_cta ,  curve(product) )
per_cta   = num_producers × box / 248 ns          [tma.issue.warp], no plateau
```


A tiling decision therefore asks two questions, not one: *how many bytes in
flight across the grid*, and *is any single CTA being asked for more than it can
absorb?*

| in flight | delivered | % of ceiling |
|---:|---:|---:|
| 256 KB | 978 GB/s | 32% |
| 512 KB | 1868 GB/s | 62% |
| 768 KB | 2393 GB/s | 79% |
| 1.0 MiB | 2605 GB/s | 86% |
| 1.5 MiB | 2738 GB/s | 91% |
| 3.0 MiB | 2861 GB/s | 95% |
| 4.0 MiB | 2934 GB/s | 97% |
| 6.2 MiB | 2989 GB/s | 99% |

Below ~256 KB the curve is exactly linear, because each warp is issuing one TMA
every 248 ns and nothing else binds. Above it, the curve bends as DRAM takes
over, and the ceiling itself keeps creeping up with in-flight bytes until it
flattens near **3.02 TB/s** — 90% of the datasheet peak.

## The two questions this was built to answer

### Q1 — the tile is large. How few SMs still saturate?

Measured, stages 4. The first two columns hold the same 32 KB per CTA and the
same 128 KB of in-flight smem, split differently:

| CTAs | 1w × 32 KB | 2w × 16 KB | 2w × 24 KB |
|---:|---:|---:|---:|
| 8 | 969 (32%) | 965 (32%) | 1061 (35%) |
| 16 | 1791 (59%) | 1802 (60%) | 1836 (61%) |
| 24 | 2306 (76%) | 2324 (77%) | 2417 (80%) |
| 32 | 2622 (87%) | 2589 (86%) | **2731 (90%)** |
| 48 | **2775 (92%)** | **2767 (92%)** | 2610 (86%) |
| 64 | 2808 (93%) | 2796 (93%) | 2781 (92%) |
| 96 | 2804 (93%) | 2769 (92%) | 2786 (92%) |
| 128 | **2962 (98%)** | 2933 (97%) | 2932 (97%) |
| 132 | 2943 (97%) | 2903 (96%) | **3018 (100%)** |

**~48 CTAs at 32 KB per CTA, ~94 at 16 KB, ~188 at 8 KB reach 90%.** Each
doubling of the per-CTA product halves the CTA count — the product law again.
At 32 KB per CTA, **a third of the machine is already at 92%**, so a kernel that
needs bandwidth and not compute does not need the whole GPU.

Two things to read off it. The `1w × 32 KB` and `2w × 16 KB` columns are the
same to within 1.2% at all nine CTA counts, because they are the same
configuration to the machine. And `2w × 24 KB` — which costs 64 KB more smem —
is ahead only where the grid is small, then dips *below* the others at 48 CTAs
(2610 against 2775). That dip is real and reproducible, and it is the transition
region flagged under open gaps.

### Q1b — what does a second producer warp buy?

**About 10%, and only on a small grid. A third buys nothing.**

Per-CTA delivery, measured at 8 CTAs where nothing aggregate can bind, is linear
in `num_producers × box` and then flat:

| `num_producers × box` | 32 KB | 36 KB | 40 KB | 44 KB | 48 KB | 56 KB |
|---|---:|---:|---:|---:|---:|---:|
| 2 producer warps | 121.3 | 131.5 | 132.1 | 132.4 | 132.6 | 132.8 |
| 4 producer warps | 121.3 | 131.2 | 131.9 | 132.2 | 132.5 | 132.6 |

Linear extrapolation from `tma.issue.warp` would give 182 GB/s at 48 KB; the machine
gives 132.6. The 2-warp and 4-warp columns agree within 0.3% at every point, so
**the ceiling is a property of the CTA, not of the warp** — four warps do not
beat two.

One warp is capped at a 32 KB box by `tma.bytes.txn.max`, which already delivers
121 GB/s = **91% of the per-CTA ceiling**. That gap is the whole value of the
second warp. At the full grid it is smaller still: 132 CTAs × 1 warp × 32 KB
measures 2943 GB/s and 132 × 2 × 24 KB measures 3018 — **2.6%**.

**How much per-CTA budget is useful depends on the grid**, and it is easy to get
wrong in either direction.

| config | per-CTA | in-flight smem | at 8 CTAs | at 132 CTAs |
|---|---:|---:|---:|---:|
| 1 warp × 32 KB | 32 KB | 128 KB | 121.1/CTA | 2943 |
| 2 warps × 16 KB | 32 KB | 128 KB | 121.3/CTA | 2903 |
| **2 warps × 18 KB** | **36 KB** | **144 KB** | **131.5/CTA** | — |
| 2 warps × 24 KB | 48 KB | 192 KB | 132.6/CTA | **3018** |
| 2 warps × 28 KB | 56 KB | 224 KB | 132.8/CTA | 3006 |

The first two rows are the same configuration as far as the machine is
concerned — equal product, equal smem, equal bandwidth, differently split.

On a **small grid** the per-CTA ceiling binds, so **36 KB per CTA is the
efficient point**: 32 → 48 KB is +9.3% and 48 → 56 KB is +0.2%. On a **full
grid** the aggregate curve binds instead, extra per-CTA bytes still raise the
total in flight, and more *is* better: 32 → 48 KB per CTA measures **+4.0%** at
132 CTAs. What holds everywhere is the top end — **48 → 56 KB is worth nothing
at any grid size** (132 CTAs: 3018 → 3006). So the 227 KB smem cap is not the
constraint to design against; ~48 KB × stages of in-flight smem is, and the
remaining ~35 KB is better spent on the consumer side or on occupancy.

### Q1c — the CTA-count floor

There is no per-CTA ceiling to divide the device ceiling by, so the CTA-count
floor comes from the measured aggregate curve directly: **~48 CTAs for 90%**, and
at 8 KB boxes ~128 CTAs (`tma.ctas.dev.dram.knee`) — the opposite of a small pool
of copy SMs.

### Q2 — every SM is busy. How small can one CTA's TMA be?

Measured at 132 CTAs (one per SM), one producer warp, stages 4:

| box | 1 KB | 2 KB | 3 KB | 4 KB | 6 KB | 8 KB | 12 KB | 16 KB | 24 KB | 32 KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 507 | 1000 | 1479 | 1906 | 2419 | 2676 | 2742 | 2801 | 2917 | 2943 |
| % ceil | 17 | 33 | 49 | 63 | 80 | 88 | 91 | 93 | 97 | 97 |

**~10.9 KB for 90%, ~23 KB for 95%.** And a hard consequence: 99% would need a
43 KB box, which is **above the 32 KB descriptor maximum** — so *one producer
warp per SM on a 132-CTA grid cannot pass ~97%, however the tile is chosen.* The
fix is a second producer warp or a second CTA per SM; it is never a bigger box.

`python3 scripts/frontier.py --table` prints both from the curve, and
`--min-ctas` / `--min-box` answer one configuration at a time with the
extrapolation warnings attached.

## The rest of the unit, briefly

- **`tma.issue.warp` = 248 ns per TMA per producer warp**, box-independent from
  1 KB to the 32 KB descriptor maximum. Delivered bandwidth is
  `n_ctas × num_producers × box_bytes / 248 ns` until the machine caps it.
- **`tma.lat.warp` = 598 ns + 4.8 ns/KB** — 636 / 677 / 751 ns at 8 / 16 / 32 KB,
  measured at ring stages 1 where nothing overlaps. The stages-2 rows come in at
  exactly half of each (0.506 / 0.511 / 0.521), which is what proves stages 1 is
  reading latency and not issue overhead.
- **`tma.stages.warp.knee`**: the covering stage count is `ceil(latency / 248 ns)` — **3 at or
  below 16 KB boxes**, 4 at 32 KB. Measured: at 8 KB stages 3 and 4 give 265.0
  and 264.7 ns, at 32 KB stages 3 is 281.6 against 271.8. The blanket "stages 4"
  costs a quarter of the ring's smem for nothing below 16 KB. Above the covering
  stages nothing improves at any box.
- **`tma.bw.dev.dip` — 5-8%, at 36–56 CTAs, only above the per-CTA knee.** A real
  anti-scaling band: adding CTAs makes it slower. Cause unknown.
- **`tma.bw.cta.geom` ≤ 3.4%.** A contiguous box, 128 B strips at 2 KB stride and at
  8 KB stride are indistinguishable at 32 CTAs and 3.4% apart at the ceiling.
  **The gather hypothesis is dead.**
- **`tma.bytes.txn.max` = 32 KB.** `boxDim[0] × elemSize ≤ swizzle width` (so 128 B
  rows under SW128) and `boxDim[1] ≤ 256`. Both enumerated against the driver.
- **`tma.bw.dev.l2`**: the 248 ns is source-independent — L2 and DRAM agree to 1–3%
  below 64 CTAs — and L2 shows no ceiling of its own through 6.45 TB/s.
- **`tma.bw.cta.warps`**: warps and CTAs are the same currency on DRAM — 21
  matched pairs, worst disagreement 0.9%. On L2 the equivalence BREAKS (22.1%
  worst), because there one CTA per SM out-delivers two holding the same bytes.

## Using it in a design

**Budget the copy column, don't assume it.**

```
python3 scripts/frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304
```

`txns_per_warp = K_per_CTA / BK`, so the floor is
`max(txns_per_warp × 248 ns, bytes / 3.17 TB/s)`.

Three design rules fall out, and the third is the one that gets missed:

1. **Bytes per TMA is a first-class term.** 8 KB → 16 KB is 1.92× on the copy
   column, free. This is why `BK=128` beat `BK=64` in the FFN task-loop.
2. **Check `tma.bytes.txn.max` before choosing `BK`.** A descriptor whose strided
   dimension is short caps the box: an activation shaped `(M_PAD=64, D)` is
   stuck at 8 KB per TMA whatever the tile. *That*, not the gather, is the
   reason to pre-block an activation — ~2× rather than ~3%.
3. **A second producer warp is worth ~10%, a third nothing** — and the useful
   in-flight smem per CTA stops at `36 KB × stages`. Buy the second warp when the
   grid is pinned small (a persistent kernel at 1 CTA/SM); do not buy the third,
   and do not grow both boxes to fill smem.
4. **More CTAs does not move an issue-bound floor.** An M-split adds CTAs, but
   every CTA still walks the whole K, so its per-warp serial chain — and the
   wall clock — is unchanged. More CTAs buys aggregate bandwidth, which a
   transaction-bound kernel is not short of. To move it: bigger `BK`, split-K,
   or more producer warps.

And when the kernel is short, use `ld.bw.dev.dram`'s `1.85 + MB/2.77` instead of
`tma.bw.dev.dram`: the 3.02 TB/s is a steady-state figure measured over ~92 µs, and a
5 µs kernel never gets there.

## The mid-grid dip

Walking the CTA ladder finely found something the coarse sweeps had shown as a
single odd point. At **56 KB per CTA** (2 warps × 28 KB):

| CTAs | 32 | 36 | 40 | 44 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GB/s | **2727** | 2610 | 2532 | 2556 | 2523 | 2615 | **2787** |

Five consecutive CTA counts sit 5–8% *below* both their neighbours. At 48 KB per
CTA the dip is shallower and narrower (~5%, 44–56 CTAs). At **32 KB per CTA there is no dip at all**: the ladder rises
monotonically to a plateau by ~40 CTAs and stays there.

The onset is the clean evidence: 32 → 36 CTAs moves the **same 66.1 MB** and
takes 4.5% longer. (From 40 CTAs up the probe's k_tile_count floor makes those rows move
more bytes, so only the onset is byte-matched — stated because it is the obvious
objection.)

**The cause is unknown.** What is actionable is the shape: do not size a grid
into 36–56 CTAs with a per-CTA product above the knee. 32 CTAs and ≥64 CTAs are
both faster than anything between.

## Per SM, not per CTA — and the L2 figure's range

Splitting the same per-SM bytes two ways, L2-resident so the DRAM ceiling is out
of the way, with placement read from `%smid` rather than assumed:

| per-SM in-flight | 1 CTA/SM (2 warps) | 2 CTA/SM (1 warp each) |
|---|---|---|
| 8 KB | 29.9 | 29.8 |
| 16 KB | 54.4 | 54.4 |
| 32 KB | 108.5 | 106.0 |
| 40 KB | 132.6 | 135.2 |
| 48 KB | 131.4 | **101.6** |
| 56 KB | **137.5** | **98.0** |

GB/s per SM. Below 40 KB the split does not matter; above it one wide CTA
out-delivers two narrow ones holding the same bytes, which is
`tma.bw.sm.cta.scale`.

**There is no per-SM ceiling to quote.** The figure moves with CTA count — 208
GB/s at 4 SMs against 138.5 at 132 on L2 — so any single number is a point on a
curve, not a ceiling. Measuring it cold is an open gap.

**What caps the L2 number.** Per-SM shared memory is 233472 B, so at stages 4
the largest per-SM in-flight this probe can reach is ~57 KB. 56 KB is therefore
the end of the range, not a comfortable margin past the knee — the knee at 40 KB
has three points beyond it and that is what makes it a knee, but 18.2 TB/s is
the largest L2 figure *reachable this way*, not proof that L2 has nothing left.

## How few SMs saturate each source

Dividing the device ceiling by 132 describes a grid choice, not the machine.
At 56 KB of per-SM in-flight, one CTA per SM, placement confirmed by `%smid`:

| SMs | DRAM GB/s | L2 GB/s |
|---|---|---|
| 4 | 830 | 836 |
| 8 | 1602 | 1634 |
| 16 | 2531 | 3133 |
| 23 | **2781 (92%)** | 3761 |
| 32 | **2875 (95%)** | 4421 |
| 64 | 2962 | 10794 |
| 132 | 3102 | 18285 |

**DRAM is 92% saturated at 23 SMs and 95% at 32**; the other 100 SMs are worth
8%. L2 keeps climbing over the same walk, which is what separates a device
ceiling from anything per-SM. The design consequence is the useful one: a
DRAM-bound copy needs ~32 SMs, so in a persistent kernel that leaves ~100 for
math -- an argument for sizing the producer grid to the bandwidth rather than to
the output tiling. See `tma.ctas.dev.dram.knee`.

## What gates these numbers

Every rate on this page is taken behind `verify()`, the probe's Q0 pass: the
rate kernel's own coordinate walk replayed at ring stages 1, each delivered box
compared against the source tile, run before any sweep and aborting the job on
failure. Without it a wrong descriptor, box or coordinate delivers the **wrong
bytes at the right speed** and no row here would look suspicious.

Its two halves cover different things, and the difference is the limit of the
claim. The **unswizzled** descriptor is compared byte-exact, which is what pins
the strides, the box geometry and the walk. The **SW128** descriptor the sweeps
actually use is compared as a multiset of elements, because inverting the
swizzle on the host would mean asserting the smem layout rather than testing it;
that still fails on a wrong coordinate, box or stride, and does not check the
intra-box permutation.

## Element width does not move either descriptor cap -- `tma.bytes.txn.dtype`

Both caps are in BYTES: the box row must not exceed the swizzle width, and
`boxDim[1]` must not exceed 256, at 1, 2 and 4 byte elements alike.

Legality enumerated over (element width) x (swizzle) x (box row bytes) for
`uint8`, `float16`, `bfloat16`, `float32` and `tfloat32` -- all five produce an
IDENTICAL table:

| swizzle | box row accepted up to | rejects |
|---|---|---|
| SW32 | 32 B | the next step, rc=1 |
| SW64 | 64 B | the next step, rc=1 |
| SW128 | 128 B | the next step, rc=1 |
| SW_NONE | >= 256 B | -- |

Issue rate at byte-identical geometry (8192 B row stride, 128 B box row) across
the same five: 253.2 / 253.1 / 253.0 / 253.1 / 253.1 ns per transaction at an
8 KB box, 267.2-269.0 at 16 KB, 358.6-359.5 at 28 KB -- a 0.08-0.67% spread
against a 6% noise floor. sm90 rejects the 4-bit tensor map types outright.

**Isolation.** The rate rows hold the descriptor's BYTE geometry fixed -- same
row stride, same box row bytes, same rows, therefore the same box and the same
transaction count -- so element width is the only axis. Element counts differ 4x
across the set while every byte quantity is identical.

**Falsifier.** If the caps tracked elements rather than bytes, `uint8` would
accept a 128-element (128 B) row under SW64 where `bfloat16` accepts only 64;
it does not -- both stop at 64 B. If the copy engine cared about element width,
the byte-identical rate rows would separate; they agree to 0.7%.

**Why it matters.** Halving the element width halves the ELEMENTS a legal box
carries, not its bytes. Porting a tile to fp8 by keeping its element counts
drops it out of SW128 into SW64, halving the maximum box and DOUBLING the
transactions per byte moved -- and since `tma.issue.warp` is a per-transaction
cost that is itself element-width-independent, that exactly cancels fp8's byte
saving on the copy path.

Job 565743, `probes/units/tma_ring/tma_ring.py`.

## Open gaps

- **Whether L2 has more than 18.2 TB/s.** Per-SM shared memory caps in-flight
  bytes at ~57 KB at stages 4, so the sweep runs out of range before L2
  necessarily does. A shallower ring would trade `tma.stages.warp.knee` for
  reach, which is a different measurement.
- **Why two CTAs underperform one at the same per-SM bytes.** The 1.40x is
  measured and reproducible; the mechanism is not established. Per-CTA
  outstanding-transaction limits and mbarrier arbitration are both untested.
- **The intra-box swizzle permutation.** Covered only up to a multiset by the
  Q0 gate above. A descriptor that delivered the right elements to the wrong
  offsets *inside* the box would pass. Nothing here depends on that ordering —
  the consumer is wgmma, which reads through the same layout — but a probe that
  did would need the byte-exact form.
- **Multicast.** `cp.async.bulk.tensor…multicast` is untested here; a cluster
  that broadcasts one TMA to N CTAs should count as one transaction against
  `tma.issue.warp` and N× on the product. Unverified — do not assume it.
- **The top curve bin** (8.25 MiB) holds one distinct configuration measured
  twice, not several. `tma.bw.dev.dram` is solid at 6.19 MiB, thinner above it.
- **1-D and 3-D+ descriptors, `im2col` mode.** All 2-D here. Element width is
  covered for legality and rate (`tma.bytes.txn.dtype`) but every OTHER sweep
  in this unit is bf16 only.
- **Why the mid-grid dip exists.** Characterised above, unexplained. It is the
  strongest reason in this unit to *measure* a candidate grid rather than
  interpolate one off the curve, and the two-term model stays up to 15%
  optimistic inside the band. `frontier.py` flags it rather than pretending.
- **The product law's lower edge.** `24 × 32 KB` sits 4% below other
  configurations at the same product — small-grid effects (`ld.ctas.dev.knee`) beginning
  to bind.
