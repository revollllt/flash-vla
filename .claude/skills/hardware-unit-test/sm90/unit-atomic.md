# Unit: global atomics and gmem counters

**Probe** `probes/units/gmem_atomic/gmem_atomic.{cu,py}` · **Constants** `atom.ratio.ret`,
`atom.ratio.place`, `atom.rate.addr`, `atom.ratio.width`, `atom.ratio.scope`, `atom.lat.dev.hop`

```
# launcher and output path are this host's; the probe itself takes neither
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/gmem_atomic/gmem_atomic.py \
    --json profiles/hardware-unit-test/atomic.json
```

**Everything here resolves in L2, not DRAM** — the probe's addresses fit in the
50 MB L2 by construction, which is where atomics are executed. A reduction whose
output is genuinely cold is not covered; see the gaps.

## The one thing to take away

**Layout is worth 6.3×; everything else in this unit is worth ≤1.3×.**

The atomic unit is per-transaction, like TMA. So the levers are how many
transactions you issue and how they land, not how many bytes each carries:

| lever | worth | how |
|---|---|---|
| **address placement** | **6.3×** | pack a warp's lanes into one 128 B line |
| **width** | **3.8×** bytes, free | `red.global.add.v4.f32` retires as fast as `.u32` |
| return value | 1.30× | `red` not `atom` when the old value is unused |
| scope | 1.00× | free — choose it for correctness only |

And one number that dwarfs all of them in the wrong direction: **a single
contended address sustains 1.36 Gop/s, 386× below the uncontended best.**

## Throughput, measured

`red.u32`, 33792 threads, by number of distinct addresses and their spacing:

| addresses | threads sharing each | 4 B apart | 128 B apart |
|---:|---:|---:|---:|
| 1 | 33792 | 1.4 | 1.4 |
| 4 | 8448 | 5.4 | 2.3 |
| 32 | 1056 | 12.6 | 10.5 |
| 256 | 132 | 27.2 | **64.9** |
| 2048 | 16 | **165.4** | 70.5 |
| 16384 | 2 | **525.6** | 83.1 |
| 65536 | 1 | 401.4 | 85.1 |

*(Gop/s. Bold = the better placement at that sharing level.)*

Read the two columns against each other, not down one. **The crossover is at
roughly 100 threads per address.** Below it, packing a warp into one line wins
by 5.4–6.3× — the hardware merges lanes that land in the same line, and giving
each lane its own line throws that away. Above it, the packed layout funnels
every thread through a handful of lines and loses by 2.4×.

**The peak multiple does not reproduce; the shape does.** Re-measured twice on
2026-08-30, every row above agrees to ~1% except the two highest-address 4 B
rows, which move in opposite directions at 0.4% within-run spread. The ratio is
quoted from the 16384 row, so the constant inherits that: it has read **6.30 /
5.61 / 5.39** across three runs. Design against the crossover and the direction,
not the multiple. [protocol.md rule 14b]

That single table decides a reduction's layout, and it decides it in opposite
directions for a split-K accumulate (few lanes per address → pack) and a
histogram (many lanes per bin → spread, or privatise).

## The counter protocol

Ping-pong between two CTAs, 2000 rounds, timed from the host so the number never
crosses two SMs' unsynchronised clocks:

| advance | 0 observers | 6 | 30 | 130 |
|---|---:|---:|---:|---:|
| `red.release.gpu.add` | 651.1 | 648.5 | 648.4 | 653.8 |
| `st.release.gpu` | 564.4 | 570.4 | 574.1 | 563.9 |

*(ns per arrive→observe hop.)*

Two results, both load-bearing for a task-graph megakernel:

- **A hop costs ~650 ns.** At `tma.bw.dev.dram` that is the time to move ~2 MB. A task
  doing less work than that is dominated by the ordering around it.
- **Observers are free.** 130 CTAs polling the same counter cost 0.4% more than
  none — so **one counter can gate the whole machine**, and a broadcast tree is
  solving a problem this hardware does not have.

And where a single producer owns the counter, a plain `st.release` flag is 13%
cheaper than an atomic increment. Use the atomic only when several producers
must accumulate into the same counter.

## `red.global.add` or a second kernel?

The question is usually asked as "how many partials before atomics lose". That
framing is wrong here — **it is layout, not size, that decides it.**

Accumulating `N×P` f32 partials, against writing them out and running a reduce
kernel (`launch.lat.dev.ramp` = 1240 ns, `tma.bw.dev.dram` = 3.02 B/ns, and the two-kernel path
moves the partials twice):

```
atomic path   t = 4·N·P / BW_atomic
two-kernel    t = 1240 + 8·N·P / 3.02
```

- **Well-laid-out** (packed, `BW_atomic` = 2.1 TB/s): `1.90·N·P` against
  `1240 + 2.65·N·P`. The atomic path is cheaper at *every* size — it moves half
  the bytes and pays no launch.
- **Badly laid out** (scattered u32, 340 GB/s): `11.8·N·P` against the same, and
  the second kernel wins above **N·P ≈ 136** — which is to say, immediately.

So: fix the layout and never write the second kernel. The decision only looks
like a size threshold when the atomics were going to be scattered anyway.

## Open gaps

- **Packed *and* wide is unmeasured.** `atom.ratio.place` was measured at u32 and
  `atom.ratio.width` at 128 B spacing; the combination — `v4.f32` at 16 B spacing, so
  four lanes fill a line — is the configuration a good split-K reduction would
  actually use, and its rate is an inference from two separate sweeps rather
  than a reading. **The 2.1 TB/s in the arithmetic above is the packed-u32
  number, not a measured packed-v4 number.** Measure it before trusting it.
- **Cold atomics.** Everything here is L2-resident. An atomic to an address that
  misses L2 pays a DRAM round trip that none of these numbers include.
- **A6 — what polling costs the neighbours.** `atom.lat.dev.hop` says observers are free
  *to each other*. It does not say what 130 spinning CTAs do to a concurrent
  kernel's TMA bandwidth, which is the version of the question a megakernel
  actually asks. The probe needs a mode that runs pollers alongside the
  streaming loop and reports the bandwidth loss.
- **`.acquire` / `.release` cost.** `atom.ratio.scope` shows scope is free under
  *relaxed* semantics. The ordering qualifiers themselves are only measured
  inside `atom.lat.dev.hop`'s round trip, where they cannot be separated from it.
- **Contention within a CTA** — `.cta`-scoped atomics to shared memory are a
  different unit entirely and are not covered.
