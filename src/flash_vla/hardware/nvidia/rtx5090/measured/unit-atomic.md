# unit: atomic — global reductions, and which levers still pay

The skill's `gmem_atomic` probe through `lab/sm120/run_skill_probe.py`. Every
number resolves in L2, not DRAM.

```bash
CUDA_HOME=... CUTLASS_DIR=... python3 lab/sm120/run_skill_probe.py gmem_atomic
```

## The headline: one sm90 rule is dead here, the biggest one survives

**`[atom.ratio.ret]` — `red` and `atom` cost the same.** 92.04 against 90.78
Gop/s uncontended, inside the 3–6% run noise, and the same for `f32`, `exch` and
`cas`. sm90's rule — "any accumulate that does not need the old value should be
`red`, worth 30%" — **buys nothing on this part**. The return value is free.

**`[atom.ratio.place]` — address layout is still worth 5.5×.** 65536 addresses
at 4 B spacing reach 503.1 Gop/s; the same count at 128 B spacing reaches 92.0.
Close to sm90's 5.4–6.3×, and still the largest lever in the unit.

| addresses | spacing | threads/address | Gop/s |
|---:|---:|---:|---:|
| 1 | 4 B | 33792 | 2.42 |
| 4 | 4 B | 8448 | 9.70 |
| 32 | 4 B | 1056 | 15.53 |
| 256 | 4 B | 132 | 31.05 |
| 2048 | 4 B | 16.5 | 240.0 |
| 16384 | 4 B | 2.1 | 449.0 |
| 65536 | 4 B | 0.5 | 503.1 |

**`[atom.rate.addr]` — one address is worth 2.42 Gop/s.** 1.8× sm90's 1.36, so a
contended counter hurts less here — but it is still 208× below the uncontended
rate, so the advice is unchanged: more accumulators, nothing else.

**`[atom.ratio.width]` — the widest atomic is free.** `u32`, `f32`, `f16x2`,
`bf16x2`, `v2f32` and `v4f32` all run within 1% of each other on operation rate,
so `red.v4f32` moves **4.25×** the bytes for nothing.

**`[atom.ratio.scope]` — scope is free.** `cta`, `gpu` and `sys` agree to 0.6%
for both `red` and `atom`, contended and uncontended. Choose scope for
correctness; a too-narrow one fails silently and data-dependently.

**`[atom.lat.dev.hop]` — 400 ns per arrive→observe hop, and observers are free.**

| observers | CTAs | ns/hop (`red.release.add`) | ns/hop (`st.release`) |
|---:|---:|---:|---:|
| 0 | 2 | 400.4 | 393.2 |
| 6 | 8 | 400.7 | 393.3 |
| 30 | 32 | 402.4 | 393.5 |
| 130 | 132 | 413.1 | 395.7 |

Against sm90's 651 ns this is 1.6× cheaper, so ordering a task separately costs
less here than the H100 numbers imply. At `[ld.bw.dev.dram]` a 400 ns hop is
about 0.6 MB of traffic — the amount of work a task must exceed to be worth
ordering on its own.
