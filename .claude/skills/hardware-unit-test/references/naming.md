# Tag naming — the grammar every constant is filed under

A tag is how a constant gets cited in a spec, so it has one job: let a reader
tell **what** was measured, **per what**, and **under what condition**, without
opening the file. The scheme below exists because the previous one did not.

## The grammar

```
<engine>.<quantity>.<scope>[.<condition>...]
```

Dots throughout, so a tag needs no quoting in a shell, in YAML, or inside a
Markdown citation like `[tma.bw.dev.dram]`.

| Part | Values | What it fixes |
|---|---|---|
| `engine` | `tma` `ld` `wgmma` `mma` `ffma` `sched` `launch` `atom` `cluster` `coop` | which hardware unit produced it |
| `quantity` | `bw` `issue` `lat` `stages` `bytes` `ctas` `count` `rate` `ratio` `util` `regs` `clock` `xover` `feedtax` | **the unit, implied by this field alone** |
| `scope` | `warp` `wg` `cta` `sm` `dev` `txn` `addr` `n` | the DENOMINATOR — what the quantity is "per" |
| `condition` | `dram` `l2` `ss` `rs` `chain` `indep` `ilp` `mem` `bar` `knee` `max` `curve` `dip` `geom` `warps` | source, instruction form, stall kind, or which feature of the curve |

Units implied by `quantity`, so a tag can be dimension-checked by eye:

```
bw     GB/s          issue  ns/txn or cyc/instr   lat    ns or cyc
stages count         bytes  B                     ctas   count
rate   ops/s         ratio  dimensionless         util   dimensionless
regs   count         clock  GHz                   xover  the axis value at the crossing
```

`scope` may be omitted only when the engine already fixes the denominator —
`cluster.lat.sync`, `cluster.count.max`.

### Field order carries all the meaning

There is no `@` or `/` separating the parts, so nothing but position says which
field is which. Apply it consistently: **quantity, then scope, then condition.**

This bites in exactly one way. `tma.bw.geom.cta` reads naturally in English and
is wrong — it puts the condition before the scope. The correct form is
`tma.bw.cta.geom`: bandwidth, per CTA, under the geometry condition.

## Why the grammar is shaped this way

An earlier scheme used `UPPER-KEBAB` names invented here, after the thing
rather than the measurement. It was removed from the repo on 2026-08-30. Three
failures it produced, all of which had already happened, are what the grammar
above is built to prevent:

- **Scope was ambiguous.** Three different scopes -- device-wide, per-CTA and
  the streaming-load ceiling -- all wore the word `CEIL`, so a reader could not
  tell which one a citation meant.
- **The source was invisible.** Two bandwidth ceilings, one DRAM-sourced and one
  L2-sourced, and only one said so -- by naming a cache instead of a quantity.
- **Units were not recoverable from the tag**, so a constant could be multiplied
  by the wrong thing without the expression looking wrong.
- **It did not scale.** Once instruction form, scope and dependence structure all
  have to appear in one name, `UPPER-KEBAB` produces things like
  `WGMMA-RS-LATENCY-CHAINED-PER-WARPGROUP`.

A second, smaller retirement followed inside the dotted grammar itself: three
tags spelled the ring depth as `.depth.` while `pipeline.stages.wg.knee` spelled
the same quantity `.stages.`. They are all `.stages.` now, per
[`vocabulary.md`](vocabulary.md)'s CUTLASS-sourced word.

## Migration

Complete. The invented `UPPER-KEBAB` spellings were renamed throughout the repo
on 2026-08-30 and the compatibility map was deleted with them, so there is one
spelling of every tag and `--tag` resolves nothing else:

```
$ python3 scripts/constants.py --tag <any old UPPER-KEBAB name>
no constant tagged '...'
```

The rename itself is recorded in
`.agents/notes/implemented/architecture/2026-08-28-constant-tag-rename.md`,
which is the one file still holding the old spellings -- its subject is the
rename, so it has to.

## Adding a tag

1. Pick `quantity` first — it fixes the unit, and if no listed quantity fits,
   the measurement probably is not one number.
2. Pick `scope` by asking "per what?" and answering with a denominator, not a
   topic.
3. Add `condition` only to distinguish it from a sibling that would otherwise
   collide. `tma.bw.dev.dram` needs `dram` because `tma.bw.dev.l2` exists.
4. Never name a cache, a PTX mnemonic, or a probe-internal word (`frame`,
   `trip`, `sweep`) where a machine quantity is meant.
