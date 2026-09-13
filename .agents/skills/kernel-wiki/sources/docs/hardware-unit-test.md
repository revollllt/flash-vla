---
id: doc-hardware-unit-test
title: "hardware-unit-test: the measured machine constants of this repository's machines"
url: ../../../hardware-unit-test/SKILL.md
source_category: official-doc
architectures: [sm90, sm120]
tags: [tma, wgmma, mbarrier, cluster, pdl, mma-sync]
retrieved_at: 2026-09-07
---

# hardware-unit-test

The sibling skill holds hardware unit tests, probes that isolate one engine
and measure it, and the table of constants they produce, each filed under a
tag of the form `engine.quantity.scope[.condition]` with the machine,
toolchain, validity range and the sweep that would have refuted it.

**One table per hardware axis**, under
`src/flash_vla/hardware/<vendor>/<arch>/measured/`, each with its own
`machine:` block naming the GPU and its `arch`. Today that is
`sm90-h100-sxm5` (42 constants) and `sm120-rtx5090` (29). `constants.py`
loads every axis unless `--machine` selects one.

**The tag namespace is shared across axes and the numbers are not.**
`ld.bw.dev.dram` exists in both tables with different values, so a bracketed
citation means *the machine this page's `architectures` names*. A page that
cites a constant for an architecture it does not list is citing the wrong
machine, and neither the validator nor the tag can catch it.

`python3 .agents/skills/hardware-unit-test/scripts/constants.py --tag <tag>`
renders one constant in full; `--validate` states what `status: measured`
must clear. A wiki page cites a constant by its bracketed tag and never
restates the number; the validator resolves every bracketed tag against this
table.

## What it is the authority for

Machine numbers: TMA delivery rate, latency and saturation frontier; global
atomic throughput and counter-hop latency; wgmma issue rate against N, stage
knee and second-warpgroup ratio; launch and grid-ramp cost; streaming
bandwidth ceilings including the cold-burst curve; cluster barrier cost and
placement limit; occupancy knees.

Not every unit exists on every axis, and that is itself the finding: the
sm120 table carries `mma.sync` rate by accumulator width where the sm90 table
carries wgmma issue rate, because consumer Blackwell has no `wgmma`.

## What it is not

A benchmark of any kernel, or a datasheet. Roofline peaks live in the skill's
`spec.py`; per-kernel latency belongs to `doc-benchmark-kernel`.
