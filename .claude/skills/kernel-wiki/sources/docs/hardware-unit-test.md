---
id: doc-hardware-unit-test
title: "hardware-unit-test: the measured machine constants of this repository's H100"
url: ../../../hardware-unit-test/SKILL.md
source_category: official-doc
architectures: [sm90]
tags: [tma, wgmma, mbarrier, cluster, pdl, mma-sync]
retrieved_at: 2026-09-07
---

# hardware-unit-test

The sibling skill holds hardware unit tests, probes that isolate one engine
and measure it, and the table of constants they produce, each filed under a
tag of the form `engine.quantity.scope[.condition]` with the machine,
toolchain, validity range and the sweep that would have refuted it.

`python3 .claude/skills/hardware-unit-test/scripts/constants.py --tag <tag>`
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

## What it is not

A benchmark of any kernel, or a datasheet. Roofline peaks live in the skill's
`spec.py`; per-kernel latency belongs to `doc-benchmark-kernel`.
