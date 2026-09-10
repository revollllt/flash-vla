---
name: hardware-unit-test
description: Look up or measure GPU primitive capabilities such as bandwidth, launch cost, tensor throughput or synchronization. Use when a kernel design needs an applicable machine limit; candidate kernel timing belongs to benchmark-kernel.
---

# Hardware capabilities

Own measured hardware constants and their validity ranges. A datasheet peak,
measured primitive capability and model/kernel latency answer different questions.

1. Look up the relevant tag in the hardware axis's `measured/` table. Check its
   GPU, shape, cache/source, concurrency and toolchain conditions before reuse.
2. If the data cannot answer the current question, choose the smallest existing
   probe that isolates that mechanism. Change one meaningful variable.
3. Run on the selected GPU with the configured CUDA toolchain. Record raw output
   and conditions; do not turn a local node observation into a hardware rule.
4. Return the applicable value/range and implication for the design. Update the
   shared table only for a reusable hardware finding, not every kernel trial.

## Example

Look up a TMA issue cost from the project root before designing a pipeline:

```bash
python .claude/skills/hardware-unit-test/scripts/constants.py --tag tma.issue.warp
```

If the selected geometry needs a targeted probe:

```bash
python .claude/skills/hardware-unit-test/probes/units/tma_ring/tma_ring.py \
  --sweeps A --json artifacts/tma.json
```

Use [the probe protocol](references/protocol.md) for isolation and cache policy.
`CUDA_HOME`, `CUTLASS_DIR` and `HW_UNIT_TEST_CACHE` configure builds;
`HUT_CONSTANTS_ROOT` selects an external measured table. Host setup and scheduling
are user-local. Compare implemented kernels with
[benchmark-kernel](../benchmark-kernel/SKILL.md), and inspect their counters with
[ncu-report](../ncu-report/SKILL.md).
