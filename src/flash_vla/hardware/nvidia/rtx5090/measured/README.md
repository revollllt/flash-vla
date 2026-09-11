# sm120 — the measured layer of the RTX 5090 hardware axis

What this machine actually costs, beside the datasheet peaks in `../spec.py`:
those two are the roofline's denominator and the ceiling's. **Nothing here
transfers to another architecture**, and nothing from the sm90 table transfers
*to* here; the arch-independent layer — what to measure and how — stays in the
`hardware-unit-test` skill's `references/category-*.md` and
`references/protocol.md`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (GB202), 170 SMs, `sm_120a` |
| smem | 101376 B per CTA (99 KB opt-in), 102400 B per SM |
| L2 | 96 MB · GDDR7 datasheet peak 1.792 TB/s |
| clocks | not pinned — noise floor **not yet established** |
| toolchain | CUDA 13.1 (V13.1.80), driver 580.142, PTX ISA 9.1 |

## The state of this table

The instruction set, the static limits and the **`launch` unit** are measured.
`tma`, `mma`, `atomic` and `coop` are not, and `constants.py --validate` reports
four GAPs — that is the accurate picture, not an oversight.

**The floor model now has a denominator**: `[ld.bw.dev.dram]` is
`t_us = 3.93 + MB/1.539` here, against sm90's `1.85 + MB/2.77`. Both terms are
worse — fixed cost 2.12×, marginal rate 1.80× — while the marginal rate is 90%
of this part's datasheet peak, so the memory system is efficient and the
*overhead* is what moved. A launch costs 2.05 µs against sm90's 1.24 and, unlike
sm90, does not rise with grid size: there is no ramp to amortise.

Still true, and still the thing to guard: nothing else transfers. Quoting an
sm90 `tma`, `mma`, `atomic` or `coop` constant here produces a number with no
evidence behind it.

## Consulting

```bash
# from the repository root
python3 .agents/skills/hardware-unit-test/scripts/constants.py --machine sm120
python3 .agents/skills/hardware-unit-test/scripts/constants.py --tag isa.wgmma.absent
python3 .agents/skills/hardware-unit-test/scripts/constants.py --validate
```

Cite tags — `[cluster.count.max]`, `[smem.bytes.cta.max]` — so a claim traces to
the probe that established it.

## What is measured here

### Instruction set (`isa-support.md`)

**The port is not a flag change.** `wgmma` and `tcgen05` are both refused on
`sm_120a`, so consumer Blackwell has no asynchronous MMA of any kind and every
tensor-core mainloop in this repository has to be rebuilt on warp-level
`mma.sync`. The producer side survives whole: TMA, `mbarrier` with transaction
counts, `cp.async`, `ldmatrix`/`stmatrix`, `elect.sync`, `fence.proxy.async` and
PDL all assemble unchanged.

Three features need the architecture-conditional `sm_120a` target and are
*refused*, not degraded, on plain `sm_120`: `setmaxnreg`, block-scaled `mma`, and
`ldmatrix .m16n16`. Warp specialisation in this repository uses the first of
those.

51 instructions were decided by `ptxas` itself, one minimal `.entry` each. The
full matrix, the verbatim diagnostics and the per-unit consequences are in
[isa-support.md](isa-support.md).

### Static limits (`[cluster.count.max]`, `[smem.bytes.cta.max]`)

Clusters work here, up to 8, with distributed shared memory — launched and
checked rather than assumed, because a widely-mirrored community summary claims
consumer Blackwell is limited to cluster size 1 and that is false on this device.
Cluster 16 is refused.

Shared memory per block is **99 KB against sm90's 227**, and max warps per SM is
**48 against 64**. A tile budget inherited from sm90 overshoots shared memory by
2.29x and fails at launch rather than degrading.

### Software stack (`toolchain.md`)

TileLang 0.1.11 works on `sm_120`, warp specialisation included — it compiles to
an `mbarrier` producer/consumer pipeline and emits no `setmaxnreg`, so plain
`sm_120` suffices for it. torch bf16 and CUDA graph capture/replay both work.
nvcc compiles `sm_90a` here (compile-only), which makes a shared-tile refactor
verifiable by PTX equality. See [toolchain.md](toolchain.md).

### Profiler visibility (`ncu-metrics.md`)

NCU counter names moved on GB202: the whole `sm__inst_executed_pipe_tensor_op_*`
family is gone (26 metrics on GH100, 0 here) and `dram__bytes_read` is now
`dram__bytes_op_read`, so existing saved queries fail to resolve rather than
returning zero. Counter access is **denied on this host** (`ERR_NVGPUCTRPERM`),
so NCU is unavailable — but the CUPTI timeline is not affected and the top-down
model profile still runs. See [ncu-metrics.md](ncu-metrics.md).

## What is NOT measured here, and what would measure it

| unit | blocked | next |
|---|---|---|
| `tma` | ring depth, box size, CTA count for any copy pipeline | skill's `tma_ring` uses only surviving instructions; needs `arch_flags=["-gencode","arch=compute_120a,code=sm_120a"]` and a run |
| `mma` | every tensor-core tile decision | needs a **new** probe: the skill's `mma_rate` is built on `wgmma` |
| `atomic` | reduction layout | skill's `gmem_atomic` has no sm90-only construct; run it |
| `coop` | persistent-kernel budgets | skill's `coop_launch` likewise |

Of these, `launch` and `mma` are the two that block real design work: the first
because every fusion decision in this repository is denominated in launch cost,
the second because there is no longer a choice of tensor-core instruction and
the one that remains has never been characterised on this part.

## The biggest thing still untested here

**What TMA `.multicast::cluster` costs here.** No longer a question of whether it
works — NVIDIA's Target ISA Notes say it assembles on any `sm_90`-or-higher
target but is optimized only for `sm_90a`/`sm_10xa`/`sm_11xa` and their families,
and "may have substantially reduced performance on other targets". `sm_120a` is
not on that list. So the open question is the size of the penalty, and no probe
here issues a multicast TMA or counts its delivered bytes. An sm_90a design that
treats multicast as a free fan-out does not port on that assumption.

Second: **the noise floor.** sm90's table carries ~6% and an explicit "clocks not
pinnable"; this machine's variability has not been characterised at all, so no
comparison here has an error bar yet.
