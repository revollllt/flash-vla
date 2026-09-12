# unit: mma — the only tensor-core path, and what it reaches

On sm90 this unit has two halves and the choice between them is the interesting
part: `[mma.xover.n.wgmma]` says tile N below 32 favours `mma.sync`, above it
favours `wgmma`. **Neither half of that survives here.** ptxas refuses every
`wgmma` form on `sm_120a` and `tcgen05` is datacenter Blackwell only
[isa.wgmma.absent], so there is one tensor-core instruction and no crossover to
measure. The question becomes how close that one instruction gets to the peak.

Probe: `lab/sm120/mma_unit.cu`.

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120f,code=sm_120f -o /tmp/mma_unit lab/sm120/mma_unit.cu
/tmp/mma_unit 20
```

## Claims, and what would have refuted them

**`[mma.issue.warp]` — 32.1 cycles per `m16n8k16`, and the accumulator count
barely moves it.**

| independent accumulator sets | cycles/mma | vs best |
|---:|---:|---:|
| 1 | 34.45 | 1.07× |
| 2 | 33.10 | 1.03× |
| 4 | 32.57 | 1.01× |
| 8 | 32.28 | 1.00× |
| 16 | 32.13 | 1.00× |

*Isolation*: operands register-resident, one warp alone on the device, no shared
memory and no memory system in the number. The accumulators are summed into a
value stored only under a condition that is false, so ptxas cannot drop the
chain and no store traffic enters the measurement.

*Falsifier, and it was checked*: a flat curve is exactly what an eliminated loop
body looks like, so the SASS was counted rather than trusted. Each
`mma_issue<NACC>` kernel contains exactly `NACC` `HMMA.16816.F32.BF16`
instructions in its loop body — 1, 2, 4, 8, 16 — and nothing was hoisted.

**The flatness is the finding.** sm90's `[mma.stages.warp.knee]` says hold four
independent accumulator sets, and there it is worth 4×: a single chain is
latency-bound at 25.14 cycles against an issue interval of 6.26. Here one chain
costs 34.45 and sixteen cost 32.13. The instruction is **issue-limited, not
latency-limited**, on this part, so the sm90 rule buys 7% and the registers it
reserves are better spent elsewhere.

**`[mma.ceiling.dev.bf16]` — 253 TFLOP/s, 60% of the derived peak.**

| CTAs | warps/CTA | TFLOP/s | of 419.4 |
|---:|---:|---:|---:|
| 170 (1/SM) | 4 | 248.8 | 59% |
| 170 | 8 | 252.5 | 60% |
| 170 | 12 | 253.2 | 60% |
| 340 (2/SM) | 4 | 249.7 | 60% |
| 340 | 8 | 251.8 | 60% |
| 340 | 12 | 249.9 | 60% |

Four warps per SM already saturates it; twelve warps and a second CTA per SM add
nothing. That much mirrors sm90's "one warpgroup saturates the tensor core".

**`[mma.ratio.dev.fp8]` — fp8 is 1.97× bf16, at the same 59% efficiency.**

A 60% shortfall invites two different explanations: the instruction is
inefficient, or `spec.py`'s derived peak is wrong. NVIDIA publishes no dense
tensor table for this SKU — only "3352 AI TOPS", which is FP4 with sparsity — so
the ladder in `spec.py` is derived, and a derived denominator is exactly the kind
of thing that should be checked before a 60% is quoted.

| | measured | derived peak | efficiency |
|---|---:|---:|---:|
| bf16 → fp32 | 253 TFLOP/s | 419.4 | 60% |
| fp8 e4m3 → fp32 | 497.8 TFLOP/s | 838.8 | 59% |

fp8 measures **1.97×** bf16. The ladder's premise — each halving of element
width doubles throughput — is therefore measured rather than assumed, and the
efficiency is the *same* at both widths. So the 60% belongs to `mma.sync` and
not to an error in the peak.

**`[mma.feedtax.warp.ldmatrix]` — 1.27× at low reuse, 1.03× at high.**

| mmas per `ldmatrix` pair | cycles/mma | vs register-resident |
|---:|---:|---:|
| 8 | 41.03 | 1.27× |
| 16 | 36.97 | 1.15× |
| 32 | 34.44 | 1.07× |
| 64 | 33.34 | 1.03× |

Comparable to sm90's ≤1.18× but it bites at a lower reuse factor, so a mainloop
that reloads every 8 mmas pays 27% here.

## What this says against sm90

| | sm90 (H100) | sm_120 (RTX 5090) |
|---|---|---|
| instructions available | `mma.sync` **and** `wgmma` | `mma.sync` only |
| `mma.sync` per-warp issue | 6.26 cyc | **32.1 cyc** |
| accumulator sets needed | 4 (worth 4×) | **1** (worth 1.07×) |
| `mma.sync` ceiling | 63% of peak | **60% of peak** |
| best reachable | **95%**, via `wgmma` | **60%** |

The per-instruction efficiency is nearly the same on both machines — 63% against
60% — which is the reassuring half. The unforgiving half is the last row: on
H100 a kernel that needed more than 63% could reach for `wgmma` and get 95%. Here
60% is the whole ceiling, because the instruction that would have gone higher
does not exist.

## What is not established

That 60% is the *hardware's* limit rather than this shape's. Only
`m16n8k16` bf16 and `m16n8k32` fp8 were measured, both with register-resident
operands. `spec.py` also lists FP6 and FP4 inputs and a block-scaled
`mma.sync.kind::mxf8f6f4` form that needs `sm_120a`/`sm_120f`
[isa.target.a_required]; none of those was measured, and the block-scaled path
is the one NVIDIA built this tensor core around. A kernel that needs more than
253 TFLOP/s of bf16 has nowhere to go, but a kernel that can use narrower inputs
has the ladder available and the ladder is now measured to hold at one step.
