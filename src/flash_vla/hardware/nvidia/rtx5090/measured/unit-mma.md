# unit: mma — the only tensor-core path, and it already runs at the peak

On sm90 this unit is half about `wgmma` and half about the crossover between it
and `mma.sync`. Neither survives here: ptxas refuses every `wgmma` form on
`sm_120a` and `tcgen05` is datacenter Blackwell only [isa.wgmma.absent], so
there is one tensor-core instruction and no crossover to measure.

Probes: `lab/sm120/mma_unit.cu` (issue interval, accumulator knee, ldmatrix tax)
and `lab/sm120/mma_clock.cu` (the ceiling, measured clock-free).

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120f,code=sm_120f -o /tmp/mma_clock lab/sm120/mma_clock.cu
/tmp/mma_clock 20
```

## The correction this unit exists to record

An earlier version of this file said `mma.sync` reaches **60% of peak** and that
there is nothing to reach past it with. **Both halves were wrong**, and the way
they were wrong is worth more than the number.

The claim compared a *measured* 253 TFLOP/s against a *derived* 419.4 TFLOP/s
peak. Two independent errors, in opposite directions:

1. **The peak had no accumulator width.** `spec.py` derived 419.4 from the FP32
   lane peak. NVIDIA does publish 419 TF dense for this part — and on their own
   forums it is stated to be the **FP16-with-FP16-accumulate** figure. Consumer
   Blackwell runs **fp32 accumulate at half rate**. Every mainloop in this
   repository accumulates in fp32, so its peak was 209.6, not 419.4.
2. **The measurement used wall time against an assumed clock.** 253 TFLOP/s is
   real, and it is *above* the 209.6 that the marketed 2.407 GHz boost implies —
   which should have been the tell, since a kernel cannot exceed the peak. This
   part runs at ~2.89 GHz under a pure tensor load.

Multiplying a measured numerator by a datasheet denominator is how a kernel at
100% of the hardware looks like it is at 60% of it.

**The fix is to measure the clock-free quantity.** FLOP per cycle per SM is a
property of the hardware; every TFLOP/s figure, NVIDIA's included, is it times a
clock.

## The measurement

`mma_clock.cu` records per-SM `clock64()` spans alongside host wall time, so the
per-cycle rate and the achieved clock both fall out and neither is assumed.

| dtype | accumulate | warps/SM | FLOP/cycle/SM | TFLOP/s | GHz |
|---|---|---:|---:|---:|---:|
| bf16 | fp32 | 4 | 507.5 | 248.9 | 2.884 |
| bf16 | fp32 | 8 | **511.5** | 252.4 | 2.903 |
| bf16 | fp32 | 12 | **511.5** | 253.2 | 2.912 |
| fp8 e4m3 | fp32 | 4 | 1015.6 | 497.9 | 2.884 |
| fp8 e4m3 | fp32 | 8 | **1023.0** | 497.9 | 2.863 |
| fp16 | **fp16** | 4 | 929.5 | 449.2 | 2.842 |
| fp16 | **fp16** | 8 | **1023.9** | 496.9 | 2.855 |

Three clean powers of two: **512**, **1024**, **1024**.

**`[mma.rate.sm.bf16]` — 512 FLOP/cycle/SM is the peak, and `mma.sync` reaches
it.** Not 60% of something; 99.9% of 512. The instruction leaves nothing on the
table.

**`[mma.ratio.sm.acc]` — fp32 accumulate is exactly half rate.** 1023.9 against
511.5 at the same shape. This confirms NVIDIA's 419 TF as the fp16-accumulate
figure (1024 × 170 × 2.407 GHz = 419.2) and settles that a fp32 mainloop's peak
is 209.6 at that clock.

**`[mma.ratio.sm.fp8]` — fp8 is exactly 2× bf16** at equal accumulator width,
as the width ladder implies.

**`[mma.clock.sm]` — ~2.89 GHz under load**, above the 2.407 GHz marketed boost
*and* above the 2.550 GHz the driver reports as this device's maximum. That is
20% of headroom that a datasheet-derived TFLOP/s figure silently loses.

## Everything is internally consistent

The single-warp figure from `mma_unit.cu` closes the loop. Each SM has four
sub-partitions with one tensor core each; one warp gets one of them.

- One warp measured **32.13 cycles** per `m16n8k16`.
- 4096 FLOP ÷ 32 cycles = 128 FLOP/cycle per tensor core.
- × 4 tensor cores = **512 FLOP/cycle/SM**. ✓

It also explains the one result that looked anomalous before:

**`[mma.stages.warp.knee]` — accumulator count barely matters (1.07× from 1 set
to 16).** sm90's rule says hold four independent accumulator sets, and there it
is worth 4× because a single chain is latency-bound. Here the tensor core is
**issue**-limited at 32 cycles per instruction, so one dependence chain already
saturates it and there is nothing for more chains to overlap with. The registers
that rule reserves are better spent elsewhere.

**`[mma.issue.warp]` — 32.1 cycles per mma per warp**, against sm90's 6.26. That
is not inefficiency, it is a smaller tensor core: 128 FLOP/cycle per tensor core
here against Hopper's ~946.

**`[mma.feedtax.warp.ldmatrix]` — 1.27× at 8 mmas per `ldmatrix` pair**, 1.15×
at 16, 1.07× at 32, 1.03× at 64. Comparable to sm90's ≤1.18× but biting at a
lower reuse factor.

## What this says against sm90

| | sm90 (H100) | sm_120 (RTX 5090) |
|---|---|---|
| instructions available | `mma.sync` **and** `wgmma` | `mma.sync` only |
| bf16 fp32-acc peak, FLOP/cycle/SM | ~3784 | **512** |
| what the best instruction reaches | 95% of peak, via `wgmma` | **~100% of peak**, via `mma.sync` |
| device bf16 fp32-acc, sustained | ~850 TFLOP/s | **~253 TFLOP/s** |
| fp32 accumulate penalty | none | **2×** |

The corrected picture is more useful than the wrong one, and points somewhere
different. There is no instruction to switch to and no efficiency to recover:
`mma.sync` is already at the hardware ceiling, and the ceiling is simply 3.4×
lower than H100's device-wide. A kernel that is tensor-bound here cannot be
tuned out of it.

What *is* available, and is not available on H100, is the accumulator and the
input width. fp16 accumulation doubles the rate; fp8 input doubles it again.
Both are real 2× levers on this part and neither exists on Hopper, where fp32
accumulate runs at full rate.

## What is not established

The fp4 and block-scaled forms, which need `sm_120a`/`sm_120f`
[isa.target.a_required] and are what NVIDIA built this tensor core around. The
marketed 838 TFLOP/s dense fp8 implies 2048 FLOP/cycle/SM — twice what fp8 with
fp32 accumulate measures — which is consistent with the half-rate rule applying
to fp8 too, but the narrower-accumulator fp8 form was not measured. Nor was
`tf32`, which the instruction accepts.

Whether fp16 accumulation is numerically acceptable for any call site in this
repository is a correctness question, not a hardware one, and nothing here
answers it.
