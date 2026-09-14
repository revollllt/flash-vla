# CPU screen: fusing expert QK and full-row softmax

Stop the straightforward query-row CTA fusion. A transposed four-query design
is resource-feasible but remains an uncertain, different layout rather than
a clear low-cost win. This screen does not establish that every possible
QK/softmax fusion is slower. No kernel was implemented or compiled.

## Current work and recoverable budget

Current triton_qk_attention.py has Q(400,256), K(1018,256), one 32x32x64
QK tile, grid (13,32)=416 CTAs. It stores FP32 logits(400,1018).
The unchanged native softmax launches 400 CTAs of 256 threads. Each thread
holds four columns at tid+256*i; it applies FP32 scale and runtime mask,
warp XOR reductions, an eight-warp shared reduction, expf, and BF16 rn output.
There are two CTA barriers and 64 shared bytes in the existing softmax.
PV still consumes materialized BF16 P.

The retained profile019 is before the new Triton QK, so its QK duration must
not be presented as the current QK timing. It has:
- old torch QK: 803.068 us over 180 launches;
- unchanged native softmax: 383.206 us, 180 launches, median 2.112 us;
- preceding QK-to-softmax GPU gap: 15.8916 us total, median 0.0957 us.

Eliminating the whole softmax launch duration plus this gap would offer
0.3991 ms per model invocation before replacement work. A real fusion must
still execute its math; this is a favorable ceiling on that portion, not an
expected saving. It could also eliminate the logits store/load:
1628800 bytes each, 3257600 bytes/call or 586368000 bytes over 180 calls.
That is logical traffic, not measured DRAM traffic, and is not an independent
time saving to add to the softmax duration above.

Trace: /home/ubuntu/flash-vla/results/pi05-rtx5090/gpt6-run-01/profiles/019-expert/0_shipped_action_expert.json
Summary: /home/ubuntu/flash-vla/results/pi05-rtx5090/gpt6-run-01/profile-019-expert-summary.json

## One CTA owns the complete row reduction

The table assumes 1024 padded key positions, with the six invalid positions
masked. "Work" is the physical tensor-MMA arithmetic relative to the useful
query rows, excluding the common key tail padding. "Score" is the logical
FP32 score storage; an unsliced MMA fragment can be larger for padded queries.

| Orientation / queries per CTA | CTAs | Work | Score KiB | K reads MB/call |
|---|---:|---:|---:|---:|
| Q as MMA M, 32 | 13 | 1x | 128 | 6.776 |
| Q as MMA M, 16 | 25 | 1x | 64 | 13.030 |
| Q as MMA M, 4 | 100 | 4x | 16 | 52.122 |
| Q as MMA M, 2 | 200 | 8x | 8 | 104.243 |
| Q as MMA N, 8 | 50 | 1x | 32 | 26.061 |
| Q as MMA N, 4 | 100 | 2x | 16 | 52.122 |
| Q as MMA N, 2 | 200 | 4x | 8 | 104.243 |

K reads assume every CTA loads its entire 521216-byte K operand once.
They do not predict L1/L2 hits or DRAM bytes. Current tiled QK logically reads
K once per query tile, about 6.776 MB, and Q once per key tile, about 6.554 MB.

The conventional orientation has a 16-query MMA floor. The transposed form
computes K@Q.T, placing query count on the eight-column MMA axis. This halves
the minimum useful query grouping and must not be dismissed using the
conventional m16 padding argument. Its FP32 result must be rearranged back to
query-major order before the original row softmax. The n8 case is an
ISA-granularity theoretical layout, not a verified Triton tl.dot layout.
This screen did not compile that N dimension or inspect its lowering;
Triton may impose a larger tile or different padding. The table therefore
does not establish a directly expressible Triton kernel or its actual work.

Hardware evidence from measured/unit-mma.md gives 512 BF16-with-FP32-accumulate
FLOPs/cycle/SM. A full-key conventional m16 tile needs at least 16384 SM cycles;
the transposed n8 tile needs at least 8192. These exclude all loads, score
layout changes, reductions, exp, stores and launch work. With 170 SMs:
- conventional 16-query / 25-CTA layout exposes only 25 SMs;
- conventional two-query / 200-CTA layout requires at least two CTA workloads
  on some SMs, giving at least 32768 pure MMA cycles there;
- transposed eight- or four-query layouts have 50 or 100 CTAs and at least
  8192 pure MMA cycles each;
- transposed two-query / 200-CTA layout has at least 16384 pure MMA cycles on
  the SMs assigned two CTAs.

At an illustrative 2.9 GHz, 8192/16384/32768 cycles mean 2.82/5.65/11.30 us.
This uses the historical MMA unit's clock scale, not a new clock measurement.
The cheap layouts that fill all SMs therefore spend much of or more than the
existing QK+softmax time scale on MMA alone.

## Resource feasibility and remaining uncertainty

The measured limits are 99 KiB shared/CTA, 100 KiB shared/SM and 64K registers/SM.
A 32-query full score tile already exceeds shared capacity. Keeping it in
registers would average 256 FP32 accumulator registers per thread at 128
threads before operands and softmax state; increasing threads does not repair
the 13-CTA grid.

A 16-query CTA can stream key chunks and retain scores in shared:
64 KiB scores + 8 KiB Q + two 64x64 BF16 K buffers (16 KiB) +
about 1 KiB row-reduction scratch is about 89 KiB. It fits on paper, but leaves
only 25 active SMs and serializes or groups sixteen row softmax operations.

The transposed four-query case is the borderline survivor. Its logical scores
need 16 KiB, with up to a 32 KiB padded full MMA accumulator if held at once.
At 256 threads that padded accumulator averages 32 FP32 scalars/thread,
before operands, indexing and reductions. Streaming key chunks can fit
16 KiB shared scores + 2 KiB Q + two 128x64 BF16 K buffers (32 KiB) below 99 KiB.
These are layout budgets, not compiler register/shared allocation results.
Four independent copies of the exact native 256-thread reduction would use
1024 threads; fewer threads require serial row groups or a different reduction
mapping. A standard tl.softmax is not evidence of retaining the exact expf,
rounding and reduction sequence.

This transposed case raises total logical K reads to 52 MB while removing only
3.26 MB of score traffic (and reducing repeated Q reads). K is only 0.52 MB
and can be cache-resident, so this alone is not proof of a slowdown. The repo
does not provide a measured L2 regime for this access pattern. There is no
counter-supported timing margin for it, and current QK is already a short
kernel. It is not promoted to implementation in this bounded screen.

Conditionally, if the current card, driver, runtime and compiler path support
the required eight-CTA clusters, DSM access and cluster synchronization, one
16-query group could split across eight key CTAs: 25 clusters / 200 CTAs,
8 KiB score per CTA, no duplicated QK. Shared/DSM exchange could then gather
full rows for softmax. This screen did not test that support on the current
card/software combination; historical hardware notes do not validate this
prospective kernel path. These counts are conditional layout arithmetic,
not evidence of availability. It would also be a new multi-CTA pipeline,
with unmeasured communication cost, outside this bounded fusion task.

Sources: current triton_qk_attention.py and fused_attention.cu; profile019;
src/flash_vla/hardware/nvidia/rtx5090/measured/unit-mma.md and isa-support.md.
All work in this screen was source/JSON reading and CPU arithmetic. No Torch
or Triton import, GPU work, JIT, NVCC, or model loading was performed.

This fusion direction stops after the CPU screen; no implementation follows.
