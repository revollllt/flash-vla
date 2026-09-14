# CPU screen: full-row BF16 softmax fused with expert PV

One fixed mapping is resource-feasible and testable: output tile 16x32, K64,
four warps, 200 CTAs. It retains complete FP32 logits, computes a full-row
softmax, explicitly rounds every probability to BF16 in shared memory, then
performs BF16-input/FP32-accumulator PV. Its main risk is eightfold duplicated
softmax work. There is no speed prediction or demonstrated end-to-end gain.
This screen performed only source/JSON reads and CPU arithmetic.

Inspected main revision: 6982545. Current route is triton-qk-attention.
Q/out is BF16[400,256], logits FP32[400,1018], V BF16[1018,256], mask BF16[1018].
QK remains the current 32x32x64 Triton kernel and still writes the full logits.
The new boundary would replace only native softmax + torch PV, including its
split-K reduction.

## The single mapping

Grid=(25,8), block=128. Each CTA owns 16 query rows and 32 output features.
Each of its four warps computes four rows sequentially. For one row, lane l
holds keys l+32*i for i=0..31: 32 FP32 values/lane. It applies the same 0.0625
scale and runtime additive mask, computes the full 1018-key maximum and sum,
uses expf, normalizes, and explicitly converts the final probabilities with
BF16 round-to-nearest. Six padded keys use -inf scores and zero P/V contributions.

The 16x1024 BF16 probability tile lives in CTA shared memory. All probability
rows are finished before the PV phase. After a CTA barrier, four warp MMA tiles
of 16x8 cover the CTA's 16x32 output. Sixteen K64 iterations consume the shared
P and staged V and accumulate FP32, followed by BF16 output. No partial-softmax
PV is accumulated or rescaled; normalization precedes each probability's BF16
rounding. This excludes the ordinary online FlashAttention rounding order.

This warp-row reduction tree differs from the current eight-warp, 256-thread
native reduction. It retains the requested FP32/full-row/BF16 sequence, but P
is not claimed bitwise equal. The same expf/scalar fusion settings also need
preserving. Existing numerical tolerances must check P and final PV on actual
inputs before timing. If retaining the native reduction order bitwise is an
additional requirement, this mapping does not qualify: emulating its two
virtual groups on 128 threads would process 16 rows serially and require at
least 32 CTA reductions/barriers, materially changing the proposal.

A future implementation would need a CuTe/CUTLASS warp MMA path consuming this
shared P directly; calling a device GEMM would create another launch. The
installed generic global-input Stream-K GEMM is not a drop-in fused mainloop.
The exact CuTe copy/operand layouts have not been instantiated or compiled.
No new framework, general FlashAttention rewrite, cluster, or DSM path is proposed.

## Parallelism, padding, and resources

The mapping reuses the only locally promising 018 PV output tile, avoiding the
slower 16x16 PV tile and the already rejected 1018->1024 global padding dispatch
idea. Its 200 CTAs exceed 170 SMs; this is not proof of actual residency or
balanced assignment. Unlike the prior QK+softmax fusion screen, it does not
recompute QK or require a 16x1024 FP32 MMA result in a CTA.

M=400 and N=256 divide their tiles exactly. Only K pads 1018->1024 (+0.5894%).
The logical four-warp m16n8k16 decomposition has no additional M/N padding:
256 MMA instructions/CTA, 51200/call, 209715200 padded FLOPs versus 208486400
useful FLOPs. This is the intended ISA layout, not a compiled new lowering.

| Live allocation or work | Lower bound / concrete budget |
|---|---:|
| BF16 P shared, 16x1024 | 32768 B |
| Three BF16 V stages, 64x32 | 12288 B |
| One shared runtime mask, padded to 1024 | 2048 B |
| Total before swizzle padding/pipeline state | 47104 B = 46 KiB |
| Softmax held values | 32 FP32 scalars/lane, 16384 B/CTA |
| PV accumulator | 4 FP32 scalars/lane, 2048 B/CTA |

Softmax held values need not overlap the PV accumulator's lifetime. Indices,
reduction state, operand fragments, pipeline state, and compiler allocation are
additional. Neither 32 nor 4 is a prediction of total registers/thread.
The shared budget is below the measured 99 KiB/CTA limit and permits at most
two such allocations within 100 KiB/SM before padding; this does not establish
occupancy. The 018 three-stage count is only the fixed starting budget.

Measured BF16/FP32 tensor throughput is 512 FLOP/cycle/SM at 4+ warps.
Each CTA's MMA-only work is at least 2048 SM cycles. With 200 CTAs on 170 SMs,
some SM must perform at least two CTA workloads, at least 4096 MMA cycles
(~1.41 us at the historical illustrative 2.9 GHz). All softmax, memory,
barriers, launch, and scheduling costs are excluded. It is not a runtime
estimate or proof of where the extra CTAs execute.

## Duplicated work and traffic

Every query's P is recomputed once for each N32 output tile: eight copies.
Actual exponent evaluations rise from 407200 to 3257600/call; full padded
slots total 3276800. Four rows per warp execute sequentially. Fewer CTAs do
not remove this arithmetic. Warp reductions avoid per-row CTA barriers, but
their sum order differs from the control.

Logical global bytes per call, excluding cache hits and transaction rounding:

| Object | Proposed mapping |
|---|---:|
| FP32 score reads, eight copies | 13030400 B |
| BF16 V reads, once per 16-query group | 13030400 B |
| BF16 mask loaded once per CTA into shared | 407200 B |
| BF16 output writes | 204800 B |
| Global BF16 P writes/reads | 0 |

The score working set itself is only 1628800 B; V is 521216 B. The native
softmax reads scores once and writes 814400 B of P. Relative specifically to
native softmax + the 018 tiled PV, the fusion removes 814400 B P stores plus
6515200 B tiled P reads, but adds 11401600 B score rereads: net +4072000 B
logical score/P global traffic. V tiling is the same. The current cuBLAS
split-K PV has a different mapping, so these exact P-read counts must not be
attributed to that control without further evidence.

The fusion also writes 6553600 B of padded BF16 P to shared over all CTAs;
warp MMA A reuse can reread it within the CTA. Shared transactions and conflicts
depend on the uncompiled layout. Cache capacity alone does not establish hit
rate. The repo's cold-DRAM bandwidth fit and 340-CTA cold-read knee do not price
these immediate score rereads; no relevant L2/SFU throughput measurement exists
in the consulted unit results. Do not multiply the native softmax time by eight
as a timing prediction, or assume rereads are free.

## Existing timing budget and what the new screen would establish

Profile019 used the older torch QK; its QK time is not current. The unchanged
downstream kernels totaled across 180 calls:
- native softmax: 383.206 us (2.129 us/call);
- torch PV main kernel: 736.895 us (4.094 us/call);
- split-K reduce: 279.468 us (1.553 us/call).

The replaced region is 1399.569 us, or 7.775 us/call of profiler kernel duration.
The softmax->PV gap adds 15.544 us total (0.096 us median). These diagnostic
numbers are a budget scale, not the uninstrumented baseline for a new trial.
Two graph nodes disappear, but the measured empty-node cost is about
0.45-0.52 us near this grid scale, not the 2.05 us in-stream launch cost.

018's 16x32x64 PV was one 200-CTA/128-thread MMA kernel, passed nine actual
pairs, and improved the local nine-pair boundary by 0.76-0.79 us/call. Its
deployment experiment was reverted as inconclusive: eight-round mean advantage
0.05724 ms with reverse-order medians overlapping amid clock drift. The new
hypothesis therefore needs benefit beyond simply repeating that PV change.
Softmax launch/P materialization disappear, but eightfold softmax arithmetic
and changed reduction order are new costs/risks. Neither the 018 local gain
nor the profile softmax duration can be booked as a deployed saving.

The minimal future test is this one layout on one saved actual logits/mask/V
pair: inspect BF16 P and PV numerical error first, then compare the complete
softmax+PV boundary with identical immutable inputs. Only a clear local gain
would justify representative calls and the full current-QK attention boundary.
No tile sweep or additional PV-only tuning is motivated by this screen.

## Mask, prefix reuse, and Q alias

The pipeline shares one runtime key mask across queries and appends each step's
50 K/V rows after the 968-row prefix. V has a single KV head; all 400 flattened
query rows share it. Prefix V is reusable across steps of a layer, but logits
and P change with Q, so no prefix probabilities or normalization can be cached.
The finite BF16 mask value must be read, not replaced by an assumed prefix cut
or -inf. Cache residency across the deployed layers/steps is not established.

Out aliases Q in the actual pipeline. This fusion reads only logits, mask and V,
after the same-stream QK kernel has completed reading Q. It can overwrite Q
without copying or retaining the original Q in production. A full-attention
ABBA must still reset Q or cycle immutable snapshots before QK; a softmax/PV
probe alone does not read Q and can overwrite its output freely.

Sources: triton_qk_attention.py, fused_attention.py/.cu, inherited
h100/pi05/pipeline.py:104/145/208; profile019 and its expert trace; 018 PV
README/dispatch.json and measurements/018-decision.json; measured/unit-mma.md,
unit-launch.md, constants.yaml; prior gpt6-attention-qk-softmax-screen/README.md.
