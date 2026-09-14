# Fixed 128-thread native softmax screen

The fixed experiment did not establish a gain above its control drift, and
is stopped. Full tested source remains at 6f8aec5125349b7e820d857d3a25c188de70af18
on branch gpt6-pi05-softmax-128. That commit includes the 80818f9 vision revert.
This evidence-only delivery leaves the production native source unchanged
and retains its exact experimental delta in pi05_attention_softmax_128.patch.
When that delta is applied, both 256- and 128-thread instantiations live in the
same fused_attention library; the production launch remains 256 threads.

The native source is minimally templated for Threads, Items=1024/Threads,
and Warps=Threads/32. Both versions retain scalar coalesced loads, FP32
logits*scale followed by runtime BF16 mask addition, max subtraction, expf,
FP32 sums, reciprocal division, multiplication and explicit BF16 RN output.
The loader and its -O3, --fmad=false, -arch=sm_120a flags are unchanged.
No fast-math option, 512-thread variant, new softmax algorithm or QK fusion
is introduced. Changing the column-to-thread map changes the FP32 sum
association; bitwise equality of probabilities is measured, not assumed.

019 and 021 traces show 383.206 and 383.423 us over 180 softmax calls,
medians 2.112 and 2.113 us. The current kernel has 400 CTAs, 256 threads,
32 registers/thread, and 64 logical shared bytes. At source level, 128 threads
reduce the four five-step warp XOR reductions from 160 to 80 warp shuffle
instructions per CTA and the replicated shared lane loads from 64 to 16
per reduction phase. However, each thread retains eight instead of four
values and has longer local max/sum chains. The two barriers, launch count,
padded 1024-element work, and total data traffic are unchanged.
These are competing hypotheses, not a performance prediction.

## Fixed validation and timing

The existing nine-case capture helper in
lab/sm120/pi05_attention_qk_triton_confirm.py records steps 0/4/9 crossed
with layers 0/8/17, before out=Q overwrites the input. It is present in the
base checkout and is imported by this script; execute the new script by
its file path so the existing sibling import resolves. Capture uses the
current shipped model, seed 42, and saves its full identity in the snapshot.

For each case, control and candidate share exactly the same logits,
probability, working-Q and saved input addresses. Both execute the current
production Triton 32x32x64 QK, selected native softmax, then unchanged
torch.mm(P,V) with out=Q. Probability and output metrics use the existing
shallow tolerance without adjustment. All six metrics, exact-equality flags
and changed-element counts are saved after each case. Numerical failure is
saved and raised before timing. No blanket exactness is claimed from nine cases.

After all nine pass, one A1/B1/B2/A2 comparison includes the identical Q reset
inside every complete attention invocation. Each graph has 16 sequences of
nine calls, five warm replays and 15 measured replays. All 15 raw samples,
including the first, are retained in each leg. This is a warm working set
below L2 capacity, with no flush; it does not recreate intervening full-model
traffic. Compilation, model loading and numerical checks precede timing.
The report retains both control/candidate drift and their conservative gap.
A local gain is not a deployed end-to-end gain.

## Reproduce in the granted exclusive window

Use the complete tested revision 6f8aec5, or apply the exact attached delta
in an experimental checkout before running the script:

~~~sh
git apply lab/sm120/pi05_attention_softmax_128.patch
~~~

From /home/ubuntu/flash-vla-gpt6-backbone:

~~~sh
source /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-env.sh
/home/ubuntu/flash-vla/.venv/bin/python lab/sm120/pi05_attention_softmax_128_probe.py prepare \
  --snapshot /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-softmax-128-nine.safetensors \
  --checkpoint /home/ubuntu/models/pi05_belt_cup_pytorch \
  --checkpoint-id kai0/pi05-belt-cup/orbax-39999+openpi-convert-pi05_aloha
/home/ubuntu/flash-vla/.venv/bin/python lab/sm120/pi05_attention_softmax_128_probe.py time \
  --snapshot /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-softmax-128-nine.safetensors \
  --out /home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-softmax-128-screen.json
~~~

## Measured result: stop

Snapshot and measurement revisions are both 6f8aec5. The captured identity
records the retained 021 route: triton-qk-attention, triton-qkv-finish,
dual-ffn, and the original cutlass-vision FFN up. Preparation triggered a
normal rebuild of the existing same-target CUTLASS library, completed before
numerics and timing. The fixed softmax library was compiled with the original
flags plus -Xptxas=-v for the resource log.

| Native variant | Registers/thread | Shared bytes | Stack / spill bytes |
|---|---:|---:|---:|
| control 256 | 32 | 64 | 0 / 0 |
| candidate 128 | 39 | 32 | 0 / 0 |

These resource changes do not establish the cause of the observed timings.

All nine cases pass the existing shallow probability and full-output metrics.
The changed sum association produces 18 changed probability elements in total;
only step4_layer0 has exactly matching P. Maximum P absolute error is
3.814697265625e-6 and worst relative RMS is 2.9642270338733675e-7.
Seven of nine complete attention outputs are exact. The other two have seven
changed elements in total, maximum absolute error 0.0078125, worst relative
RMS 2.7800134738669127e-5, and minimum cosine 0.9999999996135823.
These are small measured differences, not bitwise parity.

One complete-chain ABBA, including identical Q reset:

| Leg | Median us / 9 calls | IQR us / 9 calls |
|---|---:|---:|
| A1: 256 | 109.9400 | 109.8810–109.9800 |
| B1: 128 | 109.8780 | 109.8750–109.9010 |
| B2: 128 | 109.9180 | 109.8790–109.9420 |
| A2: 256 | 110.0400 | 110.0210–110.0590 |

The apparent mean gain is 0.0920 us per nine calls (0.01022 us/call),
while A drifts by 0.1000 us and B by 0.0400 us. The conservative separation
min(A)-max(B) is only 0.0220 us per nine calls; A1 overlaps candidate IQRs.
This does not establish a stable useful complete-chain gain. The direction
stops without a production route, deployed benchmark, repeated comparison,
512-thread variant or another reduction scheme. No timing cause is assigned.

Tracked compact evidence:
- results/rtx5090-pi05/gpt6-attention-softmax-128/local.json: all nine case
  metrics and all 60 measured samples, including every leg's first sample;
- results/rtx5090-pi05/gpt6-attention-softmax-128/ptxas.txt: both variants'
  compiler resource output.

The original snapshot and raw prepare/time/build logs remain under
/home/ubuntu/flash-vla/artifacts/rtx5090-pi05/gpt6-softmax-128-*.
Initial checks were py_compile and scoped diff review. The granted window
then completed compilation, actual capture, numerics and this one ABBA.
GPU/NVCC/JIT ownership was released immediately after measurement completion.
