# Fixed 128-thread native softmax screen

This experiment changes only the native CTA width from 256 to 128. Both
instantiations live in the same fused_attention library; the existing launch
and all production routes remain on 256 threads. Its base includes 80818f9,
which withdraws the inconclusive vision rounded-GELU deployment candidate.

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

No measurements have been made at source preparation time. The only initial
checks are Python syntax compilation and a scoped diff review.
