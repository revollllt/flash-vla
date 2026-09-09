# Agent Note: the Pi0.5 Target — what openpi fixes, and what the fold buys

Status: implemented

## Problem

Pi0.5 shares Pi0's SigLIP tower and its Gemma-2B backbone, so the H100 Target
could reuse most of Pi0's kernels — but three model-level differences decide
what the Target may and may not do, and each of them was a research question
before it was an implementation:

- π0.5 is described publicly as a two-stage, hierarchical policy, which would
  have meant a text decode, a second KV-cache segment and a variable-length
  sub-task loop inside the graph.
- The action expert's norms become adaptive (AdaRMSNorm), adding a modulation
  projection at every norm site. Taken naively that is a large increase in the
  weight stream of the stage that is already weight-bandwidth bound.
- The robot state stops being a projected token in the suffix and is
  discretized into the language prompt instead, which makes the prefix longer,
  makes part of it padding, makes the action expert's RoPE offset
  data-dependent, and puts a tokenizer on the per-inference path.

The bring-up answered all three against the reference implementation. The
answers are the Target's contract, and they lived nowhere but the temporary
bring-up plan that this change deletes.

## Decision

### The reference the Target is defined against

The I/O contract is openpi's, verbatim: tokenization, discretization, padding
length, sequence layout and action semantics. Everything below was verified
against the local openpi checkout at `15a9616` and against the real PaliGemma
tokenizer model.

Bit-identity with an earlier implementation is *not* the bar; the bar is the
precision gate. A transformation that is mathematically exact but rounds
differently (a row scaling moved across a reduction, an fp32 accumulator in
place of a bf16 intermediate) is in scope and is often more accurate than what
it replaces. Changing *what* is computed — a shorter pad, a different
discretization, a fused approximation of a non-linear op — is a model change
and is out of scope.

**Pi0.5 is not hierarchical in this checkpoint.** openpi has one `Pi0` class
with a `pi05=True` flag; `sample_actions` takes the same code path for both:
one prefix prefill into the KV cache, then ten flow-matching steps. There is no
text decode, no logits head, no second KV-cache segment and no sub-task token
loop — openpi's own README states that only the flow-matching head is
supported for π0.5 training and inference. The two-stage inference of the blog
post is a serving-layer construction, not part of these weights. Every
"sub-task generation stage" work item, the two-graph-chaining risk and the
variable-sub-task-length risk are therefore void.

**Three structural differences from Pi0, and only three.** The SigLIP tower
(So400m/14, 27 layers, width 1152, FFN 4304, 16 heads) and the Gemma-2B
backbone (18 layers, width 2048, FFN 16384, 8 query heads over 1 KV head, head
dim 256) are identical. The Gemma-300M action expert keeps its shapes (18
layers, width 1024, FFN 4096, 8 query / 1 KV, head dim 256) and changes in
three ways: its norms become adaptive; the state leaves the suffix for the
prompt, so the suffix is `action_horizon` rows rather than `1 + action_horizon`
and the timestep reaches the model as an AdaRMS condition instead of a
concatenated MLP input; and `max_token_len` goes from 48 to 200.

Checkpoint-dependent shape profiles: `pi05_base` and `pi05_aloha` have horizon
50 with a discrete state input and are the profile this Target is built for;
`pi05_droid` has horizon 15; `pi05_libero` has horizon 10 and
`discrete_state_input=False`, so that model consumes no state at all and is not
a usable baseline for the state path.

Real-checkpoint reference runs require an explicit upstream training configuration.
The adapter rejects Pi0, state-free Pi0.5, unmerged LoRA variants, and incompatible
action dimensions or prompt padding before allocating the model. The selected
action horizon is the shape used by both implementations, and the resolved
configuration remains reference provenance in the report. Configuration checks
do not prove the checkpoint's origin, tensor ABI, robot preprocessing, numerical
correctness or policy quality; those still require their own observed evidence.
An available conversion config.json must agree with the selected reference
configuration on its recorded model fields. A contradiction is an error before
model allocation.

Production Pi0.5 construction accepts the original OpenPI PyTorch checkpoint
with a separate immutable ID and digest. The existing adapter normalizes its
weights; schedule-dependent folds are rebuilt for every construction. Input
seed changes do not replace real weights, and synthetic weights cannot be
relabeled through real-checkpoint provenance options. Weight-free declaration
checks configuration and supplies workload identity; it does not certify a
checkpoint tensor schema or establish correctness.

Checkpoint compatibility evidence inspects stored safetensors names and shapes
against an OpenPI model on meta, validates shared-parameter aliases against
actual reference parameter identity, and checks the existing normalized layout
against the Target schema. The explicit supported reference configuration owns
the inference semantics; its resolution and the observed normalized shapes
produce the onboarding contract. No tensor values are loaded or hashed.
This establishes structural compatibility under that configuration, not the
checkpoint's training origin, numerical correctness or robot policy quality.
Transition receipts additionally bind the observed weight and fixture identities
to the requested measurement context; that context's environment comes from
the caller's recorded measurement setup, not this CPU structural inspection.


**AdaRMSNorm, exactly.** A modulation dense layer maps the condition (1024) to
3072 with bias and zero init; the result splits into `scale`, `shift` and
`gate`; the site computes `rms(x) * (1 + scale) + shift` and the residual
becomes `x + f(x_hat) * gate`. It is applied at the pre-attention norm, the
pre-FFN norm and the expert's *final* norm, whose gate is discarded — 18 × 2 +
1 = **37 sites**. An AdaRMS site has no learned per-channel weight (the
conditioned branch replaces it), so Pi0's trick of folding `(1 + norm.weight)`
into the following GEMM has nothing to fold on the action-expert side; the
vision and backbone sides are unchanged. Only the action expert is adaptive;
the backbone keeps plain RMSNorm.

**Discrete state, exactly.** openpi bins the normalized state with
`digitize(state, linspace(-1, 1, 257)[:-1]) - 1` and writes
`f"Task: {task}, State: {' '.join(bins)};\nAction: "`. Bin values span −1 … 255
(257 distinct), because `digitize` returns 0 below −1, which becomes −1; the
string `"-1"` is a legal state token. The branch-free closed form is
`b = clamp(floor(x * 128) + 128, -1, 255)`, verified bit-exact over 400 k
samples including every bin edge in both fp32 and fp64. `floor((x + 1) * 128)`
is **wrong**: it disagrees for tiny negative `x`, because `x * 128` is an exact
power-of-two scaling and `x + 1` is not. Token count for a 32-dim state is
75 … 151, mean 131 over 3300 prompts; openpi pads to 200 regardless.

Worked example. The bin mapping over the normalized range, with −1 as the
out-of-range clip:

| state | −1.5 | −1.0 | −0.5 | −1e-5 | 0.0 | +0.5 | +0.99 | +1.0 | +2.0 |
|---|---|---|---|---|---|---|---|---|---|
| bin | **−1** | 0 | 64 | 127 | 128 | 192 | 254 | 255 | 255 |

A 4-dim toy state `[0.0, -0.5, 0.9, -1.2]` becomes bins `[128, 64, 243, -1]`
and the prompt `"Task: fold the towel, State: 128 64 243 -1;\nAction: "`, which
the PaliGemma tokenizer turns into 27 pieces: a per-episode head
(`<bos> Task : ▁fold ▁the ▁towel , ▁State :`), one group per state value
(`▁ 1 2 8`, `▁ 6 4`, `▁ 2 4 3`, `▁- 1`), and a fixed tail
(`; \n Action : ▁`).

At the real 32-dim size a typical prompt is 128 tokens, of which about 116 are
the state block, and the prefix is three 256-token image views followed by the
language block padded to 200 — 968 rows, of which `768 + n_valid` carry data.
Positions are `cumsum(input_mask) - 1`, so a valid language token *j* always
lands at position 768 + *j* regardless of the pad, padded rows repeat the last
valid position and are masked out, and the action rows take positions
`n_valid … n_valid + chunk - 1`. **That offset is the one quantity that varies
per inference**, which is why `action_expert_rope` is filled per call and the
backbone's rope table stays static.

### AdaRMS costs nothing at inference, because it is folded

Naively the 37 modulation projections are 37 × Dense(1024 → 3072) =
**116.5 M parameters**, +37.4 % on the 311.4 M-parameter action expert, on the
stage that is purely weight-bandwidth bound.

They are never streamed, because the condition
`swish(time_mlp_out(swish(time_mlp_in(posemb(t)))))` **depends only on the
timestep**, and the flow-matching schedule is fixed at `t = 1.0, 0.9, …, 0.1`.
All 37 sites × 10 steps of `(scale, shift, gate)` are therefore compile-time
constants of the Target, folded once at checkpoint load:

| table | shape | bytes (bf16) |
|---|---|---|
| `(1 + scale)` | 10 × (18×2+1) × 1024 | 0.76 MB |
| `gate` | 10 × 18×2 × 1024 | 0.74 MB |
| `shift @ W_qkv` | 10 × 18 × 2560 | 0.92 MB |
| `shift @ [W_gate·W_up]` | 10 × 18 × 8192 | 2.95 MB |
| `shift @ W_out` (final norm) | 10 × 32 | negligible |

About 5.4 MB in total, about 0.2 MB read per step, against 116.5 M parameters
that never enter the weight stream. This is the same class of transformation
Pi0 already applies to its fixed timestep schedule; Pi0.5 simply offers a
larger fold.

What remains at runtime, per action-expert norm site, lands on three different
axes of the consuming GEMM, which is why one of the three is hard: `(1 + scale)`
is per-K, *inside* the reduction, and cannot be moved out; `shift @ W` is
per-N, an epilogue add; `gate` is per-N, an epilogue multiply; the RMS factor
stays per-M and rides the epilogue as before. Expressing `(1 + scale)` as a row
scaling of the weight (`diag(s) @ W`) is taken only for the final norm, whose
`W_out` is 1024 × 32 so ten copies cost 0.65 MB; for the QKV projection ten
copies would be 944 MB and for the FFN 3.0 GB, so there the scale rides on the
activation tile.

### The longer prefix is the real structural cost

`768 + 200 = 968` backbone rows instead of Pi0's 768. Unlike AdaRMS this does
not fold away. Derived by dividing the stage's minimal traffic and math by the
datasheet H100 peaks — the arithmetic the bring-up used, recorded here as the
derivation it is; `python -m benchmarks floor` is the live authority and
divides by measured, tagged constants:

| stage | Pi0 | Pi0.5 | source of the change |
|---|---:|---:|---|
| `vision_encoder`, 27 layers | 0.66 ms | 0.66 ms | unchanged |
| `llm_backbone`, 18 layers (compute-bound, ~linear in sequence) | 3.08 ms | **3.88 ms** | 768 → 968 rows |
| `action_expert`, 10 × 18 layers (weight-bound) | 1.90 ms | ~1.90 ms | AdaRMS folded |
| sum of per-stage floors | **5.65 ms** | **~6.44 ms** | |

The action expert's key length goes 819 → **1018** and its M goes 51 → **50**,
so every tile configuration carried over from Pi0 is at the wrong geometry.

The pad length is **not** shortened. Because positions are `cumsum(mask) - 1`
and padded columns are masked out of the softmax, the output is invariant to
the pad as long as it never truncates, and about 64 % of the 200 columns are
padding, so a shorter pad would recover roughly 0.16 ms of the backbone floor.
That is a deployment shape-profile decision for whoever owns the deployment,
not an inference optimization; the Target matches openpi at 200.

### The prompt decomposes, provably

The prompt is `head(task) + Σ_d " {b_d}" + ";\nAction: "`. SentencePiece
segmentation of the number block is **independent of its neighbours** for this
tokenizer: every number is preceded by a space, which becomes its own `▁`
piece, and digits are individual pieces, so nothing is shared across value
boundaries. The piece count is therefore a pure function of the bin — 2 pieces
for `-1` and `0`–`9` (11 values), 3 for `10`–`99` (90 values), 4 for `100`–`255`
(156 values) — and the prefix length is a 32-entry table sum, computable
anywhere, host or device. The head and the tail encode identically standalone.
The tables are `NUM[257][4]` and `LEN[257]` built once ever, a fixed tail, and
a head cached once per episode.

### Tokenization stays on the host, behind the vision stage

Tokenization depends only on `state`; the vision tower depends only on
`images`; both inputs arrive together. The Target therefore captures the pass
as separate stages and declares a host slot between `vision_encoder` and
`llm_backbone`: the host tokenizes and stages the prompt inputs while the
vision graph runs. The variable-length prompt mask and the data-dependent
action-expert RoPE table get a free host-side home in the same slot, so no
device-side length propagation is needed anywhere.

Measured on this cluster's login-node CPU with the real PaliGemma tokenizer:
openpi's SentencePiece path costs 45.5 µs per call, the table lookup 15.9 µs
and is exact. Either is hidden entirely by the stage split.

The contrast with vLLM/SGLang matters and is easy to get backwards. Their
overlapped scheduler hides CPU work because there is always a *next* request's
metadata to prepare while the current step runs; that buys throughput, not the
latency of one call. Here there is exactly one call in flight, so the only
thing to overlap against is *this* call's own GPU work — which exists precisely
because the vision tower does not depend on the state. A vLLM-style scheduler
does not help a single-shot latency target; this specific dependency split
does.

### The action expert is memory-latency-bound — as a hypothesis

At chunk 50 the action expert is memory-bound (every call site's arithmetic
intensity is far below the H100 ridge point), but it is **memory-*latency*-bound
rather than bandwidth-bound**: call sites at full CTA occupancy still reach only
a fraction of streaming bandwidth, because shared-memory footprint caps
resident warps per SM and too few warps cannot keep enough loads in flight to
hide HBM latency. Wave quantization at M = 50 compounds this for the QKV
projection and the attention, but is the smaller effect.

This is recorded as a hypothesis with its conditions — chunk 50, 10 steps, 18
layers, the shipped plan, cold weights — and not as a measurement. The
measurements it was derived from came from per-kernel commands that no longer
exist, so the numbers are not reproducible. Re-derive it with
`python -m benchmarks kernels --target h100/pi05` for the per-call-site times
and `python -m benchmarks floor --target h100/pi05` for what they are being
compared against. The practical consequence, if it holds, is that the action
expert's bandwidth floor is not reachable by tiling: the levers are a smaller
per-CTA shared-memory footprint (more resident warps) or less weight traffic
per step, not a better tile shape.

## Alternatives considered

- **Stream the modulation weights.** Rejected: the condition is a pure function
  of the fixed timestep schedule, so all 37 sites × 10 steps are constants of
  the Target and belong in the fold.
- **Fold `(1 + scale)` into the weight everywhere** (`diag(s) @ W`, ten copies).
  Taken for the final norm only; for QKV and the FFN the ten copies are 944 MB
  and 3.0 GB, so the scale rides on the activation tile there.
- **Shorten the pad below 200.** Rejected under the ground rule: the output is
  invariant to it, so it is a shape-profile decision of the deployment, not an
  optimization, and taking it would break the openpi I/O contract this Target
  is defined against.
- **Tokenize on the device, inside the graph.** Rejected for now: it is
  achievable (SentencePiece never has to run on the GPU; after the piece-count
  proof the only variable input is 32 values from a 257-symbol alphabet, and
  every step is a fixed-shape kernel over data-dependent values), but it adds
  about six launches to a stage that is already latency-bound in order to save
  host time the stage split already hides for free, and it makes the token ids
  device-only, so the tokenizer check and any debugging need an explicit
  readback path. Build it only if the stage split ever costs more than it saves.
- **A host node inside the graph** (`cudaLaunchHostFunc`, which is
  stream-capturable). Rejected: it buys nothing over the stage split, it
  serializes the stream and adds a driver-thread dispatch, and torch's graph
  API does not expose it, so it would need a custom op. Recorded so it is not
  rediscovered.
- **`pi05_libero` as the baseline for the state path.** Rejected: it sets
  `discrete_state_input=False`, so that model consumes no state at all.
- **Bit-identity with an earlier implementation as the bar.** Relaxed by the
  Target owner: the bar is the precision gate against openpi, which admits
  exact-but-differently-rounded transformations.

## Consequences

- The action expert's suffix has no state token, so Pi0's per-head mask
  structure is dead here: the whole prefix is bidirectional and the padding is
  a per-key, query-independent hole, so the mask is a 1-D additive vector over
  `968 + chunk` rows. It uses a large finite negative rather than `-inf`, so an
  all-masked row softmaxes to uniform rather than NaN.
- Padded embedding rows are zeroed rather than left uninitialized: a padded
  query row's attention output feeds the next layer and a NaN there survives
  the mask, while zeros stay finite through RMSNorm. Buffers the host rewrites
  per call are zeroed at allocation for the same reason — an uninitialized
  index buffer is a crash in the embedding gather, not a wrong number.
- The weight schema splits at "does it depend on the inference schedule":
  `models/pi05/spec.py`'s `weight_shapes` is the model contract, one entry per
  openpi tensor up to lossless relayout; `runtime_shapes(steps)` is what a
  Target loads; `weights.fold` maps one to the other. `models/` never learns
  what a Target's capture looks like. `language_embeds` stops being a weight
  and becomes a resident vocabulary table, because the prompt now carries the
  state and changes every call.
- openpi's rotary frequencies are bf16 in the reference implementation: the
  bf16 cast covers registered buffers, so the inverse-frequency table is
  quantized at construction and a deep prefix position is several radians out
  of phase. The baseline adapter recomputes the frequencies; re-*casting* is
  not enough, because the forward pass already widens to fp32. Without this the
  reference check fails in a way that presents as a Target bug — error zero at
  position 0, growing linearly, norms preserved.
- Weight names keep their `decoder_*` / `encoder_*` prefixes: they are the
  model contract and the key set the openpi adapters and the fold match on.
  Stage, call-site and buffer names use the full `llm_backbone_*` /
  `action_expert_*` forms.

## Open questions

1. **Which checkpoint is the target?** `pi05_base` (horizon 50) is the
   assumption everywhere above. `pi05_droid` (15) and `pi05_libero` (10, and no
   state at all) are different shape profiles and each needs its own tuning
   pass and its own acceptance entry.
2. **Task-text policy.** The head tokens are cached per episode. If the task
   string can change between two consecutive `forward` calls without an
   explicit `set_task`, that cache is wrong. The calling convention needs
   confirming.

## Verification

The bring-up recorded no job ids for these runs; each item names the check that
covers it today.

- **The fold is exact.** `python -m eval.pi05.fold` — host only, no GPU, no
  checkpoint. Worst fp32 relative deviation 8.5e-7 across all 18 layers × 10
  steps × {qkv, ffn-gate, ffn-up} plus the output projection, i.e. rounding; in
  bf16, which is what the Target stores, 2.6e-3, the cost of rounding the
  tables once. `--openpi` additionally pins the time embedding to openpi's own
  sinusoidal embedding.
- **The tokenizer decomposition holds.** `python -m eval.pi05.tokenize` with
  `PALIGEMMA_TOKENIZER` set — host only. 7 task strings (including empty,
  underscored, newlined, and one long enough to truncate) × {200 random states,
  the range extremes, a ramp, an fp64 case, and all 257 bin values broadcast to
  every slot}: 3248 encode cases and 401 k discretization samples, 0 mismatches
  token for token and mask for mask, truncation branch exercised 464 times.
- **The backbone matches openpi.** `python -m eval.pi05.reference --stage
  llm_backbone --layers 18`, under the openpi interpreter, on random weights:
  layer-0 cosine 0.99994, deepest layer 0.99728, worst per-layer step 0.00022,
  padded rows finite. Layer 0 is the reading that matters — it carries no
  accumulated error, so a structural bug shows there at full size; the drift to
  0.9973 is 45 bf16 layers compounding on random weights, and the smooth
  per-layer step is what rules out a bug at any single layer.
- **The action expert matches openpi.** `python -m eval.pi05.reference --stage
  action_expert`, on random weights, with openpi's own prefix cache
  transplanted in by default so that the reading isolates action-expert wiring
  and kernels from prefix drift: at 1 step cosine 0.9999911 (max_abs 0.0176,
  rms 0.00488), at 10 steps 0.9999841, and the full pass 0.9999840 — the prefix
  drift washes out because the expert attends over 1018 keys and averages.
  `--steps 1` is the gated reading; the flow loop is a chaotic map on random
  weights and depth is an amplifier, not a defect.
- **The AdaRMS kernels match torch.** `lab/pi05/kernels.py` (`--only A|B|C|D`
  runs one variant): the QKV/RoPE variant at cosine 0.9999959, the scaled-gate
  variant at 1.0000000, the gated residual at 0.9999999 (K=2048) and 1.0000000
  (K=4096), the masked FlashDecoding split at 0.9999957. Against a torch
  recomputation rather than openpi, which has no AdaRMSNorm kernel to compare
  with — this is what localizes a failure to a kernel rather than to the
  weights.
- **The graph itself.** `python -m eval.smoke` checks the declarations without
  a device; `python -m eval.correctness --target h100/pi05` compares the
  shipped plan against the reference route in lockstep on the declared stage
  outputs.
- **Not covered:** none of this uses trained weights. Passing a real
  `pi05_base` checkpoint adds the conversion of trained values and says nothing
  more about the code; policy quality needs the LIBERO slot, which is empty.

## Related notes

- [explicit graph and ModelRunner](2026-09-06-explicit-graph-and-model-runner.md):
  the form the Target takes — contract, graph, plans — and the naming this note
  uses.
- [runtime/Target boundary and acceptance first](2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  the acceptance registry and the floor's standing as guidance.
- [deployment configuration and the lab workspace](../process/2026-09-06-deploy-config-and-lab.md):
  where the trial scripts cited above live.
