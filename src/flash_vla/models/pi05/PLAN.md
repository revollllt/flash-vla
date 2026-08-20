# Pi0.5 × H100 — model support plan

> **Temporary.** This file exists only while the Pi0.5 target is being brought up.
> Delete it once `hardware/nvidia/h100/pi05/` is landed, tuned, and documented in
> its own README. Nothing in the package may import or depend on it.

Goal: add the Pi0.5 target to flash-vla, reusing the H100 target infrastructure.
Pi0.5 lives at `src/flash_vla/hardware/nvidia/h100/pi05/` and must not import
anything from `hardware/nvidia/h100/pi0/` (target isolation, ARCHITECTURE.md).
`runtime/` and `tuning/` are shared.

### Ground rule

**This is an inference-optimization target, not a model change.** The I/O
contract — tokenization, discretization, padding length, sequence layout,
action semantics — is openpi's, verbatim. A transformation is allowed only when
it is *provably equivalent* (algebraic folds, exact closed forms, mathematically
identical fusions), and the proof or the parity measurement belongs next to it.
Anything that would move a number, however defensibly, is out of scope here.

Concretely: `max_token_len = 200`, matching openpi (`pi0_config.py:38`). The
output happens to be invariant to a shorter pad (§1.2), and that is worth
knowing, but we do not take it — it is a shape-profile decision for whoever owns
the deployment, not an optimization.

---

## 0. Fact base

Verified against the local openpi checkout at `15a9616`
(`/data/user/jzou521/codes/cuda/openpi-official`), and against the real
PaliGemma tokenizer at
`/data/user/jzou521/models/openpi/big_vision/paligemma_tokenizer.model`.

### 0.1 Pi0.5 is *not* hierarchical in openpi

`Pi0` is one class; `pi05=True` is a flag (`models/pi0.py:66`). `sample_actions`
(`models/pi0.py:216-279`) takes the **same** code path for both: one prefix
prefill into the KV cache, then 10 flow-matching steps. There is no text decode,
no logits head, no second KV-cache segment, no sub-task token loop.

openpi's README states it directly: *"in this repository, we currently only
support the flow matching head for both π0.5 training and inference."* The
two-stage / high-level-subtask inference described in the π0.5 blog post is a
serving-layer construction, not part of these weights or this code.

**Consequence:** every "Stage A sub-task generation" work item from the previous
draft of this plan is deleted. So is the two-graph-chaining risk and the
variable-sub-task-length risk.

### 0.2 The three real differences

| aspect | pi0 | pi0.5 | source |
|---|---|---|---|
| SigLIP vision | So400m/14, 27L, w=1152, ffn=4304, 16 heads | **identical** | `models/pi0.py:81-89` |
| prefix Gemma-2B | 18L, w=2048, ffn=16384, 8 q-heads / 1 kv-head, hd=256 | **identical** | `models/gemma.py:79-87` |
| action expert Gemma-300M | 18L, w=1024, ffn=4096, 8 q / 1 kv, hd=256 | **same shapes**, norms become AdaRMS | `models/gemma.py:69-78` |
| state input | `state_proj(state)` → suffix token 0 | **discretized into language tokens in the prefix** | `models/pi0.py:151-157`, `models/tokenizer.py:23-30` |
| timestep | `concat(action, time)` → 2-layer MLP → action tokens | `time_mlp_in/out` (swish ×2) → **AdaRMS cond** | `models/pi0.py:162-177` |
| suffix length | `1 + action_horizon` | **`action_horizon`** (no state token) | `models/pi0.py:151` |
| `max_token_len` | 48 | **200** | `models/pi0_config.py:38-39` |

Checkpoint-dependent shape profiles (`training/config.py`):

| config | `action_horizon` | `discrete_state_input` | note |
|---|---|---|---|
| `pi05_base`, `pi05_aloha` | 50 | True | the default target |
| `pi05_droid` | 15 | True | decoder M=15 |
| `pi05_libero` | 10 | **False** | `pi05=True` + `discrete_state_input=False` ⇒ **the model consumes no state at all** (`config.py:745`). Do not use it as the parity baseline for the state path. |

### 0.3 AdaRMSNorm, exactly

`models/gemma.py:112-131`, and the PyTorch mirror at
`models_pytorch/transformers_replace/models/gemma/modeling_gemma.py:49-104`:

```
modulation           = Dense(cond)              # 1024 -> 3072, with bias, zero-init
scale, shift, gate   = split(modulation, 3)
x_hat                = rms(x) * (1 + scale) + shift
y                    = x + f(x_hat) * gate      # gated residual
```

Notes that matter for the checkpoint adapter and the kernels:

- Applied at `pre_attention_norm_1`, `pre_ffw_norm_1`, **and the expert's final
  norm** (`models/gemma.py:409-411` passes `adarms_cond[i]` to `final_norms`;
  its gate is discarded). That is 18×2 + 1 = **37 sites**.
- An AdaRMS site has **no learned per-channel `weight`** — the `if cond_dim is
  not None` branch replaces it (`modeling_gemma.py:57-64`). So the pi0 adapter
  trick of folding `(1 + norm.weight)` into the following GEMM
  (`eval/baselines/openpi.py:_fold_norm`) has nothing to fold on the decoder
  side. Encoder side is unchanged.
- Only the action expert uses AdaRMS; the prefix expert keeps plain RMSNorm
  (`models/pi0.py:80`, `use_adarms=[False, True]`).

### 0.4 Discrete state, exactly

`models/tokenizer.py:23-30`:

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
full_prompt = f"Task: {cleaned_text}, State: {' '.join(map(str, ds))};\nAction: "
tokens = sp.encode(full_prompt, add_bos=True)
```

Measured facts (see §3 for the experiments):

- Bin values span **-1 … 255** (257 distinct), because `digitize` returns 0 for
  `x < -1`, which becomes -1. The string `"-1"` is a legal state token.
- Exact branch-free closed form, verified bit-exact over 400 k samples including
  every bin edge, in both fp32 and fp64:
  `b = clamp(floor(x * 128) + 128, -1, 255)`.
  (`floor((x+1)*128)` is **wrong** — it disagrees for tiny negative `x`, where
  `x+1` rounds to 1.0. `x*128` is an exact power-of-two scaling, `x+1` is not.)
- Token count for a 32-dim state: **75 … 151, mean 131** over 3300 prompts.
  openpi pads to 200 regardless.

#### Worked example

Bin mapping (normalized state ∈ [-1, 1] → 256 uniform bins; `-1` is the
out-of-range clip):

| state | -1.5 | -1.0 | -0.5 | -1e-5 | 0.0 | +0.5 | +0.99 | +1.0 | +2.0 |
|---|---|---|---|---|---|---|---|---|---|
| bin | **-1** | 0 | 64 | 127 | 128 | 192 | 254 | 255 | 255 |

A 4-dim toy state `[0.0, -0.5, 0.9, -1.2]` → bins `[128, 64, 243, -1]` →

```
"Task: fold the towel, State: 128 64 243 -1;\nAction: "
```

which the PaliGemma tokenizer turns into 27 tokens:

```
<bos> Task : ▁fold ▁the ▁towel , ▁State :        <- head, fixed per episode
▁ 1 2 8 | ▁ 6 4 | ▁ 2 4 3 | ▁- 1                 <- the 4 state values
; \n Action : ▁                                   <- tail, fixed forever
```

Two structural facts fall out and drive everything in §3:

1. Each value is `▁` (or `▁-`) followed by its decimal digits, one token per
   digit. Nothing is shared across value boundaries.
2. The piece count is a pure function of the bin: 2 for `-1` and `0`–`9`,
   3 for `10`–`99`, 4 for `100`–`255`.

At the real 32-dim size, a typical prompt is 128 tokens of which 116 are the
state block, and it lands in the prefix like this (3 views, `L_pad` = 200):

```
[  0 .. 255]  image view 0        SigLIP, 256 tokens
[256 .. 511]  image view 1
[512 .. 767]  image view 2
[768 .. 895]  language, 128 valid tokens   <- the state lives here
[896 .. 967]  padding, 72 tokens, mask=False
              encoder_seq_len = 768 + 200 = 968
```

`positions = cumsum(input_mask) - 1`, so valid language token *j* always gets
position 768+*j* regardless of the pad, padded rows repeat the last valid
position and are masked out, and the 50 action tokens get positions
`n_valid … n_valid+49` = 896…945 here — **the one quantity that varies per
inference** (§2.2, decoder RoPE).

Contrast with pi0 on the same observation: `tokenized_prompt` is empty
(`encoder_seq_len = 768`), and the state enters as `state_proj(32→1024)`, a
single continuous token at the head of the suffix (`suffix = 1 + 50 = 51`).
Pi0.5 deletes that token and re-encodes the same information as ~116 language
tokens in the prefix (`suffix = 50`).

---

## 1. What this actually costs

### 1.1 AdaRMS is free if it is pre-folded

Naively, 37 × Dense(1024→3072) = **116.5 M params**, i.e. **+37.4 %** on the
311.4 M-param action expert. The decoder is purely weight-bandwidth bound, so
that would move its 10-step floor from 1.86 ms to **2.55 ms**.

It does not have to cost anything, because

```
adarms_cond = swish(time_mlp_out(swish(time_mlp_in(posemb_sincos(t)))))
```

**depends only on the timestep**, and the flow-matching schedule is fixed at
`t = 1.0, 0.9, …, 0.1` (`models/pi0.py:228,278`; the pi0 adapter already
reconstructs exactly this loop at `eval/baselines/openpi.py:277-284`). So all 37
sites × 10 steps of `(scale, shift, gate)` are compile-time constants of the
target.

Fold at checkpoint load:

| table | shape | bytes (bf16) |
|---|---|---|
| `(1 + scale)` | 10 × (18×2+1) × 1024 | 0.76 MB |
| `gate` | 10 × 18×2 × 1024 | 0.74 MB |
| `shift @ W_qkv` | 10 × 18 × 2560 | 0.92 MB |
| `shift @ [W_gate·W_up]` | 10 × 18 × 8192 | 2.95 MB |
| `shift @ W_out` (final norm) | 10 × 32 | negligible |

~5.4 MB total; ~0.2 MB streamed per step. The 116.5 M-param modulation weights
never touch the inference weight stream.

This is the same class of transformation the pi0 target already does with
`decoder_action_fused_time_biases`. The previous draft's claim that "the fold is
architecturally wrong now" is **backwards** — Pi0.5 offers a *larger* fold than
Pi0.

What is left at runtime, per decoder norm site:

1. `(1 + scale)` — a **per-K-column** scale on the GEMM's A operand. This is the
   one thing that does not commute with the existing lazy-RMS trick (which
   relies on the RMS factor being *per-row*, so it can ride the epilogue). A is
   only 50×1024 here, so an extra global→register→shared trip on A is free
   relative to the weight stream; whether that or a producer-side fusion is
   right is a **tile-dataflow question, not a coding question** — see §4.
2. `shift @ W` — a per-N constant bias. Identical in form to the `Bias_local[j]`
   already in `kernels/fused_norm.py:tl_fused_rms_matmul_bias_res`.
3. `gate` — a per-N vector multiply in the residual epilogue: `x + y * gate`.

Expected net effect on the decoder floor: **~0**.

### 1.2 The prefix gets longer — this is the real cost

`encoder_seq_len = 768 + 200 = 968` instead of 768. This is the dominant
structural cost of Pi0.5, and unlike AdaRMS it does not fold away.

| stage | pi0 floor | pi0.5 floor | source of the change |
|---|---|---|---|
| vision 27L | 0.66 ms | 0.66 ms | unchanged |
| encoder 18L (compute-bound, ~linear in seq) | 3.08 ms | **3.88 ms** | 768 → 968 tokens |
| decoder 10×18L (weight-bound) | 1.90 ms | ~1.90 ms | AdaRMS folded, §1.1 |
| **sum of per-stage maxima** | **5.65 ms** | **~6.44 ms** | |

Decoder key length goes 819 → 968 + 50 = **1018**; decoder M goes 51 → **50**.

Recorded but **not taken**, per the ground rule: because
`positions = cumsum(input_mask) - 1` and padded columns are masked out of the
softmax, the model output is invariant to the pad length as long as it never
truncates. Measured token length is 75–151 for a 32-dim state plus a short task
string, so ~64 % of the 200 columns are padding and a shorter pad would recover
~0.16 ms of encoder floor. That is a deployment shape-profile decision, not an
inference optimization — we match openpi at 200.

---

## 2. Architecture delta

```
src/flash_vla/models/pi05/            # NEW (this directory)
  PLAN.md                             # this file; delete when the target lands
  spec.py                             # shapes + folded-table schema
  weights.py                          # random_checkpoint
  tokenize.py                         # host-side table tokenizer (see §3)

src/flash_vla/hardware/nvidia/h100/pi05/   # NEW target, mirrors pi0/
  engine.py  pipeline.py  buffers.py  ops.py
  backends/tilelang/{wrappers,fused_wrappers,autotune}.py
  backends/tilelang/kernels/          # AdaRMS variants + prefix embed/mask/rope
```

### 2.1 Two weight layouts, and where the line is

**Landed** as `models/pi05/spec.py` + `models/pi05/weights.py`.

`spec.weight_shapes()` is the **model contract**: one entry per OpenPI tensor,
up to relayout that is lossless and schedule-independent — transposes, the q/k/v
concat, the RoPE permutation, and absorbing a *plain* per-channel RMSNorm scale
into the single GEMM that consumes it (unchanged from Pi0, and it only applies
to vision and the prefix, since the action expert's norms are adaptive). Read it
side by side with OpenPI's state dict.

`spec.runtime_shapes(steps)` is what an execution target loads.
`weights.fold(checkpoint, steps)` maps one to the other. The line between them
is exactly *"does it depend on the inference schedule"* — anything that does
stays out of the model contract, so `models/` never learns what a target's
capture looks like, and the target never re-derives the model.

```
openpi state_dict --[eval/baselines: rename, stack, pack]--> weight_shapes()
weight_shapes()   --[models/pi05/weights.fold(ckpt, steps)]-> runtime_shapes(steps)
runtime_shapes()  --[engine]--------------------------------> static device buffers
```

Note this is a slight correction of Pi0's own arrangement: Pi0 folds its fixed
timestep schedule inside `eval/baselines/openpi.py:target_checkpoint`, so
`models/pi0/spec.py` describes the already-folded layout
(`decoder_action_fused_time_biases`). Pi0.5 separates the two because the fold
is much larger here. Pi0 is not being changed.

Delta vs `models/pi0/spec.py`, in the model contract:

- Removed: `decoder_state_in_proj_{w,b}` (no state token);
  `decoder_action_fused_in_proj_w`, `decoder_action_fused_time_biases`,
  `decoder_action_mlp_{w,b}` (no `action_time_mlp`).
- Added: `decoder_action_in_proj_{w,b}`, `decoder_time_mlp_{in,out}_{w,b}`,
  `decoder_ada_rms_{attn,ffn}_{w,b}` per layer and `decoder_ada_rms_final_{w,b}`
  — **37 sites, not 36**; the expert's final norm is adaptive too
  (`models/gemma.py:409-411`).
- `language_embeds` stops being a weight and becomes `vocab_embeddings
  (257152, 2048)` — 1.05 GB bf16, resident, ~0.6 MB read per pass — because the
  prompt now contains the state and changes every call, whereas Pi0's is fixed
  at load (`engine.py:60`).

What the fold produces, measured from the real shapes:

| | keys | decoder params | per step | 10 steps | floor at 3.35 TB/s |
|---|---:|---:|---:|---:|---:|
| `weight_shapes()` | 19 | 430.1 M | 860.2 MB | 8.60 GB | 2.57 ms |
| `runtime_shapes(10)` | 16 | 314.5 M | 628.9 MB | 6.29 GB | **1.88 ms** |

The new per-step tables total 5.35 MB bf16, 0.53 MB read per step. The final
norm folds furthest: its scale, its shift and the Euler `dt` all collapse into
`decoder_action_out_proj_w (10, 1024, 32)` and `_b (10, 32)`, which keeps Pi0's
`out += bias + rms(x) @ weight` call shape unchanged.

**Gate (passing):** `python -m eval.correctness.pi05.fold_equivalence` — host
only, no GPU, no checkpoint. Worst float32 relative deviation 8.5e-7 across all
18 layers × 10 steps × {qkv, ffn-gate, ffn-up} plus the output projection, i.e.
rounding. In bfloat16, which is what the target stores, 2.6e-3 — the cost of
rounding the tables once, the same trade `fused_vs_unfused.py` documents for
Pi0. `--openpi` additionally pins the time embedding to
`create_sinusoidal_pos_embedding`.

### 2.2 Engine / buffer plan delta

- `forward(images, state, noise, prompt_token_ids, n_valid)` — or, with the
  in-graph tokenizer of §3.3, `forward(images, state, noise)` plus a
  once-per-episode `set_task(prompt)`.
- New static buffers: `prompt_token_ids (200,) int32`, `prefix_mask_bias
  (968 + chunk,) bf16`, `prefix_len (1,) int32`.
- **`decoder_rope_weights` becomes data-dependent.** Suffix positions are
  `n_valid_prefix + 0…chunk-1`, and `n_valid_prefix` varies per call.
  `buffers.py:64` currently bakes `encoder_seq_len` in as a constant. Either
  recompute the 50×256 table on the host and H2D it (25 KB), or compute it in
  the graph from `prefix_len` (§3.3). The **encoder** rope table stays static:
  padding sits at the end of the language block, so valid token *j* always lands
  at position 768+*j*.
- **Padded embedding rows must be zeroed**, not left uninitialised. A padded
  query row's attention output feeds the next layer, and a NaN there survives
  the mask (`0 * NaN = NaN`). Zeros stay finite all the way through (RMSNorm of
  a zero row gives zero).

### 2.3 Attention mask delta

pi0's `kernels/base.py:tl_softmax_mask0` carries the condition
`(gi >= NUM_HEADS) | (j <= ENC_LEN)`. That exists **solely** to stop the state
token from attending to the action block. Pi0.5 has no state token — its suffix
`ar_mask` is `[True] + [False]*(H-1)`, so **all `H` query rows share one mask
row**.

What replaces it is prefix padding, which is also a per-key, query-independent
vector. Net: the decoder mask degrades from a 2-D block structure to a 1-D
additive `0 / -inf` vector of length `968 + chunk` (1018 at chunk 50) — strictly
friendlier for the FlashDecoding kernel. The encoder attention needs the same
vector over 968.

Use a large finite negative, not `-inf`: openpi uses `-2.3819763e38`
(`models/gemma.py:225`) and pi0's kernel uses `-3.0e38`, so an all-masked row
softmaxes to uniform rather than NaN. Keep that.

---

### 2.4 Kernel inventory, and what v1 does not attempt

Surveyed against the 19 call sites in `hardware/nvidia/h100/pi0/ops.py`.

| stage | call sites | Pi0.5 |
|---|---|---|
| vision | all 5 | **untouched** — same weights, same M=768; not even a re-tune |
| encoder | all 5 | kernels untouched, M 768 → 968; **re-tune only** |
| decoder | `decoder_state_proj`, `decoder_action_mlp` | **deleted** (no state token, no `action_time_mlp`) |
| | `decoder_action_in_proj` | existing `tl_matmul_bias` at M=50, K=32, N=1024; Pi0.5's `action_in_proj` has no activation and no time term |
| | `decoder_action_out_proj` | **unchanged kernel** — `tl_fused_rms_matmul_bias_res` already computes `R + Bias + rms(A) @ B`, which is exactly the folded form; it just indexes `W'[s]`/`b'[s]` per step |
| | `decoder_norm_qkv_rope`, `decoder_norm_gated_ffn`, `decoder_out_proj_residual`, `decoder_ffn_down_residual`, `decoder_attention` | need new kernel variants |

Three things that look like kernels and are not: the prefix padding mask (2 KB
H2D), the data-dependent decoder RoPE table (25 KB H2D), and the embed gather
(`torch.index_select(out=)` plus an in-place scale, which captures fine). With
the two-graph split of §3.2 the host computes all three behind the vision
tower, so **§4.2 needs no kernel at all**.

#### Which axis each folded term lands on

This is the whole reason one of the three is hard. `scale`, `shift` and `gate`
are per-*hidden-channel* and broadcast across all 50 action tokens
(`models/gemma.py:129`, where the `[:, None, :]` inserts the token axis). In
the consuming GEMM that puts them on three different axes:

| term | axis | vs the K reduction | cost |
|---|---|---|---|
| `rstd` | M (token, A's rows) | outside | epilogue — this is the existing lazy-RMS trick |
| `(1 + scale)` | **K (hidden, A's columns)** | **inside** | cannot be moved out |
| `shift @ W` | N (output) | after | epilogue add, free |
| `gate` | N (output) | after | epilogue multiply, free |

`sum_k A[i,k] * s[k] * W[k,n]` — `s[k]` sits inside the sum. Equivalently it is
a row scaling of W (`diag(s) @ W`), which is what the final norm actually does
(`W_out` is 1024x32, so ten copies cost 0.65 MB); for qkv that would be 944 MB
and for the FFN 3.0 GB, so there it has to ride on A.

#### The one collision, and the v1 decision

Only `tl_fused_rms_gate` collides. Its mainloop accumulates
`Pow[i,j] += A_shared[i,j]**2` from the *same* tile `T.gemm` consumes, and the
RMS needs x unscaled while the GEMM needs x scaled — with the gemms ahead of
the accumulation, in-place scaling is wrong on either side of it.
`tl_qkv_gemm_rope` does not collide (F arrives as a parameter, no accumulation
in its mainloop) and `tl_fused_rms_matmul_bias_res` does not need a scale at
all.

**Decision: v1 falls back to the separate `tl_rms_factor` for that site.** Pi0
already has that path unfused — `_rms_factor` then `tl_scaled_gate`, which
takes F as a parameter — so the Pi0.5 variant is built on `tl_scaled_gate`, not
on `tl_fused_rms_gate`.

That buys a uniform design rather than two. Both remaining GEMM variants become
the same change applied twice: *take F as a parameter, scale the A tile by a
1024-entry K-vector, add an N-bias in the epilogue.*

v1 kernel list:

1. `tl_ada_qkv_gemm_rope` — from `tl_qkv_gemm_rope`. The bias goes in **between
   the F multiply and the rotation**: the folded form is
   `q = rstd * ((x*s) @ W_q) + shift @ W_q`, then `RoPE(q)`. Adding it after the
   rotation is a silent error. Bias applies to all three of Q/K/V; RoPE only to
   Q/K.
2. `tl_ada_scaled_gate` — from `tl_scaled_gate`. Two N-biases, both inside,
   before the gelu.
3. `tl_matmul_gated_res` — from `tl_matmul_res`, one per-N multiply in the
   epilogue. `decoder_out_proj_residual` and `decoder_ffn_down_residual` share it.
4. `tl_fd_flat_split` mask variant — Pi0's `(gi >= NUM_HEADS) | (j <= ENC_LEN)`
   predicate exists only to stop the state token attending to the action block
   and is dead in Pi0.5. It is replaced by an additive mask vector, because the
   padding is a hole in the middle of the key range (`[768+n_tok, 968)`, with
   the suffix at `[968, 1018)`), not a suffix of it.

The scale vector is 1024 bf16 = 2 KB, loaded once at kernel entry rather than
per mainloop iteration (`_DEC_QKV` runs 8 iterations at BLOCK_K=128,
`tl_scaled_gate` 4 at BLOCK_K=256).

**Cost of the fallback, to be measured not assumed.** The decoder goes from 7
graph nodes per layer to 8, i.e. 1260 → 1440 nodes over 18 layers x 10 steps.
Every decoder kernel is under one wave and latency-bound, so the marginal cost
is the added kernel's own latency and not just node overhead; `tl_rms_factor`
at M=50, K=1024 touches 100 KB. Record the per-stage split before and after so
v2 has a number to beat.

**What v2 revisits.** Whether `tl_fused_rms_gate` can feed an unscaled `Pow` and
a scaled `gemm` from one tile — a second BLOCK_M x BLOCK_K shared buffer for the
scaled copy (32 KB single-buffered at `_FUSED_GATE`, consumed in the same
iteration), or reordering to Pow → scale → gemm and paying the serialization.
Shared-memory budget and occupancy decide it, with numbers, in the dataflow spec.

## 3. Tokenization: measurements and the three options

The `state` is inside the prompt, so tokenization runs **per inference**, on the
host, ahead of graph replay. Measured on this cluster's login-node CPU with the
real PaliGemma tokenizer:

| approach | µs / call | exactness |
|---|---:|---|
| openpi `PaligemmaTokenizer.tokenize` (sentencepiece) | **45.5** | reference |
| table lookup, vectorized numpy (`Pi05Tokenizer.encode`) | **15.9** | exact, 0/3248 gate cases |
| same in C/C++ | ~1–2 (est.) | exact |
| in-graph on device (§3.3) | **0** (host) | exact |

45 us is 0.30 % of the 15.18 ms pi0 baseline. Measure before optimizing; the
options below are ordered by cost.

### 3.1 The decomposition is verified, not assumed

The prompt is `head(task) + Σ_d " {b_d}" + ";\nAction: "`. SentencePiece
segmentation of the number block is **independent of neighbours** for this
tokenizer, because every number is preceded by a space, which becomes its own
`▁` piece, and digits are individual pieces:

```
'▁', '1', '3', '6',   '▁', '1', '1', '9',   …   '▁', '1', '1', '4', ';', '\n', 'Action', ':', '▁'
```

Verified by `eval.correctness.pi05.tokenize_parity`, which is the standing gate:
7 task strings (including empty, underscored, newlined, and one long enough to
truncate) × {200 random states, the range extremes, a ramp, a float64 case, and
all 257 bin values broadcast to every slot} — **0 mismatches** against
`sp.encode` of the full string, token for token and mask for mask.

Piece counts are a pure function of the bin value: 2 pieces for `-1` and `0`–`9`
(11 values), 3 for `10`–`99` (90 values), 4 for `100`–`255` (156 values). So the
prefix length is a 32-entry table sum — computable anywhere, host or device.

Tables needed: `NUM[257][4] int32` + `LEN[257] int32` (built once, ever),
`TAIL` (once, ever), `HEAD(task)` (once **per episode**, ~20 µs, off the
per-inference path). The head/tail encode identically standalone — also verified.

### 3.2 Option A — keep it on the host, hide it behind the vision tower

The cheapest structural fix, and the one to do first. Tokenization depends only
on `state`; the vision tower (27 layers, ~2.4 ms) depends only on `images`. Both
inputs arrive together.

Split the capture into two graphs:

```
replay(graph_vision)        # async, ~2.4 ms of GPU work, depends on images only
tokenize(state) on CPU      # 14–41 µs, overlaps
copy_(prompt_token_ids); copy_(prefix_mask_bias); copy_(decoder_rope_weights)
replay(graph_rest)          # encoder + decoder
```

Cost: one extra graph launch (~5 µs) and the loss of kernel overlap across the
vision/encoder boundary. Removes 100 % of the host tokenization latency, and it
also gives the variable-length `decoder_rope_weights` and mask vectors a free
host-side home — no device-side length propagation needed anywhere.

This is the same idea as vLLM/SGLang's overlapped scheduler, but note the
difference in why it works. Their async scheduler hides CPU work because there
is always a *next* request's metadata to prepare while the current step runs;
that helps throughput, not the latency of one call. Here there is exactly one
call in flight, so the only thing to overlap against is *this call's own* GPU
work — which exists precisely because the vision tower does not depend on the
state. Do not expect a vLLM-style scheduler to help a single-shot latency
target; expect this specific dependency split to.

### 3.3 Option B — do it in the graph, on the device

Fully achievable, because SentencePiece never has to run on the GPU. After §3.1
the only per-inference-variable input is 32 values from a 257-symbol alphabet.

In-graph node chain, all trivial kernels:

1. `b = clamp(floor(state * 128) + 128, -1, 255)` — 32 elements, exact (§0.4).
2. `len_d = LEN[b_d + 1]`, exclusive scan over 32 elements → per-dim write offset,
   and `n_state_tokens`.
3. scatter `NUM[b_d + 1][:len_d]` into `prompt_token_ids` after the static
   `HEAD`, then append `TAIL`; zero-fill to 200. Write `prefix_len`.
4. embed gather: `encoder_x[768 + i] = embed_table[prompt_token_ids[i]] * sqrt(2048)`,
   zero for `i >= n_tok`. ~0.6 MB read.
5. `prefix_mask_bias[j] = 0 if j < 768 + n_tok else -3e38`, and 0 again over the
   suffix range.
6. `decoder_rope_weights[i] = rope(prefix_len + i)` — 50×256, computed from the
   device-side `prefix_len`.

Every one of these is a fixed-shape kernel over data-dependent *values*, which
is exactly what CUDA graphs allow. Nothing here needs dynamic shapes.

Trade-offs, honestly: it puts six more launches into the graph (~15 µs of
launch latency in a stage that is already latency-bound) to save 16-45 us of
host time that Option A already hides for free, and the token ids become
device-only, which makes the parity gate and any debugging need an explicit
readback path. **Do Option A first. Only build Option B if the two-graph split
turns out to cost more than it saves.**

### 3.4 Option C — host node inside the graph

`cudaLaunchHostFunc` is stream-capturable and becomes a graph host node, so the
tokenizer could literally live inside the graph, ordered after the vision
subgraph. `torch.cuda.CUDAGraph` does not expose this, so it needs a custom op;
the host node also serialises the stream and adds a driver-thread dispatch
(~5–20 µs). It buys nothing over Option A. Recorded here only so it is not
re-discovered as a new idea.

---

## 4. Work items

### 4.1 Package skeleton
- [x] `models/pi05/tokenize.py` — the §3.1 tables, `Pi05Tokenizer.encode` and the
      verbatim upstream `reference` beside it. `pi05` extra added to pyproject.
- [x] `models/pi05/spec.py` (`weight_shapes` / `runtime_shapes`) and
      `weights.py` (`fold`, `random_checkpoint`) per §2.1.
- [ ] `hardware/nvidia/h100/pi05/` skeleton, `op_table` wired like pi0.
- [ ] README layout table.
- **Gate (tokenizer, passing):**
  ```
  PYTHONPATH=src PALIGEMMA_TOKENIZER=<paligemma_tokenizer.model> \
      python -m eval.correctness.pi05.tokenize_parity
  ```
  Host-only, no GPU and no openpi checkout. Currently 3248 encode cases and
  401 k `discretize` samples, 0 mismatches, truncation branch exercised 464
  times; host latency 15.9 us table vs 45.5 us SentencePiece.
- **Gate:** engine constructs and captures on random weights.

### 4.2 Prefix path — **landed**

`hardware/nvidia/h100/pi05/` now covers vision, the prompt embedding gather, and
the 18 encoder layers that build the KV cache. The backend is Pi0's, copied per
the target-isolation rule; only `attention.py` changed, to take a mask.

- [x] Host table tokenizer + `set_task()` / per-episode head cache.
- [x] `vocab_embeddings` residency + embed gather. Two torch nodes rather than a
      kernel: `prompt_embed_scale` carries `sqrt(width)` on valid rows and zero
      on padding, so one multiply both applies the embedder scale and zeroes the
      padding.
- [x] Prefix padding mask vector; encoder attention consumes it. Because the
      whole prefix is bidirectional the mask is per-key and query-independent —
      one additive vector, not a 2-D structure.
- [x] Data-dependent `decoder_rope_weights`; static encoder rope table.
- [x] Two-graph split. `forward` copies images, replays the vision graph,
      tokenizes on the host while it runs, copies 27 KB of prompt inputs, replays
      the prefix graph.
- **Gate (passing):**
  ```
  CMD='-m eval.correctness.pi05.prefix_parity --layers 18' \
  PYTHON=<openpi venv> PALIGEMMA_TOKENIZER=<tokenizer> \
      sbatch --export=ALL sbatch/run.sbatch
  ```
  Layer 0 cosine 0.99994, deepest 0.99728, worst per-layer step 0.00022, padded
  rows finite. Read layer 0: it carries no accumulated error, so a structural
  bug shows there at full size. The drift to 0.9973 is 45 bfloat16 layers on
  random weights compounding, which `fused_vs_unfused.py` already documents as
  the amplifier; the smooth step is what rules out a bug at any single layer.

Three things this shook out, all worth keeping:

1. **OpenPI's rotary frequencies are bfloat16.**
   `to_bfloat16_for_selected_params` casts the whole module, and `inv_freq` is a
   registered buffer, so `10000**(-1/128)` becomes 0.9296875 instead of
   0.9305720. Phase is frequency times position, so at prefix position 900 the
   reference is several radians out and its K is simply a different rotation.
   This cost most of the debugging: it presents as *our* bug, with error zero at
   position 0 and growing linearly, norms preserved, and un-rotating with the
   correct phase not helping. `openpi05.restore_rope_precision` recomputes the
   frequencies; note re-*casting* is not enough, since the forward pass already
   widens to float32 and the quantization happened at construction.
   Pi0 is exposed to the same thing at position ≤ 818 and reads it through ten
   denoising steps rather than directly; not investigated here.
2. **Uninitialized index buffers are a crash, not a wrong number.** `prompt_token_ids`
   is `torch.empty` and warmup runs before the first `forward`, so `index_select`
   trapped on an out-of-range id and poisoned the context. Buffers the host
   rewrites per call are now zeroed at allocation.
3. **`M = 968` is not a multiple of any tuned `BLOCK_M`.** Checked explicitly:
   TileLang bounds-checks the scalar stores in `tl_rope_scatter_bf16`, so nothing
   is written past `encoder_seq_len`. Worth re-checking for any new kernel.

### 4.3 AdaRMS — kernels **landed**, integration next
- [x] Adapter-side fold: `weights.fold` builds the per-step tables (§2.1).
      Gated by `eval.correctness.pi05.fold_equivalence`.
- [x] Tile-dataflow spec written, reviewed and approved:
      `specs/tile/pi05-adarms-decoder.md`. One spec, four instantiations, because
      they share one decision.
- [x] Four kernels in `backends/tilelang/kernels/adarms.py`, plus their call-site
      wrappers. The op table is now 18 entries.
- [x] **Gate (passing):** `python -m eval.correctness.pi05.kernel_parity`
      (`--only A|B|C|D` runs one variant).

      | variant | cosine vs torch | max_abs |
      |---|---|---|
      | A `tl_ada_qkv_gemm_rope` | 0.9999959 | 3.1e-2 |
      | B `tl_ada_scaled_gate` | 1.0000000 | 3.1e-2 |
      | C `tl_matmul_gated_res` K=2048 | 0.9999999 | 4.9e-4 |
      | C `tl_matmul_gated_res` K=4096 | 1.0000000 | 4.9e-4 |
      | D `tl_fd_flat_split_mask` | 0.9999957 | 3.1e-5 |

      Against a torch recomputation, not against OpenPI: OpenPI has no AdaRMSNorm
      kernel to compare with, only a whole model. This gate is what localizes a
      failure to the kernel rather than the weights, which is the split that
      resolved the prefix bring-up.
- [x] Wire `pipeline.decoder()` and extend the engine to the full pass. Three
      graphs now: vision | prefix | decoder. See §4.7 for the measurement.
- [ ] Suffix parity against `PI0Pytorch(Pi0Config(pi05=True))` at `--steps 1`
      with `--layers` bisection.
- [ ] Re-tune. Every config is Pi0's, carried over unchanged, and every decoder
      shape moved (M 51 -> 50, keys 819 -> 1018). Deliberate: changing the tiling
      and the maths in one step makes a numerical failure un-bisectable.

**What the kernels cost, and the one trap.** `S` -- the per-K `(1 + scale)` --
is staged one `BLOCK_K` slice per mainloop iteration, alongside A and W. The
spec originally said to hold the whole 1024-entry vector in shared memory,
loaded once, and that **deadlocked**: under warp specialization a
global→shared `T.copy` outside `T.Pipelined` lowers to a producer-warp TMA with
no matching consumer arrival, so the kernel compiles, launches, and never
returns. It cost a 17-minute job that sat silent after the kernel finished
compiling. The origin kernels only ever load vectors into *fragments*, which is
a per-thread load with no barrier, so there was no precedent for the shape and
the argument for it — "keep the pipelined body free of an extra copy" — had it
backwards. Recorded in the spec's `deviations`.

Two consequences worth carrying forward: a TileLang kernel that deadlocks does
not fail, it hangs, so the gate takes `--only` and new kernels get one job each;
and `global→shared` outside a pipelined body is the shape to distrust.

### 4.4 Full parity — **landed**

`eval/correctness/pi05/suffix_parity.py`, against
`PI0Pytorch(Pi0Config(pi05=True))` on random weights.

| mode | steps | cosine | max_abs | rms |
|---|---:|---:|---:|---:|
| transplanted KV cache | 1 | **0.9999911** | 0.0176 | 0.00488 |
| transplanted KV cache | 10 | 0.9999841 | 0.0457 | 0.00634 |
| full pass | 10 | 0.9999840 | 0.0457 | 0.00636 |

**The KV cache is transplanted by default, on purpose.** The gate hands our
decoder OpenPI's own prefix cache instead of the one our encoder built. Our
prefix agrees to a layer-0 cosine of 0.99994 but drifts to 0.9973 by layer 17 on
random weights; feeding that into the decoder would mix two error sources in one
number. Transplanting leaves only decoder wiring and decoder kernels — which is
what nothing else covered, and the class of mistake (wrong per-step table slice,
wrong residual aliasing, wrong suffix RoPE offset) that produces a plausible
wrong answer.

`--steps 1` is the gated reading, because the flow loop is a chaotic map on
random weights and depth is an amplifier, not a defect.

**The prefix drift barely reaches the actions.** Full-pass and transplanted
agree to seven digits at 10 steps (0.9999840 vs 0.9999841), so the 0.9973 KV
cache drift washes out: the decoder attends over 1018 keys and averages.

One gate bug worth remembering: `--layers` originally truncated only *our*
decoder, so an 18-layer reference was being compared against our 1-layer run and
reported a meaningless 0.995. `openpi05.truncate_expert` now cuts both, and both
its `ModuleList` and its config count have to move — `GemmaModel` iterates
`self.layers[: self.config.num_hidden_layers]`.

Still open: none of this uses trained weights. `--checkpoint <pi05_base>` adds
the conversion of trained values; it says nothing more about the code.

### 4.5 Tuning — not started, and now the top item
Every config is Pi0's and every decoder shape moved. §4.7 says where to aim.
- Re-tune every call site at the new geometry: encoder M = 968, decoder M = 50,
  decoder keys = 1018. New sites: embed gather, the four AdaRMS variants.
- `autotune.sweep_kernel` with `correct=` on every candidate — some tilings are
  wrong, not slow.
- BLOCK_M=64 is a cliff, not a knob: below it TileLang drops off wgmma onto
  Ampere `mma.sync` (measured, see the spec's `mma_m` check). A sweep that
  straddles it is comparing two different kernels.

### 4.7 First end-to-end measurement

`python -m benchmarks e2e-pi05`, H100 SXM5, 3 views, prompt padded to 200,
chunk 50, 10 steps, 18 layers, median of 30. **Untuned** — every tile config is
Pi0's, carried over unchanged.

| stage | median | floor (§1.2) | above floor |
|---|---:|---:|---:|
| vision | 2.494 ms | 0.66 | 3.78x |
| prefix | 7.933 ms | 3.88 | 2.04x |
| decoder | 8.635 ms | 1.88 | 4.59x |
| **forward()** | **19.241 ms** | 6.42 | **3.00x** |

Launch overhead — three graph launches plus the input copies — is 0.179 ms,
0.9 % of the pass, so the stage sum accounts for the whole number.

**The host tokenizer is fully hidden, as designed.** It costs 0.152 ms on the
compute node (an order of magnitude more than the 16 us measured on the login
node, worth knowing), and `forward` measures 19.209 ms by wall clock against
19.241 ms by CUDA event — the wall clock is *no worse*, so none of that host
time is on the critical path. That is the two-graph split of §3.2 paying off,
measured rather than assumed.

**Where the time is.** The decoder is the worst stage at 4.59x its floor and the
largest absolute gap at 6.8 ms. Three known contributors, none yet separated:
the v1 AdaRMS fallback adds a `tl_rms_factor` launch per layer (180 extra nodes
over 18 layers x 10 steps); keys grew 819 -> 1018; and `S` is now staged per
mainloop iteration. Pi0's decoder measured 7.08 ms, so Pi0.5 is 1.55 ms worse on
a stage whose floor did not move.

Vision at 3.78x is Pi0's number unchanged (Pi0: ~2.41 ms) on identical work, so
it is a Pi0 debt inherited, not something Pi0.5 introduced.

Not comparable to Pi0's 15.18 ms headline: that is prompt 0, encoder M=768,
decoder keys 819. Pi0.5's prefix is 26 % longer by construction.

These are latency numbers; §4.4's suffix parity is what makes them mean
anything, and it now passes end to end at cosine 0.99998.

### 4.8 Profile and wave quantization

`python -m benchmarks profile-pi05`. Per-kernel self-device-time inside a graph
replay, plus the grid each decoder call site launches, derived from the live
configs in `wrappers.py` rather than restated.

**Wave quantization is severe, and it is the largest tuning target.** At M=50 the
decoder cannot fill the machine by tiling M -- one m-tile of 64 covers the whole
chunk -- so the CTA count is set almost entirely by `N / BLOCK_N`. Two kernels
carrying 37.3 % of decoder time run on a third to two thirds of the SMs:

| kernel | CTAs | SMs used | self | share | floor at full occupancy |
|---|---:|---:|---:|---:|---:|
| `tl_matmul_gated_res` | 128 | 0.97 | 2.527 ms | 28.4 % | 2.450 ms |
| `tl_ada_scaled_gate` | 128 | 0.97 | 2.238 ms | 25.2 % | 2.170 ms |
| **`tl_fd_flat_split_mask`** | **42** | **0.32** | 1.693 ms | 19.0 % | **0.539 ms** |
| **`tl_ada_qkv_gemm_rope`** | **80** | **0.61** | 1.629 ms | 18.3 % | **0.987 ms** |
| `tl_rms_factor` | 25 | 0.19 | 0.477 ms | 5.4 % | 0.090 ms |
| `tl_fd_flat_combine` | 200 | 1.00 | 0.268 ms | 3.0 % | — |
| `action_out_proj` | 4 | 0.03 | 0.042 ms | 0.5 % | — |

Ranked by what a fix is worth:

1. **`tl_fd_flat_split_mask`, ~1.15 ms.** The grid is 7 m-blocks × `NUM_SPLIT`,
   and `NUM_SPLIT` is 6 because `_FD_SPLIT` requests 7 — a Pi0 constant tuned at
   819 keys. At `BLOCK_N=64` the guard admits up to **16** splits (112 CTAs,
   0.85 SM); at `BLOCK_N=32` up to **32** (224 CTAs, full). Nothing about the
   kernel changes, only the requested split count.
2. **`tl_ada_qkv_gemm_rope`, ~0.64 ms.** `BLOCK_N=32` gives 2560/32 = 80 CTAs.
   `BLOCK_N=16` gives 160, and at 85 KB of smem two CTAs fit per SM, so all 160
   are resident — full coverage. `BLOCK_N=8` gives 320 at 75 KB, three per SM.
3. **`tl_rms_factor`, ~0.39 ms**, but the real fix is deleting it, not tiling it:
   1.33 us per call across 360 calls is launch latency, not work.

**What the v1 AdaRMS fallback actually cost: ~0.24 ms**, not the 0.72 ms
estimated from a 2 us-per-launch guess. Half of `tl_rms_factor`'s 360 calls are
new — Pi0 already ran one per layer for its qkv site — so the fallback is 180
launches at 1.33 us. Cheaper than assumed, which lowers the priority of the v2
fusion relative to the two wave-quantization items above.

The two kernels already at 0.97 SM carry 53.6 % of the decoder between them and
have almost no wave headroom. Whatever is left in them is not occupancy.

**Prefix.** `tl_matmul_gate` alone is 3.763 ms, 49.6 % of the stage — the
encoder gated FFN, which Pi0's README already records as compute-bound at 60 %
MFU. The embed gather is 5 us, negligible as predicted.

Correcting an earlier estimate in this file: the encoder attention was flagged as
0.32–0.64 ms of avoidable score-matrix traffic. Measured, the whole attention is
0.59 ms (softmax 0.241 + two GEMMs 0.350), so a flash-style MQA kernel is worth
at most ~0.35 ms, not the ~0.6 ms implied. It drops below both decoder items.

### 4.6 Packaging
- `hardware/.../pi05/README.md` with the measured profile; sbatch example.
- **Delete this file.**

---

## 5. Open questions for the owner

1. **Which checkpoint is the target?** `pi05_base` (horizon 50) is the default
   assumption above. `pi05_droid` (15) and `pi05_libero` (10, and no state at
   all) are different shape profiles and would each need their own tuning pass.
2. ~~`L_pad`~~ — **decided: 200**, matching openpi. See the ground rule.
3. **Task-text policy.** The head tokens are cached per episode. If the task
   string can change between two consecutive `forward()` calls without an
   explicit `set_task()`, that cache is wrong — confirm the calling convention.
