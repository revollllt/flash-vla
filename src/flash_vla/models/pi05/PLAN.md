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

### 4.2 Prefix path — do this first
It changes the engine API and the buffer plan, so everything else sits on top.
- [x] Host table tokenizer + `set_task()` / per-episode head cache
  (`models/pi05/tokenize.py`).
- `vocab_embeddings` residency + embed-gather kernel (`×sqrt(2048)`, zero-fill).
- Prefix padding mask vector; encoder + decoder attention consume it.
- Data-dependent `decoder_rope_weights`; static encoder rope table.
- Two-graph split (§3.2).
- **Gate:** prefix prefill parity against `PI0Pytorch(Pi0Config(pi05=True))` —
  compare `past_key_values` per layer, at the same `max_token_len = 200`.

### 4.3 AdaRMS
- [x] Adapter-side fold: `weights.fold` builds the per-step tables (§2.1) from
      `time_mlp_in/out` + `posemb_sincos` + each site's modulation Dense, over
      the fixed schedule. Gated by `eval.correctness.pi05.fold_equivalence`.
- [ ] **Kernels go through the tile-dataflow gate.** Write the tile-level
  dataflow spec first — CTA tiling, mainloop, stage depth, warp specialisation,
  per-instruction mma iters — and get it reviewed before any kernel code. The
  open design question is where `(1 + scale)` is applied: on the A tile inside
  the consuming GEMM's mainloop, or fused into the producing residual kernel's
  epilogue (which also has the full 1024-wide row and could emit the RMS
  reduction directly). Do not pick this in code.
  The other two folded terms need no new kernel shape: `shift @ W` is the
  `Bias_local[j]` pattern already in `kernels/fused_norm.py`, and `gate` is a
  per-N multiply in the residual epilogue.
- **Gate:** `eval/correctness/pi05/fused_vs_unfused.py`, then suffix parity
  against openpi with `--steps 1` and `--layers` bisection.

### 4.4 Full parity + capture
- `eval/correctness/pi05/openpi_parity.py` mirroring the pi0 one, against
  `pi05_base`.
- Dump a real `pi05_base` state dict and **confirm the adapter key names** —
  they are currently inferred from `gemma_pytorch.py:42-57` +
  `modeling_gemma.py:430`, expected to be
  `paligemma_with_expert.gemma_expert.model.layers.{i}.{input_layernorm,post_attention_layernorm}.dense.{weight,bias}`
  and `...gemma_expert.model.norm.dense.{weight,bias}`, with **no**
  `...layernorm.weight` on the expert side.
- **Gate:** full-pass parity at the pi0 tolerance (max/mean/RMS/P99 + cosine).

### 4.5 Tuning
- Re-tune every call site at the new geometry: encoder M = 968, decoder M = 50,
  decoder keys = 1018. New sites: embed gather, AdaRMS
  variants of `norm_qkv_rope` / `norm_gated_ffn` / `action_out_proj`.
- `autotune.sweep_kernel` with `correct=` on every candidate — some tilings are
  wrong, not slow.
- **Gate:** median-of-30, cold, in-graph; per-stage split recorded against the
  §1.2 floors and the pi0 15.18 ms baseline.

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
