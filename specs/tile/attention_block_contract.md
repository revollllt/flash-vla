# Pi0.5 action-expert attention block: interface and measurement contract

Status: normative for every implementation of the block, current and future.
Owner: the attention-block optimization line
([Agent Note](../../.agents/notes/rejected/architecture/2026-08-27-attention-block-taskloop.md), rejected 2026-08-27 with the measured attribution);
work plan in [`attention_block_plan.md`](attention_block_plan.md).

This document fixes three things so that every optimization step is compared
against the same thing: **(1)** the tensors an implementation receives and
produces, with shape, dtype and majorness; **(2)** the numerical contract and
parity gate; **(3)** how the block's latency is measured. An implementation
that needs a different external tensor changes this document first, in the
same PR.

The interface is the fused kernel's ABI, `attn_taskloop_launch` in
[`sm90_attn_task_desc.cuh`](../../src/flash_vla/hardware/nvidia/h100/pi05/backends/cuda/kernels/sm90_attn_task_desc.cuh),
which stays authoritative for shapes and geometry. Two torch mirrors of that
ABI exist and the harness is built on them:

- [`attn_block_reference.py`](../../src/flash_vla/hardware/nvidia/h100/pi05/backends/cuda/attn_block_reference.py)
  -- the block as one call (`AttnBlockReference.forward`), same buffers, same
  mutation, no tiling; plus `make_inputs()` producing one layer-step of inputs
  at the ABI's shapes. This is the block-level parity reference.
- [`attn_reference.py`](../../src/flash_vla/hardware/nvidia/h100/pi05/backends/cuda/attn_reference.py)
  -- the same ABI decomposed task body by task body over the kernel's split
  structure, geometry parsed from the header. This is what stage bisection
  diffs against.

Both are checked against the hardware-independent algorithm in
[`models/pi05/reference.py`](../../src/flash_vla/models/pi05/reference.py);
`eval/correctness/pi05/kernel_parity.py` owns the rounding contract when a
parity number is disputed.

## 1. Scope: one layer-step of the attention half, RMS factor outside

```
out = out + ( MQA( RoPE(F * ((x*s) @ W_qkv) + b), K_cache, V_cache, mask ) @ W_o ) * g
K_cache[PREFIX_LEN:KEYS] = K_step,   V_cache[PREFIX_LEN:KEYS] = V_step
```

`F = rsqrt(mean(x^2) + eps)` is an **input** (`rms_factor`), exactly as in
the FFN prototype: the RMS reduction stays outside this launch and joins as a
task kind only in the full-layer scope. Every implementation measured under
this contract therefore receives the factor and is timed without it; the
TileLang control passes the same factor into `tl_ada_qkv_gemm_rope` directly
rather than running `tl_rms_factor`. The block **excludes** the FFN half,
`action_in_proj` and `action_out_proj`.

## 2. Shape profile

The reference profile is the tuned pipeline configuration and the only one
the fused kernels compile for. TileLang implementations treat the same
numbers as runtime values. Names are the header's.

| symbol | value | meaning |
|---|---:|---|
| `M` | 50 | action chunk = query rows |
| `M_PAD` | 64 | padded rows of every activation; one wgmma m64 tile |
| `D` | 1024 | action-expert width |
| `H` | 8 | query heads (multi-query: one K/V head) |
| `DH` | 256 | head width |
| `QKV_W` | 2560 | `H*DH + DH + DH` |
| `PREFIX_LEN` | 968 | `3 views x 256 image tokens + 200 prompt` |
| `KEYS` | 1018 | `PREFIX_LEN + M` |
| `KEYS_PAD` | 1024 | `KEYS` rounded up so every kv-split walks the same trip count |

A different `num_views` or `prompt_len` is a different profile: fused kernels
recompile, and the harness is invoked with the matching constants. The launch
validates `prefix_len` / `n_ctas` against its compiled profile and fails
rather than silently reading the wrong cache rows.

## 3. Interface tensors

All tensors are CUDA device memory, contiguous, **row-major** (last dimension
contiguous), bf16 unless stated. "inout" tensors are mutated in place.
Argument names are those of `attn_taskloop_launch` and of
`AttnBlockReference.forward`.

### 3.1 Activations and per-inference inputs

| name | shape | role | notes |
|---|---|---|---|
| `x` | `(M_PAD, D)` | in | projection input; rows `[M, M_PAD)` unspecified (the harness zero-fills them) |
| `rms_factor` | `(M_PAD,)` | in | `rsqrt(mean(x^2)+eps)` on real rows, **zero on pad rows** |
| `rope` | `(M_PAD, DH)` | in | cos in even columns, sin in odd; positions `n_valid + m` |
| `key_mask` | `(KEYS_PAD,)` | in | additive: `0` on real keys, `MASK_NEG = -3.0e38` on `[n_valid, PREFIX_LEN)` and on `[KEYS, KEYS_PAD)`; identical for every query row |
| `k_cache` | `(KEYS_PAD, DH)` | inout | rows `[0, PREFIX_LEN)` read-only prefix; rows `[PREFIX_LEN, KEYS)` written by this block (rotated K); rows `[KEYS, KEYS_PAD)` **never written**, finite (host zero-fills at allocation) |
| `v_cache` | `(KEYS_PAD, DH)` | inout | as `k_cache`, unrotated V |
| `out` | `(M_PAD, D)` | inout | residual in, `residual + gated projection` out; **may alias `x`** (the pipeline aliases them) and an implementation must be correct in both cases |

`n_valid` (the tokenized prompt length) is a host quantity that only shapes
`key_mask` and `rope`; no kernel receives it.

**Pad-row policy.** The whole block is row-independent along `M`: every stage
maps row `m` to row `m`, and the only cross-row path -- K/V into the cache --
writes real rows only. Pad rows of `x`, `out`, `q_buf` and `o_buf` are
therefore *unspecified*: an implementation may write them (the fused epilogue
does, unmasked, because it is cheaper) and no consumer may read them as real.
Parity compares real rows only. The two hard rules are the ones that keep the
invariant true: `rms_factor` pad rows are zero, and cache rows `>= KEYS` are
never written.

### 3.2 Weights and folded AdaRMSNorm vectors (per layer, per step)

| name | shape | role | layout notes |
|---|---|---|---|
| `w_qkv` | `(D, QKV_W)` | in | `(K, N)`; columns `[0, H*DH)` are Q with column `h*DH + d`, then K `[H*DH, H*DH+DH)`, then V. RoPE pairs are adjacent `(2p, 2p+1)`: the checkpoint permutation is done offline by `spec.weight_shapes()`, never at runtime |
| `qkv_bias` | `(QKV_W,)` | in | `shift @ w_qkv`, folded |
| `ada_scale` | `(D,)` | in | `1 + scale`; multiplies the A operand inside the contraction |
| `w_o` | `(H*DH, D)` | in | `(K, N)`; contraction index `h*DH + d` |
| `ada_gate` | `(D,)` | in | multiplies the projection before the residual add |

Weights are consumed **as stored**: no implementation may require a
pre-blocked or transposed weight without changing this table. (The FFN
pre-blocks its gate/up weights; the attention geometry does not need it.)

### 3.3 Caller-allocated scratch, part of the ABI

| name | shape | dtype | meaning |
|---|---|---|---|
| `q_buf` | `(H, M_PAD, DH)` | bf16 | **head-major** rotated Q: head `h`'s queries are one contiguous `(M_PAD, DH)` slab |
| `o_buf` | `(H, M_PAD, DH)` | bf16 | head-major attention output; `o_proj` reads head `h` as k-slice `[h*DH, (h+1)*DH)` |
| `qkv_partial`, `attn_partial`, `attn_lse`, `out_partial` | sizes in the header | f32 / bf16 / f32 / f32 | split partials; split 0 reduces |
| `counters` | `(kCount,)` = 160 | u32 | zeroed on-stream before every launch (graph-capturable memset) |
| `table`, `dbg`, `timeline` | header | i32 / i64 / i64 | task table; watchdog record; optional per-task `%globaltimer` stamps `(N_CTAS, TASK_SLOTS, 5)` for critical-path analysis (null disables) |

`q_buf` and `o_buf` are *observable* stage buffers: the fused family fills
them and the harness diffs them against `attn_reference`. The partials,
counters, table and watchdog buffer are scheduling state with no algorithmic
meaning and are never compared. The TileLang control implements the same
Python call signature, ignores the scratch it does not need, and leaves
`q_buf` / `o_buf` untouched; the harness skips the stage diff for it.

### 3.4 Pipeline binding

The pipeline (`h100/pi05/buffers.py`) allocates the decoder-side buffers
with these padded extents -- `decoder_x`, `decoder_rope_weights`,
`decoder_norm_factor_buf` at `M_PAD` rows, `encoder_K` / `encoder_V` /
`prefix_mask_bias` at `KEYS_PAD` -- and hands its call sites the leading-row
views of the unpadded shapes. An implementation behind a pipeline call site
recovers the padded base from the view and must refuse a view that is not
backed by one (`backends/cuda/wrappers.py`). Pad entries are `MASK_NEG` on
the mask and zero elsewhere, and no kernel writes a pad row; the row
independence of section 3.1 is what makes that sufficient.

The per-op form (`standalone` mode, `STANDALONE_OPS`) additionally offers a
token-major combine (op 6) that writes the attention output as
`(M, H * DH)` into the pipeline's `decoder_q_buf`, the layout the TileLang
`decoder_out_proj_residual` consumes; only the `M` real rows are written.

## 4. Numerical contract and parity gate

Rounding points, shared by the two torch mirrors and the kernel:

1. `a = bf16(x * s)` is formed in bf16 before the QKV contraction (the kernel
   forms it in shared memory inside the mainloop).
2. QKV accumulates in fp32; the epilogue applies `rms_factor` then adds `b`,
   **then** rotates the Q and K columns in fp32. `q_buf` and the cache rows
   are stored bf16.
3. Logits, running max, softmax and the P·V accumulation are fp32; split
   partials may be bf16 (the fused design chooses bf16 for the largest
   buffer); the combine is fp32; `o_buf` is bf16.
4. `o_proj` accumulates in fp32; `out + acc * g` is fp32; the store is bf16.

`AttnBlockReference` keeps the whole chain in fp32 and rounds only at buffer
boundaries, so agreement to a few `1e-4` in cosine is expected, not to the
last bit.

Gate, applied to every implementation in every configuration it is measured
in (`eval/correctness/pi05/prefix_parity.error_metrics`; bar `cosine > 0.999`,
`max_abs` reported):

- `out[:M]` against `AttnBlockReference.forward` on the same inputs.
- `k_cache[PREFIX_LEN:KEYS]`, `v_cache[PREFIX_LEN:KEYS]`: same gate.
- Cache rows `[KEYS, KEYS_PAD)` and prefix rows `[0, PREFIX_LEN)` are
  bit-unchanged.
- Both aliasing modes (`out is x`, `out is not x`) pass.
- **Replay safety**: two consecutive invocations from identical inputs (fresh
  `out`, fresh caches, fresh counters) produce bit-identical outputs.
  Deterministic reduction order is required; atomic fp32 accumulation into
  outputs is not allowed.
- **Stage bisection** (fused family): with the truncated task table that runs
  one kind, `q_buf` and the cache suffix (`kQkvProj`), `o_buf`
  (`kAttention`, with `q_buf` and caches pre-filled from `attn_reference`),
  and `out` (`kOutProj`, with `o_buf` pre-filled) each pass the same bar
  before the full table is run, because a persistent-kernel bug hangs rather
  than fails.

Performance is never reported for a configuration whose parity did not pass
in the same process invocation.

## 5. Fused family: stage contracts

The stage boundaries are `attn_reference`'s functions; the harness pre-fills
inputs from them and diffs outputs against them.

| task kind | reads | writes | reference functions |
|---|---|---|---|
| `kQkvProj` | `x, rms_factor, ada_scale, w_qkv, qkv_bias, rope` | `q_buf`, cache suffix rows | `qkv_partial` -> `qkv_epilogue` -> `qkv_scatter` |
| `kAttention` | `q_buf, k_cache, v_cache, key_mask` | `o_buf` | `attention_partial` -> `attention_combine` |
| `kOutProj` | `o_buf, w_o, ada_gate, out` | `out` | `out_proj_partial` -> `out_proj_epilogue` |

A geometry change (split counts, `BN`, `KEYS_PAD`) edits the header only;
`attn_reference` re-derives its decomposition from it, so the stage contract
cannot drift from the kernel.

## 6. Measurement contract

### 6.1 The quantity

**Block latency** = GPU time from the start of the first kernel to the end of
the last kernel or memset that one block invocation issues, with the
invocation captured in a CUDA graph and replayed, against a cold L2. Gaps
between the nodes of the graph are inside the span: the pipeline pays them,
and removing them is most of what fusion buys, so a measurement that dropped
them would hide the effect under test.

For the fused kernel this is the counter memset plus one kernel. For the
TileLang control it is the four-node span `tl_ada_qkv_gemm_rope` ->
`tl_fd_flat_split_mask` -> `tl_fd_flat_combine` -> `tl_matmul_gated_res`,
each called with the contract's tensors (`rms_factor[:M]` passed in, Q in the
control's own token-major scratch). `tl_rms_factor` is outside the span on
both sides.

### 6.2 Primary timer: CUPTI over a CUDA graph

```python
from flash_vla.bench import bench_gpu_time, KernelResult
samples = bench_gpu_time(block_fn, input_args=(...),   # one block per call
                         enable_cupti=True, use_cuda_graph=True,
                         cold_l2_cache=True, repeat_iters=REPS)
```

- `cupti-python 13.0.1` is installed in the repository venv; the harness
  refuses to fall back to events silently and records which backend ran.
- `use_cuda_graph=True` captures one call and replays it; each sample is the
  span of §6.1 (`flash_vla.bench.timer.bench_gpu_time_with_cupti`).
- Cold L2 comes from the `2 x L2` flush the timer issues before every replay;
  no rotating weight sets are needed. (A rotating-set scheme would need at
  least 6 sets for this block: one layer's attention weights plus cache are
  ~10.5 MB, and 3 sets would sit inside the 50 MB L2.)
- `REPS >= 30`; report `median`, `min`, `p99`, `n` via `KernelResult`, with
  `flops` / `bytes` from the reference shapes so achieved TB/s is printed.

### 6.3 Diagnostics recorded with every measurement

- **Per-kernel CUPTI records** for one replay: kernel name, grid, duration.
  `span - sum(kernel durations)` is the launch/ramp gap term and is reported
  next to the span. This is how the control's per-kernel numbers are
  obtained; the previously recorded `10.44 / 10.25 / 5.42 us` are not cited
  because they predate the current toolchain.
- **Cross-check timer**: the FFN harness's event-timed graph with rotating
  cold sets (`ffn_taskloop_parity.run_bench` method, with enough sets to
  exceed L2), reported alongside. The two timers agree to within the noise
  floor when the harness is sound; disagreement is a harness bug to fix
  before any kernel conclusion.
- **Environment**: node (`[job] node=`), CUDA, torch, `nvcc` flags, git
  commit, whether `pin_gpu_clocks` succeeded (`[warn] could not lock GPU
  clocks` means it did not).

### 6.4 Comparison rules

- Candidate and control are measured **in the same process and the same
  job**, interleaved (A B A B, at least three rounds each) whenever the
  claimed difference is below twice the noise floor. The noise floor on this
  cluster without pinned clocks is ~6%; cross-job deltas below that are not
  conclusions.
- The control for the block is the TileLang composition through the same
  call signature. The control for a fused-family experiment is the previous
  fused revision, rebuilt from source in the same job.
- Floors and targets divide by `hardware-unit-test/sm90/constants.yaml` tags
  (`tma.issue.warp`, `tma.bw.cta.dram`, `ld.bw.dev.dram`, `launch.lat.dev.ramp`, ...), never by
  datasheet peaks. A result is classified against the binding term of its
  floor, or it is marked exploratory.
- A performance claim in a note or PR names: the harness command, job id,
  node, timer backend, `n`, and the control it beat.

### 6.5 Harness

`eval/correctness/pi05/attention_block_parity.py` is the single entry point.
Inputs come from `attn_block_reference.make_inputs` on the device; the block
gate is `AttnBlockReference`; stage bisection uses `attn_reference`; timing
is §6.2 with the §6.3 dump; `--bench` runs only after parity passed in the
same invocation. Diagnostics: `--stage-bench` (one task kind on its truncated
table, against its floor), `--timeline` (per-task stamps: dependency wait,
first frame, mainloop, join, epilogue, plus the five slowest tasks per
kind), `ATTN_NVCC_DEFINES` ablation builds with `--force-timeline`,
`sbatch/profile_attn.sh` for ncu. Jobs go through `sbatch/pi05_cuda.sh`.
