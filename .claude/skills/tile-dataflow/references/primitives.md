# Named non-MMA primitives

Some non-MMA computations recur with the same state, the same ordering hazards
and the same numerical traps every time. Spelling them out entry by entry in
`non_mma` invites each spec to get them subtly wrong in a new way. So they get
**names and contracts** here, and a spec references the name plus its
parameters — the way FlashAttention's repo carries a `Softmax<kNRows>` object
rather than open-coding the rescale in every kernel.

Referencing a primitive is not a licence to skip the fields that vary. `where`,
`cost`, `touches` and `on_critical_path` are still per-kernel and still required;
what the primitive fixes is the *algorithm*, its *state*, and the *hazards*.

## `online_softmax`

The streaming softmax of FlashAttention: one pass over the key blocks, carrying
a running max and sum per query row, rescaling the output accumulator whenever
the max moves.

**State**, carried across mainloop iterations, one pair per query row, in
registers:

```
  m   running row max      f32, initialised to -inf
  l   running row sum      f32, initialised to 0
```

Both belong in `mainloop.loop_carried` alongside the output accumulator. A spec
that lists only the accumulator is wrong.

**Per iteration**, given the score block `S[rows, block]` fresh from the QK MMA:

```
  m_new  = max(m, reduce_max(S, axis=block))
  alpha  = exp2((m - m_new) * scale_log2)          # rescale factor for the carry
  P      = exp2(S * scale_log2 - m_new * scale_log2)
  l      = l * alpha + reduce_sum(P, axis=block)
  acc_o  = acc_o * alpha                            # <-- THE HAZARD, see below
  acc_o += P @ V                                    # the second MMA
  m      = m_new
```

**Epilogue**: `acc_o = acc_o / l`, and the log-sum-exp a split kernel needs to
combine partials is `log2(l) + m * scale_log2`.

### The three things that go wrong

1. **The `acc_o *= alpha` must precede this iteration's `P@V` MMA.** It is a
   CUDA-core pass over the whole output accumulator, sitting between two MMAs on
   the same registers, and it is the reason an attention kernel's L3 timeline is
   not the same shape as a GEMM's. Put it after, and the previous block's
   contribution is scaled by the wrong factor — a silent, data-dependent error
   that a short parity test can easily miss.
2. **Work in log2 space, not natural log.** Fold `1/sqrt(d)` (and any other
   scalar) into `scale_log2 = scale * log2(e)` once at launch, then use
   `ex2.approx` throughout. `exp` costs several instructions where `exp2` costs
   one, and this sits in the innermost loop.
3. **Fully masked rows.** If a row can have zero valid keys, `m` stays `-inf`,
   `m - m_new` is `NaN`, and the NaN propagates into `acc_o` where the running
   max update then hides it. Either clamp `m` to a large finite negative before
   the subtraction, or state in the spec why no row can be fully masked. Both
   are acceptable; silence is not.

### Parameters a spec must supply

| Parameter | Meaning |
|---|---|
| `rows` | how many query rows one instance carries state for — sets the register cost, `2 * rows` f32 per thread-group |
| `block` | the key-block extent reduced per iteration — this is the reduction's cost |
| `span` | where `reduce_max` / `reduce_sum` live: lane-local if the score fragment layout keeps a row within a thread, otherwise a `shfl` tree. Getting this wrong is the usual order-of-magnitude cost error |
| `first_iter` | `specialised` (skip the rescale and the max compare on iteration 0) or `uniform` |
| `masked_rows` | `check_inf` / `clamped` / `impossible because ...` |
| `p_cast` | the dtype `P` is cast to before the second MMA, usually bf16, and where that rounding lands |

### Split variants

A split-KV kernel runs `online_softmax` per split and then combines. The combine
is its own `non_mma` entry, not part of this primitive: it reads each split's
`(acc_o, lse)` and does a second, tiny online pass over the splits. State the
split count and whether any split can be empty — a split that starts past the
key count produces a garbage partial that the combine's max update will hide.

## `row_rms` — STUB, do not reference by name yet

**Contract not written**: no parameters table, no worked instance. Read it as
background on the fusion and its rounding trap, not as a name to cite.

`F[m] = rsqrt(mean_k(x[m, k]^2) + eps)`, the RMSNorm factor, when it is fused
into the GEMM that consumes it rather than run as its own kernel.

The fusion is legal because **row scaling commutes with the K reduction**:
`rms(x) @ W` and `(x @ W) * rstd(x)[:, None]` are the same value, so the factor
does not have to exist before the GEMM starts. Accumulate the sum of squares
from the same shared-memory tile the MMA consumes, and apply the factor in the
epilogue.

The trap is that **where the factor is applied changes the rounding**, and
therefore the function:

- applied to the **f32 accumulator in the epilogue** — mathematically exact, one
  rounding;
- applied to the **bf16 A tile inside the mainloop** — a second rounding per
  element, and it requires the factor to exist *before* the mainloop, which
  defeats the fusion unless the tile is held resident and read twice.

Neither is wrong. The spec must say which, `non_mma.dtype` must record it, and
the parity reference must mirror it — two implementations differing only here
will not compare bit-for-bit, and the difference is real but uninteresting.

Also note: the factor is usually **stored to bf16** by a standalone
`rms_factor` kernel. A fused version that keeps it in f32 is *more* accurate and
therefore does not match the unfused one. Say which you are reproducing.

## `split_reduce` — STUB, do not reference by name yet

**Contract not written.** No parameters table, no worked instance in this skill.
A spec that writes `primitive: split_reduce` looks specified and is not, which
is worse than writing the reduction out. Written out below is what is known;
promote it to a real primitive when a kernel here needs it.

The cross-CTA reduction a split-K GEMM needs, when the partials stay inside a
cluster instead of going through global memory. Its barrier is **not** removable:
plain `st.shared::cluster` pushes have no transaction counter behind them, so
deleting it is a data race rather than a saving. And `mbarrier.arrive` defaults
to `.release.CTA` — CUTLASS's `ClusterBarrier::arrive(smem, cta_id, pred)` emits
a bare `mbarrier.arrive.shared::cluster`, which orders nothing for a peer reading
the DSMEM stores it was meant to announce. Use `cute::cluster_sync()`, or explicit
`.release.cluster` / `.acquire.cluster` on both sides.

Two schedules, and the push form is better:

- **pull**: `cluster_sync()`, then every rank reads its own rows out of all
  peers' smem via `mapa`. Needs two barriers.
- **push**: every rank writes each owner the rows that owner will reduce, then
  **one** `cluster_sync()`, then each rank sums its partials out of its own
  smem. measured 0.65 us cheaper on one split-K kernel (`[MEAS]`, one machine), and it lets the local partial buffer alias
  retired mainloop stages — the received buffer cannot alias, because a peer
  writes it while this CTA may still be in its mainloop.

Sum the partials in a fixed rank order so the result is deterministic across
runs; split-K already reorders the reduction relative to an unsplit kernel, and
a non-deterministic order makes that difference unreproducible on top.
