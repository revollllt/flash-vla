# Interview bank

Questions worth asking, and the ones people forget. Read the section for the
archetype you are working on; skip the rest.

The method is in SKILL.md: batch by section, propose a default with its
arithmetic rather than quizzing, derive whatever follows from settled fields.
This file is the *content* — what to ask about once you are asking well.

## Contents

- [Opening — before any section](#opening--before-any-section)
- [Universal: per-section questions](#universal-per-section-questions)
- [Archetype: GEMM](#archetype-gemm)
- [Archetype: attention / flash](#archetype-attention--flash)
- [Archetype: grouped / MoE](#archetype-grouped--moe)
- [Archetype: reduction / normalization / elementwise-fused](#archetype-reduction--normalization--elementwise-fused)
- [Questions people forget](#questions-people-forget)

## Opening — before any section

Four questions settle more downstream fields than anything else. Ask them
together, first.

1. **What are the real shapes, and which are dynamic at runtime?** Not the
   general case — the shapes this kernel will actually run at. A kernel for
   M=51, K=1024 is a different kernel from the same math at M=8192.
2. **What is the regime: latency at small M, or throughput at large M?**
   Decides persistent-vs-wave, tile size, and whether split-K/split-KV is even
   on the table.
3. **What is the fusion boundary?** What is inside this kernel and what stays
   outside. Epilogue fusion is the cheapest performance in the whole design and
   the easiest to forget to ask about.
4. **Is there a reference implementation, and what is the numerical contract?**
   Bitwise-vs-tolerance, and where the accumulation happens in fp32.

If the user has an existing kernel to port or beat, read it now. Extracting its
spec first (see the reverse-engineering section of SKILL.md) turns most of the
remaining interview into confirmation.

## Universal: per-section questions

**Grid.** Persistent or wave — and if persistent, what does the scheduler
guarantee about load balance? What rasterization order, and is L2 reuse of A or
of B the one worth optimizing? Is a cluster worth it (only when one operand is
genuinely shared between adjacent CTAs)?

**Mainloop.** Which axis is the mainloop, when there is more than one candidate?
Does the step size have an external constraint — a quantization scale
granularity, a KV page size, a swizzle atom? Is the extent guaranteed divisible
by the step, and if not, predication or padding?

**Pipeline.** How deep can smem afford, and how deep does the load latency
need? These two usually disagree; the reviewer needs to know which one bound
the answer. Is anything worth aliasing to buy a stage?

**Warp groups.** Producer/consumer, or cooperating math groups? (See
`spec-schema.md` §4 — the two are different kernels, not variants.) If
cooperating: what is the dependency between groups, and what named barrier
enforces it?

**Iters.** What MMA shape, and is the N dimension the full tile N or a split?
Where does the accumulator live and does it survive the mainloop? What runs
between the MMA batch and the barrier release?

**Epilogue.** Fused or separate kernel? Does the output need a smem round-trip
(TMA store does; `st.global` does not)? Any cross-CTA reduction, and if so what
combines it?

## Archetype: GEMM

- Which operand is K-major and which is MN-major? This determines the wgmma
  descriptor major-ness and whether a transpose is needed in smem.
- Is there **quantization scaling**, and at what granularity? Per-tensor,
  per-channel, per-128-block, per-32-block (MX/NV formats)? This is the field
  that decides whether the accumulator can stay in the wgmma accumulator for
  the whole mainloop or must be drained and promoted every K-block — a
  first-order structural difference, not a detail.
- Is K large enough to need split-K? If yes: atomics or a combine kernel, and
  what is the partial's dtype?
- Does the epilogue fuse a bias, an activation, a residual add, or a cast? In
  what order?
- Is B reused across many A tiles (weights) such that persistent CTAs holding B
  resident would pay off?
- Is one of M/N/K tiny (a projection from 32 to 1024, a single decode token)?
  Below one wave the whole analysis changes: the kernel is launch- and
  bandwidth-bound, deep pipelines and warp specialization cost more than they
  return, and the honest answer may be a much simpler kernel.

## Archetype: attention / flash

- Are Q, K, V separate tensors or one packed buffer? Head dims for QK and for
  V — MLA has `d_k=576, d_v=512`, and specs that assume they are equal break.
- MHA, MQA, or GQA — how many Q heads share a KV head? This sets the effective
  M of the QKᵀ GEMM, which is what decides compute-bound vs memory-bound in
  decode.
- Causal, sliding-window, or full? Where is the mask applied, and can whole
  KV blocks be skipped rather than masked?
- Is the KV cache paged? Page size becomes the mainloop step, and the block
  table becomes an indirection the TMA descriptor must handle.
- **Online softmax state**: confirm explicitly that running max `m` and running
  sum `l` are loop-carried, where they live, and when the accumulator gets
  rescaled. This is the single most-omitted item in attention specs.
- Split-KV across CTAs? If yes, the combine kernel needs its own spec, and
  `softmax_lse` becomes an output.
- Prefill or decode? Decode with small `q_seqlen` puts the whole M dimension
  below one wgmma tile, and the kernel becomes a different shape entirely.
- Is the P matrix (post-softmax) handed to the PV GEMM from registers or from
  smem? Register handoff avoids a round trip but forces the two GEMMs into the
  same warp group; smem handoff is what lets two warp groups cooperate.

## Archetype: grouped / MoE

- Contiguous-grouped (one ragged M axis with a group offset array) or masked
  (fixed padded extent per group with a validity count)? These are different
  schedulers.
- Is the group boundary aligned to the CTA tile M? When it is not, either the
  tile is split across groups (wrong) or the scheduler must skip — say which.
- Does the tensor map need to be rewritten per group (changing base pointer or
  inner stride)? On Hopper that means updating a TMA descriptor in smem and
  fencing it, which is a real pipeline hazard worth its own spec line.
- How is load imbalance across experts handled — dynamic scheduler, or accept
  the tail?
- Is the routing/permutation inside this kernel or a separate one?

## Archetype: reduction / normalization / elementwise-fused

- Is the reduction axis contained within one CTA, or does it span CTAs? Within
  a CTA the whole `pipeline` section usually collapses; across CTAs you need a
  two-pass or atomic scheme.
- Is this actually bandwidth-bound? If yes, say so in `regime` and stop
  optimizing tile shapes — the questions that matter become vectorization
  width, coalescing, and how many passes over the data.
- What fuses in: RMSNorm + gate + up-projection is one kernel or three, and the
  answer changes the buffer plan completely.
- Is the intermediate precision fp32 even when inputs and outputs are bf16?

## Questions people forget

Ask these when the corresponding section looks finished. Each one has bitten a
real kernel.

- **The tail.** `mainloop.tail`. What happens when the extent is not divisible
  by the step.
- **The first iteration.** Is the accumulator cleared by the MMA's `ScaleOut::Zero`
  on iteration 0, or pre-zeroed? Both work; unstated is a bug.
- **The last iteration.** Who releases the final stage's empty barrier, and does
  the producer need an extra round of waits before the barriers can be
  destroyed? (DeepGEMM needs exactly this under TMA multicast.)
- **Barrier arrival counts.** Not 1 by default. Count the arriving warps, and
  multiply by cluster size for multicast.
- **The phase bit.** `(iter / depth) & 1`. Omitted → deadlock.
- **Smem alignment.** TMA needs 128-byte alignment; 128-byte swizzle needs the
  buffer 1024-byte aligned.
- **Register budget under `setmaxnreg`.** The producer's dealloc and the math
  group's alloc must sum within the SM's per-thread budget at the stated
  occupancy.
- **What the epilogue costs in smem.** It is often the largest single buffer
  and it is not staged — leaving it out of the smem check overstates the
  affordable depth by a whole stage.
- **Whether the output buffer can alias a staged input.** Frequently yes after
  the mainloop drains, and frequently worth a stage.
- **Dynamic values that are secretly static.** If M is "dynamic" but always 51
  in production, saying so unlocks an entire class of specialization. Ask.
