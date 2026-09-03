# Agent Note: Fused encoder MQA attention as a CUDA kernel

Status: implemented

Date: 2026-09-03

## Problem

The Pi0.5 encoder ran its attention as a torch.compile chain of three kernels
(cuBLAS QK^T, a Triton softmax, cuBLAS PV) that materializes a 7744 x 968
score matrix: 30.7 us per layer, 0.52 ms over 17 layers. Library SDPA
backends were rejected earlier (cudnn 60 us, efficient 82 us, flash refuses
the additive float mask), and the TileLang fused kernel reached only 25.6
us/layer, under the promotion bar
(`.agents/notes/rejected/architecture/2026-09-03-encoder-mqa-flash-attention.md`).

## Decision

`enc_attn.cu` on the shared SM90 tile primitives, launched through a C ABI
and ctypes like the decoder task loops. One CTA owns 64 flattened query rows
(one wave of 121 CTAs); Q is loaded once and stays resident; two TMA producer
warps fill independent K and V frame rings; one math warpgroup runs the
log2-domain online softmax with the fragment reuse that feeds P straight into
the P.V wgmma. The three tensor maps are encoded once per buffer triple on
the host, since `cuTensorMapEncodeTiled` is not graph-capture safe.

The kernel writes into a new `encoder_attn_out` buffer instead of returning a
fresh tensor, and `pipeline.encoder` selects it. `ENC_ATTN_ROUTE=torch`
restores the reference chain; that switch is what makes an A/B for this call
site possible, and the chain remains the parity reference.

## Alternatives considered

- **Two math warpgroups splitting the key blocks (FlashAttention-3
  ping-pong), which the rejected TileLang note named as the design that would
  clear the bar.** Measured 28.4 us/layer against 22.9 for one warpgroup, with
  or without the named-barrier stagger. Shared memory caps the ring at six
  32 KB frames, so a second warpgroup halves the lookahead each one gets; this
  kernel is bound by producer depth, not by softmax overlap. The decisive
  evidence is that halving the ring instead (depth 3 -> 2, one warpgroup)
  costs the same 5.4 us/layer.
- **Independent K and V ring depths with separate issuing warps** (K4/V2,
  K3/V3): equal within 1%. Total ring capacity is what matters, not its split
  across the two streams nor the number of issuing warps. K5/V1 is 32.8 us --
  both streams need at least two slots.
- **A lone producer warp instead of a producer warpgroup.** `setmaxnreg` is
  warpgroup-wide, so a single producer warp cannot hand its registers to the
  math side: ptxas capped the kernel at 168 registers, spilled 192 B and
  serialized the wgmma pipeline (C7511). Three idle warps are cheaper.

## Consequences

- The prefix stage runs 125 kernels per inference instead of 159; two launches
  per layer disappear. `benchmarks/layer_breakdown.py` carries the new
  sequence (`attn:fused`).
- `buffers.py` gains `encoder_attn_out` (4 MB) so the fused kernel has a
  stable, capture-safe destination.
- The kernel holds 226 KB of shared memory and one CTA per SM. Its shape
  constants (head dim 256, 64-row tiles, 64-key blocks) are compiled in;
  `keys` and the row count are runtime arguments, and the ragged last key
  block is covered by the TMA's out-of-bounds zero fill plus a padded mask
  slice.
- The remaining gap to the arithmetic floor is 2.3x (22.1 us in-graph against
  a ~10 us tensor-core time for 16 key blocks), and the named cause is ring
  capacity. Cluster multicast of the K/V frames is the untested lever: every
  CTA re-reads the whole 1 MB stream, so a cluster of 4 would cut both the L2
  traffic and the per-CTA TMA issue count.

## Verification


- Kernel gate: `eval/correctness/pi05/enc_attn_parity.py` -- the fused kernel
  against the torch chain it replaced, against an fp32 recomputation over
  valid query rows, and a separate finiteness check on the padded rows (the
  mask scaled by log2(e) overflows to -inf, so a fully-masked block would
  yield NaN without the kernel's running-max floor). It is also the
  regression gate the tile library's README names for `tile/sm90/*.cuh`
  edits, alongside the two task-loop parity scripts.
- Kernel parity vs fp32 torch over valid query rows, every candidate:
  cosine 0.9999973, max relative error 3.8e-3 (bf16 rounding). Harness and
  ledger in the task workspace `artifacts/ktasks/mqa-cuda/`.
- Per-kernel, same cold-rotating graph timer as the chain, in-job control:
  22.9 us/layer against 30.7 (jobs 589060, 589061, 589066, 589067, 589109,
  589110, 589111; the in-job chain time spans 30.63-30.90 us across six
  nodes, so the numbers compare directly).
- Route gates: `prefix_parity` against OpenPI (job 589128, openpi env) passes,
  layer-0 K/V cosine unchanged to 6e-8 and the largest per-layer change 3.0e-5
  in the favourable direction; `plan_parity` on the pdl plan, 1 step x 18
  layers (job 589129) passes.
- End to end, same-job A/B/A on ACD1-33 (job 589207, 30 reps per leg), prefix
  stage min: torch 6.302, cuda 6.146, torch 6.302. Forward wall min 16.274 /
  16.062 / 16.233. The two prefix reference legs
  reproduce exactly (6.302 / 6.302) and the wall legs to 0.041 ms, so the
  -0.156 ms prefix is 3.8x its own control spread and the -0.19 ms wall is
  4.6x its own. Neither clears the partition's 6% noise floor, which is why
  the same-job A/B/A design is the evidence and a cross-job comparison of
  these numbers would not be.
- In-graph trace (job 589127): `attn:fused` 22.14 us/layer over 125 launches,
  no measurable inter-kernel gap in the prefix graph.
