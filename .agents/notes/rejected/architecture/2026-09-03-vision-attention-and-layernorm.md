# Agent Note: The vision stage's non-GEMM overhead stays as it is

Status: rejected — every candidate measured below the promotion bar, and the
one that looked best in isolation is slower in the captured graph

Date: 2026-09-03

## Problem

The vision stage spends 82.5 us per layer over 27 layers. Three GEMM call
sites account for 62.5 us of that; the rest is attention (a cuDNN workspace
memset, a 3.4 us graph-node gap and an 8.7 us `scaled_dot_product_attention`
kernel, 12.9 us together for 0.91 GFLOP against a 1.07 us arithmetic floor),
two LayerNorms at 2.5 us each, and inter-node idle. The lane asked whether a
fused attention kernel reading the packed QKV projection directly, a
LayerNorm folded into an adjacent GEMM, or a re-tuned fc2 tile could recover
any of it.

## Decision

Keep the cuDNN SDPA route, both LayerNorm launches, and the shipped
`_VIS_FFN_DOWN`. Nothing is wired; the candidate kernel and the probes stay
in the task workspace.

## Why the fused attention loses (measured)

A TileLang kernel that reads the packed `(views, 256, 3*16*72)` buffer
directly, runs one online-softmax pass and writes the output layout the
out-projection expects — no permute, no workspace, no memset — reaches
10.42 us against 11.77 us for the production route in the isolated
cold-weight timer, a 1.35 us/layer gain that is already under the bar.

In the captured 27-layer graph the sign flips. Replacing the pair with the
fused kernel changes the per-layer wall from 75.95 to 78.47 us, i.e.
**+0.068 ms over the stage**:

| | baseline | candidate |
|---|---:|---:|
| memset + SDPA / fused kernel | 0.74 + 8.45 | 9.76 |
| kernel sum per layer | 71.91 | 71.87 |
| graph wall per layer | 75.95 | 78.47 |

Two facts follow, and they are the reusable part. The isolated timer
**overstates** the production route because it charges a cold QKV read and
the node gap, while in the graph the projection GEMM has just written that
buffer: cuDNN's kernel costs 8.45 us there, not 11. And the 3.4 us gap is not
an artifact of the memset→SDPA edge that fusing removes — with one fewer
node the candidate's total gap grew from 4.04 to 6.60 us per layer.

## Why the kernel cannot be made faster in TileLang

head_dim 72 is hostile to every fast shared-memory fill path, so the kernel
is stuck on element-wise predicated loads (tens of thousands of 2-byte loads
per CTA):

- **TMA**: 72 bf16 columns are 144 B, not one of the swizzle widths, so
  TileLang pads the shared layout and `LowerBulk` refuses the bulk copy
  ("cannot fall back to normal copy ... padded shared layout").
- **Vectorized copy into a padded tile**: writing the 72 real columns and the
  8 pad columns is two writes to one buffer inside the pipelined loop, which
  the pipeline planner rejects as overlapping stage writes.
- **Unpadded 72-wide tile**: compiles, but a wgmma K extent of 72 produces
  wrong results (silently, not as an error).

Unblocking any of these means changing the QKV layout — padding each head to
80 columns or splitting Q/K/V — which the projection GEMM would have to
produce. That projection is a cuBLAS call today, so it means a hand-written
GEMM with a scatter epilogue. It is the one site where the hand-written sm90
GEMM did beat cuBLAS (12.53 vs 13.65 us, see the 2026-09-03 short-K GEMM
note), so the follow-up is a two-kernel co-design, not a fix inside this lane.

## Why the LayerNorm fold loses (priced, not built)

Both consumers are cuBLAS(Lt) calls, so folding means a hand-written GEMM,
and those are measured at 12.53 vs 13.65 us (qkv, wins) and 21.55 vs 16.29
(fc1, loses) in the same timer.

- fc1: the fold begins 5.26 us behind and saves a 2.56 us launch, so it is
  2.7 us worse before the prologue costs anything.
- qkv: LayerNorm + GEMM is 16.61 us in-graph and the bar is 3.7 us/layer, so
  the prologue would have to be free. It cannot be: the statistics span the
  full K=1152 row and no CTA holds one, so at BN=128 the 27 N-tiles of a row
  block each re-read the same 128x1152 rows to recompute the same statistics
  (~46 MB against a 1.7 MB input), and 162 CTAs breaks the one-wave rule that
  dominates these shapes.

Folding into the *producer* GEMM's epilogue instead fails for the mirror
reason: its output tile is 128 of 1152 columns, so nine CTAs hold each row
and the statistics need a cross-CTA reduction — atomics plus a finalize
launch, which re-adds the launch being removed, or a cooperative grid sync
the vision graph does not use.

## Alternatives considered

- Another SDPA backend: already measured worse (cudnn 11.71, flash 12.81,
  efficient 16.63 us against the default 11.87).
- Re-tuning fc2 in the graph rather than in isolation: the right question,
  and deeper pipelining at a smaller BK does help — 64x128x64 with 6 stages
  is 0.64 us/layer faster than the shipped 64x128x128 with 3 — but that is
  0.017 ms, inside the stage's 6 % noise floor. Deeper rings at BK=128 are
  worse, 216-CTA and 54-CTA tilings much worse. Recorded for a future lane to
  bundle, not promoted alone.

## Verification

Jobs, all with parity passing (fused kernel cos 0.99999988 against the
production route on engine data; every fc2 config cos 0.9999987):
589117 and 589183 (candidate sweeps, isolated timer), 589149 (the TMA and
unpadded forms, 72/72 compile and 72/72 numeric failures), 589125 (fc2 tile
configs in the full-layer graph), 589216 (both surviving candidates in the
27-layer graph, the measurement that decided the lane).

No route-parity or e2e job was run: the contract does not warrant one below
the bar, and the combined candidate is negative.
