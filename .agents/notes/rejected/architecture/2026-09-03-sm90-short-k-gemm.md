# Agent Note: A hand-written SM90 GEMM for the short-K call sites

Status: rejected — at the vision and encoder shapes the kernel reaches cuBLAS
only at one site (+8%) and loses at the other four (2026-09-03). One wave of
CTAs is worth more than every other tiling property, and the shapes do not
let a hand-written kernel hold one wave while also cutting TMA issue count or
epilogue cost.

Uses the [SM90 tile primitive library](../../implemented/architecture/2026-09-02-sm90-tile-primitive-library.md).
Supersedes the "a fixed-shape kernel may take either site back" hypothesis
left open by [prefix GEMM epilogues](../../implemented/architecture/2026-09-03-prefix-gemm-epilogue.md)
and [vision GEMM epilogues](../../implemented/architecture/2026-09-02-vision-gemm-epilogue.md).

## Problem

The vision and encoder GEMMs are 7-22 us kernels at M=768/968 with K around
1-4K. cuBLAS reaches roughly half the measured tensor-core ceiling on them,
and the standing argument was that a library GEMM cannot know the shape:
it cannot size its tile so one wave of CTAs covers the problem, cannot pick
the cluster multicast shape for it, and cannot fuse bias, GELU or an in-place
residual. Those three are exactly what a fixed-shape kernel can do, so the
lane built one on the tile primitive library and measured it against cuBLAS
in the same cold-weight graph timer.

## Decision

Keep cuBLAS (and the TileLang gated-FFN body) at every short-K site; do not
wire the kernel. It stays in the tree unwired as the starting point for a
successor, with its configuration table and ablation switches.

Measured, same timer, cold rotating weights, parity passing everywhere
(cos >= 0.9999986, max_rel <= 3.6e-3):

| site | M x N x K | production | best candidate | verdict |
|---|---|---:|---:|---|
| vision qkv | 768 x 3456 x 1152 | 13.65 (cuBLAS) | 12.53 | +8%, worth 0.03 ms e2e |
| vision fc1 | 768 x 4304 x 1152 | 16.29 (cuBLASLt GELU) | 21.55 | loses |
| vision fc2 | 768 x 1152 x 4304 | 20.01 (TileLang) | 25.29 | loses |
| vision o_proj | 768 x 1152 x 1152 | 6.57 (cuBLAS) | 9.62 | loses |
| encoder o_proj | 968 x 2048 x 2048 | 13.95 (`addmm_`) | 18.69 | loses |

Only qkv beats its route, by less than the 10% aim, and 27 layers of 1.1 us
is 0.03 ms — under the 0.10 ms promotion bar on its own. Nothing is
promotable, so nothing was wired and no parity or e2e route job was run.

## Why it cannot pull away (measured)

An ablation that rebuilds the kernel once per column, all rows from one job:

| column | vision qkv | vision fc1 | vision o_proj |
|---|---:|---:|---:|
| full | 12.48 | 21.86 | 9.72 |
| copy alone (no mma) | 10.97 | 19.28 | 8.50 |
| math + epilogue (no load) | 11.34 | — | 6.15 |
| epilogue (full − no store) | 3.14 | 11.2 | 1.90 |

Two facts follow. First, the copy and math columns are each about 11 us at
qkv and already overlap almost perfectly, so the kernel's floor is its copy
column, not its math. Second, the copy column divided by the box count is
305 ns (qkv) and 236 ns per box (o_proj), i.e. the per-transaction TMA issue
cost applies **per SM**: splitting issue across four producer warps did not
parallelize it. Box count per CTA is `2 * K / BK` and nothing else, so the
only lever on the copy column is a larger BK — which, under the 32 KB box
limit, caps BM and BN at 128 and therefore raises the tile count.

That collides with the property that dominates everything else here: the
grid must be **one wave, as close under 132 CTAs as possible**. A config at
138 tiles (1.05 waves) measured 36 us against 21.5 for the same site at 126
tiles, because the handful of SMs that run a second tile set the critical
path; configs that leave SMs idle (84, 102, 64 tiles) are slower in
proportion. At N = 3456, 4304, 1152 and 2048 no BN that is a multiple of 64
lands the tile count just under 132, so the swizzled C staging tile — worth
3.1 us at qkv and 11.2 at fc1 by the ablation — is unreachable at the tile
counts that matter. The shapes force a choice between one wave, cheap
epilogues, and few TMA boxes, and cuBLAS's own schedule already sits at that
compromise.

## Alternatives considered

- Rotating each cluster's K start to grow unique bytes in flight: measured
  20% worse; it destroys the lockstep L2 miss-merge between CTAs.
- Cluster multicast: 2x1 helps at every site (and is in the best configs);
  4x1 and 2x2 are neutral to much worse.
- BK=128 with 3-D chunked boxes (halves box count): no gain at o_proj, and
  at qkv it forces 162 tiles, i.e. a second wave.
- Direct bf16x2 global stores instead of the staged TMA store: 3.1 us worse.
- Persistent multi-tile CTAs so one tile's epilogue hides under the next
  tile's mainloop: not built. It is the one untried structural answer to
  both the exposed epilogue and the wave imbalance, and is what a successor
  should start from — but it only pays where the tile count exceeds 132,
  which at these shapes means accepting smaller tiles and therefore more
  total TMA boxes, the cost the ablation says is already binding.

## Consequences

- No production route changes; the vision and encoder GEMM sites keep the
  routes chosen by the two epilogue notes above.
- `backends/cuda/kernels/sm90_gemm.cu` and its ctypes launcher stay in the
  tree, unwired and unreferenced by any op table, with a 21-entry config
  table and the `GEMM_ABL_*` ablation switches that produced the table above.
- The measured per-SM TMA issue cost and the one-wave dominance are the two
  reusable findings; a successor kernel at any of these shapes should price
  its box count and its tile count before its tile shape.

## Verification

Ablation decomposition job 589035 (ACD1-55, six rebuilds in one job so the
columns are same-node comparable); configuration sweeps 588835 (ACD1-20) and
589059 (ACD1-6); earlier rounds 588755-588822. Harness
`artifacts/ktasks/sm90-short-k-gemm/bench_gemm.py` (`graph_time_cold`,
rotating cold weights, cuBLAS references timed in the same job), ledger
`candidates.jsonl` c0-c4. Parity is against fp32 torch per candidate; no
route-level parity or e2e job was run because no candidate was promotable.
