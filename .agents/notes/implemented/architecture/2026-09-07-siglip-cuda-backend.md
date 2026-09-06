# Agent Note: the shared SigLIP vision backends

Status: implemented

## Problem

The `vision_encoder` segment is the campaign's first lane
([optimization campaign plan](../../proposed/performance/2026-09-06-optimization-campaign-plan.md),
lane B). Both H100 Targets run the same SigLIP tower at the same shapes, and
job 598964's floor report put the segment at 2.05 ms (Pi0.5) and 2.54 ms (Pi0)
against a measured ceiling of 1.01 ms on both, with 245 launches per pass.

Two things made the gap look larger than it is. The 0.49 ms between the two
Targets is almost entirely two call sites: Pi0 runs TileLang GEMMs where Pi0.5
runs LayerNorm followed by a GEMM with a fused cuBLASLt epilogue, at identical
shapes. And the four GEMM sites hold most of the remaining headroom, which the
lane's contract proposed to take with a warp-specialized kernel folding the
LayerNorm into the GEMM's A side.

## Decision

1. **A shared component package with two backends.**
   `hardware/nvidia/h100/siglip/` owns the vision kernels, their reference
   mirrors and their backend factories, written once and registered by both
   Targets. `siglip-cublas` is the torch-plus-cuBLASLt form of the two
   pre-norm projections; `siglip-cuda` adds a hand-written LayerNorm and a
   fused attention kernel. `geometry.py` is the single dimension mirror and
   fails at import if the two model specs ever disagree about the tower.
2. **Attention is one kernel, and the pad stays in shared memory.**
   `siglip_attn.cu` replaces the three launches of the torch route with one,
   reading the packed QKV projection as the graph declares it. head_dim 72 is
   neither a swizzle width nor a multiple of the wgmma K step, so the score
   contraction runs at K=80 with the tail held at zero. TMA cannot produce
   that image, because it writes a box contiguously; cp.async can, with nine
   16-byte chunks per row and a pad column written once per frame. The packed
   buffer keeps its declared layout, so the projection and the attention stay
   independently routable and the backend declares no route constraint.
3. **The LayerNorm is worth its own kernel.** torch's generic `F.layer_norm`
   measured about 3.4 us per call above the tuned TileLang body, which is what
   made the pure-torch route a regression on Pi0.5. `siglip_norm.cu` is one
   CTA per row, fp32 statistics, one rounding on output; it also writes a
   caller-provided buffer, so the two projections take their workspace from
   the runner's injected allocator as the component-package note intends.
4. **The four GEMM call sites keep cuBLASLt and TileLang.** This reverses the
   contract's primary candidate, on the prior lane's evidence rather than on a
   new measurement, and the reasoning is below.

## Why the LayerNorm fold was not built

[A hand-written SM90 GEMM for the short-K call sites](../../rejected/architecture/2026-09-03-sm90-short-k-gemm.md)
already built this kernel family on the same tile library at these same
shapes. It beat cuBLAS at one site by less than the promotion bar and lost at
the other four. Its ablation named the binding cost: the copy column rather
than the math is the kernel's floor, the per-transaction TMA issue cost applies
per SM and does not parallelize across producer warps, and box count per CTA is
twice K over BK and nothing else. One wave of CTAs dominates every other tiling
property.

Folding the LayerNorm into the A side requires the row block to be resident in
shared memory, which caps BLOCK_M at 64. That halves the A box and doubles the
tile count, so at the QKV shape the critical path goes from about 36 boxes per
SM to about 72; cluster-2 multicast with split issue and a wider N tile recover
it only to about 45, against cuBLAS's own position. The fold's saving is real
but small in the captured graph, where the separate LayerNorm launch costs
about 2.5 us rather than the 13 us the cold isolated timer charges it. The
candidate prices at roughly 0.08 ms per site, under the bar.

The thesis that the fold adds no traffic is true about bytes and false about
transactions, and transactions are what binds at short K. That is the
reusable finding: **the floor model's measured ceiling is not reachable at
these shapes, because a third constant — the per-SM TMA issue cost — binds
before either the bandwidth or the tensor-core term the ceiling divides by.**

## Alternatives considered

- **Build the fold anyway and measure.** Rejected: the prior lane's ablation
  predicts the outcome from arithmetic this task does not change, and the
  budget is better spent on the two candidates whose prior rejection does not
  transfer.
- **A padded or split QKV produced by the projection's epilogue**, giving the
  attention kernel aligned TMA. Priced and deferred: it buys aligned loads at
  the cost of an atomic route group over two call sites and a second write of
  the QKV buffer, and the shared-memory pad makes it unnecessary.
- **Another SDPA backend.** Already measured worse by the earlier vision lane.
- **Keeping torch's `F.layer_norm`.** Rejected on its own A/B/A: it is what
  makes the route a regression on Pi0.5.
- **Routing `siglip-cublas` on Pi0.5.** Rejected on its own A/B/A: Pi0.5's
  TileLang route already reaches the same cuBLASLt GEMMs through a faster
  LayerNorm.

## Consequences

- **Pi0's shipped plan routes its two pre-norm vision projections and its
  vision attention to `siglip-cuda`.** Pi0.5's does not: its best candidate
  measured 0.094 ms against the registry's 0.10 ms bar, because its TileLang
  LayerNorm already reaches the same cuBLASLt projections. Both Targets carry
  both backends either way, and the candidate plans under `lab/plans/` isolate
  the LayerNorm, the attention and the combined route.
- The package imports `models/`, `runtime/` and the tile library and no
  Target; the grep the component-package note requires stays empty.
- `benchmarks kernels` needs no new case: it derives cases from the routed op
  table, so routing a call site is what registers it.
- The attention kernel's `acc_to_aregs` is a second copy of the
  FlashAttention-3 accumulator reinterpretation that the Pi0.5 encoder
  attention kernel also carries. It belongs in `tile/sm90` with an
  `eval.tile_sm90` case once a third consumer appears.
- The four GEMM sites are closed to this lane's budget with a named blocker
  rather than an exhausted one.

## Verification

Login node: `python -m eval.smoke`; `python -c "import flash_vla"`; both
kernels compile for sm_90a, and the attention kernel's PTX carries the padded
contractions the design requires (five wgmma m64n64k16 for K=80, four
m64n80k16 for the output) at 136 registers with no spill, so no mainloop was
eliminated.

GPU, clocks unlocked, same process per A/B/A, 100 reps. Each job's legs are on
one node; nodes differ between jobs, so only within-job deltas are compared:

- Job 599790, ACD1-8: baselines for both Targets on both plans, the floor
  denominator on the current runtime, and the torch-LayerNorm candidate.
- Job 599825, ACD1-8: T2 parity caught the attention kernel's softmax
  denominator unreduced across the quad (rel_rms 3.156 at cosine 0.988) before
  any timing was read. Both metrics were needed: an error that large in
  magnitude still leaves the direction almost right, so cosine alone would have
  called it close.
- Job 599835, ACD1-28: the LayerNorm candidate's A/B/A on both Targets, and
  `eval.correctness` reporting `replay_determinism` false for the attention
  kernel while every tolerance held. No tolerance gate would have caught that,
  and the T2 parity did not: one warm call usually finds the data already
  there.
- Job 599860, ACD1-28: parity and the A/B/A after both attention fixes.

## Evidence summary

Instrument discipline: `benchmarks floor` is cited only for per-call-site
attribution and `ceiling_us_each`; every segment- and chunk-level number here
is `benchmarks latency`, same process, 100 reps, clocks unlocked, `min` read,
with the delta taken against the mean of the run's two reference control legs.

Vision-encoder call sites, in-graph attribution, job 598964 on ACD1-55, with
the measured ceiling each site is judged against:

| call site | ceiling us | Pi0.5 us | Pi0 us |
|---|---:|---:|---:|
| `patch_embed` (x1) | 3.52 | 13.09 | 13.31 |
| `norm_qkv` | 7.28 | 16.50 | 22.29 |
| `attention` | 4.41 | 11.16 | 11.46 |
| `out_proj_residual` | 4.73 | 7.28 | 8.51 |
| `norm_ffn_up` | 9.99 | 19.11 | 28.10 |
| `ffn_down_residual` | 10.82 | 22.27 | 22.22 |

Candidate results:

| candidate | job, node | Target | chunk min delta | vision min delta | control spread | verdict |
|---|---|---|---:|---:|---:|---|
| torch LayerNorm + cuBLASLt | 599790, ACD1-8 | Pi0 | -0.286 | -0.292 | 0.0166 | superseded |
| torch LayerNorm + cuBLASLt | 599790, ACD1-8 | Pi0.5 | +0.066 | +0.107 | 0.0346 | rejected |
| CUDA LayerNorm + cuBLASLt | 599835, ACD1-28 | Pi0 | **-0.426** | -0.371 | 0.0246 | kept |
| CUDA LayerNorm + cuBLASLt | 599835, ACD1-28 | Pi0.5 | -0.001 | +0.013 | 0.0436 | neutral |

Parity, T2 mirrors on random weights at the production shape, gated on the
registry's shallow bf16 pair (`rel_rms` < 6.6e-2, cosine > 0.99978):

| call site | rel_rms | cosine |
|---|---:|---:|
| `norm_qkv` | 3.57e-5 | 0.9999999994 |
| `norm_ffn_up` | 8.31e-5 | 0.9999999995 |
| `attention` | 1.53e-3 | 0.9999988 |

The in-engine 1x1 gate passes on both Targets for every kept candidate, with
`replay_determinism` and finiteness.

Promotion, `python -m eval.gate --reference shipped --baseline --reps 100`,
job 599893 on ACD1-20. The candidate is the Target's shipped route plus the
vision change and the reference leg is `shipped`, so the A/B/A measures the
deployed configuration rather than the change against the reference route:

| Target | verdict | chunk min, shipped / candidate / shipped | improve | spread | tail |
|---|---|---|---:|---:|---:|
| Pi0 | **pass** | 15.3036 / 14.8819 / 15.3118 | -0.4217 | 0.0081 | 0.127 |
| Pi0.5 | fail | 15.9386 / 15.8450 / 15.9185 | -0.0936 | 0.0201 | 0.081 |

Every gate passed on both Targets, `baseline_layer0` included, and both runs
were valid with the tail well inside the 0.5 ms bound. Pi0.5 fails on the
candidate rule alone and by 0.0064 ms; it is kept and not promoted. Pi0's
vision segment goes 2.5075 to 2.0680 ms in the same run.
