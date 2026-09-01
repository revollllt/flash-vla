# FFN task loop: in-place latency probes (L2 prefetch, TMA geometry) — rejected

Status: rejected (2026-09-02), measured. Two candidates under contract
`ffn-taskloop-inflight` (workspace `artifacts/ktasks/ffn-taskloop-inflight/`,
ledger has per-round numbers). Both numerically clean; neither moved e2e.

## Problem

Post-PDL, `ffn_taskloop_kernel` is 42.5% of the decoder critical path
(17.95 us/layer marginal vs an 8.35 us weight floor). NCU (583993 /
lineinfo 584074): long_scoreboard 10.6-11.9 cyc/issue on the ring waits,
barrier 8.2 on the GU->DR slot seam (`ffn_taskloop.cu:858` region);
DRAM read 40% in GU, 21.8% in DR.

## Probe 1 — DR-weight L2 prefetch from the reserved warp during GU

- 8 tiles at slot entry: +2.2 us/layer (+0.4 ms e2e, 584115/584116).
- 4 cold tiles after the stationary XFS lands: +1.0 us/layer (+0.18 ms vs a
  same-node no-define control, 584173 vs 584174).
- Cost per prefetch instruction 250-275 ns = `[tma.issue.warp]` (248 ns)
  within 10%. Working hypothesis: TMA descriptor ops, prefetch included,
  serialize in the SM's TMA unit across warps (`[tma.bw.cta.warps]` was
  measured bandwidth-bound and does not rule this out). A HUT probe of two
  warps issuing small boxes is needed before citing it as a constant.

## Probe 2 — DR TMA geometry to the saturation standard

Audit against `frontier.py --min-box` (>= 13.5 KB/box at 128-132 CTAs x 1
warp): GU meets it (32 KB, 4 txns = 0.99 us column); DR did not (8 KB weight
boxes, 8 txns = 1.98 us; 8 KB hidden boxes capped by the row-major 128 B
SW128 row limit, 16 txns = 3.97 us). Candidate: hidden stored [FF, M_PAD]
M-contiguous so DR reads one 32 KB [M64 x K256] box per stage, 16 KB weight
boxes, MN-major A operand; K order unchanged.

- Gate 584506: hidden/out cosine 1.0000000, replay x3 — bit-clean.
- e2e 584507/584508 (ACD1-58): pdl 7.418/7.438 with tilelang legs
  8.271/8.330; normalized to tilelang, -0.853/-0.892 vs the baseline band
  -0.841..-0.878 => neutral.

## What the two nulls establish

1. DR is neither bandwidth-bound nor TMA-issue-bound: its 4 us copy column
   was overlapped by a serial chain — counter observe RTT -> first
   activation load -> 4 wgmma stages -> split-K join RTT -> fold -> residual
   RMW. Shortening the column moved nothing; the chain is the pole.
2. Adding TMA operations to the GU phase costs ~248 ns each regardless of
   which warp issues them.
3. GU's 40% DRAM at 32 KB boxes and ring depth 3 (one below
   `[tma.stages.warp.knee]`) has no mechanism yet: per-CTA delivery is
   ~13 GB/s against a 24 GB/s fair share. Withdrawn without measurement:
   "contention knee", "hidden_ready gating" (as a stated fact), and the
   2 CTA/SM retile as next lever (`[tma.bw.sm.cta.scale]` favors one wide
   CTA 1.40x on cache-hot feeds).

## Successor direction

Measure before changing: a hardware-unit-test probe of single-CTA delivery
at GU's exact geometry (1 producer warp x 32 KB x depth 3 vs 4, 128 CTAs)
names GU's gap; a two-warp small-box issue probe settles per-SM vs per-warp
TMA serialization. Kernel-side, the DR lever is chain length, not bytes:
the split-K=4 join (a counter RTT plus a 3-partial fold per tile) was
justified by the copy column, which no longer binds once boxes are at the
cap — S=2 halves joins and partial traffic at an 8-stage column that still
fits under the chain.

## Verification for any successor

gu/dr/full parity + plan_parity both depths + A/B/A e2e with a same-node
no-define control job; normalize cross-node comparisons on the tilelang leg.
