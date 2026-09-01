# Decoder PDL chain: dependent launches with dep-free prefetch above the wait

Status: implemented (2026-09-02). Owner of the decoder-wide PDL launch chain;
extends [cooperative XFS producer + PDL](2026-08-28-cooperative-xfs-pdl.md),
which owns the single producer->FFN edge this note generalizes.

## Problem

The decoder graph is a single-stream linear chain (~1280 nodes); each launch
pays 0.95-1.24 us of grid ramp inside its self-time (graph replay does not
remove it), and each kernel's dependency-free loads (weights, prefix K/V)
start only after its predecessor fully drains. Before this change one of ~6
per-layer boundaries had PDL, with its trigger after the producer's grid sync
(~90% point) and its wait above the dependency-free weight ring.

## Decision

Extend PDL over every boundary the CUDA backend owns, with the vllm#49791 /
sglang#23965 discipline: triggers fire at primary entry, dependents launch
under the primary's tail, and every dependency-free load is hoisted above
`cudaGridDependencySynchronize()`.

- FFN wait is a mode (0 off / 1 entry-wait / 2 role-split). Mode 2 releases
  the weight-loader and reserved warps; only readers of producer-written
  state (XFS activations, readiness counters, residual) wait. The wait stays
  after the barrier-init `__syncthreads` or the release is void.
- The chain rms -> qkv -> attention -> combine gains entry triggers and waits
  at each first dependent read: qkv's sole dependent operand is `rms_factor`
  (its whole weight/x mainloop runs pre-wait); attention pre-issues the mask
  and all pure-prefix K/V frames (`frame_needs_kv` splits them; 968/1024 keys
  are prefix) and waits only before suffix frames + Q; combine waits at top.
  Combine does not trigger: its TileLang successor cannot carry the launch
  attribute (cooperative launcher, deferred per the 2026-08-28 note).
- The fused producer gets a `TRIGGER_AT_ENTRY` JIT variant (entry trigger,
  post-sync trigger skipped) selected only on the PDL chain; the base plan
  keeps the conservative post-sync placement.
- Route surface: one `cuda-pdl` variant backend (same wrappers,
  `pdl_chain=True`); which boundaries overlap is chosen by which call sites a
  plan routes there (`attn-ffn-cuda-fused-producer-pdlffn` = FFN half only,
  `...-pdl` = full chain). Atomic-route guards extended; mixing cuda and
  cuda-pdl inside one atomic route is rejected.

## Safety argument (source-verified)

- A trigger is added only where the successor carries a wait (PTX
  griddepcontrol chain rule). TileLang 0.1.11 / tvm_ffi: kernels without
  `pdl_sync` carry no programmatic attribute, so no TileLang successor
  early-launches without a wait.
- No WAR hazard across overlap windows: the TileLang kernels form hard
  serialization fences per layer, confining overlap to
  {rms -> qkv -> attention -> combine} and {producer -> ffn}; qkv's
  post-wait epilogue writes cache rows [968,1018) while pre-wait attention
  reads rows <= 960; no scratch aliasing across the chain.
- Cross-proxy: every generic-store -> TMA-read hop keeps
  `fence.proxy.async` after its wait; combine's `__ldcg` reads rely on the
  prerequisite's bulk-store flush, preserved because `griddepcontrol.wait`
  spans full grid completion.
- Early triggers are memory-safe by PTX semantics: `.wait` guarantees
  prerequisite completion + visibility; `.launch_dependents` only schedules.

## Alternatives

- TileLang launcher patch for a cooperative PDL consumer (combine ->
  producer edge, now the largest un-overlapped boundary at ~7 us of
  producer runtime): deferred again, same reasoning as 2026-08-28. Next task.
- `pdl_sync` inside TL kernels (ffn -> next rms edge): rejected — worth only
  ~0.025 ms total and costs `__restrict__` in that kernel.
- Trigger-placement tuning beyond entry: no headroom — the trace shows the
  three new edges already near-fully overlapped.
- Megakernel: unchanged endgame; this chain is the incremental bubble-fill
  and its per-edge dependency audit feeds the planner design.

## Verification and results

Gates: `plan_parity --steps 1 --layers {1,18}` (cos > 0.999 + bit-identical
third replay) pass for both plans on every job below; 10x18 reported-only
fail by design (chaotic flow map, same slot as main). Bit-identical replay
doubles as the race check.

Same-process A/B/A (tilelang / base / -pdlffn / -pdl / base / tilelang),
torch 2.13.0+cu130, clocks unpinned — read min. Decoder-stage min ms:

| job | node | base (two legs) | -pdlffn | -pdl |
|---|---|---|---|---|
| 583718 (reps 30, co-located) | ACD1-1 | 7.665 / 7.649 | 7.651 | 7.778 (outlier) |
| 583719 (reps 30, co-located) | ACD1-1 | 7.653 / 7.707 | 7.670 | 7.506 |
| 583759 (reps 50) | ACD1-13 | 7.597 / 7.612 | 7.606 | 7.458 |
| 583760 (reps 50) | ACD1-13 | 7.570 / 7.600 | 7.592 | 7.447 |
| 583922 (reps 50, entry trigger) | ACD1-50 | 7.599 / 7.621 | 7.591 | **7.399** |
| 583923 (reps 50, entry trigger) | ACD1-1 | 7.598 / 7.631 | 7.592 | **7.391** |

Promotion rule (-0.20 ms vs best same-process base on two nodes, gates
green, base unregressed) met by the entry-trigger revision: -0.200 /
-0.207 ms. Decoder best moves 7.659 -> 7.391 ms (tilelang 8.2-8.4).

Trace evidence (job 583761, ACD1-58, Chrome trace + pair-overlap analysis):
mean overlap rms->qkv +0.92 us, qkv->attn +6.90 us, attn->combine +11.70 us;
the pre-existing producer->ffn edge overlapped only +0.52 us with the
post-sync trigger — which is why the FFN-only variant measures noise
(-0.01..+0.02 ms in all six jobs): its window only cashes in composition
with the overlapped attention chain. Caveat for future profiling: post-PDL,
kernel self-times include wait time (attention reads ~13 us, combine
~13.6 us in-trace) — per-kernel trace durations are no longer latency.

## Required verification for successors

Any change to trigger or wait placement re-runs the parity pair on both
node generations plus one A/B/A e2e job per generation; any new PDL edge
needs the WAR audit repeated for its overlap window.
