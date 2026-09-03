# pi05-latency-loop — state

Goal: pi0.5 H100 forward wall min on ACD1-33 (pdl plan, `sbatch/profile_pi05.sh`)
from 16.838 ms (job 585554) to <= 14.5 ms, via repeated
profile -> analyze -> optimize -> validate rounds over vision, prefix (LLM
backbone) and decoder (action expert), bf16 numerics preserved.

## Baseline and environment

- Revision: main after the tile-library commit (see `git log -1` at init);
  the goal's measured baseline is job 585554 on HEAD 66bc9b6:
  vision 2.019 / prefix 6.919 / decoder 7.445 / wall 16.838 ms (min).
  Per-kernel decoder numbers: non-PDL trace 585140 (PDL traces inflate).
- Node for comparisons: ACD1-33 (`sbatch -w ACD1-33`); cross-node deltas are
  normalized on a tilelang control leg; clocks unpinned, noise floor 6%.
- Env: `.venv` (torch 2.13.0+cu130, tilelang 0.1.11), CUDA 13.1 + gcc 13.3
  modules via `sbatch/_common.sh`; openpi parity env
  `/data/user/jzou521/codes/cuda/openpi-official/.venv`.
- Machine constants: `.claude/skills/hardware-unit-test/sm90/constants.yaml`
  (850 TFLOP/s bf16, 2.77 TB/s DRAM marginal, 1.24 us launch ramp,
  `tma.bw.dev.burst` cold-burst curve).

## Active truth (authority order)

1. latest user confirmation (this goal: target 14.5 ms; bf16 with
   reduction-order-level tolerance, no FP8/quant/approximation; custom kernel
   replaces cuBLAS when faster, aim >= 10%).
2. `.agents/notes/` — decisions; one note per shipped/rejected decision.
3. `artifacts/ktasks/<task>/{contract.md,candidates.jsonl}` — per-kernel task
   contracts and ledgers (gitignored, absolute path under the main repo).
4. `todo.md` (this dir) — the ranked headroom queue; item-level state lives
   only there.
5. this file — phase, gates, evidence, next action.

Skills that bind each step: `kernel-design` (contract -> reference -> parity
-> candidate loop -> promotion), `benchmark-kernel`, `gpu-profiler-analysis`,
`ncu-report`, `hardware-unit-test`, `server-usage`.

## Phases and gates

| # | gate | evidence | state |
|---|---|---|---|
| G1 | sm90 short-K GEMM (tile library) beats cuBLAS at vision qkv/fc1 and is promoted, or is rejected with evidence | sweep JSON same-timer vs cuBLAS; parity; e2e A/B | closed: REJECTED (note 2026-09-03-sm90-short-k-gemm; qkv +8% only, fc1 loses) |
| G2 | prefix short-K sites (o_proj, qkv_rope) and gate_up/down epilogues re-measured and promoted/rejected | same | closed: gate_up/o_proj/down promoted (588760, prefix 6.306 ms); the custom-GEMM retake is moot after G1 |
| G3 | decoder GU L2-warm: HUT decisive pair run; kernel candidate promoted/rejected | HUT probe JSON + constant; ffn parity; plan_e2e A/B/A both node gens | closed: REJECTED (constant tma.bw.dev.burst.warm recorded; candidate null e2e; note in rejected/) |
| G4 | prefix MQA flash attention promoted/rejected | attention parity vs torch chain; e2e | closed: TileLang form rejected; CUDA form PROMOTED (todo#4b, -0.19 ms wall) |
| G5 | decoder small-kernel fusions (rms_factor->qkv, combine->producer) promoted/rejected | plan parity; e2e | closed: BOTH REJECTED (rms fold +0.08, +FFN entry trigger +0.56, combine bound 0.15-0.2 ms) |
| G6 | full re-profile at final HEAD on ACD1-33; wall min <= 14.5 ms; `profiles/pi05/latency-breakdown.md` rewritten | e2e/profile/traces JSON; layer_breakdown sequence assert | open |
| G7 | every promoted change has: parity gate passed, Agent Note, ledger line, local commit | git log + notes | open |
| G8 | final independent review: 3 read-only reviewers (correctness/tests, design/boundaries, security/maintainability), no unresolved high-severity finding | review reports | open |

Each round: (1) profile (`CAPTURE_TRACES=1 CAPTURE_PLAN=attn-ffn-cuda-fused-producer-pdl sbatch -w ACD1-33 sbatch/profile_pi05.sh`, plus a non-PDL profile when decoder per-kernel numbers are needed); (2) analyze with `benchmarks/layer_breakdown.py` and re-rank `todo.md`; (3) run the top items as kernel-design tasks, parallel subagents in worktrees where file ownership is disjoint; (4) validate + promote; (5) update this file.

## Execution contract

- Promotion bar per change: e2e min improvement >= 0.10 ms in a same-job A/B
  (or same node vs the current baseline), all parity gates pass, note +
  ledger written, local commit. A custom kernel replaces cuBLAS at a site
  when it is faster in the same timer; aim for >= 10%.
- Numerics: bf16 outputs; reduction-order differences allowed; whole-stage
  gates are `prefix_parity` (openpi env, layer-0 cosine and smooth drift vs
  the recorded baseline `artifacts/ktasks/encoder-qkv-rope/runs/prefix_parity_base_585323.json`)
  and `plan_parity` for the decoder.
- Retry: 3 attempts per item, then record and defer; deferred items keep
  their gate open. Slurm queue waits are recoverable: open an independent
  lane (next site's profile/contract, HUT probe, note work) while waiting.
- Subagents: `Agent` forks with `isolation: worktree`; in a worktree run
  `ln -s <MAIN>/.venv .venv`, `mkdir -p sbatch/logs`, and
  `git submodule update --init third_party/cutlass` (or symlink to MAIN's
  checkout WITHOUT committing it); before merging, `git checkout HEAD -- third_party`
  and reject any submodule typechange. Never edit `kernels/base.py` or a
  `.cu/.cuh` while a TileLang/CUDA build job submitted from that tree is
  queued or running.
- Commits: local only, on main after review of the agent branch; message
  per `.claude/rules` with the Co-Authored-By / Claude-Session trailer. No
  push.
- Progress report after each productive loop:
  `Progress [..] N% (k/8 gates)` / this loop + remaining / next action.
- Learning: after each round summarize disproven assumptions; write to
  `.claude/skills/kernel-design/references/wiki/` only from promoted or
  rejected candidates (established results); memory updates need the user.

## Evidence so far

- 2026-09-02 rounds 0-1 (before this goal): -1.03 ms via vision smem
  epilogue + cuBLASLt sites and encoder QKV+RoPE fusion; SDPA backends
  rejected for prefix attention; HUT: GU phase at ~90% of the cold-burst
  ceiling. Notes: `2026-09-02-vision-gemm-epilogue`,
  `2026-09-02-encoder-qkv-rope-epilogue`, proposed
  `2026-09-02-ffn-gu-dram-ceiling`.

## Initialization TODO

- none (tile library + CUTLASS submodule committed at init; sweep harness at
  `artifacts/ktasks/vision-gemm-retune/sweep_vision.py` already times cuBLAS
  in the same timer and is the G1 harness).

## Round 2 results

- todo#3 prefix epilogues MERGED to main (cherry-pick of 6a607f4): prefix
  6.919 -> 6.306 ms, wall 16.838 -> 16.254 (588760, ACD1-33). Lesson: the
  smem-staged epilogue HURTS at K=16384 (ffn:down 97 -> 105 us) -- the
  short-K rule is short-K only. cuBLAS `addmm_` took o_proj and ffn:down.

- G3 CLOSED as rejected (commits 427d050, 6d50792). Lesson: an isolated
  warmth win (2.5x on the burst) did not survive the real inter-launch
  traffic / DR collision; the decoder FFN's remaining lever is stream
  continuity (megakernel scope), not warmth.
- Process lesson: removing an agent worktree deletes its gitignored
  `artifacts/ktasks/<task>/` workspace -- COPY the workspace to MAIN first
  (the G3 ledger/patch source was lost this way; the note carries the verdict).

## Reachability, revised after job 589178

The FFN bandwidth family is closed by a measured bound (both weight streams
free = 2.39 us/layer = 0.43 ms ceiling), which also retires the megakernel's
continuity argument for this kernel. What replaced it is larger: the
DownResidual dependency chain, 10.5 us/layer on 0.54 GFLOP with 0.25 us of
stream, i.e. ~1.89 ms of decoder time whose cost is round trips and joins.
todo#10 owns it.

## Reachability of the 14.5 ms target (recorded 2026-09-03, superseded above)

At 16.254 ms with G1/G3/G4/G5 rejected, the remaining identified queue is
todo#4b CUDA MQA attention (~0.25), #7 vision LN + SDPA (~0.27), #9 decoder
GU->DR stream continuity (~0.36), #6 fc2 L2 (~0.1, now ownerless after G1) =
~1.0 ms, landing near 15.3 ms. Closing the last ~0.8 ms needs a lever not
yet on the queue; the decoder is the only stage with structural headroom
(7.44 ms measured, ~3.5 ms at the cold-burst rate for its 6.3 GB of weight
traffic) and the megakernel's cross-layer stream continuity is the named
mechanism, currently out of scope by the goal's own exclusion. Surface this
to the user rather than silently missing the number.

## Round 2 closed

Merged: prefix epilogues (fc9832c) = wall 16.838 -> 16.254 (588760).
Rejected with evidence: G3 GU L2-warm, G4 TileLang MQA, G5 both decoder folds.
G1 interrupted by a session quota at candidate c2; its sources are committed
on branch `sm90-gemm-wip` (34faaca, not wired) and its ledger/runs are in
`artifacts/ktasks/sm90-short-k-gemm/`. Baseline for round 3 is 16.254 ms
(vision 2.012 / prefix 6.306 / decoder 7.442); note the decoder measured
7.325 on ACD1-33 in the G5 jobs, i.e. ~0.12 ms of node/time drift.

Round 3 lanes: G1 resume (4 issuing warps + 3-D boxes; the kernel is
TMA-ISSUE bound at [tma.issue.warp] 248 ns/txn, not DRAM bound), todo#4b
(CUDA 2-math-WG MQA attention), todo#7 (vision LayerNorm prologue + SDPA
replacement).

## Round 2 (active, superseded)

Three lanes running as fork subagents in worktrees (started at goal
activation): G1 `sm90-short-k-gemm` (branch sm90-short-k-gemm), G3
`ffn-gu-warm` (branch ffn-gu-warm), todo#3 `prefix-gemm-epilogue` (branch
prefix-gemm-epilogue). Coordinator reviews each branch, strips submodule
typechanges, cherry-picks onto main, then re-profiles.

## Next action

Round 2, G1: open `artifacts/ktasks/sm90-short-k-gemm/contract.md` — one
sm90 GEMM template on `cuda/tile/sm90` (TMA + wgmma pipeline, persistent
tile schedule sized to 132 SMs, epilogues bias / bias+GELU / bias+residual
in place, optional LayerNorm prologue), first target vision qkv
(768x3456x1152) vs cuBLAS 13.5 us; in parallel a subagent runs the G3 HUT
decisive pair (L2 residency of 16.8 MB across the attention chain).
