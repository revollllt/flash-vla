# Agent Note: a 64-wide DownResidual tile is slower at every split — the epilogue is per-owner and serial

Status: rejected — measured (job 589409). The 64-column output tile that
`2026-09-03-ffn-dr-chain.md` named as its successor is 2.1-4.4 us/layer worse
on the DownResidual column at all three splits tried, and the mechanism
explains why the predicted compounding could never have arrived.

Date: 2026-09-03

## Problem

The preceding lane priced every link of the DownResidual chain by deleting it
and found none clears the promotion bar alone: readiness poll 1.05 us, split-K
join wait 0.96, partial fold 1.37, residual read 0.69, against a 10.5 us phase
whose compute floor is 0.64 us. It named one structural change that attacks
several links at once and did not build it: widen the output tile from 32 to
64 columns, so that 32 tiles become 16 (halving the joins and the fold
traffic) while the wgmma moves off the m64n32k16 atom
(`[wgmma.issue.wg.ss]`: 62% of peak) onto m64n64k16 (93%), the 1.49x GatedUp
already takes.

## Decision

Reject the wide tile. The output-tile width and the split are one geometry --
halving the tile count halves the workers unless the split grows to match --
so all three reachable operating points were measured, in one job on one node
with the baseline row rebuilt beside them.

| geometry | tiles | DR workers | stages/split | txns/CTA | partials | gu-only | fused | Δfused | dr-only | Δdr-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 wide, S=4 (shipped) | 32 | 128 | 8 | 25 | 0.75 MB | 11.88 | 19.46 | — | 10.52 | — |
| 64 wide, S=4 | 16 | 64 | 8 | 25 | 0.75 MB | 11.82 | 22.60 | +3.14 | 12.61 | +2.10 |
| 64 wide, S=8 | 16 | 128 | 4 | 13 | 1.75 MB | 11.81 | 23.75 | +4.30 | 13.50 | +2.98 |
| 64 wide, S=2 | 16 | 32 | 16 | 49 | 0.25 MB | 11.81 | 25.19 | +5.74 | 14.89 | +4.37 |

`gu-only` is flat at 11.81-11.88 across all four rows, so the change is
DownResidual-local and the deltas are not drift. The best wide variant costs
+0.57 ms over 180 launches where the lane was opened expecting -0.5 ms.

## Why the predicted compounding does not exist

- **The wgmma gain applies to 0.64 us.** The phase's compute floor is 0.64 us
  of a 10.5 us phase, so moving from 62% to 93% of the atom's peak can return
  at most ~0.2 us. The 1.49x is real and irrelevant at this ratio; GatedUp
  benefited because its column is the one doing the arithmetic.
- **The epilogue is per-owner and serial, so halving the owners lengthens the
  critical path.** Only split 0 folds the partials, reads the residual and
  stores; those were priced at 1.37 + 0.69 us. Their TOTAL work is fixed by
  the output area, so a 64-wide tile leaves each owner doing twice the fold
  and twice the read-modify-write while there are half as many owners running
  concurrently. Wall time follows one owner's serial chain, which doubles.
  Halving a join COUNT does not halve a join COST when the count was already
  running in parallel and the per-owner work is what grows.
- **The monotone ordering across splits confirms it.** More workers did not
  rescue it (S=8, 128 workers, is worse than S=4, 64 workers) and neither did
  fewer transactions (S=8 issues 13 per CTA against 25 and is still worse);
  the worst row is the one with the longest per-owner chain (S=2, 16 stages).
  Nothing here is bound by worker count or transaction count -- the same two
  hypotheses the preceding lane refuted with its own bounds.

## Alternatives considered

- **Keeping the parameterisation in the source.** It is neutral at the default
  (the baseline row reproduces the preceding lane's 19.57-19.64 / ~11.9 /
  10.31-10.65 band exactly), but it is configurability that exists only to
  express a rejected geometry. Reverted per the rejected-flag precedent; the
  patch is in the task workspace.
- **A 64-wide tile with the epilogue split across CTAs**, so the owner's fold
  is shared. That is a different design -- a reduction task kind, which
  `reduction-own-task-kind` covers and which the split-K planner already
  sketches for a future queue format. It is not a variant of this change and
  was not measured.

## Consequences

- DownResidual's shipped geometry (32 columns, S=4, 128 workers) stands, and
  is now measured to be the best of the four reachable points, not merely the
  incumbent.
- With the byte hypothesis, the transaction hypothesis, four chain links and
  now the tile geometry all measured, incremental change to this kernel is
  closed. Remaining ideas for the decoder FFN must change what the phase
  computes or who computes it, not how the same work is shaped.
- Recorded for whoever revisits the epilogue: its cost is proportional to the
  output area per owner and sits on the critical path, so any redesign must
  keep per-owner epilogue work constant or move it off the owner entirely.

## Verification

Job 589409 (ACD1-13, driver 610.43.02, CUDA 13.1, torch 2.13.0+cu130, clocks
not lockable so the 6% noise floor applies). One job, one node, four builds:
each row ran `ffn_taskloop_parity --modes gu,dr,full` before its bench and
passed -- worst cosine 0.9999999 at the shipped width and 0.9997693 at 64,
identical to seven digits across all three splits, so the deviation tracks the
instruction and swizzle rather than the schedule; it stays inside the 0.999
gate and was not root-caused because the geometry is rejected on speed. All
geometries compile clean (login-node check over the four pairs). No route or
e2e job was run: the best candidate is 3.14 us/layer the wrong way, which no
end-to-end measurement would reverse. Budget: 1 GPU job of 6, three
consecutive non-improvements, stopped per contract.
