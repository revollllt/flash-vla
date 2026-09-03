# Agent Note: The GatedUp weight ring is already at its knee at depth 3

Status: rejected — a deeper weight ring is slower, measured three ways. The
refutation of the predicted gradient is the durable result.

Date: 2026-09-03

## Problem

The GatedUp accounting
(`.agents/notes/rejected/architecture/2026-09-03-ffn-gu-accounting.md`) found
that 9.6 of the phase's 11.6 us/layer is the copy pipeline and its waits, and
that the pole is neither bandwidth nor transaction count but how many weight
frames can be in flight: the ring is 3 deep against 4 trips, and depth 3 -> 2
costs +1.52 us/layer. It proposed taking the weight ring to 4, paid for by a
3-deep rotating activation ring, and priced it at about the 1.52 us the
gradient measures — roughly 0.27 ms over 180 launches. It explicitly did not
verify that the gradient is symmetric across the steady-state boundary.

It is not symmetric. That is this note.

## Decision

Keep the shipped geometry: 4 stationary activation frames, weight ring 3 deep.

### The constraint the proposal was really fighting

Seven 32 KiB frames fit under the 232448 B data-plane cap. The mainloop has
four trips and each wants an activation frame and a weight frame, so eight
slots are wanted and seven exist. Exactly one frame must be refilled
mid-flight whatever the split, and the refill cannot start until the math
retires the trip that used it. Moving the shortage from the weight ring to
the activation ring therefore moves which stream issues late; it does not
remove the late issue. (Production makes this cleaner than it looks: the call
site passes `task_count = 1`, so the "stationary across tasks" property the
current design buys is never used and rotating the activation adds no loads.)

### Measured (job 591080, ACD1-2, six builds in one job, reps 50)

`bench_gu.py` medians; every valid row also ran
`ffn_taskloop_parity --modes gu,dr,full` and passed at worst cosine
0.9999999, which is also the proof that the flags are behavior-neutral when
unset. Frames are 32 KiB each.

| row | act | w | frames | numerics | gu-only | Δ | fused | Δ |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| c0 shipped | 4 | 3 | 7 | valid | 11.824 | — | 19.424 | — |
| c1 stale act | 3 | 4 | 7 | BOUND | 12.011 | **+0.187** | 19.563 | +0.139 |
| c2 rotating | 3 | 4 | 7 | valid | 12.048 | +0.224 | 19.792 | +0.368 |
| c3 rotating | 3 | 3 | 6 | valid | 11.851 | +0.027 | 19.195 | −0.229 |
| c4 rotating | 4 | 3 | 7 | valid | 11.840 | +0.016 | 19.355 | −0.069 |
| c5 stale act | 2 | 5 | 7 | BOUND | 12.101 | +0.277 | 19.643 | +0.219 |

**c1 is the decisive row.** It gives the weight ring depth 4 while charging
the activation side nothing — the wrapped trip reads a stale frame, so the
numerics are void and no refill is issued. If depth were worth anything this
is where it would show, with no payment to obscure it. It is +0.187 us/layer:
slower. c5 takes the ring to depth 5 and is slower still, monotonically.

**The gradient is one-sided.** Depth 3 -> 2 costs 1.52 us; depth 3 -> 4 and
3 -> 5 cost 0.19 and 0.28. Depth 3 is at or just past the knee for a
four-trip mainloop that keeps one wgmma group outstanding
(`[wgmma.stages.wg.knee]`: 4 in flight, `wait_group >= 1`). Below it the
pipeline starves; above it there is nothing left to cover and the extra
frames only add barrier and issue work.

**The rotation protocol is nearly free** (c4, same geometry with the protocol
running: +0.016 gu-only), so the proposal's cost estimate was right and only
its benefit estimate was wrong.

## Alternatives considered

- Trade the frame the other way (c3, six frames): neutral in GatedUp (+0.027)
  and it frees 32 KiB of the data plane. It buys no latency — the CTA is at
  1 per SM either way, since two would need 393488 B against the 232448 B cap
  — but a future lane that wants shared memory for something else can have
  that frame for free. Not a reason to change the shipped source now.
- Shrink `GATED_UP_BLOCK_K` to 128 so eight smaller frames fit: rejected
  without a build. In-flight weight bytes are the invariant the depth
  argument rests on (3 x 32 KiB = 6 x 16 KiB), and this lane has just shown
  more in-flight bytes do not help.

## Consequences

- The shipped kernel is unchanged. The `FFN_GU_ACT_FRAMES` /
  `FFN_GU_WEIGHT_DEPTH` / `FFN_GU_ACT_ROTATE` / `FFN_GU_BOUND_STALE_ACT`
  flags, including a working rotating-activation implementation with the
  release-on-retirement protocol, live in the workspace patch
  (`artifacts/ktasks/gu-ring-depth/patch_ring_flags.diff`), not in the source.
- Every bandwidth- and pipeline-side idea for the GatedUp phase now has a
  measured bound: warming, prefetch, box retiling, stream continuity, and now
  ring depth. What remains in the phase is the 2.04 us of gelu/product
  arithmetic the accounting lane priced, and it is arithmetic, not overhead.
- A general rule worth carrying: a depth gradient measured downward does not
  predict the gain upward. Reducing a ring below the knee exposes latency the
  pipeline was covering; increasing it above the knee covers nothing. Price
  the direction you intend to move.

## Verification

One GPU job of the seven allowed (591080, ACD1-2, reps 50, six builds in the
same job so no cross-job normalization is needed; the c0 row's 11.824 sits at
the low end of the accounting lane's 11.56-12.45 baseline band). No
`plan_parity` or `plan_e2e` run: the per-kernel gate to earn one was
1.0 us/layer on the fused column and the best row here is 0.23 us in the
wrong direction. Workspace, ledger, patch and job log:
`artifacts/ktasks/gu-ring-depth/`.

Single-job caveat: each row is one median of 50 reps, and the deltas at
0.02-0.28 us are near the resolution the previous lanes treated as signal.
The verdict does not depend on resolving them, because the thesis predicted
-1.5 us and no row is negative in GatedUp at all.
