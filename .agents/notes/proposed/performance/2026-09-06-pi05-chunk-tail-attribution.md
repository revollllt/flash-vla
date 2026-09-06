# Agent Note: attribute the Pi0.5 chunk-latency tail

Status: proposed

## Problem

The deployment bound of the acceptance registry (`eval/acceptance.py`,
`deployment.jitter_ms` = 0.5: the chunk latency's `p99 - min` on the
candidate leg) is the one latency number the owner set, and Pi0.5 does not
meet it. Across the gate runs of 2026-09-06 (19 A/B/A runs, 57 Pi0.5 legs of
100 forwards each, nodes ACD1-1, -15, -19, -33, -55, -58, unlocked clocks,
shared nodes) 22 Pi0.5 legs had a wall-clock tail above 0.5 ms; the 6 Pi0
legs measured the same way had none (0.09-0.13 ms). The gate therefore ends
`fail` or `blocked` on Pi0.5 and has never produced a `pass` of record, while
every correctness gate, the baseline tier and the candidate rule pass.

What is known, from the evidence records under `artifacts/refactor/pr3/`,
`pr4/gate` and `pr5/gate` on this machine (each leg's `metrics` block):

- **It is Pi0.5-specific and plan-independent.** Tails appear on `shipped`
  (cuda / cuda-pdl routes) and on `reference` (all tilelang) legs alike, so
  no backend or the PDL chain is the cause. Pi0, measured on the same nodes
  in the same jobs, never shows one.
- **It is sporadic in time, not in a path.** `chunk_latency` (wall clock)
  and `device_latency` (CUDA events around the same forward) are two
  separate loops of 100 forwards. A tail lands in one or the other or both:
  final-2 leg 1 read wall 0.20 / device 11.7 ms; final-4 leg 1 wall 0.13 /
  device 15.9; final2-1 leg 2 wall 11.8 / device 11.3. One or two forwards
  in a hundred are late, at random moments.
- **The magnitudes cluster.** 2.4-3.0 ms (nine legs), 4.2-4.8 ms (ten),
  7.0, 8.8, 10.9-11.8 (five), 13.1, 15.9 and 17.3 ms (one each): roughly
  multiples of 2.5-4.5 ms, up to one whole forward (min 15.7-17.4 ms).
- **The host slot alone is not late.** `host_time.prompt`, the slot timed
  by itself (tokenize `state` on the host, stage four pinned copies), has a
  tail of 0.015-0.13 ms in every leg.
- **The segments alone are not late either.** Replayed standalone,
  `vision_encoder` tails at 0.04-0.06 ms, `action_expert` at 0.2-1.7 ms,
  and `llm_backbone` at 0.7-1.6 ms on both Targets, in every leg. The
  backbone's consistent ~1 ms standalone tail is a second, separate
  observation; it does not add up to the chunk tail and Pi0 has it too.
- **The collector is not the whole story.** With `gc.disable()` three legs
  read 0.10 / 0.15 / 0.24 ms (job 598959); with the collector frozen after
  capture the tail returned in 7 of 27 legs (jobs 598975, 599008, 599019,
  599021). A full collection during a forward is one mechanism, not the only
  one.
- **Two host-side defects were already removed** and did not remove the
  tail: the sampled `state` resident on the device (a synchronization inside
  `forward`, job 598952) and collections inside a capture (job 598959).

The structural difference between the two Targets is where the host sits in
the forward. Pi0's forward is three graph launches back to back: if the host
thread is descheduled after launching, the GPU keeps working and the wall
clock does not notice unless the preemption outlasts the remaining GPU work.
Pi0.5's forward is `vision_encoder` replay, then the host slot (tokenize the
robot state, four pinned host-to-device copies), then the `llm_backbone`
and `action_expert` replays (`src/flash_vla/hardware/nvidia/h100/pi05/pipeline.py`,
`g.host("prompt")`; `src/flash_vla/runtime/runner.py`, `forward`). The GPU
finishes vision in ~2 ms and then waits for the host: any host delay at that
moment (a scheduler slice, a page fault, a collection, a stalled pinned copy)
is paid in full by the GPU timeline, which is why both the wall clock and
the device events see it. That is a hypothesis, not a finding.

## Proposal

One lane, three decisive experiments on one node in one job, each an A/B/A
through `benchmarks latency` (`python -m benchmarks latency --target
h100/pi05 --plan shipped --reps 100`, three legs, both plans), with the
harness extended first so the run explains itself:

1. **Instrument before hypothesizing.** Per leg, `benchmarks/latency.py`
   records: involuntary and voluntary context switches of the process during
   the timed loop (`/proc/self/status`), the collector's collections with
   their generation and wall-clock time (`gc.callbacks`), the per-forward
   sample list (not only min / median / p99, so a late forward can be
   matched to a timestamp), and a 10 Hz sample of the GPU's SM clock and
   `clocks_event_reasons` (`nvidia-smi --query-gpu=...`) and of other
   compute processes on the device. A late forward is then attributable
   from the record rather than argued about.
2. **Move the host dependency off the GPU's critical path.** The prompt
   slot needs only `state`, which is a host input available before the
   first launch. Reorder Pi0.5's program so the host slot runs before the
   `vision_encoder` replay and the three replays follow back to back
   (`pipeline.py`, one placement change). Expected cost: up to the slot's
   0.15-0.17 ms on chunk `min`, since tokenization no longer overlaps
   vision. If the tail disappears, the mechanism is host lateness at the
   mid-forward dependency, whatever its source, and the reordering (or a
   host slot overlapped by a second thread) is the fix.
3. **Pin the host.** With the original order, run the legs with the process
   bound to one core of the job's cgroup (`os.sched_setaffinity`) and
   compare; if the tail goes, the source is scheduling contention on the
   shared node, and a deployment host needs a reserved core (record it as a
   deployment requirement in the acceptance note).
4. **Isolate the pinned copies.** With the original order, replace the four
   `copy_into` host-to-device copies by device-to-device copies from a
   device staging buffer written once; if the tail goes, the source is
   PCIe / pinned-memory contention with other jobs on the node.

Each experiment is a lab script under `lab/pi05/`, its Slurm wrapper under
`lab/sbatch/`, and its evidence in `artifacts/`; whichever change ships goes
into the deployment path in its own PR with the gate's `pass` as evidence.

## Alternatives considered

- Loosen or reformulate the bound (p95, or the fraction of late forwards):
  not this lane's call. The owner set `p99 - min <= 0.5 ms` as the stability
  requirement; a bound that hides the tail does not make the controller
  meet its tick.
- Require an exclusive node for deployment measurements: defensible only
  after the mechanism is known; if the cause is the mid-forward host
  dependency, an exclusive node hides a defect the robot host will still
  have.
- Keep the collector off in the harness: it removed the tail in three legs,
  but the framework cannot own the application's collector policy, and the
  frozen-collector runs show the tail without a collection being the obvious
  cause.

## Acceptance criteria

- A named mechanism with a same-node A/B in the record: the late forwards
  in the control legs line up with a recorded event (context switch,
  collection, clock event, other process, copy stall), and the treated leg
  has none above 0.5 ms.
- Pi0.5 chunk `p99 - min <= 0.5 ms` on both plans in at least three
  consecutive A/B/A runs on shared nodes, or a stated deployment
  requirement (a reserved core, a policy) under which it holds, recorded in
  the acceptance note.
- `python -m eval.gate --target h100/pi05 --baseline` returns `pass` once,
  with the evidence path in the note.
- The harness keeps the per-leg attribution record after the lane, so the
  next tail is read, not re-investigated.

## Risks

- The control spread on ACD1-1 sat above the 0.10 ms validity limit in
  five of eight runs; a node that does not hold still invalidates the A/B/A
  before the tail can be read. Pick the node by `sinfo` GRES usage and soak
  longer if the first run reads `blocked`.
- Reordering the host slot changes the pipeline's overlap and costs chunk
  `min`; the promotion bar (0.10 ms) applies to that regression like any
  other, so the lane should measure both orders in one job.
- The runner's collector handling was amended after the runs listed here:
  it no longer freezes the heap (freezing in the constructor made every
  runner permanent, since a runner references itself through its captured
  segments), and by the owner's ruling the deployment path is graph replay
  with no collector policy of its own. Re-baseline on the amended runtime;
  the collector-on runs from before the freeze (598904 to 598959) are the
  comparable ones.

## Related notes

- [acceptance is deployability](../../implemented/architecture/2026-09-06-acceptance-is-deployability.md):
  the bound, the two host-side fixes already made, and the run table.
- [explicit graph and ModelRunner](../../implemented/architecture/2026-09-06-explicit-graph-and-model-runner.md):
  the program order and the host slot contract this lane may change.
