# Agent Note: the Pi0.5 chunk tail is the host slot's thread pool

Status: implemented

## Problem

The deployment bound of the acceptance registry (`eval/acceptance.py`,
`deployment.jitter_ms` = 0.5: the chunk latency's `p99 - min` on the candidate
leg) is the one latency number the owner set, and Pi0.5 did not meet it. Across
the gate runs of 2026-09-06 (19 A/B/A runs, 57 Pi0.5 legs of 100 forwards
each, six nodes, unlocked clocks, shared nodes) 22 Pi0.5 legs had a wall-clock
tail above 0.5 ms; the 6 Pi0 legs measured the same way had none. The gate
therefore ended `fail` or `blocked` on Pi0.5 and had never produced a `pass`,
while every correctness gate, the baseline tier and the candidate rule passed.

The tail was plan-independent, node-independent, sporadic in time, and absent
from Pi0. Two host-side defects had already been found and removed by the same
bound (a sampled state resident on the device, and collections inside a
capture) without removing it. Nothing in the report said which forward was
late, when, or what else the machine was doing, so every remaining hypothesis
was an argument rather than a finding.

## Decision

1. **A latency report carries its own attribution.** Each leg of
   `benchmarks/latency.py` records, beside its statistics, every timed loop's
   per-forward samples with their offsets and wall clock, the process's
   per-forward context-switch and page-fault deltas, the loop's
   voluntary/nonvoluntary switch delta, the cyclic collector's collections with
   generation and duration, a 10 Hz record of the device's clocks, clock event
   reasons and other compute processes, and the samples that ran above `min`
   plus the registry's own jitter bound with the events that fall inside them.
   The block is additive and the statistics keep their shape, so the gate and
   the candidate rule read the report they always did and the gate keeps the
   block in its evidence record.

   The instrument may not create the artifact it looks for. A 10 Hz sampling
   thread would hand the GIL over at the interpreter's switch interval, which
   is the magnitude under investigation, so the device sampler is a subprocess
   writing to a file that is parsed afterwards, the per-sample counters are one
   `getrusage` outside the timed region, `/proc` is read twice per loop rather
   than per forward, and the sample lists are preallocated. Every instrument
   fails into a recorded error string.

2. **The mechanism: Pi0.5's `prompt` host slot evaluated elementwise torch
   expressions on the GPU's critical path, and torch dispatched them across its
   intra-op thread pool.** The wait at that pool's barrier has no bound. The
   slot's wall time was bimodal, 0.34 ms in 96.5% of forwards and 8 to 16 ms in
   the rest with nothing in between, and its excess was 117% of the whole
   forward's excess because the synchronize that follows shortens as the GPU
   catches up. The slot sits between the vision replay and the backbone, with
   about 2 ms of vision work to hide behind, so a stall of that size is paid in
   full by the chunk. Pi0 has no host slot, which is the entire asymmetry.

3. **What depends only on the token count is tabulated, not computed.** The
   tokenizer pads to the prompt length and marks a prefix of `n_tokens` valid,
   so the embedding scale, the attention mask and the decoder's rotary table
   are functions of `n_tokens` alone and it ranges over `[0, prompt_len]`.
   `hardware/nvidia/h100/pi05/prefix.py` builds those tables at construction by
   running the same expression once per reachable count, so they are
   bit-identical to the per-call arithmetic by construction rather than by
   argument; the slot selects a row and copies. The task string does not enter
   the tables, so changing it needs no rebuild.

4. **A host slot on the critical path performs no elementwise work whose size
   could reach the thread pool.** This is the invariant the fix establishes and
   the one a future host slot has to keep. It is a property of the deployment
   path, not of the measurement: the framework sets no thread count and owns no
   scheduling policy.

## Alternatives considered

Each was run as a same-node A/B, one engine per plan, eight legs per plan
alternating control and treated, 100 forwards per leg (job 599723). None
removed the tail, and none changed the count of blocking waits per forward.

- **Move the host slot ahead of the vision replay** so three replays follow
  back to back: rejected. It left the tail (treated legs broke the bound in
  four of four cells on the shipped plan) and cost chunk `min` 0.25 to 0.31 ms,
  above the registry's promotion bar, which is the overlap the slot was buying.
- **Bind the measuring thread to one core of the job's cgroup**: rejected. Clean
  on one plan and not the other, with no change to the blocking-wait count; the
  record shows late forwards carry no involuntary preemption, which a
  scheduling explanation would have needed.
- **Serve the four staged copies from device memory** instead of pinned host
  memory: rejected. The copies were never the problem: over 3000 forwards their
  maximum is 0.147 ms and they block never.
- **Disable the cyclic collector**: rejected. The tail is unchanged with the
  collector off, and on the amended runtime a leg runs one collection lasting
  0.03 to 0.09 ms, none of them inside a late forward. The earlier
  collector-off reading was three legs of an event that appears in about a
  quarter of them.
- **Loosen or reformulate the bound**, or require an exclusive node for
  deployment measurements: not taken. The bound found a real defect in the
  deployment path, which is what it is for.

## Consequences

- Pi0.5's host slot costs 0.078 ms of host time per inference instead of
  0.348 ms, and its worst case over 3000 forwards is 0.196 ms instead of
  13.605 ms. The tables cost about 5.6 MB and half a second of construction at
  this Target's shape, both proportional to the padded prompt length.
- A Pi0.5 forward now blocks the host 0.32 times, the same as a Pi0 forward.
  The seven blocking waits a Pi0.5 forward used to perform were the pool's
  workers, not the caller's wait for the GPU.
- The tail bound has now found three host-side defects in the deployment path
  and passed nothing that did not deserve it.
- A tail is read from the record rather than re-investigated: the attribution
  block names, per late forward, the collections, switches, faults, clock
  events and other processes that fall inside it.
- `lab/pi05/tail_probe.py` is the step-level reading that loop statistics
  cannot give, and `lab/pi05/tail_experiments.py` is how a host-side treatment
  is compared against its own control on one engine.

## Verification

- Login node: `python -m eval.smoke` passes; `benchmarks latency --help` and
  `eval.gate --help` import. The three tables were checked bit-identical to the
  expressions they replace for all 201 reachable token counts. Isolated over
  20000 calls on the same box with the same tokenizer, the slot's host half
  read median 0.170 ms, p99 93.3 ms, maximum 99.7 ms and 519 calls above 2 ms
  before, and median 0.026 ms, p99 0.034 ms, maximum 1.66 ms and none above
  2 ms after. That the pool is the mechanism was checked directly: the same
  expression with one intra-op thread reads a p99 of 0.044 ms against 93.2 ms
  with the default pool.
- GPU, `sbatch -p acd_u --gres=gpu:1`, clocks unlocked, shared nodes:

  | run | job | node | what it established |
  |---|---|---|---|
  | re-baseline and Pi0 control, with attribution | 599723 | ACD1-12 | the tail survives the amended runtime; late forwards carry no collection, no page fault, no clock event and no other process on the device, and the GPU idles 35-40% of a burst at full clock |
  | the four treatments, both plans, 8 legs each | 599723 | ACD1-12 | none removes the tail; none changes the blocking-wait count |
  | step probe, both Targets, 1000 forwards each | 599781 | ACD1-55 | the tail is the `prompt` slot, and Pi0 has none |
  | three A/B/A runs and the gate, before | 599781 | ACD1-55 | candidate tails 3.414, 2.747 and 0.145 ms; gate `fail` on the tail bound at 2.671 ms with every other gate and the baseline tier passing |
  | split probe, both plans, 3000 forwards each | 599786 | ACD1-55 | the tokenization half is bimodal to 13.6 ms; the copies never exceed 0.147 ms |
  | split probe, three A/B/A runs and the gate, after | 599815 | ACD1-8 | the slot's host half maxes at 0.196 ms and one forward in 3000 exceeds the bound on either plan |

- The reproducible commands are `python -m benchmarks latency --target
  h100/pi05 --plan reference --plan shipped --plan reference --reps 100`,
  `python -m lab.pi05.tail_probe --target h100/pi05 --split-host --reps 3000`
  and `python -m eval.gate --target h100/pi05 --baseline --reps 100`, submitted
  through `lab/sbatch/tail.sh`.

## Related notes

- [acceptance is deployability](../../implemented/architecture/2026-09-06-acceptance-is-deployability.md):
  the bound this lane answers to, and the two host-side defects it found first.
- [explicit graph and ModelRunner](../../implemented/architecture/2026-09-06-explicit-graph-and-model-runner.md):
  the program order and the host slot contract, both unchanged by this lane.
