# Agent Note: acceptance is deployability

Status: implemented

## Problem

The acceptance registry named its latency objective as "the gap to the
structural floor", and the floor model's structural tier carried only a
launch term. On the Pi0.5 Target that made the gap 6.9 ms of "kernel-level
work", while the optimization loop that had just closed had shown, kernel by
kernel, that the action expert sits at its cold-delivery ceiling and its
dependency-chain latency. The objective pointed the flywheel at work that did
not exist, and it had no stop condition a number could decide.

At the same time no metric in the registry carried a human bound, so a
closed-loop policy whose tail latency missed control ticks could not fail a
gate; the candidate rule read the control-leg spread as its only threshold,
so one quiet pair of legs could promote a 0.02 ms change and a noisy pair
could turn every delta into "indistinguishable" without invalidating the run;
a pure refactor could never pass a gate that demanded a distinguishable
improvement; and the official-baseline tier ran under the project interpreter,
which cannot import OpenPI, so every gate ended `blocked`.

## Decision

1. **Acceptance means deployable.** The human defines the accuracy
   requirements, one latency bound and the budget; the floor model is guidance
   for the Analyze step and is not an objective. `eval/acceptance.py` carries
   no `objective` key.
2. **The one human latency number is a tail bound.** `deployment.jitter_ms`
   (0.5 ms for both Targets, set by the owner) bounds the candidate leg's
   chunk-latency `p99 - min` in the A/B/A run. p99 is read as a stability
   property under deployment conditions (clocks unlocked, shared node). When
   the reference leg of the same run violates the bound too, the node is the
   cause and the verdict is `blocked`, not `fail`.
3. **An absolute promotion bar, and a validity limit on the run.**
   `promotion_bar_ms` (0.10 ms, the bar the Pi0.5 loop used for every recorded
   promotion) is what a performance candidate must beat on chunk `min`, on
   top of the control spread. `control_spread_max_ms` (0.10 ms, equal to the
   bar) is the most the two control legs may disagree on chunk `min`: a run
   whose noise is below the effect it must decide is valid, above it
   `benchmarks latency` marks the run invalid and the gate blocks. Measured
   spreads on this cluster's shared, unlocked nodes ran 0.007 to 0.12 ms
   across five runs (jobs 598904, 598948), which is why the limit is the
   bar and not half of it. The control spread keeps its role as the minimum
   detectable effect for the other statistics.
4. **Two candidate modes.** `improve` for a performance candidate; `no_regression`
   for a refactor or a correctness fix, which passes when no chunk statistic
   regresses by more than the larger of the bar and that statistic's own
   control spread (the `min` spread is not the median's noise, and a
   regression below the bar is not one the bar would have promoted). Each
   leg soaks for `soak_s` seconds before its warmup so an unlocked GPU's
   clocks settle before anything is read. The mode is recorded in the
   evidence.

   The protocol's legacy "unlocked" label means the benchmark and repository
   job scripts inherit device state without requesting clock changes. It does
   not certify administrator-level locked-clock bounds. Reports distinguish
   this execution policy and the explicit Slurm GPU frequency request from
   application-clock observations and unobserved effective locked-clock bounds.
   Policy and observed environment changes are already part of segment identity
   and are checked before and after each leg. A/B/A, soak and the control-spread
   limit remain required; unobserved constraints limit attribution and cannot
   be converted into a claim of globally unlocked hardware. Recording only an
   always-missing hardware-lock certificate would prevent the actual execution
   policy from being represented; recording "unlocked verified" from application
   clocks would invent evidence. Neither alternative is accepted.
   Slurm's request variable is documented in
   [sbatch output environment](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES).
5. **A stop condition as data.** `stop.headroom_pct` (10): a Target's
   optimization stops when the deployment bound holds and no candidate is
   left, when the budget is spent, or when every call site measures within
   this fraction of its measured ceiling in the floor report (the floor
   report's flag for that is the floor model's next change).
6. **The baseline tier runs under its own interpreter.** Each Target entry
   names `baseline_python`, selected by machine environment
   (`OPENPI_PYTHON` for Pi0/Pi0.5, `LINGBOT_PYTHON` for LingBot); the gate runs
   the official-baseline scripts under it with the
   repository on the path, the requested input seed and construction options,
   and records `unavailable` when
   that interpreter does not exist or a script reports itself unavailable.
   Each tier judges the Target's reference route, the oracle every candidate
   is compared against in-engine. Pi0's tier builds that route from the
   OpenPI checkpoint's weights (`OPENPI_PI0_CHECKPOINT`) and requires a
   separate immutable checkpoint ID (`OPENPI_PI0_MODEL_REVISION`, its legacy
   environment spelling, or the CLI option). No machine path supplies an
   implicit identity. Missing configuration or assets make the tier unavailable.
   Reusing these environment interfaces avoids a second runtime-discovery
   framework; hardcoded machine defaults would prevent relocation and could
   silently select the maintainer's checkpoint. Pi0.5 accepts an explicit
   checkpoint/configuration
   and separate ID/digest, or uses deterministic random weights when no
   checkpoint is requested. Construction options apply to declarations, all
   numerical comparisons, timing legs and optional floor work. Registered
   shallow depths remain fixed; "full" depths use the declared workload.
   Unsupported official-adapter options fail explicitly. These controls do not
   replace the adapter's required workload, checkpoint or numerical evidence.
   Pi0.5's two official reports explicitly identify the full-prefix and
   single-step expert checks. Both the gate and Campaign import require these
   stages exactly once and compare their actual identities against the
   registered depths; only the expert's flow-step axis differs from the full
   workload. Other shape fields, architecture and ExecutionVariant still match.

7. **Thresholds are keys.** A registry check's `threshold` names a key of the
   precision policy's tolerances; `eval/correctness.py` reads that key, for the
   precision the reference runner reports, rather than one hard-coded name.
   The `overhead` metric reports `p99` beside `min` and `median`.
8. **Sample inputs arrive as deployment delivers them.** An input a host
   slot consumes (Pi0.5's robot state, tokenized on the host) is sampled
   into pinned host memory, values drawn on the device first so dumps stay
   comparable. Resident on the device it forced a synchronization inside
   `forward`, which cost Pi0.5 0.4-0.6 ms of chunk `min` and put every
   host hiccup during the wait on the chunk (jobs 598904, 598952).
9. **The cyclic collector is out of the capture, and that is all.**
   `ModelRunner` disables Python's collector while the stages are captured
   (a collection inside a capture invalidated it, CUDA error 901, job
   598959) and collects once afterwards. It does not freeze the heap: a
   runner references itself through its captured segments, so a frozen
   runner would never be collected, and the freeze that shipped for one day
   made every runner permanent. The owner's ruling: the collector is
   preparation only; the deployment path is graph replay and owns no
   collector policy. The collector-off reading of job 598959 (0.10-0.24 ms
   tails against 2.5-3 ms) stays as evidence for the tail lane.

## Alternatives considered

- Keep the gap to the structural floor as the objective and add the missing
  structural terms: rejected by the owner. The roofline is a datasheet
  quantity and guidance only; measured ceilings say how far a call site is
  from what the machine delivered, and neither is a destination.
- An absolute deployment tick in milliseconds: rejected by the owner in favor
  of a bound relative to the run's own `min`; p99 is a stability metric, and
  the absolute level is what the optimization moves.
- Bootstrap intervals or interleaved A/B/A/B legs for the delta: rejected as
  heavier than the question; the absolute bar plus a validity limit on the
  control spread is what the recorded promotions actually used.
- Fail, rather than block, when the reference leg also violates the tail
  bound: rejected; on a shared unlocked node that is evidence about the node.

## Consequences

- A refactor is gateable: `python -m eval.gate --mode no-regression`.
- A candidate that improves chunk `min` by less than 0.10 ms is not promotable
  however clean its A/B/A; the kernel-design contract's promotion line cites
  this bar rather than restating a number.
- A run with a control spread above 0.10 ms blocks and is rerun rather than
  read.
- The tail bound can fail a candidate on its own; it can also block a run on
  a noisy node, which is recorded as such.
- Reports from before this change carry an `objective` string and no
  `deployment` block; they are not re-read.
- The tail bound found three host-side defects before it passed anything: a
  mid-forward synchronization, collector pauses, and a host slot whose
  elementwise work reached torch's intra-op thread pool. All three were in the
  deployment path, not in the measurement, which is what a deployment bound is
  for.

## Verification

- Login node: `python -m eval.smoke` passes; `eval.gate --help`,
  `eval.correctness --help` import; the registry merges every Target entry.
- GPU, six jobs on shared, unlocked nodes, the last two on the branch head.
  Every gate the registry defines was exercised and reached the verdict it
  should:

  | run | job | node | outcome |
  |---|---|---|---|
  | Pi0.5 shipped vs reference, `--baseline` | 598904 | ACD1-55 | every in-engine gate passed; the baseline tier passed under the OpenPI interpreter; chunk `min` -1.12 ms with spread 0.007 ms; `blocked` on the tail bound, both legs at 4.5-4.7 ms |
  | same, after the host-state fix | 598952 | ACD1-15 | chunk `min` 16.25 -> 15.69 ms; tail 2.97 ms on the candidate leg, 0.13 ms on the reference: `fail` |
  | same, collector on / off | 598959 | ACD1-1 | on: the capture was invalidated (CUDA 901); off: tails 0.10 / 0.15 / 0.24 ms, the tail bound passed |
  | five runs on the branch head | 598975 | ACD1-1 | shipped vs reference: every gate passed, tail 0.095 ms, `blocked` on a 0.17 ms first-run spread; Pi0 with the reference route as candidate: `fail` on the candidate rule (+1.34 ms), tails 0.09 ms; jitter overridden to 0: `blocked` with both legs over; collector counted: 1669 collections, 18 full, tails 0.12 / 0.17 ms |
  | no-regression, a plan against itself | 598948, 598975, 599008 | ACD1-19, ACD1-1 | first crashed (the plan-name candidate leg), then `fail` on a 0.065 ms median move over a 0.050 ms `min` spread, then the candidate rule and the tail bound passed (`blocked` only because `--baseline` was not given) |
  | shipped vs reference, rerun | 599008 | ACD1-1 | gates and rule passed; spread 0.1004 ms, 0.4 us over the limit; the candidate leg's tail 2.94 ms against the reference's 0.15 |
  | both, `--baseline`, last attempt | 599019 | ACD1-1 | every gate and the baseline tier passed on both; spreads 0.122 and 0.113 ms, `blocked`; one candidate leg's tail 11.5 ms, the self-test's 0.22 / 0.20 |
  | Pi0 shipped vs reference, `--baseline`, after the tier could run its script | 599777 | ACD1-55 | every in-engine gate and the baseline tier passed (`baseline_layer0` gate, `baseline_depth` report); chunk `min` -1.347 ms with spread 0.030 ms; tails 0.128 / 0.036 ms: **`pass`**, Pi0's first of record; `artifacts/gate/hardware_nvidia_h100_pi0/shipped-2026-09-07T02:44:03.json` |
  | campaign close: each shipped plan vs its pre-campaign plan, `--baseline` | 600566 | ACD1-3 | Pi0: every gate and the baseline tier passed, chunk `min` -0.711 ms (spread 0.069), tails 0.122 / 0.099: `pass`. Pi0.5: every gate passed, `fail` on the candidate rule at +0.070 ms for two plans that build the identical program (spread 0.007): the difference between two engine instances of one program, the noise the bar sits above; tails 0.067 / 0.079 |

- **The Pi0.5 tail this bound blocked on is closed.** It was the third
  host-side defect the bound found: the `prompt` slot evaluated elementwise
  torch expressions on the GPU's critical path and the wait at the intra-op
  thread pool's barrier is unbounded. Pi0.5 now passes on both plans, and every
  leg carries the attribution record that named it
  ([the Pi0.5 chunk tail is the host slot's thread pool](../performance/2026-09-07-pi05-chunk-tail-is-the-host-slots-thread-pool.md)).

## Related notes

- [runtime/Target boundary and acceptance first](2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  the registry this note redefines; its Decision §2–§4 are amended to point
  here.
- [explicit graph and ModelRunner](2026-09-06-explicit-graph-and-model-runner.md):
  the runner and the shipped / reference plans the gate compares.
