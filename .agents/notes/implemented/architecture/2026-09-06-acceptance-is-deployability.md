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
5. **A stop condition as data.** `stop.headroom_pct` (10): a Target's
   optimization stops when the deployment bound holds and no candidate is
   left, when the budget is spent, or when every call site measures within
   this fraction of its measured ceiling in the floor report (the floor
   report's flag for that is the floor model's next change).
6. **The baseline tier runs under its own interpreter.** Each Target entry
   names `baseline_python` (the OpenPI environment, `OPENPI_PYTHON` to
   override); the gate runs the official-baseline scripts under it with the
   repository on the path, and records `unavailable` only when that
   interpreter does not exist.
7. **Thresholds are keys.** A registry check's `threshold` names a key of the
   precision policy's tolerances; `eval/correctness.py` reads that key, for the
   precision the reference runner reports, rather than one hard-coded name.
   The `overhead` metric reports `p99` beside `min` and `median`.

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

## Verification

- Login node: `python -m eval.smoke` passes; `eval.gate --help`,
  `eval.correctness --help` import; the registry merges every Target entry.
- GPU (job PENDING): `python -m eval.gate --target h100/pi05 --baseline` on
  the shipped plan against the reference; `--mode no-regression` with the
  reference as its own candidate; Pi0 with the reference route as the
  candidate against the shipped plan (the intended negative test).

## Related notes

- [runtime/Target boundary and acceptance first](2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  the registry this note redefines; its Decision §2–§4 are amended to point
  here.
- [explicit graph and ModelRunner](2026-09-06-explicit-graph-and-model-runner.md):
  the runner and the shipped / reference plans the gate compares.
