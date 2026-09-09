# Agent Note: Bounded optimization evidence and early failure

Status: implemented

## Problem

Route ambiguity, summing overlapping durations and evaluating after known
correctness failures make experiments expensive and conclusions unreliable.
The owner's plan also needs durable failures, controlled comparisons and
source applicability without adding hashes or new acceptance policy.

## Decision

Use explicit experiment inputs, declared source copies, loaded artifact paths
and retained stage evidence around the existing tools. Treat GPU ownership as
Slurm state, not an expiring timestamp. Budget related trials together and retain
confirmation resources. A worker lost before accounting requires explicit cost
recovery after confirmed termination.

Qualification uses the experiment's recorded construction options and seed.
Checkpoint, reference configuration and fixture selection must reach the evaluator
unchanged; otherwise an explicit real-asset experiment can silently evaluate the
factory defaults. Preflight and qualification share spec.options; the experiment index retains
those options alongside conditions so recovery and duplicate detection retain
the same selection. Passing them as separate argv
values preserves paths and prompts without shell interpretation.

Separate numerical oracle from performance incumbent. Existing required-check
failure or missing official evidence stops qualification before latency work;
floor is optional diagnostic work. Preserve same-trace intervals, sums, unions
and occurrence makespans. Overlapping groups without a joint ceiling model cannot
justify automatic floor stopping.

Keep reference applicability and experiment reopening explicit. A prior invalid
run is not a negative mechanism result; missing artifacts leave identity unknown.
Latest-source review follows a gate pass, and promotion never edits shipped.
Qualification of distinct source versions remains unsupported through the
existing evaluator; its supported mode is route variants in one source tree.

Agent continuation begins with canonical discovery and compact published facts,
then current Campaign state and unresolved hypotheses. Raw profiles and historical
stdout are read only for a specific unresolved question. Compatible checkpoint
changes reuse the lineage and portable implementation, then rebuild/retune,
validate and re-anchor before another iteration. Agent instructions must preserve
the evaluator's actual supported source-loading boundary; they cannot turn
unsupported source-version qualification into an acceptance claim.

## Alternatives considered

A full snapshot/cache/distributed platform exceeds the current use. New hashes,
frozen contracts, baselines and acceptance gates conflict with the owner's
constraints. Generalizing all backend loading before the first component runs
would create unsupported interfaces. Treating profiled overlap as recoverable
E2E benefit confuses observation with causation.

## Consequences

The controller is a bounded tool, not an autonomous performance verdict. Its
small index survives loss of large artifacts, but source dependency selection
remains explicit. The QKV wider-tile candidate is rejected on two independent
cold-weight experiments; the TMA probe narrows a mechanism without claiming a
vision speedup. No candidate combination currently requires promotion.

## Verification

The execution checklist links focused CPU tests, the real shallow PDL profile,
component A/A-prime diagnostics, two production shapes, GPU fixture/replay
checks, calibration and independent QKV/TMA runs. CPU old/new gate experiments
retain identical fail/blocked verdicts while avoiding latency and floor calls
once their prerequisites fail. They do not measure Agent productivity.
See `docs/optimization-results.md` and `artifacts/optimization/job-accounting.txt`.
